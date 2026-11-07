"""Real SB3/physics integration plus split and diagnostics contract tests."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

pytest.importorskip('stable_baselines3')
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env as sb3_check
from gymnasium.utils.env_checker import check_env
from escort_rl.env import G1EscortEnv
from escort_rl.ppo_training import (read_config,streams,make_vec,TrainingScenarios,checkpoint,exact_learn,provenance)
from escort_rl.ppo_evaluation import validation_cases,reserved,EpisodeMetrics,aggregate,selection_key,promotion_pass
from escort_rl.stage1_scenarios import load_config,sample_episode
from escort_rl.training_callbacks import ResearchCallback


def settings():
    c=read_config();return c,load_config(c['environment_path']),json.loads(Path(c['split_path']).read_text())


def test_splits_and_sampling():
    c,env,split=settings();cases=validation_cases(env,split)
    assert len(cases)==60
    assert all(x['episode_seed']%4==1 and not reserved(x['metadata'],split) for x in cases)
    signatures=[(x['family'],x['metadata']['scenario_id']) for x in cases]
    assert len(set(signatures))==60
    assert cases==validation_cases(env,split)
    draws=[sample_episode(4*i,config=env).scenario_metadata()['family'] for i in range(800)]
    for family,p in env['families'].items():assert abs(draws.count(family)/800-p)<.06
    s=streams(42,2);assert len(set([s['model'],s['evaluation'],*s['workers']]))==4
    assert s==streams(42,2)


def test_real_environment_checkers_and_terminal_contract():
    env=G1EscortEnv()
    try:
        check_env(env,skip_render_check=True);sb3_check(env)
        obs,_=env.reset(seed=42)
        assert obs.shape==(40,) and obs.dtype==np.float32 and not obs[22:].any()
        _,r,t,tr,info=env.step(np.zeros(3,np.float32))
        assert not t and not tr and np.isfinite(r)
        assert info['physics_steps']==100 and info['gait_calls']==10
        assert sum(info['reward_components'].values())==r
    finally:env.close()


def test_training_wrapper_rng_and_no_privileged_inputs(tmp_path):
    c,config,split=settings();env=TrainingScenarios(G1EscortEnv(config=config),split,123)
    try:
        state=env.rng_state();obs,info=env.reset();meta=env.current.copy()
        assert meta['episode_seed']%4==0 and not reserved(meta,split)
        env.set_rng_state(state);again,_=env.reset()
        assert np.array_equal(obs,again) and meta==env.current
        actor=obs.copy();env.current['family']='fake_evaluator_value'
        assert np.array_equal(actor,obs)
        assert obs.shape==(40,) and 'family' not in info
    finally:env.close()


def test_two_workers_independent():
    c,_,_=settings();v=make_vec(c,2,42)
    try:
        obs=v.reset();assert obs.shape==(2,40)
        states=v.env_method('rng_state');assert states[0]!=states[1]
        _,rewards,_,infos=v.step(np.array([[.1,0,0],[-.1,0,0]],np.float32))
        assert np.isfinite(rewards).all()
        assert all(i['physics_steps']==100 and i['gait_calls']==10 for i in infos)
        assert infos[0]['scenario_diagnostics']['episode_seed']!=infos[1]['scenario_diagnostics']['episode_seed']
    finally:v.close()


def test_optimizer_checkpoint_fresh_process_and_resume(tmp_path):
    torch.set_num_threads(1);c,_,_=settings();v=make_vec(c,1,42)
    callback=None
    try:
        before=provenance(c);model=PPO('MlpPolicy',v,n_steps=8,batch_size=8,n_epochs=2,
            policy_kwargs={'net_arch':dict(pi=[128,128],vf=[128,128])},seed=31,device='cpu')
        obs=v.reset();action,_=model.predict(obs,deterministic=True)
        assert action.shape==(1,3) and np.isfinite(action).all()
        # Actor and critic consume the SAME 40-float tensor. No evaluator data in rollout buffer.
        assert model.policy.mlp_extractor.policy_net[0].in_features==40
        assert model.policy.mlp_extractor.value_net[0].in_features==40
        initial={k:t.clone() for k,t in model.policy.state_dict().items()}
        callback=ResearchCallback(tmp_path,1,1000,lambda *args:None)
        exact_learn(model,16,callback)
        assert model.num_timesteps==16 and model._n_updates==4
        assert any(not torch.equal(t,initial[k]) for k,t in model.policy.state_dict().items())
        assert model.rollout_buffer.observations.shape[-1]==40
        path=tmp_path/'checkpoint';checkpoint(model,path,before)
        with pytest.raises(FileExistsError):checkpoint(model,path)
        expected,_=model.predict(obs,deterministic=True)
        np.save(tmp_path/'obs.npy',obs)
        script="from stable_baselines3 import PPO; import numpy as np,sys; m=PPO.load(sys.argv[1],device='cpu'); a,_=m.predict(np.load(sys.argv[2]),deterministic=True); np.save(sys.argv[3],a)"
        subprocess.run([sys.executable,'-c',script,str(path),str(tmp_path/'obs.npy'),str(tmp_path/'actions.npy')],check=True)
        assert np.array_equal(expected,np.load(tmp_path/'actions.npy'))
        loaded=PPO.load(path,env=v,device='cpu');assert loaded.num_timesteps==16
        assert loaded.policy.optimizer.state_dict()['state']
        loaded.learn(8,reset_num_timesteps=False);assert loaded.num_timesteps==24
        after=provenance(c);assert before==after
    finally:
        if callback:callback.close()
        v.close()


def test_metrics_and_selection():
    m=EpisodeMetrics();info=dict(reason='success',elapsed_s=.2,evaluation=dict(formation_error_m=.1,
        accompanied_robot_progress_m=3.,robot_advance_m=3.2),reward_components={'formation':-.1,'progress':.4},
        body_velocity=[.2,0,0],applied_command=[.3,0,0],raw_action=[2,0,0],clipped_action=[1,0,0])
    m.add(.3,info);e=m.summary('straight',4)
    assert np.isclose(sum(e['reward_components'].values()),e['return_total'])
    assert e['action_clipped_fraction']==1
    summary=aggregate([e]);worse=dict(summary,success_rate=0,mean_return=1000)
    assert selection_key(summary)>selection_key(worse)


def test_orchestrated_resume_and_two_worker_budget(tmp_path):
    from escort_rl.ppo_training import run_training,DEFAULT_PPO_CONFIG
    import yaml
    c=yaml.safe_load(DEFAULT_PPO_CONFIG.read_text())
    c['environment_config']=str(DEFAULT_PPO_CONFIG.parent/'stage1.yaml')
    c['split_config']=str(DEFAULT_PPO_CONFIG.parent/'stage1_splits.json')
    c['ppo'].update(n_steps=8,batch_size=8,n_epochs=1)
    path=tmp_path/'ppo.yaml';path.write_text(yaml.safe_dump(c))
    first=run_training(path,42,2,16,tmp_path/'first',validation=False)
    assert first['new_transitions']==16 and first['optimizer_updates']==1
    second=run_training(path,42,2,8,tmp_path/'second',resume=tmp_path/'first/final.zip',validation=False)
    assert second['start_timesteps']==16 and second['end_timesteps']==24
    assert second['optimizer_updates']==2
    assert (tmp_path/'second/final.rng.pt').exists()
    with pytest.raises(ValueError,match='10,000'):
        run_training(path,42,2,10001,tmp_path/'forbidden',validation=False)


def test_sb3_timeout_semantics():
    from stable_baselines3.common.vec_env import DummyVecEnv
    c,config,split=settings();config['environment']['time_limit_s']=.2
    v=DummyVecEnv([lambda:TrainingScenarios(G1EscortEnv(config=config),split,42)])
    try:
        v.reset();_,_,done,infos=v.step(np.zeros((1,3),np.float32))
        assert done[0] and infos[0]['TimeLimit.truncated']
        assert infos[0]['reason']=='time_limit' and infos[0]['terminal_observation'].shape==(40,)
    finally:v.close()
