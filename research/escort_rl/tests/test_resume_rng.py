"""Regression: actual setup/load/first reset, independent streams, repeated restarts."""
import copy
import random
import json
from pathlib import Path
from collections import Counter
import numpy as np
import pytest
import torch
import yaml
from stable_baselines3 import PPO
from escort_rl.ppo_training import read_config,make_vec,checkpoint,provenance,run_training
from escort_rl.sb3_restart import clear_pending_resets,restore_checkpoint_rng,PROTOCOL
from escort_rl.stage1_scenarios import load_config,sample_episode
from escort_rl.ppo_evaluation import reserved,validation_cases


def config_file(tmp_path):
    c=read_config();env=load_config(c['environment_path']);env['environment']['time_limit_s']=.2
    p=tmp_path/'env.yaml';p.write_text(yaml.safe_dump(env,sort_keys=False));c['environment_path']=str(p)
    return c,env,json.loads(Path(c['split_path']).read_text())


def expected_sequences(states,config,split,count):
    result=[]
    for saved in states:
        rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(saved['rng']);items=[]
        while len(items)<count:
            seed=int(rng.integers(0,2**60))*split['seed_modulus']+split['train_residue']
            meta=sample_episode(seed,config=config).scenario_metadata()
            if not reserved(meta,split):items.append((seed,meta['scenario_id'],meta['family'],meta['route_length_m']))
        result.append(items)
    return result


def sampled(vec):
    return [(m['episode_seed'],m['scenario_id'],m['family'],m['route_length_m']) for m in vec.get_attr('current')]


@pytest.mark.parametrize('workers',[1,2])
def test_multiple_completed_episodes_and_three_resumes(tmp_path,workers):
    torch.set_num_threads(1);c,env,split=config_file(tmp_path);v=make_vec(c,workers,73)
    snapshots=[];observed=[];original_start=None;model=None
    try:
        model=PPO('MlpPolicy',v,seed=19,n_steps=2,batch_size=2,n_epochs=1,device='cpu')
        clear_pending_resets(v);v.reset();original_start=sampled(v)
        # Multiple completed physical episodes + real optimizer updates before checkpoint.
        model.learn(4*workers,reset_num_timesteps=False)
        assert all(s['episodes']>=5 for s in v.env_method('rng_state'))
        path=tmp_path/'base';checkpoint(model,path,provenance(c));start_steps=model.num_timesteps
    finally:v.close()
    for restart in range(3):
        state=torch.load(str(path)+'.rng.pt',weights_only=False)
        expected=expected_sequences(state['environment_streams'],env,split,4)
        previous_weights={k:t.clone() for k,t in model.policy.state_dict().items()}
        previous_optimizer=copy.deepcopy(model.policy.optimizer.state_dict())
        v=make_vec(c,workers,73)
        try:
            model=PPO.load(path,env=v,device='cpu',force_reset=True)
            assert any(s is not None for s in v._seeds) # The exact SB3 hazard exists.
            v.set_options({'unexpected_pending_option':'must be cancelled'})
            assert restore_checkpoint_rng(model,v,state)==PROTOCOL
            assert torch.equal(torch.get_rng_state(),state['torch'])
            assert np.array_equal(np.random.get_state()[1],state['numpy'][1])
            assert np.random.get_state()[0]==state['numpy'][0] and np.random.get_state()[2:]==state['numpy'][2:]
            assert random.getstate()==state['python']
            assert v._seeds==[None]*workers and v._options==[{}]*workers
            assert all(torch.equal(previous_weights[k],t) for k,t in model.policy.state_dict().items())
            assert model.policy.optimizer.state_dict()['param_groups']==previous_optimizer['param_groups']
            for key,values in previous_optimizer['state'].items():
                for field,value in values.items():
                    actual=model.policy.optimizer.state_dict()['state'][key][field]
                    assert torch.equal(value,actual) if torch.is_tensor(value) else value==actual
            assert model.num_timesteps==state['timesteps']
            v.reset()
            assert torch.equal(torch.get_rng_state(),state['torch'])
            for j in range(4):
                current=sampled(v)
                for worker in range(workers):
                    assert current[worker]==expected[worker][j]
                    assert current[worker][0]%4==0
                    assert current[worker][0]!=original_start[worker][0]
                    observed.append(current[worker]);snapshots.append((restart,worker,current[worker]))
                if workers==2:assert current[0][0]!=current[1][0]
                if j<3:
                    _,_,done,_=v.step(np.zeros((workers,3),np.float32));assert done.all()
            model.learn(2*workers,reset_num_timesteps=False)
            assert model.num_timesteps==start_steps+(restart+1)*2*workers
            path=tmp_path/f'restart_{restart}';checkpoint(model,path,provenance(c))
        finally:v.close()
    assert len({s[0] for s in observed})==len(observed)
    assert len({s[1] for s in observed})==len(observed)
    assert len({round(s[3],5) for s in observed})>len(observed)//2
    val={c['episode_seed'] for c in validation_cases(env,split)}
    assert not val & {s[0] for s in observed}
    assert all(s[0]%4 != split['final_test_residue'] for s in observed)
    print('RNG_EVIDENCE',json.dumps(dict(workers=workers,resumes=3,seeds=len(observed),
        unique_seeds=len({s[0] for s in observed}),unique_signatures=len({s[1] for s in observed}),
        families=dict(Counter(s[2] for s in observed)),per_worker={str(w):[s[2][0] for s in snapshots if s[1]==w] for w in range(workers)})))


def test_corrupt_restore_is_explicit_and_atomic_for_worker_states(tmp_path):
    c,_,_=config_file(tmp_path);v=make_vec(c,1,42)
    try:
        model=PPO('MlpPolicy',v,n_steps=2,batch_size=2);clear_pending_resets(v);v.reset()
        p=tmp_path/'checkpoint';checkpoint(model,p,provenance(c));state=torch.load(str(p)+'.rng.pt',weights_only=False)
        before=v.env_method('rng_state')
        for bad in ({},dict(state,environment_streams=[]),dict(state,timesteps=123),
                    dict(state,environment_streams=[{'rng':{},'episodes':1}])):
            with pytest.raises((ValueError,KeyError)):restore_checkpoint_rng(model,v,bad)
            assert before==v.env_method('rng_state')
        restore_checkpoint_rng(model,v,state)
        with pytest.raises(RuntimeError,match='reseed'):v.envs[0].reset(seed=123)
        with pytest.raises(ValueError):v.envs[0].reset(options={'family':'straight'})
    finally:v.close()


def test_legacy_checkpoint_rejected_without_hash_bypass(tmp_path):
    from escort_rl.contract import PACKAGE
    legacy=PACKAGE/'runs/ppo_stage1_20261105/pilot/final.zip'
    # Existing research artifact is intentionally read-only; required for this repository audit.
    assert legacy.exists()
    with pytest.raises(ValueError,match='restart_protocol'):
        run_training(PACKAGE/'configs/ppo_stage1.yaml',42,2,4,tmp_path/'rejected',legacy,validation=False)


def test_unsupported_sb3_version_rejected(monkeypatch):
    import escort_rl.sb3_restart as adapter
    monkeypatch.setattr(adapter.importlib.metadata,'version',lambda _: '2.8.0')
    with pytest.raises(RuntimeError,match='2.7.0'):clear_pending_resets(None)


def test_comparison_cli_budget_without_executing_experiment(tmp_path,monkeypatch):
    import importlib.util,sys
    from escort_rl.contract import PACKAGE
    spec=importlib.util.spec_from_file_location('diagnostic_train_cli',PACKAGE/'tools/train_stage1_ppo.py')
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli);calls=[]
    def record(config,seed,workers,timesteps,output,resume):
        calls.append(dict(config=config,seed=seed,workers=workers,timesteps=timesteps,output=output,resume=resume))
        output.mkdir();return dict(new_transitions=timesteps)
    monkeypatch.setattr(cli,'run_training',record)
    monkeypatch.setattr(sys,'argv',['train_stage1_ppo.py','--chunks','2','--workers','2','--output',str(tmp_path/'comparison')])
    cli.main()
    assert len(calls)==2 and sum(c['timesteps'] for c in calls)==20000
    assert calls[0]['resume'] is None and calls[1]['resume']==calls[0]['output']/'final.zip'
    assert calls[0]['seed']==calls[1]['seed']==42
