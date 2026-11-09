"""Trained-optimizer critic preparation must never change the actor or its states."""
import importlib.util
import copy
import numpy as np
import pytest
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from escort_rl.contract import PACKAGE

spec=importlib.util.spec_from_file_location('critic_prep',PACKAGE/'diagnostics/prepare_stage2_critic.py');prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)


class Toy(gym.Env):
    observation_space=gym.spaces.Box(-1,1,(40,),np.float32)
    action_space=gym.spaces.Box(-1,1,(3,),np.float32)
    def reset(self,seed=None,options=None):return np.zeros(40,np.float32),{}
    def step(self,a):return np.zeros(40,np.float32),0.,True,False,{}


def make_model():
    torch.set_num_threads(1)
    model=PPO('MlpPolicy',Toy(),device='cpu',seed=7,n_steps=8,batch_size=8,policy_kwargs=dict(net_arch=dict(pi=[128,128],vf=[128,128])))
    # Prime nonempty Adam states as a trained-checkpoint stand-in before the test.
    loss=sum(p.square().mean() for p in model.policy.parameters());model.policy.optimizer.zero_grad();loss.backward();model.policy.optimizer.step()
    return model


def data():
    rng=np.random.default_rng(9);episodes=24;n=30;count=episodes*n
    x=rng.uniform(-.8,.8,(count,40)).astype(np.float32);ids=np.repeat(np.arange(episodes,dtype=np.int32),n);families=np.repeat(np.tile(prep.FAMILIES,4),n)
    fail=np.isin(families,['crossing','diagonal']);x[:,22]=np.where(fail,1,-1);y=(np.where(fail,-5.,2.)+.2*x[:,0]).astype(np.float32)
    return dict(observations=x,returns=y,episode_ids=ids,families=families,near_intrusion=fail.copy())


def test_stratified_whole_episode_holdout_and_missing_family_rejected():
    d=data();c=prep.read_preparation(prep.DEFAULT_PREP);a,b=prep.split_episodes(d,c);a2,b2=prep.split_episodes(d,c)
    assert np.array_equal(a,a2) and np.array_equal(b,b2)
    assert not set(d['episode_ids'][a]) & set(d['episode_ids'][b])
    assert set(d['families'][b])==set(prep.FAMILIES)
    bad={k:v[d['families']!='diagonal'] for k,v in d.items()}
    with pytest.raises(ValueError,match='diagonal'):prep.split_episodes(bad,c)


def test_fit_freezes_actor_and_optimizer_then_value_only_reset_and_reload(tmp_path):
    model=make_model();d=data();c=prep.read_preparation(prep.DEFAULT_PREP);c.update(updates=700,score_interval=100)
    protected=prep.digest_state(prep.protected_weights(model));actor_optimizer=prep.digest_state(prep.optimizer_actor_state(model));counter=model.num_timesteps
    result=prep.fit(model,d,c)
    assert result['accepted'] and not result['final_gate_reasons']
    assert result['final']['near_intrusion']['negative_fraction']>=.8
    assert result['final']['holdout']['mse']<result['initial']['holdout']['mse']
    assert protected==prep.digest_state(prep.protected_weights(model))
    assert actor_optimizer==prep.digest_state(prep.optimizer_actor_state(model)) and model.num_timesteps==counter
    reset=prep.reset_value_optimizer(model);assert len(reset)==6
    for name,p in model.policy.named_parameters():
        assert (p not in model.policy.optimizer.state) if name.startswith(prep.CRITIC_PREFIXES) else (p in model.policy.optimizer.state)
    assert actor_optimizer==prep.digest_state(prep.optimizer_actor_state(model))
    model.save(tmp_path/'prepared.zip');other=PPO.load(tmp_path/'prepared.zip',device='cpu')
    assert protected==prep.digest_state(prep.protected_weights(other))
    assert actor_optimizer==prep.digest_state(prep.optimizer_actor_state(other))
    model.get_env().close()


def test_current_actor_collection_one_full_g1_episode_without_model_updates(tmp_path):
    from escort_rl.stage2.training import load_stage2
    from escort_rl.stage2.scenarios import DEFAULT
    model,_=load_stage2(prep.DEFAULT_MODEL,DEFAULT);model.policy.set_training_mode(False)
    c=prep.read_preparation(prep.DEFAULT_PREP);c['episodes']=1
    protected=prep.digest_state(prep.protected_weights(model));optimizer=prep.digest_state(model.policy.optimizer.state_dict());counter=model.num_timesteps
    d=prep.collect(model,DEFAULT,c,tmp_path)
    assert d['observations'].shape[1]==40 and d['observations'].dtype==np.float32 and np.isfinite(d['returns']).all()
    assert len(set(d['episode_ids']))==1 and len(d['returns'])>1
    episodes=__import__('json').loads((tmp_path/'episodes.json').read_text());assert episodes[0]['episode_seed']%4==0
    assert model.num_timesteps==counter and protected==prep.digest_state(prep.protected_weights(model))
    assert optimizer==prep.digest_state(model.policy.optimizer.state_dict())


def test_gate_explicitly_rejects_unimproved_critic():
    model=make_model();d=data();c=prep.read_preparation(prep.DEFAULT_PREP);train,hold=prep.split_episodes(d,c);s=prep.scores(model,d,train,hold)
    assert 'overall_holdout_not_better_than_parent_and_constant' in prep.gate(s,s,c)
    model.get_env().close()


def test_prepared_checkpoint_preserves_parent_rng_and_counters(tmp_path,monkeypatch):
    import json
    from escort_rl.stage2.training import load_stage2
    from escort_rl.stage2.scenarios import DEFAULT
    original_root=prep.ROOT
    monkeypatch.setattr(prep,'ROOT',tmp_path)
    registered=[];monkeypatch.setattr(prep,'register',lambda *args:registered.append(True))
    def mock_collection(model,path,c,out):
        d=data();np.savez_compressed(out/'rollouts.npz',**d)
        prep.write(out/'collection.json',dict(test_fixture=True,transitions=len(d['returns'])))
        return d
    monkeypatch.setattr(prep,'collect',mock_collection)
    result=prep.prepare(prep.DEFAULT_MODEL,DEFAULT,prep.DEFAULT_PREP,tmp_path/'prepared',updates=1500)
    assert result['status']=='accepted' and result['ppo_timesteps']==10000 and result['stage2_ppo_transitions']==5000 and registered
    source_state=torch.load(prep.DEFAULT_MODEL.with_suffix('.rng.pt'),weights_only=False)
    final_state=torch.load(tmp_path/'prepared/prepared.rng.pt',weights_only=False)
    assert source_state['environment_streams']==final_state['environment_streams']
    assert torch.equal(source_state['torch'],final_state['torch']) and source_state['python']==final_state['python']
    assert np.array_equal(source_state['numpy'][1],final_state['numpy'][1])
    model,meta=load_stage2(tmp_path/'prepared/prepared.zip',DEFAULT)
    parent,_=load_stage2(prep.DEFAULT_MODEL,DEFAULT)
    assert model.gradient_steps==parent.gradient_steps==394
    assert prep.digest_state(prep.protected_weights(model))==prep.digest_state(prep.protected_weights(parent))
    assert prep.digest_state(prep.optimizer_actor_state(model))==prep.digest_state(prep.optimizer_actor_state(parent))
    assert meta['critic_preparation']['actor_unchanged'] and meta['critic_preparation']['training_rng_unchanged']
