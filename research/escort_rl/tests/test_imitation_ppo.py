"""Focused regression checks for explicit BC import and value-only preparation."""
import copy
import json
import numpy as np
import pytest
import torch
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from escort_rl.contract import PACKAGE
from escort_rl.ppo_training import read_config,checkpoint,TimedPPO,exact_learn
from escort_rl.imitation_ppo import (checked_source,initialize_actor,subset,ACTOR_PREFIXES,
    CRITIC_PREFIXES,returns,fit_critic,development_cases,noise_pass,preservation_pass)
from escort_rl.sb3_restart import restore_checkpoint_rng,clear_pending_resets
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback


class Toy(gym.Env):
    observation_space=gym.spaces.Box(-1,1,(40,),np.float32)
    action_space=gym.spaces.Box(-1,1,(3,),np.float32)
    def __init__(self):self.rng=np.random.default_rng(42);self.count=0
    def reset(self,seed=None,options=None):
        self.count+=1;return self.rng.uniform(-1,1,40).astype(np.float32),{}
    def step(self,a):return self.reset()[0],1.,True,False,{}
    def rng_state(self):return dict(rng=copy.deepcopy(self.rng.bit_generator.state),episodes=self.count)
    def set_rng_state(self,s):self.rng.bit_generator.state=s['rng'];self.count=s['episodes']


def model(seed=1):
    torch.set_num_threads(1)
    return TimedPPO('MlpPolicy',DummyVecEnv([Toy]),seed=seed,n_steps=8,batch_size=8,n_epochs=1,
                    policy_kwargs=dict(net_arch=dict(pi=[128,128],vf=[128,128])))


def config():return read_config(PACKAGE/'configs/ppo_stage1_imitation_warmstart_v1.yaml')


def test_checked_source_and_reject_hash():
    c=config();source=checked_source(c);assert source.num_timesteps==0
    c=copy.deepcopy(c);c['imitation']['actor_sha256']='bad'
    with pytest.raises(ValueError,match='hash mismatch'):checked_source(c)


def test_import_only_actor_and_sigma():
    a=model(1);b=model(2);critic=subset(a.policy,CRITIC_PREFIXES)
    initialize_actor(a,b,.05)
    assert all(torch.equal(v,subset(a.policy,ACTOR_PREFIXES)[k]) for k,v in subset(b.policy,ACTOR_PREFIXES).items())
    assert all(torch.equal(v,subset(a.policy,CRITIC_PREFIXES)[k]) for k,v in critic.items())
    assert np.allclose(a.policy.log_std.detach().exp(),.05)
    assert not a.policy.optimizer.state and a.num_timesteps==0
    a.num_timesteps=1
    with pytest.raises(ValueError,match='fresh'):initialize_actor(a,b,.05)
    a.get_env().close();b.get_env().close()


def test_return_termination_and_bootstrap():
    assert np.allclose(returns([1,2],.5),[2,2])
    assert np.allclose(returns([1,2],.5,4),[3,4])


def test_critic_fit_actor_frozen_whole_episode_holdout():
    m=model();rng=np.random.default_rng(7);x=rng.uniform(-1,1,(1000,40)).astype(np.float32)
    y=(x[:,0]*4+x[:,1]*2-3).astype(np.float32);ids=np.repeat(np.arange(10),100)
    before=subset(m.policy,ACTOR_PREFIXES+('log_std',));cfg=config()['imitation'].copy();cfg['critic_updates']=500
    result=fit_critic(m,x,y,ids,cfg)
    assert result['passes'] and result['final']['holdout_explained_variance']>.8
    assert not set(result['train_episodes']) & set(result['holdout_episodes'])
    assert all(torch.equal(v,subset(m.policy,ACTOR_PREFIXES+('log_std',))[k]) for k,v in before.items())
    assert not m.policy.optimizer.state
    assert all(p.requires_grad for p in m.policy.parameters())
    m.get_env().close()


def test_development_namespace_and_gates():
    c=config();cases=development_cases(c)
    assert len(cases)==12 and all(x['episode_seed']%4==0 for x in cases)
    assert cases==development_cases(c)
    summary=dict(success_rate=1.,per_family={f:dict(success_rate=1.) for f in ('straight','gentle_turn','stop_start')},
                 formation_mean_m=.06,failures={'success':12})
    assert noise_pass(summary,c['imitation']) and preservation_pass(summary)
    summary['failures']['supervisor_failure']=1
    assert not noise_pass(summary,c['imitation']) and not preservation_pass(summary)


class Continue(BaseCallback):
    def _on_step(self):return True


def test_prepared_checkpoint_optimizer_and_rng_resume(tmp_path):
    m=model();s=model(2);initialize_actor(m,s,.05);clear_pending_resets(m.get_env())
    m.set_logger(configure(str(tmp_path/'logs'),[]));exact_learn(m,16,Continue())
    checkpoint(m,tmp_path/'saved',dict(initialization_protocol='imitation_pretrained_ppo_v1'))
    state=torch.load(tmp_path/'saved.rng.pt',weights_only=False);before=subset(m.policy,ACTOR_PREFIXES)
    v=DummyVecEnv([Toy]);loaded=TimedPPO.load(tmp_path/'saved.zip',env=v)
    restore_checkpoint_rng(loaded,v,state)
    assert loaded.num_timesteps==16 and loaded.policy.optimizer.state
    assert all(torch.equal(value,subset(loaded.policy,ACTOR_PREFIXES)[k]) for k,value in before.items())
    assert v.env_method('rng_state')==state['environment_streams']
    loaded.set_logger(configure(str(tmp_path/'logs2'),[]));exact_learn(loaded,16,Continue())
    assert loaded.num_timesteps==32 and loaded.gradient_steps==4
    v.close();m.get_env().close();s.get_env().close()
