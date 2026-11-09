import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
import gymnasium as gym

p=Path(__file__).resolve().parents[1]/'diagnostics/stage2_teacher_bootstrap.py'
spec=importlib.util.spec_from_file_location('stage2_teacher_bootstrap',p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

def test_gate_rejects_missing_and_failed_families():
    rows=[dict(family=f,success=True,reason='success') for f in mod.FAMILIES]
    assert mod.teacher_gate(rows)==[]
    rows[0]['success']=False;rows[0]['reason']='pedestrian_intrusion'
    assert mod.teacher_gate(rows)
    assert mod.teacher_gate([])
    rows.append(dict(family='straight',success=True,reason='physics_failure'))
    assert 'critical_failure' in mod.teacher_gate(rows)

def test_split_is_disjoint_whole_episodes():
    families=np.repeat(mod.FAMILIES,20);ids=np.repeat(np.arange(len(mod.FAMILIES)*4),5)
    train,hold=mod.episode_split(ids,families)
    assert not set(ids[train])&set(ids[hold])
    assert set(families[train])==set(mod.FAMILIES)==set(families[hold])
    with pytest.raises(ValueError):mod.episode_split(np.zeros(6,int),np.asarray(mod.FAMILIES))

class Toy(gym.Env):
    observation_space=gym.spaces.Box(-1,1,(40,),dtype=np.float32)
    action_space=gym.spaces.Box(-1,1,(3,),dtype=np.float32)
    def reset(self,seed=None,options=None):super().reset(seed=seed);return np.zeros(40,np.float32),{}
    def step(self,a):return np.zeros(40,np.float32),0.,False,True,{}

def test_actor_fit_preserves_critic_variance_and_counters():
    torch.set_num_threads(1)
    model=PPO('MlpPolicy',Toy(),policy_kwargs=dict(net_arch=dict(pi=[128,128],vf=[128,128])),seed=42,n_steps=8,batch_size=8)
    before={n:p.detach().clone() for n,p in model.policy.state_dict().items()}
    rng=np.random.default_rng(6);x=rng.uniform(-1,1,(120,40)).astype(np.float32);y=np.stack([x[:,0]*.2,x[:,22]*.3,x[:,23]*.1],axis=1)
    families=np.repeat(mod.FAMILIES,20);ids=np.repeat(np.arange(24),5)
    report=mod.fit_actor(model,x,y,ids,families,updates=100)
    assert report['critic_and_log_std_unchanged'] and model.num_timesteps==0
    assert any(not torch.equal(v,model.policy.state_dict()[n]) for n,v in before.items() if n.startswith(mod.ACTOR_PREFIXES))
    assert all(torch.equal(v,model.policy.state_dict()[n]) for n,v in before.items() if not n.startswith(mod.ACTOR_PREFIXES))

def test_collection_enforces_training_partition(tmp_path):
    with pytest.raises(ValueError):mod.collect(tmp_path/'invalid',count=3,seed=852001)
