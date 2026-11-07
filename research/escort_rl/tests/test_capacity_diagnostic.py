"""Capacity-test boundaries and exploration-ablation isolation."""
import copy
import json
import numpy as np
import torch
from stable_baselines3 import PPO
from escort_rl.capacity_diagnostic import fit_actor,dataset_partition,actor_parameters,teacher_function
from escort_rl.env import G1EscortEnv
from escort_rl.ppo_training import read_config
from escort_rl.contract import PACKAGE


def test_episode_partition_and_actor_only_updates():
    torch.set_num_threads(1);env=G1EscortEnv()
    try:
        c=read_config();model=PPO('MlpPolicy',env,seed=11,device='cpu',**c['ppo'])
        rng=np.random.default_rng(11);x=np.zeros((40,40),np.float32)
        for i in range(40):
            x[i,0:2]=rng.uniform(-.1,.1,2);x[i,5]=1.;x[i,6]=-.5;x[i,8:10]=1.;x[i,15]=-1.
        y=np.array([teacher_function()(obs,env.config) for obs in x],np.float32)
        ids=np.repeat(np.arange(4,44,4),4);train,hold=dataset_partition(ids)
        assert not set(ids[train])&set(ids[hold])
        old={k:v.clone() for k,v in model.policy.state_dict().items()}
        r=fit_actor(model,x,y,ids,40,42)
        assert np.isfinite(r['holdout_mse']) and r['updates']==40
        assert sum(p.numel() for p in actor_parameters(model))==22147
        assert any(not torch.equal(v,model.policy.state_dict()[k]) for k,v in old.items() if k.startswith('action_net'))
        assert all(torch.equal(v,model.policy.state_dict()[k]) for k,v in old.items() if k.startswith(('mlp_extractor.value_net','value_net')) or k=='log_std')
        assert model.num_timesteps==0 and model._n_updates==0
    finally:env.close()


def test_low_exploration_changes_only_log_std():
    a=read_config(PACKAGE/'configs/ppo_stage1_fixed_legacy_v1.yaml')
    b=read_config(PACKAGE/'configs/ppo_stage1_low_exploration_v1.yaml')
    assert b['ppo']['policy_kwargs'].pop('log_std_init')==-1.
    assert a==b
    assert np.isclose(np.exp(-1),.36787944117)
