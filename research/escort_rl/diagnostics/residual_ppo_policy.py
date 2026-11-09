"""SB3 2.7 residual PPO: Gaussian LATENT action, explicit execution transform.
No probability density is assigned to the clipped final body command.
"""
import copy,sys
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import DiagGaussianDistribution
sys.path.insert(0,str(Path(__file__).resolve().parent))
import stage2_residual as core
LATENT_LIMIT=20.0  # SB3 requires finite Box bounds; these are NOT velocity limits.

class MaskedGaussian(DiagGaussianDistribution):
    def __init__(self,mask):super().__init__(3);self.mask=mask
    def sample(self):return torch.where(self.mask[:,None],super().sample(),torch.zeros_like(self.distribution.mean))
    def mode(self):return torch.where(self.mask[:,None],self.distribution.mean,torch.zeros_like(self.distribution.mean))
    def log_prob(self,actions):return torch.where(self.mask,super().log_prob(actions),torch.zeros(actions.shape[0],device=actions.device))
    def entropy(self):return torch.where(self.mask,super().entropy(),torch.zeros(len(self.mask),device=self.mask.device))

def execute(context,observation,latent):
    x=np.atleast_2d(np.asarray(observation,np.float32));u=np.atleast_2d(np.asarray(latent,np.float32))
    if x.shape[1]!=40 or u.shape!=(len(x),3) or not np.isfinite(x).all() or not np.isfinite(u).all():raise ValueError('Invalid observation/latent action')
    if np.any(x[:,[33,39]]>=.5):raise ValueError('Only single-pedestrian Stage 2 supported')
    base=np.atleast_2d(context.base_action(x));out=base.copy();raw=base.copy();mask=core.gate_mask(x)
    if mask.any():
        delta=context.c['residual_limit']*torch.tanh(torch.as_tensor(np.clip(u[mask],-LATENT_LIMIT,LATENT_LIMIT))).numpy();raw[mask]=base[mask]+delta;out[mask]=np.clip(raw[mask],-1,1)
    return out,base,raw

class ResidualActorCriticPolicy(ActorCriticPolicy):
    def __init__(self,*args,residual_source,latent_sigma=.01,**kwargs):
        self.residual_source=str(residual_source);self.latent_sigma=float(latent_sigma)
        kwargs['ortho_init']=False;kwargs['use_sde']=False
        super().__init__(*args,**kwargs)
    def _build(self,lr_schedule):
        # Loading the immutable reference must not reseed the model-init RNG.
        with torch.random.fork_rng():self.context,_=core.load(self.residual_source)
        self.residual_net=copy.deepcopy(self.context.net)
        for p in self.context.net.parameters():p.requires_grad_(False)
        self.value_net=nn.Sequential(nn.Linear(core.FEATURES,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,1))
        self.log_std=nn.Parameter(torch.full((3,),float(np.log(self.latent_sigma))),requires_grad=False)
        self.optimizer=self.optimizer_class([p for p in self.parameters() if p.requires_grad],lr=lr_schedule(1),**self.optimizer_kwargs)
    def inputs(self,obs):
        x=obs.detach().cpu().numpy().astype(np.float32);base=np.atleast_2d(self.context.base_action(x));mask=core.gate_mask(x)
        if np.any(x[:,[33,39]]>=.5):raise ValueError('Unsupported pedestrian slots')
        z=torch.as_tensor(core.features(x,base,self.context.bc,self.context.c),device=self.device)
        mean=torch.zeros((len(x),3),device=self.device)
        if mask.any():
            active=core.features(x[mask],base[mask],self.context.bc,self.context.c);mean[torch.as_tensor(mask,device=self.device)]=self.residual_net(torch.as_tensor(active,device=self.device))
        return z,mean,torch.as_tensor(mask,device=self.device)
    def get_distribution(self,obs):
        _,mean,mask=self.inputs(obs);return MaskedGaussian(mask).proba_distribution(mean,self.log_std)
    def forward(self,obs,deterministic=False):
        z,mean,mask=self.inputs(obs);dist=MaskedGaussian(mask).proba_distribution(mean,self.log_std);actions=dist.get_actions(deterministic=deterministic)
        return actions,self.value_net(z),dist.log_prob(actions)
    def evaluate_actions(self,obs,actions):
        z,mean,mask=self.inputs(obs);dist=MaskedGaussian(mask).proba_distribution(mean,self.log_std)
        return self.value_net(z),dist.log_prob(actions),dist.entropy()
    def predict_values(self,obs):
        z,_,_=self.inputs(obs);return self.value_net(z)
    def _predict(self,observation,deterministic=False):return self.get_distribution(observation).get_actions(deterministic=deterministic)
    def assert_frozen(self):
        self.context.assert_frozen()
        if self.log_std.requires_grad:raise RuntimeError('Exploration variance unexpectedly trainable')

class LatentResidualEnv(gym.Wrapper):
    def __init__(self,env,source):
        super().__init__(env)
        with torch.random.fork_rng():self.context,_=core.load(source)
        self.action_space=gym.spaces.Box(-LATENT_LIMIT,LATENT_LIMIT,shape=(3,),dtype=np.float32);self.last_obs=None
    def reset(self,**kwargs):
        obs,info=self.env.reset(**kwargs);self.last_obs=obs.copy();return obs,info
    def scenario_metadata(self):return self.env.scenario_metadata()
    def step(self,latent):
        out,base,raw=execute(self.context,self.last_obs,latent);obs,r,t,tr,info=self.env.step(out[0]);self.last_obs=obs.copy()
        info.update(ppo_latent=np.asarray(latent).tolist(),ppo_base_action=base[0].tolist(),ppo_executed_action=out[0].tolist(),ppo_residual_action=(out[0]-base[0]).tolist(),ppo_action_saturated=bool(np.any(np.abs(raw[0])>1)))
        return obs,r,t,tr,info

class ExecutedPolicy:
    """Only this adapter feeds the ordinary normalized-action Stage2Env/evaluator."""
    def __init__(self,model):self.model=model
    def predict(self,observation,deterministic=True):
        latent,_=self.model.predict(observation,deterministic=deterministic);out,_,_=execute(self.model.policy.context,observation,latent)
        return (out[0] if np.asarray(observation).ndim==1 else out),None
