"""Opt-in reward candidate. Original rewards.py and base environment stay unchanged."""
import numpy as np
from .env import G1EscortEnv
from .rewards import EpisodeEvaluator

VERSIONS=('legacy_v1','formation_progress_v2','tracking_stop_v3')


class FormationProgressEvaluator(EpisodeEvaluator):
    """Same weights/outcomes/highwater; progress cap follows the desired escort point.

    project_robot is evaluator-only route geometry. No future pose/time query is used.
    """
    def __init__(self,config,initial_robot_progress,route_length,project_robot):
        super().__init__(config,initial_robot_progress,route_length)
        self.project_robot=project_robot
        self.formation_highwater=float(initial_robot_progress)

    def update(self,**kwargs):
        client=kwargs['client'];h=client.yaw
        R=np.array([[np.cos(h),-np.sin(h)],[np.sin(h),np.cos(h)]])
        desired=np.array([client.x,client.y])+R@self.config['environment']['desired_offset_m']
        cap=float(self.project_robot(desired))
        actual_client_progress=kwargs['client_progress']
        # Two nondecreasing ledgers: bounded formation progress and raw robot progress.
        # Both must advance, preventing credit from client motion while the robot waits.
        bounded=min(float(kwargs['robot_progress']),cap)
        increment=min(max(0.,bounded-self.formation_highwater),
                      max(0.,float(kwargs['robot_progress'])-self.highwater))
        self.formation_highwater=max(self.formation_highwater,bounded)
        # The unchanged base equation gates eligibility and computes all other terms.
        # This equivalent synthetic budget makes its increment equal the approved one;
        # restore the true client progress in diagnostics below.
        budget=self.highwater+increment
        reward,term,trunc,info=super().update(**dict(kwargs,client_progress=budget))
        info['client_progress_m']=actual_client_progress
        info['desired_formation_progress_m']=cap
        info['reward_version']='formation_progress_v2'
        return reward,term,trunc,info


class FormationProgressEnv(G1EscortEnv):
    """Reward-only adapter; inherits original reset, stepping, observations and physics."""
    def reset(self,**kwargs):
        observation,info=super().reset(**kwargs)
        original=self._evaluator
        self._evaluator=FormationProgressEvaluator(self.config,original.initial_progress,
            original.route_length,self._scenario.project_robot)
        info['reward_version']='formation_progress_v2'
        return observation,info


def make_environment(config_path=None,*,config=None,reward_version='legacy_v1'):
    if reward_version not in VERSIONS:raise ValueError(f'Unknown reward version: {reward_version}')
    if reward_version=='tracking_stop_v3':
        from .tracking_objective import TrackingStopEnv
        cls=TrackingStopEnv
    else:
        cls=G1EscortEnv if reward_version=='legacy_v1' else FormationProgressEnv
    kwargs={} if config_path is None else {'config_path':config_path}
    if config is not None:kwargs['config']=config
    return cls(**kwargs)
