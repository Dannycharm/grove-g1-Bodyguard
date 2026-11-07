"""Opt-in v3 task: robust position + interval velocity tracking + sampled stop hold.

No future trajectory enters actor/critic. This changes reward AND completion criteria;
legacy/v2 remain available. No assertion of policy-invariant shaping or learned superiority.
"""
from pathlib import Path
import numpy as np
import yaml
from .contract import PACKAGE
from .env import G1EscortEnv
from .reward_variants import FormationProgressEvaluator

SPEC=PACKAGE/'configs/tracking_stop_v3.yaml'


def settings():
    c=yaml.safe_load(SPEC.read_text())
    required={'schema_version','version','formation_scale_m','formation_cost_cap','velocity_scale_m_s',
              'velocity_weight','velocity_cost_cap','stop_linear_speed_m_s','stop_instantaneous_speed_cap_m_s','stop_max_excursion_m','stop_yaw_rate_rad_s','stop_hold_s'}
    if set(c)!=required or c['schema_version']!=1 or c['version']!='tracking_stop_v3':raise ValueError('Invalid tracking_stop_v3 configuration')
    if any(not np.isfinite(c[k]) or c[k]<=0 for k in required-{'schema_version','version'}):raise ValueError('Positive finite tracking parameters required')
    return c


def robust_cost(value,cap):
    z=max(0.,float(value))
    return min(float(cap),z*z if z<=1 else 2*z-1)


def desired_point(client,offset):
    h=client.yaw;R=np.array([[np.cos(h),-np.sin(h)],[np.sin(h),np.cos(h)]])
    return np.array([client.x,client.y])+R@np.asarray(offset)


class TrackingStopEvaluator(FormationProgressEvaluator):
    def __init__(self,config,initial_robot_progress,route_length,project_robot,measure,
                 initial_robot_xy,initial_client,objective=None):
        super().__init__(config,initial_robot_progress,route_length,project_robot)
        self.objective=settings() if objective is None else dict(objective)
        self.measure=measure;self.previous_robot=np.array(initial_robot_xy,float)
        self.previous_desired=desired_point(initial_client,config['environment']['desired_offset_m'])
        self.hold_samples=[]

    def update(self,**kwargs):
        arrived=kwargs['client_arrived'];elapsed=float(kwargs['elapsed']);dt=float(kwargs['dt']);o=self.objective
        # Preserve all original safety/loss/time-limit handling; handle endpoint completion here.
        _,terminated,truncated,info=super().update(**dict(kwargs,client_arrived=None))
        point=desired_point(kwargs['client'],self.config['environment']['desired_offset_m'])
        average_robot=(np.array(kwargs['robot_xy'])-self.previous_robot)/dt
        average_target=(point-self.previous_desired)/dt
        velocity_error=average_robot-average_target
        self.previous_robot=np.array(kwargs['robot_xy'],float);self.previous_desired=point
        components=info['reward_components']
        components['formation']=-self.config['reward']['formation_weight']*dt*robust_cost(info['formation_error_m']/o['formation_scale_m'],o['formation_cost_cap'])
        components['velocity_tracking']=-o['velocity_weight']*dt*robust_cost(np.linalg.norm(velocity_error)/o['velocity_scale_m_s'],o['velocity_cost_cap'])
        row=self.measure();speed=float(np.linalg.norm(row['linear_velocity_world'][:2]));yaw_rate=row['euler_yaw_rate_world']
        geometry=bool(arrived is not None and info['formation_error_m']<=self.config['outcomes']['final_formation_tolerance_m']
                      and info['robot_advance_m']>=self.config['outcomes']['minimum_robot_progress_fraction']*self.route_length)
        qualified=geometry and info['accompanied_robot_progress_m']>=self.config['outcomes']['minimum_robot_progress_fraction']*self.route_length
        interval_speed=float(np.linalg.norm(average_robot))
        quiet_native=(speed<=o['stop_instantaneous_speed_cap_m_s'] and yaw_rate is not None and abs(yaw_rate)<=o['stop_yaw_rate_rad_s'])
        if qualified and quiet_native and not terminated:
            self.hold_samples.append((elapsed,np.array(kwargs['robot_xy'],float)))
            while len(self.hold_samples)>1 and self.hold_samples[1][0]<=elapsed-o['stop_hold_s']+1e-9:
                self.hold_samples.pop(0)
        else:self.hold_samples=[]
        span=0. if not self.hold_samples else elapsed-self.hold_samples[0][0]
        window_speed=0. if span<=1e-12 else float(np.linalg.norm(self.hold_samples[-1][1]-self.hold_samples[0][1])/span)
        excursion=0. if not self.hold_samples else max(float(np.linalg.norm(x-self.hold_samples[0][1])) for _,x in self.hold_samples)
        held=span if window_speed<=o['stop_linear_speed_m_s'] and excursion<=o['stop_max_excursion_m'] else 0.
        if not terminated and arrived is not None:
            if held+1e-9>=o['stop_hold_s']:
                terminated=True;truncated=False;info['reason']='success'
                components['terminal_success']=self.config['reward']['success_reward']
            elif elapsed-arrived+1e-9>=self.config['outcomes']['recovery_window_s']:
                terminated=True;truncated=False;info['reason']='recovery_timeout'
                components['terminal_failure']=-self.config['reward']['failure_penalty']
        info.update(reward_version='tracking_stop_v3',endpoint_geometry_qualified=geometry,
            endpoint_formation_and_progress_qualified=bool(qualified),stop_hold_s=held,
            stop_interval_speed_m_s=interval_speed,stop_native_speed_m_s=speed,stop_yaw_rate_rad_s=yaw_rate,
            stop_window_speed_m_s=window_speed,stop_window_excursion_m=excursion,
            relative_velocity_error_m_s=float(np.linalg.norm(velocity_error)),
            robot_interval_velocity_world=average_robot.tolist(),desired_interval_velocity_world=average_target.tolist(),
            success_definition='endpoint formation/progress plus sampled planar/yaw stop hold')
        reward=float(sum(components.values()))
        if not np.isfinite(reward):raise RuntimeError('Nonfinite tracking objective')
        return reward,terminated,truncated,info


class TrackingStopEnv(G1EscortEnv):
    def __init__(self,*args,**kwargs):
        self.tracking_config=settings()
        super().__init__(*args,**kwargs)

    def reset(self,**kwargs):
        obs,info=super().reset(**kwargs);original=self._evaluator;row=self._runner.measure()
        self._evaluator=TrackingStopEvaluator(self.config,original.initial_progress,original.route_length,
            self._scenario.project_robot,self._runner.measure,row['position'][:2],self._scenario.client_pose_at(0.),self.tracking_config)
        info['reward_version']='tracking_stop_v3'
        return obs,info
