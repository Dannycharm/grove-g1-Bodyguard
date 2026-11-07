"""Evaluator-only formation, nonrepeatable robot progress and explicit episode outcomes."""
import numpy as np


def formation_error(robot_xy,client,offset):
    h=client.yaw;R=np.array([[np.cos(h),-np.sin(h)],[np.sin(h),np.cos(h)]])
    local=R.T@(np.asarray(robot_xy)-[client.x,client.y])
    return local-np.asarray(offset),local


class EpisodeEvaluator:
    def __init__(self,config,initial_robot_progress,route_length):
        self.config=config;self.initial_progress=float(initial_robot_progress);self.highwater=float(initial_robot_progress)
        self.route_length=float(route_length);self.loss_seconds=0.;self.previous_action=np.zeros(3)
        self.accompanied_progress=0.

    def update(self,*,elapsed,dt,robot_xy,client,robot_progress,client_progress,client_arrived,action,
               fallen=False,emergency=False,physics_failure=False,out_of_bounds=False,torso_tilt=0.):
        c=self.config;r=c['reward'];o=c['outcomes']
        error,local=formation_error(robot_xy,client,c['environment']['desired_offset_m'])
        norm=float(np.linalg.norm(error));distance=float(np.linalg.norm(np.asarray(robot_xy)-[client.x,client.y]))
        self.loss_seconds=self.loss_seconds+dt if distance>o['client_distance_limit_m'] or not client.valid else 0.
        eligible=(norm<=r['accompaniment_error_limit_m'] and distance<=o['client_distance_limit_m']
                  and local[0]<=r['ahead_tolerance_m'] and client.valid)
        new_progress=max(0.,min(float(robot_progress),float(client_progress))-self.highwater) if eligible else 0.
        self.accompanied_progress+=new_progress
        # Even ineligible racing consumes traversed progress; it cannot be banked for later credit.
        self.highwater=max(self.highwater,float(robot_progress))
        robot_advance=max(0.,self.highwater-self.initial_progress)
        reason='running';terminated=False;truncated=False
        if physics_failure:reason='physics_failure'
        elif fallen:reason='fall'
        elif emergency:reason='supervisor_failure'
        elif out_of_bounds:reason='out_of_bounds'
        elif self.loss_seconds+1e-9>=o['client_loss_grace_s']:reason='client_lost'
        elif client_arrived:
            if (norm<=o['final_formation_tolerance_m'] and
                    robot_advance>=o['minimum_robot_progress_fraction']*self.route_length and
                    self.accompanied_progress>=o['minimum_robot_progress_fraction']*self.route_length):
                reason='success'
            elif elapsed-client_arrived>=o['recovery_window_s']-1e-9:reason='recovery_timeout'
        if reason!='running':terminated=True
        elif elapsed+1e-9>=c['environment']['time_limit_s']:reason='time_limit';truncated=True
        components=dict(formation=r['formation_weight']*(np.exp(-(norm/r['formation_scale_m'])**2)-1)*dt,
            progress=r['progress_weight']*new_progress,
            command_change=-r['command_change_weight']*float(np.sum((np.asarray(action)-self.previous_action)**2)),
            torso_stability=-r['torso_stability_weight']*max(0.,float(torso_tilt))*dt,
            terminal_success=r['success_reward'] if reason=='success' else 0.,
            terminal_failure=-r['failure_penalty'] if terminated and reason!='success' else 0.)
        self.previous_action=np.asarray(action).copy()
        return float(sum(components.values())),terminated,truncated,dict(reason=reason,reward_components=components,
            formation_error_client_frame=error.tolist(),formation_error_m=norm,client_distance_m=distance,
            robot_progress_m=float(robot_progress),client_progress_m=float(client_progress),new_robot_progress_m=new_progress,
            robot_advance_m=robot_advance,accompanied_robot_progress_m=self.accompanied_progress,client_loss_seconds=self.loss_seconds)
