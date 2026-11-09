"""Stage 2 adapter; unchanged AGILE/dynamics and unchanged first 22 observations."""
import numpy as np
import gymnasium as gym
from ..tracking_objective import TrackingStopEnv
from ..mujoco_runner import MujocoRunner
from .scenarios import load_config,sample_episode,DEFAULT


def swept_clearance(previous,current,radius):
    a=np.asarray(previous,float);v=np.asarray(current,float)-a
    fraction=0. if np.dot(v,v)<1e-24 else float(np.clip(-np.dot(a,v)/np.dot(v,v),0,1))
    return float(np.linalg.norm(a+fraction*v)-radius)


class Stage2Env(TrackingStopEnv):
    def __init__(self,stage2_config=DEFAULT):
        self.stage2_config=load_config(stage2_config);self._stage2_episode=None
        self._episode_min_clearance=float('inf');self._step_min_clearance=float('inf');self._previous_relative=None;self._intrusion=False
        super().__init__(config_path=self.stage2_config['stage1_config'])
        # Preserve original Stage 1 source and its checkpoint hashes. Initialization
        # briefly creates its runner, then replaces/closes it once; resets use only
        # the final independent two-proxy runner. No live authority exists.
        old=self._runner
        try:self._runner=MujocoRunner(scene='flat',actors=2,actor_names=['escort_client','escort_pedestrian_1'])
        finally:old.close()
        self._runner.actor_driver=self._drive_client
        self._client_slot=int(self._runner.model.body_mocapid[self._runner._id(self._runner.mj.mjtObj.mjOBJ_BODY,'escort_client')])
        self._pedestrian_slot=int(self._runner.model.body_mocapid[self._runner._id(self._runner.mj.mjtObj.mjOBJ_BODY,'escort_pedestrian_1')])
        self._torso=self._runner._id(self._runner.mj.mjtObj.mjOBJ_BODY,'torso_link')
        self._deviation_start=None;self._recovery_durations=[]

    def _drive_client(self,simulation_time,data):
        super()._drive_client(simulation_time,data)
        if not hasattr(self,'_pedestrian_slot'):return
        t=0. if self._origin is None else max(0.,simulation_time-self._origin)
        state=None if self._stage2_episode is None else self._stage2_episode.pedestrian_pose_at(t)
        if state is None:
            data.mocap_pos[self._pedestrian_slot]=[0,0,-10];data.mocap_quat[self._pedestrian_slot]=[1,0,0,0];return
        data.mocap_pos[self._pedestrian_slot]=[state.x,state.y,0]
        data.mocap_quat[self._pedestrian_slot]=[np.cos(state.yaw/2),0,0,np.sin(state.yaw/2)]
        if self._origin is not None:
            root=self._runner.model.jnt_qposadr[self._runner._free_joint]
            relative=np.array([state.x,state.y])-data.qpos[root:root+2]
            radius=self.stage2_config['pedestrian']['radius_m']+self.stage2_config['clearance']['robot_pelvis_footprint_radius_m']
            clearance=float(np.linalg.norm(relative)-radius) if self._previous_relative is None else swept_clearance(self._previous_relative,relative,radius)
            self._previous_relative=relative
            self._episode_min_clearance=min(self._episode_min_clearance,clearance);self._step_min_clearance=min(self._step_min_clearance,clearance)
            self._intrusion|=clearance<0.

    def _observe(self):
        obs,row,client,saturated,tilt=super()._observe()
        if self._stage2_episode is None:return obs,row,client,saturated,tilt
        ped=self._stage2_episode.pedestrian_pose_at(self._elapsed)
        if ped is not None:
            w,x,y,z=row['quaternion'];yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
            R=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
            relative=R.T@(np.array([ped.x,ped.y])-row['position'][:2]);velocity=R.T@(np.array([ped.vx,ped.vy])-row['linear_velocity_world'][:2])
            slot=np.r_[relative/self.config['observation']['position_scale_m'],velocity/self.config['observation']['relative_velocity_scale_m_s'],self.stage2_config['pedestrian']['radius_m'],float(ped.valid)]
            obs=obs.copy();obs[22:28]=np.clip(slot,-1,1);saturated+=int(np.sum(np.abs(slot)>1))
        return obs.astype(np.float32),row,client,saturated,tilt

    def reset(self,*,seed=None,options=None):
        gym.Env.reset(self,seed=seed);options={} if options is None else dict(options)
        if set(options)-{'family','episode_seed'}:raise ValueError('Supported options: family, episode_seed')
        episode_seed=int(options['episode_seed']) if 'episode_seed' in options else int(self.np_random.integers(0,2**60))
        self._stage2_episode=sample_episode(episode_seed,options.get('family'),self.stage2_config)
        self._origin=None;self._previous_relative=None;self._intrusion=False;self._episode_min_clearance=float('inf');self._step_min_clearance=float('inf');self._deviation_start=None;self._recovery_durations=[]
        metadata=self._stage2_episode.scenario_metadata()
        obs,info=super().reset(options=dict(family=metadata['client_family'],episode_seed=episode_seed))
        self._scenario=self._stage2_episode
        self._previous_relative=None;self._intrusion=False;self._episode_min_clearance=float('inf');self._step_min_clearance=float('inf')
        self._drive_client(self._runner.data.time,self._runner.data)
        info.update(stage='stage2',scenario_id=metadata['scenario_id'],episode_seed=episode_seed,pedestrian_count=metadata['pedestrian_count'],pedestrian_slot_schema='[relative_xy/16, relative_world_velocity_in_robot_heading_axes/2, radius_m/1m, validity]')
        return obs,info

    def step(self,action):
        self._step_min_clearance=float('inf')
        obs,reward,term,trunc,info=super().step(action)
        if info['reason']=='physics_failure':return obs,reward,term,trunc,info
        has_ped=self._stage2_episode.pedestrian is not None
        g=self.stage2_config['clearance'];clearance=self._step_min_clearance
        penalty=-g['shaping_weight']*self.config['environment']['dt_s']*min(1.,max(0.,(g['preferred_surface_margin_m']-clearance)/g['preferred_surface_margin_m']))**2 if has_ped else 0.
        components=info['reward_components'];components['pedestrian_clearance']=penalty
        if self._intrusion and info['reason'] not in ('fall','supervisor_failure','physics_failure'):
            term=True;trunc=False;info['reason']='pedestrian_intrusion';info['evaluation']['reason']='pedestrian_intrusion';components['terminal_success']=0.;components['terminal_failure']=-self.config['reward']['failure_penalty'];self._done=True
        error=info['evaluation']['formation_error_m']
        if error>self.config['outcomes']['final_formation_tolerance_m'] and self._deviation_start is None:self._deviation_start=info['elapsed_s']
        elif error<=self.config['outcomes']['final_formation_tolerance_m'] and self._deviation_start is not None:
            self._recovery_durations.append(info['elapsed_s']-self._deviation_start);self._deviation_start=None
        ped=self._stage2_episode.pedestrian_pose_at(self._elapsed)
        info['evaluation']['reward_version']='stage2_tracking_clearance_v1'
        info.update(pedestrian_position=None if ped is None else [ped.x,ped.y,0.],pedestrian_velocity_world=None if ped is None else [ped.vx,ped.vy],stage='stage2',reward_version='stage2_tracking_clearance_v1',pedestrian_count=int(has_ped),
            geometric_surface_clearance_m=float(clearance) if has_ped else None,
            minimum_geometric_surface_clearance_m=float(self._episode_min_clearance) if has_ped else None,
            pedestrian_intrusion=bool(self._intrusion),clearance_semantics='Swept pelvis-disc surrogate versus scripted pedestrian disc, not a MuJoCo physical contact',
            formation_recovery_durations_s=list(self._recovery_durations),pending_formation_recovery_s=0. if self._deviation_start is None else info['elapsed_s']-self._deviation_start)
        info['evaluation']['reward_components']=components
        return obs,float(sum(components.values())),term,trunc,info
