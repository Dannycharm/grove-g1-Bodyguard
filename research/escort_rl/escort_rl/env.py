"""Isolated full-G1 Stage 1 Gymnasium environment. No ROS/DDS or training process."""
import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .mujoco_runner import MujocoRunner
from .stage1_scenarios import load_config,validate_config,sample_episode,DEFAULT_CONFIG
from .observations import SIZE,build_observation
from .rewards import EpisodeEvaluator


class G1EscortEnv(gym.Env):
    metadata={'render_modes':[],'render_fps':5}

    def __init__(self,config_path=DEFAULT_CONFIG,*,config=None,render_mode=None):
        super().__init__()
        if render_mode is not None:raise ValueError('Headless only; render_mode must be None')
        self.config=load_config(config_path) if config is None else validate_config(config)
        self.render_mode=None
        self.action_space=spaces.Box(-1.,1.,(3,),np.float32)
        self.observation_space=spaces.Box(-1.,1.,(SIZE,),np.float32)
        self._caps=np.asarray(self.config['environment']['command_caps'],float)
        self._runner=MujocoRunner(scene='flat',actors=1,actor_names=['escort_client'])
        self._runner.actor_driver=self._drive_client
        self._client_slot=int(self._runner.model.body_mocapid[self._runner._id(self._runner.mj.mjtObj.mjOBJ_BODY,'escort_client')])
        self._torso=self._runner._id(self._runner.mj.mjtObj.mjOBJ_BODY,'torso_link')
        self._scenario=None;self._origin=None;self._done=True;self._last_observation=None
        self._elapsed=0.;self._previous_applied=np.zeros(3);self._evaluator=None

    def _drive_client(self,simulation_time,data):
        if self._scenario is None:return
        t=0. if self._origin is None else max(0.,simulation_time-self._origin)
        state=self._scenario.client_pose_at(t)
        data.mocap_pos[self._client_slot]=[state.x,state.y,0.]
        data.mocap_quat[self._client_slot]=[np.cos(state.yaw/2),0.,0.,np.sin(state.yaw/2)]

    def _observe(self):
        row=self._runner.measure();R=self._runner.data.xmat[self._torso].reshape(3,3)
        omega=np.zeros(6)
        self._runner.mj.mj_objectVelocity(self._runner.model,self._runner.data,
            self._runner.mj.mjtObj.mjOBJ_BODY,self._torso,omega,0)
        gravity=R.T@np.array([0.,0.,-1.]);angular=R.T@omega[:3]
        client=self._scenario.client_pose_at(self._elapsed)
        observation,saturated=build_observation(client,row,gravity,angular,self._previous_applied,
            self.config['environment']['desired_offset_m'],self.config)
        return observation,row,client,saturated,max(0.,1+gravity[2])

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        options={} if options is None else dict(options)
        if set(options)-{'family','episode_seed'}:raise ValueError('Supported reset options: family, episode_seed')
        episode_seed=int(options['episode_seed']) if 'episode_seed' in options else int(self.np_random.integers(0,2**63-1))
        self._scenario=sample_episode(episode_seed,options.get('family'),self.config)
        self._scenario.reset_episode();self._origin=None;self._elapsed=0.;self._done=True
        self._previous_applied=np.zeros(3);self._last_observation=None
        spawn=self._scenario.robot_pose
        self._runner.reset(base_xy=spawn[:2],base_yaw=spawn[2])
        self._drive_client(self._runner.data.time,self._runner.data)
        warm_steps=round(self.config['environment']['warmup_s']/self._runner.model.opt.timestep)
        for i in range(warm_steps):
            if i%100==0:self._runner.command([0.,0.,0.])
            self._runner.step_physics()
        row=self._runner.measure();e=self.config['environment']
        if (self._runner.data.eq_active[self._runner.weld] or row['emergency'] or row['fallen'] or
                row['height']<e['stable_min_height_m'] or row['upright']<e['stable_min_upright_cos']):
            raise RuntimeError('Reset stabilization failed; episode was not started')
        self._origin=float(self._runner.data.time);self._active_steps=0
        self._drive_client(self._origin,self._runner.data)
        self._runner.mj.mj_forward(self._runner.model,self._runner.data)
        self._evaluator=EpisodeEvaluator(self.config,self._scenario.project_robot(row['position'][:2]),self._scenario.length)
        self._done=False
        observation,row,client,saturated,_=self._observe();self._last_observation=observation.copy()
        info=dict(scenario_id=self._scenario.scenario_metadata()['scenario_id'],episode_seed=episode_seed,
            elapsed_s=0.,warmup_simulation_s=float(self._runner.data.time),warmup_physics_steps=warm_steps,
            warmup_gait_calls=self._runner.gait.calls,startup_weld_released=True,observation_saturated_features=saturated,
            data_provenance='isolated_headless_mujoco_and_scripted_client_pose_proxy')
        return observation,info

    def step(self,action):
        if self._done:raise RuntimeError('Call reset() before step() and after episode completion')
        raw=np.asarray(action,float)
        if raw.shape!=(3,) or not np.isfinite(raw).all():raise ValueError('Expected three finite normalized actions')
        clipped=np.clip(raw,-1,1);command=clipped*self._caps
        self._runner.command(command)
        before_steps=self._runner.physics_steps;before_gait=self._runner.gait.calls
        steps=round(self.config['environment']['dt_s']/self._runner.model.opt.timestep)
        failure=False;fall=False
        try:
            for _ in range(steps):
                previous_time=self._runner.data.time;self._runner.step_physics()
                if abs(self._runner.data.time-previous_time-self._runner.model.opt.timestep)>1e-8:
                    raise RuntimeError('Physics clock rollback/automatic reset detected')
                d=self._runner.data
                fall |= bool(d.xpos[self._runner.pelvis,2]<.45 or d.xmat[self._runner.pelvis,8]<.5)
        except (RuntimeError,FloatingPointError,self._runner.mj.FatalError) as exc:
            failure=True;failure_detail=str(exc)
        executed=self._runner.physics_steps-before_steps;self._active_steps+=executed
        self._elapsed=self._active_steps*self._runner.model.opt.timestep
        self._previous_applied=np.clip(self._runner.controller.applied.astype(float)/self._caps,-1,1)
        if failure:
            # Explicit irrecoverable failure, finite last causal observation; no fake state advance.
            self._done=True
            components={k:0. for k in ('formation','progress','command_change','torso_stability','terminal_success')}
            components['terminal_failure']=-float(self.config['reward']['failure_penalty'])
            terminal_observation=self._last_observation.copy();terminal_observation[8:10]=0.
            return terminal_observation,sum(components.values()),True,False,dict(reason='physics_failure',
                physics_failure_detail=failure_detail,
                reward_components=components,physics_steps=executed,gait_calls=self._runner.gait.calls-before_gait,
                elapsed_s=self._elapsed,observation_reused_after_physics_failure=True)
        observation,row,client,saturated,tilt=self._observe()
        bounds=np.array(self.config['scenario']['bounds_xy_m'])
        xy=np.asarray(row['position'][:2]);outside=bool(np.any(xy<bounds[:,0]) or np.any(xy>bounds[:,1]))
        reward,terminated,truncated,evaluation=self._evaluator.update(elapsed=self._elapsed,
            dt=executed*self._runner.model.opt.timestep,robot_xy=xy,client=client,
            robot_progress=self._scenario.project_robot(xy),client_progress=self._scenario.client_progress(self._elapsed),
            client_arrived=self._scenario.duration_s if self._elapsed+1e-9>=self._scenario.duration_s else None,
            action=self._previous_applied,fallen=fall or row['fallen'],emergency=row['emergency'],
            out_of_bounds=outside,torso_tilt=tilt)
        if not np.isfinite(reward):raise RuntimeError('Nonfinite reward')
        self._done=terminated or truncated;self._last_observation=observation.copy()
        info=dict(reason=evaluation['reason'],elapsed_s=self._elapsed,
            physics_steps=executed,gait_calls=self._runner.gait.calls-before_gait,
            raw_action=raw.tolist(),clipped_action=clipped.tolist(),raw_command=(raw*self._caps).tolist(),
            clipped_command=command.tolist(),filtered_command=command.tolist(),
            applied_command=row['applied'],command_filter='none; reference timeout/caps preserved',
            observation_saturated_features=saturated,evaluation=evaluation,
            reward_components=evaluation['reward_components'],robot_position=row['position'],
            client_position=[client.x,client.y,0.],body_velocity=row['body_velocity'],
            velocity_convention=row['velocity_convention'],contacts=row['contacts'],
            height=row['height'],upright=row['upright'],supervisor_failure=row['emergency'])
        return observation,float(reward),bool(terminated),bool(truncated),info

    def scenario_metadata(self):
        """Evaluator-only API. Never passed into observation construction/normalization."""
        if self._scenario is None:raise RuntimeError('No episode sampled')
        return self._scenario.scenario_metadata()

    def close(self):
        if self._runner is not None:self._runner.close();self._runner=None
