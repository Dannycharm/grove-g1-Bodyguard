"""Stage 1 adapter: existing analytic generator drives scalar distance on a line/arc.

No ROS imports, mailbox or competing general trajectory generator. Future routes/metadata
are evaluator APIs. client_pose_at() returns only the state at the requested current time.
"""
from dataclasses import asdict
import copy
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np
from .contract import ROOT, PACKAGE, sha256

# Repository-native dependency, without requiring an install or modifying sys.path.
# Callers set PYTHONPATH to both research packages, as documented.
from escort_scenarios.generator import generate, load_config as load_yaml, ActorState

DEFAULT_CONFIG=PACKAGE/'configs/stage1.yaml'
FAMILIES=('straight','gentle_turn','stop_start')
VERSION='stage1_progress_adapter_v1'


def load_config(path=DEFAULT_CONFIG):
    return validate_config(load_yaml(path))


def validate_config(c):
    required={
        'scenario':{'speed_m_s','route_length_m','start_xy_m','heading_rad','curvature_abs_m_inv','pause_s','stop_fraction',
            'max_acceleration_m_s2','max_turn_rate_rad_s','bounds_xy_m','geometry_margin_m','formation_error_xy_m',
            'heading_error_rad','sample_dt_s','max_sample_attempts'},
        'environment':{'dt_s','warmup_s','time_limit_s','command_caps','desired_offset_m','stable_min_height_m','stable_min_upright_cos'},
        'observation':{'position_scale_m','relative_velocity_scale_m_s','offset_scale_m','robot_velocity_scales','torso_angular_scale_rad_s'},
        'reward':{'formation_weight','formation_scale_m','progress_weight','accompaniment_error_limit_m','ahead_tolerance_m',
            'command_change_weight','torso_stability_weight','success_reward','failure_penalty'},
        'outcomes':{'client_distance_limit_m','client_loss_grace_s','final_formation_tolerance_m','recovery_window_s','minimum_robot_progress_fraction'}}
    if not isinstance(c,dict) or set(c)!={'schema_version','scene','families',*required}:
        raise ValueError('Missing or unknown Stage 1 configuration sections')
    for section,names in required.items():
        if not isinstance(c[section],dict) or set(c[section])!=names:
            raise ValueError(f'Missing or unknown keys in {section}')
        for name,value in c[section].items():
            a=np.asarray(value,dtype=float)
            if not np.isfinite(a).all():raise ValueError(f'Nonfinite configuration value: {section}.{name}')
    if c.get('schema_version')!=1 or c.get('scene')!='flat':
        raise ValueError('Stage 1 requires schema_version 1 and the verified obstacle-free flat scene')
    if set(c.get('families',{}))!=set(FAMILIES):raise ValueError('Exactly three Stage 1 families required')
    probabilities=np.array(list(c['families'].values()),float)
    if not np.isfinite(probabilities).all() or (probabilities<0).any() or not np.isclose(probabilities.sum(),1,rtol=0,atol=1e-10):
        raise ValueError('Family probabilities must be nonnegative and sum to one')
    s=c['scenario'];e=c['environment'];o=c['outcomes']
    for key in ['speed_m_s','route_length_m','start_xy_m','heading_rad','curvature_abs_m_inv','pause_s','stop_fraction']:
        a=np.asarray(s[key],float)
        if a.shape!=(2,) or not np.isfinite(a).all() or a[0]>a[1]:raise ValueError(f'Invalid range: {key}')
    if not 0<s['speed_m_s'][0]<=s['speed_m_s'][1]<=.25:raise ValueError('Stage 1 peak speed must be in (0,.25] m/s')
    if s['route_length_m'][0]<3:raise ValueError('Route must extend at least 3 m to prevent stationary completion')
    if not 0<s['stop_fraction'][0]<=s['stop_fraction'][1]<1:raise ValueError('Stop fraction must lie inside route')
    if s['pause_s'][0]<1 or s['pause_s'][1]>3:raise ValueError('Pause must lie in [1,3] s')
    if s['curvature_abs_m_inv'][0]<=0 or s['curvature_abs_m_inv'][1]*s['route_length_m'][1]>=math.pi/2:
        raise ValueError('Gentle arc must have positive curvature magnitude and total turn < pi/2')
    for key in ['max_acceleration_m_s2','max_turn_rate_rad_s','geometry_margin_m','sample_dt_s']:
        if not np.isfinite(s[key]) or s[key]<=0:raise ValueError(f'Invalid positive scenario value: {key}')
    b=np.array(s['bounds_xy_m'],float)
    if b.shape!=(2,2) or not np.isfinite(b).all() or np.any(b[:,0]>=b[:,1]):raise ValueError('Invalid scene bounds')
    caps=np.array(e['command_caps'],float)
    if caps.shape!=(3,) or not np.isfinite(caps).all() or (caps<=0).any() or np.any(caps>[.6,.6,.9]):
        raise ValueError('Command caps must not exceed frozen AGILE limits')
    if e['dt_s']!=.2 or e['warmup_s']<1:raise ValueError('Require 5 Hz actions and >=1s startup stabilization')
    if o['client_loss_grace_s']<=0 or o['recovery_window_s']<=0:raise ValueError('Positive loss/recovery windows required')
    if not 0<o['minimum_robot_progress_fraction']<=1:raise ValueError('Invalid completion progress fraction')
    for section,names in {
        'environment':['time_limit_s','stable_min_height_m','stable_min_upright_cos'],
        'observation':['position_scale_m','relative_velocity_scale_m_s','offset_scale_m','torso_angular_scale_rad_s'],
        'outcomes':['client_distance_limit_m','final_formation_tolerance_m'],
        'reward':['formation_scale_m','accompaniment_error_limit_m','failure_penalty','success_reward']}.items():
        if any(c[section][name]<=0 for name in names):raise ValueError(f'Expected positive scale/limit in {section}')
    if any(c['reward'][name]<0 for name in c['reward'] if name!='ahead_tolerance_m'):
        raise ValueError('Reward weights must be nonnegative')
    if c['reward']['ahead_tolerance_m']>0:raise ValueError('Stage 1 does not reward progress ahead of the client')
    if np.shape(c['observation']['robot_velocity_scales'])!=(3,) or np.any(np.asarray(c['observation']['robot_velocity_scales'])<=0):
        raise ValueError('Three positive robot velocity scales required')
    if np.shape(e['desired_offset_m'])!=(2,) or e['desired_offset_m'][0]>=0:
        raise ValueError('A behind-client planar formation offset is required')
    if np.shape(s['formation_error_xy_m'])!=(2,) or np.any(np.asarray(s['formation_error_xy_m'])<0) or s['heading_error_rad']<0:
        raise ValueError('Invalid nonnegative spawn error range')
    if type(s['max_sample_attempts']) is not int or s['max_sample_attempts']<1:raise ValueError('Positive integer max_sample_attempts required')
    if abs(round(e['warmup_s']/.002)*.002-e['warmup_s'])>1e-10:raise ValueError('Warmup must be an integer physics interval')
    return copy.deepcopy(c)


class Stage1Episode:
    def __init__(self,scenario,metadata,start,heading,curvature,robot_pose):
        self._progress=scenario;self._metadata=metadata
        self.start=np.array(start,float);self.heading=float(heading);self.curvature=float(curvature)
        self.robot_pose=np.asarray(robot_pose,float)
        self.length=metadata['route_length_m'];self.duration_s=scenario.duration_s
        self.reset_episode()

    def reset_episode(self):self._last_time=0.

    def point_at_progress(self,s):
        k=self.curvature
        local=np.array([s,0.]) if abs(k)<1e-12 else np.array([math.sin(k*s)/k,(1-math.cos(k*s))/k])
        h=self.heading;R=np.array([[math.cos(h),-math.sin(h)],[math.sin(h),math.cos(h)]])
        return self.start+R@local

    def client_pose_at(self,elapsed_time):
        state=self._progress.pose_at('client',elapsed_time)
        distance=state.x;yaw=self.heading+self.curvature*distance
        tangent=np.array([math.cos(yaw),math.sin(yaw)]);normal=np.array([-tangent[1],tangent[0]])
        xy=self.point_at_progress(distance);v=state.vx*tangent
        a=state.ax*tangent+self.curvature*state.vx**2*normal
        # Tangent parameterization retains endpoint/stop heading even at zero speed.
        return ActorState('client','client','mujoco_world',state.time_s,state.timestamp_s,
                          float(xy[0]),float(xy[1]),0.,yaw,float(v[0]),float(v[1]),0.,
                          float(a[0]),float(a[1]),self.curvature*state.vx,
                          provenance='headless_simulation_pose_proxy')

    def client_progress(self,t):return self._progress.pose_at('client',t).x

    def project_robot(self,xy):
        h=self.heading;R=np.array([[math.cos(h),-math.sin(h)],[math.sin(h),math.cos(h)]])
        x,y=R.T@(np.asarray(xy)-self.start);k=self.curvature
        return float(x if abs(k)<1e-12 else math.atan2(k*x,1-k*y)/k)

    def scenario_metadata(self):return copy.deepcopy(self._metadata)

    def validate(self,config):
        s=config['scenario'];bounds=np.array(s['bounds_xy_m']);margin=s['geometry_margin_m']
        # Analytic arc extrema plus endpoints establish continuous route bounds, not just samples.
        phases=[self.heading,self.heading+self.curvature*self.length]
        progress=[0.,self.length]
        if abs(self.curvature)>1e-12:
            for n in range(-8,9):
                for phase in (n*math.pi,n*math.pi+math.pi/2):
                    arc_s=(phase-self.heading)/self.curvature
                    if 0<arc_s<self.length:progress.append(arc_s)
        points=[self.point_at_progress(x) for x in progress]+[self.robot_pose[:2]]
        offset=np.asarray(config['environment']['desired_offset_m'],float)
        desired_progress=[0.,self.length]
        derivative_angle=math.atan2(self.curvature*offset[0],1-self.curvature*offset[1])
        if abs(self.curvature)>1e-12:
            for n in range(-8,9):
                for phase in (n*math.pi,n*math.pi+math.pi/2):
                    arc_s=(phase-self.heading-derivative_angle)/self.curvature
                    if 0<arc_s<self.length:desired_progress.append(arc_s)
        for arc_s in desired_progress:
            heading=self.heading+self.curvature*arc_s
            R=np.array([[math.cos(heading),-math.sin(heading)],[math.sin(heading),math.cos(heading)]])
            points.append(self.point_at_progress(arc_s)+R@offset)
        points=np.asarray(points)
        if np.any(points<bounds[:,0]+margin) or np.any(points>bounds[:,1]-margin):
            raise ValueError('Route or robot spawn violates continuous flat-scene free-space bounds/margin')
        max_speed=self._metadata['peak_speed_limit_m_s']
        if abs(self.curvature)*max_speed>s['max_turn_rate_rad_s']+1e-10:raise ValueError('Turn-rate bound exceeded')
        # Generator tangential acceleration budget and centripetal bound combine orthogonally.
        tangential=self._metadata['tangential_acceleration_limit_m_s2']
        if math.hypot(tangential,abs(self.curvature)*max_speed**2)>s['max_acceleration_m_s2']+1e-10:
            raise ValueError('Combined acceleration bound exceeded')
        return dict(valid=True,continuous_bounds=True,geometry='original flat plane; no nonrobot obstacles',
                    speed_limit_m_s=max_speed,acceleration_limit_m_s2=s['max_acceleration_m_s2'],
                    turn_rate_limit_rad_s=s['max_turn_rate_rad_s'])


def sample_episode(seed,family=None,config=None):
    c=load_config() if config is None else validate_config(config)
    rng=np.random.default_rng(int(seed));names=list(FAMILIES)
    family=str(rng.choice(names,p=[c['families'][n] for n in names])) if family is None else family
    if family not in FAMILIES:raise ValueError(f'Unsupported Stage 1 family: {family}')
    s=c['scenario'];last_error=None
    for attempt in range(s['max_sample_attempts']):
        length=float(rng.uniform(*s['route_length_m']));speed=float(rng.uniform(*s['speed_m_s']))
        start=rng.uniform(*s['start_xy_m'],size=2);heading=float(rng.uniform(*s['heading_rad']))
        k=float(rng.choice([-1,1])*rng.uniform(*s['curvature_abs_m_inv'])) if family=='gentle_turn' else 0.
        centripetal=abs(k)*speed**2
        tangential=math.sqrt(max(0,s['max_acceleration_m_s2']**2-centripetal**2))
        pause=float(rng.uniform(*s['pause_s'])) if family=='stop_start' else 0.
        stop_fraction=float(rng.uniform(*s['stop_fraction'])) if family=='stop_start' else None
        points=[[0,0],[length,0]] if stop_fraction is None else [[0,0],{'xy':[length*stop_fraction,0],'dwell_s':pause},[length,0]]
        generator_config=dict(schema_version=1,family='straight',seed=int(seed),frame_id='stage1_route_distance',
            time_origin_s=0.,sample_dt_s=s['sample_dt_s'],pedestrian_count=0,
            bounds={'x':[-1,length+1],'y':[-1,1]},
            limits={'max_speed_mps':speed,'max_acceleration_mps2':tangential,'max_turn_rate_rps':s['max_turn_rate_rad_s']},
            randomization={'position_jitter_m':0,'speed_scale':[1,1]},
            actors=[{'id':'client','role':'client','initial_yaw':0,'waypoints':points}])
        scenario=generate(generator_config)
        h=heading;R=np.array([[math.cos(h),-math.sin(h)],[math.sin(h),math.cos(h)]])
        error=rng.uniform(-np.array(s['formation_error_xy_m']),np.array(s['formation_error_xy_m']))
        robot_xy=start+R@(np.array(c['environment']['desired_offset_m'])+error)
        robot_pose=[*robot_xy,heading+float(rng.uniform(-s['heading_error_rad'],s['heading_error_rad']))]
        metadata=dict(schema_version=1,adapter_version=VERSION,generator_version='0.1.0',
            python_version=sys.version.split()[0],numpy_version=np.__version__,family=family,seed=int(seed),
            route_length_m=length,peak_speed_limit_m_s=speed,tangential_acceleration_limit_m_s2=tangential,
            speed_semantics='sampled peak bound; quintic starts/ends at rest, not a constant cruise speed',
            start_xy_m=start.tolist(),heading_rad=heading,curvature_m_inv=k,pause_s=pause,stop_fraction=stop_fraction,
            robot_spawn_xy_yaw=robot_pose,pedestrian_count=0,client_id='client',frame='mujoco_world',
            generator_resolved_config=scenario.resolved_config,generator_source_sha256=sha256(ROOT/'research/escort_scenarios/escort_scenarios/generator.py'),
            config_sha256=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest(),
            adapter_source_sha256=sha256(__file__),evaluator_only=True,sample_attempt=attempt)
        episode=Stage1Episode(scenario,metadata,start,heading,k,robot_pose)
        try:episode.validate(c)
        except ValueError as exc:last_error=str(exc);continue
        episode._metadata['duration_s']=episode.duration_s
        episode._metadata['mean_route_speed_m_s']=length/episode.duration_s
        episode._metadata['scenario_id']=hashlib.sha256(json.dumps(episode._metadata,sort_keys=True).encode()).hexdigest()
        return episode
    raise ValueError(f'No valid Stage 1 geometry after {s["max_sample_attempts"]} attempts: {last_error}')
