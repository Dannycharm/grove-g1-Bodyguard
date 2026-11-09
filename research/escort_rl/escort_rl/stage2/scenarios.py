"""Client replay plus one analytic pedestrian using the existing quintic generator."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from escort_scenarios.generator import generate,load_config as read_yaml
from ..contract import PACKAGE
from ..stage1_scenarios import load_config as load_stage1,sample_episode as stage1_sample,FAMILIES as STAGE1

DEFAULT=PACKAGE/'stage2_config/stage2.yaml'
ENCOUNTERS=('crossing','head_on','diagonal')
FAMILIES=(*STAGE1,*ENCOUNTERS)


def load_config(path=DEFAULT):
    path=Path(path).resolve();c=read_yaml(path)
    if set(c)!= {'schema_version','version','stage1_config','stage1_ppo_config','families','pedestrian','clearance','validation'} or c['schema_version']!=1 or c['version']!='stage2_single_pedestrian_v1':raise ValueError('Invalid Stage 2 schema/version')
    if set(c['families'])!=set(FAMILIES) or any(v<0 or not np.isfinite(v) for v in c['families'].values()) or not np.isclose(sum(c['families'].values()),1):raise ValueError('Invalid Stage 2 family probabilities')
    for key in ('peak_speed_m_s','encounter_duration_fraction','timing_jitter_s','half_path_length_m','diagonal_angle_degrees','head_on_lane_offset_m'):
        a=np.asarray(c['pedestrian'][key]);
        if a.shape!=(2,) or not np.isfinite(a).all() or a[0]>a[1]:raise ValueError('Invalid pedestrian range: '+key)
    p=c['pedestrian'];g=c['clearance'];v=c['validation']
    if not 0<p['encounter_duration_fraction'][0]<=p['encounter_duration_fraction'][1]<1:raise ValueError('Encounter must occur during client route')
    if not 0<p['peak_speed_m_s'][0]<=p['peak_speed_m_s'][1] or not 90<p['diagonal_angle_degrees'][0]<=p['diagonal_angle_degrees'][1]<180:raise ValueError('Invalid approach speed/angle')
    if any(not np.isfinite(x) or x<=0 for x in (p['max_acceleration_m_s2'],p['radius_m'],p['initial_robot_distance_m'],g['robot_pelvis_footprint_radius_m'],g['preferred_surface_margin_m'])) or g['shaping_weight']<0 or g['intrusion_threshold_m']!=0:raise ValueError('Invalid clearance geometry/settings')
    if type(p['max_attempts']) is not int or p['max_attempts']<1 or v['seed_modulus']!=4 or v['train_residue']!=0 or v['validation_residue']!=1 or v['final_test_residue']!=2:raise ValueError('Invalid attempts or seed partition')
    for k in ('pedestrian_success','each_encounter_success','stage1_success','stage1_each_family_success'):
        if not 0<v[k]<=1:raise ValueError('Invalid promotion probability')
    c['stage1_config']=str((path.parent/c['stage1_config']).resolve());c['stage1_ppo_config']=str((path.parent/c['stage1_ppo_config']).resolve())
    load_stage1(c['stage1_config']);return c


class Episode:
    def __init__(self,base,family,pedestrian=None,details=None):
        self.base=base;self.family=family;self.pedestrian=pedestrian;self.details=details or {}
    def __getattr__(self,key):return getattr(self.base,key)
    def pedestrian_pose_at(self,t):return None if self.pedestrian is None else self.pedestrian.at(t)
    def scenario_metadata(self):
        m=self.base.scenario_metadata();m.update(stage='stage2',family=self.family,client_family=m['family'],pedestrian_count=int(self.pedestrian is not None),pedestrian_id='pedestrian_1' if self.pedestrian else None,encounter=copy.deepcopy(self.details))
        m['scenario_id']=hashlib.sha256(json.dumps(m,sort_keys=True).encode()).hexdigest();return m


def sample_episode(seed,family=None,config=None):
    c=load_config() if config is None else copy.deepcopy(config);rng=np.random.default_rng(np.random.SeedSequence([int(seed),2]))
    if family is None:family=str(rng.choice(list(c['families']),p=list(c['families'].values())))
    if family not in FAMILIES:raise ValueError('Unknown Stage 2 family')
    base_config=load_stage1(c['stage1_config']);base=stage1_sample(seed,family if family in STAGE1 else 'straight',base_config)
    if family in STAGE1:return Episode(base,family)
    p=c['pedestrian'];bounds=np.asarray(base_config['scenario']['bounds_xy_m']);margin=base_config['scenario']['geometry_margin_m'];last=None
    for attempt in range(p['max_attempts']):
        fraction=float(rng.uniform(*p['encounter_duration_fraction']));time=base.duration_s*fraction
        client=base.client_pose_at(time);heading=client.yaw;u=np.array([np.cos(heading),np.sin(heading)]);n=np.array([-u[1],u[0]])
        anchor=np.array([client.x,client.y])+u*base_config['environment']['desired_offset_m'][0]
        side=int(rng.choice([-1,1]));offset=side*float(rng.uniform(*p['head_on_lane_offset_m'])) if family=='head_on' else 0.
        anchor+=n*offset
        angle=side*np.pi/2 if family=='crossing' else np.pi if family=='head_on' else side*np.deg2rad(rng.uniform(*p['diagonal_angle_degrees']))
        direction=np.array([np.cos(heading+angle),np.sin(heading+angle)])
        length=float(rng.uniform(*p['half_path_length_m']));start=anchor-direction*length;end=anchor+direction*length
        if np.any(np.vstack([start,end])<bounds[:,0]+margin) or np.any(np.vstack([start,end])>bounds[:,1]-margin):last='pedestrian route outside bounds';continue
        speed=float(rng.uniform(*p['peak_speed_m_s']));jitter=float(rng.uniform(*p['timing_jitter_s']))
        cfg=dict(schema_version=1,family='straight',seed=int(seed),frame_id='mujoco_world',time_origin_s=0.,sample_dt_s=.02,pedestrian_count=1,
            bounds=dict(x=bounds[0].tolist(),y=bounds[1].tolist()),limits=dict(max_speed_mps=speed,max_acceleration_mps2=p['max_acceleration_m_s2'],max_turn_rate_rps=1.),randomization=dict(position_jitter_m=0.,speed_scale=[1.,1.]),
            actors=[dict(id='unused_generator_carrier_client',role='client',initial_yaw=heading,waypoints=[base.start.tolist(),base.start.tolist()]),
                    dict(id='pedestrian_1',role='pedestrian',initial_yaw=heading+angle,waypoints=[dict(xy=start.tolist(),dwell_s=0.),end.tolist()])])
        trajectory=generate(cfg).actors[1];duration=trajectory.duration_s;delay=time+jitter-duration/2
        if delay<0:last='encounter too early for bounded pedestrian acceleration';continue
        cfg['actors'][1]['waypoints'][0]['dwell_s']=delay;trajectory=generate(cfg).actors[1]
        if np.linalg.norm(start-base.robot_pose[:2])<p['initial_robot_distance_m']:last='pedestrian spawn too close to G1';continue
        times=np.arange(0,max(base.duration_s,trajectory.duration_s)+.05,.05)
        clearance=min(np.hypot(base.client_pose_at(float(t)).x-trajectory.at(float(t)).x,base.client_pose_at(float(t)).y-trajectory.at(float(t)).y)-2*p['radius_m'] for t in times)
        clearance_lower_bound=clearance-.025*(speed+base.scenario_metadata()['peak_speed_limit_m_s'])
        if clearance_lower_bound<p['minimum_client_surface_clearance_m']:last='scripted client/pedestrian paths overlap or lack clearance';continue
        details=dict(adapter_version='stage2_encounters_v1',peak_speed_m_s=speed,encounter_fraction=fraction,target_time_s=time,actual_midpoint_time_s=delay+duration/2,timing_jitter_s=jitter,approach_angle_rad=float(angle),head_on_lane_offset_m=offset,pedestrian_delay_s=delay,pedestrian_duration_s=duration,start_xy=start.tolist(),end_xy=end.tolist(),side=side,sample_attempt=attempt,client_pedestrian_min_surface_clearance_m=float(clearance),client_pedestrian_continuous_clearance_lower_bound_m=float(clearance_lower_bound),client_pedestrian_validation_dt_s=.05,generator_config=cfg,carrier_client='Required only by generator schema; unused. Actual client is unchanged Stage1Episode.',evaluator_only=True)
        return Episode(base,family,trajectory,details)
    raise ValueError(f'No valid {family} encounter in {p["max_attempts"]} attempts: {last}')
