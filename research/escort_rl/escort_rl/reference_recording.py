"""Read-only bag normalization and evaluator analysis. No ROS graph or command authority.

Streams deliberately remain unaligned: wall/controller stamps, mapped relay stamps,
recorder timestamps and native simulation time are distinct clock domains.
"""
import numpy as np
from .contract import load_contract
from .measurements import rotation_wxyz

TOPICS={
    '/joint_states':'sensor_msgs/msg/JointState',
    '/cmd_vel':'geometry_msgs/msg/Twist',
    '/g1_sensor_relay/actor_ground_truth':'g1_msgs/msg/SimulatorActorSnapshot',
    '/g1_sensor_relay/base_state':'nav_msgs/msg/Odometry',
    '/agile_controller/inferring':'std_msgs/msg/Bool',
    '/locomotion_safety_controller/transition_event':'lifecycle_msgs/msg/TransitionEvent',
    '/diagnostics':'diagnostic_msgs/msg/DiagnosticArray',
}

def stamp_ns(header):
    s=header['stamp'];return int(s['sec'])*1_000_000_000+int(s['nanosec'])

def vector(v):
    result=[float(v[k]) for k in ('x','y','z')]
    if not np.isfinite(result).all():raise ValueError('Nonfinite vector')
    return result

def pose(p):
    q=p['orientation'];quat=[float(q[k]) for k in ('w','x','y','z')]
    rotation_wxyz(quat)
    return dict(position=vector(p['position']),quaternion_wxyz=quat)

def normalize(topic,message,bag_timestamp_ns):
    result=dict(topic=topic,bag_timestamp_ns=int(bag_timestamp_ns),evaluator_only=True)
    if 'header' in message:
        result.update(header_stamp_ns=stamp_ns(message['header']),frame_id=message['header']['frame_id'])
    if topic=='/joint_states':
        names=message['name'];desired=load_contract(verify=False)['observation_joints']
        if len(names)!=len(set(names)):raise ValueError('Duplicate joint names')
        slots={n:i for i,n in enumerate(names)}
        result.update(kind='joint_state',joint_order=desired)
        for source,target in [('position','joint_position'),('velocity','joint_velocity'),('effort','joint_effort')]:
            value=message[source]
            if len(value) not in (0,len(names)):raise ValueError('JointState array/name length mismatch')
            result[target]=[float(value[slots[n]]) if n in slots and len(value) and np.isfinite(value[slots[n]]) else None for n in desired]
        result['all_body_joints_valid']=all(v is not None and np.isfinite(v) for key in ('joint_position','joint_velocity') for v in result[key])
    elif topic=='/cmd_vel':
        result.update(kind='command_received',requested=[float(message['linear']['x']),float(message['linear']['y']),float(message['angular']['z'])],
                      applied_command=None,clock='recorder arrival only; Twist has no header',
                      semantics='publisher command only; not proof of timeout/clamped/executed command')
    elif topic=='/g1_sensor_relay/actor_ground_truth':
        # Drop all actor roster, trajectories, alignment transforms and nonrobot truth.
        result.update(kind='pelvis_snapshot',simulation_time_s=float(message['simulation_time_s']),
                      epoch=int(message['epoch']),valid=bool(message['pelvis_valid']))
        if result['valid']:result.update(pose(message['pelvis_pose']))
    elif topic=='/g1_sensor_relay/base_state':
        result.update(kind='base_state',child_frame_id=message['child_frame_id'],**pose(message['pose']['pose']),
                      linear_velocity_pelvis=vector(message['twist']['twist']['linear']),
                      angular_velocity_pelvis=vector(message['twist']['twist']['angular']),
                      semantics='source-verified imu site origin/body axes; confirm frame params and loaded model',
                      simulation_time_s=None,clock='relay mapped capture stamp; not exact native time')
    elif topic=='/agile_controller/inferring':
        result.update(kind='inferring',inferring=bool(message['data']),
                      semantics='latched first-inference flag, not per-tick success or safety status')
    elif topic.endswith('/transition_event'):
        result.update(kind='lifecycle_transition',transition_id=int(message['transition']['id']),
                      start_state=message['start_state'],goal_state=message['goal_state'],
                      semantics='lifecycle state transition, not the safety emergency latch')
    elif topic=='/diagnostics':
        result.update(kind='diagnostics',status=message['status'],
                      semantics='source hardware diagnostics report temperatures, not safety latch')
    else:raise ValueError('Unsupported topic')
    return result


def summarize(events,inventory):
    snapshots=[e for e in events if e['kind']=='pelvis_snapshot' and e['valid']]
    joints=[e for e in events if e['kind']=='joint_state']
    commands=[e for e in events if e['kind']=='command_received']
    estimates=[]
    for before,after in zip(snapshots,snapshots[1:]):
        dt=after['simulation_time_s']-before['simulation_time_s']
        if before['epoch']!=after['epoch'] or not 0<dt<=.5:continue
        velocity=(np.array(after['position'])-before['position'])/dt
        R=rotation_wxyz(after['quaternion_wxyz'])
        R0=rotation_wxyz(before['quaternion_wxyz'])
        delta=np.arctan2(R[1,0],R[0,0])-np.arctan2(R0[1,0],R0[0,0])
        estimates.append(dict(simulation_time_s=after['simulation_time_s'],dt_s=dt,
            linear_velocity_world_estimate=velocity.tolist(),linear_velocity_pelvis_estimate=(R.T@velocity).tolist(),
            euler_yaw_rate_estimate=float(np.arctan2(np.sin(delta),np.cos(delta))/dt)))
    result=dict(topic_inventory=inventory,counts={t:sum(e['topic']==t for e in events) for t in TOPICS},
        nonzero_command_messages=sum(any(abs(v)>1e-8 for v in e['requested']) for e in commands),
        command_stream_observed=bool(commands),missing_commands_are_zero=False,
        full_dynamics_parity_ready=False,
        unavailable_without_additional_evidence=['loaded checkpoint identity','loaded scene/model and compiled engine settings',
          'exact executed command and receipt/timeout/inference times','joint targets/gains','safety latch and startup-weld state',
          'matched initial state and repeated matching trials','exact native time for joint/controller/base_state samples'],
        snapshot_derivative_semantics='backward difference at native sim time; evaluator-only estimate, not native twist',
        snapshot_count=len(snapshots),joint_samples_with_complete_body_state=sum(e['all_body_joints_valid'] for e in joints))
    if estimates:
        speeds=np.linalg.norm(np.array([e['linear_velocity_world_estimate'][:2] for e in estimates]),axis=1)
        result['snapshot_planar_speed_estimate_mean_max_m_s']=[float(speeds.mean()),float(speeds.max())]
        result['snapshot_native_duration_s']=snapshots[-1]['simulation_time_s']-snapshots[0]['simulation_time_s']
    if snapshots:
        offsets=np.array([e['header_stamp_ns']/1e9-e['simulation_time_s'] for e in snapshots])
        result['relay_stamp_offset_range_s']=float(np.ptp(offsets))
        result['snapshot_epoch_count']=len({e['epoch'] for e in snapshots})
    if joints:
        values=np.array([e['joint_position'] for e in joints if e['all_body_joints_valid']])
        if values.size:result['max_joint_position_range_rad']=float(np.ptp(values,axis=0).max())
        velocities=np.array([e['joint_velocity'] for e in joints if e['all_body_joints_valid']])
        if velocities.size:result['max_abs_joint_velocity_rad_s']=float(np.max(np.abs(velocities)))
    return result,estimates
