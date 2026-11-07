import numpy as np

def response_metrics(rows, command, hold_start, stop_start, config, velocity_key="body_velocity", stop_command_time=None):
    conventions={r.get("velocity_convention", "unlabeled_legacy") for r in rows}
    if len(conventions)!=1:
        raise ValueError("Mixed velocity conventions require explicit conversion")
    t=np.array([r['time'] for r in rows]); v=np.array([r[velocity_key] for r in rows])
    cmd=np.clip(command,[-.6,-.6,-.9],[.6,.6,.9])
    steady=(t>=stop_start-2)&(t<stop_start)
    achieved=v[steady].mean(axis=0)
    active=np.abs(cmd)>1e-8
    crossed=np.all(np.where(active, v*np.sign(cmd)>=np.abs(cmd)*config['latency_fraction'],True),axis=1)
    indices=np.flatnonzero((t>=hold_start)&(t<stop_start)&crossed)
    latency=float(t[indices[0]]-hold_start) if active.any() and len(indices) else None
    hold_end=stop_start
    if stop_command_time is not None:stop_start=float(stop_command_time)
    stopped=(np.linalg.norm(v[:,:2],axis=1)<config['stop_threshold_m_s'])&(np.abs(v[:,2])<config['stop_threshold_rad_s'])
    stop_index=None
    dwell=round(config['stop_dwell_s']/config['sample_period_s'])
    for i in np.flatnonzero((t>=stop_start)&(t<stop_start+config['stop_s'])):
        if i+dwell<=len(rows) and t[i+dwell-1]<stop_start+config['stop_s'] and stopped[i:i+dwell].all():
            stop_index=i; break
    start=np.searchsorted(t,stop_start)
    stop_distance=None
    if stop_index is not None:
        xy=np.array([r['position'][:2] for r in rows])
        stop_distance=float(np.linalg.norm(np.diff(xy[start:stop_index+1],axis=0),axis=1).sum())
    return dict(velocity_convention=(next(iter(conventions)) if velocity_key=="body_velocity" else LEGACY_VELOCITY_CONVENTION),
        hold_end_time=float(hold_end),stop_command_time=float(stop_start),
        requested=list(command),applied=cmd.tolist(),achieved=achieved.tolist(),
        error=(achieved-cmd).tolist(),latency_s=latency,
        stopping_time_s=None if stop_index is None else float(t[stop_index]-stop_start),
        stopping_distance_m=stop_distance,fall=any(r['fallen'] for r in rows),
        emergency=any(r['emergency'] for r in rows),
        qualification='body angular z is yaw-rate approximation when tilted; latency is first 50% response; null means no crossing/no settled stop within window')


VELOCITY_CONVENTION = 'pelvis_origin_in_pelvis_axes_v2'
LEGACY_VELOCITY_CONVENTION = 'pelvis_inertial_com_in_inertial_axes_v1'


def euler_yaw_rate(rotation_body_to_world, angular_velocity_world):
    """Derivative of world ZYX yaw; undefined when the projected body x-axis vanishes."""
    R = np.asarray(rotation_body_to_world, dtype=float)
    omega = np.asarray(angular_velocity_world, dtype=float)
    denominator = R[0, 0]**2 + R[1, 0]**2
    if denominator < 1e-12:
        return None
    return float(omega[2] - R[2, 0] *
                 (R[0, 0]*omega[0] + R[1, 0]*omega[1]) / denominator)


def robot_velocities(mj, model, data, pelvis, imu_site):
    """Read-only velocities. World is native MuJoCo; pelvis axes are xmat, not ximat.

    mjOBJ_BODY velocity is centered at the body's inertial COM. The existing imu site
    is at the pelvis origin and aligned to pelvis axes, matching Grove baseStateLoop.
    No state, control, policy input or integration history is written here.
    """
    R = data.xmat[pelvis].reshape(3, 3)
    I = data.ximat[pelvis].reshape(3, 3)
    com_world = np.zeros(6)
    origin_world = np.zeros(6)
    mj.mj_objectVelocity(model, data, mj.mjtObj.mjOBJ_BODY, pelvis, com_world, 0)
    mj.mj_objectVelocity(model, data, mj.mjtObj.mjOBJ_SITE, imu_site, origin_world, 0)
    linear_body = R.T @ origin_world[3:]
    angular_body = R.T @ origin_world[:3]
    legacy_linear = I.T @ com_world[3:]
    legacy_angular = I.T @ com_world[:3]
    yaw_rate = euler_yaw_rate(R, origin_world[:3])
    return dict(
        measurement_schema_version=2, velocity_convention=VELOCITY_CONVENTION,
        velocity_point='pelvis_origin',velocity_axes='pelvis',world_frame='mujoco_world',
        data_provenance='isolated_headless_mujoco_state',
        com_position_world=data.xipos[pelvis].tolist(),
        body_velocity=[float(linear_body[0]), float(linear_body[1]), float(angular_body[2])],
        linear_velocity_world=origin_world[3:].tolist(),
        linear_velocity_pelvis=linear_body.tolist(),
        angular_velocity_world=origin_world[:3].tolist(),
        angular_velocity_pelvis=angular_body.tolist(),
        euler_yaw_rate_world=yaw_rate, euler_yaw_rate_valid=yaw_rate is not None,
        com_linear_velocity_world=com_world[3:].tolist(),
        com_linear_velocity_pelvis=(R.T @ com_world[3:]).tolist(),
        com_linear_velocity_inertial=legacy_linear.tolist(),
        legacy_com_velocity=[float(legacy_linear[0]), float(legacy_linear[1]), float(legacy_angular[2])],
        legacy_com_velocity_convention=LEGACY_VELOCITY_CONVENTION)


def rotation_wxyz(quaternion):
    """Quaternion body-to-world rotation, without silently normalizing invalid samples."""
    q=np.asarray(quaternion,dtype=float)
    if q.shape!=(4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>1e-5:
        raise ValueError('Expected finite unit wxyz quaternion')
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
