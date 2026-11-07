"""Causal numerical actor contract; no scenario, route, family or evaluator object accepted."""
import numpy as np

SIZE=40
SLICES={'client_position':(0,2),'relative_velocity':(2,4),'client_heading_sin_cos':(4,6),
        'desired_client_frame_offset':(6,8),'client_valid_fresh':(8,10),
        'robot_pelvis_velocity':(10,13),'torso_projected_gravity':(13,16),
        'torso_angular_velocity':(16,19),'previous_applied_action':(19,22),'pedestrian_slots':(22,40)}


def build_observation(client,robot,torso_gravity,torso_angular,previous_applied,desired_offset,config):
    x,y,z=robot['position'];w,qx,qy,qz=robot['quaternion']
    yaw=np.arctan2(2*(w*qz+qx*qy),1-2*(qy*qy+qz*qz))
    R=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
    relative=R.T@(np.array([client.x,client.y])-[x,y])
    relative_v=R.T@(np.array([client.vx,client.vy])-np.array(robot['linear_velocity_world'][:2]))
    delta=client.yaw-yaw;o=config['observation']
    values=np.r_[relative/o['position_scale_m'],relative_v/o['relative_velocity_scale_m_s'],
        np.sin(delta),np.cos(delta),np.array(desired_offset)/o['offset_scale_m'],float(client.valid),float(client.valid),
        np.array(robot['body_velocity'])/o['robot_velocity_scales'],torso_gravity,
        np.array(torso_angular)/o['torso_angular_scale_rad_s'],previous_applied,np.zeros(18)]
    if values.shape!=(SIZE,) or not np.isfinite(values).all():raise RuntimeError('Nonfinite/invalid causal observation')
    return np.clip(values,-1,1).astype(np.float32),int(np.count_nonzero(np.abs(values)>1))
