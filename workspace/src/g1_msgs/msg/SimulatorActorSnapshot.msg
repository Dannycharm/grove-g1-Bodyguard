# EVALUATOR-ONLY current simulator truth; never connect directly to a robot policy.
# header is capture time mapped onto relay ROS wall time, frame_id=mujoco_world.
std_msgs/Header header
uint64 epoch
float64 simulation_time_s
geometry_msgs/Pose sensor_pose
string sensor_frame_id
geometry_msgs/Pose pelvis_pose
bool pelvis_valid
string[] body_names
geometry_msgs/Pose[] body_poses
