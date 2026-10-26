# One causal observation. Never a future trajectory. Pose/velocity use array header.frame_id.
string actor_id
string role
geometry_msgs/Pose pose
geometry_msgs/Twist velocity
float64 heading
bool valid
bool velocity_valid
string observation_source
string invalid_reason
