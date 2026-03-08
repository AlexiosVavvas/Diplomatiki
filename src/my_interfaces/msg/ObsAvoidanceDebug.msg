# ObsAvoidanceDebug Message
# This ROS message contains information about an agent's state regarding the obstacle avoidance tuning

std_msgs/Header header

float64 psi
float64 h
float64 hdot
float64 hddot
float64 two_alpha_h_hdot
float64 alpha2_h
float64[] beta
float64[] u_safe