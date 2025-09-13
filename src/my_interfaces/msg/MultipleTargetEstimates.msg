## MultipleTargetEstimates Message
# This message contains an array of target estimates for multiple detected targets.

std_msgs/Header header           # Standard ROS header with timestamp and frame_id

int32 num_of_targets
SingleTargetEstimate[] target_estimates

SingleTargetEstimate[] ground_truths