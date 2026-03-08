# Synchronization barrier message for multi-agent clock synchronization
# Each agent publishes this to signal they've completed a simulation step

int32 agent_id              # ID of the agent publishing this message
int32 sim_step              # Current simulation step index
float64 sim_time            # Current simulation time
builtin_interfaces/Time timestamp  # ROS timestamp when message was sent
