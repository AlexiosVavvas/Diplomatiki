# ROS Bag Player with Dummy Agents

This script (`bagplay.sh`) allows you to play ROS bags while spawning dummy agent nodes that your programs can recognize.

## Features

- Spawns configurable number of dummy agent nodes (default: `agent_1`, `agent_2`)
- Plays ROS bag files
- Properly terminates all processes when Ctrl+C is pressed
- Customizable agent names and prefixes
- Status publishing from each dummy agent

## Usage

```bash
# Basic usage with default 2 agents (agent_1, agent_2)
./bagplay.sh rosbag/test

# Specify number of agents
./bagplay.sh -n 4 rosbag/airplane_hermes_1

# Custom agent prefix
./bagplay.sh --prefix drone_ --num-agents 3 rosbag/drones_in_small_space

# Show help
./bagplay.sh --help
```

## Options

- `-n, --num-agents NUM`: Number of dummy agent nodes to spawn (default: 2)
- `-p, --prefix PREFIX`: Prefix for agent node names (default: 'agent_')
- `-h, --help`: Show help message

## What it does

1. **Sources ROS environment**: Automatically sources your local workspace or system ROS installation
2. **Spawns dummy agents**: Creates nodes with names like `agent_1`, `agent_2`, etc.
3. **Publishes status**: Each agent publishes status messages on `/{agent_name}/status`
4. **Plays ROS bag**: Starts the bag playback
5. **Clean shutdown**: When you press Ctrl+C, all processes are properly terminated

## Files

- `bagplay.sh`: Main script
- `dummy_agent.py`: Python script that creates individual dummy agent nodes
- `test_dummy_agents.sh`: Test script to verify dummy agents work correctly

## Testing

You can test the dummy agents independently:

```bash
# Test the dummy agents
./test_dummy_agents.sh

# Check running nodes
ros2 node list

# Listen to agent status
ros2 topic echo /agent_1/status
```

## Troubleshooting

1. **Permission denied**: Make sure scripts are executable with `chmod +x *.sh`
2. **ROS not sourced**: Script tries to auto-source ROS, but you can manually source before running
3. **Python not found**: Make sure Python 3 and rclpy are installed
4. **Nodes not visible**: Check if ROS_DOMAIN_ID matches between processes

## Example Output

```
Starting ROS bag playback with dummy agents...
Bag path: rosbag/test
Number of agents: 2
Agent prefix: agent_
Press Ctrl+C to stop all processes

Sourced local ROS workspace
Spawning dummy agent nodes...
Starting dummy node: agent_1
Starting dummy node: agent_2
All dummy agent nodes started.

Starting ROS bag playback...
[INFO] [rosbag2_player]: Playing from 'rosbag/test'
...
```