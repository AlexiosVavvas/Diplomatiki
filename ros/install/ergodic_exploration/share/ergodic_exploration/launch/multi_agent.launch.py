#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # Declare launch arguments for customization
    num_agents_arg = DeclareLaunchArgument(
        'num_agents',
        default_value='5',
        description='Number of agents to launch'
    )
    
    # Define agent configurations (agent_id, x_pos, y_pos)
    agent_configs = [
        {'agent_id': '1', 'x_pos': '1', 'y_pos': '1', 'antenna_rad': '6.0', 'antenna_range_flag': 'true'},
        {'agent_id': '2', 'x_pos': '3', 'y_pos': '2', 'antenna_rad': '6.0', 'antenna_range_flag': 'true'},
        {'agent_id': '3', 'x_pos': '5', 'y_pos': '7', 'antenna_rad': '6.0', 'antenna_range_flag': 'true'},
        {'agent_id': '4', 'x_pos': '7', 'y_pos': '8', 'antenna_rad': '6.0', 'antenna_range_flag': 'true'},
        {'agent_id': '5', 'x_pos': '9', 'y_pos': '1', 'antenna_rad': '6.0', 'antenna_range_flag': 'true'},
    ]
    
    # Create nodes list
    nodes = []
    
    # Launch each agent
    for i, config in enumerate(agent_configs):
            
        agent_node = Node(
            package='ergodic_exploration',
            executable='agent_node',
            name=f'agent_{config["agent_id"]}',
            parameters=[],
            arguments=[
            '--agent_id', config['agent_id'],
            '--init_pos', config['x_pos'], config['y_pos'],
            '--antenna_rad', config['antenna_rad'],
            '--antenna_range_flag', config['antenna_range_flag'],
            '--ros-args', '--log-level', 'WARN'
            ],
            output='screen',
            emulate_tty=True,
        )
        nodes.append(agent_node)
    
    return LaunchDescription([
        num_agents_arg,
    ] + nodes)
