#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # Declare launch arguments
    agent1_id_arg = DeclareLaunchArgument('agent1_id', default_value='1')
    agent1_x_arg = DeclareLaunchArgument('agent1_x', default_value='9')
    agent1_y_arg = DeclareLaunchArgument('agent1_y', default_value='3')
    
    agent2_id_arg = DeclareLaunchArgument('agent2_id', default_value='2')
    agent2_x_arg = DeclareLaunchArgument('agent2_x', default_value='5')
    agent2_y_arg = DeclareLaunchArgument('agent2_y', default_value='7')
    
    agent3_id_arg = DeclareLaunchArgument('agent3_id', default_value='3')
    agent3_x_arg = DeclareLaunchArgument('agent3_x', default_value='2')
    agent3_y_arg = DeclareLaunchArgument('agent3_y', default_value='1')
    
    # Agent nodes
    agent1 = Node(
        package='ergodic_exploration',
        executable='agent_node',
        name='agent_1',
        arguments=[
            '--agent_id', LaunchConfiguration('agent1_id'),
            '--init_pos', LaunchConfiguration('agent1_x'), LaunchConfiguration('agent1_y')
        ],
        output='screen',
        emulate_tty=True,
    )
    
    agent2 = Node(
        package='ergodic_exploration',
        executable='agent_node',
        name='agent_2',
        arguments=[
            '--agent_id', LaunchConfiguration('agent2_id'),
            '--init_pos', LaunchConfiguration('agent2_x'), LaunchConfiguration('agent2_y')
        ],
        output='screen',
        emulate_tty=True,
    )
    
    agent3 = Node(
        package='ergodic_exploration',
        executable='agent_node',
        name='agent_3',
        arguments=[
            '--agent_id', LaunchConfiguration('agent3_id'),
            '--init_pos', LaunchConfiguration('agent3_x'), LaunchConfiguration('agent3_y')
        ],
        output='screen',
        emulate_tty=True,
    )
    
    return LaunchDescription([
        agent1_id_arg, agent1_x_arg, agent1_y_arg,
        agent2_id_arg, agent2_x_arg, agent2_y_arg,
        agent3_id_arg, agent3_x_arg, agent3_y_arg,
        agent1,
        agent2,
        agent3,
    ])
