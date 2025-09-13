#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    
    # Define agent configurations (agent_id, x_pos, y_pos)
    agent_configs = [
        {'agent_id': '1', 'x_pos': '-16',  'y_pos': '-4', 'l_bounds' : ['-20.0', '20.0', '-20.0', '20.0'], 'antenna_rad': '0.1', 'antenna_range_flag': 'false', 'talk_alike_flag' : 'false', 'agent_config' : 'src/ergodic_exploration/launch/three_drones_big_agents.yaml'},
        {'agent_id': '2', 'x_pos': '-6',  'y_pos': '3',  'l_bounds' : ['-20.0', '20.0', '-20.0', '20.0'], 'antenna_rad': '0.1', 'antenna_range_flag': 'false', 'talk_alike_flag' : 'false', 'agent_config' : 'src/ergodic_exploration/launch/three_drones_big_agents.yaml'},
        {'agent_id': '3', 'x_pos': '18',  'y_pos': '4',   'l_bounds' : ['-20.0', '20.0', '-20.0', '20.0'], 'antenna_rad': '0.1', 'antenna_range_flag': 'false', 'talk_alike_flag' : 'false', 'agent_config' : 'src/ergodic_exploration/launch/three_drones_big_agents.yaml'},
        {'agent_id': '4', 'x_pos': '7',  'y_pos': '-10',  'l_bounds' : ['-20.0', '20.0', '-20.0', '20.0'], 'antenna_rad': '0.1', 'antenna_range_flag': 'false', 'talk_alike_flag' : 'false', 'agent_config' : 'src/ergodic_exploration/launch/three_drones_big_agents.yaml'},
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
            '--agent_config', config['agent_config'],
            '--agent_id', config['agent_id'],
            '--init_pos', config['x_pos'], config['y_pos'],
            '--l_bounds'] + config['l_bounds'] + [
            '--antenna_rad', config['antenna_rad'],
            '--antenna_range_flag', config['antenna_range_flag'],
            '--talk_alike_flag', config['talk_alike_flag'],
            '--same_l_bounds_flag', 'true',
            '--kmax', '10',
            '--ros-args', '--log-level', 'WARN'
            ],
            output='screen',
            emulate_tty=True,
        )
        nodes.append(agent_node)
    
    return LaunchDescription([

    ] + nodes)
