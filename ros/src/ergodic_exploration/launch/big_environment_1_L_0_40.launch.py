#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    
    # Define agent configurations (agent_id, x_pos, y_pos)
    agent_configs = [
        {'agent_id': '1', 'x_pos': '9',  'y_pos': '5', 'l_bounds' : ['0.0', '40.0', '0.0', '40.0'],  'model_type': 'SimpleCarSecondOrder', 'antenna_rad': '0.1', 'antenna_range_flag': 'false', 'talk_alike_flag' : 'false'},
        {'agent_id': '2', 'x_pos': '15', 'y_pos': '16', 'l_bounds' : ['0.0', '40.0', '0.0', '40.0'],  'model_type': 'SimpleCarSecondOrder', 'antenna_rad': '0.1', 'antenna_range_flag': 'false', 'talk_alike_flag' : 'false'},
        {'agent_id': '3', 'x_pos': '25', 'y_pos': '18', 'l_bounds' : ['0.0', '40.0', '0.0', '40.0'],  'model_type': 'SimpleCarSecondOrder', 'antenna_rad': '0.1', 'antenna_range_flag': 'false', 'talk_alike_flag' : 'false'},
        # {'agent_id': '4', 'x_pos': '35', 'y_pos': '18', 'l_bounds' : ['0.0', '40.0', '0.0', '40.0'],  'model_type': 'SimpleCarSecondOrder', 'antenna_rad': '0.1', 'antenna_range_flag': 'false', 'talk_alike_flag' : 'false'},
        # {'agent_id': '5', 'x_pos': '8',  'y_pos': '25', 'l_bounds' : ['0.0', '40.0', '0.0', '40.0'],  'model_type': 'SimpleCarSecondOrder', 'antenna_rad': '0.1', 'antenna_range_flag': 'false', 'talk_alike_flag' : 'false'},
        # {'agent_id': '6', 'x_pos': '18', 'y_pos': '30', 'l_bounds' : ['0.0', '40.0', '0.0', '40.0'],  'model_type': 'SimpleCarSecondOrder', 'antenna_rad': '0.1', 'antenna_range_flag': 'false', 'talk_alike_flag' : 'false'},
        # {'agent_id': '7', 'x_pos': '30', 'y_pos': '35', 'l_bounds' : ['0.0', '40.0', '0.0', '40.0'],  'model_type': 'SimpleCarSecondOrder', 'antenna_rad': '0.1', 'antenna_range_flag': 'false', 'talk_alike_flag' : 'false'},
        # {'agent_id': '8', 'x_pos': '12', 'y_pos': '15', 'l_bounds' : ['0.0', '40.0', '0.0', '40.0'],  'model_type': 'SimpleCarSecondOrder', 'antenna_rad': '0.1', 'antenna_range_flag': 'false', 'talk_alike_flag' : 'false'},
        # {'agent_id': '9', 'x_pos': '22', 'y_pos': '22', 'l_bounds' : ['0.0', '40.0', '0.0', '40.0'],  'model_type': 'DoubleIntegrator', 'antenna_rad': '0.1', 'antenna_range_flag': 'false', 'talk_alike_flag' : 'false'},
        # {'agent_id': '10', 'x_pos': '32', 'y_pos': '28', 'l_bounds' : ['0.0', '40.0', '0.0', '40.0'],  'model_type': 'DoubleIntegrator', 'antenna_rad': '0.1', 'antenna_range_flag': 'false', 'talk_alike_flag' : 'false'},
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
            '--model_type', config['model_type'],
            '--agent_id', config['agent_id'],
            '--init_pos', config['x_pos'], config['y_pos'],
            '--l_bounds'] + config['l_bounds'] + [
            '--antenna_rad', config['antenna_rad'],
            '--antenna_range_flag', config['antenna_range_flag'],
            '--talk_alike_flag', config['talk_alike_flag'],
            '--same_l_bounds_flag', 'true',
            '--kmax', '4',
            '--obstacles_yaml', 'src/ergodic_exploration/launch/big_environment_1_L_0_40.yaml',
            '--ros-args', '--log-level', 'WARN'
            ],
            output='screen',
            emulate_tty=True,
        )
        nodes.append(agent_node)
    
    return LaunchDescription([

    ] + nodes)
