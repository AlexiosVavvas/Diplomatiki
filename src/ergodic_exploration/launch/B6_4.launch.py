#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

# def _uNomAddTrim(x, t):
#     if t < 0.1:
#         return dynamic_model.u_trim + np.array([0, 0.1, 0, 0.05])
#     else:
#         return dynamic_model.u_trim + np.array([0, 0, 0, 0.05])


def generate_launch_description():
    
    # Define agent configurations (agent_id, x_pos, y_pos)
    agent_configs = [
        {'agent_id': '1', 'y_pos': '80',  'x_pos': '110', 'z_pos':'-400', 'init_psi_deg': '180', 'l_bounds' : ['-20.0', '180.0', '-20.0', '180.0'],  'antenna_rad': '0.01', 'antenna_range_flag': 'true', 'talk_alike_flag' : 'false', 'agent_config': 'src/ergodic_exploration/launch/B6_agent.yaml'},
        {'agent_id': '2', 'y_pos': '110',  'x_pos': '80', 'z_pos':'-400', 'init_psi_deg': '-90', 'l_bounds' : ['-20.0', '180.0', '-20.0', '180.0'],  'antenna_rad': '0.01', 'antenna_range_flag': 'true', 'talk_alike_flag' : 'false', 'agent_config': 'src/ergodic_exploration/launch/B6_agent.yaml'},
        {'agent_id': '3', 'x_pos': '80',   'y_pos': '50',  'z_pos':'-400', 'init_psi_deg': '90',  'l_bounds' : ['-20.0', '180.0', '-20.0', '180.0'],  'antenna_rad': '0.01', 'antenna_range_flag': 'true', 'talk_alike_flag' : 'false', 'agent_config': 'src/ergodic_exploration/launch/B6_agent.yaml'},
        {'agent_id': '4', 'x_pos': '50',   'y_pos': '80',  'z_pos':'-400', 'init_psi_deg': '0',  'l_bounds' : ['-20.0', '180.0', '-20.0', '180.0'],  'antenna_rad': '0.01', 'antenna_range_flag': 'true', 'talk_alike_flag' : 'false', 'agent_config': 'src/ergodic_exploration/launch/B6_agent.yaml'},
    ]
    
    # Extract all agent IDs for sync (as strings for command line)
    all_agent_ids = [config['agent_id'] for config in agent_configs]
    
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
            '--init_pos', config['x_pos'], config['y_pos'], config['z_pos'], config['init_psi_deg'],
            '--l_bounds'] + config['l_bounds'] + [
            '--antenna_rad', config['antenna_rad'],
            '--antenna_range_flag', config['antenna_range_flag'],
            '--talk_alike_flag', config['talk_alike_flag'],
            '--same_l_bounds_flag', 'true',
            # Clock synchronization - all agents sync with each other
            # sync_freq: 50 iterations = 0.025s sim time (good for collision testing at 16 m/s)
            '--sync_clocks_flag', 'true',
            '--sync_agent_ids'] + all_agent_ids + [
            '--sync_freq', '1',
            '--obstacles_yaml', 'src/ergodic_exploration/launch/B6_obs.yaml',
            '--kmax', '2',
            # '--ros-args', '--log-level', 'WARN',
            # '--show_init_phi', 'true',
            ],
            output='screen',
            emulate_tty=True,
        )
        nodes.append(agent_node)
    
    return LaunchDescription([

    ] + nodes)
