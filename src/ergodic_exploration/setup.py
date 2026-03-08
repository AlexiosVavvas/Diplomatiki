from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ergodic_exploration'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']) + ['my_erg_lib'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py') + glob('launch/*.launch.yaml')),
        (os.path.join('share', package_name, 'meshes'), glob('meshes/*.stl')),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'scipy', 
        'matplotlib',
        'flightgear-python'
    ],
    zip_safe=True,
    maintainer='alex',
    maintainer_email='alexios.vavvas@gmail.com',
    description='TODO: Package description',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'agent_node = ergodic_exploration.agent_node:main',
            'environment_node = ergodic_exploration.environment:main',
            'fg_visualizer_node = ergodic_exploration.fg_visualizer_node:main',
            'aircraft_data_converter = ergodic_exploration.aircraft_data_converter:main',
            'joystick_node = ergodic_exploration.joystick_node:main',
            'agent_node_airplane_teleop = ergodic_exploration.agent_node_airplane_teleop:main',
            'tf_visualizer_airplane = ergodic_exploration.tf_visualizer_airplane:main',
        ],
    },
)
