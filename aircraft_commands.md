# Determine trim speed: 
python more/trim_speed_analysis.py

# --------------------------------------------

./launch_flightgear.sh

ros2 run ergodic_exploration fg_visualizer_node

ros2 run ergodic_exploration aircraft_data_converter

ros2 run plotjuggler plotjuggler -l plotjuggler_layout.xml --window_title "Aircraft Simulation" -n

python dashboard_ros.py --top_view_only --pos_inverted 1

ros2 run ergodic_exploration joystick_node 

ros2 launch ergodic_exploration fixed_wing_teleop_agent.launch.py

