# Determine trim speed: 
python more/trim_speed_analysis.py

# --------------------------------------------

./launch_flightgear.sh

ros2 run ergodic_exploration fg_visualizer_node

ros2 run ergodic_exploration aircraft_data_converter

ros2 run plotjuggler plotjuggler -l plotjuggler_layout_obs_3.xml --window_title "Aircraft Simulation" -n

python dashboard_ros.py --top_view_only --pos_inverted 1 --skip-points 1
python dashboard_ros.py --3d --z-bounds 390 410 --pos_inverted 1 --wing-trails 2 --skip-points 10 --follow 1 --no-axis --fancy --max-path-points 20000 --camera-distance 50

-----------------------

ros2 run ergodic_exploration joystick_node 

ros2 launch ergodic_exploration fixed_wing_teleop_agent.launch.py

-----------------------
