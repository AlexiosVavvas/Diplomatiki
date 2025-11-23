# Aircraft Analysis and Visualization

## Overview
This guide explains how to visualize and analyze fixed-wing aircraft simulation data. The system connects ROS2 nodes with FlightGear flight simulator for real-time 3D visualization and data plotting.

## Quick Start

### 1. Launch FlightGear Simulator
```bash
./launch_flightgear.sh
```
This starts the FlightGear flight simulator in visualization mode, ready to receive aircraft position and attitude data via UDP.

### 2. Start FlightGear Visualizer Node
```bash
ros2 run ergodic_exploration fg_visualizer_node
```
This node subscribes to `agent_1/data` topic and sends the aircraft state to FlightGear via UDP for real-time 3D visualization.

### 3. Launch Dashboard (Top View)
```bash
python dashboard_ros.py --top_view_only --pos_inverted 1
```
Displays a 2D top-down view of the aircraft trajectory.

**Note**: `--pos_inverted 1` is required because aircraft use **NED (North-East-Down)** convention where:
- **North** = +X
- **East** = +Y  
- **Down** = +Z (altitude is negative upward)

### 4. Start Aircraft Data Converter
```bash
ros2 run ergodic_exploration aircraft_data_converter
```
This node converts `agent_1/data` to a more detailed `agent_1/aircraft_data` topic with comprehensive flight information including:

- **Position**: North, East, Down, Altitude (NED frame)
- **Attitude**: Roll, Pitch, Yaw (radians and degrees)
- **Body Velocities**: Forward (u), Sideways (v), Downward (w), Airspeed
- **NED Velocities**: North, East, Down, Climb Rate, Ground Speed
- **Angular Rates**: Roll rate (p), Pitch rate (q), Yaw rate (r)
- **Aerodynamic Angles**: Angle of Attack (α), Sideslip Angle (β)

### 5. Launch PlotJuggler (Data Plotting)
```bash
ros2 run plotjuggler plotjuggler -l plotjuggler_layout.xml --window_title "Aircraft Simulation"
```

**PlotJuggler** is a real-time plotting tool for ROS topics. It allows you to:
- Plot multiple aircraft parameters simultaneously
- Track time-series data (altitude, velocity, angles, etc.)
- Analyze flight dynamics in real-time
- Load pre-configured layouts (like `plotjuggler_layout.xml`)

#### Installing PlotJuggler
```bash
sudo apt install ros-humble-plotjuggler-ros
```


## System Architecture

```
agent_1/data (AgentData)
    ↓
    ├──→ fg_visualizer_node ──→ UDP ──→ FlightGear (3D View)
    ├──→ dashboard_ros.py ──────────→ 2D Top View
    └──→ aircraft_data_converter ──→ agent_1/aircraft_data (AircraftData)
              ↓
         PlotJuggler (Real-time Plots)
```

## Coordinate Systems

- **NED Frame**: North-East-Down (aircraft standard)
  - Origin: Reference point on ground
  - X-axis: North
  - Y-axis: East
  - Z-axis: Down (altitude is negative upward)

- **Body Frame**: Aircraft-fixed reference
  - X-axis: Through aircraft nose (forward)
  - Y-axis: Through right wing
  - Z-axis: Through belly (down)

## Typical Workflow

1. Launch FlightGear for 3D visualization
2. Start the visualizer node to send data to FlightGear
3. Run the dashboard for 2D trajectory view
4. (Optional) Start data converter for detailed analysis
5. (Optional) Open PlotJuggler to plot flight parameters

All components run concurrently and update in real-time as the aircraft simulation runs.
