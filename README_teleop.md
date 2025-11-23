# Fixed-Wing Aircraft Teleoperation

Manual control of fixed-wing aircraft models using an Arduino joystick instead of the ergodic controller.

## Overview

The teleoperation system allows you to directly control the aircraft's control surfaces (elevator, ailerons, rudder) and throttle using an Arduino joystick. This is useful for:

- **Testing aircraft dynamics** - Feel how the aircraft responds to control inputs
- **Manual exploration** - Fly the aircraft yourself to explore the environment
- **Debugging** - Compare manual control with ergodic control behavior
- **Training data collection** - Generate training data for learning-based controllers

## Prerequisites

### 1. Arduino Joystick Setup

Connect your Arduino joystick to the computer via USB. The joystick should output data in the format:
```
/*value1,value2,value3,...,valueN*/
```

The system expects these channels:
- **Channel 1**: Throttle stick
- **Channel 3**: Aileron stick (roll control)
- **Channel 5**: Elevator stick (pitch control)
- **Channel 7**: Rudder stick (yaw control)
- **Channel 9**: Switch (reset to trim)

### 2. Serial Port Permissions

```bash
# Add your user to the dialout group
sudo usermod -a -G dialout $USER

# Or give temporary access (until device is unplugged)
sudo chmod 666 /dev/ttyUSB0

# Then log out and back in
```

### 3. Install Python Dependencies

```bash
# Install pyserial (NOT serial)
pip install pyserial
```

### 4. Verify Arduino Connection

```bash
# Check if Arduino is detected
ls /dev/ttyUSB*

# Test raw serial output (Ctrl+A then K to exit screen)
sudo screen /dev/ttyUSB0 9600
```

## Joystick Control Mapping

### Arduino Joystick Layout

| Control Surface | Joystick Channel | Description |
|----------------|------------------|-------------|
| **Throttle** | Channel 1 | Full range: min to max throttle |
| **Aileron** (Roll) | Channel 3 | Left/Right → roll left/right |
| **Elevator** (Pitch) | Channel 5 | Up/Down → pitch up/down |
| **Rudder** (Yaw) | Channel 7 | Left/Right → yaw left/right |
| **Reset Switch** | Channel 9 | Toggle to reset aircraft to trim state |

### Control Surface Details

The joystick values are normalized to [-1, 1] during calibration and then mapped to the aircraft model's input limits:

- **Elevator**: Controls pitch angle (nose up/down)
  - Model limits: typically ±25° (±0.4363 radians)
  - Used for climbing/descending and speed control

- **Ailerons**: Controls roll angle (banking left/right)
  - Model limits: typically ±25° (±0.4363 radians)
  - Used for turning

- **Rudder**: Controls yaw angle (nose left/right)
  - Model limits: typically ±25° (±0.4363 radians)
  - Used for coordination in turns and crosswind correction

- **Throttle**: Controls engine power
  - Model limits: 0-100% (0.0 to 1.0)
  - Affects forward speed

## Usage

### Manual Node Launch 

If you want more control, launch nodes individually:

```bash
# Terminal 1: Joystick node (Arduino interface with calibration)
ros2 run ergodic_exploration joystick_node

# Terminal 2: Teleoperation agent
ros2 run ergodic_exploration agent_node_airplane_teleop \
    --agent_id 1 \
    --init_pos 50 0 -200 \
    --l_bounds -20 180 -20 180 \
    --kmax 4 \
    --agent_config src/ergodic_exploration/launch/fixed_wing_free_fly_agent.yaml

# Terminal 3: Environment node (visualization)
ros2 run ergodic_exploration environment_node
```

## Visualization

For detailed instructions on visualization with FlightGear and PlotJuggler, see [README_aircraft.md](README_aircraft.md).

**Quick reference:**
```bash
# FlightGear 3D visualization
./launch_flightgear.sh
ros2 run ergodic_exploration fg_visualizer_node

# PlotJuggler for data plotting
ros2 run plotjuggler plotjuggler -l plotjuggler_layout.xml
```

**Teleoperation-specific topics to monitor:**
- `/joystick` - Raw joystick inputs (throttle, aileron, elevator, rudder, switch)
- `/agent_1/data` - Agent state (position, attitude, velocities, control inputs)
- `/agent_1/aircraft_data` - Detailed flight parameters (after running aircraft_data_converter)

## Joystick Calibration

When the joystick node starts, it will guide you through calibration:

1. **Throttle Calibration**: 
   - Press Enter when ready
   - Move throttle stick through its full range for 5 seconds
   
2. **Aileron Calibration**:
   - Press Enter when ready
   - Move aileron stick fully left and right for 5 seconds

3. **Elevator Calibration**:
   - Press Enter when ready
   - Move elevator stick fully up and down for 5 seconds

4. **Rudder Calibration**:
   - Press Enter when ready
   - Move rudder stick fully left and right for 5 seconds

5. **Switch Calibration**:
   - Toggle the switch several times
   - System detects the two states automatically

After calibration, the system displays the detected ranges and starts publishing normalized joystick data.

## Configuration

### Agent Configuration File

The teleoperation node uses the same configuration files as `agent_node`. Example configuration:

```yaml
# launch/fixed_wing_free_fly_agent.yaml
model_type: "FixedWing12DOFTrainer"

dynamics:
  dt: 0.001             # Simulation timestep (seconds)
  v_trim: 10.0          # Trim speed (m/s)
  use_linear_f: false   # Use linear dynamics model
  use_linear_fx_fu: false
```

### Joystick Node Parameters

You can customize the joystick node behavior:

```bash
ros2 run ergodic_exploration joystick_node \
    --ros-args \
    -p port:=/dev/ttyUSB0 \
    -p baudrate:=9600 \
    -p publish_rate:=50.0 \
    -p calibration_duration:=5.0
```

Parameters:
- `port`: Serial port device (default: /dev/ttyUSB0)
- `baudrate`: Communication speed (default: 9600)
- `publish_rate`: Publishing frequency in Hz (default: 50.0)
- `calibration_duration`: Calibration time per control in seconds (default: 5.0)

## Flying Tips

### Basic Flight

1. **Takeoff**: 
   - Aircraft starts at altitude (80m by default)
   - Increase throttle to 70-80%
   - Keep wings level (ailerons centered)

2. **Level Flight**:
   - Maintain trim speed (~15 m/s)
   - Small pitch adjustments with elevator
   - Use rudder to maintain heading

3. **Turning**:
   - Bank with ailerons (roll)
   - Add rudder in same direction (coordinated turn)
   - Pull back on elevator slightly to maintain altitude

4. **Recovering from Upset**:
   - Press reset button (A/X) to return to trim
   - Or manually: level wings, neutral pitch, adjust throttle

### Advanced Maneuvers

- **Climbing**: Increase throttle + nose up (elevator back)
- **Descending**: Reduce throttle + nose down (elevator forward)
- **Tight Turns**: More bank angle + more elevator + coordinated rudder
- **Slow Flight**: Reduce throttle to ~40%, nose up to maintain altitude

### Common Issues

**Aircraft diving/climbing uncontrollably**:
- Check trim settings in config file
- Press reset button to stabilize
- Adjust throttle to appropriate level

**Aircraft spinning/tumbling**:
- Too much control input at low speed
- Press reset button
- Use gentler control inputs

**Controls feel backwards**:
- Check axis inversion in code
- Some controllers have different axis conventions

## Monitoring and Debugging

### View Real-time State

```bash
# Monitor agent data
ros2 topic echo /agent_1/data

# View joystick input
ros2 topic echo /joystick
```

### Check Control Inputs

The node logs control information periodically:

```
[teleop_agent_1] t=5.23s | Pos: (50.1, 0.2, 79.8) | V: 10.2 m/s | 
φ=2.3° θ=4.1° ψ=180.5° | Throttle: 0.65
```

## Troubleshooting

### Arduino Not Detected

```bash
# Check device permissions
ls -l /dev/ttyUSB0
# Should show crw-rw---- with dialout group

# Add yourself to dialout group if needed
sudo usermod -a -G dialout $USER
# Log out and back in

# Or give temporary access
sudo chmod 666 /dev/ttyUSB0
```

### Joystick Node Not Publishing

```bash
# Check if joystick node is running
ros2 node list | grep joystick

# Check for data on joystick topic
ros2 topic hz /joystick

# Restart with different port if needed
ros2 run ergodic_exploration joystick_node --ros-args -p port:=/dev/ttyUSB1
```

### Aircraft Model Not Loading

```bash
# Verify config file exists
ls src/ergodic_exploration/launch/fixed_wing_free_fly_agent.yaml

# Check for syntax errors in YAML
cat src/ergodic_exploration/launch/fixed_wing_free_fly_agent.yaml
```

### Control Inputs Not Responding

1. Verify joystick topic is publishing: `ros2 topic hz /joystick`
2. Check calibration ranges (run joystick_node again)
3. Test with serial monitor: `sudo screen /dev/ttyUSB0 9600`
4. Check node logs for errors
5. Verify Arduino is sending correct format: `/*value1,value2,...*/`

## Comparison with Ergodic Control

After flying manually, you can compare with autonomous ergodic control:

```bash
# Launch normal ergodic agent
ros2 run ergodic_exploration agent_node \
    --agent_id 0 \
    --init_pos 5.0 5.0 \
    --agent_config src/ergodic_exploration/agent_configs/fixed_wing_12dof_trainer.yaml
```

This helps understand:
- How ergodic controller handles aircraft dynamics
- Differences in trajectory smoothness
- Control input usage patterns
- Exploration efficiency

## Building After Changes

If you modify the code, rebuild:

```bash
cd ~/dipl  # Or your workspace directory
colcon build --packages-select ergodic_exploration
source install/setup.bash
```

## Example Scenarios

### Free Flight Exploration

```bash
ros2 launch ergodic_exploration fixed_wing_teleop.launch.py \
    init_pos:="5.0 5.0" \
    l_bounds:="0.0 50.0 0.0 50.0"
```

### Multiple Aircraft (Advanced)

Launch multiple teleoperation nodes with different IDs and joystick devices:

```bash
# Agent 0 with joystick js0
ros2 run ergodic_exploration agent_node_airplane_teleop --agent_id 0 &

# Agent 1 with joystick js1  
# (requires modifying code to accept device parameter)
ros2 run ergodic_exploration agent_node_airplane_teleop --agent_id 1 &
```

## Further Development

Possible enhancements to the teleoperation node:

1. **Flight modes**: Add altitude hold, heading hold, etc.
2. **Multiple joysticks**: Support for multiple controllers
3. **Record/playback**: Record manual flights for replay
4. **HUD overlay**: Display flight info in RViz
5. **Haptic feedback**: Vibration for stall warning, etc.
6. **Custom axis mapping**: Runtime parameter configuration

## Related Files

- `agent_node_airplane_teleop.py` - Main teleoperation node
- `fixed_wing_teleop.launch.py` - Launch file
- `model_dynamics.py` - Aircraft dynamics models
- `agent_configs/fixed_wing_12dof_trainer.yaml` - Configuration

## Support

For issues or questions:
1. Check terminal output for error messages
2. Verify joystick connection with `jstest`
3. Test joy_node independently: `ros2 topic echo /joy`
4. Review configuration file syntax

Enjoy flying! 🛩️
