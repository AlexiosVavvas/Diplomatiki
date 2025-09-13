# Agent Configuration System

This directory contains YAML configuration files for different agent types in the ergodic exploration system. The configuration system allows you to define agent parameters in structured YAML files instead of hardcoding them in the source code.

## Overview

The agent configuration system replaces the hardcoded model parameters in `agent_node.py` with flexible YAML-based configurations. This makes it easier to:

- Create new agent configurations without modifying code
- Share and version control agent parameters
- Quickly switch between different agent setups
- Experiment with different parameter combinations

## Configuration Structure

Each agent configuration file contains the following main sections:

### 1. Agent Section
- `model_type`: The dynamics model to use (e.g., "DoubleIntegrator", "SingleIntegrator")
- `dynamics`: Model-specific dynamics parameters (dt, mass, damping, etc.)
- `control`: Ergodic control parameters (limits, gains, timing, safety parameters)
- `system`: System-level parameters (max iterations, publish frequency)

### 2. Targets Section
- `real_positions`: Ground truth target positions [x, y, z]
- `ekf`: Extended Kalman Filter parameters for target estimation

### 3. Flags Section
- System behavior flags (localise_targets, update_eid, save_images)

## Available Configurations

| Configuration File | Model Type | Description |
|-------------------|------------|-------------|
| `default.yaml` | DoubleIntegrator | Default configuration with balanced performance |
| `single_integrator.yaml` | SingleIntegrator | Simple point mass model |
| `double_integrator.yaml` | DoubleIntegrator | Point mass with inertia |
| `simple_boat_second_order.yaml` | SimpleBoatSecondOrder | Marine vehicle dynamics |
| `simple_car_second_order.yaml` | SimpleCarSecondOrder | Ground vehicle dynamics |
| `fixed_wing_12dof_trainer.yaml` | FixedWing12DOFTrainer | Aircraft dynamics model |
| `quadcopter.yaml` | Quadcopter | Quadrotor dynamics with 12 DOF and LQR stabilization |

## Usage

### Basic Usage
To use a specific configuration, specify the `--agent_config` argument:

```bash
# Use double integrator configuration
ros2 run ergodic_exploration agent_node.py --agent_id 1 --agent_config src/ergodic_exploration/agent_configs/double_integrator.yaml

# Use boat configuration
ros2 run ergodic_exploration agent_node.py --agent_id 2 --agent_config src/ergodic_exploration/agent_configs/simple_boat_second_order.yaml
```

### Command Line Overrides
You can still override specific parameters via command line arguments:

```bash
# Override model type from configuration
ros2 run ergodic_exploration agent_node.py --agent_id 1 --agent_config src/ergodic_exploration/agent_configs/default.yaml --model_type SingleIntegrator

# Override antenna radius
ros2 run ergodic_exploration agent_node.py --agent_id 1 --agent_config src/ergodic_exploration/agent_configs/default.yaml --antenna_rad 5.0
```

### Default Behavior
If no `--agent_config` is specified, the system uses `src/ergodic_exploration/agent_configs/default.yaml`.

## Creating Custom Configurations

To create a new agent configuration:

1. Copy an existing configuration file that's closest to your needs
2. Modify the parameters as required
3. Save with a descriptive name (e.g., `my_custom_agent.yaml`)
4. Use with `--agent_config path/to/my_custom_agent.yaml`

### Key Parameters to Consider

#### Control Limits
- `u_limits_init`: Initial control limits applied at startup
- `u_limits`: Full control limits applied after `time_to_apply_ulimits` seconds
- `time_to_apply_ulimits`: Time to wait before applying full limits

#### Timing Parameters
- `ts`: Sampling time (how often ergodic control is computed)
- `t_h`: Prediction horizon for control optimization
- `delta_t_erg`: Time window for ergodic cost calculation

#### Safety Parameters (CBF - Control Barrier Function)
- `delta_safe`: Safety margin for obstacles
- `kappa_obs`: Obstacle avoidance gain
- `rho_obs`: Obstacle influence distance

## Example Configuration

```yaml
agent:
  model_type: "DoubleIntegrator"
  
  dynamics:
    dt: 0.0012
    damping: 2
    
  control:
    u_limits_init: [[-50, 50], [-50, 50]]
    u_limits: [[-50, 50], [-50, 50]]
    Q: 8
    R: [[0.001, 0.0], [0.0, 0.001]]
    ts: 0.03
    t_h: 0.5

targets:
  real_positions:
    - [2.0, 2.0, 0.0]
    - [4.0, 8.0, 0.0]
  
  ekf:
    sensor_range: 3.0
    R_diag: [0.1, 0.1]

flags:
  localise_targets: true
  update_eid: false
```

## Backward Compatibility

The old command line arguments are still supported for overriding configuration values:
- `--model_type`: Override model type from config
- `--antenna_rad`: Override antenna radius
- `--kmax`: Override maximum Fourier modes
- `--antenna_range_flag`: Override antenna range flag
- `--talk_alike_flag`: Override talk alike flag
- `--same_l_bounds_flag`: Override same L bounds flag
