# Obstacle Configuration with YAML

## Overview
The agent node now supports loading obstacle configurations from a YAML file instead of hardcoding them in the Python code. This makes it easy to modify obstacle layouts without changing the source code.

## YAML Configuration File Structure

The configuration file contains two main sections:

### 1. Regular Obstacles (`obstacles`)
These are fixed-position obstacles that you define explicitly:

```yaml
obstacles:
  # Circle obstacle example
  - pos: [5.0, 5.0]           # Position [x, y]
    dimensions: 0.6           # Radius for circles
    obs_type: 'circle'        # Type: 'circle'
    obs_name: "Circle 1"     # Descriptive name
    # kappa: 1.0              # Optional - uses KAPPA_OBS default if not specified
    # rho0: 0.15              # Optional - uses RHO_OBS default if not specified

  # Rectangle obstacle example  
  - pos: [7.0, 3.0]           # Position [x, y] (center)
    dimensions: [2.0, 1.5]    # [width, height] for rectangles
    obs_type: 'rectangle'     # Type: 'rectangle'
    kappa: 1.2               # Optional - custom value (overrides default)
    rho0: 0.25               # Optional - custom value (overrides default)
    obs_name: "Rectangle 1"   # Descriptive name
```

### 2. Wall Obstacles (`walls`)
Wall obstacles come in two types: **dynamic walls** and **fixed walls**.

#### Dynamic Walls (Auto-positioning)
These are boundary walls that automatically adapt to the agent's exploration bounds:

```yaml
walls:
  - obs_type: 'wall'         # Type: 'wall'
    wall_type: 'bottom'      # 'bottom', 'top', 'left', or 'right'
    obs_name: "Bottom Wall"  # Descriptive name
    # kappa: 0.5             # Optional - uses KAPPA_WALL default if not specified
    # rho0: 1.5              # Optional - uses RHO_WALL default if not specified
```

#### Fixed Walls (Explicit positioning)
These are custom walls with manually specified position and dimensions:

```yaml
walls:
  - obs_type: 'wall'              # Type: 'wall'
    pos: [5.0, 3.0]               # Explicit position [x, y]
    dimensions: [0, 4.0]          # Normal vector [nx, ny] - magnitude = visual length
    obs_name: "Custom Wall"       # Descriptive name
    # kappa: 0.5                  # Optional - uses KAPPA_WALL default if not specified
    # rho0: 1.5                   # Optional - uses RHO_WALL default if not specified
```

**Note**: `dimensions` for walls represents the **normal vector** whose magnitude determines the visual length of the wall. The wall itself is mathematically infinite for collision calculations.

## Usage

### Running with Custom Obstacle File
```bash
python agent_node.py --agent_id 1 --obstacles_yaml /path/to/your/obstacles.yaml
```

### Running with Default File
If no `--obstacles_yaml` argument is provided, it defaults to `obstacles_config.yaml` in the current directory:
```bash
python agent_node.py --agent_id 1
```

## Obstacle Types

### Circle Obstacles
- `pos`: Center position `[x, y]`
- `dimensions`: Single value representing the radius
- `obs_type`: Must be `'circle'`

### Rectangle Obstacles  
- `pos`: Center position `[x, y]`
- `dimensions`: Array `[width, height]`
- `obs_type`: Must be `'rectangle'`

### Wall Obstacles
Wall obstacles support two configuration modes:

#### Dynamic Walls
- `wall_type`: One of `'bottom'`, `'top'`, `'left'`, `'right'`
- Position and dimensions are automatically calculated based on agent's L1/L2 bounds
- `obs_type`: Must be `'wall'`

#### Fixed Walls  
- `pos`: Explicit position `[x, y]`
- `dimensions`: Normal vector `[nx, ny]` where magnitude = visual wall length
  - For vertical walls: `[0, length]` (e.g., `[0, 4.0]` - vertical wall, 4 units long)
  - For horizontal walls: `[length, 0]` (e.g., `[3.0, 0]` - horizontal wall, 3 units long)
  - Wall is mathematically infinite for collision detection, dimensions only affect visualization
- `obs_type`: Must be `'wall'`

## Parameters

### Required Parameters for All Obstacles
- `pos`: Position (except walls which are auto-calculated)
- `dimensions`: Size specification (varies by type)
- `obs_type`: Type of obstacle
- `obs_name`: Human-readable name for the obstacle

### Optional Parameters
- `kappa`: Obstacle parameter used in calculations
  - If not provided, uses `KAPPA_OBS` default for regular obstacles
  - If not provided, uses `KAPPA_WALL` default for wall obstacles  
- `rho0`: Obstacle parameter used in calculations
  - If not provided, uses `RHO_OBS` default for regular obstacles
  - If not provided, uses `RHO_WALL` default for wall obstacles

**Note**: Default values are determined by the agent's model type and are automatically passed from the code.

## Examples

See `obstacles_config.yaml` for a complete example with:
- 9 circular obstacles in a 3x3 grid
- 2 rectangular obstacles
- 4 boundary walls (dynamic)
- Examples of custom fixed walls (commented out)

### Example: Creating Custom Walls

```yaml
walls:
  # Boundary walls (dynamic - auto-positioned)
  - obs_type: 'wall'
    wall_type: 'bottom'
    kappa: 0.5
    rho0: 1.5
    obs_name: "Bottom Boundary"
  
  # Custom vertical wall in the middle (fixed)
  - obs_type: 'wall'
    pos: [5.0, 5.0]           # Center of wall
    dimensions: [0, 3.0]      # Normal vector: vertical, magnitude 3.0 (visual length)
    kappa: 1.0
    rho0: 0.8
    obs_name: "Center Divider"
  
  # Custom horizontal wall (fixed)  
  - obs_type: 'wall'
    pos: [2.5, 7.0]           # Center of wall
    dimensions: [4.0, 0]      # Normal vector: horizontal, magnitude 4.0 (visual length)
    kappa: 1.0
    rho0: 0.8
    obs_name: "Top Barrier"
```

### Converting from Code to YAML

If you have existing wall code like:
```python
Obstacle(pos=[agent.L1_min + agent.L1_size/2, agent.L2_min], 
         dimensions=[0, +L1], obs_type='wall', 
         kappa=KAPPA_WALL, rho0=RHO_WALL, obs_name="Bottom Wall")
```

Convert it to YAML as:
```yaml
- obs_type: 'wall'
  pos: [5.0, 0.0]      # Replace with actual calculated values  
  dimensions: [0, 10]   # Normal vector: replace +L1 with actual L1 value (visual length)
  kappa: 0.5           # Replace KAPPA_WALL with actual value
  rho0: 1.5            # Replace RHO_WALL with actual value  
  obs_name: "Bottom Wall"
```

**Important**: The `dimensions` parameter represents the **normal vector** whose magnitude determines the visual length of the wall. The wall is mathematically infinite for collision calculations.

## Error Handling

The system will:
- Print warnings if the YAML file is not found
- Display parsing errors if the YAML format is invalid
- Show missing key errors if required parameters are not provided
- Continue with an empty obstacle list if loading fails

## Modifying Obstacles

To change the obstacle layout:
1. Edit your YAML configuration file
2. Restart the agent node
3. The new obstacle configuration will be loaded automatically

No code compilation is needed - just modify the YAML file and restart!