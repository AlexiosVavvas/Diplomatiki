# Dashboard ROS2 Visualization

Real-time multi-agent visualization dashboard for ROS2.

## Basic Usage

```bash
# Source ROS2 and run
source install/setup.bash
python dashboard_ros.py
```

## Common Options

### Filter Agents
```bash
# Show only agents 1 and 2
python dashboard_ros.py --agents 1 2
```

### 3D Visualization
```bash
# Enable 3D view
python dashboard_ros.py --3d
```

### Aircraft Mode (NED Convention)
```bash
# For airplane agents using NED coordinates
python dashboard_ros.py --3d --pos_inverted 1 2
```

### Wing Trails
```bash
# Add smoke trails from wing tips (wingspan in meters)
python dashboard_ros.py --3d --pos_inverted 1 --wing-trails 10
```

### Camera Follow Mode
```bash
# Follow agent 1 with keyboard camera control
python dashboard_ros.py --3d --pos_inverted 1 --follow 1 --camera-distance 150
```

**Camera controls:**
- `↑/↓` — Elevation angle
- `←/→` — Azimuth angle  
- `+/-` — Zoom in/out
- `n` — Cycle to next agent

### Performance Options
```bash
# Reduce data points for smoother rendering
python dashboard_ros.py --3d --skip-points 3 --max-path-points 500
```

### Clean Screenshots
```bash
# Hide axis for cleaner visuals
python dashboard_ros.py --3d --no-axis
```

### Publication-Quality Visuals (Fancy Mode) 
```bash
# Clean light theme with gradient trails, ground shadows, professional styling
python dashboard_ros.py --3d --fancy --no-axis --wing-trails 10 --pos_inverted 1
```

Features:
- Clean white background (paper-ready)
- Gradient trajectory trails that fade from light to saturated
- Subtle ground shadow projection for depth
- Light gray ground grid
- Professional obstacle styling with red edge highlights
- Gradient wing ribbon trails

## Keyboard Controls

| Key | Action |
|-----|--------|
| `e` | Manual refresh |
| `a` | Toggle auto-refresh |
| `c` | Clear plots |
| `q` | Quit |

## Full Example

```bash
# Aircraft visualization with all features (publication-ready)
python dashboard_ros.py \
    --pos_inverted 1 2 3 \
    --3d \
    --fancy \
    --wing-trails 10 \
    --follow 1 \
    --camera-distance 200 \
    --z-bounds 300 500 \
    --max-path-points 3000 \
    --skip-points 9 \
    --no-axis 
```
