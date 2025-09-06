# Single obstacle message for ergodic exploration
# Supports circle, rectangle, and wall obstacle types

string obs_type          # Type of obstacle: "circle", "rectangle", or "wall"
string obs_name          # Name/ID of the obstacle (optional, for debugging)
geometry_msgs/Point position     # Obstacle position [x, y, z]
float64[] dimensions     # Dimensions based on type:
                         # - circle: [radius]
                         # - rectangle: [width, height]  
                         # - wall: [normal_x, normal_y]
float64 kappa            # Potential field parameter
float64 rho0            # Obstacle vicinity distance