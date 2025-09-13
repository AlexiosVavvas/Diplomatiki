# Target estimate message for ergodic exploration
# Contains 3D position estimate and 3x3 covariance matrix

# Target identification
int32 target_id -1               # Unique identifier for the target

# Position estimate (3D)
geometry_msgs/Point position     # Estimated position [x, y, z]

# Covariance matrix (3x3 flattened to 9 elements)
# Represents uncertainty in position estimate
# Order: [xx, xy, xz, yx, yy, yz, zx, zy, zz]
float64[9] covariance [0,0,0,0,0,0,0,0,0]  # 3x3 covariance matrix (row-major order)
