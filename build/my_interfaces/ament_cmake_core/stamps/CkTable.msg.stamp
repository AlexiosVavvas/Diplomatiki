# Table of floats message
# Represents a square table/matrix of float values

string model_type         # Model type
float64[] l_bounds        # [x_min, x_max, y_min, y_max] L1, L2 boundaries for ergodic search in space

int32 table_size          # Size of the square table (size x size)
float64[] ck_values       # Flattened array of float values (row-major order)
float64[] ck_values_average_in_range  # Mean Ck array for agents in range as seen by a particular agent

float64 total_erg_cost    # Total ergodic cost for all agents now
float64 total_erg_cost_in_range # Total ergodic cost for all in range agents now
float64 erg_cost_reduction_perc # Reduction percentage of Total ergodic cost for all in range agents now. For example if init is 100 and now 90 -> 10%

geometry_msgs/Point position     # Current Agent position [x, y, z] (used to simulate short antenna range)
