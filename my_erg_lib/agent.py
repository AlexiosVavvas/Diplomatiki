from my_erg_lib.basis import Basis
import numpy as np
from my_erg_lib.model_dynamics import SingleIntegrator, DoubleIntegrator, Quadcopter
from my_erg_lib.eid import Sensor, EKF
import time

class Agent():
    def __init__(self, L1, L2, Kmax, dynamics_model, phi=None, x0=None, agent_id=None):
        self.agent_id = agent_id

        # Space Parameters
        self.L1 = L1
        self.L2 = L2
        self.Kmax = Kmax
        
        # Connecting model dynamics
        self.model = dynamics_model
        self.model.reset(x0)

        # Lets connect a sensor to track the target position 
        self.sensor = Sensor(sensor_range=0.2,
                             R=np.diag([0.035, 0.01]))          # TODO: Be able to adjust those parameters from outside
        self.real_target_position = np.array([0.8, 0.4, 0])     # Real target position (Ground Truth)   # TODO: Maybe take it from an env?
        self.a = np.array([0.2, 0.6, 0])                        # Current target position estimate      # TODO: When to update it?
        self.ekf = EKF(a_init = self.a,
                       sigma_init = np.eye(3)*1e3,
                       R = np.diag([0.1, 0.1]),         # Sensor noise covariance
                       Q = np.eye(3) * 1e-5,            # Process noise covariance
                       a_limits=[[0, L1], [0, L2], [0, 10]])    # TODO: Why does it look like an ellipse and not a circle at first?

        # Initialise obstacle list
        self.obstacle_list = []

        # Initialize the basis object
        self.basis = Basis(L1, L2, Kmax, phi_=phi, precalc_phik_coeff=False, num_gauss_points=15)

    def modifedPhiForObstacles(self, phi, obs_to_exclude=None, obs_list=None):
        '''
        Wrapper function modifying the original phi function to take into account the obstacles
        Zeros out the phi function in the obstacle area
        '''
        assert callable(phi), "phi must be a callable function."

        # Determine which obstacles to use
        if obs_list is not None:
            pass
        elif obs_to_exclude == "All" or obs_list == []:
            phi.obs_list = []
            return phi
        elif obs_to_exclude == "None":
            # Use all obstacles
            obs_list = self.obstacle_list
        elif obs_to_exclude is not None:
            # Exclude specified obstacles
            obs_list = [obs for obs in self.obstacle_list if obs.name_id not in obs_to_exclude]
        else:
            raise ValueError("obs_to_exclude must be 'All', 'None', or a list of obstacle names.")


        phi.obs_list = obs_list
        def phi_w_obs(x):
            # If we are inside an obstacle, return 0, we dont want to explore ergodically there
            # TODO: If obstacles change position, we need to update the phi coefficients
            for obs in obs_list:
                if obs.withinReach(x):
                    return 0
            return phi(x)
        
        # Attach the obstacle list to the phi function
        phi_w_obs.obs_list = obs_list

        return phi_w_obs

    def updateEIDphiFunction(self, NUM_GAUSS_POINTS=30, P_UPPER_LIM=20, HTA_SCALE=1, FINAL_FI_CLIP=10):

        def phi(x):

            # Append the self.model.state[2] to the state vector
            x = np.append(x, self.model.state[2])
            M  = self.ekf.measurement_model.M   # Number of target estimation states

            # -----------------------------------------------------------------------------
            
            # Expected Information Matrix (Φ)
            # Now we have to integrate the product of I(a) * p(a). We will use Gauss for speed
            # Get Gauss-Legendre quadrature points and weights
            a1_points, a1_weights = np.polynomial.legendre.leggauss(NUM_GAUSS_POINTS)
            a2_points, a2_weights = np.polynomial.legendre.leggauss(NUM_GAUSS_POINTS)
            # Transform from [-1,1] to [0,L1] and [0,L2]
            a1_points  = 0.5 * self.L1 * (a1_points + 1)
            a2_points  = 0.5 * self.L2 * (a2_points + 1)
            a1_weights = 0.5 * self.L1 * a1_weights
            a2_weights = 0.5 * self.L2 * a2_weights

            # -----------------------------------------------------------------------------
            
            # Precalculate I matrices using vectorized approach (cleaner version)
            sigma_inv = np.linalg.inv(self.sensor.R)

            # Create array of all target positions at once
            a1_grid, a2_grid = np.meshgrid(a1_points, a2_points, indexing='ij')
            a_positions = np.column_stack([
                a1_grid.flatten(), 
                a2_grid.flatten(), 
                np.zeros(NUM_GAUSS_POINTS**2)
            ])  # Shape: (NUM_GAUSS_POINTS^2, 3)

            # Vectorized H computation
            H_all = self.ekf.measurement_model.H(a_positions, x)  # Shape: (NUM_GAUSS_POINTS^2, 2, 3)
            
            # Compute Fisher Information matrices vectorized
            # We want: H.T @ sigma_inv @ H for each position
            # H_all.shape = (N, 2, 3), we need H.T which is (N, 3, 2)

            # Transpose H to get H.T
            H_T = H_all.transpose(0, 2, 1)  # Shape: (N, 3, 2)

            # Using einsum
            H_T_sigma_inv   = np.einsum('nij,jk->nik', H_T, sigma_inv)          # Shape: (N, 3, 2)
            I_matrices_flat = np.einsum('nij,njk->nik', H_T_sigma_inv, H_all)   # Shape: (N, 3, 3)

            # Reshape back to grid format
            I_matrices = I_matrices_flat.reshape(NUM_GAUSS_POINTS, NUM_GAUSS_POINTS, M, M)

            # -----------------------------------------------------------------------------
            
            # Precompute probability values
            p_values = np.zeros((NUM_GAUSS_POINTS, NUM_GAUSS_POINTS))
            # Calculate all probabilities at once
            all_probs = self.ekf.p(np.array([[a1, a2, 0] for a1 in a1_points for a2 in a2_points]), upper_lim_to_normalise=P_UPPER_LIM)
            # Reshape the results back to a grid
            p_values = all_probs.reshape(NUM_GAUSS_POINTS, NUM_GAUSS_POINTS)
            # Normalize the probabilities
            # p_values *= 10/np.max(p_values) 
            # p_values = np.clip(p_values, 0, 20)  # Clip to [0, 20] range

            # -----------------------------------------------------------------------------
            
            # Fisher Information calculation

            # Instead of nested loops, use broadcasting and matrix operations
            # Create weight matrices for broadcasting
            a1_weights_grid, a2_weights_grid = np.meshgrid(a1_weights, a2_weights, indexing='ij')
            weights_combined = a1_weights_grid * a2_weights_grid  # Shape: (NUM_GAUSS_POINTS, NUM_GAUSS_POINTS)

            # Multiply I_matrices by probabilities and weights element-wise
            # I_matrices shape:       (NUM_GAUSS_POINTS, NUM_GAUSS_POINTS, M, M)
            # weights_combined shape: (NUM_GAUSS_POINTS, NUM_GAUSS_POINTS)
            # p_values shape:         (NUM_GAUSS_POINTS, NUM_GAUSS_POINTS)
            weighted_I = I_matrices * weights_combined[:, :, np.newaxis, np.newaxis] * p_values[:, :, np.newaxis, np.newaxis]

            # Sum over the quadrature points (first two dimensions)
            FI = np.sum(weighted_I, axis=(0, 1))  # Shape: (M, M)
            
            # -----------------------------------------------------------------------------
            
            # Return the determinant of the Fisher Information matrix
            det = np.linalg.det(FI) * HTA_SCALE
            # Clip to 0, 10
            det = np.clip(det, 0, FINAL_FI_CLIP)
            
            return det

        # Check if the previous phi has an obs_list attribute
        if hasattr(self.basis.phi, 'obs_list'):
            # Save it, we will need it to modify the phi function later
            obs_list = self.basis.phi.obs_list.copy()
        else:
            # TODO: Check if this can ever happen
            raise ValueError("The previous phi function does not have an obs_list attribute.")

        # Attach the phi function to the agent
        self.basis.phi = phi

        # Handle obstacle exclusion from the function above
        self.basis.phi = self.modifedPhiForObstacles(self.basis.phi, obs_list=obs_list)

        # Precalculate the phi coefficients
        self.basis.precalcAllPhiK() 



    def withinBounds(self, x):
        '''
        Check if the state is within the bounds of the system
        '''
        # Check if the 2 first ergodic dimension are within the bounds L1, L2
        if x[0] < 0 or x[0] > self.L1 or x[1] < 0 or x[1] > self.L2:
            print(f"--> ATTENTION: State out of bounds: {x}")

        # Check if model is quadcopter
        if isinstance(self.model, Quadcopter):
            # Check if the 3rd dimension is within the bounds
            z = self.model.state[2]
            if z < 0 or z > self.model.z_target * 20:
                print(f"--> Quad is getting out of hand in the Z dim: {z:.2f} m")
