# Ergodic Library
from my_erg_lib.basis import Basis
import numpy as np
from my_erg_lib.model_dynamics import SingleIntegrator, DoubleIntegrator, Quadcopter, SimpleBoatSecondOrder, SimpleCarSecondOrder
from my_erg_lib.eid import Sensor, EKF
import my_erg_lib.Utilities as utils
import time
from my_erg_lib.obstacles import Obstacle, saveObstaclesToMemory, removeObstaclesFromMemory, updateObstaclePositionInMemory

# ROS Library
from rclpy.node import Node
from my_interfaces.msg import CkTable, AgentData, SingleObstacle, MultipleObstacles, SingleTargetEstimate, MultipleTargetEstimates
import re

class Agent(Node):
    def __init__(self, L1_BOUNDS, L2_BOUNDS, Kmax, dynamics_model, phi=None, x0=None, agent_id=None, 
                 antenna_rad=np.inf, antenna_range_flag=False, same_l_bounds_flag=True, real_target_positions=None, ekf_params=None,
                 KAPPA_OBS_VIRTUAL=1, RHO_OBS_VIRTUAL=0.75):
        
        self.agent_id = agent_id if agent_id is not None else 0
        self.time_since_start = 0.0

        super().__init__(f"agent_{self.agent_id}")  # ROS node with name "agent_{id}"
        
        # Space Parameters
        # assert L_BOUNDS is a list of 2 elements [min, max]
        assert isinstance(L1_BOUNDS, (list, tuple)) and len(L1_BOUNDS) == 2, "L1_BOUNDS must be a list or tuple of 2 elements [min, max]."
        assert isinstance(L2_BOUNDS, (list, tuple)) and len(L2_BOUNDS) == 2, "L2_BOUNDS must be a list or tuple of 2 elements [min, max]."
        self.L1_min = L1_BOUNDS[0]
        self.L1_max = L1_BOUNDS[1]
        self.L2_min = L2_BOUNDS[0]
        self.L2_max = L2_BOUNDS[1]
        assert self.L1_min < self.L1_max, f"L1_BOUNDS are not valid. Lower bound must be less than upper bound. ({self.L1_min} <=? {self.L1_max})"
        assert self.L2_min < self.L2_max, f"L2_BOUNDS are not valid. Lower bound must be less than upper bound. ({self.L2_min} <=? {self.L2_max})"
        self.L1_size = self.L1_max - self.L1_min
        self.L2_size = self.L2_max - self.L2_min

        self.Kmax = Kmax
        self.antenna_rad = antenna_rad      # Antenna radius in meters (for CK sharing among agents)
        self.antenna_range_flag = antenna_range_flag
        self.same_l_bounds_flag = same_l_bounds_flag  # Whether to communicate only with agents having same L bounds [Default: True]
        self.limits_changed_flag = False    # We have the option to start with lower u_bounds. After a while change them to higher. This flags states whether the change has been made yet or not
        
        # Connecting model dynamics
        self.model = dynamics_model
        # Make sure x0 is within the space limits L1_min->L1_max, L2_min->L2_max if given
        if x0 is not None:
            assert len(x0) == self.model.num_of_states, "Initial position x0 must have the same length as the model's state vector."
            assert self.L1_min <= x0[0] <= self.L1_max and self.L2_min <= x0[1] <= self.L2_max, "Initial position x0 must be within the limits of the space." # TODO: Assuming x[0], x[1] are the x, y coordinates
            self.model.reset(x0)

        # Real target position (Ground Truth)
        if real_target_positions is not None:
            # Assert it contains x, y, z positions for as many targets
            assert all(len(pos) == 3 for pos in real_target_positions), "Each real target position must be a 3D coordinate (x, y, z)."
            self.real_target_positions = np.asarray(real_target_positions)

        # TODO: Fix comparing real targ pos to agents pos for different dynamics models
        self.num_of_targets = 0     # Number of target estimates so far
        self.target_estimates = []

        default_ekf_params = {
            # New target estimate EKF parameters
            'sigma_init': np.eye(3)*5e-1,      # 1e-1 may be more appropriate
            'R': np.diag([0.1, 0.1]),          # Sensor noise covariance
            'Q': np.eye(3) * 1e-4,             # Process noise covariance (1e-5)
            'a_limits': [[self.L1_min, self.L1_max], [self.L2_min, self.L2_max], [0, 10]],
            # Sensor Parameters
            'sensor_range': 2,                  # Sensor range in [m]
            'sensor_R': np.diag([0.01, 0.01])   # Sensor noise covariance
        }
        if ekf_params is not None:
            # Merge user params with defaults (user params override defaults)
            self.ekf_params = {**default_ekf_params, **ekf_params}
        else:
            self.ekf_params = default_ekf_params

        # Multiple EKF instances - one per target estimate
        self.ekfs = []

        # Lets connect a sensor to track the target position 
        self.sensor = Sensor(sensor_range=self.ekf_params['sensor_range'],
                             R=self.ekf_params['sensor_R'])

        # Initialise obstacle list
        self.obstacle_list = []
        # Parameters for virtual obstacles (other agents etc)
        self.KAPPA_OBS_VIRTUAL = KAPPA_OBS_VIRTUAL
        self.RHO_OBS_VIRTUAL = RHO_OBS_VIRTUAL

        # Initialize the basis object
        self.basis = Basis(L1_BOUNDS, L2_BOUNDS, Kmax, phi_=phi, precalc_phik_coeff=False, num_gauss_points=22)
        # TODO: Make a modular length basis ergodic memory, so that it can be changed later on

        # Create ROS publishers and subscribers
        self.ck_publisher = self.create_publisher(CkTable, f'agent_{self.agent_id}/ck', 10)
        self.data_publisher = self.create_publisher(AgentData, f'agent_{self.agent_id}/data', 10)
        self.target_est_publisher = self.create_publisher(MultipleTargetEstimates, f'agent_{self.agent_id}/target_estimates', 10)
        self.known_obst_publisher = self.create_publisher(MultipleObstacles, f'agent_{self.agent_id}/known_obstacles', 10)
        # Timer to periodically publish data
        self.publish_timer = self.create_timer(0.5, self.publishTargetAndObstacleData)

        # Agents in the same network
        self.discovered_agents = set()
        self.in_range_agents = set()  # Agents within antenna range for CK calculations
        self.ck_subscribers = {}  # Dict to store subscribers for other agents
        self.agent_ck_data = {}  # Dict to store CK data from other agents {agent_id: ck_table}
        self.agent_positions = {}  # Dict to store positions of other agents {agent_id: (x, y)}
        self.agent_model_types = {}  # Dict to store model types of other agents {agent_id: model_type}
        self.agent_l_bounds = {}  # Dict to store L bounds of other agents {agent_id: [x_min, x_max, y_min, y_max]}
        self.talk_alike_flag = False # Whether to communicate only with similar model (boats with boats, cars with cars, etc)
        
        # Timer for periodic agent discovery (every second)
        self.discovery_timer = self.create_timer(1.0, self.discoverAgentsInROS)


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

    def updateEIDphiFunction(self, NUM_GAUSS_POINTS=10, P_UPPER_LIM=10, HTA_SCALE=5e-3, FINAL_FI_CLIP=10, ALWAYS_ADD=0):
        # Check to see if there are any targets to estimate, otherwise we dont need this
        if self.num_of_targets == 0:
            return

        # -----------------------------------------------------------------------------

        # Expected Information Matrix (Φ)
        # We will have to integrate the product of I(a) * p(a)
        # Get Gauss-Legendre quadrature points and weights
        a1_points, a1_weights = np.polynomial.legendre.leggauss(NUM_GAUSS_POINTS)
        a2_points, a2_weights = np.polynomial.legendre.leggauss(NUM_GAUSS_POINTS)
        # Transform from [-1,1] to [L1_min, L1_max] and [L2_min, L2_max]
        a1_points  = 0.5 * self.L1_size * (a1_points + 1) + self.L1_min
        a2_points  = 0.5 * self.L2_size * (a2_points + 1) + self.L2_min
        a1_weights = 0.5 * self.L1_size * a1_weights
        a2_weights = 0.5 * self.L2_size * a2_weights

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
        # -----------------------------------------------------------------------------
        
        # Precompute probability values
        # Here we can incorporate more target estimates than one
        p_values = np.zeros((NUM_GAUSS_POINTS, NUM_GAUSS_POINTS))
        for ekf in self.ekfs:
            # Calculate all probabilities at once # TODO: How is the 3rd dimension's computation affected with 0ing it out here?
            all_probs = ekf.p(np.array([[a1, a2, 0] for a1 in a1_points for a2 in a2_points]), upper_lim_to_normalise=P_UPPER_LIM) # Could use P_UPPER_LIM to make them all be of equal height
            # Reshape the results back to a grid
            p_values += all_probs.reshape(NUM_GAUSS_POINTS, NUM_GAUSS_POINTS)

        # Normalize the probabilities
        p_values *= P_UPPER_LIM / np.max(p_values) 

        def phi(x):
            # Append the self.model.state[2] to the state vector
            x = np.append(x, self.model.state[2])
            M  = self.sensor.measurement_model.M   # Number of target estimation states

            # -----------------------------------------------------------------------------

            # Vectorized H computation (Assuming they all have the same measurement model)
            H_all = self.sensor.measurement_model.H(a_positions, x)  # Shape: (NUM_GAUSS_POINTS^2, 2, 3)
            
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
            res = np.linalg.det(FI) * HTA_SCALE + ALWAYS_ADD
            # Clip to not exceed a predefined value (usually for stability reasons)
            res = np.clip(res, 0, FINAL_FI_CLIP)
            
            return res

        # Check if the previous phi has an obs_list attribute
        if hasattr(self.basis.phi, 'obs_list'):
            # Save it, we will need it to modify the phi function later
            obs_list = self.basis.phi.obs_list.copy()
        else:
            # Although i dont think this can happen, will leave it here for agility
            raise ValueError("The previous phi function does not have an obs_list attribute.")

        # Attach the phi function to the agent
        self.basis.phi = phi

        # Handle obstacle exclusion from the function above
        self.basis.phi = self.modifedPhiForObstacles(self.basis.phi, obs_list=obs_list)

        # Precalculate the phi coefficients
        self.basis.precalcAllPhiK() 

    def spawnNewTargetEstimate(self, measurement=None, init_position=None, current_time=time.time()):
        """
        Spawn a new target based on the measurement. The initial position will be in the direction of the measurement.
        This will create a new EKF instance for the new target and add it to the list of targets
        """
        if measurement is not None:
            beta = measurement[0]  # Azimuth angle in radians
            elev = measurement[1]  # Elevation angle in radians
            # Calculate the new target position based on the measurement
            a_init = np.array([
                self.model.state[0] + self.sensor.sensor_range/2 * np.sin(beta),  # x position
                self.model.state[1] + self.sensor.sensor_range/2 * np.cos(beta),  # y position
                0                                                                 # z position (assuming flat ground)
            ])
            # Clip to L1_min->L1_max, L2_min->L2_max
            a_init[0] = np.clip(a_init[0], self.L1_min, self.L1_max)
            a_init[1] = np.clip(a_init[1], self.L2_min, self.L2_max)
        elif init_position is not None:
            a_init = np.asarray(init_position)
        else:
            raise ValueError("Either 'measurement' or 'a_init' must be provided to spawn a new target.")
        
        ekf_ = EKF(ekf_id=self.ekfs[-1].id + 1 if self.ekfs != [] else 0,  # Increment ID if exists, else start from 0
                   a_init=a_init,
                   sigma_init=self.ekf_params['sigma_init'],     # 1e-1 may be more appropriate
                   R=self.ekf_params['R'],                       # Sensor noise covariance
                   Q=self.ekf_params['Q'],                       # Process noise covariance (1e-5)
                   a_limits=self.ekf_params['a_limits'],
                   time_now=current_time)
        
        self.ekfs.append(ekf_)
        self.target_estimates.append(a_init)  # Add the initial estimate to the list of target estimates
        self.num_of_targets += 1  # Increment the number of targets

    def _mergeTargets(self, ekf_ids):
        """
        Merge multiple targets by averaging their estimates and covariance matrices.
        This will remove all but the first target from the list of targets.
        
        Args:
            *ekf_ids: Variable number of EKF IDs to merge
        """
        if len(ekf_ids) < 2:
            raise ValueError("At least two EKF IDs must be provided for merging.")
        
        # Find indices corresponding to the EKF IDs
        target_indices = []
        for ekf_id in ekf_ids:
            idx = None
            for i, ekf in enumerate(self.ekfs):
                if ekf.id == ekf_id:
                    idx = i
                    break
            if idx is None:
                raise ValueError(f"EKF with ID {ekf_id} not found in the list of current ekfs.")
            target_indices.append(idx)
        
        # Sort indices in descending order to avoid index shifting when deleting
        sorted_indices = sorted(target_indices, reverse=True)
        first_idx = sorted_indices[-1]  # The smallest index (will be kept)
        
        # Calculate weighted average of estimates
        estimates_to_merge = [self.target_estimates[idx] for idx in target_indices]
        new_estimate = np.mean(estimates_to_merge, axis=0)
        
        # Calculate "average" of covariance matrices using Log-Euclidean mean (# TODO: Can this be done at all?)
        sigmas_to_merge = [self.ekfs[idx].sigma_k_1 for idx in target_indices]
        new_sigma = utils.logEuclideanMean(sigmas_to_merge)
        
        # Update the first target's estimate and covariance
        self.target_estimates[first_idx] = new_estimate
        self.ekfs[first_idx].sigma_k_1 = new_sigma
        self.ekfs[first_idx].last_time_updated = np.max([self.ekfs[idx].last_time_updated for idx in target_indices])  # Update the last updated time to the earliest one
        
        # Remove the other targets (in descending order to avoid index shifting)
        for idx in sorted_indices[:-1]:  # Skip the last one (smallest index)
            del self.target_estimates[idx]
            del self.ekfs[idx]
        

        print(f"--> Merging targets with IDs {ekf_ids}. New estimate: {new_estimate}, New covariance: {new_sigma}")
        self.num_of_targets -= (len(ekf_ids) - 1)

    def mergeTargetsIfNeeded(self, MERGE_THRESHOLD=2.0, EUCL_DIST_THRESHOLD=0.2, SIMILAR_MEASUREMENTS_ANGLE_THRESHOLD_RAD=30*np.pi/180):
        """
        Check if any targets should be merged based on Bhattacharyya distance between their distributions
        """
        if len(self.ekfs) < 2:
            return
        
        def bhattacharyyaDistance(mu1, sigma1, mu2, sigma2):
            """
            Calculate Bhattacharyya distance between two multivariate Gaussian distributions
            """
            # Mean difference
            delta_mu = mu1 - mu2
            
            # Average covariance matrix
            sigma_avg = 0.5 * (sigma1 + sigma2)
            
            try:
                # First term: quadratic form with average covariance
                sigma_avg_inv = np.linalg.inv(sigma_avg)
                term1 = 0.125 * delta_mu.T @ sigma_avg_inv @ delta_mu
                
                # Second term: determinant ratio
                det_avg = np.linalg.det(sigma_avg)
                det_product = np.linalg.det(sigma1) * np.linalg.det(sigma2)
                term2 = 0.5 * np.log(det_avg / np.sqrt(det_product))
                
                return term1 + term2
            except np.linalg.LinAlgError:
                # If matrices are singular, return infinite distance
                return np.inf
        
        targets_to_merge = []
        
        # Check all pairs of targets
        for i in range(len(self.ekfs)):
            for j in range(i + 1, len(self.ekfs)):
                mu1 = self.target_estimates[i]
                sigma1 = self.ekfs[i].sigma_k_1
                mu2 = self.target_estimates[j] 
                sigma2 = self.ekfs[j].sigma_k_1
                
                bh_distance = bhattacharyyaDistance(mu1, sigma1, mu2, sigma2)
                eucl_dist = np.linalg.norm(mu1 - mu2)
                # print(f"D between {self.ekfs[i].id} and {self.ekfs[j].id}: {bh_distance:.3f} \t Eucl Dist: {eucl_dist:.3f}")
                estimate_centers_distance = np.linalg.norm(mu1[:2] - mu2[:2])  # Only consider x, y positions
                
                if bh_distance < MERGE_THRESHOLD and estimate_centers_distance < EUCL_DIST_THRESHOLD:
                    # Estimate Measurements for these 2 targets
                    meas1 = self.sensor.measurement_model.Y(self.target_estimates[i], self.model.state[:3])
                    meas2 = self.sensor.measurement_model.Y(self.target_estimates[j], self.model.state[:3])
                    # Are the 2 estimated measurents close enought to at least 2 measurments in agent.measurements_raw??
                    # This way we check the existence of 2 targets close by in the real world
                    how_many_are_close = 0
                    for est_m in [meas1, meas2]:
                        for m in self.sensor.measurements_raw:
                            # Check if angle_diff < 30 deg (e.g)
                            angle_diff = np.abs((est_m[0] - m[0] + np.pi) % (2 * np.pi) - np.pi) # Normalise angle diff to [-pi, pi]
                            if m is not None and (angle_diff < SIMILAR_MEASUREMENTS_ANGLE_THRESHOLD_RAD):  # TODO: Check this threshold
                                how_many_are_close += 1

                    # I need the 2 estimates pairing with at least 2 real measurements each => 2x2=4+
                    if how_many_are_close < 4: 
                        targets_to_merge.append([self.ekfs[i].id, self.ekfs[j].id])
                        # print(f"Targets {self.ekfs[i].id} and {self.ekfs[j].id} should be merged (Bhattacharyya distance: {bh_distance:.3f} \t Eucl Dist: {estimate_centers_distance:.3f})")
        
        # Check if an ekf_id is more than once in the targets_to_merge list. Merge every pair. For example, if we have targets_to_merge = [[1, 2], [2, 3], [4, 5]], we should merge [[1, 2, 3], [4, 5]]
        targets_to_merge = utils.mergeOverlappingPairsAllTheWay(targets_to_merge)
        
        # Merge the targets
        for ekf_ids in targets_to_merge:
            self._mergeTargets(ekf_ids)

    def associateTargetsWithMahalanobis(self, measurements, agent_position, ASSOCIATION_THRESHOLD=5):
        """
        Associate measurements using Mahalanobis distance
        Returns a list of associated measurements for each target estimate.
        Example:
            7 Targets, 3 Measurements
            - measurements = [np.array([0.8, 0.4]), np.array([0.3, 0.7]), np.array([0.5, 0.2])]
            - associated_measurements = [np.array([0.8, 0.4]), None, np.array([0.5, 0.2]), None, None, None, None]
            The index in the associated_measurements corresponds to the target index in the self.target_estimates list.
            Although 3 meas where given, only 2 were associated with targets. This can be due to low association threshold (measurement could not be matched to existing targets)

        """
        associated_measurements = [None] * len(self.target_estimates)
        used_measurements = set()
        # Get rid of None measurements
        measurements = [m for m in measurements if m is not None]
        if len(measurements) == 0:
            return associated_measurements  # No measurements to associate
        
        # Calculate association costs (Mahalanobis distances)
        association_matrix = np.full((len(self.target_estimates), len(measurements)), np.inf)   # Array [num_targets x num_measurements] -> np.inf
        
        for target_idx, (target_estimate, ekf) in enumerate(zip(self.target_estimates, self.ekfs)):
            # Predicted measurement for this target
            predicted_z = self.sensor.measurement_model.Y(target_estimate, agent_position)
            
            # Innovation covariance S = H*P*H' + R
            H = self.sensor.measurement_model.H(target_estimate, agent_position)
            S = H @ ekf.sigma_k_1 @ H.T + self.sensor.R
            S_inv = np.linalg.solve(S, np.eye(S.shape[0]))
            
            for meas_idx, measurement in enumerate(measurements):
                # Innovation (measurement residual)
                diff = (measurement - predicted_z + np.pi) % (2 * np.pi) - np.pi  # Normalise angle diff to [-pi, pi] # TODO: Measurements need to be angles, its not modular enough
                
                # Mahalanobis distance
                mahal_dist = np.sqrt(diff.T @ S_inv @ diff)
                association_matrix[target_idx, meas_idx] = mahal_dist
        
        # # For simplicity, we use a greedy approach to associate targets with measurements
        # for target_idx in range(len(self.target_estimates)):
        #     best_measurement_idx = None
        #     min_distance = np.inf
            
        #     for meas_idx in range(len(measurements)):
        #         if meas_idx in used_measurements :
        #             continue
                

        #         distance = association_matrix[target_idx, meas_idx]
        #         if distance < min_distance and distance < ASSOCIATION_THRESHOLD:
        #             # # If distance from agent to specific target is more than sensing radius, and the estimated angle more than enough, skip it
        #             # xy_dist = np.linalg.norm(self.model.state[:2] - self.target_estimates[target_idx][:2])
        #             # beta, _ = self.sensor.measurement_model.Y(self.target_estimates[target_idx], self.model.state[:3])
        #             # delta_angle = np.abs(beta - measurements[meas_idx][0])  # Angle difference in radians

        #             # if xy_dist > self.sensor.sensor_range and delta_angle > 15 * np.pi / 180: ! TODO: Handle this better
        #             #     continue    ! TODO: If target is inside, estimate is barely outside but in the same line, its wrong
        #             #     # pass

        #             min_distance = distance
        #             best_measurement_idx = meas_idx
            
        #     if best_measurement_idx is not None:
        #         associated_measurements[target_idx] = measurements[best_measurement_idx]
        #         used_measurements.add(best_measurement_idx)
        # For simplicity, we use a greedy approach to associate targets with measurements

        # Find the best associations globally instead of target-by-target
        remaining_targets = set(range(len(self.target_estimates)))
        remaining_measurements = set(range(len(measurements)))

        while remaining_targets and remaining_measurements:
            # Find the global minimum distance among remaining targets and measurements
            best_target_idx = None
            best_measurement_idx = None
            min_distance = np.inf
            
            for target_idx in remaining_targets:
                for meas_idx in remaining_measurements:
                    distance = association_matrix[target_idx, meas_idx]
                    if distance < min_distance and distance < ASSOCIATION_THRESHOLD:
                        min_distance = distance
                        best_target_idx = target_idx
                        best_measurement_idx = meas_idx
            
            # If we found a valid association, make it
            if best_target_idx is not None and best_measurement_idx is not None:
                associated_measurements[best_target_idx] = measurements[best_measurement_idx]
                remaining_targets.remove(best_target_idx)
                remaining_measurements.remove(best_measurement_idx)
            else:
                # No more valid associations possible
                break

        # Debug Printing ---------------
        # print(f"\n--> Association Matrix:\n{association_matrix}")
        # # print the same matrix but zero out the values that have not been associated
        # association_matrix_display = np.zeros_like(association_matrix)
        # for target_idx in range(len(self.target_estimates)):
        #     for meas_idx in range(len(measurements)):
        #         if (associated_measurements[target_idx] is not None and 
        #             np.array_equal(associated_measurements[target_idx], measurements[meas_idx])):
        #             association_matrix_display[target_idx, meas_idx] = association_matrix[target_idx, meas_idx]
        # print(f"--> Association Matrix (with unassociated values zeroed out):\n{association_matrix_display}")
        # # print measurements in deg
        # measurements_deg = np.array([np.degrees(meas[0]) for meas in measurements])
        # print(f"--> Measurements (in degrees): \n{measurements_deg}\n")


        # import vis
        # vis.plotMeasurementsAndTargets(self, measurements, associated_measurements, fig_num=1,
        #                                save_fig_filename=None)
                                    #    save_fig_filename="images/measurementsEKF")

        return associated_measurements

    def _removeTargetEstimate(self, ekf_id):
        """
        Remove a target from the list of targets by its EKF ID
        """
        for i, ekf in enumerate(self.ekfs):
            if ekf.id == ekf_id:
                del self.ekfs[i]
                del self.target_estimates[i]
                self.num_of_targets -= 1
                return
        
        print(f"--> No target found with EKF ID {ekf_id}.")

    def searchAndRemoveOldTargetEstimates(self, current_time=time.time(), MAX_AGE_SEC=60):
        """
        Search for old target estimates and remove them if they are older than MAX_AGE_SEC
        """
        for ekf in self.ekfs:
            age = current_time - ekf.last_time_updated
            if age > MAX_AGE_SEC:
                ekf_id = ekf.id
                self._removeTargetEstimate(ekf_id)
                print(f"--> Removed target with EKF ID {ekf_id} ({age:.1f} [s] old).")


    def withinBounds(self, x):
        '''
        Check if the state is within the bounds of the system
        '''
        # Check if the 2 first ergodic dimension are within the bounds L1, L2
        if not (self.L1_min <= x[0] <= self.L1_max and self.L2_min <= x[1] <= self.L2_max):
            print(f"--> ATTENTION: State out of bounds: {x}")

        # Check if model is quadcopter
        if isinstance(self.model, Quadcopter):
            # Check if the 3rd dimension is within the bounds
            z = self.model.state[2]
            if z < 0 or z > self.model.z_target * 20:
                print(f"--> Quad is getting out of hand in the Z dim: {z:.2f} m")


    # ======= Potential Function Calculations =======
    def calcPotentialU(self, x):
        """
        Calculate the potential function U at a given state x.
        This function sums the potential contributions from all obstacles in the obstacle list.
        """
        U = 0
        for obs in self.obstacle_list:
            U += obs.U(x[:2])
        return U

    # def calcPotentialUGradient(self, x):
    #     """
    #     Calculate the gradient of the potential function U at a given state x.
    #     """
    #     grad_U = np.zeros(2)
    #     for obs in self.obstacle_list:
    #         grad_U += obs.gradU(x[:2])

    #     return grad_U
    
    def calcPotentialUAndGradient(self, x):
        """
        Calculate both the potential function U and its gradient at a given state x.
        This is more efficient than calling calcPotentialU and calcPotentialUGradient separately,
        as it computes rho only once per obstacle.
        """
        U = 0
        grad_U = np.zeros(2)
        for obs in self.obstacle_list:
            U_obs, grad_U_obs = obs.UandGradU(x[:2])
            U += U_obs
            grad_U += grad_U_obs
        return U, grad_U

    def calcH(self, x, delta=0.0, u_value_precomputed=None):
        """
        Calculate h(x), the CBF (Control Barrier Function) value at a given state x.
        This function is used to ensure safety constraints are satisfied.
        """
        U = self.calcPotentialU(x) if u_value_precomputed is None else u_value_precomputed
        h = 1 / (1 + U) - delta
        return h

    def calcHGradient(self, x, also_return_h_flag=False):
        """
        Calculate the gradient of h(x) at a given state x.
        This function is used to ensure safety constraints are satisfied.
        ATTENTION: Returns 2x1 vector, only for positional dimensions. Need to append accordingly in order to multiply by f(x) later on.
        """
        U, grad_U = self.calcPotentialUAndGradient(x)
        h_grad = -grad_U / (1 + U)**2

        # Append zeros for the other dimensions if needed (e.g., for quadcopter)
        if self.model.num_of_states > 2:
            h_grad = np.append(h_grad, np.zeros(self.model.num_of_states - 2))

        if also_return_h_flag:
            h = self.calcH(x, u_value_precomputed=U)
            return h, h_grad
        else:
            return h_grad

    def calcHessianH(self, x, epsilon=1e-3):
        # Lets use finite differences to calculate the Hessian of h(x)
        hessian_h = np.zeros((self.model.num_of_states, self.model.num_of_states))

        for i in range(self.model.num_of_states):
            for j in range(i, self.model.num_of_states):  # compute for j >= i to exploit symmetry
                if i == j:
                    x_plus = x.copy()
                    x_plus[i] += epsilon
                    h_plus = self.calcH(x_plus)

                    x_minus = x.copy()
                    x_minus[i] -= epsilon
                    h_minus = self.calcH(x_minus)

                    hessian_h[i, i] = (h_plus - 2 * self.calcH(x) + h_minus) / (epsilon ** 2)
                else:
                    x_pp = x.copy()
                    x_pp[i] += epsilon
                    x_pp[j] += epsilon
                    h_pp = self.calcH(x_pp)

                    x_pm = x.copy()
                    x_pm[i] += epsilon
                    x_pm[j] -= epsilon
                    h_pm = self.calcH(x_pm)

                    x_mp = x.copy()
                    x_mp[i] -= epsilon
                    x_mp[j] += epsilon
                    h_mp = self.calcH(x_mp)

                    x_mm = x.copy()
                    x_mm[i] -= epsilon
                    x_mm[j] -= epsilon
                    h_mm = self.calcH(x_mm)

                    hessian_value = (h_pp - h_pm - h_mp + h_mm) / (4 * epsilon ** 2)
                    hessian_h[i, j] = hessian_value
                    hessian_h[j, i] = hessian_value  # symmetry


        return hessian_h

    def calcUsafe(self, x, udef_now, alpha_1=1.0, alpha_2=1.0, delta=0.0):

        # Calculate CBF function h(x) and ∇(x)
        h, grad_h = self.calcHGradient(x[:2], also_return_h_flag=True)
        # Calculate CBF Hessian
        hess_h = np.zeros((self.model.num_of_states, self.model.num_of_states))
        # hess_h = self.calcHessianH(x, epsilon=1e-4)

        # System Dynamics
        f = self.model.f(x, udef_now)
        f_x = self.model.f_x(x, udef_now)
        g = self.model.h(x)

        h_dot = grad_h.T @ f    
        h_ddot = f.T @ hess_h @ f + grad_h.T @ f_x @ f

        PSI = h_ddot + 2 * alpha_1 * h_dot + alpha_2 * h
        beta = (f.T @ hess_h + grad_h.T @ f_x) @ g

        if PSI >= 0:
            # No need to change the control input if PSI > 0, we are not in a danger zone
            u_safe = np.zeros_like(udef_now)      
        else:
            if np.linalg.norm(beta) < 1e-6:
                u_safe = np.zeros_like(udef_now)
            else:
                # If we have a boat, we need to prioritize rudder over thrust (to avoid using reverse thrust)
                if isinstance(self.model, SimpleBoatSecondOrder):
                    # Create weighting matrix to prioritize rudder over thrust
                    # Assume control input order is [thrust, rudder]
                    W = np.diag([1.0, self.model.rudder_priority])  # Higher weight on rudder
                    
                    # Weighted least squares solution for safety control
                    beta_weighted = W @ beta.T
                    u_safe_weighted = -beta_weighted / (np.linalg.norm(beta_weighted)**2) * PSI
                    
                    # Transform back to original control space
                    u_safe = W @ u_safe_weighted
                    
                    # Additional constraint: don't allow thrust to go above maximum (less negative = less forward thrust)
                    if len(u_safe) >= 1:  # Ensure we have thrust control
                        # If the safety control would make total thrust positive (reverse), redistribute to rudder
                        total_thrust = udef_now[0] + u_safe[0]
                        
                        if total_thrust > self.model.max_allowed_rev_thr:  # total_thrust becoming positive means reverse
                            thrust_excess = total_thrust - self.model.max_allowed_rev_thr
                            u_safe[0] = self.model.max_allowed_rev_thr - udef_now[0]  # Adjust thrust to maximum allowed (0 or negative)

                            # If we have rudder control, increase rudder authority to compensate
                            if len(u_safe) >= 2:
                                # Choose redistribution sign robustly:
                                if np.abs(beta[1]) > 1e-5:
                                    sign_steer = np.sign(beta[1])
                                else:
                                    # Fallback: use cross product of velocity and grad_h to pick turning side
                                    v = f[:2]        # translational velocity (x,y)
                                    g2 = grad_h[:2]  # gradient in x,y
                                    cross = v[0]*g2[1] - v[1]*g2[0]
                                    # If cross>0 -> turning in one direction, cross<0 -> the other
                                    sign_steer = np.sign(cross) if np.abs(cross) > 1e-6 else 1.0

                                additional_steer = -thrust_excess * sign_steer * self.model.rudder_priority
                                u_safe[1] += additional_steer
                                
                elif isinstance(self.model, SimpleCarSecondOrder):
                    # Create weighting matrix to prioritize steering over thrust
                    # Assume control input order is [thrust, steering]
                    W = np.diag([1.0, self.model.steer_priority])  # Higher weight on steering
                    
                    # Weighted least squares solution for safety control
                    beta_weighted = W @ beta.T
                    u_safe_weighted = -beta_weighted / (np.linalg.norm(beta_weighted)**2) * PSI
                    
                    # Transform back to original control space
                    u_safe = W @ u_safe_weighted

                    # print(f"--> Car Safety Control: PSI={PSI:.4f}, beta={beta}, u_safe_initial={u_safe}")

                    # Additional constraint: don't allow thrust to go above maximum (less negative = less forward thrust)
                    if len(u_safe) >= 1:  # Ensure we have thrust control
                        # If the safety control would make total thrust positive (reverse), redistribute to rudder
                        total_thrust = udef_now[0] + u_safe[0]
                        
                        if total_thrust > self.model.max_allowed_rev_thr:  # total_thrust becoming positive means reverse
                            thrust_excess = total_thrust - self.model.max_allowed_rev_thr
                            u_safe[0] = self.model.max_allowed_rev_thr - udef_now[0]  # Adjust thrust to maximum allowed (0 or negative)

                            # If we have rudder control, increase rudder authority to compensate
                            if len(u_safe) >= 2:
                                # Choose redistribution sign robustly:
                                if np.abs(beta[1]) > 1e-5:
                                    sign_steer = np.sign(beta[1])
                                else:
                                    # Fallback: use cross product of velocity and grad_h to pick turning side
                                    # use cross product between velocity vector and gradient to pick side
                                    v = f[:2]         # translational velocity (x,y) -> car: (u*cos, u*sin)
                                    g2 = grad_h[:2]
                                    cross = v[0]*g2[1] - v[1]*g2[0]
                                    sign_steer = np.sign(cross) if np.abs(cross) > 1e-6 else np.sign(beta[1]) if np.abs(beta[1])>1e-6 else 1.0

                                additional_steer = -thrust_excess * sign_steer * self.model.steer_priority
                                u_safe[1] += additional_steer

                # TODO: Quad relative degree is more than 2 for roll, pitch, yaw. So cbf returns actions for only throttle (due to our linearization)
                # elif isinstance(self.model, Quadcopter):

                #     # input order assumed: [thrust, yaw, pitch, roll]
                #     m = self.model.m
                #     hover_thrust = m * 9.81

                #     # weights: make thrust very expensive to change; attitude and yaw cheap(er)
                #     w_thrust = 100.0       # very large -> penalize thrust changes
                #     w_yaw   = 0.1
                #     w_pitch = 0.1
                #     w_roll  = 0.1
                #     W = np.diag([w_thrust, w_yaw, w_pitch, w_roll])
                #     Winv = np.linalg.inv(W)

                #     beta_vec = np.asarray(beta).reshape(-1)   # shape (4,)
                #     if np.linalg.norm(beta_vec) < 1e-9:
                #         print(f"[Quadcopter CBF] beta_vec norm too small ({np.linalg.norm(beta_vec):.2e}), setting u_safe = 0")
                #         u_safe = np.zeros_like(udef_now)
                #     else:
                #         denom = beta_vec @ (Winv @ beta_vec)
                #         print(f"[Quadcopter CBF] denom: {denom:.4e}, PSI: {PSI:.4e}")
                #         if np.abs(denom) < 1e-12:
                #             print(f"[Quadcopter CBF] denom too small ({denom:.2e}), setting u_safe = 0")
                #             u_safe = np.zeros_like(udef_now)
                #         else:
                #             # weighted-LS closed form
                #             u_delta = - (Winv @ beta_vec) * (PSI / denom) * 10 ** -4 * 3  # proposed delta to nominal
                #             print(f"[Quadcopter CBF] u_delta (proposed): {u_delta}")

                #             # enforce a minimum allowed total thrust (so we don't "dive through")
                #             min_thrust_fraction = 0.95   # tune: 0.6..0.95 depending on safety desired
                #             min_allowed_total_thrust = min_thrust_fraction * hover_thrust
                #             proposed_total_thrust = udef_now[0] + u_delta[0]
                #             print(f"[Quadcopter CBF] proposed_total_thrust: {proposed_total_thrust:.4f}, min_allowed_total_thrust: {min_allowed_total_thrust:.4f}")

                #             if proposed_total_thrust < min_allowed_total_thrust:
                #                 print(f"[Quadcopter CBF] proposed_total_thrust below minimum, fixing thrust to {min_allowed_total_thrust:.4f}")
                #                 # fix thrust delta to reach the minimum allowed total thrust
                #                 u_delta_thrust_fixed = min_allowed_total_thrust - udef_now[0]
                #                 # compute remaining RHS after accounting for the fixed thrust contribution
                #                 b0 = beta_vec[0] * u_delta_thrust_fixed * 10
                #                 residual_rhs = -PSI - b0

                #                 # redistribute residual among attitude channels (indices 1..3)
                #                 beta_others = beta_vec[1:]
                #                 W_others = W[1:, 1:]
                #                 try:
                #                     Winv_others = np.linalg.inv(W_others)
                #                 except np.linalg.LinAlgError:
                #                     print("[Quadcopter CBF] Winv_others not invertible, using thrust only")
                #                     # fallback: cannot invert, return with clipped thrust only
                #                     u_safe = np.array([u_delta_thrust_fixed, 0.0, 0.0, 0.0])
                #                 else:
                #                     denom2 = beta_others @ (Winv_others @ beta_others)
                #                     print(f"[Quadcopter CBF] denom2: {denom2:.4e}, residual_rhs: {residual_rhs:.4e}")
                #                     if np.abs(denom2) < 1e-12:
                #                         print("[Quadcopter CBF] denom2 too small, using thrust only")
                #                         # can't redistribute, use thrust only (clipped)
                #                         u_safe = np.array([u_delta_thrust_fixed, 0.0, 0.0, 0.0])
                #                     else:
                #                         u_others_delta = - (Winv_others @ beta_others) * (residual_rhs / denom2)
                #                         print(f"[Quadcopter CBF] u_others_delta: {u_others_delta}")
                #                         u_safe = np.concatenate(([u_delta_thrust_fixed], u_others_delta))
                #             else:
                #                 print(f"[Quadcopter CBF] using u_delta as u_safe: {u_delta}")
                #                 u_safe = u_delta

                #     # Clip to allowed uLimits
                #     u_safe = np.clip(u_safe, self.erg_c.uLimits[:, 0], self.erg_c.uLimits[:, 1])

                else:
                    # Standard least squares solution for safety control
                    u_safe = -beta.T / (np.linalg.norm(beta)**2) * PSI
                    

        # Apply control limits
        u_safe = np.clip(u_safe, self.erg_c.uLimits[:, 0], self.erg_c.uLimits[:, 1])


        return u_safe
    
    def calcUsafeTIMING(self, x, udef_now, alpha_1=1.0, alpha_2=1.0, delta=0.0):
        import time
        
        SAMPLING_TIME = 0.03  # Hardcoded sampling time in seconds
        timing_results = {}
        
        # Start total timing
        t_start_total = time.perf_counter()

        # Calculate CBF function h(x) and ∇(x)
        t_start = time.perf_counter()
        h, grad_h = self.calcHGradient(x[:2], also_return_h_flag=True)
        timing_results['calcHGradient'] = time.perf_counter() - t_start
        
        # Calculate CBF Hessian
        t_start = time.perf_counter()
        hess_h = np.zeros((self.model.num_of_states, self.model.num_of_states))
        # hess_h = self.calcHessianH(x, epsilon=1e-4)
        timing_results['hessian_creation'] = time.perf_counter() - t_start

        # System Dynamics
        t_start = time.perf_counter()
        f = self.model.f(x, udef_now)
        timing_results['model.f'] = time.perf_counter() - t_start
        
        t_start = time.perf_counter()
        f_x = self.model.f_x(x, udef_now)
        timing_results['model.f_x'] = time.perf_counter() - t_start
        
        t_start = time.perf_counter()
        g = self.model.h(x)
        timing_results['model.h'] = time.perf_counter() - t_start

        # Compute h_dot and h_ddot
        t_start = time.perf_counter()
        h_dot = grad_h.T @ f
        timing_results['h_dot_computation'] = time.perf_counter() - t_start
        
        t_start = time.perf_counter()
        h_ddot = f.T @ hess_h @ f + grad_h.T @ f_x @ f
        timing_results['h_ddot_computation'] = time.perf_counter() - t_start

        # Compute PSI and beta
        t_start = time.perf_counter()
        PSI = h_ddot + 2 * alpha_1 * h_dot + alpha_2 * h
        timing_results['PSI_computation'] = time.perf_counter() - t_start
        
        t_start = time.perf_counter()
        beta = (f.T @ hess_h + grad_h.T @ f_x) @ g
        timing_results['beta_computation'] = time.perf_counter() - t_start

        # Safety control computation
        t_start = time.perf_counter()
        if PSI >= 0:
            # No need to change the control input if PSI > 0, we are not in a danger zone
            u_safe = np.zeros_like(udef_now)      
        else:
            if np.linalg.norm(beta) < 1e-6:
                u_safe = np.zeros_like(udef_now)
            else:
                # Standard least squares solution for safety control
                u_safe = -beta.T / (np.linalg.norm(beta)**2) * PSI
        timing_results['u_safe_computation'] = time.perf_counter() - t_start

        # Apply control limits
        t_start = time.perf_counter()
        u_safe = np.clip(u_safe, self.erg_c.uLimits[:, 0], self.erg_c.uLimits[:, 1])
        timing_results['control_clipping'] = time.perf_counter() - t_start
        
        # Total time
        timing_results['TOTAL'] = time.perf_counter() - t_start_total

        # Print timing results
        print("\n" + "="*70)
        print(f"{'Operation':<30} {'Time (ms)':<15} {'% of Ts (30ms)':<20}")
        print("="*70)
        for operation, elapsed_time in timing_results.items():
            time_ms = elapsed_time * 1000
            percentage = (elapsed_time / SAMPLING_TIME) * 100
            if operation == 'TOTAL':
                print("-"*70)
            print(f"{operation:<30} {time_ms:>10.4f} ms   {percentage:>10.2f}%")
        print("="*70 + "\n")

        input("Press Enter to continue...")

        return u_safe

    # ROS Related Functions -------------------------------------

    def publishCk(self, ck):
        """
        Publish the ck values to a ROS topic
        """
        msg = CkTable()
        msg.model_type = self.model.type
        msg.table_size = self.Kmax + 1
        msg.l_bounds = [float(self.L1_min), float(self.L1_max), float(self.L2_min), float(self.L2_max)]
        msg.ck_values = ck.flatten().tolist()
        msg.ck_values_average_in_range = self.erg_c.ck_aver_others.flatten().tolist()
        msg.total_erg_cost = float(self.erg_c.total_erg_cost)
        msg.total_erg_cost_in_range = float(self.erg_c.total_erg_cost_in_range)
        msg.erg_cost_reduction_perc = float((self.erg_c.init_erg_cost - self.erg_c.total_erg_cost_in_range)/self.erg_c.init_erg_cost) if self.erg_c.init_erg_cost > 0 else 0.0
        msg.position.x = float(self.model.state[0])
        msg.position.y = float(self.model.state[1])
        msg.position.z = 0.0        # Assuming 2D plane for ergodic exploration
        self.ck_publisher.publish(msg)

    def publishData(self, state_now, u_input_now, erg_cost_now, active_cbf_flag, time_now, delta_t_Ts):
        """
        Publish agent data to a ROS topic
        """
        msg = AgentData()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = f"agent_{self.agent_id}"
        msg.simulation_time = float(time_now)
        msg.delta_t_ts = delta_t_Ts
        msg.num_of_states = self.model.num_of_states
        msg.num_of_inputs = self.model.num_of_inputs
        msg.states = [float(x) for x in state_now.flatten()]
        msg.inputs = [float(x) for x in u_input_now.flatten()]
        msg.ergodic_cost = float(erg_cost_now)
        msg.active_cbf_flag = bool(active_cbf_flag)
        msg.in_range_agents_ids = [int(id_) for id_ in self.getInRangeAgentIds()]

        self.data_publisher.publish(msg)
    
    def publishTargetAndObstacleData(self):
        """
        Publish target estimates and known obstacles to ROS topics
        """
        # Publish target estimates
        if len(self.target_estimates) > 0:
            target_estimates = []
            ground_truths = []
            for ekf, est in zip(self.ekfs, self.target_estimates):
                msg = SingleTargetEstimate()
                msg.target_id = ekf.id
                msg.position.x = float(est[0])
                msg.position.y = float(est[1])
                msg.position.z = float(est[2])
                msg.covariance = [float(x) for x in ekf.sigma_k_1.flatten()]
                target_estimates.append(msg)
            
            for idx, gt in enumerate(self.real_target_positions):
                msg = SingleTargetEstimate()
                msg.target_id = idx  # Use index as ID since gt is a numpy array
                msg.position.x = float(gt[0])
                msg.position.y = float(gt[1])
                msg.position.z = float(gt[2])
                ground_truths.append(msg)

            msg = MultipleTargetEstimates()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = f"agent_{self.agent_id}"
            msg.num_of_targets = len(target_estimates)
            msg.target_estimates = target_estimates
            msg.ground_truths = ground_truths

            self.target_est_publisher.publish(msg)

        # Publish known obstacles
        if len(self.obstacle_list) > 0:
            obstacles = []
            for obs in self.obstacle_list:
                msg = SingleObstacle()
                msg.obs_name = obs.name_id
                msg.obs_type = obs.type
                msg.position.x = float(obs.pos[0])
                msg.position.y = float(obs.pos[1])
                if obs.type == "rectangle":
                    msg.dimensions = [obs.width, obs.height]
                elif obs.type == "circle":
                    msg.dimensions = [obs.r]
                elif obs.type == "wall":
                    msg.dimensions = [float(obs.n[0]), float(obs.n[1]), float(obs.wall_length)]
                else:
                    raise ValueError(f"Unknown obstacle type: {obs.type}")
                msg.kappa = float(obs.kappa)
                msg.rho0 = float(obs.rho0)
                obstacles.append(msg)
            msg = MultipleObstacles()
            msg.num_of_obstacles = len(obstacles)
            msg.obstacles = obstacles
            self.known_obst_publisher.publish(msg)

    def discoverAgentsInROS(self):
        """Discover active agent nodes and create subscribers for them"""
        # Get list of all nodes
        node_names = self.get_node_names()
        
        # Look for agent nodes (assuming they follow pattern like 'agent_X' or contain 'agent')
        agent_pattern = re.compile(r'agent[_\-]?(\d+)', re.IGNORECASE)
        current_agents = set()
        
        for node_name in node_names:
            match = agent_pattern.search(node_name)
            if match:
                disc_agent_id = int(match.group(1))
                current_agents.add(disc_agent_id)
                
                # Create subscriber if this is a new agent
                if disc_agent_id not in self.discovered_agents and disc_agent_id != self.agent_id:
                    self.createAgentSubscriber(disc_agent_id)
        
        # Remove subscribers for agents that are no longer active
        inactive_agents = self.discovered_agents - current_agents
        for disc_agent_id in inactive_agents:
            self.removeAgentSubscriber(disc_agent_id)
        
        # Update discovered agents
        if current_agents != self.discovered_agents:
            self.get_logger().info(f'Active agents: {sorted(current_agents)}')
            self.discovered_agents = current_agents
        
        # Update in-range agents based on antenna radius
        self.updateInRangeAgents()

    def updateInRangeAgents(self):
        """Update the list of agents within antenna range"""
        current_in_range = set()
        my_position = np.array([self.model.state[0], self.model.state[1]])
        
        for agent_id in self.discovered_agents:
            if agent_id == self.agent_id:
                continue
                
            # Check if we have position data for this agent
            if agent_id in self.agent_positions:
                other_position = np.array(self.agent_positions[agent_id])
                distance = np.linalg.norm(my_position - other_position)
                
                if distance <= self.antenna_rad:
                    current_in_range.add(agent_id)
        
        # Log changes in in-range agents
        if current_in_range != self.in_range_agents:
            newly_in_range = current_in_range - self.in_range_agents
            newly_out_of_range = self.in_range_agents - current_in_range
            
            if newly_in_range:
                self.get_logger().info(f'Agents now in range: {sorted(newly_in_range)}')
            if newly_out_of_range:
                self.get_logger().info(f'Agents now out of range: {sorted(newly_out_of_range)}')
                
            self.in_range_agents = current_in_range

    def createAgentSubscriber(self, agent_id):
        """Create a subscriber for a specific agent"""
        # Create agent data subscriber
        subscriber = self.create_subscription(
            CkTable,
            f'agent_{agent_id}/ck',
            lambda msg, aid=agent_id: self.agentCkCallback(msg, aid),
            10
        )
        self.ck_subscribers[agent_id] = subscriber
        self.get_logger().info(f'Created subscriber for agent_{agent_id}/ck')

        # Lets add a virtual obstacle in a safe initial position (far from current agent) to avoid colliding with one another
        # This will be updated to the correct position when the first message is received
        safe_initial_pos = [self.model.state[0] + 1000.0, self.model.state[1] + 1000.0]  # Far away initial position
        virtual_obs = Obstacle(pos=safe_initial_pos, dimensions=0.6, obs_type='circle', kappa=self.KAPPA_OBS_VIRTUAL, rho0=self.RHO_OBS_VIRTUAL, obs_name=f"agent_{agent_id}")
        saveObstaclesToMemory(self, [virtual_obs])
        

    def removeAgentSubscriber(self, agent_id):
        """Remove subscriber and data for an inactive agent"""
        if agent_id in self.ck_subscribers:
            # Destroy agent data subscriber
            self.destroy_subscription(self.ck_subscribers[agent_id])
            del self.ck_subscribers[agent_id]
            self.get_logger().info(f'Removed subscriber for agent_{agent_id}')
        
        # Remove stored CK data for this agent
        if agent_id in self.agent_ck_data:
            del self.agent_ck_data[agent_id]
            
        # Remove stored position data for this agent
        if agent_id in self.agent_positions:
            del self.agent_positions[agent_id]
            
        # Remove stored model type data for this agent
        if agent_id in self.agent_model_types:
            del self.agent_model_types[agent_id]
            
        # Remove stored l_bounds data for this agent
        if agent_id in self.agent_l_bounds:
            del self.agent_l_bounds[agent_id]
            
        # Remove from in-range agents
        self.in_range_agents.discard(agent_id)

        # Lets remove the virtual obstacle in the position of the discoverd agent in memory
        removeObstaclesFromMemory(self, [f"agent_{agent_id}"])

    def agentCkCallback(self, msg, agent_id):
        """Callback function to handle CK data from other agents"""
        try:
            # Store agent model type from the message
            self.agent_model_types[agent_id] = msg.model_type
            
            # Store agent l_bounds from the message
            self.agent_l_bounds[agent_id] = list(msg.l_bounds)
            
            # Check if talk_alike_flag is enabled and if model types match
            if self.talk_alike_flag:
                if msg.model_type != self.model.type:
                    # Log that we're ignoring this message due to model type mismatch
                    self.get_logger().info(f'Ignoring CK data from agent_{agent_id} (model type: {msg.model_type}) - not compatible with my model type: {self.model.type}')
                    return
            
            # Check if same_l_bounds_flag is enabled and if l_bounds match
            if self.same_l_bounds_flag:
                my_l_bounds = [self.L1_min, self.L1_max, self.L2_min, self.L2_max]
                # Convert msg.l_bounds to list for proper comparison
                msg_l_bounds = list(msg.l_bounds)
                if msg_l_bounds != my_l_bounds:
                    # Log that we're ignoring this message due to l_bounds mismatch
                    self.get_logger().info(f'Ignoring CK data from agent_{agent_id} (l_bounds: {msg_l_bounds}) - not compatible with my l_bounds: {my_l_bounds}')
                    return
            
            # Store agent position from the message
            self.agent_positions[agent_id] = (msg.position.x, msg.position.y)

            # Find the obstacle with name "agent_{id}" and update virtual obstacle position in memory (list: self.obstacle_list)
            # If the virtual obstacle doesn't exist yet, create it
            obstacle_found = False
            for obs in self.obstacle_list:
                if obs.name_id == f"agent_{agent_id}":
                    obs.pos = np.asarray([msg.position.x, msg.position.y])
                    obstacle_found = True
                    break
            
            if not obstacle_found:
                # Create virtual obstacle if it doesn't exist
                virtual_obs = Obstacle(pos=[msg.position.x, msg.position.y], dimensions=0.6, obs_type='circle', kappa=self.KAPPA_OBS_VIRTUAL, rho0=self.RHO_OBS_VIRTUAL, obs_name=f"agent_{agent_id}")
                saveObstaclesToMemory(self, [virtual_obs])

            # Convert flattened row-major form back to table
            table_size = msg.table_size
            ck_flat = np.array(msg.ck_values)
            
            # Reshape to (table_size, table_size) matrix
            ck_table = ck_flat.reshape((table_size, table_size))

            # If table_size is less than my Kmax+1, pad with zeros otherwise truncate
            if table_size < self.Kmax + 1:
                padded_ck = np.zeros((self.Kmax + 1, self.Kmax + 1))
                padded_ck[:table_size, :table_size] = ck_table
                ck_table = padded_ck
            elif table_size > self.Kmax + 1:
                ck_table = ck_table[:self.Kmax + 1, :self.Kmax + 1]

            # Store the CK data for this agent
            self.agent_ck_data[agent_id] = ck_table
            
            # Log reception (optional, can be commented out to reduce verbosity)
            # self.get_logger().info(f'Received CK data from agent_{agent_id}, table size: {table_size}')
            
        except Exception as e:
            self.get_logger().error(f'Error processing CK data from agent_{agent_id}: {str(e)}')

    def getAgentCkData(self, agent_id=None, in_range_only=False):
        """
        Get CK data from other agents
        
        Args:
            agent_id: If specified, return CK data for that specific agent.
                     If None, return dict of all agents' CK data.
            in_range_only: If True, only return data for agents within antenna range.
        
        Returns:
            If agent_id specified: numpy array of CK table or None if not available
            If agent_id is None: dict {agent_id: ck_table} for all discovered agents
        """
        def _isCompatibleAgent(aid):
            """Helper function to check if agent is compatible based on flags"""
            # Check if in range if requested
            if in_range_only and aid not in self.in_range_agents:
                return False
            
            # Check if model types match if talk_alike_flag is enabled
            if self.talk_alike_flag:
                agent_model_type = self.agent_model_types.get(aid, None)
                if agent_model_type is None or agent_model_type != self.model.type:
                    return False
            
            # Check if l_bounds match if same_l_bounds_flag is enabled
            if self.same_l_bounds_flag:
                agent_l_bounds = self.agent_l_bounds.get(aid, None)
                my_l_bounds = [self.L1_min, self.L1_max, self.L2_min, self.L2_max]
                if agent_l_bounds is None or agent_l_bounds != my_l_bounds:
                    return False
            
            return True
        
        if agent_id is not None:
            # Check if specific agent is compatible
            if not _isCompatibleAgent(agent_id):
                return None
            return self.agent_ck_data.get(agent_id, None)
        else:
            # Return data from compatible agents
            return {aid: ck_data for aid, ck_data in self.agent_ck_data.items() 
                   if _isCompatibleAgent(aid)}

    def getInRangeAgentIds(self):
        """Get list of currently in-range agent IDs (excluding self)"""
        if self.antenna_range_flag:
            # Filter in-range agents based on compatibility flags
            compatible_agents = set()
            for aid in self.in_range_agents:
                # Check talk_alike_flag compatibility
                if self.talk_alike_flag:
                    if self.agent_model_types.get(aid, None) != self.model.type:
                        continue
                
                # Check same_l_bounds_flag compatibility
                if self.same_l_bounds_flag:
                    my_l_bounds = [self.L1_min, self.L1_max, self.L2_min, self.L2_max]
                    if self.agent_l_bounds.get(aid, None) != my_l_bounds:
                        continue
                
                compatible_agents.add(aid)
            
            return sorted(list(compatible_agents))
        else:
            return self.getDiscoveredAgentIds()

    def getDiscoveredAgentIds(self):
        """Get list of currently discovered agent IDs (excluding self)"""
        compatible_agents = set()
        for aid in (self.discovered_agents - {self.agent_id}):
            # Check talk_alike_flag compatibility
            if self.talk_alike_flag:
                if self.agent_model_types.get(aid, None) != self.model.type:
                    continue
            
            # Check same_l_bounds_flag compatibility
            if self.same_l_bounds_flag:
                my_l_bounds = [self.L1_min, self.L1_max, self.L2_min, self.L2_max]
                if self.agent_l_bounds.get(aid, None) != my_l_bounds:
                    continue
            
            compatible_agents.add(aid)
        
        return sorted(list(compatible_agents))

    def updateVirtualObstaclesBasedOnCompatibility(self):
        """
        Update virtual obstacles based on current compatibility flags.
        Remove obstacles for incompatible agents and ensure obstacles exist for compatible agents.
        """
        # Get currently compatible agents
        compatible_agents = set()
        for aid in self.discovered_agents:
            if aid == self.agent_id:
                continue
                
            # Check talk_alike_flag compatibility
            if self.talk_alike_flag:
                if self.agent_model_types.get(aid, None) != self.model.type:
                    continue
            
            # Check same_l_bounds_flag compatibility
            if self.same_l_bounds_flag:
                my_l_bounds = [self.L1_min, self.L1_max, self.L2_min, self.L2_max]
                if self.agent_l_bounds.get(aid, None) != my_l_bounds:
                    continue
            
            compatible_agents.add(aid)
        
        # Find existing virtual obstacles for agents
        existing_virtual_obstacles = set()
        obstacles_to_remove = []
        
        for obs in self.obstacle_list:
            if obs.name_id and obs.name_id.startswith("agent_"):
                try:
                    agent_id_str = obs.name_id.replace("agent_", "")
                    agent_id = int(agent_id_str)
                    existing_virtual_obstacles.add(agent_id)
                    
                    # If this agent is no longer compatible, mark obstacle for removal
                    if agent_id not in compatible_agents:
                        obstacles_to_remove.append(obs.name_id)
                        self.get_logger().info(f'Removing virtual obstacle for incompatible agent_{agent_id}')
                except ValueError:
                    # Not a valid agent ID format, skip
                    pass
        
        # Remove obstacles for incompatible agents
        if obstacles_to_remove:
            removeObstaclesFromMemory(self, obstacles_to_remove)
        
        # Create obstacles for compatible agents that don't have them yet
        for agent_id in compatible_agents:
            if agent_id not in existing_virtual_obstacles:
                # Create virtual obstacle if we have position data for this agent
                if agent_id in self.agent_positions:
                    pos = self.agent_positions[agent_id]
                    virtual_obs = Obstacle(pos=[pos[0], pos[1]], dimensions=0.6, obs_type='circle', kappa=self.KAPPA_OBS_VIRTUAL, rho0=self.RHO_OBS_VIRTUAL, obs_name=f"agent_{agent_id}")
                    saveObstaclesToMemory(self, [virtual_obs])
                    self.get_logger().info(f'Created virtual obstacle for compatible agent_{agent_id} at position {pos}')
                else:
                    # Create at safe initial position if no position data yet
                    safe_initial_pos = [self.model.state[0] + 1000.0, self.model.state[1] + 1000.0]
                    virtual_obs = Obstacle(pos=safe_initial_pos, dimensions=0.6, obs_type='circle', kappa=self.KAPPA_OBS_VIRTUAL, rho0=self.RHO_OBS_VIRTUAL, obs_name=f"agent_{agent_id}")
                    saveObstaclesToMemory(self, [virtual_obs])
                    self.get_logger().info(f'Created virtual obstacle for compatible agent_{agent_id} at safe initial position')
            