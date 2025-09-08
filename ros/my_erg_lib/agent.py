# Ergodic Library
from my_erg_lib.basis import Basis
import numpy as np
from my_erg_lib.model_dynamics import SingleIntegrator, DoubleIntegrator, Quadcopter
from my_erg_lib.eid import Sensor, EKF
import my_erg_lib.Utilities as utils
import time

# ROS Library
from rclpy.node import Node
from my_interfaces.msg import CkTable, AgentData, SingleObstacle, MultipleObstacles, SingleTargetEstimate, MultipleTargetEstimates  # type: ignore
import re

class Agent(Node):
    def __init__(self, L1, L2, Kmax, dynamics_model, phi=None, x0=None, agent_id=None, antenna_rad=np.inf, antenna_range_flag=False):
        self.agent_id = agent_id if agent_id is not None else 0
        self.time_since_start = 0.0

        super().__init__(f"agent_{self.agent_id}")  # ROS node with name "agent_{id}"
        
        # Space Parameters
        self.L1 = L1
        self.L2 = L2
        self.Kmax = Kmax
        self.antenna_rad = antenna_rad  # Antenna radius in meters (for CK sharing among agents)
        self.antenna_range_flag = antenna_range_flag
        
        # Connecting model dynamics
        self.model = dynamics_model
        # Make sure x0 is within the space limits 0->L1, 0->L2 if given
        if x0 is not None:
            assert len(x0) == self.model.num_of_states, "Initial position x0 must have the same length as the model's state vector."
            assert 0 <= x0[0] <= L1 and 0 <= x0[1] <= L2, "Initial position x0 must be within the limits of the space." # TODO: Assuming x[0], x[1] are the x, y coordinates
            self.model.reset(x0)

        # TODO: Maybe make a separate target class for this?
        # Multiple targets setup
        self.real_target_positions = [      # Real target position (Ground Truth)   # TODO: Maybe take it from an env?
            np.array([2, 2, 0]),        # Target 1
            np.array([4, 8, 0]),        # Target 2
            np.array([8, 6, 0])         # Target 3
        ]
        # TODO: Fix comparing real targ pos to agents pos for different dynamics models
        self.num_of_targets = 0     # Number of target estimates so far
        self.target_estimates = []
        
        # Multiple EKF instances - one per target estimate
        self.ekfs = []

        # Lets connect a sensor to track the target position 
        self.sensor = Sensor(sensor_range=2,
                             R=np.diag([0.01, 0.01]))         

        # Initialise obstacle list
        self.obstacle_list = []

        # Initialize the basis object
        self.basis = Basis(L1, L2, Kmax, phi_=phi, precalc_phik_coeff=False, num_gauss_points=22)
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
                self.model.state[0] + self.sensor.sensor_range/2 * np.sin(beta),  # x position # TODO: Validate those
                self.model.state[1] + self.sensor.sensor_range/2 * np.cos(beta),  # y position
                0                                                         # z position (assuming flat ground)
            ])
            # Clip to 0->L1, 0->L2
            a_init[0] = np.clip(a_init[0], 0, self.L1)
            a_init[1] = np.clip(a_init[1], 0, self.L2)
        elif init_position is not None:
            a_init = np.asarray(init_position)
        else:
            raise ValueError("Either 'measurement' or 'a_init' must be provided to spawn a new target.")
        
        ekf_ = EKF(ekf_id=self.ekfs[-1].id + 1 if self.ekfs != [] else 0,  # Increment ID if exists, else start from 0
                   a_init=a_init,
                   sigma_init=np.eye(3)*5e-1,    # 1e-1 may be more appropriate
                   R=np.diag([0.1, 0.1]),        # Sensor noise covariance    
                   Q=np.eye(3) * 1e-4,           # Process noise covariance (1e-5)
                   a_limits=[[0, self.L1], [0, self.L2], [0, 10]],
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
        if not (0 <= x[0] <= self.L1 and 0 <= x[1] <= self.L2):
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

    def calcPotentialUGradient(self, x):
        """
        Calculate the gradient of the potential function U at a given state x.
        """
        grad_U = np.zeros(2)
        for obs in self.obstacle_list:
            grad_U += obs.gradU(x[:2])

        return grad_U

    def calcH(self, x, delta=0.0):
        """
        Calculate h(x), the CBF (Control Barrier Function) value at a given state x.
        This function is used to ensure safety constraints are satisfied.
        """
        U = self.calcPotentialU(x)
        h = 1 / (1 + U) - delta
        return h

    def calcHGradient(self, x):
        """
        Calculate the gradient of h(x) at a given state x.
        This function is used to ensure safety constraints are satisfied.
        ATTENTION: Returns 2x1 vector, only for positional dimensions. Need to append accordingly in order to multiply by f(x) later on.
        """
        U = self.calcPotentialU(x)
        grad_U = self.calcPotentialUGradient(x)
        h_grad = -grad_U / (1 + U)**2

        # Append zeros for the other dimensions if needed (e.g., for quadcopter)
        if self.model.num_of_states > 2:
            h_grad = np.append(h_grad, np.zeros(self.model.num_of_states - 2))

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

        # Calculate CBF function h(x)
        h = self.calcH(x, delta)
        # Calculate CBF gradient
        grad_h = self.calcHGradient(x[:2])
        # Calculate CBF Hessian
        hess_h = self.calcHessianH(x, epsilon=1e-4)

        # System Dynamics
        f = self.model.f(x, udef_now)
        f_x = self.model.f_x(x, udef_now)
        g = self.model.h(x)

        h_dot = grad_h.T @ f    
        h_ddot = f.T @ hess_h @ f + grad_h.T @ f_x @ f

        PSI = h_ddot + 2 * alpha_1 * h_dot + alpha_2 * h
        # PSI = h_ddot + alpha_1 * h_dot + alpha_2 * h
        beta = (f.T @ hess_h + grad_h.T @ f_x) @ g

        if PSI >= 0:
            # No need to change the control input if PSI > 0, we are not in a danger zone
            u_safe = np.zeros_like(udef_now)      
        else:
            if np.linalg.norm(beta) < 1e-6:
                u_safe = np.zeros_like(udef_now)
            else:
                u_safe = - beta.T / (np.linalg.norm(beta)**2) * PSI
                pass

        u_safe = np.clip(u_safe, self.erg_c.uLimits[:, 0], self.erg_c.uLimits[:, 1])


        return u_safe
    


    # ROS Related Functions -------------------------------------

    def publishCk(self, ck):
        """
        Publish the ck values to a ROS topic
        """
        msg = CkTable()
        msg.table_size = self.Kmax + 1
        msg.ck_values = ck.flatten().tolist()
        msg.ck_values_average_in_range = self.erg_c.ck_aver_others.flatten().tolist()
        msg.total_erg_cost = float(self.erg_c.total_erg_cost)
        msg.total_erg_cost_in_range = float(self.erg_c.total_erg_cost_in_range)
        msg.position.x = float(self.model.state[0])
        msg.position.y = float(self.model.state[1])
        msg.position.z = 0.0        # Assuming 2D plane for ergodic exploration
        self.ck_publisher.publish(msg)

    def publishData(self, state_now, u_input_now, erg_cost_now, active_cbf_flag, time_now):
        """
        Publish agent data to a ROS topic
        """
        msg = AgentData()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = f"agent_{self.agent_id}"
        msg.simulation_time = float(time_now)
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
                    msg.dimensions = [float(obs.n[0]), float(obs.n[1])]
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
            
        # Remove from in-range agents
        self.in_range_agents.discard(agent_id)

    def agentCkCallback(self, msg, agent_id):
        """Callback function to handle CK data from other agents"""
        try:
            # Store agent position from the message
            self.agent_positions[agent_id] = (msg.position.x, msg.position.y)
            
            # Convert flattened row-major form back to table
            table_size = msg.table_size
            ck_flat = np.array(msg.ck_values)
            
            # Reshape to (table_size, table_size) matrix
            ck_table = ck_flat.reshape((table_size, table_size))
            
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
        if agent_id is not None:
            # Check if specific agent is in range if requested
            if in_range_only and agent_id not in self.in_range_agents:
                return None
            return self.agent_ck_data.get(agent_id, None)
        else:
            if in_range_only:
                # Return only data from agents within range
                return {aid: ck_data for aid, ck_data in self.agent_ck_data.items() 
                       if aid in self.in_range_agents}
            else:
                return self.agent_ck_data.copy()

    def getInRangeAgentIds(self):
        """Get list of currently in-range agent IDs (excluding self)"""
        return sorted(list(self.in_range_agents)) if (self.antenna_range_flag == True) else self.getDiscoveredAgentIds()

    def getDiscoveredAgentIds(self):
        """Get list of currently discovered agent IDs (excluding self)"""
        return sorted(list(self.discovered_agents - {self.agent_id}))
            