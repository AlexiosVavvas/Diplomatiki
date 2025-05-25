import numpy as np
import time
# TODO: A cool idea could be a measurement model measuring only distance R(x, y). This way we simultate the RF beacon giving us the signal intensity at each point and trying to find the exact location of the stolen property

# class MeasurementModel:
class MeasurementModel_NonVectorized:
    def __init__(self):
        self.mu = 2  # μ: Number of measurements (beta, phi)
        self.M = 3   # M: Number of estimated states (xt, yt, zt)

    def Y(self, a, x):
        """	
        Measurement function
            Y = [beta, phi]
        where:
            beta = atan2(yt - yq, xt - xq)                          # Azimuth angle         
            phi = atan2(zt - zq, sqrt((xt - xq)^2 + (yt - yq)^2))   # Elevation angle
        """
        assert len(a) == self.M, "'a' must be a 3D vector [xt, yt, zt]"
        assert len(x) == self.M, "'x' must be a 3D vector [xq, yq, zq]"

        xt, yt, zt = a
        xq, yq, zq = x

        # Measurement function
        # Azimuth angle
        if (yt - yq) == 0 and (xt - xq) == 0:
            # We are right above it
            beta = 0
        elif (yt - yq) == 0:
            # We view it right in front of us
            beta = np.pi / 2
        else:
            beta = np.arctan2(xt - xq, yt - yq)
            
        # Elevation angle
        if (xt == xq) and (yt == yq):
            # We are right above it
            phi = np.pi / 2
        elif zt == zq:
            # We view it from the same height
            phi = 0
        else:
            phi = np.arctan2(zq-zt, np.sqrt((xt - xq) ** 2 + (yt - yq) ** 2))  # Elevation angle

        return np.array([beta, phi])

    def H(self, a, x):
        """
        Jacobian of the measurement function
        """
        assert len(a) == self.M, "'a' must be a 3D vector [xt, yt, zt]"
        assert len(x) == self.M, "'x' must be a 3D vector [xq, yq, zq]"

        xt, yt, zt = a
        xq, yq, zq = x

        both_zero = (yt - yq) == 0 and (xt - xq) == 0
        # Jacobian of the measurement function
        Y1_a1 = (yt - yq)/((xq-xt)**2 + (yq-yt)**2) if not both_zero else 0
        Y1_a2 = (xq - xt)/((xq-xt)**2 + (yq-yt)**2) if not both_zero else 0
        Y1_a3 = 0
        Y2_a1 = ((xq-xt)*(zq-zt))/(np.sqrt((xq-xt)**2 + (yq-yt)**2) * ((xq-xt)**2 + (yq-yt)**2 + (zq-zt)**2)) if not both_zero else 0
        Y2_a2 = ((yq-yt)*(zq-zt))/(np.sqrt((xq-xt)**2 + (yq-yt)**2) * ((xq-xt)**2 + (yq-yt)**2 + (zq-zt)**2)) if not both_zero else 0
        Y2_a3 = - 1 / (np.sqrt((xq-xt)**2 + (yq-yt)**2) * (1 + ((zq-zt)**2)/((xq-xt)**2 + (yq-yt)**2))) if not both_zero else 0

        H_ = np.array([[Y1_a1, Y1_a2, Y1_a3],
                        [Y2_a1, Y2_a2, Y2_a3]])

        return H_
    
# class MeasurementModelVEC:
class MeasurementModel:
    """
    Measurement model for the target state estimation
        - mu: Number of measurements (beta, phi)
        - M: Number of estimated states (xt, yt, zt)
        - Y: Measurement function
        - H: Jacobian of the measurement function

    Note: Vectorized version of the measurement model
        - Handles multiple target positions and multiple agent positions
        - Returns measurements and Jacobian matrices in a vectorized manner
        - Supports both single and multiple target/agent positions
    """
    def __init__(self):
        self.mu = 2  # μ: Number of measurements (beta, phi)
        self.M = 3   # M: Number of estimated states (xt, yt, zt)

    def Y(self, a, x):
        """	
        Measurement function (vectorized)
            Y = [beta, phi]
        where:
            beta = atan2(xt - xq, yt - yq)                          # Azimuth angle         
            phi = atan2(zq - zt, sqrt((xt - xq)^2 + (yt - yq)^2))   # Elevation angle
        
        Args:
            a: Target position(s) [xt, yt, zt] - shape (3,) or (N, 3)
            x: Agent position(s) [xq, yq, zq] - shape (3,) or (M, 3)
        
        Returns:
            Y_: Measurement(s) - shape (2,) or (N, 2) or (M, 2)
        """
        # Convert inputs to numpy arrays
        a = np.asarray(a)
        x = np.asarray(x)
        
        # Determine which input is vectorized
        a_vectorized = a.ndim == 2
        x_vectorized = x.ndim == 2
        
        # Handle different cases
        if not a_vectorized and not x_vectorized:
            # Both single positions - original behavior
            assert len(a) == self.M, "'a' must be a 3D vector [xt, yt, zt]"
            assert len(x) == self.M, "'x' must be a 3D vector [xq, yq, zq]"
            a = a.reshape(1, -1)
            x = x.reshape(1, -1)
            squeeze_output = True
        elif a_vectorized and not x_vectorized:
            # Multiple targets, single agent - NEW FUNCTIONALITY
            assert a.shape[1] == self.M, "'a' must have shape (N, 3)"
            assert len(x) == self.M, "'x' must be a 3D vector [xq, yq, zq]"
            x = np.tile(x.reshape(1, -1), (a.shape[0], 1))  # Broadcast x to match a
            squeeze_output = False
        elif not a_vectorized and x_vectorized:
            # Single target, multiple agents - EXISTING FUNCTIONALITY
            assert len(a) == self.M, "'a' must be a 3D vector [xt, yt, zt]"
            assert x.shape[1] == self.M, "'x' must have shape (M, 3)"
            a = np.tile(a.reshape(1, -1), (x.shape[0], 1))  # Broadcast a to match x
            squeeze_output = False
        else:
            # Both vectorized - not supported for now
            raise ValueError("Both 'a' and 'x' cannot be vectorized simultaneously")
        
        # Now both a and x have shape (N, 3)
        xt, yt, zt = a[:, 0], a[:, 1], a[:, 2]  # Shape: (N,) each
        xq, yq, zq = x[:, 0], x[:, 1], x[:, 2]  # Shape: (N,) each
        
        # Calculate differences
        dx = xt - xq  # Shape: (N,)
        dy = yt - yq  # Shape: (N,)
        dz = zq - zt  # Shape: (N,)
        
        # Azimuth angle (beta)
        # Handle special cases vectorized
        both_zero_xy = (dy == 0) & (dx == 0)
        dy_zero = (dy == 0) & (dx != 0)
        
        beta = np.arctan2(dx, dy)  # Default case
        beta = np.where(both_zero_xy, 0, beta)  # Right above target
        beta = np.where(dy_zero, np.pi / 2, beta)  # dy == 0 but dx != 0
        
        # Elevation angle (phi)
        # Calculate horizontal distance
        horizontal_dist = np.sqrt(dx**2 + dy**2)
        
        # Handle special cases
        both_zero_xy = (dx == 0) & (dy == 0)  # Right above target
        dz_zero = (dz == 0)  # Same height
        
        phi = np.arctan2(dz, horizontal_dist)  # Default case
        phi = np.where(both_zero_xy, np.pi / 2, phi)  # Right above target
        phi = np.where(dz_zero & ~both_zero_xy, 0, phi)  # Same height but not directly above
        
        # Stack measurements
        Y_ = np.stack([beta, phi], axis=1)  # Shape: (N, 2)
        
        # Squeeze if input was single position
        if squeeze_output:
            Y_ = Y_.squeeze(0)  # Shape: (2,)
        
        return Y_


    def H(self, a, x):
        """
        Jacobian of the measurement function
        Vectorized version that accepts multiple target positions OR multiple agent positions
        
        Args:
            a: Target position(s) [xt, yt, zt] - shape (3,) or (N, 3)
            x: Agent position(s) [xq, yq, zq] - shape (3,) or (M, 3)
        
        Returns:
            H_: Jacobian matrix - shape (2, 3) or (N, 2, 3) or (M, 2, 3)
        """
        # Convert inputs to numpy arrays
        a = np.asarray(a)
        x = np.asarray(x)
        
        # Determine which input is vectorized
        a_vectorized = a.ndim == 2
        x_vectorized = x.ndim == 2
        
        # Handle different cases
        if not a_vectorized and not x_vectorized:
            # Both single positions - original behavior
            assert len(a) == self.M, "'a' must be a 3D vector [xt, yt, zt]"
            assert len(x) == self.M, "'x' must be a 3D vector [xq, yq, zq]"
            a = a.reshape(1, -1)
            x = x.reshape(1, -1)
            squeeze_output = True
        elif a_vectorized and not x_vectorized:
            # Multiple targets, single agent - YOUR USE CASE
            assert a.shape[1] == self.M, "'a' must have shape (N, 3)"
            assert len(x) == self.M, "'x' must be a 3D vector [xq, yq, zq]"
            x = np.tile(x.reshape(1, -1), (a.shape[0], 1))  # Broadcast x to match a
            squeeze_output = False
        elif not a_vectorized and x_vectorized:
            # Single target, multiple agents - original implementation
            assert len(a) == self.M, "'a' must be a 3D vector [xt, yt, zt]"
            assert x.shape[1] == self.M, "'x' must have shape (M, 3)"
            a = np.tile(a.reshape(1, -1), (x.shape[0], 1))  # Broadcast a to match x
            squeeze_output = False
        else:
            # Both vectorized - not supported for now
            raise ValueError("Both 'a' and 'x' cannot be vectorized simultaneously")
        
        # Now both a and x have shape (N, 3)
        xt, yt, zt = a[:, 0], a[:, 1], a[:, 2]  # Shape: (N,) each
        xq, yq, zq = x[:, 0], x[:, 1], x[:, 2]  # Shape: (N,) each
        
        # Calculate differences (match original H method signs)
        dx = xq - xt  # Shape: (N,)
        dy = yq - yt  # Shape: (N,)
        dz = zq - zt  # Shape: (N,)
        
        # Calculate common terms
        dx2_dy2 = dx**2 + dy**2  # Shape: (N,)
        both_zero = (dy == 0) & (dx == 0)  # Shape: (N,)
        
        # Calculate denominators with safe division
        denom1 = np.where(both_zero, 1, dx2_dy2)  # Avoid division by zero
        sqrt_dx2_dy2 = np.sqrt(np.maximum(dx2_dy2, 1e-12))  # Avoid sqrt of zero
        total_dist_sq = dx2_dy2 + dz**2
        
        # Jacobian elements for azimuth angle (Y1) - match original signs
        Y1_a1 = np.where(both_zero, 0, -dy / denom1)  # -(yq-yt) = (yt-yq)
        Y1_a2 = np.where(both_zero, 0, dx / denom1)   # (xq-xt)
        Y1_a3 = np.zeros_like(Y1_a1)  # Shape: (N,)
        
        # Jacobian elements for elevation angle (Y2)
        denom2 = sqrt_dx2_dy2 * total_dist_sq
        Y2_a1 = np.where(both_zero, 0, dx * dz / denom2)  # Match original: ((xq-xt)*(zq-zt))
        Y2_a2 = np.where(both_zero, 0, dy * dz / denom2)  # Match original: ((yq-yt)*(zq-zt))
        
        # For Y2_a3, we need to handle the arctangent derivative
        elevation_denom = sqrt_dx2_dy2 * (1 + (dz**2) / np.maximum(dx2_dy2, 1e-12))
        Y2_a3 = np.where(both_zero, 0, -1 / elevation_denom)  # Keep negative sign
        
        # Stack into Jacobian matrices
        H_ = np.stack([
            np.stack([Y1_a1, Y1_a2, Y1_a3], axis=1),  # Shape: (N, 3)
            np.stack([Y2_a1, Y2_a2, Y2_a3], axis=1)   # Shape: (N, 3)
        ], axis=1)  # Final shape: (N, 2, 3)
        
        # Squeeze if input was single position
        if squeeze_output:
            H_ = H_.squeeze(0)  # Shape: (2, 3)
        
        return H_


class Sensor:
    def __init__(self, sensor_range, sensor_id=0, R=None):
        """
        Measurement unit to simulate sensor.
        Given the actual real positions it returns a simulated measurement 
            [beta: azimuth angle, phi: elevation angle]
            - sensor_range: The range of the sensor [m]
            - sensor_id: The ID of the sensor
            - R: Sensor noise covariance matrix
        """
        self.sensor_id = sensor_id

        assert sensor_range > 0, "'sensor_range' must be a positive number."
        self.sensor_range = sensor_range
        
        # Lets connect and a measurement model
        self.measurement_model = MeasurementModel()
        self.mu = self.measurement_model.mu  # Number of measurements

        # Actual Sensor noise covariance matrix
        assert R is None or (R.shape[0] == self.mu and R.shape[1] == self.mu), f"R must be a μxμ=({self.mu}x{self.mu}) matrix."
        self.R = np.eye(self.mu) * 0.035 if R is None else np.asarray(R)

    def getMeasurement(self, real_target_position, agent_position):
        """
        Simulate a sensor measurement
            - real_target_position: The real position of the target
            - agent_position: The position of the agent
        Returns: A simulated measurement (None if out of range)
        """
        assert len(real_target_position) == len(agent_position), "'real_target_position' and 'agent_position' must be the same length."

        # Check if the target is within range
        distance = np.sqrt((real_target_position[0] - agent_position[0]) ** 2 + (real_target_position[1] - agent_position[1]) ** 2)
        if distance > self.sensor_range:
            return None
        
        # z = Y(a, x) + δ where δ is N(0, R)
        # Simulate noise
        noise = np.random.multivariate_normal(mean=np.zeros(self.mu), cov=self.R)
        
        # Get the measurement
        measurement = self.measurement_model.Y(real_target_position, agent_position) + noise

        return measurement


class EKF:
    """
    Extended Kalman Filter (EKF) for target state estimation
        - a_init: Initial target state estimate
        - sigma_init: Initial covariance matrix
        - R: Sensor noise covariance matrix
        - Q: Process noise covariance matrix
        - a_limits: Limits for the target state estimate
    """
    def __init__(self, a_init, sigma_init=None, R=None, Q=None, a_limits=None):
        # Target State Estimate
        self.M = len(a_init)  # Number of estimated states
        self.a_k_1 = np.asarray(a_init)
        # Lets set the limits if exist as well
        if a_limits is not None:
            # a_limits should be in the form [[a1_min, a1_max], [a2_min, a2_max], [a3_min, a3_max], ...]
            for i in range(self.M):
                assert len(a_limits[i]) == 2, "'a_limits' must be a matrix in the form [[a1_min, a1_max], [a2_min, a2_max], ...]"
                assert a_limits[i][0] < a_limits[i][1], "'a_limits' low value must be less than high values."
            self.a_limits = np.asarray(a_limits)
        else:
            self.a_limits = np.array([[-np.inf, np.inf] for i in range(self.M)])  # No limits


        # Initial Covariance Matrix Sigma (Σ)
        self.sigma_k_1 = np.eye(self.M) * 1  if sigma_init is None else np.asarray(sigma_init)

        # Measurement Model
        self.measurement_model = MeasurementModel()
        self.mu = self.measurement_model.mu  # Number of measurements

        # Estimated Sensor Noise covariance
        assert R is None or (R.shape[0] == self.mu and R.shape[1] == self.mu), f"R must be a μxμ=({self.mu}x{self.mu}) matrix."
        self.R = np.eye(self.mu) * 0.035 if R is None else np.asarray(R)

        # Process Noise covariance (Not used in the paper, but makes the uncertainty grow especially when no measurements are available)
        assert Q is None or (Q.shape[0] == self.M and Q.shape[1] == self.M), f"Q must be a MxM=({self.M}x{self.M}) matrix."
        self.Q = np.eye(self.M) * 1e-4 if Q is None else np.asarray(Q)


    def predict(self):
        """
        Prediction model for the target state
            - a: Previous state estimate (a_k|k-1)
          => Returns the predicted state estimate (a_k|k) and covariance (Σ_k|k)
        """
        # Trivial since we assume static target model
        ak_k_1 = self.a_k_1            # a_k|k-1
        sigmaK_k_1 = self.sigma_k_1    # Σ_k|k-1
        return ak_k_1, sigmaK_k_1


    # zk: New sensor measurement
    def update(self, xk, zk, update_internal_state=True):
        """
        Updates the belief "a" based on the new measurement "zk"
            - ak_k_1: Previous state estimate (a_k|k-1)
            - zk: New measurement (z_k)
        """
        # Predict target state
        ak_k_1, sigmaK_k_1 = self.predict()
        sigmaK_k_1 += self.Q 

        if zk is not None:
            # Compute expected measurement
            zk_hat = self.measurement_model.Y(ak_k_1, xk)  # Y(a_k|k-1)

            # Linearize the measurement model (get Jacobian)
            Hk = self.measurement_model.H(ak_k_1, xk)

            # Compute Kalman Gain
            S = Hk @ sigmaK_k_1 @ Hk.T + self.R
            S_inv = np.linalg.solve(S, np.eye(S.shape[0]))
            Kk = sigmaK_k_1 @ Hk.T @ S_inv

            # Update state 
            zk_minus_zk_hat = (zk - zk_hat + np.pi) % (2 * np.pi) - np.pi  # Normalise angle diff to [-pi, pi] 

            ak = ak_k_1 + Kk @ zk_minus_zk_hat
            ak = np.clip(ak, self.a_limits[:, 0], self.a_limits[:, 1])

            # Update covariance matrix
            sigmaK = (np.eye(len(sigmaK_k_1)) - Kk @ Hk) @ sigmaK_k_1

            # Track NIS: Normalized Innovation Squared -> High when new measurement is far from the predicted one
            # NIS = zk_minus_zk_hat.T @ S_inv @ zk_minus_zk_hat
        else:
            # No measurement available - no correction step
            ak = ak_k_1  # State remains the same
            sigmaK = sigmaK_k_1  # Covariance only grows due to Q (already added above)
            # NIS = 0

        # Update internal state
        if update_internal_state:
            self.a_k_1 = ak
            self.sigma_k_1 = sigmaK

        return ak, sigmaK


    def p(self, a_array, upper_lim_to_normalise=None):
        """
        Vectorized probability density function of the target state
        Multivariate Gaussian distribution
            p(a_) = N(a_, a, Σ)
            - a_array: Array of evaluation positions (N x M matrix where N is number of points, M is dimensions)
            - a: Target state (self.a_k_1)
            - Σ: Covariance matrix (self.sigma_k_1)
        
        Returns: Array of probability densities (N,)
        Math: 
            p(a_) = 1 / ((2 * π)^(M/2) * |Σ|^0.5) * exp(-0.5 * (a_ - a).T @ Σ_inv @ (a_ - a))
            where:
                - |Σ|:   Determinant of the covariance matrix
                - Σ_inv: Inverse of the covariance matrix
        """
        a_array = np.asarray(a_array)
        
        # Handle single point case
        if a_array.ndim == 1:
            a_array = a_array.reshape(1, -1)
        
        # Calculate differences: (a_array - self.a_k_1)
        a_t = self.a_k_1.copy(); a_t[2] = 0  # Set zt to 0
        diff = a_array - a_t  # Shape: (N, M)
        
        # Calculate normalization constant
        norm_const = 1 / ((2 * np.pi)**(self.M/2) * np.linalg.det(self.sigma_k_1) ** 0.5)
        
        # Calculate inverse of covariance matrix once
        sigma_inv = np.linalg.inv(self.sigma_k_1)
        
        # Vectorized quadratic form calculation
        # For each point: (a_i - self.a_k_1).T @ sigma_inv @ (a_i - self.a_k_1)
        quadratic_forms = np.sum((diff @ sigma_inv) * diff, axis=1)
        
        # Calculate exponentials
        exp_terms = np.exp(-0.5 * quadratic_forms)

        if upper_lim_to_normalise is not None:
            if norm_const > upper_lim_to_normalise:
                norm_const = upper_lim_to_normalise

        return norm_const * exp_terms
    


if __name__ == "__main__":
    # Example usage with environment configurations
    L1, L2 = 1.0, 1.0  # Environment dimensions
    
    # Connect sensor to track target position
    sensor = Sensor(sensor_range=0.2,
                    R=np.diag([0.035, 0.035]))
    
    # Real target position (Ground Truth)
    real_target_position = np.array([0.5, 0.5, 0])
    
    # Initial target position estimate
    a = np.array([0.6, 0.4, 0])
    
    # Initialize EKF
    ekf = EKF(a_init=a,
              sigma_init=np.eye(3)*0.01,
              R=np.diag([0.035, 0.035]),
              a_limits=[[0, L1], [0, L2], [0, 10]])
    
    # Agent position
    agent_position = np.array([0.5, 0.1, 0.1])
    
    # Lets plot p(a) for a in 0->L1, 0->L2
    # Precompute probability values
    N_POINTS = 40
    p_values = np.zeros((N_POINTS, N_POINTS))
    # Calculate all probabilities at once
    all_probs = ekf.p(np.array([[a1, a2, 0] for a1 in np.linspace(0, L1, N_POINTS) for a2 in np.linspace(0, L2, N_POINTS)]))
    # Reshape the results back to a grid
    p_values = all_probs.reshape(N_POINTS, N_POINTS)

    # print 1 / ((2 * π)^(M/2) * |Σ|^0.5)
    print(f"Normalization constant: {1 / ((2 * np.pi)**(ekf.M/2) * np.linalg.det(ekf.sigma_k_1) ** 0.5)}")


    # Plotting
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(15, 6))
    
    # 2D plot
    ax1 = plt.subplot(121)
    im = ax1.imshow(p_values.T, extent=(0, L1, 0, L2), origin='lower', aspect='auto')
    plt.colorbar(im, ax=ax1, label='Probability Density')
    ax1.scatter(real_target_position[0], real_target_position[1], c='red', label='Real Target Position')
    ax1.scatter(agent_position[0], agent_position[1], c='blue', label='Agent Position')
    ax1.set_title('2D Probability Density Function of Target Position')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.legend()

    # Add debug prints
    print(f"Real target position: {real_target_position}")
    print(f"Initial estimate (a): {a}")
    print(f"Current EKF estimate: {ekf.a_k_1}")
    print(f"EKF covariance diagonal: {np.diag(ekf.sigma_k_1)}")
    
    # Test the peak location
    peak_prob = ekf.p(ekf.a_k_1.reshape(1, -1))[0]
    print(f"Probability at EKF estimate: {peak_prob}")
    
    
    # 3D plot
    ax2 = plt.subplot(122, projection='3d')
    X, Y = np.meshgrid(np.linspace(0, L1, N_POINTS), np.linspace(0, L2, N_POINTS))
    surf = ax2.plot_surface(X, Y, p_values.T, cmap='viridis', alpha=0.8)
    ax2.scatter(real_target_position[0], real_target_position[1], 
               ekf.p(real_target_position.reshape(1, -1))[0], 
               c='red', s=100, label='Real Target Position')
    ax2.scatter(agent_position[0], agent_position[1], 0, c='blue', s=100, label='Agent Position')
    ax2.set_title('3D Probability Density Function of Target Position')
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    ax2.set_zlabel('Probability Density')
    ax2.legend()

    ax1.scatter(ekf.a_k_1[0], ekf.a_k_1[1], c='green', marker='x', s=100, label='EKF Estimate')
    ax2.scatter(ekf.a_k_1[0], ekf.a_k_1[1], peak_prob, c='green', marker='x', s=100, label='EKF Estimate')
    
    plt.tight_layout()
    plt.show()










# if __name__ == "__main__VectorizedTimingTests":
    # Add the EKF estimate to the plot
#     # test the vectorized measurement model H vs the original one
#     mm = MeasurementModelVectorized()
#     mm_orig = MeasurementModel()

#     # single target, single agent
#     a = np.array([0.5, 0.5, 0])
#     x = np.array([0.4, 0.6, 0.1])
    
#     H_vec = mm.H(a, x)
#     H_orig = mm_orig.H(a, x)
    
#     assert np.allclose(H_vec, H_orig), "H (vectorized) does not match H (original)"

#     # Use larger test cases for better timing measurement
#     N_TARGETS = 1000  # Increased for better timing resolution
#     N_AGENTS = 500
    
#     # multiple targets, single agent
#     a_large = np.random.rand(N_TARGETS, 3)  # Random target positions
#     x_single = np.array([0.4, 0.6, 0.1])
    
#     # Warm up runs (JIT compilation, cache warming)
#     _ = mm.H(a_large[:10], x_single)
#     for target in a_large[:10]:
#         _ = mm_orig.H(target, x_single)
    
#     # Actual timing with multiple runs
#     N_RUNS = 10
#     vec_times = []
#     orig_times = []
    
#     for _ in range(N_RUNS):
#         start_time = time.perf_counter()
#         H_vec = mm.H(a_large, x_single)
#         vec_times.append(time.perf_counter() - start_time)
        
#         start_time = time.perf_counter()
#         H_orig_list = []
#         for target in a_large:
#             H_orig_list.append(mm_orig.H(target, x_single))
#         H_orig = np.stack(H_orig_list, axis=0)
#         orig_times.append(time.perf_counter() - start_time)
    
#     vec_time_multi = np.mean(vec_times)
#     orig_time_multi = np.mean(orig_times)
    
#     assert H_vec.shape == (N_TARGETS, 2, 3), "H (vectorized) shape mismatch for multiple targets, single agent"
#     assert np.allclose(H_vec, H_orig), "H (vectorized) does not match H (original) for multiple targets"

#     # single target, multiple agents
#     a_single = np.array([0.5, 0.5, 0])
#     x_large = np.random.rand(N_AGENTS, 3)
    
#     # Warm up
#     _ = mm.H(a_single, x_large[:10])
#     for agent in x_large[:10]:
#         _ = mm_orig.H(a_single, agent)
    
#     vec_times_agents = []
#     orig_times_agents = []
    
#     for _ in range(N_RUNS):
#         start_time = time.perf_counter()
#         H_vec = mm.H(a_single, x_large)
#         vec_times_agents.append(time.perf_counter() - start_time)
        
#         start_time = time.perf_counter()
#         H_orig_list = []
#         for agent in x_large:
#             H_orig_list.append(mm_orig.H(a_single, agent))
#         H_orig = np.stack(H_orig_list, axis=0)
#         orig_times_agents.append(time.perf_counter() - start_time)
    
#     vec_time_agents = np.mean(vec_times_agents)
#     orig_time_agents = np.mean(orig_times_agents)
    
#     assert H_vec.shape == (N_AGENTS, 2, 3), "H (vectorized) shape mismatch for single target, multiple agents"
#     assert np.allclose(H_vec, H_orig), "H (vectorized) does not match H (original) for multiple agents"

#     print(f"H method timing results ({N_TARGETS} targets, {N_AGENTS} agents, {N_RUNS} runs):")
#     print(f"Multiple targets - Vectorized: {vec_time_multi:.6f}s, Original: {orig_time_multi:.6f}s (speedup: {orig_time_multi/vec_time_multi:.1f}x)")
#     print(f"Multiple agents - Vectorized: {vec_time_agents:.6f}s, Original: {orig_time_agents:.6f}s (speedup: {orig_time_agents/vec_time_agents:.1f}x)")

#     # Similar improvements for Y method tests
#     # Y method tests with larger datasets
#     vec_times_Y = []
#     orig_times_Y = []
    
#     for _ in range(N_RUNS):
#         start_time = time.perf_counter()
#         Y_vec = mm.Y(a_large, x_single)
#         vec_times_Y.append(time.perf_counter() - start_time)
        
#         start_time = time.perf_counter()
#         Y_orig_list = []
#         for target in a_large:
#             Y_orig_list.append(mm_orig.Y(target, x_single))
#         Y_orig = np.stack(Y_orig_list, axis=0)
#         orig_times_Y.append(time.perf_counter() - start_time)
    
#     vec_time_multi_Y = np.mean(vec_times_Y)
#     orig_time_multi_Y = np.mean(orig_times_Y)
    
#     assert Y_vec.shape == (N_TARGETS, 2), "Y (vectorized) shape mismatch for multiple targets, single agent"
#     assert np.allclose(Y_vec, Y_orig), "Y (vectorized) does not match Y (original) for multiple targets"

#     # Single target, multiple agents for Y
#     vec_times_agents_Y = []
#     orig_times_agents_Y = []
    
#     for _ in range(N_RUNS):
#         start_time = time.perf_counter()
#         Y_vec = mm.Y(a_single, x_large)
#         vec_times_agents_Y.append(time.perf_counter() - start_time)
        
#         start_time = time.perf_counter()
#         Y_orig_list = []
#         for agent in x_large:
#             Y_orig_list.append(mm_orig.Y(a_single, agent))
#         Y_orig = np.stack(Y_orig_list, axis=0)
#         orig_times_agents_Y.append(time.perf_counter() - start_time)
    
#     vec_time_agents_Y = np.mean(vec_times_agents_Y)
#     orig_time_agents_Y = np.mean(orig_times_agents_Y)
    
#     assert Y_vec.shape == (N_AGENTS, 2), "Y (vectorized) shape mismatch for single target, multiple agents"
#     assert np.allclose(Y_vec, Y_orig), "Y (vectorized) does not match Y (original) for multiple agents"

#     print(f"\nY method timing results ({N_TARGETS} targets, {N_AGENTS} agents, {N_RUNS} runs):")
#     print(f"Multiple targets - Vectorized: {vec_time_multi_Y:.6f}s, Original: {orig_time_multi_Y:.6f}s (speedup: {orig_time_multi_Y/vec_time_multi_Y:.1f}x)")
#     print(f"Multiple agents - Vectorized: {vec_time_agents_Y:.6f}s, Original: {orig_time_agents_Y:.6f}s (speedup: {orig_time_agents_Y/vec_time_agents_Y:.1f}x)")

#     print("\nAll tests passed!")



