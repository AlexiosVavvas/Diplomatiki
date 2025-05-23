import numpy as np

class MeasurementModel:
    # TODO: Make it actually modular. Have one measuring both beta, phi and one only beta for speed etc.
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
            beta = np.arctan2(xt - xq, yt - yq) if (yt - yq) != 0 else np.pi / 2
            
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
        # Jacobian of the measurement function
        Y1_a1 = (yt - yq)/((xq-xt)**2 + (yq-yt)**2)
        Y1_a2 = (xq - xt)/((xq-xt)**2 + (yq-yt)**2)
        Y1_a3 = 0
        Y2_a1 = ((xq-xt)*(zq-zt))/(np.sqrt((xq-xt)**2 + (yq-yt)**2) * ((xq-xt)**2 + (yq-yt)**2 + (zq-zt)**2))
        Y2_a2 = ((yq-yt)*(zq-zt))/(np.sqrt((xq-xt)**2 + (yq-yt)**2) * ((xq-xt)**2 + (yq-yt)**2 + (zq-zt)**2))    
        Y2_a3 = - 1 / (np.sqrt((xq-xt)**2 + (yq-yt)**2) * (1 + ((zq-zt)**2)/((xq-xt)**2 + (yq-yt)**2)))

        return np.array([[Y1_a1, Y1_a2, Y1_a3],
                         [Y2_a1, Y2_a2, Y2_a3]])
    


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
        self.R = np.eye(self.mu) * 0.035 if R is None else R

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
        self.sigma_k_1 = np.eye(self.M) * 1  if sigma_init is None else sigma_init

        # Measurement Model
        self.measurement_model = MeasurementModel()
        self.mu = self.measurement_model.mu  # Number of measurements

        # Estimated Sensor Noise covariance
        assert R is None or (R.shape[0] == self.mu and R.shape[1] == self.mu), f"R must be a μxμ=({self.mu}x{self.mu}) matrix."
        self.R = np.eye(self.mu) * 0.035 if R is None else R

        # Process Noise covariance (Not used in the paper, but makes the uncertainty grow especially when no measurements are available)
        assert Q is None or (Q.shape[0] == self.M and Q.shape[1] == self.M), f"Q must be a MxM=({self.M}x{self.M}) matrix."
        self.Q = np.eye(self.M) * 1e-4 if Q is None else Q


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
        sigmaK_k_1 += self.Q  if zk is None else 0  # TODO: Maybe ask about it. It seems logical though

        # Compute expected measurement
        zk_hat = self.measurement_model.Y(ak_k_1, xk)  # Y(a_k|k-1)

        # Linearize the measurement model (get Jacobian)
        Hk = self.measurement_model.H(ak_k_1, xk)

        # Compute Kalman Gain
        S = Hk @ sigmaK_k_1 @ Hk.T + self.R
        S_inv = np.linalg.solve(S, np.eye(S.shape[0]))
        Kk = sigmaK_k_1 @ Hk.T @ S_inv

        # Update state 
        ak = ak_k_1 + Kk @ (zk - zk_hat) if zk is not None else ak_k_1
        ak = np.clip(ak, self.a_limits[:, 0], self.a_limits[:, 1])

        # Update covariance matrix
        sigmaK = (np.eye(len(sigmaK_k_1)) - Kk @ Hk) @ sigmaK_k_1

        # Update internal state
        if update_internal_state:
            self.a_k_1 = ak
            self.sigma_k_1 = sigmaK

        return ak, sigmaK


    def p(self, a):
        """
        Probability density function of the target state
        Mutlivariate Gaussian distribution
            p(a_) = N(a_, a, Σ)
            - a_: Evaluation Position
            - a: Target state
            - Σ: Covariance matrix

        """
        # Probability density function of the target state
        return 1 / ((2 * np.pi)**(self.M/2) * np.linalg.det(self.sigma_k_1) ** 0.5) * np.exp(-0.5 * (a - self.a_k_1).T @ np.linalg.inv(self.sigma_k_1) @ (a - self.a_k_1))

