from my_erg_lib.basis import Basis
import numpy as np
from my_erg_lib.model_dynamics import SingleIntegrator, DoubleIntegrator, Quadcopter
from my_erg_lib.eid import Sensor, EKF

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
        self.sensor = Sensor(sensor_range=0.2)                  # TODO: Be able to adjust those parameters from outside
        self.real_target_position = np.array([0.5, 0.5, 0])     # Real target position (Ground Truth) # TODO: Maybe take it from an env?
        self.a = np.array([0.7, 0.4, 0])                        # Current target position estimate # TODO: When to update it?
        self.ekf = EKF(a_init = self.a,
                       sigma_init = np.eye(3)*1)                # TODO: Why does it look like an ellipse and not a circle at first?

        # Initialise obstacle list
        self.obstacle_list = []

        # Initialize the basis object
        self.basis = Basis(L1, L2, Kmax, phi_=phi, precalc_phik_coeff=False)

    def modifedPhiForObstacles(self, phi, obs_to_exclude=None):
        '''
        Wrapper function modifying the original phi function to take into account the obstacles
        Zeros out the phi function in the obstacle area
        '''
        assert callable(phi), "phi must be a callable function."
        if obs_to_exclude == "All":
            phi.obs_list = []
            return phi

        obs_list = [obs for obs in self.obstacle_list if obs.name_id not in obs_to_exclude] if obs_to_exclude != None else self.obstacle_list
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

    def updateEIDphiFunction(self):

        def phi(x):

            # Append the self.model.state[2] to the state vector
            # TODO: Check dimensions before and after appending
            x = np.append(x, self.model.state[2])

            # Lets get a measurement from the environment: z -> [azimuth, elevation]
            z = self.sensor.getMeasurement(self.real_target_position, x)    # TODO: Maybe we should use the target estimate instead of the real target position here
            if z is None:
                # If the target is out of range, return 0
                return 0
            
            # Using this measurement, we can update the target position estimate using the EKF
            a, sigma = self.ekf.update(xk=x, zk=z, update_internal_state=False)

            # Fisher Information Matrix (I)
            M = 3 # Number of target estimation states
            mu = self.ekf.measurement_model.num_of_measurements # Number of measurements
            def I(a):
                H = self.ekf.measurement_model.H(a, x) # Jacobian Y_a | x
                sigma = self.sensor.R # Sensor noise covariance matrix
                
                I_ = np.zeros((M, M)) # FIM: Fisher Information Matrix
                for i in range(M):
                    for j in range(M):
                        I_[i, j] = float(H[:, i].reshape((mu, 1)).T @ np.linalg.inv(sigma) @ H[:, j].reshape((mu, 1)))
                return I_
            
            # Expected Information Matrix (Φ)

            # Now we have to integrate the product of I(a) * p(a). We will use Gauss for speed
            # Get Gauss-Legendre quadrature points and weights
            NUM_GAUSS_POINTS = 4 # TODO: Handle this better
            a1_points, a1_weights = np.polynomial.legendre.leggauss(NUM_GAUSS_POINTS)
            a2_points, a2_weights = np.polynomial.legendre.leggauss(NUM_GAUSS_POINTS)
            # Transform from [-1,1] to [0,L1] and [0,L2]
            a1_points  = 0.5 * self.L1 * (a1_points + 1)
            a2_points  = 0.5 * self.L2 * (a2_points + 1)
            a1_weights = 0.5 * self.L1 * a1_weights
            a2_weights = 0.5 * self.L2 * a2_weights
            # Precalculate I for every a1, a2 and put it in a dictionary
            I_ = {}
            for k in range(NUM_GAUSS_POINTS):
                for l in range(NUM_GAUSS_POINTS):
                    a1, a2 = a1_points[k], a2_points[l]
                    I_[a1, a2] = I([a1, a2, 0])

            FI = np.zeros((M, M))
            for i in range(M):
                for j in range(M):
                    # Compute the integral

                    result = 0.0
                    for k in range(NUM_GAUSS_POINTS):
                        for l in range(NUM_GAUSS_POINTS):
                            result += a1_weights[k] * a2_weights[l] * I_[a1_points[k], a2_points[l]][i, j] * self.ekf.p([a1_points[k], a2_points[l], 0])
                    
                    FI[i, j] = result # TODO: Maybe .copy() here?
            # print det 
            det = np.linalg.det(FI)
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

        # Precalculate the phi coefficients
        # self.basis.precalcAllPhiK() # TODO: Check if this is needed

        # Handle obstacle exclusion from the function above
        self.basis.phi = self.modifedPhiForObstacles(self.basis.phi, obs_to_exclude=obs_list)



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
