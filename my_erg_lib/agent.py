from my_erg_lib.basis import Basis
import numpy as np
from my_erg_lib.model_dynamics import SingleIntegrator, DoubleIntegrator, Quadcopter

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
