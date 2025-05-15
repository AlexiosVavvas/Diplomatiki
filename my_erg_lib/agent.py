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

        # Initialize the basis object
        self.basis = Basis(L1, L2, Kmax, phi)


    def setPhi(self, phi=None):
        """Set the target distribution."""
        self._phi = phi if phi is not None else lambda s: 2  # Default to constant 0 function if not provided
        self.basis.phi = self._phi


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
