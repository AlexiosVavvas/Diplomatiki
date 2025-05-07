from agent import Agent
from model_dynamics import SingleIntegrator
from ergodic_controllers import DecentralisedErgodicController
import numpy as np

agent = Agent(L1=1.0, L2=1.0, Kmax=5, dynamics_model=SingleIntegrator(), phi=None)

def uFunc(x, t):
    # Example control function: a simple sinusoidal control
    u_ = np.zeros((2,))
    u_[0] = np.random.uniform(-1, 1)  # Random control for x1
    u_[1] = np.random.uniform(-1, 1)  # Random control for x2
    return u_ * 0


erg_c = DecentralisedErgodicController(agent, phi=None, num_of_agents=1, R=np.eye(2), uNominal=None)
erg_c.calcNextActionTriplet(ti=0, T=1)