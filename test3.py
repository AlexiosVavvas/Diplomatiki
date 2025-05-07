from agent import Agent
from model_dynamics import SingleIntegrator
from ergodic_controllers import DecentralisedErgodicController
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def phiExample(s):
    # Complex function with multiple peaks, valleys, and non-linearities
    x, y = s[0], s[1]
    
    # Multiple Gaussian bumps
    bumps = 3 * np.exp(-30 * ((x-0.2)**2 + (y-0.3)**2)) + \
            4 * np.exp(-40 * ((x-0.7)**2 + (y-0.8)**2)) + \
            2 * np.exp(-25 * ((x-0.5)**2 + (y-0.1)**2)) + \
            5 * np.exp(-35 * ((x-0.9)**2 + (y-0.5)**2))
    
    # Sinusoidal variations
    # waves = 2 * np.sin(8 * np.pi * x) * np.cos(6 * np.pi * y)
    
    # Polynomial trend
    # trend = (x - 0.4)**2 * (y - 0.6)**2 * 5
    
    # Sharp ridge
    # ridge = 3 * np.exp(-100 * (x - y)**2)
    
    # Combine all components
    return bumps + 2


def uFunc(x, t):
    # Example control function: a simple sinusoidal control
    u_ = np.ones((2,))
    u_[0] = np.random.uniform(-1, 1)  # Random control for x1
    u_[1] = np.random.uniform(-1, 1)  # Random control for x2
    return u_ * 0.1

# Generate Agent and connect to an ergodic controller object
agent = Agent(L1=1.0, L2=1.0, Kmax=5, 
            #   dynamics_model=SingleIntegrator(), phi=phiExample)
                dynamics_model=SingleIntegrator(), phi=lambda s: 2)
agent.erg_c = DecentralisedErgodicController(agent, uNominal=None, T_sampling=0.01, T_horizon=0.05, Q=1)

# Set time intervals
agent.model.dt = 0.004  # Time step size

x1 = np.linspace(0, 1, 100)
x2 = np.linspace(0, 1, 100)

for k1 in range(agent.basis.Kmax+1):
    for k2 in range(agent.basis.Kmax+1):
        fkdx = np.zeros((len(x1), len(x2)))
        for i, x1_val in enumerate(x1):
            for j, x2_val in enumerate(x2):
                hk = agent.basis.calcHk(k1, k2)
                fkdx[i, j] = agent.basis.dFk_dx([x1_val, x2_val], k1, k2, hk)[1]

        plt.figure()
        plt.imshow(fkdx, extent=(0, 1, 0, 1), origin='lower', cmap='viridis')
        plt.title(f'Fk_dx for k1={k1}, k2={k2}')
        plt.colorbar(label='Fk_dx Value')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.show()

