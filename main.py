from basis import Basis, ReconstructedPhi, ReconstructedPhiFromCk
import numpy as np
import matplotlib.pyplot as plt

import cProfile
import pstats
from pstats import SortKey

def phiExample(s):
    # Complex function with multiple peaks, valleys, and non-linearities
    x, y = s[0], s[1]
    
    # Multiple Gaussian bumps
    bumps = 3 * np.exp(-30 * ((x-0.2)**2 + (y-0.3)**2)) + \
            4 * np.exp(-40 * ((x-0.7)**2 + (y-0.8)**2)) + \
            2 * np.exp(-25 * ((x-0.5)**2 + (y-0.1)**2)) + \
            5 * np.exp(-35 * ((x-0.9)**2 + (y-0.5)**2))
    
    # Sinusoidal variations
    waves = 2 * np.sin(8 * np.pi * x) * np.cos(6 * np.pi * y)
    
    # Polynomial trend
    trend = (x - 0.4)**2 * (y - 0.6)**2 * 5
    
    # Sharp ridge
    ridge = 3 * np.exp(-100 * (x - y)**2)
    
    # Combine all components
    return bumps + waves + trend + ridge

def plotPhi(base, phi_new, x_traj=None):
    x1 = np.linspace(0, 1, 50)
    x2 = np.linspace(0, 1, 50)

    # Plot both in a 1x2 matplotlib figure as heatmap colors
    import matplotlib.pyplot as plt
    from matplotlib import cm

    Z_original = np.zeros((len(x1), len(x2)))
    Z_reconstructed = np.zeros((len(x1), len(x2)))

    for i in range(len(x1)):
        for j in range(len(x2)):
            Z_original[i, j] = phiExample([x1[i], x2[j]])
            Z_reconstructed[i, j] = phi_new([x1[i], x2[j]])
        
    fig = plt.figure(figsize=(12, 6))
    # Not 3d, just 2d with imshow color
    ax1 = fig.add_subplot(121)
    im1 = ax1.imshow(Z_original, extent=(0, 1, 0, 1), origin='lower', cmap=cm.viridis)
    ax1.set_title('Original Function')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_aspect('auto')
    plt.colorbar(im1, ax=ax1, label='Function Value')

    ax2 = fig.add_subplot(122)
    im2 = ax2.imshow(Z_reconstructed, extent=(0, 1, 0, 1), origin='lower', cmap=cm.viridis)
    ax2.set_title('Reconstructed Function')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_aspect('auto')
    plt.colorbar(im2, ax=ax2, label='Function Value')

    if x_traj is not None:
        ax2.plot(x_traj[:, 1], x_traj[:, 0], 'r-', label='Trajectory')

    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------------
def main():
    from agent import Agent
    from model_dynamics import SingleIntegrator
    from ergodic_controllers import DecentralisedErgodicController

    def uFunc(x, t):
        # Example control function: a simple sinusoidal control
        u_ = np.ones((2,))
        u_[0] = np.random.uniform(-1, 1)  # Random control for x1
        u_[1] = np.random.uniform(-1, 1)  # Random control for x2
        return u_ * 0

    agent = Agent(L1=1.0, L2=1.0, Kmax=5, 
                  dynamics_model=SingleIntegrator(), phi=lambda s: 2, Ts=0.01, T_horizon=0.15)
    agent.erg_c = DecentralisedErgodicController(agent, uNominal=uFunc, T=agent.Ts, Q=1)
    
    # Set time intervals
    agent.model.dt = 0.001  # Time step size
    states = []

    x0 = np.array([0.5, 0.5])  # Initial state
    agent.model.reset(x0)  # Reset the model to the initial state
    t = np.arange(0, 1, 0.01)  # Time vector
    u = []

    from tqdm import tqdm
    for i in tqdm(range(0, len(t))):
        us, _, _ = agent.erg_c.calcNextAction(t[i])
        u.append(us)
        states.append(agent.model.step(us))

    states = np.array(states)
    u = np.array(u)

    # Visualize the trajectory
    plt.figure(figsize=(8, 6))
    plt.plot(t, states[:, 0], 'b-', label='state 0')
    plt.plot(t, states[:, 1], 'r-', label='state 1')
    plt.legend()
    plt.grid()
    
    plt.figure(figsize=(8, 6))
    plt.plot(t, u[:, 0], 'g-', label='control 1')
    plt.plot(t, u[:, 1], 'r-', label='control 2')
    plt.legend()
    plt.grid()
    plt.show()









# -----------------------------------------------------------------------------------

# Add at the end of file
if __name__ == "__main__":
    main()
    # # Profile the main function
    # profiler = cProfile.Profile()
    # profiler.enable()
    
    # main()
    # print(f"\n\n\n\nProfiling {main.__name__}()")
    
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats(SortKey.CUMULATIVE)
    # # Filter to only show functions from the current file
    # stats.print_stats("agent.py|basis.py|target_distribution.py|model_dynamics.py")  # Show only your modules