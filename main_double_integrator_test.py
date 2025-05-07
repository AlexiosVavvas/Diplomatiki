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
    # waves = 2 * np.sin(8 * np.pi * x) * np.cos(6 * np.pi * y)
    
    # Polynomial trend
    # trend = (x - 0.4)**2 * (y - 0.6)**2 * 5
    
    # Sharp ridge
    # ridge = 3 * np.exp(-100 * (x - y)**2)
    
    # Combine all components
    return bumps + 2 #+ waves + trend + ridge

def plotPhi(agent_basis, phi_new, x_traj=None):
    phi_old = agent_basis.phi

    x1 = np.linspace(0, 1, 50)
    x2 = np.linspace(0, 1, 50)

    # Plot both in a 1x2 matplotlib figure as heatmap colors
    import matplotlib.pyplot as plt
    from matplotlib import cm

    Z_original = np.zeros((len(x1), len(x2)))
    Z_reconstructed = np.zeros((len(x1), len(x2)))

    for i in range(len(x1)):
        for j in range(len(x2)):
            Z_original[i, j] = phi_old([x1[i], x2[j]])
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

# -----------------------------------------------------------------------------------
def main():
    from agent import Agent
    from model_dynamics import SingleIntegrator, DoubleIntegrator
    from ergodic_controllers import DecentralisedErgodicController
    import time

    def uFunc(x, t):
        # Example control function: a simple sinusoidal control
        u_ = np.ones((2,))
        u_[0] = np.random.uniform(-1, 1)  # Random control for x1
        u_[1] = np.random.uniform(-1, 1)  # Random control for x2
        return u_ * 0.1

    # Generate Agent and connect to an ergodic controller object
    agent = Agent(L1=1.0, L2=1.0, Kmax=4, 
                  dynamics_model=SingleIntegrator(dt=0.005), phi=phiExample)
                #   dynamics_model=SingleIntegrator(dt=0.005), phi=lambda s: 2)
    agent.erg_c = DecentralisedErgodicController(agent, uNominal=None, T_sampling=0.02, T_horizon=0.3, Q=1, deltaT_erg=0.9)
    
    x0 = np.array([0.1, 0.9])  # Initial state
    agent.model.reset(x0)  # Reset the model to the initial state
    
    states_list = [x0.copy()]  
    t_list = [0]  # Time vector
    u_list = [np.zeros((2,))]  # Control action list
    erg_cost_list = []
    delta_erg_cost = 10  # Threshold for ergodic cost

    ti = t_list[0]; ti_indx = 0
    
    TMAX = 0.5 # [s]
    TMIN = 0.5
    # Initialize timing variables
    last_iter_time = time.time()
    delta_time = 1
    
    i = 0; IMAX = 300
    while i < IMAX:
    # while delta_erg_cost > 1e-5 and t_list[i] < TMAX:
        # Start timing this iteration
        iter_start_time = time.time()
        
        if t_list[i] % agent.erg_c.Ts < 0.01:
            ti = t_list[i]; ti_indx = i
            # Calculate ergodic control for the sample step
            us, tau, lamda_dur, erg_cost = agent.erg_c.calcNextActionTriplet(t_list[i])
            erg_cost_list.append(erg_cost); print(f"Ergodic cost: {erg_cost:.3f} \t i: {i}/{IMAX} \t perc: {i/IMAX:.2%} \t dt/Ts: {delta_time/agent.erg_c.Ts:.2f}")
            # Update the action mask
            agent.erg_c.updateActionMask(ti, us, tau, lamda_dur)
        
        us_ = agent.erg_c.ustar_mask[i - ti_indx]
        
        if us_.all() != 0:
            # Apply the control action to the agent's model
            u = us_
            # print(f"Stepping with ERGODIC control action: {u}")
        elif agent.erg_c.uNominal is not None:
            # If no ergodic control is available, use the nominal control
            u_nom = agent.erg_c.uNominal(agent.model.state, t_list[i])
            # Apply the nominal control action to the agent's model
            u = u_nom
            # print(f"Stepping with NOMINAL control action: {u}")
        else:
            # If no ergodic control and no nominal control, just step the model with zero control
            u = np.zeros((2,))
            # print("Stepping with ZERO control action.")
        
        u = u.clip(-3, 3)
        agent.model.step(u)  # Step the model with the control action
        agent.erg_c.past_states_buffer.push(agent.model.state.copy())  # Store the state in the buffer

        u_list.append(u.copy())
        states_list.append(agent.model.state.copy())

        t_list.append(t_list[i] + agent.model.dt)
        delta_erg_cost = abs(erg_cost_list[-1] - erg_cost_list[-2]) if len(erg_cost_list) > 1 else 10
        
        # Calculate delta time for this iteration
        current_time = time.time()
        delta_time = current_time - last_iter_time
        last_iter_time = current_time
        
        i += 1

    states_list = np.array(states_list)
    u_list = np.array(u_list)

    # Visualize the trajectory
    plt.figure(figsize=(8, 6))
    plt.plot(t_list, states_list[:, 0], 'b-', label='state 0')
    plt.plot(t_list, states_list[:, 1], 'r-', label='state 1')
    plt.legend()
    plt.grid()
    
    plt.figure(figsize=(8, 6))
    plt.plot(t_list, u_list[:, 0], 'g-', label='control 1')
    plt.plot(t_list, u_list[:, 1], 'r-', label='control 2')
    plt.legend()
    plt.grid()

    # ergodic cost
    plt.figure(figsize=(8, 6))
    plt.plot(erg_cost_list, 'g-', label='Ergodic Cost')
    plt.legend()
    plt.grid()

    # vis traj
    phi_rec = ReconstructedPhi(agent.basis, precalc_phik=False)
    plotPhi(agent.basis, phi_rec, x_traj=states_list)
    # plt.show()
    
    ck_ = agent.basis.calcCkCoeff(states_list, x_buffer=None, ti=ti, T=agent.erg_c.T)
    # ck_ = agent.basis.calcCkCoeff(agent.erg_c.past_states_buffer.get(), x_buffer=None, ti=ti, T=agent.erg_c.T)
    phi_rec_from_ck = ReconstructedPhiFromCk(agent.basis, ck_)
    plotPhi(agent.basis, phi_rec_from_ck, x_traj=states_list)
    plt.show()






# -----------------------------------------------------------------------------------

# Add at the end of file
if __name__ == "__main__":
    # main()
    # Profile the main function
    profiler = cProfile.Profile()
    profiler.enable()
    
    main()
    print(f"\n\n\n\nProfiling {main.__name__}()")
    
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats(SortKey.CUMULATIVE)
    # Filter to only show functions from the current file
    stats.print_stats("agent.py|basis.py|target_distribution.py|model_dynamics.py|ergodic_controllers.py")  # Show only your modules