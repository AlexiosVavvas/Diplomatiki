from my_erg_lib.basis import Basis, ReconstructedPhi, ReconstructedPhiFromCk
import numpy as np
import matplotlib.pyplot as plt

import cProfile
import pstats
from pstats import SortKey

def phiExample(s, L1=1.0, L2=1.0):
    # Complex function with multiple peaks, valleys, and non-linearities
    x, y = s[0], s[1]
    
    # Multiple Gaussian bumps
    # Generate random bump positions within the L1, L2 boundaries
    bump_positions = [
        (0.2 * L1, 0.3 * L2), 
        (0.7 * L1, 0.8 * L2), 
        (0.5 * L1, 0.1 * L2), 
        (0.9 * L1, 0.5 * L2)
    ]
    bump_heights = [3, 4, 2, 5]
    bump_widths = [30, 40, 25, 35]
    
    bumps = 0
    for i in range(len(bump_positions)):
        pos_x, pos_y = bump_positions[i]
        height = bump_heights[i]
        width = bump_widths[i]
        bumps += height * np.exp(-width * ((x-pos_x)**2 + (y-pos_y)**2))
    
    # Sinusoidal variations
    # waves = 2 * np.sin(8 * np.pi * x) * np.cos(6 * np.pi * y)
    
    # Polynomial trend
    # trend = (x - 0.4)**2 * (y - 0.6)**2 * 5
    
    # Sharp ridge
    # ridge = 3 * np.exp(-100 * (x - y)**2)
    
    # Combine all components
    return bumps + 2 #+ waves + trend + ridge

def plotPhi(agent, phi_new, x_traj=None):
    phi_old = agent.basis.phi

    x1 = np.linspace(0, agent.L1, 50)
    x2 = np.linspace(0, agent.L2, 50)

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
    im1 = ax1.imshow(Z_original, extent=(0, agent.L1, 0, agent.L2), origin='lower', cmap=cm.viridis)
    ax1.set_title('Original Function')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_aspect('auto')
    plt.colorbar(im1, ax=ax1, label='Function Value')

    ax2 = fig.add_subplot(122)
    im2 = ax2.imshow(Z_reconstructed, extent=(0, agent.L1, 0, agent.L2), origin='lower', cmap=cm.viridis)
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
    from my_erg_lib.agent import Agent
    from my_erg_lib.model_dynamics import DoubleIntegrator, SingleIntegrator
    from my_erg_lib.ergodic_controllers import DecentralisedErgodicController
    import time

    def uFunc(x, t):
        #  Potential force pussing away from the boundaries [0xL1=2, 0xL2=2]
        pass 
 

    # Generate Agent and connect to an ergodic controller object
    agent = Agent(L1=2.0, L2=2.0, Kmax=5, 
                  dynamics_model=DoubleIntegrator(dt=0.005), phi=lambda s: phiExample(s, L1=2.0, L2=2.0))
                #   dynamics_model=SingleIntegrator(dt=0.005), phi=lambda s: 2)
    agent.erg_c = DecentralisedErgodicController(agent, uNominal=None, Q=2, uLimits=[[-10, 10], [-10, 10]],
                                                 T_sampling=0.1, T_horizon=0.3, deltaT_erg=0.3*10,
                                                 barrier_weight=1e3, barrier_eps=0.1)
    
    x0 = np.array([0.5, 0.4, 0, 0])  # Initial state
    agent.model.reset(x0)  # Reset the model to the initial state
    
    states_list = [x0.copy()]  
    t_list = [0]  # Time vector
    u_list = [np.zeros((2,))]  # Control action list
    erg_cost_list = []

    ti = t_list[0]; ti_indx = 0
    Ts_iter = int(agent.erg_c.Ts / agent.model.dt)  # Number of iterations per sampling time
    
    # Initialize timing variables
    initial_time = time.time()
    last_iter_time = time.time()
    delta_time = 1
    
    i = 0; IMAX = 4000
    while i < IMAX:
        # If multiple of Ts, calculate ergodic action
        if i % Ts_iter == 0:
            ti = t_list[i]; ti_indx = i
            # Calculate ergodic control for the sample step
            us, tau, lamda_dur, erg_cost = agent.erg_c.calcNextActionTriplet(t_list[i])
            erg_cost_list.append(erg_cost)
            if i % 1 == 0:
                print(f"Ergodic cost: {erg_cost:.3f} \t i: {i}/{IMAX} \t perc: {i/IMAX:.2%} \t dt/Ts: {delta_time/agent.erg_c.Ts:.2f}\t remaining: {delta_time * (IMAX-i)/Ts_iter:.0f} s\t elapsed: {time.time()-initial_time:.1f} s ({time.time()-initial_time + delta_time * (IMAX-i)/Ts_iter:.0f} s)")
                # Debug print if agent inside boundaries
                agent.withinBounds(agent.model.state[:2])
            # Update the action mask
            agent.erg_c.updateActionMask(ti, us, tau, lamda_dur)
        
        # Continue with simulation of agent
        us_ = agent.erg_c.ustar_mask[i - ti_indx]
        
        if us_.all() != 0:
            # Apply the control action to the agent's model
            u = us_
        elif agent.erg_c.uNominal is not None:
            # If no ergodic control is available, use the nominal control
            u_nom = agent.erg_c.uNominal(agent.model.state, t_list[i])
            # Apply the nominal control action to the agent's model
            u = u_nom
        else:
            # If no ergodic control and no nominal control, just step the model with zero control
            u = np.zeros((2,))
        
        agent.model.step(u)  # Step the model with the control action
        agent.erg_c.past_states_buffer.push(agent.model.state.copy()[:2])  # Store the state in the buffer

        u_list.append(u.copy())
        states_list.append(agent.model.state.copy())

        t_list.append(t_list[i] + agent.model.dt)
        delta_erg_cost = abs(erg_cost_list[-1] - erg_cost_list[-2]) if len(erg_cost_list) > 1 else 10
        
        # Calculate delta time for this iteration
        current_time = time.time()
        # delta_time is for when we calculated ergodic control. Otherwise we dont care, its fast
        delta_time = current_time - last_iter_time if (i%Ts_iter == 0) else delta_time
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
    plotPhi(agent, phi_rec, x_traj=None)
    # plt.show()
    
    ck_ = agent.basis.calcCkCoeff(states_list, x_buffer=None, ti=ti, T=agent.erg_c.T)
    # ck_ = agent.basis.calcCkCoeff(agent.erg_c.past_states_buffer.get(), x_buffer=None, ti=ti, T=agent.erg_c.T)
    phi_rec_from_ck = ReconstructedPhiFromCk(agent.basis, ck_)
    plotPhi(agent, phi_rec_from_ck, x_traj=states_list)
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