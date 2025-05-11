import numpy as np

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

# Function to be used for phi with specific L1 and L2 values
def phi_func(s):
    return phiExample(s, L1=2.0, L2=2.0)

# -----------------------------------------------------------------------------------
def main():
    from my_erg_lib.agent import Agent
    from my_erg_lib.model_dynamics import DoubleIntegrator, SingleIntegrator
    from my_erg_lib.ergodic_controllers import DecentralisedErgodicController
    from my_erg_lib.basis import ReconstructedPhi, ReconstructedPhiFromCk
    import matplotlib.pyplot as plt
    from vis import plotPhi
    import time

    # TODO: Seperate actual simulation dt from trajectory prediciton dt. This way we can speed up prediction without affecting actual simulation time.
    # TODO: Something is wrong with the effect of the mass. Check it out
    # Generate Agent and connect to an ergodic controller object
    agent = Agent(L1=2.0, L2=2.0, Kmax=3, 
                #   dynamics_model=DoubleIntegrator(dt=0.005), phi=lambda s: 2,
                  dynamics_model=DoubleIntegrator(dt=0.005, mass=1), phi=phi_func,
                  x0=[0.5, 0.4, 0, 0])
    UMAX = 2
    agent.erg_c = DecentralisedErgodicController(agent, uNominal=None, Q=2, uLimits=[[-UMAX, UMAX], [-UMAX, UMAX]],
                                                 T_sampling=0.1, T_horizon=0.15, deltaT_erg=0.3*30,
                                                 barrier_weight=100, barrier_eps=0.4, barrier_pow=2)
    # Lists to store for plotting
    states_list = [agent.model.state.copy()]  
    t_list = [0]  # Time vector
    u_list = [np.zeros((2,))]  # Control action list
    u_before = np.zeros((2,))  # Previous control action
    erg_cost_list = []

    ti = t_list[0]; ti_indx = 0
    Ts_iter = int(agent.erg_c.Ts / agent.model.dt)  # Number of iterations per sampling time
    
    # Initialize timing variables
    initial_time = time.time()
    last_iter_time = time.time()
    delta_time = 1
    
    i = 0; IMAX = 20e3
    phi_3_ = ReconstructedPhi(agent.basis, precalc_phik=False)
    while i < IMAX:
        # If multiple of Ts, calculate ergodic action
        if i % Ts_iter == 0:
            ti = t_list[i]; ti_indx = i
            # Calculate ergodic control for the sample step
            us, tau, lamda_dur, erg_cost = agent.erg_c.calcNextActionTriplet(t_list[i])
            erg_cost_list.append(erg_cost)
            if i % 1 == 0:
                print(f"ti = {ti:.2f} s\t Erg cost: {erg_cost:.2f} \t i: {i}/{IMAX:.0f} \t perc: {i/IMAX:.2%} \t dt/Ts: {delta_time/agent.erg_c.Ts:.2f}\t remaining: {delta_time * (IMAX-i)/Ts_iter:.0f} s\t elapsed: {time.time()-initial_time:.1f} s ({time.time()-initial_time + delta_time * (IMAX-i)/Ts_iter:.0f} s)")
                # Debug print if agent inside boundaries
                agent.withinBounds(agent.model.state[:2])
            # Update the action mask
            agent.erg_c.updateActionMask(ti, us, tau, lamda_dur)

            
            # Simulation saving file
            # traj = agent.simulateForward(x0=agent.model.state, ti=ti, udef=agent.erg_c.uNominal, T=agent.erg_c.T)
            # erg_traj = traj[:, :2] # Only take the ergodic dimensions
            # ck_ = agent.basis.calcCkCoeff(erg_traj, x_buffer=agent.erg_c.past_states_buffer.get() ,ti=ti, T=agent.erg_c.T)
            # phi_rec_from_ck = ReconstructedPhiFromCk(agent.basis, ck_)
            # plotPhi2(agent, phi_rec_from_ck=phi_rec_from_ck, phi_rec_from_agent=phi_3_, all_traj=states_list)
            # plt.savefig(f"images/phi_{ti:.2f}.png")
            # plt.close()
        
        # Continue with simulation of agent
        us_ = agent.erg_c.ustar_mask[i - ti_indx]
        
        if us_.all() != 0:
            # Apply the control action to the agent's model
            u = us_
        else:
            # If no ergodic control is available, use the nominal control
            u = agent.erg_c.uNominal(agent.model.state, t_list[i])
        
        # Lets smooth out with the previous control action
        a = 0.9
        u = a * u + (1-a) * u_before  # Smooth the control action
        u_before = u.copy()

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
    for i in range(agent.model.num_of_states):
        plt.plot(t_list, states_list[:, i], label=f"state {i}")
    plt.legend()
    plt.grid()
    
    plt.figure(figsize=(8, 6))
    for i in range(agent.model.num_of_inputs):
        plt.plot(t_list, u_list[:, i], label=f"control {i}")
    plt.legend()
    plt.grid()

    # ergodic cost
    plt.figure(figsize=(8, 6))
    plt.plot(erg_cost_list, 'g-', label='Ergodic Cost')
    plt.legend()
    plt.grid()

    # vis traj
    ck_ = agent.basis.calcCkCoeff(states_list, x_buffer=None, ti=ti, T=agent.erg_c.T)
    # ck_ = agent.basis.calcCkCoeff(agent.erg_c.past_states_buffer.get(), x_buffer=None, ti=ti, T=agent.erg_c.T)
    phi_rec_from_ck = ReconstructedPhiFromCk(agent.basis, ck_)
    phi_rec = ReconstructedPhi(agent.basis, precalc_phik=False)
    plotPhi(agent, phi_rec_from_ck=phi_rec_from_ck, phi_rec_from_agent=phi_rec, all_traj=states_list)

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
    stats.print_stats("agent.py|basis.py|model_dynamics.py|ergodic_controllers.py|barrier.py|replay_buffer.py")  # Show only your modules