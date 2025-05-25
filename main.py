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
        (0.9 * L1, 0.5 * L2),
        (0.5 * L1, 0.8 * L2)
    ]
    bump_heights = [3, 4, 2, 5, 8]
    bump_widths = [30, 40, 25, 35, 20]
    
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
    return bumps + 0 #+ waves + trend + ridge

# Function to be used for phi with specific L1 and L2 values
def phi_func(s):
    return phiExample(s, L1=1.0, L2=1.0)

# -----------------------------------------------------------------------------------
def main():
    from my_erg_lib.agent import Agent
    from my_erg_lib.obstacles import Obstacle, ObstacleAvoidanceControllerGenerator
    from my_erg_lib.model_dynamics import SingleIntegrator, DoubleIntegrator, Quadcopter
    from my_erg_lib.ergodic_controllers import DecentralisedErgodicController
    from my_erg_lib.basis import ReconstructedPhi, ReconstructedPhiFromCk
    import matplotlib.pyplot as plt
    import vis
    import time

    # TODO: Something is wrong with the effect of the mass. Check it out (Double Integrator etc)
    # Set up the agent -----------------------------------------------------------------------------
    
    # ===== Dynamics Model =====
    # Double integrator model ----
    # x0 = [0.4, 0.6, 0, 0]
    # u_limits = [[-10, 10], [-10, 10]]
    # model = DoubleIntegrator(dt=0.001)
    # u_nominal = None
    # Q_ = 1
    # PREDICTION_DT = model.dt * 25
    # RELAX_FACTOR = 0.9
    # IMAX = 10e3
    # TS = 0.01; T_H = 0.2; deltaT_ERG = 0.25 * 10
    # BAR_WEIGHT = 0

    # Quadrotor model -----------
    x0 = [0.4, 0.8, 2, 0, 0, 0, 0,  0,  0,  0,  0,  0]
    UP_MTR_LIM = 2         # Motor Upper Limit Thrust in [N]
    LOW_MTR_LIM = -2       # Motor Lower Limit Thrust in [N]
    mtr_limits = [[LOW_MTR_LIM, UP_MTR_LIM], [LOW_MTR_LIM, UP_MTR_LIM], [LOW_MTR_LIM, UP_MTR_LIM], [LOW_MTR_LIM, UP_MTR_LIM]]
    model = Quadcopter(dt=0.001, x0=x0, z_target=2, motor_limits=mtr_limits, zero_out_states=["x", "y", "ψ"])
    TS = 0.1; T_H = 0.25*5; deltaT_ERG = 0.25 * 40
    Q_ = 3 # 1 with phiFunc
    u_limits = model.input_limits
    u_nominal = model.calcLQRcontrol
    PREDICTION_DT = model.dt * 40
    RELAX_FACTOR = 0.3
    IMAX = 100e3
    BAR_WEIGHT = 0 # 50

    # Agent - Ergodic Controller -------------
    # Generate Agent and connect to an ergodic controller object
    agent = Agent(L1=1.0, L2=1.0, Kmax=5, 
                  dynamics_model=model, phi=lambda s: 2, x0=x0) # phi=phi_func
    
    agent.erg_c = DecentralisedErgodicController(agent, uNominal=u_nominal, Q=Q_, uLimits=u_limits,
                                                 T_sampling=TS, T_horizon=T_H, deltaT_erg=deltaT_ERG,
                                                 barrier_weight=BAR_WEIGHT, barrier_eps=0.05, barrier_pow=2)
    
    # Avoiding Obstacles -------------------
    # Add obstacles and another controller to take them into account
    FMAX = 0.25; EPS_M = 0.2
    obs  = [Obstacle(pos=[0.2, 0.2],   dimensions=0.1,        f_max=FMAX, min_dist=0.14, eps_meters=EPS_M, obs_type='circle',    obs_name="Obstacle 1"), 
            Obstacle(pos=[0.66, 0.77], dimensions=0.12,       f_max=FMAX, min_dist=0.16, eps_meters=EPS_M, obs_type='circle',    obs_name="Obstacle 2"), 
            Obstacle(pos=[0.6, 0.5],   dimensions=0.08,       f_max=FMAX, min_dist=0.12, eps_meters=EPS_M, obs_type='circle',    obs_name="Obstacle 3"),
            Obstacle(pos=[0.15, 0.8],  dimensions=[0.2, 0.2], f_max=FMAX, min_dist=0.14, eps_meters=EPS_M, obs_type='rectangle', obs_name="Obstacle 4")]

    agent.erg_c.uNominal += ObstacleAvoidanceControllerGenerator(agent, obs_list=obs, func_name="Obstacles")


    # Avoiding Walls ----------------------
    FMAX = 1; min_dist = 1e-2; EPS_M = 0.49; e_max = agent.L1
    bar  = [Obstacle(pos=[0,        0],   dimensions=[0, +1], f_max=FMAX, min_dist=min_dist, e_max=e_max, eps_meters=EPS_M,  obs_type='wall', obs_name="Bottom Wall"),
            Obstacle(pos=[0, agent.L2],   dimensions=[0, -1], f_max=FMAX, min_dist=min_dist, e_max=e_max, eps_meters=EPS_M,  obs_type='wall', obs_name="Top Wall"   ),
            Obstacle(pos=[0,        0],   dimensions=[+1, 0], f_max=FMAX, min_dist=min_dist, e_max=e_max, eps_meters=EPS_M,  obs_type='wall', obs_name="Left Wall"  ),
            Obstacle(pos=[agent.L1, 0],   dimensions=[-1, 0], f_max=FMAX, min_dist=min_dist, e_max=e_max, eps_meters=EPS_M,  obs_type='wall', obs_name="Right Wall" )]
    
    # Add the obstacle avoidance controller to the ergodic controller
    agent.erg_c.uNominal += ObstacleAvoidanceControllerGenerator(agent, obs_list=bar, func_name="Walls")
    # Print uNominal Status
    print(agent.erg_c.uNominal)

    # Lets now update the phi_function to take into account the obstacles
    agent.basis.phi = agent.modifedPhiForObstacles(agent.basis.phi, obs_to_exclude="All")
    agent.basis.precalcAllPhiK()

    if input("\nVisualise Potential Fields? (y/n): ") == "y":
        vis.visPotentialFields(agent)

    input("Press Enter to continue...")
    # --------------------------------------------------------------------------------------------------
    
    # Lists to store for plotting
    states_list = [agent.model.state.copy()]  
    time_list = [0]  # Time vector
    u_list = [np.zeros((agent.model.num_of_inputs,))]  # Control action list
    u_before = np.zeros((agent.model.num_of_inputs,))  # Previous control action
    erg_cost_list = []
    state_target_list = [agent.model._state_target.copy()] if isinstance(agent.model, Quadcopter) else []  # State target list (only for quads with LQR)
    delta_t_Ts = []
    target_t_pos_list = [] # Contains [ti, x_est, y_est, z_est]
    target_sigma_list = [] # Contains 3x3 sigma matrix for every time step
    draw_plot_flag = False  # Flag that alternates when updating EID to plot

    ti = time_list[0]; ti_indx = 0
    Ts_iter = int(agent.erg_c.Ts / agent.model.dt)  # Number of iterations per sampling time
    
    # Initialize timing variables
    initial_time = time.time()
    last_iter_time = time.time()
    delta_time = 1
    
    i = 0
    while i < IMAX:
        # If multiple of Ts, calculate ergodic action
        if i % Ts_iter == 0:
            ti = time_list[i]; ti_indx = i
            # Calculate ergodic control for the sample step
            us, tau, lamda_dur, erg_cost = agent.erg_c.calcNextActionTriplet(time_list[i], prediction_dt=PREDICTION_DT)

            # change lamda dur only if not quadcopter
            if not isinstance(agent.model, Quadcopter):
                lamda_dur = agent.erg_c.Ts
            erg_cost_list.append(erg_cost)
            delta_t_Ts.append([ti, delta_time / agent.erg_c.Ts])

            if i % 160 == 0:
                def u_str(u):
                    res = "["
                    for j in range(len(u)):
                        res += f"{u[j]:.2f}, "
                    return res[:-2] + "]"
                print(f"ti = {ti:.2f} s\t Erg cost: {erg_cost:.2f} \t i: {i}/{IMAX:.0f} \t perc: {i/IMAX:.2%} \t Δt/Ts: {delta_time/agent.erg_c.Ts:.2f}\t remaining: {delta_time * (IMAX-i)/Ts_iter:.0f} s\t elapsed: {time.time()-initial_time:.1f} s ({time.time()-initial_time + delta_time * (IMAX-i)/Ts_iter:.0f} s)")
                print(f"{agent.model.state_string} \n u = {u_str(us)} \t (tau - ti)/dt = {(tau - ti)/agent.model.dt:.2f} \t lamda_dur = {lamda_dur:.4f} \t lamda/Ts = {lamda_dur/agent.erg_c.Ts:.2%}\n")
            
            # Debug print if agent inside boundaries
            agent.withinBounds(agent.model.state[:2])

            if np.any(np.abs(agent.model.state[:2]) > 50):
                print("--> Agent WAYY out of bounds! Stopping simulation.")
                break
            
            # Update the action mask
            if lamda_dur > 0:
                agent.erg_c.updateActionMask(ti, us, tau, lamda_dur)

            # EKF - Lets update target position estimate -------
            # Make a measurement using the sensor
            z = agent.sensor.getMeasurement(agent.real_target_position, agent.model.state[:3])
            # Update estimate using the EKF
            agent.ekf.update(xk=agent.model.state[:3], zk=z)
            agent.a = agent.ekf.a_k_1
            # Lets update our lists for plotting later
            target_t_pos_list.append([time_list[i], *agent.ekf.a_k_1])
            target_sigma_list.append(agent.ekf.sigma_k_1.copy())
                
            # Simulation saving file ----------------------------
            if draw_plot_flag:
            # if draw_plot_flag or i % 160 == 0:
                x_traj, u_traj, t_traj = agent.model.simulateForward(x0=agent.model.state, ti=ti, udef=agent.erg_c.uNominal, T=agent.erg_c.T, dt=PREDICTION_DT)
                erg_traj = x_traj[:, :2] # Only take the ergodic dimensions
                ck_ = agent.basis.calcCkCoeff(erg_traj, x_buffer=agent.erg_c.past_states_buffer.get() ,ti=ti, T=agent.erg_c.T)
                phi_rec_from_ck = ReconstructedPhiFromCk(agent.basis, ck_)
                print("Plotting phi")
                phi_3_ = ReconstructedPhi(agent.basis, precalc_phik=False)
                vis.plotPhi(agent, phi_rec_from_ck=phi_rec_from_ck, phi_rec_from_agent=phi_3_, all_traj=states_list, grid_res=30)
                plt.savefig(f"images/phiQuadWithObs_{ti:.2f}.png")
                print(f"Saved image to images/phiQuadWithObs_{ti:.2f}.png")
                plt.close()
                draw_plot_flag = False
        
        # Continue with simulation of agent
        us_ = agent.erg_c.ustar_mask[i - ti_indx]
        
        if us_.all() != 0:
            # Apply the control action to the agent's model
            u = us_
        else:
            # If no ergodic control is available, use the nominal control
            u = agent.erg_c.uNominal(agent.model.state, time_list[i])
            
        # Lets smooth out with the previous control action
        u = RELAX_FACTOR * u + (1-RELAX_FACTOR) * u_before  # Smooth the control action
        u_before = u.copy()

        # TODO: Here we should simulate forware for simulation_dt with a dt, instead of stepping. Implemend model simulation function
        agent.model.state = agent.model.step(agent.model.state, u)         # Step the model with the control action
        agent.erg_c.past_states_buffer.push(agent.model.state.copy()[:2])  # Store the state in the buffer

        # Lets update phi(x) if needed
        if i%(Ts_iter*60) == 0:
            t_ = time.time()
            print("Updating phi...")
            agent.updateEIDphiFunction(NUM_GAUSS_POINTS=10, P_UPPER_LIM=20, HTA_SCALE=5e-3, FINAL_FI_CLIP=10)
            print(f"Updated phi in {time.time()-t_:.2f} s")
            draw_plot_flag = True

            def plot():
                # Precompute probability values
                ekf = agent.ekf
                L1 = agent.L1
                L2 = agent.L2
                real_target_position = agent.real_target_position
                agent_position = agent.model.state[:2]
                a = agent.ekf.a_k_1
                
                N_POINTS = 100
                p_values = np.zeros((N_POINTS, N_POINTS))
                # Calculate all probabilities at once
                all_probs = ekf.p(np.array([[a1, a2, 0] for a1 in np.linspace(0, L1, N_POINTS) for a2 in np.linspace(0, L2, N_POINTS)]), 
                                upper_lim_to_normalise=20)
                # Reshape the results back to a grid
                p_values = all_probs.reshape(N_POINTS, N_POINTS)

                # print 1 / ((2 * π)^(M/2) * |Σ|^0.5)
                print(f"Normalization constant: {1 / ((2 * np.pi)**(ekf.M/2) * np.linalg.det(ekf.sigma_k_1) ** 0.5)}")


                # Plotting
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D

                fig = plt.figure(figsize=(15, 6))
                
                # 2D plot
                ax1 = plt.subplot(121)
                im = ax1.imshow(p_values.T, extent=(0, L1, 0, L2), origin='lower', aspect='auto')
                plt.colorbar(im, ax=ax1, label='Probability Density')
                ax1.scatter(real_target_position[0], real_target_position[1], c='red', label='Real Target Position')
                ax1.scatter(agent_position[0], agent_position[1], c='blue', label='Agent Position')
                ax1.set_title('2D Probability Density Function of Target Position')
                ax1.set_xlabel('X Position')
                ax1.set_ylabel('Y Position')
                ax1.legend()

                # Add debug prints
                print(f"Real target position: {real_target_position}")
                print(f"Initial estimate (a): {a}")
                print(f"Current EKF estimate: {ekf.a_k_1}")
                print(f"EKF covariance diagonal: {np.diag(ekf.sigma_k_1)}")
                
                # Test the peak location
                peak_prob = ekf.p(ekf.a_k_1.reshape(1, -1))[0]
                print(f"Probability at EKF estimate: {peak_prob}")
                
                
                # 3D plot
                ax2 = plt.subplot(122, projection='3d')
                X, Y = np.meshgrid(np.linspace(0, L1, N_POINTS), np.linspace(0, L2, N_POINTS))
                surf = ax2.plot_surface(X, Y, p_values.T, cmap='viridis', alpha=0.8)
                ax2.scatter(real_target_position[0], real_target_position[1], 
                        ekf.p(real_target_position.reshape(1, -1))[0], 
                        c='red', s=100, label='Real Target Position')
                ax2.scatter(agent_position[0], agent_position[1], 0, c='blue', s=100, label='Agent Position')
                ax2.set_title('3D Probability Density Function of Target Position')
                ax2.set_xlabel('X Position')
                ax2.set_ylabel('Y Position')
                ax2.set_zlabel('Probability Density')
                ax2.legend()

                ax1.scatter(ekf.a_k_1[0], ekf.a_k_1[1], c='green', marker='x', s=100, label='EKF Estimate')
                ax2.scatter(ekf.a_k_1[0], ekf.a_k_1[1], peak_prob, c='green', marker='x', s=100, label='EKF Estimate')
                
                plt.tight_layout()


                # Lets calculate phi(x) as well and plot in the same manner
                N_POINTS = int(N_POINTS/2)
                phi_values = np.zeros((N_POINTS, N_POINTS))
                # Calculate phi values at all positions
                for ii in range(N_POINTS):
                    for jj in range(N_POINTS):
                        phi_values[ii, jj] = agent.basis.phi(np.array([np.linspace(0, L1, N_POINTS)[ii], np.linspace(0, L2, N_POINTS)[jj]]))
                # Plotting phi values
                fig = plt.figure(figsize=(15, 6))
                ax1 = plt.subplot(121)
                im = ax1.imshow(phi_values.T, extent=(0, L1, 0, L2), origin='lower', aspect='auto')
                plt.colorbar(im, ax=ax1, label='Phi Value')
                ax1.set_title('Phi Function Values')
                ax1.set_xlabel('X Position')
                ax1.set_ylabel('Y Position')
                ax1.scatter(real_target_position[0], real_target_position[1], c='red', label='Real Target Position')
                ax1.scatter(agent_position[0], agent_position[1], c='blue', label='Agent Position')
                ax1.legend()
                # 3D plot of phi values
                ax2 = plt.subplot(122, projection='3d')
                X, Y = np.meshgrid(np.linspace(0, L1, N_POINTS), np.linspace(0, L2, N_POINTS))
                surf = ax2.plot_surface(X, Y, phi_values.T, cmap='viridis', alpha=0.8)
                ax2.set_title('3D Phi Function Values')
                ax2.set_xlabel('X Position')
                ax2.set_ylabel('Y Position')
                ax2.set_zlabel('Phi Value')
                ax2.legend()
                plt.tight_layout()
                plt.show()

            # plot()


        # Store states for plotting later etc --------------------
        u_list.append(u.copy())
        states_list.append(agent.model.state.copy())
        state_target_list.append(agent.model._state_target_history_for_plotting.copy() if isinstance(agent.model, Quadcopter) else [])

        time_list.append(time_list[i] + agent.model.dt)
        
        # Calculate delta time for this iteration
        current_time = time.time()
        # delta_time is for when we calculated ergodic control. Otherwise we dont care, its fast
        delta_time = current_time - last_iter_time if (i%Ts_iter == 0) else delta_time
        last_iter_time = current_time
        
        i += 1

    states_list = np.array(states_list)
    u_list = np.array(u_list)
    time_list = np.array(time_list)
    state_target_list = np.array(state_target_list)
    delta_t_Ts = np.array(delta_t_Ts)
    target_t_pos_list = np.array(target_t_pos_list)
    target_sigma_list = np.array(target_sigma_list)


    # ---------------- PLOTTING ----------------------------------------------------
    # Visualize the trajectory
    plt.figure(figsize=(8, 6))
    for i in range(agent.model.num_of_states):
        plt.plot(time_list, states_list[:, i], label=agent.model.state_names[i])
    plt.legend()
    plt.grid()
    
    plt.figure(figsize=(8, 6))
    for i in range(agent.model.num_of_inputs):
        plt.plot(time_list, u_list[:, i], marker='.', linestyle="-",label=f"control {i}")
    plt.legend()
    plt.grid()

    # ergodic cost
    plt.figure(figsize=(8, 6))
    plt.plot(erg_cost_list, 'g-', label='Ergodic Cost')
    plt.legend()
    plt.grid()

    # State target vs actual [Obstacle Forces (velocities) for Quadcopter]
    if isinstance(agent.model, Quadcopter):
        i_to_plot = [6, 7]
        fig, axes = plt.subplots(len(i_to_plot), 1, figsize=(8, 3*len(i_to_plot)))
        c_ = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        
        for idx, i in enumerate(i_to_plot):
            ax = axes[idx] if len(i_to_plot) > 1 else axes
            ax.plot(time_list[2:], state_target_list[:, i][2:], 
                   label=f"{agent.model.state_names[i]} (target)", 
                   linestyle="--", color=c_[idx])
            ax.plot(time_list[2:], states_list[:, i][2:], 
                   label=f"{agent.model.state_names[i]} (actual)", 
                   linestyle="-", color=c_[idx])
            ax.legend()
            ax.grid(True)
            ax.set_ylabel(agent.model.state_names[i])
            
        axes[-1].set_xlabel("Time [s]")
        fig.suptitle("State Targets vs Actual")
        plt.tight_layout()

    # Plot the time it took as a percentage of the sampling time
    plt.figure(figsize=(8, 5))
    plt.plot(delta_t_Ts[1:, 0], delta_t_Ts[1:, 1], 'k-', label='Δt/Ts', linewidth=0.7)
    plt.legend()
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Δt/Ts")
    plt.title("Δt/Ts")
    plt.ylim([0, np.max(delta_t_Ts[1:, 1]) * 1.3])
    plt.axhline(y=1, color='r', linestyle='--', label='Ts')

    # Lets plot the target position estimate and the sigma band around it
    # if target_pos_list is 2 dimensional and not empty:    
    if len(target_t_pos_list) > 1:
        fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
        sx = []
        sy = []
        sz = []
        for i in range(len(target_sigma_list)):
            # TODO: Sigmas are probably not correct, how are they calculated?
            sx.append(np.sqrt(target_sigma_list[i, :2, :2][0, 0]))
            sy.append(np.sqrt(target_sigma_list[i, :2, :2][1, 1]))
            sz.append(np.sqrt(target_sigma_list[i, :3, :3][2, 2]))
        sx = np.array(sx)
        sy = np.array(sy)
        sz = np.array(sz)
        # X position plot
        axes[0].plot(target_t_pos_list[:, 0], target_t_pos_list[:, 1], label="X Position Estimate", color='r')
        axes[0].fill_between(target_t_pos_list[:, 0], 
                            target_t_pos_list[:, 1] - 3 * sx, 
                            target_t_pos_list[:, 1] + 3 * sx, 
                            color='r', alpha=0.2, label="3σ band")
        axes[0].axhline(y=agent.real_target_position[0], color='r', linestyle='--', label="Actual X")
        axes[0].set_ylabel("X Position")
        axes[0].set_ylim([0, 1])
        axes[0].grid(True)
        axes[0].legend()

        # Y position plot
        axes[1].plot(target_t_pos_list[:, 0], target_t_pos_list[:, 2], label="Y Position Estimate", color='g')
        axes[1].fill_between(target_t_pos_list[:, 0], 
                            target_t_pos_list[:, 2] - 3 * sy, 
                            target_t_pos_list[:, 2] + 3 * sy, 
                            color='g', alpha=0.2, label="3σ band")
        axes[1].axhline(y=agent.real_target_position[1], color='g', linestyle='--', label="Actual Y")
        axes[1].set_xlabel("Time [s]")
        axes[1].set_ylabel("Y Position")
        axes[1].set_ylim([0, 1])
        axes[1].grid(True)
        axes[1].legend()

        # Z position plot
        axes[2].plot(target_t_pos_list[:, 0], target_t_pos_list[:, 3], label="Z Position Estimate", color='b')
        axes[2].fill_between(target_t_pos_list[:, 0], 
                            target_t_pos_list[:, 3] - 3 * sz, 
                            target_t_pos_list[:, 3] + 3 * sz, 
                            color='b', alpha=0.2, label="3σ band")
        axes[2].axhline(y=agent.real_target_position[2], color='b', linestyle='--', label="Actual Z")
        axes[2].set_xlabel("Time [s]")
        axes[2].set_ylabel("Z Position")
        axes[2].grid(True)
        axes[2].legend()

        plt.suptitle("Target Position Estimate with 3σ Confidence Bands")
        plt.tight_layout()

    # Ergodic Trajectory Plot
    ck_ = agent.basis.calcCkCoeff(states_list, x_buffer=None, ti=ti, T=agent.erg_c.T)
    phi_rec_from_ck = ReconstructedPhiFromCk(agent.basis, ck_)
    phi_rec = ReconstructedPhi(agent.basis, precalc_phik=False)
    vis.plotPhi(agent, phi_rec_from_ck=phi_rec_from_ck, phi_rec_from_agent=phi_rec, all_traj=states_list)

    plt.show()

    from vis import animateQuadcopter, plotQuadTrajWithInputs
    if isinstance(agent.model, Quadcopter):
        # def plotQuadTrajWithInputs(time_list, states_list, input_list, conv_inp_list=None, quad_model=None):
        plotQuadTrajWithInputs(time_list, states_list, u_list, conv_inp_list=None, quad_model=agent.model)	
        # Animate the quadcopter trajectory
        animateQuadcopter(time_list, states_list)






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
    stats.print_stats("agent.py|basis.py|model_dynamics.py|ergodic_controllers.py|barrier.py|replay_buffer.py|obstacles.py")  # Show only your modules