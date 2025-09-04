import os
import numpy as np

import cProfile
import pstats
from pstats import SortKey

# TODO: Integral of Phi should be = 1, and phi!=0 everywhere on the domain. Make sure EID updates respect that
def create_phi_func(L1=10.0, L2=10.0):
    """Creates a normalized phi function with multiple Gaussian bumps"""
    
    # Bump configuration: (x_pos, y_pos, height, width)
    bumps_config = [
        (0.3 * L1, 0.8 * L2, 5, 0.7),
        (0.7 * L1, 0.2 * L2, 4, 0.7), 
        (0.15 * L1, 0.4 * L2, 3, 15.2),
        (0.85 * L1, 0.6 * L2, 4.5, 6.3)
    ]
    
    def phi_unnormalized(s):
        x, y = s[0], s[1]
        bumps = sum(h * np.exp(-w * ((x-px)**2 + (y-py)**2)) 
                   for px, py, h, w in bumps_config)
        return bumps + 0.01
    
    # Calculate normalization constant
    from scipy.integrate import dblquad
    _phi_integral, _ = dblquad(lambda x, y: phi_unnormalized((x, y)), 0, L1, lambda _: 0, lambda _: L2)
    
    # Return normalized function
    return lambda s: phi_unnormalized(s) / _phi_integral * 4

# Create the phi function
phi_func = create_phi_func(L1=10.0, L2=10.0)

# -----------------------------------------------------------------------------------
def main():
    from my_erg_lib.agent import Agent
    from my_erg_lib.obstacles import Obstacle, saveObstaclesToMemory
    from my_erg_lib.model_dynamics import SingleIntegrator, DoubleIntegrator, Quadcopter
    from my_erg_lib.ergodic_controllers import DecentralisedErgodicController
    from my_erg_lib.basis import ReconstructedPhi, ReconstructedPhiFromCk
    import matplotlib.pyplot as plt
    import vis
    import time


    # Set up the agent -----------------------------------------------------------------------------
    
    # ===== Dynamics Model =====
    # Single integrator model ----
    # SingleIntegrator(dt=0.002)
    # u_limits = [[-1, 1], [-1, 1]]
    # u_nominal = None
    # INF_BUF_FLAG = True     # Whether to use infinite states buffer for ck calculation
    # Q_ = 1
    # R_ = 0.001
    # PREDICTION_DT = 0.002 * 25
    # RELAX_FACTOR = 1
    # IMAX = 100e3
    # TS = 0.01; T_H = 0.1; deltaT_ERG = 0.25 * 5
    # BAR_WEIGHT = 0
    # UPDATE_EID_FREQ = 110  # How often to update the EID phi function (30 means every 30 ergodic iterations) (or 30 x Ts [s])
    # CBF_SKIP_ITER = 8            # How often to apply the CBF safety filter (every n iterations). Skipping some cause it takes time
    # DELTA_SAFE = 0.1; ALPHA_HDOT = 100; ALPHA_H = 20; KAPPA_SAFE = 0.5; RHO_SAFE = 1.5

    # Double integrator model ----
    # DoubleIntegrator(dt=0.0012, x0=[9, 3, 0, 0], damping=2)
    ULIM = 50 # 30
    u_limits = [[-ULIM, ULIM], [-ULIM, ULIM]]
    u_nominal = None
    INF_BUF_FLAG = True         # Whether to use infinite states buffer for ck calculation
    Q_ = 8
    R_ = 0.001
    RELAX_FACTOR = 0.95         # U = RF * u + (1-RF) * u_prev
    IMAX = 2.5e3
    TS = 0.03; T_H = 0.5; deltaT_ERG = 3
    PREDICTION_DT = 0.0012 * 5  # model dt * 5
    BAR_WEIGHT = 0
    UPDATE_EID_FREQ = 110*2*3*4  # How often to update the EID phi function (30 means every 30 ergodic iterations) (or 30 x Ts [s])
    CBF_SKIP_ITER = 8            # How often to apply the CBF safety filter (every n iterations). Skipping some cause it takes time
    DELTA_SAFE = 0.1; ALPHA_HDOT = 100; ALPHA_H = 20; KAPPA_SAFE = 0.5; RHO_SAFE = 1.5


    # Quadrotor model -----------
    # TODO: Quad not working well with CBFs yet, the others do for now
    # x0 = [8, 4, 2, 0, 0, 0, 0,  0,  0,  0,  0,  0]
    # UP_MTR_LIM = 22         # Motor Upper Limit Thrust in [N]
    # LOW_MTR_LIM = -22       # Motor Lower Limit Thrust in [N]
    # mtr_limits = [[LOW_MTR_LIM, UP_MTR_LIM], [LOW_MTR_LIM, UP_MTR_LIM], [LOW_MTR_LIM, UP_MTR_LIM], [LOW_MTR_LIM, UP_MTR_LIM]]
    # model = Quadcopter(dt=0.002, x0=x0, z_target=2, motor_limits=mtr_limits, zero_out_states=["x", "y", "ψ"],
    #                    mass=2, damping=3.5, R=np.diag([1, 1, 1, 1])*1)
    # INF_BUF_FLAG = True     # Whether to use infinite states buffer for ck calculation
    # TS = 0.1; T_H = 0.1*15  # TS = 0.1, T_H = 0.25*5
    # deltaT_ERG = 0.1*10     # When using inf buffer it should be wayy smaller (w/o: deltaT_ERG = 0.1*160, w: 0.1*30)
    # Q_ = 8
    # R_ = 0.001 * np.eye(model.num_of_inputs)
    # u_limits = model.input_limits
    # u_nominal = model.calcLQRcontrol
    # PREDICTION_DT = model.dt * 40
    # RELAX_FACTOR = 0.8
    # IMAX = 20e3
    # BAR_WEIGHT = 0 # 50
    # UPDATE_EID_FREQ = 110  # How often to update the EID phi function (30 means every 30 ergodic iterations) (or 30 x Ts [s])
    # CBF_SKIP_ITER = 8            # How often to apply the CBF safety filter (every n iterations). Skipping some cause it takes time
    # DELTA_SAFE = 0.1; ALPHA_HDOT = 100; ALPHA_H = 20; KAPPA_SAFE = 0.5; RHO_SAFE = 1.5

    # del file cbf_log.txt and PSI.txt for gnuplot plotting
    if os.path.exists("logs/cbf_log.txt"):
        os.remove("logs/cbf_log.txt")
        os.remove("logs/PSI.txt")
        os.remove("logs/agent_state.txt")
        os.remove("logs/obstacles_points.txt")
        os.remove("logs/ergodic_cost.txt")
        os.remove("logs/ck_values.txt")

    # Agent - Ergodic Controller -------------

    # Generate Agent and connect to an ergodic controller object
    agent_1 = Agent(L1=10.0, L2=10.0, Kmax=4, 
                  dynamics_model=DoubleIntegrator(dt=0.0012, x0=[9, 3, 0, 0], damping=2),
                  phi=lambda s: 1/100) # phi=phi_func
                #   phi=phi_func)      # phi=phi_func

    agent_1.erg_c = DecentralisedErgodicController(agent_1, uNominal=u_nominal, Q=Q_, R = R_ * np.eye(agent_1.model.num_of_inputs), uLimits=u_limits,
                                                 T_sampling=TS, T_horizon=T_H, deltaT_erg=deltaT_ERG,
                                                 use_inf_buffer=INF_BUF_FLAG,
                                                 barrier_weight=BAR_WEIGHT, barrier_eps=0.05, barrier_pow=2)
    
    agent_2 = Agent(L1=10.0, L2=10.0, Kmax=4, 
                  dynamics_model=DoubleIntegrator(dt=0.0012, x0=[4, 8, 0, 0], damping=2), 
                  phi=lambda s: 1/100) # phi=phi_func
                #   phi=phi_func)      # phi=phi_func

    agent_2.erg_c = DecentralisedErgodicController(agent_2, uNominal=u_nominal, Q=Q_, R = R_ * np.eye(agent_2.model.num_of_inputs), uLimits=u_limits,
                                                 T_sampling=TS, T_horizon=T_H, deltaT_erg=deltaT_ERG,
                                                 use_inf_buffer=INF_BUF_FLAG,
                                                 barrier_weight=BAR_WEIGHT, barrier_eps=0.05, barrier_pow=2)

    agent_3 = Agent(L1=10.0, L2=10.0, Kmax=4, 
                  dynamics_model=DoubleIntegrator(dt=0.0012, x0=[7, 9.5, 0, 0], damping=2), 
                  phi=lambda s: 1/100) # phi=phi_func
                #   phi=phi_func)      # phi=phi_func

    agent_3.erg_c = DecentralisedErgodicController(agent_3, uNominal=u_nominal, Q=Q_, R = R_ * np.eye(agent_3.model.num_of_inputs), uLimits=u_limits,
                                                 T_sampling=TS, T_horizon=T_H, deltaT_erg=deltaT_ERG,
                                                 use_inf_buffer=INF_BUF_FLAG,
                                                 barrier_weight=BAR_WEIGHT, barrier_eps=0.05, barrier_pow=2)

    # Create a list of agents
    agent_list = [agent_1, agent_2, agent_3]

    # Avoiding Obstacles -------------------
    # Add obstacles and another controller to take them into account
    # Floor Plan Obstacle Map ---------------------
    # RHO0 = 0.15; KAPPA0 = 1
    # obs = [Obstacle(pos=[2.16, 3.04],   dimensions=[1.89, 0.69], obs_type='rectangle', kappa=KAPPA0, rho0=RHO0, obs_name="Obstacle 1"),
    #        Obstacle(pos=[4.51, 1.41],   dimensions=[0.20, 2.83], obs_type='rectangle', kappa=KAPPA0, rho0=RHO0, obs_name="Obstacle 2"),
    #        Obstacle(pos=[5.02, 2.73],   dimensions=[0.82, 0.20], obs_type='rectangle', kappa=KAPPA0, rho0=RHO0, obs_name="Obstacle 3"),
    #        Obstacle(pos=[7.37, 2.73],   dimensions=[0.78, 0.20], obs_type='rectangle', kappa=KAPPA0, rho0=RHO0, obs_name="Obstacle 4"),
    #        Obstacle(pos=[7.86, 1.58],   dimensions=[0.20, 3.17], obs_type='rectangle', kappa=KAPPA0, rho0=RHO0, obs_name="Obstacle 5"),
    #        Obstacle(pos=[7.41, 4.83],   dimensions=[0.20, 1.14], obs_type='rectangle', kappa=KAPPA0, rho0=RHO0, obs_name="Obstacle 6"),
    #        Obstacle(pos=[8.75, 4.81],   dimensions=[2.49, 0.20], obs_type='rectangle', kappa=KAPPA0, rho0=RHO0, obs_name="Obstacle 7"),
    #        Obstacle(pos=[6.79, 7.85],   dimensions=[1.06, 0.20], obs_type='rectangle', kappa=KAPPA0, rho0=RHO0, obs_name="Obstacle 8"),
    #        Obstacle(pos=[8.26, 8.87],   dimensions=[0.20, 2.27], obs_type='rectangle', kappa=KAPPA0, rho0=RHO0, obs_name="Obstacle 9"),
    #        Obstacle(pos=[4.73, 9.13],   dimensions=[3.05, 1.73], obs_type='rectangle', kappa=KAPPA0, rho0=RHO0, obs_name="Obstacle 10"),
    #        Obstacle(pos=[6.15, 7.05],   dimensions=[0.20, 2.44], obs_type='rectangle', kappa=KAPPA0, rho0=RHO0, obs_name="Obstacle 11"),
    #        Obstacle(pos=[5.40, 5.93],   dimensions=[1.32, 0.20], obs_type='rectangle', kappa=KAPPA0, rho0=RHO0, obs_name="Obstacle 12"),
    #        Obstacle(pos=[3.30, 7.05],   dimensions=[0.20, 2.44], obs_type='rectangle', kappa=KAPPA0, rho0=RHO0, obs_name="Obstacle 13"),
    #        Obstacle(pos=[1.64, 7.91],   dimensions=[0.99, 0.16], obs_type='rectangle', kappa=KAPPA0, rho0=RHO0, obs_name="Obstacle 14"),
    #        Obstacle(pos=[2.17, 1.58],   dimensions=0.35,         obs_type='circle',    kappa=KAPPA0, rho0=RHO0, obs_name="Obstacle 15")]
    # saveObstaclesToMemory(agent, obs_list=obs)
    # -----------------------------------------------

    # # Create 3x3 grid of circular obstacles with more spacing
    grid_size = 3
    # Add margin from edges
    margin_x = 0.15  # 15% margin from edges
    margin_y = 0.15  # 15% margin from edges
    obs_grid = []
    for i in range(grid_size):
        for j in range(grid_size):
            x_pos = margin_x + i * (agent_1.L1 - 2 * margin_x) / (grid_size - 1)
            y_pos = margin_y + j * (agent_1.L2 - 2 * margin_y) / (grid_size - 1)
            obs_grid.append(Obstacle(pos=[x_pos, y_pos], dimensions=0.7, obs_type='circle', kappa=1, rho0=0.15, obs_name=f"Obstacle {i*grid_size + j + 1}"))


    # Avoiding Walls ----------------------
    obs_walls  = [Obstacle(pos=[0,          0],   dimensions=[0, +1], obs_type='wall', kappa=KAPPA_SAFE, rho0=RHO_SAFE, obs_name="Bottom Wall"),
                  Obstacle(pos=[0, agent_1.L2],   dimensions=[0, -1], obs_type='wall', kappa=KAPPA_SAFE, rho0=RHO_SAFE, obs_name="Top Wall"   ),
                  Obstacle(pos=[0,          0],   dimensions=[+1, 0], obs_type='wall', kappa=KAPPA_SAFE, rho0=RHO_SAFE, obs_name="Left Wall"  ),
                  Obstacle(pos=[agent_1.L1, 0],   dimensions=[-1, 0], obs_type='wall', kappa=KAPPA_SAFE, rho0=RHO_SAFE, obs_name="Right Wall" )]

    # Save obstacles to memory
    for agent in agent_list:
        saveObstaclesToMemory(agent, obs_list=obs_grid)
        saveObstaclesToMemory(agent, obs_list=obs_walls)

    # Print uNominal Status
    # print(agent_1.erg_c.uNominal)
    
    # Visualize H-field
    # vis.visHfield(agent, L_limits=[-0.5, agent.L1+0.5, -0.5, agent.L2+0.5], delta=DELTA_SAFE, num_of_points=200)

    # Lets now update the phi_function to take into account the obstacles
    for agent in agent_list:
        agent.basis.phi = agent.modifedPhiForObstacles(agent.basis.phi, obs_to_exclude="None")
        agent.basis.precalcAllPhiK()

    # if input("\nVisualise Potential Fields? (y/n): ") == "y":
    #     vis.visPotentialFields(agent)

    # More parameters
    if input("Localise Targets? (y/n): ") == "y":
        LOCALISE_TARGETS_FLAG = True
    else:
        LOCALISE_TARGETS_FLAG = False

    if LOCALISE_TARGETS_FLAG and input("Update EID phi function? (y/n): ") == "y":
        UPDATE_EID_FLAG = True
    else:
        UPDATE_EID_FLAG = False

    if input("Save images to file? (y/n): ") == "y":
        SAVE_IMAGES_FLAG = True
    else:
        SAVE_IMAGES_FLAG = False

    dash_resp = input("Start Python Dashboard.py? (y/n): ")
    if dash_resp == "y":
        # Run python dashboard.py
        os.system("start cmd /k python dashboard.py")
    # The flag is used to write to file, so even if not spawning new window, we may need the flags on (in case there is one dashboard already running)
    DASHBOARD_FLAG = True if dash_resp != "n" else False
    for agent in agent_list:
        agent.DASHBOARD_FLAG = DASHBOARD_FLAG

    input("Press Enter to continue...")

    # --------------------------------------------------------------------------------------------------
    # Write obstacle positions to file for later visualisation with dashboard.py
    if DASHBOARD_FLAG:
        obstacle_points = []
        for obs_grid in agent_1.obstacle_list:
            obstacle_points.append(obs_grid.returnBoundaryPointsForPlotting(num_of_points=20))
        # Write to file
        with open("logs/obstacles_points.txt", "w") as f:
            for obs_points in obstacle_points:
                np.savetxt(f, obs_points, delimiter="\t")
                f.write("\n")
        del obstacle_points  # Clear the list to free memory
    # --------------------------------------------------------------------------------------------------
    
    # Lists to store for plotting - now for multiple agents
    states_list = [[agent.model.state.copy()] for agent in agent_list]
    time_list = [0]  # Time vector (shared across all agents)
    u_list = [[np.zeros((agent.model.num_of_inputs,))] for agent in agent_list]
    u_before = [np.zeros((agent.model.num_of_inputs,)) for agent in agent_list]  # Previous control action per agent
    u_safe_list = [[np.zeros((agent.model.num_of_inputs,))] for agent in agent_list]
    erg_cost_list = [[] for _ in agent_list]
    total_erg_cost_list = []  # List to store total ergodic cost from average ck
    state_target_list = [[agent.model._state_target.copy()] if isinstance(agent.model, Quadcopter) else [] for agent in agent_list]
    delta_t_Ts = [[] for _ in agent_list]
    draw_plot_flag = False  # Flag that alternates when updating EID to plot
    target_data = [{i: {'times': [], 'positions': [], 'sigmas': []} for i in range(max(1, len(agent.ekfs)))} for agent in agent_list]
    
    ti = time_list[0]; ti_indx = 0
    Ts_iter = int(agent_list[0].erg_c.Ts / agent_list[0].model.dt)  # Number of iterations per sampling time (using first agent)
    
    # Initialize timing variables
    initial_time = time.time()
    last_iter_time = time.time()
    delta_time = 1
    
    i = 0
    while i < IMAX:
        # if i == 5000:
        #     agent.real_target_positions.append(np.array([0.1, 0.1, 0]))
            
        # If multiple of Ts, calculate ergodic action
        if i % Ts_iter == 0:
            ti = time_list[i]; ti_indx = i
            
            # Process each agent
            for agent_idx, agent in enumerate(agent_list):
                agent.time_since_start = ti

                # Ergodic Control Calculation ---------------------------------------------
                # Calculate ergodic control for the sample step
                us, tau, lamda_dur, erg_cost = agent.erg_c.calcNextActionTriplet(time_list[i], prediction_dt=PREDICTION_DT)
                agent.erg_c.ck_aver_others = np.mean([other_agent.basis.ck for other_agent in agent_list if other_agent != agent])  

                # change lamda dur only if not quadcopter
                if not isinstance(agent.model, Quadcopter):
                    lamda_dur = agent.erg_c.Ts
                erg_cost_list[agent_idx].append(erg_cost)
                delta_t_Ts[agent_idx].append([ti, delta_time / agent.erg_c.Ts])

                # Debug State Information Printing (only for first agent to avoid clutter)
                if agent_idx == 0 and i % 160 == 0:
                    def u_str(u):
                        res = "["
                        for j in range(len(u)):
                            res += f"{u[j]:.2f}, "
                        return res[:-2] + "]"
                    
                    expected_time_max = IMAX/(i+1)*(time.time()-initial_time); elapsed_time_max = time.time()-initial_time
                    rem_time_max = expected_time_max - elapsed_time_max     # rem_time_simple = delta_time * (IMAX-i)/Ts_iter
                    print(f"ti = {ti:.2f} s\t Erg cost: {erg_cost:.2f} \t i: {i}/{IMAX:.0f} \t perc: {i/IMAX:.2%} \t Δt/Ts: {delta_time/agent.erg_c.Ts:.2f}\t remaining: {rem_time_max:.0f} s\t elapsed: {time.time()-initial_time:.1f} s ({time.time()-initial_time + delta_time * (IMAX-i)/Ts_iter:.0f} s) ({IMAX/(i+1)*(time.time()-initial_time):.0f} s)")
                    print(f"{agent.model.state_string} \n u = {u_str(us)} \t (tau - ti)/T = {(tau - ti)/agent.erg_c.T:.1%} \t lamda_dur = {lamda_dur:.4f} \t lamda/Ts = {lamda_dur/agent.erg_c.Ts:.2%}\n")
                
                # Debug print if agent inside boundaries
                agent.withinBounds(agent.model.state[:2])
                if np.any(np.abs(agent.model.state[:2]) > 50):
                    print(f"--> Agent {agent_idx} WAYY out of bounds! Stopping simulation.")
                    break
                
                # Update the action mask
                if lamda_dur > 0:
                    agent.erg_c.action_mask.pushAction(ti, tau, lamda_dur, us.copy())

            # Calculate total ergodic cost from average ck of all agents
            ck_total = np.mean([agent.basis.ck for agent in agent_list], axis=0)
            total_erg_cost = agent_list[0].erg_c.calcErgodicCost(ck_total)  # Use any agent since they're all the same
            total_erg_cost_list.append(total_erg_cost)

            # Multi-Target EKF update -------------------------------------------------
            if LOCALISE_TARGETS_FLAG:
                for agent_idx, agent in enumerate(agent_list):
                    # Make a measurement using the sensor
                    # if agent.model is single integrator agent_pos = agent.model.state + append a zero
                    if isinstance(agent.model, SingleIntegrator):
                        agent_pos = np.append(agent.model.state, 0)
                    elif isinstance(agent.model, DoubleIntegrator):
                        agent_pos = np.append(agent.model.state[0:2], 0)
                    else:
                        agent_pos = agent.model.state[0:3]
                    z_raw = agent.sensor.getMultipleMeasurements(agent.real_target_positions, agent_pos)
                    agent.sensor.measurements_raw = z_raw.copy()  # Store raw measurements for later use
                    # If we have some measurements and zero targets, initialize them all
                    if z_raw is not None and agent.num_of_targets == 0:
                        for measurement in z_raw:
                            agent.spawnNewTargetEstimate(measurement, current_time=ti)
                    z_associated = agent.associateTargetsWithMahalanobis(z_raw, agent_pos, ASSOCIATION_THRESHOLD=5)
                    # Update estimate using the EKF
                    for meas_id, measurement in enumerate(z_associated):
                        # Update the EKF with the measurement
                        agent.ekfs[meas_id].update(xk=agent_pos, zk=measurement, time_now=ti)
                        # Update target estimate
                        agent.target_estimates[meas_id] = agent.ekfs[meas_id].a_k_1.copy()
                    # If we have a measurement that has not been associated with any target, spawn a new target
                    z_without_none = np.array([z if z is not None else np.zeros((2,)) for z in z_associated])
                    for m in z_raw:
                        if m is not None and m not in z_without_none:
                            # Spawn a new target with the measurement
                            agent.spawnNewTargetEstimate(measurement=m, current_time=ti)
                            # Add new target to target_data dictionary
                            new_target_id = len(agent.ekfs) - 1  # Get the ID of the newly spawned target
                            target_data[agent_idx][new_target_id] = {'times': [], 'positions': [], 'sigmas': []}

                    # Store data using dictionary structure
                    for meas_id in range(len(agent.ekfs)):
                        # Ensure the target exists in target_data
                        if meas_id not in target_data[agent_idx]:
                            target_data[agent_idx][meas_id] = {'times': [], 'positions': [], 'sigmas': []}
                        
                        target_data[agent_idx][meas_id]['times'].append(time_list[i])
                        target_data[agent_idx][meas_id]['positions'].append(agent.ekfs[meas_id].a_k_1.copy())
                        target_data[agent_idx][meas_id]['sigmas'].append(agent.ekfs[meas_id].sigma_k_1.copy())

                    # Check if we need to merge targets
                    if agent.num_of_targets > 1:
                        # Chack Bhattacharyya Distance between every pair and merge as needed
                        agent.mergeTargetsIfNeeded(MERGE_THRESHOLD=3, EUCL_DIST_THRESHOLD=0.15, SIMILAR_MEASUREMENTS_ANGLE_THRESHOLD_RAD=30* np.pi/180)

                    # Also, lets check and remove outdated target estimates
                    agent.searchAndRemoveOldTargetEstimates(current_time=ti, MAX_AGE_SEC=60)

            # Simulation saving file ----------------------------
            # if draw_plot_flag:
            if (draw_plot_flag or i % 160/2 == 0) and SAVE_IMAGES_FLAG:
                # Use first agent for visualization
                agent_for_img = agent_list[0]
                x_traj, _, _ = agent_for_img.model.simulateForward(x0=agent_for_img.model.state, ti=ti, udef=agent_for_img.erg_c.uNominal, T=agent_for_img.erg_c.T, dt=PREDICTION_DT)
                erg_traj = x_traj[:, :2] # Only take the ergodic dimensions
                if INF_BUF_FLAG:
                    ck_ = agent_for_img.basis.calcCkCoeffRecursive(erg_traj, ti, agent_for_img.erg_c.T, agent_for_img.erg_c.Ts, agent_for_img.erg_c.t0_erg, x_buffer=agent_for_img.erg_c.past_states_buffer.get(), update_ck_old=False)
                else:
                    ck_ = agent_for_img.basis.calcCkCoeff(erg_traj, x_buffer=agent_for_img.erg_c.past_states_buffer.get() ,ti=ti, T=agent_for_img.erg_c.T)
                phi_rec_from_ck = ReconstructedPhiFromCk(agent_for_img.basis, ck_)
                print("Plotting phi")
                phi_3_ = ReconstructedPhi(agent_for_img.basis, precalc_phik=False)
                # Pass all agent states for visualization
                all_agent_traj = [states_list[idx] for idx in range(len(agent_list))]
                vis.plotPhi(agent_for_img, phi_rec_from_ck=phi_rec_from_ck, phi_rec_from_agent=phi_3_, all_traj=all_agent_traj, grid_res=40, ck_total=ck_total, agent_list=agent_list)
                plt.savefig(f"images/phiQuadWithObs_{ti:.2f}.png")
                print(f"Saved image to images/phiQuadWithObs_{ti:.2f}.png")
                plt.close()
                draw_plot_flag = False
        
        # Continue with simulation of agents
        for agent_idx, agent in enumerate(agent_list):
            us_ = agent.erg_c.action_mask.readAction(t_now=time_list[i])
            
            if us_ is not None:
                # Apply the control action to the agent's model
                u = us_
            else:
                # If no ergodic control is available, use the nominal control
                u = agent.erg_c.uNominal(agent.model.state, time_list[i])

            # Lets apply the CBF Safety Filter to ergodic output
            # Apply safety control every 5 iterations or every one if we were near an obstacle before
            if np.any(u_safe_list[agent_idx][-1] != 0) or i%CBF_SKIP_ITER == 0:
                u_safe = agent.calcUsafe(agent.model.state, u, alpha_1=ALPHA_HDOT, alpha_2=ALPHA_H, delta=DELTA_SAFE)
            else:
                u_safe = np.zeros((agent.model.num_of_inputs,))
            u_safe_list[agent_idx].append(u_safe.copy())  # Store the safe control action for later use
            u += u_safe.copy()  # Add the safe control action to the ergodic control action
            
            # Lets clip again to agents control input limits
            u = np.clip(u, agent.erg_c.uLimits[:, 0], agent.erg_c.uLimits[:, 1])
            # If u_magnitude is more than 100, print a warning
            if np.linalg.norm(u) > 100:
                print(f"CRITICAL: Agent {agent_idx} control action is too high: {u}")

            # Lets smooth out with the previous control action
            u = RELAX_FACTOR * u + (1-RELAX_FACTOR) * u_before[agent_idx]  # Smooth the control action
            u_before[agent_idx] = u.copy()

            # TODO: Here we should simulate forward for simulation_dt with a dt, instead of stepping. Implement model simulation function
            agent.model.state = agent.model.step(agent.model.state, u)         # Step the model with the control action
            agent.erg_c.past_states_buffer.push(agent.model.state.copy()[:2])  # Store the state in the buffer
            if INF_BUF_FLAG:
                # In inf buffer case, we have 2 buffers: one for past "ts" states and one for past "t0_erg" states. 
                # We could use one being the bigger of the two, but that complicates things. No need since past "ts" states are not much.
                agent.erg_c.past_erg_history_buffer.push(agent.model.state.copy()[:2])

            # Store states and control for plotting later
            u_list[agent_idx].append(u.copy())
            states_list[agent_idx].append(agent.model.state.copy())
            state_target_list[agent_idx].append(agent.model._state_target_history_for_plotting.copy() if isinstance(agent.model, Quadcopter) else [])

        # Lets update phi(x) if needed
        if i%(Ts_iter * UPDATE_EID_FREQ) == 0 and UPDATE_EID_FLAG:
            t_ = time.time()
            print("Updating phi...")
            # TODO: Globalize the parameters used here
            for agent_idx, agent in enumerate(agent_list):
                agent.updateEIDphiFunction(NUM_GAUSS_POINTS=10, P_UPPER_LIM=8, HTA_SCALE=8e-8, FINAL_FI_CLIP=0.01, ALWAYS_ADD=0)
            print(f"Updated phi in {time.time()-t_:.2f} s")
            draw_plot_flag = True
            # Restart ergodic memory buffer for all agents
            if INF_BUF_FLAG:
                for agent in agent_list:
                    agent.erg_c.t0_erg = max(time_list[i] - agent.erg_c.deltaT_erg, 0)  # Reset the t0_erg to the current time minus the deltaT_erg
                    
                    agent.basis.ck_bar_old = agent.basis.calcCkCoeff(agent.erg_c.past_erg_history_buffer.get(), agent.erg_c.t0_erg, agent.erg_c.deltaT_erg, do_not_divide_integral_flag=True)
                    agent.basis.ck_bar_old /= (time_list[i] + agent.erg_c.T - agent.erg_c.t0_erg)
            else:
                # Here we need to reset the buffer since it can contain a lot of outdated past state information # TODO: Check if needed
                for agent in agent_list:
                    agent.erg_c.past_states_buffer.reset(last_perc_to_keep=0.1)  # Reset the past states buffer

        # Append agent states to file (for all agents)
        if DASHBOARD_FLAG:
            with open("logs/agent_state.txt", "a") as f:
                for agent_idx, agent in enumerate(agent_list): 
                    f.write(f"{time_list[i]:.4f} {agent.model.state[0]:.4f} {agent.model.state[1]:.4f} {u_list[agent_idx][-1][0]:.4f} {u_list[agent_idx][-1][1]:.4f} {agent_idx}\n")          
            with open("logs/ergodic_cost.txt", "a") as f:
                for agent_idx, agent in enumerate(agent_list):
                    if len(erg_cost_list[agent_idx]) > 0:
                        f.write(f"{time_list[i]:.4f} {erg_cost_list[agent_idx][-1]:.4f} {1 if np.any(u_safe_list[agent_idx][-1] != 0) else 0} {agent_idx}\n")  # Save ergodic cost to file
                # Write total ergodic cost with special agent index -1
                if len(total_erg_cost_list) > 0:
                    f.write(f"{time_list[i]:.4f} {total_erg_cost_list[-1]:.4f} 0 -1\n")  # Total ergodic cost with agent_idx = -1

        # Store states for plotting later etc --------------------
        # (States and control already stored in the agent loop above)
        time_list.append(time_list[i] + agent_list[0].model.dt)  # Use first agent's dt (assuming all agents have same dt)
        
        # Calculate delta time for this iteration
        current_time = time.time()
        # delta_time is for when we calculated ergodic control. Otherwise we dont care, its fast
        delta_time = current_time - last_iter_time if (i%Ts_iter == 0) else delta_time
        last_iter_time = current_time
        
        i += 1

    # Convert lists to numpy arrays for plotting
    states_list = [np.array(agent_states) for agent_states in states_list]
    u_list = [np.array(agent_u) for agent_u in u_list]
    time_list = np.array(time_list)
    state_target_list = [np.array(agent_targets) if len(agent_targets) > 0 else [] for agent_targets in state_target_list]
    delta_t_Ts = [np.array(agent_delta) for agent_delta in delta_t_Ts]
    u_safe_list = [np.array(agent_u_safe) for agent_u_safe in u_safe_list]


    # ---------------- PLOTTING ----------------------------------------------------
    # Visualize the trajectory and control inputs in a 2x1 grid (for first agent)
    agent_idx_to_plot = 0  # Plot first agent by default
    agent_to_plot = agent_list[agent_idx_to_plot]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    
    # Top subplot: States
    for i in range(agent_to_plot.model.num_of_states):
        if i == 0:  # x position
            ax1.plot(time_list, states_list[agent_idx_to_plot][:, i], label=f'{agent_to_plot.model.state_names[i]} (Agent {agent_idx_to_plot})', color='blue', linestyle='-')
        elif i == 1:  # y position
            ax1.plot(time_list, states_list[agent_idx_to_plot][:, i], label=f'{agent_to_plot.model.state_names[i]} (Agent {agent_idx_to_plot})', color='orange', linestyle='-')
        elif i == 2:  # x velocity
            ax1.plot(time_list, states_list[agent_idx_to_plot][:, i], label=f'{agent_to_plot.model.state_names[i]} (Agent {agent_idx_to_plot})', color='blue', linestyle='--')
        elif i == 3:  # y velocity
            ax1.plot(time_list, states_list[agent_idx_to_plot][:, i], label=f'{agent_to_plot.model.state_names[i]} (Agent {agent_idx_to_plot})', color='orange', linestyle='--')
        else:  # other states
            ax1.plot(time_list, states_list[agent_idx_to_plot][:, i], label=f'{agent_to_plot.model.state_names[i]} (Agent {agent_idx_to_plot})')
    
    # Add green background where u_safe is applied (any non-zero value) for plotted agent
    u_safe_applied = np.any(np.abs(u_safe_list[agent_idx_to_plot]) > 1e-6, axis=1)  # Check if any component is non-zero
    if np.any(u_safe_applied):
        # Find continuous regions where u_safe is applied
        safe_regions = []
        start_idx = None
        for i, applied in enumerate(u_safe_applied):
            if applied and start_idx is None:
                start_idx = i
            elif not applied and start_idx is not None:
                safe_regions.append((start_idx, i-1))
                start_idx = None
        # Handle case where u_safe is applied until the end
        if start_idx is not None:
            safe_regions.append((start_idx, len(u_safe_applied)-1))
        
        # Add green background for each region
        for start_idx, end_idx in safe_regions:
            ax1.axvspan(time_list[start_idx], time_list[end_idx], 
                       alpha=0.2, color='green', zorder=0, 
                       label='Safety Control Applied' if start_idx == safe_regions[0][0] else "")
    
    ax1.axhline(y=agent_to_plot.L1, color='r', linestyle='--', label='L1')
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.legend()
    ax1.grid()
    ax1.set_title(f'Agent {agent_idx_to_plot} States')
    ax1.set_ylabel('State Values')
    
    # Bottom subplot: Control inputs
    for i in range(agent_to_plot.model.num_of_inputs):
        if i == 0:  # u1
            ax2.plot(time_list, u_list[agent_idx_to_plot][:, i], linestyle="-", label=f"control {i} (Agent {agent_idx_to_plot})", color='blue')
            ax2.plot(time_list, u_list[agent_idx_to_plot][:, i] - u_safe_list[agent_idx_to_plot][:, i], linestyle="--", label=f"control {i} - safe", color='blue')
        elif i == 1:  # u2
            ax2.plot(time_list, u_list[agent_idx_to_plot][:, i], linestyle="-", label=f"control {i} (Agent {agent_idx_to_plot})", color='orange')
            ax2.plot(time_list, u_list[agent_idx_to_plot][:, i] - u_safe_list[agent_idx_to_plot][:, i], linestyle="--", label=f"control {i} - safe", color='orange')
        else:  # other controls
            ax2.plot(time_list, u_list[agent_idx_to_plot][:, i], linestyle="-", label=f"control {i} (Agent {agent_idx_to_plot})")
            ax2.plot(time_list, u_list[agent_idx_to_plot][:, i] - u_safe_list[agent_idx_to_plot][:, i], linestyle="--", label=f"control {i} - safe")
    
    # Add green background where u_safe is applied (any non-zero value)
    if np.any(u_safe_applied):
        # Add green background for each region
        for start_idx, end_idx in safe_regions:
            ax2.axvspan(time_list[start_idx], time_list[end_idx], 
                       alpha=0.2, color='green', zorder=0, 
                       label='Safety Control Applied' if start_idx == safe_regions[0][0] else "")
    
    ax2.legend()
    ax2.grid()
    ax2.set_title('Control Inputs')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Control Values')
    
    plt.tight_layout()

    plt.figure(figsize=(8, 6))
    for i in range(agent_to_plot.model.num_of_inputs):
        if i == 0:  # u1
            plt.plot(time_list, u_safe_list[agent_idx_to_plot][:, i], linestyle="-", label=f"control {i} - safe (Agent {agent_idx_to_plot})", color='blue')
        elif i == 1:  # u2
            plt.plot(time_list, u_safe_list[agent_idx_to_plot][:, i], linestyle="-", label=f"control {i} - safe (Agent {agent_idx_to_plot})", color='orange')
        else:  # other controls
            plt.plot(time_list, u_safe_list[agent_idx_to_plot][:, i], linestyle="-", label=f"control {i} - safe (Agent {agent_idx_to_plot})")
    plt.legend()
    plt.grid()

    # ergodic cost
    plt.figure(figsize=(8, 6))
    for agent_idx, agent in enumerate(agent_list):
        plt.plot(erg_cost_list[agent_idx], label=f'Ergodic Cost - Agent {agent_idx}')
    # Plot total ergodic cost
    plt.plot(total_erg_cost_list, label='Total Ergodic Cost (Average CK)', linewidth=2, color='black', linestyle='-')
    plt.legend()
    plt.grid()
    plt.title('Ergodic Cost (Individual Agents + Total)')
    plt.xlabel('Time Step')
    plt.ylabel('Cost')

    # State target vs actual [Obstacle Forces (velocities) for Quadcopter]
    if isinstance(agent_to_plot.model, Quadcopter) and len(state_target_list[agent_idx_to_plot]) > 0:
        i_to_plot = [6, 7]
        fig, axes = plt.subplots(len(i_to_plot), 1, figsize=(8, 3*len(i_to_plot)))
        c_ = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        
        for idx, i in enumerate(i_to_plot):
            ax = axes[idx] if len(i_to_plot) > 1 else axes
            ax.plot(time_list[2:], state_target_list[agent_idx_to_plot][:, i][2:], 
                   label=f"{agent_to_plot.model.state_names[i]} (target) - Agent {agent_idx_to_plot}", 
                   linestyle="--", color=c_[idx])
            ax.plot(time_list[2:], states_list[agent_idx_to_plot][:, i][2:], 
                   label=f"{agent_to_plot.model.state_names[i]} (actual) - Agent {agent_idx_to_plot}", 
                   linestyle="-", color=c_[idx])
            ax.legend()
            ax.grid(True)
            ax.set_ylabel(agent_to_plot.model.state_names[i])
            
        axes[-1].set_xlabel("Time [s]")
        fig.suptitle(f"State Targets vs Actual - Agent {agent_idx_to_plot}")
        plt.tight_layout()

    # Plot the time it took as a percentage of the sampling time (for first agent)
    plt.figure(figsize=(8, 5))
    if len(delta_t_Ts[agent_idx_to_plot]) > 1:
        plt.plot(delta_t_Ts[agent_idx_to_plot][1:, 0], delta_t_Ts[agent_idx_to_plot][1:, 1], 'k-', label=f'Δt/Ts - Agent {agent_idx_to_plot}', linewidth=0.7)
        plt.ylim([0, np.max(delta_t_Ts[agent_idx_to_plot][1:, 1]) * 1.3])
    plt.legend()
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Δt/Ts")
    plt.title("Δt/Ts")
    plt.axhline(y=1, color='r', linestyle='--', label='Ts')

    # EKF - Lets plot the target position estimate and the sigma band around it (for first agent)
    for agent_idx, agent in enumerate(agent_list):
        if len(target_data[agent_idx]) > 0 and len(target_data[agent_idx][0]['times']) > 1 and LOCALISE_TARGETS_FLAG:  # Check if we have data
            # Create a single figure with 3 subplots for all targets
            fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            
            for meas_id in range(len(agent_list[agent_idx].ekfs)):
                # Get data for this target from dictionary
                times = np.array(target_data[agent_idx][meas_id]['times'])
                positions = np.array(target_data[agent_idx][meas_id]['positions'])
                sigmas = np.array(target_data[agent_idx][meas_id]['sigmas'])
                
                if len(times) == 0:
                    continue
                    
                color = colors[meas_id % len(colors)]
                
                # Extract standard deviations
                sx = np.sqrt(sigmas[:, 0, 0])  # X variance
                sy = np.sqrt(sigmas[:, 1, 1])  # Y variance  
                sz = np.sqrt(sigmas[:, 2, 2])  # Z variance
                
                # X position plot
                axes[0].plot(times, positions[:, 0], color=color, label=f'Target {meas_id} - Agent {agent_idx}')
                axes[0].fill_between(times, 
                                    positions[:, 0] - 3 * sx, 
                                    positions[:, 0] + 3 * sx, 
                                    color=color, alpha=0.2)
                # Only plot real target position if it exists
                if meas_id < len(agent_list[agent_idx].real_target_positions):
                    axes[0].axhline(y=agent_list[agent_idx].real_target_positions[meas_id][0], color=color, linestyle='--', alpha=0.8, label=f'Real Target {meas_id}')

                # Y position plot
                axes[1].plot(times, positions[:, 1], color=color)
                axes[1].fill_between(times, 
                                    positions[:, 1] - 3 * sy, 
                                    positions[:, 1] + 3 * sy, 
                                    color=color, alpha=0.2)
                # Only plot real target position if it exists
                if meas_id < len(agent_list[agent_idx].real_target_positions):
                    axes[1].axhline(y=agent_list[agent_idx].real_target_positions[meas_id][1], color=color, linestyle='--', alpha=0.8)

                # Z position plot
                axes[2].plot(times, positions[:, 2], color=color)
                axes[2].fill_between(times, 
                                    positions[:, 2] - 3 * sz, 
                                    positions[:, 2] + 3 * sz, 
                                    color=color, alpha=0.2)
                # Only plot real target position if it exists
                if meas_id < len(agent_list[agent_idx].real_target_positions):
                    axes[2].axhline(y=agent_list[agent_idx].real_target_positions[meas_id][2], color=color, linestyle='--', alpha=0.8)

            # Configure axes
            axes[0].set_ylabel("X Position")
            axes[0].set_ylim([0, agent_list[agent_idx].L1])
            axes[0].grid(True)
            axes[0].legend()
            
            axes[1].set_ylabel("Y Position")
            axes[1].set_ylim([0, agent_list[agent_idx].L2])
            axes[1].grid(True)

            axes[2].set_xlabel("Time [s]")
            axes[2].set_ylabel("Z Position")
            axes[2].grid(True)

            plt.suptitle(f"Agent {agent_idx} - All Targets Position Estimates with 3σ Confidence Bands")
            plt.tight_layout()

    # Ergodic Trajectory Plot (using first agent for visualization)
    agent_for_vis = agent_list[0]
    x_traj, u_traj, t_traj = agent_for_vis.model.simulateForward(x0=agent_for_vis.model.state, ti=ti, udef=agent_for_vis.erg_c.uNominal, T=agent_for_vis.erg_c.T, dt=PREDICTION_DT)
    erg_traj = x_traj[:, :2] # Only take the ergodic dimensions
    if INF_BUF_FLAG:
        ck_ = agent_for_vis.basis.calcCkCoeffRecursive(erg_traj, ti, agent_for_vis.erg_c.T, agent_for_vis.erg_c.Ts, agent_for_vis.erg_c.t0_erg, x_buffer=agent_for_vis.erg_c.past_states_buffer.get(), update_ck_old=False)
    else:
        ck_ = agent_for_vis.basis.calcCkCoeff(erg_traj, x_buffer=agent_for_vis.erg_c.past_states_buffer.get() ,ti=ti, T=agent_for_vis.erg_c.T)
    phi_rec_from_ck = ReconstructedPhiFromCk(agent_for_vis.basis, ck_)
    phi_rec = ReconstructedPhi(agent_for_vis.basis, precalc_phik=False)
    # Pass all agent trajectories to the visualization
    all_agent_states = [states_list[i][:, :2] for i in range(len(agent_list))]  # Extract only x,y positions
    vis.plotPhi(agent_for_vis, phi_rec_from_ck=phi_rec_from_ck, phi_rec_from_agent=phi_rec, all_traj=all_agent_states, ck_total=ck_total, agent_list=agent_list)

    plt.show()



    from vis import animateQuadcopter, plotQuadTrajWithInputs
    if isinstance(agent_list[0].model, Quadcopter):
        # def plotQuadTrajWithInputs(time_list, states_list, input_list, conv_inp_list=None, quad_model=None):
        plotQuadTrajWithInputs(time_list, states_list[0], u_list[0], conv_inp_list=None, quad_model=agent_list[0].model)	
        # Animate the quadcopter trajectory
        animateQuadcopter(time_list, states_list[0])






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