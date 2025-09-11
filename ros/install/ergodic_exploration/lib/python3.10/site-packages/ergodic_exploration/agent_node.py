import os
import numpy as np
import threading
import signal
import sys

import rclpy
import rclpy.parameter
import argparse

# TODO: Integral of Phi should be = 1, and phi!=0 everywhere on the domain. Make sure EID updates respect that
def createPhiFunc(L1_BOUNDS=[0.0, 10.0], L2_BOUNDS=[0.0, 10.0]):
    """Creates a normalized phi function with multiple Gaussian bumps"""
    
    L1_min, L1_max = L1_BOUNDS
    L2_min, L2_max = L2_BOUNDS
    L1_range = L1_max - L1_min
    L2_range = L2_max - L2_min
    
    # Bump configuration: (x_pos, y_pos, height, width)
    bumps_config = [
        (L1_min + 0.3 * L1_range, L2_min + 0.8 * L2_range, 5, 0.7),
        (L1_min + 0.7 * L1_range, L2_min + 0.2 * L2_range, 4, 0.7), 
        (L1_min + 0.15 * L1_range, L2_min + 0.4 * L2_range, 3, 15.2),
        (L1_min + 0.85 * L1_range, L2_min + 0.6 * L2_range, 4.5, 6.3)
    ]
    
    def phiUnnormalized(s):
        x, y = s[0], s[1]
        bumps = sum(h * np.exp(-w * ((x-px)**2 + (y-py)**2)) 
                   for px, py, h, w in bumps_config)
        return bumps + 0.01
    
    # Calculate normalization constant
    from scipy.integrate import dblquad
    _phi_integral, _ = dblquad(lambda x, y: phiUnnormalized((x, y)), L1_min, L1_max, lambda _: L2_min, lambda _: L2_max)
    
    # Return normalized function
    return lambda s: phiUnnormalized(s) / _phi_integral * 4


# Global shutdown flag
shutdown_flag = threading.Event()

def signalHandler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\nReceived interrupt signal. Shutting down gracefully...")
    shutdown_flag.set()

# Set up signal handler
signal.signal(signal.SIGINT, signalHandler)

# -----------------------------------------------------------------------------------
def main(args=None):
    from my_erg_lib.agent import Agent
    from my_erg_lib.obstacles import Obstacle, saveObstaclesToMemory
    from my_erg_lib.model_dynamics import SingleIntegrator, DoubleIntegrator, Quadcopter, SimpleBoatSecondOrder, SimpleCarSecondOrder
    from my_erg_lib.ergodic_controllers import DecentralisedErgodicController
    from my_erg_lib.basis import ReconstructedPhi, ReconstructedPhiFromCk
    import matplotlib.pyplot as plt
    import my_erg_lib.vis as vis
    import time

    # Lets parse arguments to get agent ID
    parser = argparse.ArgumentParser(description='Run agent node with dynamic ID')
    parser.add_argument('--agent_id',           type=int,            required=True,                  help='Agent ID to name the node')
    parser.add_argument('--init_pos',           type=float, nargs=2, required=False, default=[9, 3], help='Initial position as [x, y]')
    parser.add_argument('--l_bounds',           type=float, nargs=4, required=False, default=[0, 10, 0, 10], help='Initial bounds as [x_min, x_max, y_min, y_max] for ergodic exploration')
    parser.add_argument('--model_type',         type=str,            required=False, default='DoubleIntegrator', help='Dynamics model type (SingleIntegrator, DoubleIntegrator, SimpleBoatSecondOrder)')
    parser.add_argument('--antenna_rad',        type=float,          required=False, default=np.inf, help='Antenna radius in meters')
    parser.add_argument('--kmax',               type=int,            required=False, default=4,      help='Maximum Fourier modes to use for reconstruction')
    parser.add_argument('--antenna_range_flag', type=lambda x: x.lower() == 'true', required=False, default=False, help='Antenna range flag (true/false)')
    parser.add_argument('--talk_alike_flag',    type=lambda x: x.lower() == 'true', required=False, default=False, help='Weather to communicate only with similar model (boats with boats, cars with cars, etc) (true/false)')
    parser.add_argument('--same_l_bounds_flag', type=lambda x: x.lower() == 'true', required=False, default=True, help='Whether to communicate only with agents having same L bounds (true/false)')
    parsed_args, ros_args = parser.parse_known_args()  # Parse known args only, keep ROS args separate
    AGENT_ID = parsed_args.agent_id
    INIT_POS_2D = np.array(parsed_args.init_pos)
    ANTENNA_RANGE_FLAG = parsed_args.antenna_range_flag
    ANTENNA_RADIUS = parsed_args.antenna_rad if ANTENNA_RANGE_FLAG else np.inf
    MODEL_TYPE = parsed_args.model_type
    TALK_ALIKE_FLAG = parsed_args.talk_alike_flag
    SAME_L_BOUNDS_FLAG = parsed_args.same_l_bounds_flag
    KMAX = parsed_args.kmax if parsed_args.kmax > 0 else 4
    L1_BOUNDS = [parsed_args.l_bounds[0], parsed_args.l_bounds[1]]
    L2_BOUNDS = [parsed_args.l_bounds[2], parsed_args.l_bounds[3]]

    # System Read Only Parameters to set in ROS
    LOCALISE_TARGETS_FLAG = True
    UPDATE_EID_FLAG = False
    SAVE_IMAGES_FLAG = False
    IMAX = np.inf

    # Set up the agent -----------------------------------------------------------------------------
    
    # ===== Dynamics Model =====
    # Single integrator model ----
    if MODEL_TYPE == "SingleIntegrator":
        # SingleIntegrator(dt=0.002)
        u_limits_init = np.array([[-1, 1], [-1, 1]])
        u_limits = u_limits_init; time_to_apply_ulimits = 0 # [s] after which to switch u_limits
        u_nominal = None
        INF_BUF_FLAG = True     # Whether to use infinite states buffer for ck calculation
        Q_ = 1
        R_ = 0.001
        PREDICTION_DT = 0.002 * 25
        RELAX_FACTOR = 1
        IMAX = 100e3
        TS = 0.01; T_H = 0.1; deltaT_ERG = 0.25 * 5
        BAR_WEIGHT = 0
        UPDATE_EID_FREQ = 110  # How often to update the EID phi function (30 means every 30 ergodic iterations) (or 30 x Ts [s])
        CBF_SKIP_ITER = 8            # How often to apply the CBF safety filter (every n iterations). Skipping some cause it takes time
        DELTA_SAFE = 0.1; ALPHA_HDOT = 100; ALPHA_H = 20; KAPPA_WALL = 0.5; RHO_WALL = 1.5
        KAPPA_OBS = 1; RHO_OBS = 0.15

        dynamic_model = SingleIntegrator(dt=0.0012, x0=[INIT_POS_2D[0], INIT_POS_2D[1]])
        
        print("--> Using model: <SingleIntegrator>")

    # # Double integrator model ----
    elif MODEL_TYPE == "DoubleIntegrator":
        # DoubleIntegrator(dt=0.0012, x0=[9, 3, 0, 0], damping=2)
        ULIM = 50 # 30
        u_limits_init = np.array([[-ULIM, ULIM], [-ULIM, ULIM]])
        u_limits = u_limits_init; time_to_apply_ulimits = 0 # [s] after which to switch u_limits
        u_nominal = None
        INF_BUF_FLAG = True         # Whether to use infinite states buffer for ck calculation
        Q_ = 8
        R_ = 0.001
        RELAX_FACTOR = 0.95         # U = RF * u + (1-RF) * u_prev
        TS = 0.03; T_H = 0.5; deltaT_ERG = 3
        SIMUL_DT = 0.0012
        PREDICTION_DT = SIMUL_DT * 5  # model dt * 5
        BAR_WEIGHT = 0
        UPDATE_EID_FREQ = 110*2*3*4  # How often to update the EID phi function (30 means every 30 ergodic iterations) (or 30 x Ts [s])
        CBF_SKIP_ITER = 8            # How often to apply the CBF safety filter (every n iterations). Skipping some cause it takes time
        DELTA_SAFE = 0.1; ALPHA_HDOT = 100; ALPHA_H = 20; KAPPA_WALL = 0.5; RHO_WALL = 1.5
        KAPPA_OBS = 1; RHO_OBS = 0.15
        
        dynamic_model = DoubleIntegrator(dt=SIMUL_DT, x0=[INIT_POS_2D[0], INIT_POS_2D[1], 0, 0], damping=2)

        print("--> Using model: <DoubleIntegrator>")
    
    # Simple Boat Second Order model ----
    elif MODEL_TYPE == "SimpleBoatSecondOrder":
        # SimpleBoatSecondOrder(dt=0.001, x0=None, m=3.0, Iz=0.25, d_v=5.0, d_w=2.0, k_delta=4.0)
        u_limits_init = np.array([[-1, 0], [-4, 4]])
        u_limits = np.array([[-5, 0], [-4, 4]]); time_to_apply_ulimits = 10 # [s] after which to switch u_limits
        u_nominal = None
        INF_BUF_FLAG = True         # Whether to use infinite states buffer for ck calculation
        Q_ = 8
        R_ = 0.001
        RELAX_FACTOR = 0.95         # U = RF * u + (1-RF) * u_prev
        TS = 0.03*5; T_H = 0.5; deltaT_ERG = 2
        SIMUL_DT = 0.0012
        PREDICTION_DT = SIMUL_DT * 5  # model dt * 5
        BAR_WEIGHT = 0
        UPDATE_EID_FREQ = 110*2*3*4  # How often to update the EID phi function (30 means every 30 ergodic iterations) (or 30 x Ts [s])
        CBF_SKIP_ITER = 8            # How often to apply the CBF safety filter (every n iterations). Skipping some cause it takes time
        DELTA_SAFE = 0.1; ALPHA_HDOT = 100; ALPHA_H = 20; KAPPA_WALL = 0.5; RHO_WALL = 1.5
        KAPPA_OBS = 1; RHO_OBS = 0.6

        dynamic_model = SimpleBoatSecondOrder(dt=SIMUL_DT, x0=[INIT_POS_2D[0], INIT_POS_2D[1], -0.39, 0, 0])

        print("--> Using model: <SimpleBoatSecondOrder>")

    # Simple Car Second Order model ----
    elif MODEL_TYPE == "SimpleCarSecondOrder":
        # SimpleCarSecondOrder(dt=0.001, x0=None, m=8.0, L=0.9, b_v=1.0, d_v=5.0, k_delta=20.0, k_steer=5.0, Iz=0.8, d_r=1.0, u_epsilon=1e-2, max_allowed_rev_thr=-1, steer_priority=0.004)
        u_limits_init = np.array([[-1, 0], [-10, 10]])
        u_limits = np.array([[-10, 0], [-10, 10]]); time_to_apply_ulimits = 15 # [s] after which to switch u_limits
        u_nominal = None
        INF_BUF_FLAG = True         # Whether to use infinite states buffer for ck calculation
        Q_ = 8
        R_ = 0.001
        RELAX_FACTOR = 0.95         # U = RF * u + (1-RF) * u_prev
        TS = 0.03*5; T_H = 0.5; deltaT_ERG = 2
        SIMUL_DT = 0.0012 * 2
        PREDICTION_DT = SIMUL_DT * 5  # model dt * 5
        BAR_WEIGHT = 0
        UPDATE_EID_FREQ = 110*2*3*4  # How often to update the EID phi function (30 means every 30 ergodic iterations) (or 30 x Ts [s])
        CBF_SKIP_ITER = 8            # How often to apply the CBF safety filter (every n iterations). Skipping some cause it takes time
        DELTA_SAFE = 0.1; ALPHA_HDOT = 100; ALPHA_H = 20; KAPPA_WALL = 0.5; RHO_WALL = 1.5
        KAPPA_OBS = 1; RHO_OBS = 0.75

        dynamic_model = SimpleCarSecondOrder(dt=SIMUL_DT, x0=[INIT_POS_2D[0], INIT_POS_2D[1], -0.39, 0, 0, 0],
                                                m=8.0, L=0.9, b_v=1.0, d_v=5.0, k_delta=20.0, k_steer=5.0, Iz=0.8, d_r=1.0, u_epsilon=1e-2, 
                                                max_allowed_rev_thr=-1, steer_priority=0.004) # max = -0.5

        print("--> Using model: <SimpleCarSecondOrder>")

    # Quadrotor model -----------
    elif MODEL_TYPE == "Quadcopter":
        print("ERROR: Quadcopter model needs work, not available yet.")
        # Stop the program, exit main
        sys.exit(1)
    else:
        print("ERROR: Unsupported model type.")
        sys.exit(2)
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

    # Agent - Ergodic Controller -------------
    # ROS Initialization
    rclpy.init(args=ros_args)

    # Generate Agent and connect to an ergodic controller object
    agent = Agent(L1_BOUNDS=L1_BOUNDS, L2_BOUNDS=L2_BOUNDS, Kmax=KMAX, 
                  dynamics_model=dynamic_model,
                  agent_id=AGENT_ID, antenna_rad=ANTENNA_RADIUS, antenna_range_flag=ANTENNA_RANGE_FLAG,
                  same_l_bounds_flag=SAME_L_BOUNDS_FLAG,
                  phi=lambda s: 1/100) 
                #   phi=createPhiFunc(L1_BOUNDS=L1_BOUNDS, L2_BOUNDS=L2_BOUNDS))      

    agent.erg_c = DecentralisedErgodicController(agent, uNominal=u_nominal, Q=Q_, R = R_ * np.eye(agent.model.num_of_inputs), uLimits=u_limits_init,
                                                 T_sampling=TS, T_horizon=T_H, deltaT_erg=deltaT_ERG,
                                                 use_inf_buffer=INF_BUF_FLAG,
                                                 barrier_weight=BAR_WEIGHT, barrier_eps=0.05, barrier_pow=2)
    
    # System Read Only Parameters to set in ROS
    from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
    descriptor = ParameterDescriptor(read_only=True, description='Read only')
    agent.declare_parameter('id', AGENT_ID, descriptor=descriptor)
    agent.declare_parameter('localise_targets_flag', LOCALISE_TARGETS_FLAG, descriptor=descriptor)
    agent.declare_parameter('update_eid_flag', UPDATE_EID_FLAG, descriptor=descriptor)
    agent.declare_parameter('save_images_flag', SAVE_IMAGES_FLAG, descriptor=descriptor)
    agent.declare_parameter('imax', IMAX, descriptor=descriptor)
    agent.declare_parameter('init_position_2d', INIT_POS_2D.tolist(), descriptor=descriptor)
    agent.declare_parameter('model_type', agent.model.type, descriptor=descriptor)

    # Declare antenna_radius parameter (not read-only) and update agent's antenna_rad
    agent.declare_parameter('antenna_radius', ANTENNA_RADIUS)
    agent.antenna_rad = agent.get_parameter('antenna_radius').value
    agent.declare_parameter('antenna_range_flag', ANTENNA_RANGE_FLAG)
    agent.antenna_range_flag = agent.get_parameter('antenna_range_flag').value

    agent.declare_parameter('talk_alike_flag', TALK_ALIKE_FLAG)
    agent.talk_alike_flag = TALK_ALIKE_FLAG

    agent.declare_parameter('same_l_bounds_flag', SAME_L_BOUNDS_FLAG)
    agent.same_l_bounds_flag = SAME_L_BOUNDS_FLAG

    # Add parameter callback to update antenna_rad when parameter changes
    def parameterCallback(params):
        for param in params:
            if param.name == 'antenna_radius':
                agent.antenna_rad = param.value
                agent.get_logger().info(f'Updated antenna_rad to: {agent.antenna_rad}')
            if param.name == 'antenna_range_flag':
                agent.antenna_range_flag = param.value
                if param.value == True:
                    agent.get_logger().info("Enabling Antenna Range for this agent...")
                else:
                    agent.get_logger().info("Disabling Antenna Range for this agent...")
            if param.name == 'talk_alike_flag':
                agent.talk_alike_flag = param.value
                if param.value == True:
                    agent.get_logger().info("Enabling Talk Alike for this agent...")
                else:
                    agent.get_logger().info("Disabling Talk Alike for this agent...")
            if param.name == 'same_l_bounds_flag':
                agent.same_l_bounds_flag = param.value
                if param.value == True:
                    agent.get_logger().info("Enabling Same L Bounds filtering for this agent...")
                else:
                    agent.get_logger().info("Disabling Same L Bounds filtering for this agent...")

        return SetParametersResult(successful=True)

    agent.add_on_set_parameters_callback(parameterCallback)

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
            x_pos = agent.L1_min + margin_x * agent.L1_size + i * (agent.L1_size - 2 * margin_x * agent.L1_size) / (grid_size - 1)
            y_pos = agent.L2_min + margin_y * agent.L2_size + j * (agent.L2_size - 2 * margin_y * agent.L2_size) / (grid_size - 1)
            obs_grid.append(Obstacle(pos=[x_pos, y_pos], dimensions=0.6, obs_type='circle', kappa=KAPPA_OBS, rho0=RHO_OBS, obs_name=f"Obstacle {i*grid_size + j + 1}"))


    # Avoiding Walls ----------------------
    obs_walls  = [Obstacle(pos=[agent.L1_min + agent.L1_size/2, agent.L2_min                  ],  dimensions=[0, +1], obs_type='wall', kappa=KAPPA_WALL, rho0=RHO_WALL, obs_name="Bottom Wall"),
                  Obstacle(pos=[agent.L1_min + agent.L1_size/2, agent.L2_max                  ],  dimensions=[0, -1], obs_type='wall', kappa=KAPPA_WALL, rho0=RHO_WALL, obs_name="Top Wall"   ),
                  Obstacle(pos=[agent.L1_min,                   agent.L2_min + agent.L2_size/2],  dimensions=[+1, 0], obs_type='wall', kappa=KAPPA_WALL, rho0=RHO_WALL, obs_name="Left Wall"  ),
                  Obstacle(pos=[agent.L1_max,                   agent.L2_min + agent.L2_size/2],  dimensions=[-1, 0], obs_type='wall', kappa=KAPPA_WALL, rho0=RHO_WALL, obs_name="Right Wall" )]

    # Save obstacles to memory
    # saveObstaclesToMemory(agent, obs_list=obs_grid)
    saveObstaclesToMemory(agent, obs_list=obs_walls)
    # saveObstaclesToMemory(agent, obs_list=[Obstacle(pos=[5, 5],   dimensions=[10, 10], obs_type='rectangle', kappa=KAPPA_OBS, rho0=RHO_OBS, obs_name="Obstacle 1")])

    # Print uNominal Status
    # print(agent_1.erg_c.uNominal)
    
    # Visualize H-field
    # vis.visHfield(agent, L_limits=[agent.L1_min-0.5, agent.L1_max+0.5, agent.L2_min-0.5, agent.L2_max+0.5], delta=DELTA_SAFE, num_of_points=200)

    # Lets now update the phi_function to take into account the obstacles
    agent.basis.phi = agent.modifedPhiForObstacles(agent.basis.phi, obs_to_exclude="None")
    agent.basis.precalcAllPhiK()

    # Lets visualise the origial phi side by side with the reconstructed one using Kmax
    # phi_rec = ReconstructedPhi(agent.basis, precalc_phik=False)
    # vis.plotPhiOnlyOriginalAndReconstructed(agent, phi_rec_from_agent=phi_rec, grid_res=100, clip_to_min_max=False)
    # plt.show()
    # sys.exit(0)

    # --------------------------------------------------------------------------------------------------

    def _simulationFunction():
        # Lists to store for plotting - now for single agent
        states_list = [agent.model.state.copy()]
        time_list = [0]  # Time vector
        u_list = [np.zeros((agent.model.num_of_inputs,))]
        u_before = np.zeros((agent.model.num_of_inputs,))  # Previous control action
        u_safe_list = [np.zeros((agent.model.num_of_inputs,))]
        erg_cost_list = []
        state_target_list = [agent.model._state_target.copy()] if isinstance(agent.model, Quadcopter) else []
        delta_t_Ts = []
        draw_plot_flag = False  # Flag that alternates when updating EID to plot
        target_data = {i: {'times': [], 'positions': [], 'sigmas': []} for i in range(max(1, len(agent.ekfs)))}
        
        ti = time_list[0]; ti_indx = 0
        Ts_iter = int(agent.erg_c.Ts / agent.model.dt)  # Number of iterations per sampling time
        
        # Initialize timing variables
        initial_time = time.time()
        last_iter_time = time.time()
        delta_time = 1
        
        i = 0
        while i < IMAX and not shutdown_flag.is_set():
            # if i == 5000:
            #     agent.real_target_positions.append(np.array([0.1, 0.1, 0]))
                
            # If multiple of Ts, calculate ergodic action
            if not agent.limits_changed_flag and time_list[-1] >= time_to_apply_ulimits:
                agent.erg_c.uLimits = u_limits
                agent.limits_changed_flag = True
                agent.get_logger().info(f"Switching to full u_limits: {u_limits} at t = {time_list[-1]:.2f} s")

            if i % Ts_iter == 0:
                ti = time_list[i]; ti_indx = i
                
                agent.time_since_start = ti

                # Ergodic Control Calculation ---------------------------------------------
                # Calculate ergodic control for the sample step
                us, tau, lamda_dur, erg_cost = agent.erg_c.calcNextActionTriplet(time_list[i], prediction_dt=PREDICTION_DT)



                # Calculate the average CK table of all discovered agents (except the self)
                other_agent_ck_data = [ck for aid, ck in agent.getAgentCkData().items() if aid != agent.agent_id and ck is not None]
                # Append to the list above this agents ck
                other_agent_ck_data.append(agent.basis.ck)
                ck_total = np.mean(other_agent_ck_data, axis=0)
                # with this now lets calculate total ergodic cost
                agent.erg_c.total_erg_cost = agent.erg_c.calcErgodicCost(ck_total)

                # If we have limited antenna range for communication enabled
                if agent.antenna_range_flag:
                    in_range_ck_data = [ck for aid, ck in agent.getAgentCkData(in_range_only=True).items() if aid != agent.agent_id and ck is not None]
                    if len(in_range_ck_data) > 0:
                        agent.erg_c.ck_aver_others = np.mean(in_range_ck_data, axis=0)
                    else:
                        # If no agent in the neighborhood use own ck as average for in-rangers
                        agent.erg_c.ck_aver_others = agent.basis.ck
                    ck_total_in_range = np.mean(in_range_ck_data + [agent.basis.ck], axis=0) if len(in_range_ck_data) > 0 else agent.basis.ck
                    agent.erg_c.total_erg_cost_in_range = agent.erg_c.calcErgodicCost(ck_total_in_range)
                else:
                    if len(other_agent_ck_data) > 0:
                        agent.erg_c.ck_aver_others = ck_total
                    agent.erg_c.total_erg_cost_in_range = agent.erg_c.total_erg_cost
                # Reset initial cost for calculations later
                if agent.erg_c.init_erg_cost == -1:
                    agent.erg_c.init_erg_cost = agent.erg_c.total_erg_cost_in_range


                # change lamda dur only if not quadcopter
                if not isinstance(agent.model, Quadcopter):
                    lamda_dur = agent.erg_c.Ts
                erg_cost_list.append(erg_cost)
                delta_t_Ts.append([ti, delta_time / agent.erg_c.Ts])

                # Debug State Information Printing
                if i % 160 == 0:
                    def u_str(u):
                        res = "["
                        for j in range(len(u)):
                            res += f"{u[j]:.2f}, "
                        return res[:-2] + "]"
                    
                    expected_time_max = IMAX/(i+1)*(time.time()-initial_time); elapsed_time_max = time.time()-initial_time
                    rem_time_max = expected_time_max - elapsed_time_max     # rem_time_simple = delta_time * (IMAX-i)/Ts_iter
                    # print(f"ti = {ti:.2f} s\t Erg cost: {erg_cost:.2f} \t i: {i}/{IMAX:.0f} \t perc: {i/IMAX:.2%} \t Δt/Ts: {delta_time/agent.erg_c.Ts:.2f}\t remaining: {rem_time_max:.0f} s\t elapsed: {time.time()-initial_time:.1f} s ({time.time()-initial_time + delta_time * (IMAX-i)/Ts_iter:.0f} s) ({IMAX/(i+1)*(time.time()-initial_time):.0f} s)")
                    # print(f"{agent.model.state_string} \n u = {u_str(us)} \t (tau - ti)/T = {(tau - ti)/agent.erg_c.T:.1%} \t lamda_dur = {lamda_dur:.4f} \t lamda/Ts = {lamda_dur/agent.erg_c.Ts:.2%}\n")
                    agent.get_logger().info(f"ti = {ti:.2f} s\t Erg cost: {erg_cost:.2f} \t i: {i}/{IMAX:.0f} \t perc: {i/IMAX:.2%} \t Δt/Ts: {delta_time/agent.erg_c.Ts:.2f}\t remaining: {rem_time_max:.0f} s\t elapsed: {time.time()-initial_time:.1f} s ({time.time()-initial_time + delta_time * (IMAX-i)/Ts_iter:.0f} s) ({IMAX/(i+1)*(time.time()-initial_time):.0f} s)\n"
                                             f"{agent.model.state_string} \n u = {u_str(us)} \t (tau - ti)/T = {(tau - ti)/agent.erg_c.T:.1%} \t lamda_dur = {lamda_dur:.4f} \t lamda/Ts = {lamda_dur/agent.erg_c.Ts:.2%}\n")

                # Debug print if agent inside boundaries
                agent.withinBounds(agent.model.state[:2])
                if np.any(np.abs(agent.model.state[:2]) > 50):
                    agent.get_logger().fatal(f"Agent WAYY out of bounds! Stopping simulation.")
                    break
                
                # Update the action mask
                if lamda_dur > 0:
                    agent.erg_c.action_mask.pushAction(ti, tau, lamda_dur, us.copy())

                # Calculate total ergodic cost from average ck of all agents
                # ck_total = np.mean([agent.basis.ck for agent in agent_list], axis=0)
                # total_erg_cost = agent_list[0].erg_c.calcErgodicCost(ck_total)  # Use any agent since they're all the same
                # total_erg_cost_list.append(total_erg_cost)

                # Multi-Target EKF update -------------------------------------------------
                if LOCALISE_TARGETS_FLAG:
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
                            target_data[new_target_id] = {'times': [], 'positions': [], 'sigmas': []}

                    # Store data using dictionary structure
                    for meas_id in range(len(agent.ekfs)):
                        # Ensure the target exists in target_data
                        if meas_id not in target_data:
                            target_data[meas_id] = {'times': [], 'positions': [], 'sigmas': []}
                        
                        target_data[meas_id]['times'].append(time_list[i])
                        target_data[meas_id]['positions'].append(agent.ekfs[meas_id].a_k_1.copy())
                        target_data[meas_id]['sigmas'].append(agent.ekfs[meas_id].sigma_k_1.copy())

                    # Check if we need to merge targets
                    if agent.num_of_targets > 1:
                        # Chack Bhattacharyya Distance between every pair and merge as needed
                        agent.mergeTargetsIfNeeded(MERGE_THRESHOLD=3, EUCL_DIST_THRESHOLD=0.15, SIMILAR_MEASUREMENTS_ANGLE_THRESHOLD_RAD=30* np.pi/180)

                    # Also, lets check and remove outdated target estimates
                    agent.searchAndRemoveOldTargetEstimates(current_time=ti, MAX_AGE_SEC=60)

                # Simulation saving file ----------------------------
                if (draw_plot_flag or i % 160/2 == 0) and SAVE_IMAGES_FLAG:
                    # Use agent for visualization
                    x_traj, _, _ = agent.model.simulateForward(x0=agent.model.state, ti=ti, udef=agent.erg_c.uNominal, T=agent.erg_c.T, dt=PREDICTION_DT)
                    erg_traj = x_traj[:, :2] # Only take the ergodic dimensions
                    if INF_BUF_FLAG:
                        ck_ = agent.basis.calcCkCoeffRecursive(erg_traj, ti, agent.erg_c.T, agent.erg_c.Ts, agent.erg_c.t0_erg, x_buffer=agent.erg_c.past_states_buffer.get(), update_ck_old=False)
                    else:
                        ck_ = agent.basis.calcCkCoeff(erg_traj, x_buffer=agent.erg_c.past_states_buffer.get() ,ti=ti, T=agent.erg_c.T)
                    phi_rec_from_ck = ReconstructedPhiFromCk(agent.basis, ck_)
                    print("Plotting phi")
                    phi_3_ = ReconstructedPhi(agent.basis, precalc_phik=False)
                    # For single agent, pass the single trajectory
                    vis.plotPhi(agent, phi_rec_from_ck=phi_rec_from_ck, phi_rec_from_agent=phi_3_, all_traj=[states_list[:, :2]], grid_res=40, ck_total=agent.basis.ck, agent_list=[agent])
                    plt.savefig(f"images/phiQuadWithObs_{ti:.2f}.png")
                    print(f"Saved image to images/phiQuadWithObs_{ti:.2f}.png")
                    plt.close()
                    draw_plot_flag = False
            
            # Continue with simulation of agent
            us_ = agent.erg_c.action_mask.readAction(t_now=time_list[i])
            
            if us_ is not None:
                # Apply the control action to the agent's model
                u = us_
            else:
                # If no ergodic control is available, use the nominal control
                u = agent.erg_c.uNominal(agent.model.state, time_list[i])

            # Lets apply the CBF Safety Filter to ergodic output
            # Apply safety control every 5 iterations or every one if we were near an obstacle before
            if np.any(u_safe_list[-1] != 0) or i%CBF_SKIP_ITER == 0:
                u_safe = agent.calcUsafe(agent.model.state, u, alpha_1=ALPHA_HDOT, alpha_2=ALPHA_H, delta=DELTA_SAFE)
            else:
                u_safe = np.zeros((agent.model.num_of_inputs,))
            u_safe_list.append(u_safe.copy())  # Store the safe control action for later use
            u += u_safe.copy()  # Add the safe control action to the ergodic control action
            
            # Lets clip again to agents control input limits
            u = np.clip(u, agent.erg_c.uLimits[:, 0], agent.erg_c.uLimits[:, 1])
            # If u_magnitude is more than 100, print a warning
            if np.linalg.norm(u) > 100:
                agent.get_logger().warn(f"CRITICAL: Agent control action is too high: {u}")

            # Lets smooth out with the previous control action
            u = RELAX_FACTOR * u + (1-RELAX_FACTOR) * u_before  # Smooth the control action
            u_before = u.copy()

            # ROS Send data to data topic
            if not shutdown_flag.is_set() and i % 30 == 0:
                agent.publishData(state_now=agent.model.state, u_input_now=u, erg_cost_now=erg_cost, 
                                  active_cbf_flag=True if int(np.any(u_safe != 0)) == 1 else False,
                                  time_now=time_list[i])

            # TODO: Here we should simulate forward for simulation_dt with a dt, instead of stepping. Implement model simulation function
            agent.model.state = agent.model.step(agent.model.state, u)         # Step the model with the control action
            agent.erg_c.past_states_buffer.push(agent.model.state.copy()[:2])  # Store the state in the buffer
            if INF_BUF_FLAG:
                # In inf buffer case, we have 2 buffers: one for past "ts" states and one for past "t0_erg" states. 
                # We could use one being the bigger of the two, but that complicates things. No need since past "ts" states are not much.
                agent.erg_c.past_erg_history_buffer.push(agent.model.state.copy()[:2])

            # Store states and control for plotting later
            u_list.append(u.copy())
            states_list.append(agent.model.state.copy())
            state_target_list.append(agent.model._state_target_history_for_plotting.copy() if isinstance(agent.model, Quadcopter) else [])

            # Lets update phi(x) if needed
            if i%(Ts_iter * UPDATE_EID_FREQ) == 0 and UPDATE_EID_FLAG:
                t_ = time.time()
                agent.get_logger().info("Updating phi...")
                # TODO: Globalize the parameters used here
                agent.updateEIDphiFunction(NUM_GAUSS_POINTS=10, P_UPPER_LIM=8, HTA_SCALE=8e-8, FINAL_FI_CLIP=0.01, ALWAYS_ADD=0)
                agent.get_logger().info(f"Updated phi in {time.time()-t_:.2f} s")
                draw_plot_flag = True
                # Restart ergodic memory buffer
                if INF_BUF_FLAG:
                    agent.erg_c.t0_erg = max(time_list[i] - agent.erg_c.deltaT_erg, 0)  # Reset the t0_erg to the current time minus the deltaT_erg
                    
                    agent.basis.ck_bar_old = agent.basis.calcCkCoeff(agent.erg_c.past_erg_history_buffer.get(), agent.erg_c.t0_erg, agent.erg_c.deltaT_erg, do_not_divide_integral_flag=True)
                    agent.basis.ck_bar_old /= (time_list[i] + agent.erg_c.T - agent.erg_c.t0_erg)
                else:
                    # Here we need to reset the buffer since it can contain a lot of outdated past state information # TODO: Check if needed
                    agent.erg_c.past_states_buffer.reset(last_perc_to_keep=0.1)  # Reset the past states buffer
                # Reset initial cost
                agent.erg_c.init_erg_cost = -1


            # Store states for plotting later etc --------------------
            # (States and control already stored above)
            time_list.append(time_list[i] + agent.model.dt)  # Use agent's dt
            
            # Calculate delta time for this iteration
            current_time = time.time()
            # delta_time is for when we calculated ergodic control. Otherwise we dont care, its fast
            delta_time = current_time - last_iter_time if (i%Ts_iter == 0) else delta_time
            last_iter_time = current_time
            
            i += 1

        # Check if simulation ended due to shutdown signal
        if shutdown_flag.is_set():
            agent.get_logger().info("Simulation terminated by user interrupt.")
        else:
            agent.get_logger().info(f"Simulation finished in {time.time()-initial_time:.2f} s")

        # Convert lists to numpy arrays for plotting
        states_list = np.array(states_list)
        u_list = np.array(u_list)
        time_list = np.array(time_list)
        state_target_list = np.array(state_target_list) if len(state_target_list) > 0 else []
        delta_t_Ts = np.array(delta_t_Ts)
        u_safe_list = np.array(u_safe_list)

    # Start the simulation in a background thread
    sim_thread = threading.Thread(target=_simulationFunction, daemon=True)
    sim_thread.start()

    try:
        # Spin the ROS node to keep it alive
        while rclpy.ok() and not shutdown_flag.is_set():
            try:
                rclpy.spin_once(agent, timeout_sec=0.1)
            except rclpy.executors.ExternalShutdownException:
                agent.get_logger().warn("ROS external shutdown detected.")
                break
    except KeyboardInterrupt:
        agent.get_logger().warn("\nKeyboard interrupt received in main thread.")
    finally:
        # Signal shutdown to all threads
        shutdown_flag.set()
        
        # Wait for simulation thread to finish
        if sim_thread.is_alive():
            agent.get_logger().info("Waiting for simulation thread to finish...")
            sim_thread.join(timeout=3.0)  # Wait up to 3 seconds
            if sim_thread.is_alive():
                agent.get_logger().error("Warning: Simulation thread did not finish cleanly.")

        # Clear up with ROS
        try:
            if rclpy.ok():
                agent.destroy_node()
        except Exception as e:
            agent.get_logger().error(f"Error destroying node: {e}")

        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception as e:
            agent.get_logger().error(f"Error shutting down RCL: {e}")

        agent.get_logger().info("Cleanup complete. Goodbye!")

# -----------------------------------------------------------------------------------

# Add at the end of file
if __name__ == "__main__":
    main()