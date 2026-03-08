import os
import numpy as np
import threading
import signal
import sys

import rclpy
import rclpy.parameter
import argparse

# TODO: Integral of Phi should be = 1, and phi!=0 everywhere on the domain. Make sure EID updates respect that
def createPhiFunc(L1_BOUNDS=[0.0, 10.0], L2_BOUNDS=[0.0, 10.0], bumps_config_raw=None, base_level=0.01):
    """
    Creates a normalized phi function with multiple Gaussian bumps.
    
    Args:
        L1_BOUNDS: [min, max] bounds for the first dimension
        L2_BOUNDS: [min, max] bounds for the second dimension
        bumps_config_raw: List of bump configurations, each bump is a dict with:
            - x_pos_perc: x position as percentage of L1 range (0.0 to 1.0)
            - y_pos_perc: y position as percentage of L2 range (0.0 to 1.0)
            - height: height/amplitude of the bump
            - width_scale: width scaling factor (multiplied by width_base)
        base_level: Base level added to bump values to ensure some exploration
                   happens everywhere, not just at bump locations (default: 0.01)
            
    Returns:
        Normalized phi function
    """
    
    L1_min, L1_max = L1_BOUNDS
    L2_min, L2_max = L2_BOUNDS
    L1_range = L1_max - L1_min
    L2_range = L2_max - L2_min
    
    # Width scaling factor: smaller width = wider bump
    # For bump visible over ~20% of domain, use: width_scale / L_range²
    width_base = 100.0 / (L1_range * L2_range)  # Scales with domain area
    
    # Default bump configuration if none provided
    if bumps_config_raw is None:
        bumps_config_raw = [
            {'x_pos_perc': 0.7, 'y_pos_perc': 0.3, 'height': 5, 'width_scale': 0.7}
        ]
    
    # Convert percentage-based config to absolute positions
    # Bump configuration: (x_pos, y_pos, height, width)
    # width controls sharpness: larger = sharper/narrower, smaller = wider/smoother
    bumps_config = []
    for bump in bumps_config_raw:
        x_pos = L1_min + bump['x_pos_perc'] * L1_range
        y_pos = L2_min + bump['y_pos_perc'] * L2_range
        height = bump['height']
        width = width_base * bump['width_scale']
        bumps_config.append((x_pos, y_pos, height, width))
    
    def phiUnnormalized(s):
        x, y = s[0], s[1]
        bumps = sum(h * np.exp(-w * ((x-px)**2 + (y-py)**2)) 
                   for px, py, h, w in bumps_config)
        return bumps + base_level
    
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

def setupAgentConfig(parsed_args, L1_BOUNDS, L2_BOUNDS):
    """
    Load agent configuration from YAML and handle command line overrides.
    
    Returns:
        dict: Complete configuration dictionary with all parameters
    """
    import numpy as np
    from my_erg_lib.Utilities import loadAgentConfigFromYaml
    
    # Load agent configuration from YAML
    try:
        agent_config = loadAgentConfigFromYaml(parsed_args.agent_config, L1_BOUNDS, L2_BOUNDS)
    except KeyError as e:
        print(f"ERROR: Invalid agent configuration: {e}")
        sys.exit(3)
        
    if agent_config is None:
        print(f"ERROR: Failed to load agent configuration from {parsed_args.agent_config}")
        sys.exit(2)

    # Extract configuration values (with command line argument overrides)
    MODEL_TYPE = agent_config['model_type']
    if parsed_args.model_type is not None and parsed_args.model_type != MODEL_TYPE:
        MODEL_TYPE = parsed_args.model_type
        print(f"INFO: Model type overridden by command line: {MODEL_TYPE} (config had: {agent_config['model_type']})")
        
        # Warning and confirmation for model type override
        print()
        print("=" * 60)
        print("WARNING: Model type has been overridden!")
        print("=" * 60)
        print(f"Configuration file specifies: {agent_config['model_type']}")
        print(f"Command line override to:     {MODEL_TYPE}")
        print()
        print("Changing the model type can significantly affect:")
        print("  • Agent dynamics and behavior")
        print("  • Control parameters and limits")
        print("  • Required configuration parameters")
        print("  • System performance and stability")
        print()
        print("Make sure the current configuration is compatible")
        print("with the new model type before proceeding.")
        print("=" * 60)
        
        try:
            user_input = input("Do you want to continue with the overridden model type? (y/N): ").strip().lower()
            if user_input not in ['y', 'yes']:
                print("Operation cancelled by user. Exiting...")
                sys.exit(1)
            else:
                print(f"Proceeding with model type: {MODEL_TYPE}")
                print("=" * 60)
        except KeyboardInterrupt:
            print("\nOperation interrupted by user. Exiting...")
            sys.exit(1)
    
    # Get default values from config or use hardcoded defaults if not in config
    config_antenna_radius = agent_config.get('antenna_radius', np.inf)
    if 'antenna_radius' not in agent_config:
        print(f"INFO: Using hardcoded default for antenna_radius: {config_antenna_radius} \t (not specified in config)")
    
    config_kmax = agent_config.get('kmax', 4)
    if 'kmax' not in agent_config:
        print(f"INFO: Using hardcoded default for kmax: {config_kmax} \t (not specified in config)")
    
    config_antenna_range_flag = agent_config.get('antenna_range_flag', False)
    if 'antenna_range_flag' not in agent_config:
        print(f"INFO: Using hardcoded default for antenna_range_flag: {config_antenna_range_flag} \t (not specified in config)")
    
    config_talk_alike_flag = agent_config.get('talk_alike_flag', False)
    if 'talk_alike_flag' not in agent_config:
        print(f"INFO: Using hardcoded default for talk_alike_flag: {config_talk_alike_flag} \t (not specified in config)")
    
    config_same_l_bounds_flag = agent_config.get('same_l_bounds_flag', True)
    if 'same_l_bounds_flag' not in agent_config:
        print(f"INFO: Using hardcoded default for same_l_bounds_flag: {config_same_l_bounds_flag} \t (not specified in config)")
    
    ANTENNA_RADIUS = config_antenna_radius
    if parsed_args.antenna_rad is not None:
        ANTENNA_RADIUS = parsed_args.antenna_rad
        print(f"INFO: Antenna radius overridden by command line: {ANTENNA_RADIUS} (config had: {config_antenna_radius})")
    
    KMAX = config_kmax
    if parsed_args.kmax is not None and parsed_args.kmax != KMAX:
        KMAX = parsed_args.kmax
        print(f"INFO: KMAX overridden by command line: {KMAX} (config had: {config_kmax})")
    
    ANTENNA_RANGE_FLAG = config_antenna_range_flag
    if parsed_args.antenna_range_flag is not None and parsed_args.antenna_range_flag != ANTENNA_RANGE_FLAG:
        ANTENNA_RANGE_FLAG = parsed_args.antenna_range_flag
        print(f"INFO: Antenna range flag overridden by command line: {ANTENNA_RANGE_FLAG} (config had: {config_antenna_range_flag})")
    
    TALK_ALIKE_FLAG = config_talk_alike_flag
    if parsed_args.talk_alike_flag is not None and parsed_args.talk_alike_flag != TALK_ALIKE_FLAG:
        TALK_ALIKE_FLAG = parsed_args.talk_alike_flag
        print(f"INFO: Talk alike flag overridden by command line: {TALK_ALIKE_FLAG} (config had: {config_talk_alike_flag})")
    
    SAME_L_BOUNDS_FLAG = config_same_l_bounds_flag
    if parsed_args.same_l_bounds_flag is not None and parsed_args.same_l_bounds_flag != SAME_L_BOUNDS_FLAG:
        SAME_L_BOUNDS_FLAG = parsed_args.same_l_bounds_flag
        print(f"INFO: Same L bounds flag overridden by command line: {SAME_L_BOUNDS_FLAG} (config had: {config_same_l_bounds_flag})")
    
    # System parameters from config - all required
    try:
        LOCALISE_TARGETS_FLAG = agent_config['flags']['localise_targets']
        UPDATE_EID_FLAG = agent_config['flags']['update_eid']
        SAVE_IMAGES_FLAG = agent_config['flags']['save_images']
        IMAX = agent_config['system']['imax']
    except KeyError as e:
        print(f"ERROR: Missing required system flag or parameter in agent configuration: {e}")
        print(f"Please check your configuration file: {parsed_args.agent_config}")
        print("Required sections: 'flags' (localise_targets, update_eid, save_images) and 'system' (imax)")
        sys.exit(3)

    # Target positions from config - required
    try:
        REAL_TARGET_POSITIONS = agent_config['targets']['real_positions']
    except KeyError as e:
        print(f"ERROR: Missing required target configuration: {e}")
        print(f"Please check your configuration file: {parsed_args.agent_config}")
        print("Required section: 'targets' -> 'real_positions'")
        sys.exit(3)

    # EKF parameters from config - all required
    try:
        ekf_config = agent_config['targets']['ekf']
        EKF_PARAMS = {
            'sigma_init': ekf_config['sigma_init'],
            'R': ekf_config['R'],
            'Q': ekf_config['Q'],
            'a_limits': ekf_config['a_limits'],
            'sensor_range': ekf_config['sensor_range'],
            'sensor_R': ekf_config['sensor_R']
        }
    except KeyError as e:
        print(f"ERROR: Missing required EKF parameter in agent configuration: {e}")
        print(f"Please check your configuration file: {parsed_args.agent_config}")
        print("Required section: 'targets' -> 'ekf' with all EKF parameters")
        sys.exit(3)

    # Phi configuration - optional, defaults to uniform coverage
    phi_config = agent_config.get('phi', {})
    PHI_TYPE = phi_config.get('type', 'uniform')  # 'uniform' or 'gaussian_bumps'
    BASE_LEVEL = phi_config.get('base_level', 0.01)  # Base exploration level for gaussian_bumps
    
    BUMPS_CONFIG = None
    if PHI_TYPE == 'gaussian_bumps':
        bumps_raw = phi_config.get('bumps', None)
        if bumps_raw is None:
            print(f"WARNING: phi type is 'gaussian_bumps' but no 'bumps' configuration provided.")
            print(f"         Falling back to uniform coverage.")
            PHI_TYPE = 'uniform'
        else:
            # Validate and convert bumps configuration
            BUMPS_CONFIG = []
            for i, bump in enumerate(bumps_raw):
                try:
                    bump_entry = {
                        'x_pos_perc': bump['x_pos_perc'],
                        'y_pos_perc': bump['y_pos_perc'],
                        'height': bump['height'],
                        'width_scale': bump['width_scale']
                    }
                    # Validate percentage values
                    if not (0.0 <= bump_entry['x_pos_perc'] <= 1.0):
                        print(f"WARNING: Bump {i} x_pos_perc={bump_entry['x_pos_perc']} is outside [0, 1] range")
                    if not (0.0 <= bump_entry['y_pos_perc'] <= 1.0):
                        print(f"WARNING: Bump {i} y_pos_perc={bump_entry['y_pos_perc']} is outside [0, 1] range")
                    BUMPS_CONFIG.append(bump_entry)
                except KeyError as e:
                    print(f"ERROR: Bump {i} is missing required field: {e}")
                    print(f"       Each bump must have: x_pos_perc, y_pos_perc, height, width_scale")
                    sys.exit(3)
            print(f"INFO: Using gaussian_bumps phi coverage with {len(BUMPS_CONFIG)} bump(s), base_level={BASE_LEVEL}:")
            for i, bump in enumerate(BUMPS_CONFIG):
                print(f"       Bump {i+1}: pos=({bump['x_pos_perc']*100:.0f}%, {bump['y_pos_perc']*100:.0f}%), height={bump['height']}, width_scale={bump['width_scale']}")
    elif PHI_TYPE != 'uniform':
        print(f"WARNING: Unknown phi type '{PHI_TYPE}'. Using 'uniform' coverage.")
        PHI_TYPE = 'uniform'
    
    if PHI_TYPE == 'uniform':
        print(f"INFO: Using uniform phi coverage (constant density across domain)")

    print(f"Loaded configuration: Model={MODEL_TYPE}, Config file={parsed_args.agent_config}\n")
    
    # Return complete configuration
    return {
        'raw_config': agent_config,
        'MODEL_TYPE': MODEL_TYPE,
        'ANTENNA_RADIUS': ANTENNA_RADIUS,
        'KMAX': KMAX,
        'ANTENNA_RANGE_FLAG': ANTENNA_RANGE_FLAG,
        'TALK_ALIKE_FLAG': TALK_ALIKE_FLAG,
        'SAME_L_BOUNDS_FLAG': SAME_L_BOUNDS_FLAG,
        'LOCALISE_TARGETS_FLAG': LOCALISE_TARGETS_FLAG,
        'UPDATE_EID_FLAG': UPDATE_EID_FLAG,
        'SAVE_IMAGES_FLAG': SAVE_IMAGES_FLAG,
        'IMAX': IMAX,
        'REAL_TARGET_POSITIONS': REAL_TARGET_POSITIONS,
        'EKF_PARAMS': EKF_PARAMS,
        'PHI_TYPE': PHI_TYPE,
        'BUMPS_CONFIG': BUMPS_CONFIG,
        'BASE_LEVEL': BASE_LEVEL
    }

# -----------------------------------------------------------------------------------
def main(args=None):
    from my_erg_lib.agent import Agent
    from my_erg_lib.obstacles import Obstacle, saveObstaclesToMemory
    from my_erg_lib.model_dynamics import SingleIntegrator, DoubleIntegrator, Quadcopter, SimpleBoatSecondOrder, SimpleCarSecondOrder, FixedWing12DOFTrainer, FixedWing12DOFTrainerJAX
    from my_erg_lib.ergodic_controllers import DecentralisedErgodicController
    from my_erg_lib.basis import ReconstructedPhi, ReconstructedPhiFromCk
    import matplotlib.pyplot as plt
    import my_erg_lib.vis as vis
    from my_erg_lib.Utilities import loadObstaclesFromYaml
    import time

    # Lets parse arguments to get agent ID
    parser = argparse.ArgumentParser(description='Run agent node with dynamic ID')
    parser.add_argument('--agent_id',           type=int,            required=True,                  help='Agent ID to name the node')
    parser.add_argument('--init_pos',           type=float, nargs=4, required=False, default=[9, 3, 0, 0], help='Initial position as [x, y] ([x, y, z, yaw(deg)] if airplane)')
    parser.add_argument('--l_bounds',           type=float, nargs=4, required=False, default=[0, 10, 0, 10], help='Initial bounds as [x_min, x_max, y_min, y_max] for ergodic exploration')
    parser.add_argument('--agent_config',       type=str,            required=False, default='src/ergodic_exploration/agent_configs/default.yaml', help='Path to agent configuration YAML file')
    parser.add_argument('--model_type',         type=str,            required=False, default=None, help='Override model type from config (SingleIntegrator, DoubleIntegrator, etc.)')
    parser.add_argument('--antenna_rad',        type=float,          required=False, default=None, help='Override antenna radius in meters')
    parser.add_argument('--kmax',               type=int,            required=False, default=None, help='Override maximum Fourier modes for reconstruction')
    parser.add_argument('--obstacles_yaml',     type=str,                           required=False, default='None', help='Path to YAML file containing obstacle definitions')
    parser.add_argument('--antenna_range_flag', type=lambda x: x.lower() == 'true', required=False, default=None,   help='Override antenna range flag (true/false)')
    parser.add_argument('--talk_alike_flag',    type=lambda x: x.lower() == 'true', required=False, default=None,   help='Override weather to communicate only with similar model (true/false)')
    parser.add_argument('--same_l_bounds_flag', type=lambda x: x.lower() == 'true', required=False, default=None,   help='Override whether to communicate only with agents having same L bounds (true/false)')
    parser.add_argument('--show_init_phi',      type=lambda x: x.lower() == 'true', required=False, default=False,  help='Whether to show initial phi function (original + reconstructed) (true/false)')
    parser.add_argument('--sync_clocks_flag',   type=lambda x: x.lower() == 'true', required=False, default=False,  help='Enable clock synchronization across agents (true/false)')
    parser.add_argument('--sync_agent_ids',     type=int, nargs='*',                required=False, default=None,   help='List of agent IDs to sync with (e.g., --sync_agent_ids 1 2 3). If not provided, syncs with discovered agents.')
    parser.add_argument('--sync_freq',          type=int,                           required=False, default=None,   help='Sync every N iterations. Default syncs every Ts. Use smaller values (e.g., 50-100) for tighter sync during collision tests.')
    # Parse known args only, keep ROS args separate
    parsed_args, ros_args = parser.parse_known_args()  
    AGENT_ID = parsed_args.agent_id
    INIT_POS_3D = np.array(parsed_args.init_pos)
    L1_BOUNDS = [parsed_args.l_bounds[0], parsed_args.l_bounds[1]]
    L2_BOUNDS = [parsed_args.l_bounds[2], parsed_args.l_bounds[3]]
    OBSTACLES_YAML_PATH = parsed_args.obstacles_yaml
    SHOW_INIT_PHI = parsed_args.show_init_phi
    SYNC_CLOCKS_FLAG = parsed_args.sync_clocks_flag
    SYNC_AGENT_IDS = parsed_args.sync_agent_ids
    SYNC_FREQ = parsed_args.sync_freq

    # Load and setup agent configuration with override handling
    config = setupAgentConfig(parsed_args, L1_BOUNDS, L2_BOUNDS)
    agent_config = config['raw_config']
    
    # Extract final configuration values
    MODEL_TYPE = config['MODEL_TYPE']
    ANTENNA_RADIUS = config['ANTENNA_RADIUS'] 
    KMAX = config['KMAX']
    ANTENNA_RANGE_FLAG = config['ANTENNA_RANGE_FLAG']
    TALK_ALIKE_FLAG = config['TALK_ALIKE_FLAG']
    SAME_L_BOUNDS_FLAG = config['SAME_L_BOUNDS_FLAG']
    LOCALISE_TARGETS_FLAG = config['LOCALISE_TARGETS_FLAG']
    UPDATE_EID_FLAG = config['UPDATE_EID_FLAG']
    SAVE_IMAGES_FLAG = config['SAVE_IMAGES_FLAG']
    IMAX = config['IMAX']
    REAL_TARGET_POSITIONS = config['REAL_TARGET_POSITIONS']
    EKF_PARAMS = config['EKF_PARAMS']
    PHI_TYPE = config['PHI_TYPE']
    BUMPS_CONFIG = config['BUMPS_CONFIG']
    BASE_LEVEL = config['BASE_LEVEL']
    

    # ===== Dynamics Model =====
    # Load model configuration from YAML
    dynamics_config = agent_config['dynamics']
    control_config = agent_config['control']
    system_config = agent_config['system']
    
    # Extract common control parameters - all required
    try:
        u_limits_init = control_config['u_limits_init']
        u_limits = control_config['u_limits']
        time_to_apply_ulimits = control_config['time_to_apply_ulimits']
        u_nominal = None  # Will be set based on model type
        
        # Extract common ergodic control parameters - all required
        INF_BUF_FLAG = control_config['inf_buf_flag']
        Q_ = control_config['Q']
        R_ = control_config['R']
        PREDICTION_DT = dynamics_config['dt'] * control_config['prediction_dt_multiplier']
        RELAX_FACTOR = control_config['relax_factor']
        
        # Extract timing parameters - all required
        TS = control_config['ts']
        T_H = control_config['t_h']
        deltaT_ERG = control_config['delta_t_erg']
        
        # Extract optimization parameters - all required
        BAR_WEIGHT = control_config['bar_weight']
        UPDATE_EID_FREQ = control_config['update_eid_freq']
        
        # Extract safety parameters (CBF) - all required
        CBF_SKIP_ITER = control_config['cbf_skip_iter']
        DELTA_SAFE = control_config['delta_safe']
        # HOCBF Class-K function coefficients (relative degree 3)
        ALPHA_1 = control_config['alpha_1']
        ALPHA_2 = control_config['alpha_2']
        ALPHA_3 = control_config['alpha_3']
        # Input rate constraints
        ALPHA_U = control_config['alpha_u']
        CBF_KP = control_config['cbf_Kp']
        CBF_DT = control_config['cbf_dt']
        # Angle of Attack constraint parameters
        USE_AOA_CONSTRAINT = control_config.get('use_aoa_constraint', True)  # Default True for backward compatibility
        ALPHA_MAX_DEG = control_config['alpha_max_deg']
        ALPHA_AOA_1 = control_config['alpha_aoa_1']
        ALPHA_AOA_2 = control_config['alpha_aoa_2']
        SLACK_PENALTY_AOA = control_config['slack_penalty_aoa']
        KAPPA_WALL = control_config['kappa_wall']
        RHO_WALL = control_config['rho_wall']
        KAPPA_OBS = control_config['kappa_obs']
        RHO_OBS = control_config['rho_obs']
        KAPPA_OBS_VIRTUAL = control_config['kappa_obs_virtual']
        RHO_OBS_VIRTUAL = control_config['rho_obs_virtual']
        RADIUS_OBS_VIRTUAL = control_config.get('radius_obs_virtual', 5.0)  # Default 5.0 if not specified
        
        # Extract system parameters - all required
        PUBLISH_DATA_FREQ = system_config['publish_data_freq']
        
    except KeyError as e:
        print(f"ERROR: Missing required parameter in agent configuration: {e}")
        print(f"Please check your configuration file: {parsed_args.agent_config}")
        print("All parameters in 'dynamics', 'control', and 'system' sections are required.")
        sys.exit(3)
    
    # Initialize dynamic model based on configuration
    try:
        dt = dynamics_config['dt']
    except KeyError:
        print(f"ERROR: Missing required 'dt' parameter in dynamics configuration")
        print(f"Please check your configuration file: {parsed_args.agent_config}")
        sys.exit(3)
    
    # Single integrator model ----
    if MODEL_TYPE == "SingleIntegrator":
        dynamic_model = SingleIntegrator(dt=dt, x0=[INIT_POS_3D[0], INIT_POS_3D[1]])

    # Double integrator model ----
    elif MODEL_TYPE == "DoubleIntegrator":
        try:
            damping = dynamics_config['damping']
        except KeyError:
            print(f"ERROR: Missing required 'damping' parameter for DoubleIntegrator model")
            print(f"Please check your configuration file: {parsed_args.agent_config}")
            sys.exit(3)
        dynamic_model = DoubleIntegrator(dt=dt, x0=[INIT_POS_3D[0], INIT_POS_3D[1], 0, 0], damping=damping)
    
    # Simple Boat Second Order model ----
    elif MODEL_TYPE == "SimpleBoatSecondOrder":
        try:
            m = dynamics_config['m']
            Iz = dynamics_config['Iz']
            d_v = dynamics_config['d_v']
            d_w = dynamics_config['d_w']
            k_delta = dynamics_config['k_delta']
        except KeyError as e:
            print(f"ERROR: Missing required parameter '{e}' for SimpleBoatSecondOrder model")
            print(f"Please check your configuration file: {parsed_args.agent_config}")
            print("Required parameters: m, Iz, d_v, d_w, k_delta")
            sys.exit(3)
        
        dynamic_model = SimpleBoatSecondOrder(dt=dt, x0=[INIT_POS_3D[0], INIT_POS_3D[1], -0.39, 0, 0],
                                            m=m, Iz=Iz, d_v=d_v, d_w=d_w, k_delta=k_delta)

    # Simple Car Second Order model ----
    elif MODEL_TYPE == "SimpleCarSecondOrder":
        try:
            m = dynamics_config['m']
            L = dynamics_config['L']
            b_v = dynamics_config['b_v']
            d_v = dynamics_config['d_v']
            k_delta = dynamics_config['k_delta']
            k_steer = dynamics_config['k_steer']
            Iz = dynamics_config['Iz']
            d_r = dynamics_config['d_r']
            u_epsilon = dynamics_config['u_epsilon']
            max_allowed_rev_thr = dynamics_config['max_allowed_rev_thr']
            steer_priority = dynamics_config['steer_priority']
        except KeyError as e:
            print(f"ERROR: Missing required parameter '{e}' for SimpleCarSecondOrder model")
            print(f"Please check your configuration file: {parsed_args.agent_config}")
            print("Required parameters: m, L, b_v, d_v, k_delta, k_steer, Iz, d_r, u_epsilon, max_allowed_rev_thr, steer_priority")
            sys.exit(3)

        dynamic_model = SimpleCarSecondOrder(dt=dt, x0=[INIT_POS_3D[0], INIT_POS_3D[1], -0.39, 0, 0, 0],
                                           m=m, L=L, b_v=b_v, d_v=d_v, k_delta=k_delta, k_steer=k_steer, 
                                           Iz=Iz, d_r=d_r, u_epsilon=u_epsilon, 
                                           max_allowed_rev_thr=max_allowed_rev_thr, steer_priority=steer_priority)

    # Fixed Wing model
    elif MODEL_TYPE == "FixedWing12DOFTrainer" or MODEL_TYPE == "FixedWing12DOFTrainerJAX":
        try:
            v_trim = dynamics_config['v_trim']
            use_linear_f = dynamics_config['use_linear_f']
            use_linear_fx_fu = dynamics_config['use_linear_fx_fu']
        except KeyError as e:
            print(f"ERROR: Missing required parameter '{e}' for FixedWing12DOFTrainer model")
            print(f"Please check your configuration file: {parsed_args.agent_config}")
            print("Required parameters: v_trim, use_linear_f, use_linear_fx_fu")
            sys.exit(3)
        
        # Only X, Y, Z and Yaw (psi) can be changed from here. The others are overwritten by trim state. 0.053
        #                                                   x0=[x,              y,              z,              φ, θ,    ψ,                   u,   v,      w, p, q, r]
        if MODEL_TYPE == "FixedWing12DOFTrainer":
            dynamic_model = FixedWing12DOFTrainer(dt=dt,    x0=[INIT_POS_3D[0], INIT_POS_3D[1], INIT_POS_3D[2], 0, 0.12, INIT_POS_3D[3]*np.pi/180, v_trim, 0, 0, 0, 0, 0],
                                                v_trim=v_trim, use_linear_f=use_linear_f, use_linear_fx_fu=use_linear_fx_fu)
        else:
            dynamic_model = FixedWing12DOFTrainerJAX(dt=dt, x0=[INIT_POS_3D[0], INIT_POS_3D[1], INIT_POS_3D[2], 0, 0.12, INIT_POS_3D[3]*np.pi/180, v_trim, 0, 0, 0, 0, 0],
                                                v_trim=v_trim, use_linear_f=use_linear_f, use_linear_fx_fu=use_linear_fx_fu)
        
        # Add trim inputs to every input from now on using a nominal function
        try:
            u_nominal_config = control_config['u_nominal']
            if u_nominal_config == "trim":
                def _uNomAddTrim(x, t):
                    return dynamic_model.u_trim
                u_nominal = _uNomAddTrim
        except KeyError:
            print(f"ERROR: Missing required 'u_nominal' parameter for FixedWing12DOFTrainer model")
            print(f"Please check your configuration file: {parsed_args.agent_config}")
            print("Required parameter: u_nominal (should be 'trim' for FixedWing)")
            sys.exit(3)


    elif MODEL_TYPE == "Quadcopter":
        try:
            mass = dynamics_config['mass']
            damping = dynamics_config['damping']
            z_target = dynamics_config['z_target']
            motor_limits = dynamics_config['motor_limits']
            zero_out_states = dynamics_config.get('zero_out_states', None)
            Q_lqr = dynamics_config.get('Q_lqr', None)
            R_lqr = dynamics_config.get('R_lqr', None)
        except KeyError as e:
            print(f"ERROR: Missing required parameter '{e}' for Quadcopter model")
            print(f"Please check your configuration file: {parsed_args.agent_config}")
            print("Required parameters: mass, damping, z_target, motor_limits")
            sys.exit(3)
        
        # Convert Q_lqr and R_lqr lists to diagonal matrices if provided
        Q_lqr_matrix = np.diag(Q_lqr) if Q_lqr is not None else None
        R_lqr_matrix = np.diag(R_lqr) if R_lqr is not None else None
        
        # Set initial position with z_target
        x0 = [INIT_POS_3D[0], INIT_POS_3D[1], z_target, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        
        dynamic_model = Quadcopter(dt=dt, x0=x0, z_target=z_target, motor_limits=motor_limits, 
                                 zero_out_states=zero_out_states, mass=mass, damping=damping,
                                 Q=Q_lqr_matrix, R=R_lqr_matrix)
        
        # Set nominal control to LQR if specified
        try:
            u_nominal_config = control_config['u_nominal']
            if u_nominal_config == "lqr":
                u_nominal = dynamic_model.calcLQRcontrol
        except KeyError:
            print(f"ERROR: Missing required 'u_nominal' parameter for Quadcopter model")
            print(f"Please check your configuration file: {parsed_args.agent_config}")
            print("Required parameter: u_nominal (should be 'lqr' for Quadcopter)")
            sys.exit(3)
    
    else:
        print(f"ERROR: Unsupported model type: {MODEL_TYPE}")
        sys.exit(2)

    # Agent - Ergodic Controller -------------
    # ROS Initialization
    rclpy.init(args=ros_args)

    # Create phi function based on configuration
    if PHI_TYPE == 'gaussian_bumps':
        phi_func = createPhiFunc(L1_BOUNDS=L1_BOUNDS, L2_BOUNDS=L2_BOUNDS, bumps_config_raw=BUMPS_CONFIG, base_level=BASE_LEVEL)
    else:
        # Uniform coverage (default)
        phi_func = lambda s: 1/((L1_BOUNDS[1]-L1_BOUNDS[0])*(L2_BOUNDS[1]-L2_BOUNDS[0]))

    # Generate Agent and connect to an ergodic controller object
    agent = Agent(L1_BOUNDS=L1_BOUNDS, L2_BOUNDS=L2_BOUNDS, Kmax=KMAX, 
                  dynamics_model=dynamic_model,
                  agent_id=AGENT_ID, antenna_rad=ANTENNA_RADIUS, antenna_range_flag=ANTENNA_RANGE_FLAG,
                  same_l_bounds_flag=SAME_L_BOUNDS_FLAG, real_target_positions=REAL_TARGET_POSITIONS, ekf_params=EKF_PARAMS,
                  KAPPA_OBS_VIRTUAL=KAPPA_OBS_VIRTUAL, RHO_OBS_VIRTUAL=RHO_OBS_VIRTUAL, RADIUS_OBS_VIRTUAL=RADIUS_OBS_VIRTUAL,
                  phi=phi_func)      

    agent.erg_c = DecentralisedErgodicController(agent, uNominal=u_nominal, Q=Q_, R = R_, uLimits=u_limits_init,
                                                 T_sampling=TS, T_horizon=T_H, deltaT_erg=deltaT_ERG,
                                                 use_inf_buffer=INF_BUF_FLAG)
    
    # System Read Only Parameters to set in ROS
    from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
    descriptor = ParameterDescriptor(read_only=True, description='Read only')
    agent.declare_parameter('id', AGENT_ID, descriptor=descriptor)
    agent.declare_parameter('localise_targets_flag', LOCALISE_TARGETS_FLAG, descriptor=descriptor)
    agent.declare_parameter('update_eid_flag', UPDATE_EID_FLAG, descriptor=descriptor)
    agent.declare_parameter('save_images_flag', SAVE_IMAGES_FLAG, descriptor=descriptor)
    agent.declare_parameter('imax', IMAX, descriptor=descriptor)
    agent.declare_parameter('init_position_3d', INIT_POS_3D.tolist(), descriptor=descriptor)
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
                # Update virtual obstacles based on new compatibility settings
                agent.updateVirtualObstaclesBasedOnCompatibility()
            if param.name == 'same_l_bounds_flag':
                agent.same_l_bounds_flag = param.value
                if param.value == True:
                    agent.get_logger().info("Enabling Same L Bounds filtering for this agent...")
                else:
                    agent.get_logger().info("Disabling Same L Bounds filtering for this agent...")
                # Update virtual obstacles based on new compatibility settings
                agent.updateVirtualObstaclesBasedOnCompatibility()

        return SetParametersResult(successful=True)

    agent.add_on_set_parameters_callback(parameterCallback)

    # Avoiding Obstacles -------------------
    # Always load the default walls to keep us inside L bound domain for ergodic search
    obstacle_default_walls = loadObstaclesFromYaml('src/ergodic_exploration/ergodic_exploration/default_walls.yaml', L1_BOUNDS, L2_BOUNDS,
                                              kappa_obs=KAPPA_OBS, rho_obs=RHO_OBS,
                                              kappa_wall=KAPPA_WALL, rho_wall=RHO_WALL)

    saveObstaclesToMemory(agent, obs_list=obstacle_default_walls)

    # Load obstacles from custom YAML configuration file if available
    if OBSTACLES_YAML_PATH != "None":
        obstacles_from_yaml = loadObstaclesFromYaml(OBSTACLES_YAML_PATH, L1_BOUNDS, L2_BOUNDS, 
                                                kappa_obs=KAPPA_OBS, rho_obs=RHO_OBS,
                                                kappa_wall=KAPPA_WALL, rho_wall=RHO_WALL)
        # Save obstacles to memory
        if obstacles_from_yaml:
            saveObstaclesToMemory(agent, obs_list=obstacles_from_yaml)
        else:
            print("Warning: No obstacles loaded from YAML file. Using empty obstacle list.")
    
    # Print uNominal Status
    # print(agent.erg_c.uNominal)
    
    # Lets now update the phi_function to take into account the obstacles
    agent.basis.phi = agent.modifedPhiForObstacles(agent.basis.phi, obs_to_exclude="None")
    agent.basis.precalcAllPhiK()

    # ===== Clock Synchronization Setup =====
    if SYNC_CLOCKS_FLAG:
        agent.enableClockSync(expected_agent_ids=SYNC_AGENT_IDS)
        agent.get_logger().info(f"Clock synchronization ENABLED for agent {AGENT_ID}")
        if SYNC_AGENT_IDS:
            agent.get_logger().info(f"  Syncing with agents: {SYNC_AGENT_IDS}")
        else:
            agent.get_logger().info(f"  Syncing with all discovered agents")

    if SHOW_INIT_PHI:
        # Visualize H-field
        vis.visHfield(agent, L_limits=[agent.L1_min-0.5, agent.L1_max+0.5, agent.L2_min-0.5, agent.L2_max+0.5], delta=DELTA_SAFE, num_of_points=200)
        phi_rec = ReconstructedPhi(agent.basis, precalc_phik=False)

        # Lets visualise the origial phi side by side with the reconstructed one using Kmax
        vis.plotPhiOnlyOriginalAndReconstructed(agent, phi_rec_from_agent=phi_rec, grid_res=100, clip_to_min_max=False)
        
        plt.show()
        print("Using sys.exit() to stop due to --show_initial_phi == True flag")
        sys.exit(0)

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
                        # If no agent in the neighborhood, use zeros (single agent case)
                        # Using own ck would double the contribution and cause oscillations
                        agent.erg_c.ck_aver_others = np.zeros_like(agent.basis.ck)
                    ck_total_in_range = np.mean(in_range_ck_data + [agent.basis.ck], axis=0) if len(in_range_ck_data) > 0 else agent.basis.ck
                    agent.erg_c.total_erg_cost_in_range = agent.erg_c.calcErgodicCost(ck_total_in_range)
                else:
                    # Only set ck_aver_others if there are OTHER agents (not including self)
                    other_agent_ck_only = [ck for aid, ck in agent.getAgentCkData().items() if aid != agent.agent_id and ck is not None]
                    if len(other_agent_ck_only) > 0:
                        agent.erg_c.ck_aver_others = np.mean(other_agent_ck_only, axis=0)
                    else:
                        # Single agent case: use zeros to avoid doubling ck
                        agent.erg_c.ck_aver_others = np.zeros_like(agent.basis.ck)
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
                    # agent.get_logger().info(f"ti = {ti:.2f} s\t Erg cost: {erg_cost:.2f} \t i: {i}/{IMAX:.0f} \t perc: {i/IMAX:.2%} \t Δt/Ts: {delta_time/agent.erg_c.Ts:.2f}\t remaining: {rem_time_max:.0f} s\t elapsed: {time.time()-initial_time:.1f} s ({time.time()-initial_time + delta_time * (IMAX-i)/Ts_iter:.0f} s) ({IMAX/(i+1)*(time.time()-initial_time):.0f} s)\n"
                    #                          f"{agent.model.state_string} \n u = {u_str(us)} \t (tau - ti)/T = {(tau - ti)/agent.erg_c.T:.1%} \t lamda_dur = {lamda_dur:.4f} \t lamda/Ts = {lamda_dur/agent.erg_c.Ts:.2%}\n")

                # Debug print if agent inside boundaries
                agent.withinBounds(agent.model.state[:2])
                if np.any(np.abs(agent.model.state[:2]) > 50):
                    if agent.model.type != "FixedWing12DOFTrainer":
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
            active_cbf_flag = np.any(np.abs(u_safe_list[-1]) > 1e-4) or i%CBF_SKIP_ITER == 0
            if active_cbf_flag:
                u_safe = agent.calcUsafe(
                    agent.model.state, u, u_before, time_now=time_list[i],
                    alpha_1=ALPHA_1, alpha_2=ALPHA_2, alpha_3=ALPHA_3,
                    alpha_u=ALPHA_U, Kp=CBF_KP, dt=CBF_DT,
                    alpha_max_deg=ALPHA_MAX_DEG, alpha_aoa_1=ALPHA_AOA_1, 
                    alpha_aoa_2=ALPHA_AOA_2, slack_penalty_aoa=SLACK_PENALTY_AOA,
                    use_aoa_constraint=USE_AOA_CONSTRAINT,
                    delta=DELTA_SAFE
                )
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
            if not shutdown_flag.is_set() and i % PUBLISH_DATA_FREQ == 0:
                agent.publishData(state_now=agent.model.state, u_input_now=u, erg_cost_now=erg_cost, 
                                  active_cbf_flag=active_cbf_flag,
                                  time_now=time_list[i], delta_t_Ts=delta_time / agent.erg_c.Ts)

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

            # ===== Clock Synchronization =====
            # Sync frequency: use custom value or default to Ts_iter
            sync_interval = SYNC_FREQ if SYNC_FREQ is not None else Ts_iter
            if agent.sync_clocks_flag and i % sync_interval == 0:
                # Wait for all agents to reach this step before proceeding
                sync_success = agent.waitForSync(sim_step=i, sim_time=time_list[i], timeout_sec=10.0)
                if not sync_success and not shutdown_flag.is_set():
                    agent.get_logger().warn(f"Sync timeout at iteration {i}, continuing anyway...")
            elif not agent.sync_clocks_flag:
                # Original behavior: if delta_time < Ts: delay 
                if delta_time < agent.erg_c.Ts:
                    time.sleep(agent.erg_c.Ts - delta_time)
                    delta_time = agent.erg_c.Ts  # We waited the remaining time
            
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