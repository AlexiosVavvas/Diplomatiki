from scipy.linalg import logm, expm  # Add this import
import numpy as np
def logEuclideanMean(covariance_matrices):
    """Compute log-Euclidean mean of covariance matrices"""
    log_matrices = [logm(sigma) for sigma in covariance_matrices]
    mean_log = np.mean(log_matrices, axis=0)
    return expm(mean_log)


def mergeOverlappingPairsAllTheWay(pairs):
    """
    Merge all overlapping pairs into connected groups.

    Returns:
        List of merged groups where each group contains all connected elements
    Example:
        [[1, 2], [2, 3], [4, 5]] -> [[1, 2, 3], [4, 5]]
    """
    def mergeOverlappingPairs(pairs):
        """Merge pairs that share at least one common element"""
        if not pairs:
            return []
        
        result = [set(pairs[0])]
        
        for pair in pairs[1:]:
            pair_set = set(pair)
            merged = False
            
            for i, existing_group in enumerate(result):
                if pair_set & existing_group:  # If there's any overlap
                    result[i] = existing_group | pair_set
                    merged = True
                    break
            
            if not merged:
                result.append(pair_set)
        
        # Convert back to sorted lists
        return [sorted(list(group)) for group in result]

    data1 = mergeOverlappingPairs(pairs)
    data2 = []
    while data2 != data1:
        data2 = data1.copy()
        data1 = mergeOverlappingPairs(data1)
    
    return data1


import yaml
import sys

def loadObstaclesFromYaml(yaml_file_path, L1_BOUNDS, L2_BOUNDS, kappa_obs=1.0, rho_obs=0.15, kappa_wall=0.5, rho_wall=1.5):
    """
    Load obstacles from a YAML configuration file.
    
    Args:
        - yaml_file_path (str): Path to the YAML configuration file
        - L1_BOUNDS (list): [L1_min, L1_max] bounds for the domain
        - L2_BOUNDS (list): [L2_min, L2_max] bounds for the domain
        - kappa_obs (float): Default kappa value for obstacles when not specified in YAML
        - rho_obs (float): Default rho0 value for obstacles when not specified in YAML
        - kappa_wall (float): Default kappa value for walls when not specified in YAML
        - rho_wall (float): Default rho0 value for walls when not specified in YAML
        
    Returns:
        list: List of Obstacle objects
    """
    from my_erg_lib.obstacles import Obstacle
    
    obstacles = []
    obstacle_positions = []  # To track all obstacle positions for boundary checking
    
    try:
        with open(yaml_file_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Load regular obstacles
        if 'obstacles' in config:
            for obs_config in config['obstacles']:
                pos = obs_config['pos']
                dimensions = obs_config['dimensions']
                obs_type = obs_config['obs_type']
                kappa = obs_config.get('kappa', kappa_obs)  # Use default if not provided
                rho0 = obs_config.get('rho0', rho_obs)      # Use default if not provided
                obs_name = obs_config['obs_name']
                
                # Store position for boundary checking
                obstacle_positions.append(pos)
                
                obstacle = Obstacle(pos=pos, dimensions=dimensions, obs_type=obs_type, 
                                  kappa=kappa, rho0=rho0, obs_name=obs_name)
                obstacles.append(obstacle)
        
        # Load wall obstacles with dynamic positioning
        if 'walls' in config:
            L1_size = L1_BOUNDS[1] - L1_BOUNDS[0]; L1_min = L1_BOUNDS[0]; L1_max = L1_BOUNDS[1]
            L2_size = L2_BOUNDS[1] - L2_BOUNDS[0]; L2_min = L2_BOUNDS[0]; L2_max = L2_BOUNDS[1]
            
            for wall_config in config['walls']:
                kappa = wall_config.get('kappa', kappa_wall)  # Use default if not provided
                rho0 = wall_config.get('rho0', rho_wall)      # Use default if not provided
                obs_name = wall_config['obs_name']
                obs_type = wall_config['obs_type']
                
                # Check if this is a fixed wall (has explicit pos and dimensions)
                if 'pos' in wall_config and 'dimensions' in wall_config:
                    # Fixed wall with explicit parameters
                    pos = wall_config['pos']
                    dimensions = wall_config['dimensions']
                    
                    # Store position for boundary checking
                    obstacle_positions.append(pos)
                    
                    obstacle = Obstacle(pos=pos, dimensions=dimensions, obs_type=obs_type,
                                      kappa=kappa, rho0=rho0, obs_name=obs_name)
                    obstacles.append(obstacle)
                    
                elif 'wall_type' in wall_config:
                    # Dynamic wall with auto-calculated position based on wall_type
                    wall_type = wall_config['wall_type']
                    
                    # Calculate position and dimensions based on wall type
                    if wall_type == 'bottom':
                        pos = [L1_min + L1_size/2, L2_min]
                        dimensions = [0, +L1_size]
                    elif wall_type == 'top':
                        pos = [L1_min + L1_size/2, L2_max]
                        dimensions = [0, -L1_size]
                    elif wall_type == 'left':
                        pos = [L1_min, L2_min + L2_size/2]
                        dimensions = [+L2_size, 0]
                    elif wall_type == 'right':
                        pos = [L1_max, L2_min + L2_size/2]
                        dimensions = [-L2_size, 0]
                    else:
                        print(f"Warning: Unknown wall type '{wall_type}', skipping...")
                        continue
                    
                    # Store position for boundary checking (dynamic walls are always within bounds)
                    obstacle_positions.append(pos)
                    
                    obstacle = Obstacle(pos=pos, dimensions=dimensions, obs_type=obs_type,
                                      kappa=kappa, rho0=rho0, obs_name=obs_name)
                    obstacles.append(obstacle)
                    
                else:
                    print(f"Warning: Wall obstacle '{obs_name}' must have either 'wall_type' or both 'pos' and 'dimensions'. Skipping...")
                    continue
        
        # Check obstacle positions against boundaries
        if obstacle_positions:
            # Calculate min/max positions from obstacles
            x_positions = [pos[0] for pos in obstacle_positions]
            y_positions = [pos[1] for pos in obstacle_positions]
            
            min_x = min(x_positions)
            max_x = max(x_positions)
            min_y = min(y_positions)
            max_y = max(y_positions)
            
            # Check if any obstacles are outside the domain boundaries
            L1_min, L1_max = L1_BOUNDS
            L2_min, L2_max = L2_BOUNDS
            
            obstacles_outside = (min_x < L1_min or max_x > L1_max or 
                               min_y < L2_min or max_y > L2_max)
            
            if obstacles_outside:
                print("="*60)
                print("WARNING: Obstacles found outside domain boundaries!")
                print("="*60)
                print(f"Domain boundaries: X=[{L1_min}, {L1_max}], Y=[{L2_min}, {L2_max}]")
                print(f"Obstacle positions: X=[{min_x}, {max_x}], Y=[{min_y}, {max_y}]")
                print()
                
                if min_x < L1_min:
                    print(f"  • Obstacles extend {L1_min - min_x:.2f} units below X minimum ({L1_min})")
                if max_x > L1_max:
                    print(f"  • Obstacles extend {max_x - L1_max:.2f} units above X maximum ({L1_max})")
                if min_y < L2_min:
                    print(f"  • Obstacles extend {L2_min - min_y:.2f} units below Y minimum ({L2_min})")
                if max_y > L2_max:
                    print(f"  • Obstacles extend {max_y - L2_max:.2f} units above Y maximum ({L2_max})")
                    
                print()
                print("This may cause issues with ergodic exploration.")
                print("Consider adjusting obstacle positions or domain boundaries.")
                print("="*60)
                
                try:
                    user_input = input("Do you want to continue anyway? (y/N): ").strip().lower()
                    if user_input not in ['y', 'yes']:
                        print("Stopping program as requested.")
                        sys.exit(1)
                    else:
                        print("Continuing with obstacles outside boundaries...")
                except KeyboardInterrupt:
                    print("\nProgram interrupted by user.")
                    sys.exit(1)
                
    except FileNotFoundError:
        print(f"Error: Could not find YAML file at {yaml_file_path}")
        return []
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return []
    except KeyError as e:
        print(f"Error: Missing required key in YAML file: {e}")
        return []
    
    print(f"Successfully loaded {len(obstacles)} obstacles from {yaml_file_path}\n")
    return obstacles


def loadAgentConfigFromYaml(yaml_file_path, L1_BOUNDS=None, L2_BOUNDS=None):
    """
    Load agent configuration from a YAML file.
    
    Args:
        yaml_file_path (str): Path to the YAML configuration file
        L1_BOUNDS (list, optional): [L1_min, L1_max] bounds for the domain (for EKF limits)
        L2_BOUNDS (list, optional): [L2_min, L2_max] bounds for the domain (for EKF limits)
        
    Returns:
        dict: Dictionary containing agent configuration data with the following structure:
        {
            'model_type': str,
            'dynamics': dict,
            'control': dict,  
            'system': dict,
            'targets': dict,
            'flags': dict
        }
    """
    import numpy as np
    
    try:
        with open(yaml_file_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Extract main sections - all required
        if 'agent' not in config:
            raise KeyError("Missing required 'agent' section in configuration")
        if 'targets' not in config:
            raise KeyError("Missing required 'targets' section in configuration")  
        if 'flags' not in config:
            raise KeyError("Missing required 'flags' section in configuration")
            
        agent_config = config['agent']
        targets_config = config['targets'] 
        flags_config = config['flags']
        
        # Process agent configuration - all required
        if 'model_type' not in agent_config:
            raise KeyError("Missing required 'model_type' in agent configuration")
        if 'dynamics' not in agent_config:
            raise KeyError("Missing required 'dynamics' section in agent configuration")
        if 'control' not in agent_config:
            raise KeyError("Missing required 'control' section in agent configuration")
        if 'system' not in agent_config:
            raise KeyError("Missing required 'system' section in agent configuration")
            
        model_type = agent_config['model_type']
        dynamics = agent_config['dynamics']
        control = agent_config['control']
        system = agent_config['system']
        
        # Convert YAML lists to numpy arrays for control limits - required fields
        if 'u_limits_init' not in control:
            raise KeyError("Missing required 'u_limits_init' in control configuration")
        if 'u_limits' not in control:
            raise KeyError("Missing required 'u_limits' in control configuration")
        if 'R' not in control:
            raise KeyError("Missing required 'R' matrix in control configuration")
            
        control['u_limits_init'] = np.array(control['u_limits_init'])
        control['u_limits'] = np.array(control['u_limits'])
        control['R'] = np.array(control['R'])
            
        # Process EKF configuration - required section
        if 'ekf' not in targets_config:
            raise KeyError("Missing required 'ekf' section in targets configuration")
        ekf_config = targets_config['ekf']
        
        # Convert diagonal values to full covariance matrices - all required
        required_ekf_fields = ['sigma_init_diag', 'R_diag', 'Q_diag', 'sensor_range', 'sensor_R_diag']
        for field in required_ekf_fields:
            if field not in ekf_config:
                raise KeyError(f"Missing required EKF parameter: '{field}'")
        
        # Convert diagonal arrays to matrices
        diag_vals = ekf_config['sigma_init_diag']
        ekf_config['sigma_init'] = np.eye(len(diag_vals)) * np.array(diag_vals)
        
        diag_vals = ekf_config['R_diag']
        ekf_config['R'] = np.diag(diag_vals)
        
        diag_vals = ekf_config['Q_diag']
        ekf_config['Q'] = np.eye(len(diag_vals)) * np.array(diag_vals)
        
        diag_vals = ekf_config['sensor_R_diag']
        ekf_config['sensor_R'] = np.diag(diag_vals)
            
        # Add a_limits if L bounds are provided
        if L1_BOUNDS is not None and L2_BOUNDS is not None:
            ekf_config['a_limits'] = [L1_BOUNDS, L2_BOUNDS, [0, 10]]
            
        # Convert target positions to numpy arrays - required
        if 'real_positions' not in targets_config:
            raise KeyError("Missing required 'real_positions' in targets configuration")
        real_positions = targets_config['real_positions']
        real_positions_np = [np.array(pos) for pos in real_positions]
        targets_config['real_positions'] = real_positions_np
        
        # Handle special cases for imax (infinity) - required
        if 'imax' not in system:
            raise KeyError("Missing required 'imax' parameter in system configuration")
        imax_val = system['imax']
        if isinstance(imax_val, str) and imax_val == '.inf':
            system['imax'] = np.inf
            
        # Handle special case for antenna_radius (infinity) - optional
        if 'antenna_radius' in agent_config:
            antenna_radius_val = agent_config['antenna_radius']
            if isinstance(antenna_radius_val, str) and antenna_radius_val == '.inf':
                agent_config['antenna_radius'] = np.inf
                
        # Build the final configuration dictionary
        final_config = {
            'model_type': model_type,
            'dynamics': dynamics,
            'control': control,
            'system': system,
            'targets': targets_config,
            'flags': flags_config
        }
        
        # Add optional agent parameters to the final config for easy access
        optional_params = ['antenna_radius', 'kmax', 'antenna_range_flag', 'talk_alike_flag', 'same_l_bounds_flag']
        for param in optional_params:
            if param in agent_config:
                final_config[param] = agent_config[param]
        
        print(f"Successfully loaded agent configuration from {yaml_file_path}")
        print(f"Model type: {model_type}")
        return final_config
        
    except FileNotFoundError:
        print(f"ERROR: Could not find agent configuration file at {yaml_file_path}")
        return None
    except yaml.YAMLError as e:
        print(f"ERROR: Invalid YAML syntax in configuration file: {e}")
        return None
    except KeyError as e:
        # Re-raise KeyError with additional context - don't return None
        error_msg = f"Configuration validation failed: {str(e)}"
        print(f"ERROR: {error_msg}")
        raise KeyError(error_msg) from e
    except Exception as e:
        print(f"ERROR: Unexpected error loading agent configuration: {e}")
        return None