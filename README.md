# Ergodic Control Navigation

## Project Overview
This repository contains an implementation of ergodic control algorithms for multi-agent robotic systems. Ergodic control is a control strategy that drives agents to match a specified spatial distribution, making it useful for exploration, surveillance, and monitoring applications.

The core concept is to make the time-averaged statistics of an agent's trajectory match a desired spatial distribution, creating an efficient exploration pattern that focuses more time on high-importance regions while still covering the entire space.

![Quadrotor Ergodic Exploration](images/gifs/phi2_animation.gif)
*The animation shows a 12-DoF quadrotor model ergodically exploring a spatial distribution, demonstrating how the algorithm balances between visiting high-density regions while maintaining coverage of the entire domain.*

![Double Integrator w/ Obstacles](images/gifs/phi_obs_double_int_animation.gif)
*The animation shows a simple double integrator model ergodically exploring the given spatial distribution in the presence of obstacles / forbidden regions of space*

![Quadrotor Ergodic Exploration w/ Obstacles](images/gifs/phiQuadWithObs_animation.gif)
*The animation shows a 12-DoF quadrotor model ergodically exploring the given spatial distribution in the presence of obstacles / forbidden regions of space*

![Quadrotor Ergodic Exploration w/ Obstacles + EKF Target Localisation](images/gifs/phi_single_target_tracking_w_obstacles.gif)
*The animation shows a 12-DoF quad model searching for a target using bearing only measurements and localising it using an EKF filter. The EID map updates using the Expected Information Matrix in each location.*

## Repository Structure
- `my_erg_lib/`: Custom implementation of the ergodic control library
  - Contains models, controllers, and utility functions for ergodic control
- `images/`: Visualization outputs and animations
  - `gifs/`: Animations of system behavior and distribution convergence
- `more/`: Additional test scripts and experimental features
  - Integration method comparisons
  - Parallel processing implementations
  - Potential field visualization

## Key Components

### Dynamics Models
- `model_dynamics.py`: Implementation of various dynamics models including:
  - `SingleIntegrator`: Simple first-order dynamics
  - `DoubleIntegrator`: Second-order dynamics
  - `Quadcopter`: Full 12-DoF quadrotor model with realistic dynamics including:
    - Position (x, y, z)
    - Orientation (roll, pitch, yaw)
    - Linear and angular velocities
    - Motor command mixing and thrust generation
    - LQR stabilization with customizable gains for obstacle avoidance

### Control
- `ergodic_controllers.py`: Core implementation of ergodic control strategies:
  - `DecentralisedErgodicController`: Novel implementation for decentralized multi-agent settings
  - Receding-horizon implementation with trajectory optimization
  - Adjoint-based gradient descent for ergodic metric optimization
- `basis.py`: Fourier basis functions for spectral decomposition of spatial distributions
  - Integration methods: Gauss quadrature and `nquad`
  - Spectral coefficient caching for performance
  - Distribution reconstruction capabilities
- `barriers.py`: Barrier functions to enforce state and control constraints

### Obstacle Avoidance
- `obstacles.py`: Implementation of obstacle avoidance using potential fields:
  - Support for multiple obstacle types:
    - Circular obstacles with customizable radius
    - Rectangular obstacles with width and height parameters
    - Wall obstacles with normal vector definition
  - Reactive obstacle avoidance through artificial potential fields
  - Customizable repulsive forces and influence regions
  - Boundary enforcement to keep agents within exploration space

<div align="center">
<img src="images/images/potential_field_4.png" width="90%" alt="Potential field visualization">
</div>

### Target Localization and Tracking
- `eid.py`: Comprehensive multi-target localization system:
  - **Measurement Model**: Vectorized bearing-only sensor model computing azimuth and elevation angles
  - **Extended Kalman Filter (EKF)**: Real-time state estimation with uncertainty quantification
  - **Sensor Class**: Configurable range-limited sensor with realistic noise characteristics
  - **Data Association**: Mahalanobis distance-based measurement-to-target association
  - **Target Lifecycle Management**:
    - *Spawning*: Creates new target estimates from unassociated measurements
    - *Merging*: Combines nearby estimates using Bhattacharyya distance criteria
    - *Deletion*: Removes stale estimates based on age and confidence metrics
  - **Information-Driven Exploration**: EID (Expected Information Density) maps using Fisher Information Matrix
  - **Multi-Target Tracking**: Simultaneous estimation of multiple moving targets with covariance intersection

<div align="center">
<img src="images/gifs/measurementsEKF_animation_spawnTargets_Merge.gif" width="70%" alt="Multi-Target Tracking">
</div>

*The animation demonstrates multi-target localization using bearing-only measurements and EKF estimation. The system dynamically spawns new target estimates, associates measurements with existing targets, and merges or deletes estimates as needed.*

### Integration
- `agent.py`: Agent implementation that combines models and controllers
  - Boundary checking and safety mechanisms
  - Integration with obstacle avoidance system
- `replay_buffer.py`: Storage for trajectory samples for reinforcement learning
- `Utilities.py`: Helper functions for the library

### Visualization
- `vis.py`: Visualization tools including:
  - 3D rendering of quadrotor trajectories
  - Distribution visualization and comparison
  - Animation generation for ergodic coverage analysis
  - Potential field visualization for obstacle avoidance
  - Trajectory replay with time-series plotting

### Spectral Distribution Analysis
- `ReconstructedPhi` and `ReconstructedPhiFromCk`: Classes for analyzing and reconstructing spatial distributions
  - Fourier coefficient calculation for arbitrary distributions
  - Distribution reconstruction from trajectory statistics
  - Comparison between target and achieved distributions

## Dependencies
- NumPy: For numerical operations
- Matplotlib: For visualization and animation
- SciPy: For optimization and linear algebra
- PIL: For image processing and saving animations

## Usage
This library is designed for multi-agent robotic control in various scenarios:

```python
# Example usage with quadrotor model, obstacle avoidance, and multi-target tracking
import numpy as np
from my_erg_lib.agent import Agent
from my_erg_lib.model_dynamics import Quadcopter
from my_erg_lib.ergodic_controllers import DecentralisedErgodicController
from my_erg_lib.obstacles import Obstacle, ObstacleAvoidanceControllerGenerator

# Create quadrotor model with specified parameters
x0 = [0.8, 0.8, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]
model = Quadcopter(dt=0.001, x0=x0, z_target=2, 
                   motor_limits=[[-2, 2], [-2, 2], [-2, 2], [-2, 2]])

# Create target distribution function
def phi_func(s):
    x, y = s[0], s[1]
    return 3 * np.exp(-30 * ((x-0.2)**2 + (y-0.3)**2)) + 2

# Set up agent with ergodic controller
agent = Agent(L1=1.0, L2=1.0, Kmax=5, dynamics_model=model, phi=phi_func, x0=x0)
agent.erg_c = DecentralisedErgodicController(agent, uNominal=model.calcLQRcontrol, 
                                            T_sampling=0.1, T_horizon=1.25)

# Add obstacles to the environment
obstacles = [
    Obstacle(pos=[0.2, 0.2], dimensions=0.1, f_max=0.25, min_dist=0.14, 
             eps_meters=0.2, obs_type='circle', obs_name="Obstacle 1"),
    Obstacle(pos=[0.6, 0.3], dimensions=[0.2, 0.5], f_max=0.25, min_dist=0.52, 
             eps_meters=0.2, obs_type='rectangle', obs_name="Obstacle 2")
]
agent.erg_c.uNominal += ObstacleAvoidanceControllerGenerator(agent, obs_list=obstacles)

# Initialize simulation variables
time_list = [0]
Ts_iter = int(agent.erg_c.Ts / agent.model.dt)  # Iterations per sampling time
u_previous = np.zeros(agent.model.num_of_inputs)

# Main simulation loop
for i in range(10000):
    current_time = time_list[i]
    
    # Calculate ergodic control every sampling period
    if i % Ts_iter == 0:
        # Multi-target tracking
        measurements = agent.sensor.getMultipleMeasurements(
            agent.real_target_positions, agent.model.state[:3])
        
        # Data association and EKF updates
        if measurements and agent.num_of_targets == 0:
            # Initialize targets if first measurements
            for measurement in measurements:
                agent.spawnNewTargetEstimate(measurement, current_time)
        
        associated_measurements = agent.associateTargetsWithMahalanobis(
            measurements, agent.model.state[:3])
        
        # Update existing targets
        for j, measurement in enumerate(associated_measurements):
            if measurement is not None:
                agent.ekfs[j].update(agent.model.state[:3], measurement, current_time)
        
        # Spawn new targets for unassociated measurements
        for m in measurements or []:
            if not any(np.array_equal(m, am) for am in associated_measurements if am is not None):
                agent.spawnNewTargetEstimate(measurement=m, current_time=current_time)
        
        # Target management
        agent.mergeTargetsIfNeeded()
        agent.searchAndRemoveOldTargetEstimates(current_time)
        
        # Update exploration distribution periodically
        if i % (Ts_iter * 30) == 0:  # Every 30 ergodic iterations
            agent.updateEIDphiFunction()
        
        # Calculate ergodic control
        us, tau, lamda_dur, erg_cost = agent.erg_c.calcNextActionTriplet(current_time)
        agent.erg_c.updateActionMask(current_time, us, tau, lamda_dur)
    
    # Get current control action
    us_current = agent.erg_c.ustar_mask[i % Ts_iter]
    if not us_current.any():
        us_current = agent.erg_c.uNominal(agent.model.state, current_time)
    
    # Smooth control action
    u_smooth = 0.3 * us_current + 0.7 * u_previous
    u_previous = u_smooth.copy()
    
    # Apply control and step model
    agent.model.state = agent.model.step(agent.model.state, u_smooth)
    agent.erg_c.past_states_buffer.push(agent.model.state[:2])
    
    # Update time
    time_list.append(current_time + agent.model.dt)
```

## Key Features
- Spectral decomposition of target distributions using Fourier basis functions
- Receding horizon control for ergodic exploration
- LQR stabilization for complex dynamic models
- Multi-agent coordination through Fourier coefficient exchange
- Advanced integration methods (Runge-Kutta 4) for accurate dynamics simulation
- Obstacle avoidance with customizable potential fields
- Multi-target localization with bearing-only measurements
- Dynamic target management with spawning, merging, and deletion
- Information-driven exploration using Fisher Information Matrix
- Mahalanobis distance-based data association
- Comprehensive visualization tools for analysis and debugging
- Performance profiling for optimization
- Support for complex spatial distribution functions

## References
- Mavrommati, A., Tzorakoleftherakis, E., Abraham, I., and Murphey, T. D. (2017). Real-time area coverage and target localization using receding-horizon ergodic exploration. IEEE Transactions on Robotics, 34(1), 62-80. [arXiv:1708.08416](https://arxiv.org/abs/1708.08416)
- Abraham, I., and Murphey, T. D. (2018). Decentralized ergodic control: distribution-driven sensing and exploration for multiagent systems. IEEE Robotics and Automation Letters, 3(4), 2987-2994. [arXiv:1708.08416](https://arxiv.org/abs/1708.08416)