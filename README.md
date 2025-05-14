# Diplomatiki - Ergodic Control Implementation

## Project Overview
This repository contains an implementation of ergodic control algorithms for multi-agent robotic systems. Ergodic control is a control strategy that drives agents to match a specified spatial distribution, making it useful for exploration, surveillance, and monitoring applications.

The core concept is to make the time-averaged statistics of an agent's trajectory match a desired spatial distribution, creating an efficient exploration pattern that focuses more time on high-importance regions while still covering the entire space.

![Quadrotor Ergodic Exploration](images/gifs/phi1_animation.gif)

*The animation shows a 12-DoF quadrotor model ergodically exploring a spatial distribution, demonstrating how the algorithm balances between visiting high-density regions while maintaining coverage of the entire domain.*

## Repository Structure
- `my_erg_lib/`: Custom implementation of the ergodic control library
  - Contains models, controllers, and utility functions for ergodic control
- `images/`: Visualization outputs and animations
  - `gifs/`: Animations of system behavior and distribution convergence

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

### Control
- `ergodic_controllers.py`: Core implementation of ergodic control strategies:
  - `DecentralisedErgodicController`: Novel implementation for decentralized multi-agent settings
  - Receding-horizon implementation with trajectory optimization
  - Adjoint-based gradient descent for ergodic metric optimization
- `basis.py`: Fourier basis functions for spectral decomposition of spatial distributions
- `barrier.py`: Barrier functions to enforce state and control constraints

### Integration
- `agent.py`: Agent implementation that combines models and controllers
- `replay_buffer.py`: Storage for trajectory samples for reinforcement learning
- `Utilities.py`: Helper functions for the library

### Visualization
- `vis.py`: Visualization tools including:
  - 3D rendering of quadrotor trajectories
  - Distribution visualization and comparison
  - Animation generation for ergodic coverage analysis

## Dependencies
- NumPy: For numerical operations
- Matplotlib: For visualization and animation
- SciPy: For optimization and linear algebra
- PIL: For image processing and saving animations

## Usage
This library is designed for multi-agent robotic control in various scenarios:

```python
# Example usage with a quadrotor model
from my_erg_lib.agent import Agent
from my_erg_lib.model_dynamics import Quadcopter
from my_erg_lib.ergodic_controllers import DecentralisedErgodicController

# Create quadrotor model
model = Quadcopter(dt=0.001, x0=[0.6, 0.8, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# Set up agent with ergodic controller
agent = Agent(L1=2.0, L2=2.0, Kmax=5, dynamics_model=model)
agent.erg_c = DecentralisedErgodicController(agent, uNominal=model.calcLQRcontrol)

# Run simulation
for i in range(1000):
    # Calculate control action
    us, tau, lamda_dur, erg_cost = agent.erg_c.calcNextActionTriplet(time_now)
    
    # Apply control to model
    agent.model.state = agent.model.step(agent.model.state, us)
```

Key features:
- Spectral decomposition of target distributions using Fourier basis functions
- Receding horizon control for ergodic exploration
- LQR stabilization for complex dynamic models
- Multi-agent coordination through Fourier coefficient exchange
- Advanced integration methods (Runge-Kutta 4) for accurate dynamics simulation


## References
- Mavrommati, A., Tzorakoleftherakis, E., Abraham, I., and Murphey, T. D. (2017). Real-time area coverage and target localization using receding-horizon ergodic exploration. IEEE Transactions on Robotics, 34(1), 62-80. [arXiv:1708.08416](https://arxiv.org/abs/1708.08416)
- Abraham, I., and Murphey, T. D. (2018). Decentralized ergodic control: distribution-driven sensing and exploration for multiagent systems. IEEE Robotics and Automation Letters, 3(4), 2987-2994. [arXiv:1708.08416](https://arxiv.org/abs/1708.08416)