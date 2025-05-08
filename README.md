# Diplomatiki - Ergodic Control Implementation

## Project Overview
This repository contains an implementation of ergodic control algorithms for multi-agent robotic systems. Ergodic control is a control strategy that drives agents to match a specified spatial distribution, making it useful for exploration, surveillance, and monitoring applications.

## Repository Structure
- `my_erg_lib/`: Custom implementation of the ergodic control library
  - Contains models, controllers, and utility functions for ergodic control

## Key Components

### Models
- `model_dynamics.py`: Implementation of various dynamics models for different robot platforms

### Control
- `ergodic_controllers.py`: Core implementation of receding-horizon ergodic control
- `basis.py`: Fourier basis functions for spectral decomposition
- `barrier.py`: Barrier functions to enforce state constraints

### Integration
- `agent.py`: Agent implementation that combines models and controllers
- `replay_buffer.py`: Storage for trajectory samples
- `Utilities.py`: Helper functions for the library

## Dependencies
- NumPy
- Matplotlib (for visualization)
- Gym (for space definitions)

## Usage
This library is designed for multi-agent robotic control. The implementation supports:
- Spectral decomposition of target distributions
- Receding horizon control for ergodic exploration
- Multi-agent coordination through Fourier coefficient exchange

## Installation
Clone this repository and ensure you have the required dependencies.

## License
[Specify license information]

## References
[Academic papers or other references related to ergodic control]