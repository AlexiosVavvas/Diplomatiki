from my_erg_lib.agent import Agent
from my_erg_lib.model_dynamics import Quadcopter
import numpy as np

# Lets test associations with targets
agent = Agent(L1=1.0, L2=1.0, Kmax=10, dynamics_model=Quadcopter(), x0=np.array([0.3, 0.4, 2, 0, 0, 0, 0,  0,  0,  0,  0,  0])); print("\n\n")
# Targets are at (0.8, 0.4, 0) and (0.3, 0.7, 0)
agent.sensor.sensor_range = 1
measurements = agent.sensor.getMultipleMeasurements(agent.real_target_positions, agent.model.state[:3])
# suffle the measurements to simulate random order
np.random.shuffle(measurements)
print(f"Measurements: \n{measurements}")  

# Associate targets with measurements
associated_measurements = agent.associateTargetsWithMahalanobis([measurements[0]], agent.model.state[:3], ASSOCIATION_THRESHOLD=0.01)
print(f"\n\nAssociated Measurements: \n{associated_measurements}\n\n")


# Lets visualise the targets, the measurements and the associated measurements (only beta angle connecting agent with target with possitibe in +y durection is needed)
agent.real_target_positions = np.array(agent.real_target_positions)
measurements = np.array(measurements)
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
plt.scatter(agent.real_target_positions[:, 0], agent.real_target_positions[:, 1], c='red', label='Targets')

# Add numbers to targets
for i, target in enumerate(agent.real_target_positions):
    plt.annotate(str(i), (target[0], target[1]), xytext=(5, 5), textcoords='offset points', fontsize=12, color='red')

# Create arrays of starting positions for each vector
agent_x = np.full(len(measurements), agent.model.state[0])
agent_y = np.full(len(measurements), agent.model.state[1])
plt.quiver(agent_x, agent_y, 
           agent.sensor.sensor_range * np.cos(np.pi/2 - measurements[:, 0]), 
           agent.sensor.sensor_range * np.sin(np.pi/2 - measurements[:, 0]), 
           angles='xy', scale_units='xy', scale=1, color='blue', alpha=0.5, label='Measurement Vectors')

# Filter out None values from associated_measurements and plot only valid ones
valid_associated = [meas for meas in associated_measurements if meas is not None]
if valid_associated:
    valid_associated = np.array(valid_associated)
    agent_x_assoc = np.full(len(valid_associated), agent.model.state[0])
    agent_y_assoc = np.full(len(valid_associated), agent.model.state[1])
    plt.quiver(agent_x_assoc, agent_y_assoc,
              agent.sensor.sensor_range * np.cos(np.pi/2 - valid_associated[:, 0]), 
              agent.sensor.sensor_range * np.sin(np.pi/2 - valid_associated[:, 0]), 
              angles='xy', scale_units='xy', scale=1, color='green', alpha=0.5, label='Associated Vectors')

    # Add numbers to associated measurement arrows (only for valid measurements)
    valid_idx = 0
    for i, assoc_meas in enumerate(associated_measurements):
        if assoc_meas is not None:
            arrow_end_x = agent.model.state[0] + agent.sensor.sensor_range * np.cos(np.pi/2 - assoc_meas[0])
            arrow_end_y = agent.model.state[1] + agent.sensor.sensor_range * np.sin(np.pi/2 - assoc_meas[0])
            plt.annotate(str(i), (arrow_end_x, arrow_end_y), xytext=(5, 5), textcoords='offset points', fontsize=12, color='green')
            valid_idx += 1

# Plot vectors to all targets and identify unassociated ones
associated_indices = set()
for i, assoc_meas in enumerate(associated_measurements):
    if assoc_meas is not None:
        associated_indices.add(i)

agent_x_targets = np.full(len(agent.real_target_positions), agent.model.state[0])
agent_y_targets = np.full(len(agent.real_target_positions), agent.model.state[1])

# Plot associated target vectors in red
if associated_indices:
    associated_targets = agent.real_target_positions[list(associated_indices)]
    agent_x_assoc_targets = np.full(len(associated_targets), agent.model.state[0])
    agent_y_assoc_targets = np.full(len(associated_targets), agent.model.state[1])
    plt.quiver(agent_x_assoc_targets, agent_y_assoc_targets,
               associated_targets[:, 0] - agent.model.state[0], 
               associated_targets[:, 1] - agent.model.state[1], 
               angles='xy', scale_units='xy', scale=1, color='red', alpha=0.5, label='Associated Target Vectors')

# Plot unassociated target vectors in orange
unassociated_indices = [i for i in range(len(agent.real_target_positions)) if i not in associated_indices]
if unassociated_indices:
    unassociated_targets = agent.real_target_positions[unassociated_indices]
    agent_x_unassoc_targets = np.full(len(unassociated_targets), agent.model.state[0])
    agent_y_unassoc_targets = np.full(len(unassociated_targets), agent.model.state[1])
    plt.quiver(agent_x_unassoc_targets, agent_y_unassoc_targets,
               unassociated_targets[:, 0] - agent.model.state[0], 
               unassociated_targets[:, 1] - agent.model.state[1], 
               angles='xy', scale_units='xy', scale=1, color='orange', alpha=0.7, 
               label='Unassociated Target Vectors')

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Agent, Targets, Measurements and Associated Measurements')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.grid()
plt.legend()
plt.show()
