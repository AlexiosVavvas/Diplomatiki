# cd one dir back
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# -----------------------------------------------------------------------------------
from my_erg_lib.agent import Agent
from my_erg_lib.obstacles import Obstacle, ObstacleAvoidanceControllerGenerator
from my_erg_lib.model_dynamics import SingleIntegrator, DoubleIntegrator, Quadcopter
from my_erg_lib.ergodic_controllers import DecentralisedErgodicController
import matplotlib.pyplot as plt
import numpy as np

# Quadrotor model -----------
x0 = [0.5, 0.8, 2, 0]
model = DoubleIntegrator(dt=0.001, x0=x0)

# Agent - Ergodic Controller -------------
# Generate Agent and connect to an ergodic controller object
agent = Agent(L1=1.0, L2=1.0, Kmax=2, 
                dynamics_model=model, phi=lambda s: 2, x0=x0)

agent.erg_c = DecentralisedErgodicController(agent, uNominal=None, Q=1, uLimits=None,
                                                T_sampling=0.1, T_horizon=0.2, deltaT_erg=0.5,
                                                barrier_weight=50, barrier_eps=0.05, barrier_pow=2)

# Avoiding Obstacles -------------------
# Add obstacles and another controller to take them into account
FMAX = 0.25; EPS_M = 0.15
obs  = [Obstacle(pos=[0.2, 0.2],   dimensions=0.1,  f_max=FMAX, min_dist=0.12, eps_meters=EPS_M, obs_type='circle', obs_name="Obstacle 1"), 
        Obstacle(pos=[0.66, 0.77], dimensions=0.12, f_max=FMAX, min_dist=0.14, eps_meters=EPS_M, obs_type='circle', obs_name="Obstacle 2"), 
        Obstacle(pos=[0.6, 0.5],   dimensions=0.08, f_max=FMAX, min_dist=0.10, eps_meters=EPS_M, obs_type='circle', obs_name="Obstacle 3"),]

func_obs = ObstacleAvoidanceControllerGenerator(agent, obs_list=obs)


# Avoiding Walls ----------------------
FMAX = 1; min_dist = 7e-3; EPS_M = 0.2; e_max = agent.L1
bar  = [Obstacle(pos=[0,        0],   dimensions=[0, +1], f_max=FMAX, min_dist=min_dist, e_max=e_max, eps_meters=EPS_M,  obs_type='wall', obs_name="Bottom Wall"),
        Obstacle(pos=[0, agent.L2],   dimensions=[0, -1], f_max=FMAX, min_dist=min_dist, e_max=e_max, eps_meters=EPS_M,  obs_type='wall', obs_name="Top Wall"   ),
        Obstacle(pos=[0,        0],   dimensions=[+1, 1], f_max=FMAX, min_dist=min_dist, e_max=e_max, eps_meters=EPS_M,  obs_type='wall', obs_name="Left Wall"  ),
        Obstacle(pos=[agent.L1, 0],   dimensions=[-1, 0], f_max=FMAX, min_dist=min_dist, e_max=e_max, eps_meters=EPS_M,  obs_type='wall', obs_name="Right Wall" )]

# Add the obstacle avoidance controller to the ergodic controller
func_bar = ObstacleAvoidanceControllerGenerator(agent, obs_list=bar)

# --------------------------------------------------------------------------------------------------


x = np.linspace(-0.2, agent.L1+0.2, 100)
y = np.linspace(-0.2, agent.L2+0.2, 100)
f_obs = np.zeros((len(x), len(y), 2))
f_bar = np.zeros((len(x), len(y), 2))

for i in range(len(x)):
    for j in range(len(y)):
        # Calculate the potential field value at each point
        f_obs[j, i, :] = func_obs([x[i], y[j]], 0)
        f_bar[j, i, :] = func_bar([x[i], y[j]], 0)

# Create a meshgrid for plotting vectors
X, Y = np.meshgrid(x, y)

plt.figure(figsize=(10, 8))

# Setup 2x2 grid
plt.subplot(2, 2, 1)
# f_obs x-direction
skip = 2  # Skip points for better visualization
plt.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
           f_obs[::skip, ::skip, 0], np.zeros_like(f_obs[::skip, ::skip, 0]),
           scale=5, color='blue', width=0.003)
field_magnitude = np.abs(f_obs[:,:,0])
plt.contourf(X, Y, field_magnitude, cmap='viridis', alpha=0.3)
plt.colorbar(label='Field Magnitude')

# Draw the obstacle circles
for obstacle in obs:
    if obstacle.type == 'circle':
        circle = plt.Circle(obstacle.pos, obstacle.r, color='red', fill=True, alpha=0.5)
        plt.gca().add_patch(circle)

plt.title('Obstacle Avoidance Field (X-direction)')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)

plt.subplot(2, 2, 2)
# f_obs y-direction
plt.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
           np.zeros_like(f_obs[::skip, ::skip, 1]), f_obs[::skip, ::skip, 1],
           scale=5, color='red', width=0.003)
field_magnitude = np.abs(f_obs[:,:,1])
plt.contourf(X, Y, field_magnitude, cmap='viridis', alpha=0.3)
plt.colorbar(label='Field Magnitude')

# Draw the obstacle circles
for obstacle in obs:
    if obstacle.type == 'circle':
        circle = plt.Circle(obstacle.pos, obstacle.r, color='red', fill=True, alpha=0.5)
        plt.gca().add_patch(circle)

plt.title('Obstacle Avoidance Field (Y-direction)')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)

plt.subplot(2, 2, 3)
# f_bar x-direction
plt.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
           f_bar[::skip, ::skip, 0], np.zeros_like(f_bar[::skip, ::skip, 0]),
           scale=25, color='green', width=0.003)
field_magnitude = np.abs(f_bar[:,:,0])
plt.contourf(X, Y, field_magnitude, cmap='viridis', alpha=0.3)
plt.colorbar(label='Field Magnitude')
plt.title('Barrier Field (X-direction)')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)

plt.subplot(2, 2, 4)
# f_bar y-direction
plt.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
           np.zeros_like(f_bar[::skip, ::skip, 1]), f_bar[::skip, ::skip, 1],
           scale=25, color='purple', width=0.003)
field_magnitude = np.abs(f_bar[:,:,1])
plt.contourf(X, Y, field_magnitude, cmap='viridis', alpha=0.3)
plt.colorbar(label='Field Magnitude')
plt.title('Barrier Field (Y-direction)')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)

plt.tight_layout()


plt.figure(figsize=(6, 9))

# 2x1 grid setup
plt.subplot(2, 1, 1)
# Combined obstacle field (both x and y directions)
plt.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
           f_obs[::skip, ::skip, 0], f_obs[::skip, ::skip, 1],
           scale=14, color='blue', width=0.003)
field_magnitude = np.sqrt(f_obs[:,:,0]**2 + f_obs[:,:,1]**2)
plt.contourf(X, Y, field_magnitude, cmap='viridis', alpha=0.3)
plt.colorbar(label='Field Magnitude')

# Draw the obstacle circles
for obstacle in obs:
    if obstacle.type == 'circle':
        circle = plt.Circle(obstacle.pos, obstacle.r, color='red', fill=True, alpha=0.5)
        plt.gca().add_patch(circle)

plt.title('Obstacle Avoidance Field (Combined X-Y directions)')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)

plt.subplot(2, 1, 2)
# Combined barrier field (both x and y directions)
plt.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
           f_bar[::skip, ::skip, 0], f_bar[::skip, ::skip, 1],
           scale=14, color='green', width=0.003)
field_magnitude = np.sqrt(f_bar[:,:,0]**2 + f_bar[:,:,1]**2)
plt.contourf(X, Y, field_magnitude, cmap='viridis', alpha=0.3)
plt.colorbar(label='Field Magnitude')
plt.title('Barrier Field (Combined X-Y directions)')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)

plt.tight_layout()
plt.show()
