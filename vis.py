import numpy as np
from my_erg_lib.agent import Agent
from my_erg_lib.obstacles import Obstacle


def plotPhi(agent, phi_rec_from_ck, phi_rec_from_agent, all_traj=None, grid_res=50, clip_to_min_max=False):
    phi_original = agent.basis.phi

    x1 = np.linspace(0, agent.L1, grid_res)
    x2 = np.linspace(0, agent.L2, grid_res)

    # Plot in a 1x3 matplotlib figure as heatmap colors
    import matplotlib.pyplot as plt
    from matplotlib import cm

    Z_original = np.zeros((len(x1), len(x2)))
    Z_agent_fourier_rec = np.zeros((len(x1), len(x2)))
    Z_rec_from_ck = np.zeros((len(x1), len(x2)))

    for i in range(len(x1)):
        for j in range(len(x2)):
            Z_original[j, i] = phi_original([x1[i], x2[j]])
            Z_rec_from_ck[j, i] = phi_rec_from_ck([x1[i], x2[j]])
            Z_agent_fourier_rec[j, i] = phi_rec_from_agent([x1[i], x2[j]])
        
    fig = plt.figure(figsize=(18, 6))
    
    ax1 = fig.add_subplot(131)
    im1 = ax1.imshow(Z_original, extent=(0, agent.L1, 0, agent.L2), origin='lower', cmap=cm.viridis)
    ax1.set_title('Original Function Φ')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_aspect('auto')
    plt.colorbar(im1, ax=ax1, label='Function Value')

    ax2 = fig.add_subplot(132)
    im2 = ax2.imshow(Z_agent_fourier_rec,
                     extent=(0, agent.L1, 0, agent.L2), origin='lower', cmap=cm.viridis)
    ax2.set_title(f'Fourier Reconstruction (Kmax = {agent.Kmax})')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_aspect('auto')
    plt.colorbar(im2, ax=ax2, label='Function Value')

    # min and max of Z_agent_fourier_rec
    if clip_to_min_max:
        min_val = np.min(Z_agent_fourier_rec)
        max_val = np.max(Z_agent_fourier_rec)

    ax3 = fig.add_subplot(133)
    if clip_to_min_max:
        im3 = ax3.imshow(Z_rec_from_ck, extent=(0, agent.L1, 0, agent.L2), 
                        origin='lower', cmap=cm.viridis, vmin=min_val, vmax=max_val)
    else:
        im3 = ax3.imshow(Z_rec_from_ck, extent=(0, agent.L1, 0, agent.L2), 
                        origin='lower', cmap=cm.viridis)
    ax3.set_title('Reconstructed from Ck')
    ax3.set_xlabel('x1')
    ax3.set_ylabel('x2')
    ax3.set_aspect('auto')
    # x and y lims to 0 -> agent.L1 and 0 -> agent.L2
    ax3.set_xlim(0, agent.L1)
    ax3.set_ylim(0, agent.L2)
    plt.colorbar(im3, ax=ax3, label='Function Value')
    
    # Plot obstacles as circles if they exist
    for ax in [ax1, ax2, ax3]:
        if hasattr(agent, 'obstacle_list') and agent.obstacle_list:
            for obstacle in agent.obstacle_list:
                if isinstance(obstacle, Obstacle):
                    if obstacle.type == 'circle':
                        # Draw circle representing the obstacle
                        circle = plt.Circle((obstacle.pos[0]+agent.L1/(grid_res+1)/2, obstacle.pos[1]), 
                                        obstacle.r, 
                                        color='black', fill=False, linestyle='--', linewidth=1)
                        ax.add_patch(circle)
                    elif obstacle.type == 'rectangle':
                        # Draw rectangle representing the obstacle
                        rect = plt.Rectangle((obstacle.bottom_left[0]+agent.L1/(grid_res+1)/2, obstacle.bottom_left[1]), 
                                            obstacle.width, obstacle.height, 
                                            color='black', fill=False, linestyle='--', linewidth=1)
                        ax.add_patch(rect)

        # Lets also visualise the barrier
        W = 0
        ax.add_patch(plt.Rectangle((0, 0), agent.L1, agent.L2, color='black', fill=False, linestyle='--', linewidth=1))
        ax.add_patch(plt.Rectangle((W, W), agent.L1-2*W, agent.L2-2*W, color='black', fill=False, linestyle='--', linewidth=1))

    # In ax3 i want to plot the ellipse of the target pos estimate, with center at agent.a, and sigma = agent.ekf.sigma_k_1
    if hasattr(agent, 'ekf') and hasattr(agent.ekf, 'sigma_k_1'):
        sigma = agent.ekf.sigma_k_1
        center = agent.ekf.a_k_1[:2]
        # Get the eigenvalues and eigenvectors of the covariance matrix
        eigvals, eigvecs = np.linalg.eig(sigma[:2, :2])
        # Calculate the angle of rotation
        angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
        # Calculate the width and height of the ellipse
        width = 2 * np.sqrt(eigvals[0])
        height = 2 * np.sqrt(eigvals[1])
        width *= 3  # Scale to show 3 sigma case
        height *= 3  # Scale to show 3 sigma case
        # Create the ellipse
        from matplotlib.patches import Ellipse
        ellipse = Ellipse(center, width=width, height=height, angle=angle*180/np.pi, color='blue', fill=False, linestyle='--', linewidth=1)
        ax3.add_patch(ellipse)
        # Plot the estimated target position
        ax3.plot(agent.ekf.a_k_1[0], agent.ekf.a_k_1[1], 'bx', markersize=5, label='Estimated Target Position')
        # Plot the real target position
        ax3.plot(agent.real_target_position[0], agent.real_target_position[1], 'rx', markersize=5, label='Ground Truth')

    # Lets also draw the agent.sensor.sensor_range circle around the current agent position
    if hasattr(agent, 'sensor') and hasattr(agent.sensor, 'sensor_range'):
        sensor_range = agent.sensor.sensor_range
        # Draw circle representing the sensor range
        sensor_circle = plt.Circle((agent.model.state[0], agent.model.state[1]), 
                                   sensor_range, color='red', fill=False, linestyle='--', linewidth=1)
        ax3.add_patch(sensor_circle)
        # Plot the agent position
        ax3.plot(agent.model.state[0], agent.model.state[1], 'go', markersize=5, label='Agent Position')


    if all_traj is not None:
        all_traj = np.array(all_traj)
        ax3.plot(all_traj[:, 0], all_traj[:, 1], 'k-', label='Trajectory')
        # plot also the buffer
        buffer_traj = agent.erg_c.past_states_buffer.get()
        ax3.plot(buffer_traj[:, 0], buffer_traj[:, 1], 'r-', label='Buffer Trajectory')
        # if phi_3 is not None:
        #     ax2.plot(x_traj[:, 1], x_traj[:, 0], 'r-', label='Trajectory')

    plt.tight_layout()


def plotPhi3D(agent, phi_rec_from_ck, phi_rec_from_agent, all_traj=None, grid_res=50, clip_to_min_max=False):
    phi_original = agent.basis.phi

    x1 = np.linspace(0, agent.L1, grid_res)
    x2 = np.linspace(0, agent.L2, grid_res)
    X1, X2 = np.meshgrid(x1, x2)

    # Plot in a 1x3 matplotlib figure as 3D surface plots
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D

    Z_original = np.zeros((len(x1), len(x2)))
    Z_agent_fourier_rec = np.zeros((len(x1), len(x2)))
    Z_rec_from_ck = np.zeros((len(x1), len(x2)))

    for i in range(len(x1)):
        for j in range(len(x2)):
            Z_original[j, i] = phi_original([x1[i], x2[j]])
            Z_rec_from_ck[j, i] = phi_rec_from_ck([x1[i], x2[j]])
            Z_agent_fourier_rec[j, i] = phi_rec_from_agent([x1[i], x2[j]])
        
    fig = plt.figure(figsize=(24, 8))
    
    # Original function plot
    ax1 = fig.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(X1, X2, Z_original, cmap=cm.viridis, alpha=0.8)
    ax1.set_title('True Target Distribution Φ', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X Position [m]')
    ax1.set_ylabel('Y Position [m]')
    ax1.set_zlabel('Probability Density')
    ax1.set_zlim(0, 15)
    fig.colorbar(surf1, ax=ax1, label='Probability Density', shrink=0.5)

    # Fourier reconstruction plot
    ax2 = fig.add_subplot(132, projection='3d')
    surf2 = ax2.plot_surface(X1, X2, Z_agent_fourier_rec, cmap=cm.viridis, alpha=0.8)
    ax2.set_title(f'Agent Fourier Estimate (K_max = {agent.Kmax})', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X Position [m]')
    ax2.set_ylabel('Y Position [m]')
    ax2.set_zlabel('Probability Density')
    ax2.set_zlim(0, 15)
    fig.colorbar(surf2, ax=ax2, label='Probability Density', shrink=0.5)

    # Reconstruction from Ck plot
    ax3 = fig.add_subplot(133, projection='3d')
    max_z = np.max(Z_rec_from_ck)
    Z_rec_from_ck = Z_rec_from_ck/max_z * 14 if max_z > 14 else Z_rec_from_ck
    if clip_to_min_max:
        min_val = np.min(Z_agent_fourier_rec)
        max_val = np.max(Z_agent_fourier_rec)
        surf3 = ax3.plot_surface(X1, X2, Z_rec_from_ck, cmap=cm.viridis, alpha=0.8, 
                                vmin=min_val, vmax=max_val)
    else:
        surf3 = ax3.plot_surface(X1, X2, Z_rec_from_ck, cmap=cm.viridis, alpha=0.8)
    ax3.set_title('Reconstructed Distribution from C_k', fontsize=12, fontweight='bold')
    ax3.set_xlabel('X Position [m]')
    ax3.set_ylabel('Y Position [m]')
    ax3.set_zlabel('Probability Density')
    ax3.set_xlim(0, agent.L1)
    ax3.set_ylim(0, agent.L2)
    ax3.set_zlim(0, 15)
    fig.colorbar(surf3, ax=ax3, label='Probability Density', shrink=0.5)
    
    # Plot obstacles as 3D extruded shapes
    for ax in [ax1, ax2, ax3]:
        if hasattr(agent, 'obstacle_list') and agent.obstacle_list:
            for obstacle in agent.obstacle_list:
                if isinstance(obstacle, Obstacle):
                    z_min = 0
                    z_max = 15
                    
                    if obstacle.type == 'circle':
                        # Create cylindrical obstacle with gray fill
                        theta = np.linspace(0, 2*np.pi, 15)
                        z_cyl = np.linspace(z_min, z_max, 10)
                        theta_mesh, z_mesh = np.meshgrid(theta, z_cyl)
                        
                        x_cyl = obstacle.pos[0] + obstacle.r * np.cos(theta_mesh)
                        y_cyl = obstacle.pos[1] + obstacle.r * np.sin(theta_mesh)
                        
                        ax.plot_surface(x_cyl, y_cyl, z_mesh, color='gray', alpha=0.6)
                        
                    elif obstacle.type == 'rectangle':
                        # Create rectangular prism obstacle with gray fill
                        x_coords = [obstacle.bottom_left[0], obstacle.bottom_left[0] + obstacle.width]
                        y_coords = [obstacle.bottom_left[1], obstacle.bottom_left[1] + obstacle.height]
                        z_coords = [z_min, z_max]
                        
                        # Create meshgrid for the 6 faces of the rectangular prism
                        X_face, Z_face = np.meshgrid(x_coords, z_coords)
                        Y_face, Z_face2 = np.meshgrid(y_coords, z_coords)
                        X_face2, Y_face2 = np.meshgrid(x_coords, y_coords)
                        
                        # Plot all 6 faces
                        # Front and back faces
                        ax.plot_surface(X_face, np.full_like(X_face, y_coords[0]), Z_face, color='gray', alpha=0.6)
                        ax.plot_surface(X_face, np.full_like(X_face, y_coords[1]), Z_face, color='gray', alpha=0.6)
                        # Left and right faces
                        ax.plot_surface(np.full_like(Y_face, x_coords[0]), Y_face, Z_face2, color='gray', alpha=0.6)
                        ax.plot_surface(np.full_like(Y_face, x_coords[1]), Y_face, Z_face2, color='gray', alpha=0.6)
                        # Top and bottom faces
                        ax.plot_surface(X_face2, Y_face2, np.full_like(X_face2, z_coords[0]), color='gray', alpha=0.6)
                        ax.plot_surface(X_face2, Y_face2, np.full_like(X_face2, z_coords[1]), color='gray', alpha=0.6)

        # Visualize domain boundaries as wireframe
        z_min = 0
        z_max = 15
        
        # Draw domain boundary wireframe
        boundary_x = [0, agent.L1, agent.L1, 0, 0]
        boundary_y = [0, 0, agent.L2, agent.L2, 0]
        
        # Bottom and top boundaries
        for z_level in [z_min, z_max]:
            boundary_z = [z_level] * 5
            ax.plot(boundary_x, boundary_y, boundary_z, 'k--', alpha=0.3, linewidth=1)
        
        # Vertical edges
        for i in range(4):
            ax.plot([boundary_x[i], boundary_x[i]], [boundary_y[i], boundary_y[i]], 
                   [z_min, z_max], 'k--', alpha=0.3, linewidth=1)

    # In ax3, plot estimation and tracking information
    if hasattr(agent, 'ekf') and hasattr(agent.ekf, 'sigma_k_1'):
        sigma = agent.ekf.sigma_k_1
        center = agent.ekf.a_k_1[:2]
        
        # Get the eigenvalues and eigenvectors of the covariance matrix
        eigvals, eigvecs = np.linalg.eig(sigma[:2, :2])
        
        # Create uncertainty ellipse at surface level
        theta = np.linspace(0, 2*np.pi, 50)
        width = 2 * np.sqrt(eigvals[0]) * 3  # 3 sigma
        height = 2 * np.sqrt(eigvals[1]) * 3  # 3 sigma
        angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
        
        # Parametric ellipse
        ellipse_x = center[0] + width/2 * np.cos(theta) * np.cos(angle) - height/2 * np.sin(theta) * np.sin(angle)
        ellipse_y = center[1] + width/2 * np.cos(theta) * np.sin(angle) + height/2 * np.sin(theta) * np.cos(angle)
        
        # Project ellipse onto the surface
        ellipse_z = np.zeros_like(ellipse_x)
        for i in range(len(ellipse_x)):
            try:
                ellipse_z[i] = phi_rec_from_ck([ellipse_x[i], ellipse_y[i]]) + 0.5
            except:
                ellipse_z[i] = 2
        
        ax3.plot(ellipse_x, ellipse_y, ellipse_z, 'blue', linewidth=2, alpha=0.8, label='3σ Uncertainty Ellipse')
        
        # Plot estimated target position
        est_z = phi_rec_from_ck([agent.ekf.a_k_1[0], agent.ekf.a_k_1[1]]) + 1 if callable(phi_rec_from_ck) else 3
        ax3.scatter(agent.ekf.a_k_1[0], agent.ekf.a_k_1[1], est_z, 
                   color='blue', s=100, marker='x', linewidth=3, label='Estimated Target Position')
        
        # Plot real target position
        real_z = phi_rec_from_ck([agent.real_target_position[0], agent.real_target_position[1]]) + 1 if callable(phi_rec_from_ck) else 3
        ax3.scatter(agent.real_target_position[0], agent.real_target_position[1], real_z, 
                   color='red', s=100, marker='*', linewidth=2, label='True Target Position')

    # Sensor range visualization
    if hasattr(agent, 'sensor') and hasattr(agent.sensor, 'sensor_range'):
        sensor_range = agent.sensor.sensor_range
        
        # Create sensor range circle at surface level
        theta = np.linspace(0, 2*np.pi, 50)
        sensor_x = agent.model.state[0] + sensor_range * np.cos(theta)
        sensor_y = agent.model.state[1] + sensor_range * np.sin(theta)
        
        # Project sensor range onto surface
        sensor_z = np.zeros_like(sensor_x)
        for i in range(len(sensor_x)):
            try:
                sensor_z[i] = phi_rec_from_ck([sensor_x[i], sensor_y[i]]) + 0.2
            except:
                sensor_z[i] = 1
        
        ax3.plot(sensor_x, sensor_y, sensor_z, 'orange', linewidth=2, alpha=0.7, 
                linestyle='--', label='Sensor Range')
        
        # Plot agent position as a single prominent dot at the top
        agent_z = 18  # Place near the top of the plot
        ax3.scatter(agent.model.state[0], agent.model.state[1], agent_z, 
                   color='green', s=150, marker='o', edgecolor='black', linewidth=2, 
                   label='UAV Position', zorder=10)

    # Plot trajectories
    if all_traj is not None:
        all_traj = np.array(all_traj)
        
        # Project trajectory onto surface
        traj_z = np.zeros(len(all_traj))
        for i, pos in enumerate(all_traj):
            try:
                traj_z[i] = phi_rec_from_ck(pos) + 0.2 if callable(phi_rec_from_ck) else 2
            except:
                traj_z[i] = 2
        
        ax3.plot(all_traj[:, 0], all_traj[:, 1], traj_z, 'black', linewidth=2, 
                alpha=0.8, label='Flight Trajectory')
        
        # Plot buffer trajectory
        if hasattr(agent.erg_c, 'past_states_buffer'):
            buffer_traj = agent.erg_c.past_states_buffer.get()
            buffer_z = np.zeros(len(buffer_traj))
            for i, pos in enumerate(buffer_traj):
                try:
                    buffer_z[i] = phi_rec_from_ck(pos) + 0.2 if callable(phi_rec_from_ck) else 2
                except:
                    buffer_z[i] = 2
            
            ax3.plot(buffer_traj[:, 0], buffer_traj[:, 1], buffer_z, 'darkred', 
                    linewidth=2, alpha=0.8, label='Recent Trajectory')

    # Add legend outside the plot
    if hasattr(agent, 'ekf') or (hasattr(agent, 'sensor') and hasattr(agent.sensor, 'sensor_range')):
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    plt.subplots_adjust(left=0.05, right=0.85, bottom=0.1, top=0.9, wspace=0.3)

# Visualise the quad using matplotlib 3D from collected data ----------------------
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np
def animateQuadcopter(time_list, states_list,
                      frame_skip=30):
    
    # Create figure for 3D animation
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Quadcopter visualization parameters
    arm_length = 0.2
    rotor_size = 0.05

    # Skip frames to speed up animation (higher = faster)
    selected_frames = range(0, len(states_list), frame_skip)
    animation_states = states_list[::frame_skip]
    animation_times = time_list[::frame_skip]

    def update_plot(i):
        ax.clear()
        
        # Get index in the original data
        frame = selected_frames[i]
        
        # Get quadcopter state from the pre-collected data
        current_state = states_list[frame]
        x, y, z = current_state[0], current_state[1], current_state[2]
        psi, theta, phi = current_state[3], current_state[4], current_state[5]
        
        # Set axis limits
        ax.set_xlim([x - 2, x + 2])
        ax.set_ylim([y - 2, y + 2])
        ax.set_zlim([max(0, z - 2), z + 2])
        
        # Plot trajectory up to current frame
        if frame > 0:
            states_array = states_list[:frame+1:max(1,frame//50)]  # Sample trajectory points for efficiency
            ax.plot(states_array[:, 0], states_array[:, 1], states_array[:, 2], 'b-', alpha=0.7)
        
        # Draw quadcopter body
        center = np.array([x, y, z])
        
        # Apply rotation matrices for orientation
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)]
        ])
        
        R_y = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
        
        R_z = np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi), np.cos(psi), 0],
            [0, 0, 1]
        ])
        
        R = np.dot(R_z, np.dot(R_y, R_x))
        
        # Define the four arms
        arm1 = np.dot(R, np.array([arm_length, 0, 0]))
        arm2 = np.dot(R, np.array([0, arm_length, 0]))
        arm3 = np.dot(R, np.array([-arm_length, 0, 0]))
        arm4 = np.dot(R, np.array([0, -arm_length, 0]))
        
        # Draw arms
        ax.plot([center[0], center[0] + arm1[0]], [center[1], center[1] + arm1[1]], 
                [center[2], center[2] + arm1[2]], 'r-', linewidth=2)
        ax.plot([center[0], center[0] + arm2[0]], [center[1], center[1] + arm2[1]], 
                [center[2], center[2] + arm2[2]], 'g-', linewidth=2)
        ax.plot([center[0], center[0] + arm3[0]], [center[1], center[1] + arm3[1]], 
                [center[2], center[2] + arm3[2]], 'b-', linewidth=2)
        ax.plot([center[0], center[0] + arm4[0]], [center[1], center[1] + arm4[1]], 
                [center[2], center[2] + arm4[2]], 'y-', linewidth=2)
        
        # Draw rotors
        ax.scatter(center[0] + arm1[0], center[1] + arm1[1], center[2] + arm1[2], 
                color='red', s=rotor_size*100)
        ax.scatter(center[0] + arm2[0], center[1] + arm2[1], center[2] + arm2[2], 
                color='green', s=rotor_size*100)
        ax.scatter(center[0] + arm3[0], center[1] + arm3[1], center[2] + arm3[2], 
                color='blue', s=rotor_size*100)
        ax.scatter(center[0] + arm4[0], center[1] + arm4[1], center[2] + arm4[2], 
                color='yellow', s=rotor_size*100)
        
        # Set labels and title
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
        ax.set_title(f'Quadcopter Simulation - Time: {time_list[frame]:.2f}s')
        
        return ax

    # Create animation with faster interval and fewer frames
    ani = animation.FuncAnimation(fig, update_plot, frames=len(selected_frames), 
                                interval=20, blit=False, repeat=True)

    plt.show()

    # Uncomment to save animation to file
    # ani.save('quadcopter_animation.mp4', writer='ffmpeg', fps=30)
    # ------------------------------------------------------


def plotQuadTrajWithInputs(time_list, states_list, input_list, conv_inp_list=None, quad_model=None):
    
    if conv_inp_list is None:
        if quad_model is None:
            raise ValueError("If conv_inp_list is provided, quad_model must be provided as well.")
        conv_inp_list = np.array([quad_model.convertInputToMotorCommands(u) for u in input_list])
    
    time_list = np.asarray(time_list)
    states_list = np.asarray(states_list)
    input_list = np.asarray(input_list)
    conv_inp_list = np.asarray(conv_inp_list)

    # Plotting --------------------------------------------
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Position
    axs[0, 0].plot(time_list, states_list[:, 0], label='x')
    axs[0, 0].plot(time_list, states_list[:, 1], label='y')
    axs[0, 0].plot(time_list, states_list[:, 2], label='z')
    axs[0, 0].set_title('Position [m]')
    axs[0, 0].legend()
    axs[0, 0].grid()

    # Angles
    axs[0, 1].plot(time_list, states_list[:, 3] * 180 / np.pi, label='ψ')
    axs[0, 1].plot(time_list, states_list[:, 4] * 180 / np.pi, label='θ')
    axs[0, 1].plot(time_list, states_list[:, 5] * 180 / np.pi, label='φ')
    axs[0, 1].set_title('Angles [deg]')
    axs[0, 1].legend()
    axs[0, 1].grid()

    # Linear velocities
    axs[1, 0].plot(time_list, states_list[:, 6], label='x\'')
    axs[1, 0].plot(time_list, states_list[:, 7], label='y\'')
    axs[1, 0].plot(time_list, states_list[:, 8], label='z\'')
    axs[1, 0].set_title('Linear Velocities [m/s]')
    axs[1, 0].legend()
    axs[1, 0].grid()

    # Angular velocities
    axs[1, 1].plot(time_list, states_list[:, 9] * 180 / np.pi, label='ψ\'')
    axs[1, 1].plot(time_list, states_list[:, 10] * 180 / np.pi, label='θ\'')
    axs[1, 1].plot(time_list, states_list[:, 11] * 180 / np.pi, label='φ\'')
    axs[1, 1].set_title('Angular Velocities [deg/s]')
    axs[1, 1].legend()
    axs[1, 1].grid()

    plt.tight_layout()

    # Plotting control inputs --------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Control Inputs
    ax1.plot(time_list, input_list[:, 0], label='u1 - Thrust', marker = ".")
    ax1.plot(time_list, input_list[:, 1], label='u2 - Yaw Torque', marker = ".")
    ax1.plot(time_list, input_list[:, 2], label='u3 - Pitch Torque', marker = ".")
    ax1.plot(time_list, input_list[:, 3], label='u4 - Roll Torque', marker = ".")
    ax1.set_title('Control Inputs')
    ax1.legend()
    ax1.grid()

    # Motor Commands
    ax2.plot(time_list, conv_inp_list[:, 0], label='Motor 1', marker = ".")
    ax2.plot(time_list, conv_inp_list[:, 1], label='Motor 2', marker = ".")
    ax2.plot(time_list, conv_inp_list[:, 2], label='Motor 3', marker = ".")
    ax2.plot(time_list, conv_inp_list[:, 3], label='Motor 4', marker = ".")
    ax2.set_title('Motor Commands')
    ax2.legend()
    ax2.grid()

    plt.tight_layout()
    plt.show()


def simplePlot(x, y, label_list=None, title=None, x_label=None, y_label=None, y_type=None, x_lim=None, y_lim=None, T_SHOW=None, fig_num=None):    
    """
    Simple plot function to plot multiple y values against x.
    Parameters:
    - x: x values (1D array)
    - y_list: list of y values (2D array)
    - label_list: list of labels for each y value (optional)
    - T_SHOW: time in seconds to pause the plot before continuing execution (optional)
    - fig_num: figure number to reuse an existing figure (optional)
    """
    # Use interactive mode to prevent blocking
    plt.ion()
    
    # Clear and reuse figure if fig_num is provided
    if fig_num is not None:
        fig = plt.figure(fig_num)
        plt.clf()  # Clear the figure
    else:
        fig = plt.figure()
        
    if y_type == "list":
        for i, y_vals in enumerate(y):
            plt.plot(x, y_vals, label=f"y{i}" if label_list is None else label_list[i])
    elif y_type == "np.array":
        for i in range(y.shape[1]):
            plt.plot(x, y[:, i], label=f"y{i}" if label_list is None else label_list[i])
    else: 
        raise ValueError("y_type must be 'list' or 'np.array'.")
    
    plt.title(title if title is not None else "Simple Plot")
    plt.xlabel(x_label if x_label is not None else "X-axis")
    plt.ylabel(y_label if y_label is not None else "Y-axis")
    plt.xlim(x_lim if x_lim is not None else (None, None))
    plt.ylim(y_lim if y_lim is not None else (None, None))
    plt.legend()
    plt.grid()
    
    # Draw and display the figure
    plt.draw()
    
    # Pause for T_SHOW seconds if specified
    if T_SHOW is not None:
        plt.pause(T_SHOW)
    else:
        plt.pause(0.001)  # Small pause to update the figure
    
    # Return the figure number for reuse
    return fig.number


def visualiseColorForTraj(agent, ck, x_traj):
    '''
    Visualise the color for the trajectory using ck coefficients
    '''
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from my_erg_lib.basis import ReconstructedPhiFromCk
    # Reconstruct the target distribution using the coefficients
    phi_from_ck = ReconstructedPhiFromCk(agent.basis, ck)

    x1 = np.linspace(0, 1, 50)
    x2 = np.linspace(0, 1, 50)


    Z_reconstructed = np.zeros((len(x1), len(x2)))

    for i in range(len(x1)):
        for j in range(len(x2)):
            Z_reconstructed[i, j] = phi_from_ck([x1[i], x2[j]])
        
    plt.figure(figsize=(6, 6))
    plt.imshow(Z_reconstructed.T, extent=(0, 1, 0, 1), origin='lower', cmap=cm.viridis)
    plt.title(f'Reconstructed Function (Kmax={agent.Kmax})')
    plt.xlabel('x1')
    plt.ylabel('x2')

    if x_traj is not None:
        plt.plot(x_traj[:, 0], x_traj[:, 1], 'r-', label='Trajectory')

    plt.tight_layout()
    plt.show()

def visualiseCoefficients(agent, ck):
    import matplotlib.pyplot as plt
    from matplotlib import cm

    k1 = np.linspace(0, agent.Kmax, agent.Kmax+1)
    k2 = np.linspace(0, agent.Kmax, agent.Kmax+1)
    K1, K2 = np.meshgrid(k1, k2)
    Z_ck = np.zeros((len(k1), len(k2)))
    Z_phik = np.zeros((len(k1), len(k2)))
    
    for i in range(len(k1)):
        for j in range(len(k2)):
            Z_ck[i, j] = ck[i, j]
            Z_phik[i, j] = agent.basis.calcPhikCoeff(int(k1[i]), int(k2[j]))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot Ck coefficients
    im1 = ax1.imshow(Z_ck, cmap=cm.viridis, origin='lower', 
                    extent=[0, agent.Kmax, 0, agent.Kmax], aspect='equal')
    ax1.set_title('Ck Coefficients')
    ax1.set_xlabel('k1')
    ax1.set_ylabel('k2')
    fig.colorbar(im1, ax=ax1, label='Ck Value')
    
    # Plot Phi_k coefficients
    im2 = ax2.imshow(Z_phik, cmap=cm.viridis, origin='lower', 
                    extent=[0, agent.Kmax, 0, agent.Kmax], aspect='equal')
    ax2.set_title('Phi_k Coefficients')
    ax2.set_xlabel('k1')
    ax2.set_ylabel('k2')
    fig.colorbar(im2, ax=ax2, label='Phi_k Value')
    
    plt.tight_layout()
    plt.show()

from my_erg_lib.model_dynamics import Quadcopter
def visPotentialFields(agent: Agent):
    # Filter the obstacle list to only keep circle obstacles
    obs_list = [obs for obs in agent.obstacle_list if (obs.type == 'circle' or obs.type == "rectangle")] if hasattr(agent, 'obstacle_list') else []

    x = np.linspace(0, agent.L1, 100)
    y = np.linspace(0, agent.L2, 100)
    X, Y = np.meshgrid(x, y)
    skip = 2  # Skip points for better visualization

    # Check if obstacle and wall controllers exist
    has_obstacle_controller = False
    has_wall_controller = False
    obstacle_idx = -1
    wall_idx = -1
    
    if hasattr(agent.erg_c.uNominal, 'additional_functions'):
        for i, controller in enumerate(agent.erg_c.uNominal.additional_functions):
            if hasattr(controller, '__name__'):
                if controller.__name__ == "Obstacles":
                    has_obstacle_controller = True
                    obstacle_idx = i
                elif controller.__name__ == "Walls":
                    has_wall_controller = True
                    wall_idx = i

    # Initialize fields only if needed
    f_obs = None
    f_bar = None
    
    # Calculate obstacle field if it exists
    if has_obstacle_controller:
        f_obs = np.zeros((len(x), len(y), 2))
        for i in range(len(x)):
            for j in range(len(y)):
                if isinstance(agent.model, Quadcopter):
                    _ = agent.erg_c.uNominal.additional_functions[obstacle_idx]([x[i], y[j]], 0)
                    f_obs[j, i, :] = agent.model.f_command_to_controller
                else:
                    f_obs[j, i, :] = agent.erg_c.uNominal.additional_functions[obstacle_idx]([x[i], y[j]], 0)
    
    # Calculate wall/barrier field if it exists
    if has_wall_controller:
        f_bar = np.zeros((len(x), len(y), 2))
        for i in range(len(x)):
            for j in range(len(y)):
                if isinstance(agent.model, Quadcopter):
                    _ = agent.erg_c.uNominal.additional_functions[wall_idx]([x[i], y[j]], 0)
                    f_bar[j, i, :] = agent.model.f_command_to_controller
                else:
                    f_bar[j, i, :] = agent.erg_c.uNominal.additional_functions[wall_idx]([x[i], y[j]], 0)

    # Only proceed with plotting if at least one field exists
    if not (has_obstacle_controller or has_wall_controller):
        print("No obstacle or wall controllers found. Nothing to plot.")
        print("The keywords for the controllers are 'Obstacles' and 'Walls'.")
        return

    # Helper function to draw obstacles based on their type
    def draw_obstacles(ax, obstacles):
        for obstacle in obstacles:
            if obstacle.type == 'circle':
                circle = plt.Circle((obstacle.pos[0], obstacle.pos[1]), obstacle.r, 
                                   color='red', fill=True, alpha=0.5)
                ax.add_patch(circle)
            elif obstacle.type == 'rectangle':
                rect = plt.Rectangle((obstacle.bottom_left[0], obstacle.bottom_left[1]), 
                                    obstacle.width, obstacle.height, 
                                    color='red', fill=True, alpha=0.5)
                ax.add_patch(rect)
    
    # Determine the plot layout based on which fields exist
    if has_obstacle_controller and has_wall_controller:
        # Plot both obstacle and wall fields
        plt.figure(figsize=(15, 8))
        
        # Setup 2x3 grid for X and Y components
        plt.subplot(2, 3, 1)
        # f_obs x-direction
        plt.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                f_obs[::skip, ::skip, 0], np.zeros_like(f_obs[::skip, ::skip, 0]),
                scale=5, color='blue', width=0.003)
        field_magnitude = np.abs(f_obs[:,:,0])
        plt.contourf(X, Y, field_magnitude, cmap='viridis', alpha=0.3)
        plt.colorbar(label='Field Magnitude')

        # Draw the obstacles
        draw_obstacles(plt.gca(), obs_list)

        plt.title('Obstacle Avoidance Field (X-direction)')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.grid(True)

        plt.subplot(2, 3, 2)
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
        
        plt.subplot(2, 3, 3)
        # Combined x-direction (obstacle + barrier)
        combined_x = f_obs[:,:,0] + f_bar[:,:,0]
        plt.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                combined_x[::skip, ::skip], np.zeros_like(combined_x[::skip, ::skip]),
                scale=25, color='magenta', width=0.003)
        field_magnitude = np.abs(combined_x)
        plt.contourf(X, Y, field_magnitude, cmap='viridis', alpha=0.3)
        plt.colorbar(label='Field Magnitude')
        
        # Draw the obstacles
        draw_obstacles(plt.gca(), obs_list)
        
        plt.title('Combined Field (X-direction)')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.grid(True)

        plt.subplot(2, 3, 4)
        # f_obs y-direction
        plt.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                np.zeros_like(f_obs[::skip, ::skip, 1]), f_obs[::skip, ::skip, 1],
                scale=5, color='red', width=0.003)
        field_magnitude = np.abs(f_obs[:,:,1])
        plt.contourf(X, Y, field_magnitude, cmap='viridis', alpha=0.3)
        plt.colorbar(label='Field Magnitude')

        # Draw the obstacles
        draw_obstacles(plt.gca(), obs_list)

        plt.title('Obstacle Avoidance Field (Y-direction)')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.grid(True)

        plt.subplot(2, 3, 5)
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
        
        plt.subplot(2, 3, 6)
        # Combined y-direction (obstacle + barrier)
        combined_y = f_obs[:,:,1] + f_bar[:,:,1]
        plt.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                np.zeros_like(combined_y[::skip, ::skip]), combined_y[::skip, ::skip],
                scale=25, color='cyan', width=0.003)
        field_magnitude = np.abs(combined_y)
        plt.contourf(X, Y, field_magnitude, cmap='viridis', alpha=0.3)
        plt.colorbar(label='Field Magnitude')
        
        # Draw the obstacles
        draw_obstacles(plt.gca(), obs_list)
        
        plt.title('Combined Field (Y-direction)')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.grid(True)
        plt.tight_layout()

        # Combined vector fields figure
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

        # Draw the obstacles
        draw_obstacles(plt.gca(), obs_list)

        plt.title('Obstacle Avoidance Field (Combined X-Y directions)')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.grid(True)

        plt.subplot(2, 1, 2)
        # Combined barrier field (both x and y directions)
        plt.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                f_bar[::skip, ::skip, 0], f_bar[::skip, ::skip, 1],
                scale=25, color='green', width=0.003)
        field_magnitude = np.sqrt(f_bar[:,:,0]**2 + f_bar[:,:,1]**2)
        plt.contourf(X, Y, field_magnitude, cmap='viridis', alpha=0.3)
        plt.colorbar(label='Field Magnitude')
        plt.title('Barrier Field (Combined X-Y directions)')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.grid(True)
        plt.tight_layout()

        # Total combined field figure
        plt.figure(figsize=(6, 5))
        plt.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
            f_obs[::skip, ::skip, 0] + f_bar[::skip, ::skip, 0], 
            f_obs[::skip, ::skip, 1] + f_bar[::skip, ::skip, 1],
            scale=25, color='magenta', width=0.003)
        field_magnitude = np.sqrt((f_obs[:,:,0] + f_bar[:,:,0])**2 + (f_obs[:,:,1] + f_bar[:,:,1])**2)
        plt.contourf(X, Y, field_magnitude, cmap='viridis', alpha=0.3)
        plt.colorbar(label='Field Magnitude')

        # Draw the obstacles
        draw_obstacles(plt.gca(), obs_list)

        plt.title('Total Combined Field (Obstacle + Barrier)')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.grid(True)
        plt.tight_layout()
        
    elif has_obstacle_controller:
        # Only plot obstacle field
        plt.figure(figsize=(10, 5))
        
        # X component
        plt.subplot(1, 2, 1)
        plt.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                f_obs[::skip, ::skip, 0], np.zeros_like(f_obs[::skip, ::skip, 0]),
                scale=5, color='blue', width=0.003)
        field_magnitude = np.abs(f_obs[:,:,0])
        plt.contourf(X, Y, field_magnitude, cmap='viridis', alpha=0.3)
        plt.colorbar(label='Field Magnitude')
        
        # Draw the obstacles
        draw_obstacles(plt.gca(), obs_list)
        
        plt.title('Obstacle Avoidance Field (X-direction)')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.grid(True)
        
        # Y component
        plt.subplot(1, 2, 2)
        plt.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                np.zeros_like(f_obs[::skip, ::skip, 1]), f_obs[::skip, ::skip, 1],
                scale=5, color='red', width=0.003)
        field_magnitude = np.abs(f_obs[:,:,1])
        plt.contourf(X, Y, field_magnitude, cmap='viridis', alpha=0.3)
        plt.colorbar(label='Field Magnitude')
        
        # Draw the obstacles
        draw_obstacles(plt.gca(), obs_list)
        
        plt.title('Obstacle Avoidance Field (Y-direction)')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.grid(True)
        plt.tight_layout()
        
        # Combined vector field
        plt.figure(figsize=(6, 5))
        plt.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                f_obs[::skip, ::skip, 0], f_obs[::skip, ::skip, 1],
                scale=14, color='blue', width=0.003)
        field_magnitude = np.sqrt(f_obs[:,:,0]**2 + f_obs[:,:,1]**2)
        plt.contourf(X, Y, field_magnitude, cmap='viridis', alpha=0.3)
        plt.colorbar(label='Field Magnitude')
        
        # Draw the obstacles
        draw_obstacles(plt.gca(), obs_list)
        
        plt.title('Obstacle Avoidance Field (Combined X-Y directions)')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.grid(True)
        plt.tight_layout()
        
    elif has_wall_controller:
        # Only plot wall/barrier field
        plt.figure(figsize=(10, 5))
        
        # X component
        plt.subplot(1, 2, 1)
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
        
        # Y component
        plt.subplot(1, 2, 2)
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
        
        # Combined vector field
        plt.figure(figsize=(6, 5))
        plt.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                f_bar[::skip, ::skip, 0], f_bar[::skip, ::skip, 1],
                scale=25, color='green', width=0.003)
        field_magnitude = np.sqrt(f_bar[:,:,0]**2 + f_bar[:,:,1]**2)
        plt.contourf(X, Y, field_magnitude, cmap='viridis', alpha=0.3)
        plt.colorbar(label='Field Magnitude')
        plt.title('Barrier Field (Combined X-Y directions)')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.grid(True)
        plt.tight_layout()

    plt.show()


