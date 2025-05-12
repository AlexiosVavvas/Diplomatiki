import numpy as np


def plotPhi(agent, phi_rec_from_ck, phi_rec_from_agent, all_traj=None):
    phi_original = agent.basis.phi

    x1 = np.linspace(0, agent.L1, 50)
    x2 = np.linspace(0, agent.L2, 50)

    # Plot in a 1x3 matplotlib figure as heatmap colors
    import matplotlib.pyplot as plt
    from matplotlib import cm

    Z_original = np.zeros((len(x1), len(x2)))
    Z_reconstructed = np.zeros((len(x1), len(x2)))
    Z_third = np.zeros((len(x1), len(x2)))

    for i in range(len(x1)):
        for j in range(len(x2)):
            Z_original[j, i] = phi_original([x1[i], x2[j]])
            Z_reconstructed[j, i] = phi_rec_from_ck([x1[i], x2[j]])
            Z_third[j, i] = phi_rec_from_agent([x1[i], x2[j]])
        
    fig = plt.figure(figsize=(18, 6))
    
    ax1 = fig.add_subplot(131)
    im1 = ax1.imshow(Z_original, extent=(0, agent.L1, 0, agent.L2), origin='lower', cmap=cm.viridis)
    ax1.set_title('Original Function Φ')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_aspect('auto')
    plt.colorbar(im1, ax=ax1, label='Function Value')

    ax2 = fig.add_subplot(132)
    im2 = ax2.imshow(Z_third,
                     extent=(0, agent.L1, 0, agent.L2), origin='lower', cmap=cm.viridis)
    ax2.set_title(f'Fourier Reconstruction (Kmax = {agent.Kmax})')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_aspect('auto')
    plt.colorbar(im2, ax=ax2, label='Function Value')

    ax3 = fig.add_subplot(133)
    im3 = ax3.imshow(Z_reconstructed, extent=(0, agent.L1, 0, agent.L2), origin='lower', cmap=cm.viridis)
    ax3.set_title('Reconstructed from Ck')
    ax3.set_xlabel('x1')
    ax3.set_ylabel('x2')
    ax3.set_aspect('auto')
    plt.colorbar(im3, ax=ax3, label='Function Value')

    if all_traj is not None:
        all_traj = np.array(all_traj)
        ax3.plot(all_traj[:, 0], all_traj[:, 1], 'k-', label='Trajectory')
        # plot also the buffer
        buffer_traj = agent.erg_c.past_states_buffer.get()
        ax3.plot(buffer_traj[:, 0], buffer_traj[:, 1], 'r-', label='Buffer Trajectory')
        # if phi_3 is not None:
        #     ax2.plot(x_traj[:, 1], x_traj[:, 0], 'r-', label='Trajectory')

    plt.tight_layout()


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
    ax1.plot(time_list, input_list[:, 0], label='u1 - Thrust')
    ax1.plot(time_list, input_list[:, 1], label='u2 - Yaw Torque')
    ax1.plot(time_list, input_list[:, 2], label='u3 - Pitch Torque')
    ax1.plot(time_list, input_list[:, 3], label='u4 - Roll Torque')
    ax1.set_title('Control Inputs')
    ax1.legend()
    ax1.grid()

    # Motor Commands
    ax2.plot(time_list, conv_inp_list[:, 0], label='Motor 1')
    ax2.plot(time_list, conv_inp_list[:, 1], label='Motor 2')
    ax2.plot(time_list, conv_inp_list[:, 2], label='Motor 3')
    ax2.plot(time_list, conv_inp_list[:, 3], label='Motor 4')
    ax2.set_title('Motor Commands')
    ax2.legend()
    ax2.grid()

    plt.tight_layout()
    plt.show()