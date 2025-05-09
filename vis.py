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