import numpy as np
import matplotlib.pyplot as plt
from my_erg_lib.basis import Basis, ReconstructedPhiFromCk, ReconstructedPhi

def generate_dummy_trajectory(L1, L2, trajectory_type='circular', num_points=1000, T=10.0):
    """
    Generate different types of dummy trajectories for testing
    
    Args:
        L1, L2: Domain dimensions
        trajectory_type: 'circular', 'figure8', 'random_walk', 'straight_line', 'spiral'
        num_points: Number of trajectory points
        T: Total time duration
    
    Returns:
        x_traj: numpy array of shape (num_points, 2) containing (x, y) positions
        t_points: numpy array of time points
    """
    t_points = np.linspace(0, T, num_points)
    dt = T / num_points
    
    if trajectory_type == 'circular':
        # Circular trajectory in the center of domain
        center_x, center_y = L1/2, L2/2
        radius = min(L1, L2) * 0.3
        x_traj = np.zeros((num_points, 2))
        x_traj[:, 0] = center_x + radius * np.cos(2 * np.pi * t_points / T)
        x_traj[:, 1] = center_y + radius * np.sin(2 * np.pi * t_points / T)
        
    elif trajectory_type == 'figure8':
        # Figure-8 trajectory
        center_x, center_y = L1/2, L2/2
        scale_x, scale_y = L1 * 0.3, L2 * 0.2
        x_traj = np.zeros((num_points, 2))
        x_traj[:, 0] = center_x + scale_x * np.sin(2 * np.pi * t_points / T)
        x_traj[:, 1] = center_y + scale_y * np.sin(4 * np.pi * t_points / T)
        
    elif trajectory_type == 'random_walk':
        # Random walk with bounds checking
        x_traj = np.zeros((num_points, 2))
        x_traj[0] = [L1/2, L2/2]  # Start at center
        
        for i in range(1, num_points):
            # Random step
            step = np.random.normal(0, 0.1, 2)
            next_pos = x_traj[i-1] + step
            
            # Keep within bounds
            next_pos[0] = np.clip(next_pos[0], 0.1, L1-0.1)
            next_pos[1] = np.clip(next_pos[1], 0.1, L2-0.1)
            x_traj[i] = next_pos
            
    elif trajectory_type == 'straight_line':
        # Diagonal straight line
        x_traj = np.zeros((num_points, 2))
        x_traj[:, 0] = np.linspace(0.1, L1-0.1, num_points)
        x_traj[:, 1] = np.linspace(0.1, L2-0.1, num_points)
        
    elif trajectory_type == 'spiral':
        # Spiral trajectory
        center_x, center_y = L1/2, L2/2
        max_radius = min(L1, L2) * 0.4
        x_traj = np.zeros((num_points, 2))
        for i, t in enumerate(t_points):
            r = max_radius * (1 - t/T)  # Decreasing radius
            theta = 4 * np.pi * t / T   # Multiple rotations
            x_traj[i, 0] = center_x + r * np.cos(theta)
            x_traj[i, 1] = center_y + r * np.sin(theta)
    
    else:
        raise ValueError(f"Unknown trajectory type: {trajectory_type}")
    
    return x_traj, t_points

def create_test_phi_function(L1, L2, phi_type='gaussian_bumps'):
    """
    Create test target distribution functions
    """
    if phi_type == 'gaussian_bumps':
        def phi_func(s):
            x, y = s[0], s[1]
            # Multiple Gaussian bumps
            bump1 = 3 * np.exp(-10 * ((x - L1*0.3)**2 + (y - L2*0.7)**2))
            bump2 = 2 * np.exp(-8 * ((x - L1*0.7)**2 + (y - L2*0.3)**2))
            bump3 = 1.5 * np.exp(-15 * ((x - L1*0.1)**2 + (y - L2*0.1)**2))
            return bump1 + bump2 + bump3 + 0.1  # Small constant to avoid zeros
    
    elif phi_type == 'uniform':
        def phi_func(s):
            return 1.0
    
    elif phi_type == 'sinusoidal':
        def phi_func(s):
            x, y = s[0], s[1]
            return 1 + 0.5 * np.sin(2*np.pi*x/L1) * np.cos(2*np.pi*y/L2)
    
    else:
        raise ValueError(f"Unknown phi type: {phi_type}")
    
    return phi_func

def plot_phi_comparison(basis, ck_coeffs, original_phi=None, L1=1.0, L2=1.0, grid_res=50, title_suffix=""):
    """
    Plot heatmaps comparing original phi and reconstructed phi from ck coefficients
    """
    # Create grid for evaluation
    x = np.linspace(0, L1, grid_res)
    y = np.linspace(0, L2, grid_res)
    X, Y = np.meshgrid(x, y)
    
    # Reconstruct phi from ck coefficients
    phi_reconstructed = ReconstructedPhiFromCk(basis, ck_coeffs)
    
    # Evaluate reconstructed phi on grid
    Z_reconstructed = np.zeros_like(X)
    for i in range(grid_res):
        for j in range(grid_res):
            Z_reconstructed[j, i] = phi_reconstructed([X[j, i], Y[j, i]])
    
    # Create plots
    if original_phi is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Original phi
        Z_original = np.zeros_like(X)
        for i in range(grid_res):
            for j in range(grid_res):
                Z_original[j, i] = original_phi([X[j, i], Y[j, i]])
        
        im1 = axes[0].contourf(X, Y, Z_original, levels=20, cmap='viridis')
        axes[0].set_title(f'Original φ(x) {title_suffix}')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        plt.colorbar(im1, ax=axes[0])
        
        # Reconstructed phi
        im2 = axes[1].contourf(X, Y, Z_reconstructed, levels=20, cmap='viridis')
        axes[1].set_title(f'Reconstructed φ(x) from Ck {title_suffix}')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        plt.colorbar(im2, ax=axes[1])
        
        # Difference
        Z_diff = Z_original - Z_reconstructed
        im3 = axes[2].contourf(X, Y, Z_diff, levels=20, cmap='RdBu_r')
        axes[2].set_title(f'Difference (Original - Reconstructed) {title_suffix}')
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('y')
        plt.colorbar(im3, ax=axes[2])
        
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        im = ax.contourf(X, Y, Z_reconstructed, levels=20, cmap='viridis')
        ax.set_title(f'Reconstructed φ(x) from Ck {title_suffix}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    return fig

def plot_trajectory_and_coefficients(x_traj, t_points, ck_coeffs, L1, L2, title_suffix=""):
    """
    Plot the trajectory and ck coefficients
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot trajectory
    axes[0].plot(x_traj[:, 0], x_traj[:, 1], 'b-', linewidth=1, alpha=0.7)
    axes[0].plot(x_traj[0, 0], x_traj[0, 1], 'go', markersize=8, label='Start')
    axes[0].plot(x_traj[-1, 0], x_traj[-1, 1], 'ro', markersize=8, label='End')
    axes[0].set_xlim(0, L1)
    axes[0].set_ylim(0, L2)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title(f'Trajectory {title_suffix}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot ck coefficients as heatmap
    im = axes[1].imshow(ck_coeffs, origin='lower', cmap='RdBu_r', aspect='equal')
    axes[1].set_title(f'Ck Coefficients {title_suffix}')
    axes[1].set_xlabel('k2')
    axes[1].set_ylabel('k1')
    plt.colorbar(im, ax=axes[1])
    
    # Add text annotations for coefficient values
    for i in range(ck_coeffs.shape[0]):
        for j in range(ck_coeffs.shape[1]):
            axes[1].text(j, i, f'{ck_coeffs[i,j]:.3f}', 
                        ha='center', va='center', fontsize=8)
    
    plt.tight_layout()
    return fig

def test_ck_calculation(L1=5.0, L2=5.0, Kmax=5, trajectory_type='circular', 
                       phi_type='gaussian_bumps', num_points=1000, T=10.0,
                       use_infinite_buffer=False):
    """
    Main test function to verify ck calculation and phi reconstruction
    """
    print(f"Testing ck calculation with:")
    print(f"  Domain: [{L1} x {L2}]")
    print(f"  Kmax: {Kmax}")
    print(f"  Trajectory: {trajectory_type}")
    print(f"  Phi type: {phi_type}")
    print(f"  Points: {num_points}")
    print(f"  Duration: {T}s")
    print(f"  Infinite buffer: {use_infinite_buffer}")
    print("-" * 50)
    
    # Create target distribution
    phi_func = create_test_phi_function(L1, L2, phi_type)
    
    # Create basis object
    basis = Basis(L1=L1, L2=L2, Kmax=Kmax, phi_=phi_func, 
                  precalc_hk_coeff=True, precalc_phik_coeff=True)
    
    # Generate dummy trajectory
    x_traj, t_points = generate_dummy_trajectory(L1, L2, trajectory_type, num_points, T)
    
    print(f"Generated trajectory with {len(x_traj)} points")
    print(f"Trajectory bounds: x=[{x_traj[:,0].min():.3f}, {x_traj[:,0].max():.3f}], "
          f"y=[{x_traj[:,1].min():.3f}, {x_traj[:,1].max():.3f}]")
    
    # Calculate ck coefficients from trajectory
    if use_infinite_buffer:
        # For infinite buffer case, we need to simulate the recursive calculation
        # Here we'll just use the regular calculation for simplicity
        ck_coeffs = basis.calcCkCoeff(x_traj, ti=0, T=T)
    else:
        ck_coeffs = basis.calcCkCoeff(x_traj, ti=0, T=T)
    
    print(f"Calculated Ck coefficients shape: {ck_coeffs.shape}")
    print(f"Ck coefficient range: [{ck_coeffs.min():.6f}, {ck_coeffs.max():.6f}]")
    print(f"Sum of |Ck|: {np.sum(np.abs(ck_coeffs)):.6f}")
    
    # Print some coefficient values
    print("\nFirst few Ck coefficients:")
    for k1 in range(min(4, Kmax+1)):
        for k2 in range(min(4, Kmax+1)):
            print(f"C[{k1},{k2}] = {ck_coeffs[k1,k2]:.6f}")
    
    # Create reconstructed phi functions
    phi_from_ck = ReconstructedPhiFromCk(basis, ck_coeffs)
    phi_from_basis = ReconstructedPhi(basis, precalc_phik=False)
    
    # Create a simple agent-like object for vis.plotPhi
    class SimpleAgent:
        def __init__(self, L1, L2, Kmax, basis):
            self.L1 = L1
            self.L2 = L2
            self.Kmax = Kmax
            self.basis = basis
            
        class SimpleModel:
            def __init__(self, x_traj):
                self.state = x_traj[-1]  # Use last point as current state
                
        class SimpleErgC:
            def __init__(self, x_traj):
                class SimpleBuffer:
                    def __init__(self, x_traj):
                        self.trajectory = x_traj
                    def get(self):
                        return self.trajectory
                        
                self.past_states_buffer = SimpleBuffer(x_traj)
    
    # Create simple agent for visualization
    simple_agent = SimpleAgent(L1, L2, Kmax, basis)
    simple_agent.model = SimpleAgent.SimpleModel(x_traj)
    simple_agent.erg_c = SimpleAgent.SimpleErgC(x_traj)
    
    # Use vis.plotPhi to visualize the results
    import vis
    vis.plotPhi(simple_agent, phi_rec_from_ck=phi_from_ck, 
                phi_rec_from_agent=phi_from_basis, all_traj=x_traj)
    
    # Visualize the coefficients using vis.visualiseCoefficients
    vis.visualiseCoefficients(simple_agent, ck_coeffs)
    
    # Calculate some metrics for verification
    x = np.linspace(0, L1, 50)
    y = np.linspace(0, L2, 50)
    X, Y = np.meshgrid(x, y)
    
    Z_from_ck = np.zeros_like(X)
    Z_from_basis = np.zeros_like(X)
    
    for i in range(50):
        for j in range(50):
            Z_from_ck[j, i] = phi_from_ck([X[j, i], Y[j, i]])
            Z_from_basis[j, i] = phi_from_basis([X[j, i], Y[j, i]])
    
    mse = np.mean((Z_from_ck - Z_from_basis)**2)
    max_diff = np.max(np.abs(Z_from_ck - Z_from_basis))
    print(f"\nComparison metrics (Ck vs Basis coefficients):")
    print(f"  MSE: {mse:.6f}")
    print(f"  Max absolute difference: {max_diff:.6f}")
    
    plt.show()
    
    return basis, ck_coeffs, x_traj, t_points

if __name__ == "__main__":
    # Test different scenarios
    print("="*60)
    print("TESTING CK COEFFICIENT CALCULATION")
    print("="*60)
    
    # Test 1: Circular trajectory with Gaussian bumps
    # print("\n" + "="*30 + " TEST 1 " + "="*30)
    # test_ck_calculation(L1=5.0, L2=5.0, Kmax=10, trajectory_type='circular', 
    #                    phi_type='gaussian_bumps', num_points=500, T=5.0)
    
    # # Test 2: Figure-8 trajectory with uniform distribution
    # print("\n" + "="*30 + " TEST 2 " + "="*30)
    # test_ck_calculation(L1=3.0, L2=3.0, Kmax=10, trajectory_type='figure8', 
    #                    phi_type='uniform', num_points=800, T=8.0)
    
    # # Test 3: Random walk with sinusoidal distribution
    # print("\n" + "="*30 + " TEST 3 " + "="*30)
    # test_ck_calculation(L1=4.0, L2=4.0, Kmax=10, trajectory_type='random_walk', 
    #                    phi_type='sinusoidal', num_points=1000, T=10.0)

    print("\n" + "="*30 + f" TEST 4 " + "="*30)
    test_ck_calculation(L1=4.0, L2=4.0, Kmax=10, trajectory_type='random_walk',
                        phi_type='sinusoidal', num_points=500, T=10.0, use_infinite_buffer=True)

    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)
