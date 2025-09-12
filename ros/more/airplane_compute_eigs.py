"""
test_compute_eigs.py

Example: import FixedWing12DOFTrainer from your model_dynamics.py,
compute a trim for a desired airspeed, compute eigenvalues for stability analysis,
simulate, and plot results.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm

# import the class from your model_dynamics module
from src.ergodic_exploration.my_erg_lib.model_dynamics import FixedWing12DOFTrainer

def compute_eigenvalues(plane, x_trim, u_trim):
    """
    Compute eigenvalues of the linearized system around trim condition.
    Returns eigenvalues, eigenvectors, and analysis of stability.
    """
    # Get the linearized system matrices at trim
    A = plane.f_x(x_trim, u_trim)
    B = plane.f_u(x_trim)
    
    # Compute eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eig(A)
    
    print("=== EIGENVALUE ANALYSIS ===")
    print(f"System dimension: {len(eigenvals)}")
    print("\nEigenvalues (real + imaginary):")
    
    # Sort eigenvalues by real part (most unstable first)
    sorted_indices = np.argsort(eigenvals.real)[::-1]
    eigenvals_sorted = eigenvals[sorted_indices]
    eigenvecs_sorted = eigenvecs[:, sorted_indices]
    
    stable_count = 0
    unstable_count = 0
    marginally_stable_count = 0
    
    for i, eig in enumerate(eigenvals_sorted):
        real_part = eig.real
        imag_part = eig.imag
        
        # Classify stability
        if real_part < -1e-6:
            stability = "STABLE"
            stable_count += 1
        elif real_part > 1e-6:
            stability = "UNSTABLE"
            unstable_count += 1
        else:
            stability = "MARGINALLY STABLE"
            marginally_stable_count += 1
        
        # Calculate frequency and damping for complex eigenvalues
        if abs(imag_part) > 1e-6:
            freq_hz = abs(imag_part) / (2 * np.pi)
            damping_ratio = -real_part / abs(eig)
            print(f"  λ_{i+1}: {real_part:8.4f} ± {abs(imag_part):8.4f}j  |  "
                  f"f={freq_hz:6.3f}Hz, ζ={damping_ratio:6.3f}  |  {stability}")
        else:
            time_constant = -1/real_part if abs(real_part) > 1e-10 else np.inf
            print(f"  λ_{i+1}: {real_part:8.4f}                    |  "
                  f"τ={time_constant:6.2f}s               |  {stability}")
    
    print(f"\nStability Summary:")
    print(f"  Stable modes: {stable_count}")
    print(f"  Unstable modes: {unstable_count}")
    print(f"  Marginally stable modes: {marginally_stable_count}")
    
    # Overall system stability
    if unstable_count > 0:
        print(f"  OVERALL: SYSTEM IS UNSTABLE ({unstable_count} unstable modes)")
    elif marginally_stable_count > 0:
        print(f"  OVERALL: SYSTEM IS MARGINALLY STABLE")
    else:
        print(f"  OVERALL: SYSTEM IS STABLE")
    
    # Analyze aircraft-specific modes
    analyze_aircraft_modes(eigenvals_sorted, eigenvecs_sorted, plane.state_names)
    
    return eigenvals_sorted, eigenvecs_sorted, A, B

def analyze_aircraft_modes(eigenvals, eigenvecs, state_names):
    """
    Analyze and identify typical aircraft dynamic modes based on eigenvalues and eigenvectors.
    """
    print(f"\n=== AIRCRAFT MODE ANALYSIS ===")
    
    # State indices for easier reference
    # x = [X, Y, Z, phi, theta, psi, u, v, w, p, q, r]
    state_indices = {name: i for i, name in enumerate(state_names)}
    
    for i, (eig, eigvec) in enumerate(zip(eigenvals, eigenvecs.T)):
        real_part = eig.real
        imag_part = eig.imag
        
        # Get the dominant state contributions
        eigvec_abs = np.abs(eigvec)
        dominant_indices = np.argsort(eigvec_abs)[-3:][::-1]  # Top 3 contributors
        
        print(f"\nMode {i+1}: λ = {real_part:.4f} ± {abs(imag_part):.4f}j")
        print(f"  Primary state contributions:")
        for idx in dominant_indices:
            if eigvec_abs[idx] > 0.1:  # Only show significant contributions
                print(f"    {state_names[idx]}: {eigvec_abs[idx]:.3f}")
        
        # Try to classify the mode based on dominant states
        mode_type = classify_aircraft_mode(dominant_indices, eigvec_abs, state_indices, eig)
        if mode_type:
            print(f"  Identified as: {mode_type}")

def classify_aircraft_mode(dominant_indices, eigvec_abs, state_indices, eigenval):
    """
    Attempt to classify aircraft dynamic modes based on dominant states.
    """
    # Get dominant state names
    state_names_list = list(state_indices.keys())
    dominant_states = [state_names_list[i] for i in dominant_indices if eigvec_abs[i] > 0.1]
    
    real_part = eigenval.real
    imag_part = eigenval.imag
    
    # Longitudinal modes
    if any(state in dominant_states for state in ['u', 'w', 'theta', 'q']):
        if abs(imag_part) > 1e-3:  # Oscillatory
            if abs(imag_part) > 1.0:  # High frequency
                return "SHORT PERIOD (longitudinal oscillation)"
            else:  # Low frequency
                return "PHUGOID (long-period longitudinal oscillation)"
        else:  # Non-oscillatory
            return "LONGITUDINAL (non-oscillatory)"
    
    # Lateral-directional modes
    elif any(state in dominant_states for state in ['v', 'phi', 'psi', 'p', 'r']):
        if abs(imag_part) > 1e-3:  # Oscillatory
            return "DUTCH ROLL (lateral-directional oscillation)"
        else:  # Non-oscillatory
            if 'phi' in dominant_states or 'p' in dominant_states:
                return "ROLL MODE (lateral)"
            elif 'psi' in dominant_states or 'r' in dominant_states:
                return "SPIRAL MODE (directional)"
            else:
                return "LATERAL-DIRECTIONAL (non-oscillatory)"
    
    # Position/navigation modes
    elif any(state in dominant_states for state in ['X', 'Y', 'Z']):
        return "TRANSLATIONAL (position dynamics)"
    
    return "UNCLASSIFIED"

def plot_eigenvalues(eigenvals, title="Eigenvalues in Complex Plane"):
    """
    Plot eigenvalues in the complex plane with stability regions.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot eigenvalues
    real_parts = eigenvals.real
    imag_parts = eigenvals.imag
    
    # Color code by stability
    colors = []
    for eig in eigenvals:
        if eig.real < -1e-6:
            colors.append('green')  # Stable
        elif eig.real > 1e-6:
            colors.append('red')    # Unstable
        else:
            colors.append('orange') # Marginally stable
    
    scatter = ax.scatter(real_parts, imag_parts, c=colors, s=100, alpha=0.7, edgecolors='black')
    
    # Add stability boundary (imaginary axis)
    y_range = max(abs(imag_parts.max()), abs(imag_parts.min()), 1.0)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Stability boundary')
    
    # Shade stable region (left half-plane)
    ax.axvspan(ax.get_xlim()[0], 0, alpha=0.1, color='green', label='Stable region')
    ax.axvspan(0, ax.get_xlim()[1], alpha=0.1, color='red', label='Unstable region')
    
    # Add grid and labels
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Real Part')
    ax.set_ylabel('Imaginary Part')
    ax.set_title(title)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', alpha=0.7, label='Stable eigenvalues'),
                      Patch(facecolor='red', alpha=0.7, label='Unstable eigenvalues'),
                      Patch(facecolor='orange', alpha=0.7, label='Marginally stable')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Annotate eigenvalues
    for i, (real, imag) in enumerate(zip(real_parts, imag_parts)):
        ax.annotate(f'λ{i+1}', (real, imag), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.show()

def simulate(plane, x0, u_const, T=10.0, dt=0.01):
    """
    Simulate using plane.step for T seconds with fixed input u_const.
    Returns time array and state history [Nsteps x nstates].
    """
    nsteps = int(np.ceil(T / dt))
    x_hist = np.zeros((nsteps + 1, plane.num_of_states))
    t = np.zeros(nsteps + 1)
    x = x0.copy()
    plane.state = x.copy()  # keep internal state consistent if class uses it
    x_hist[0, :] = x
    for k in range(nsteps):
        x = plane.step(x, u_const, dt=dt)
        # plane.step may or may not update plane.state internally; keep x synced
        plane.state = x.copy()
        x_hist[k+1, :] = x
        t[k+1] = t[k] + dt
    return t, x_hist

def plot_results(t, x_hist, u_trim):
    # extract all state variables
    X = x_hist[:, 0]
    Y = x_hist[:, 1]
    Z = x_hist[:, 2]
    phi = x_hist[:, 3]
    theta = x_hist[:, 4]
    psi = x_hist[:, 5]
    u = x_hist[:, 6]
    v = x_hist[:, 7]
    w = x_hist[:, 8]
    p = x_hist[:, 9]
    q = x_hist[:, 10]
    r = x_hist[:, 11]

    fig, axs = plt.subplots(4, 3, figsize=(15, 12), sharex=True)
    
    # Position states
    axs[0, 0].plot(t, X)
    axs[0, 0].set_ylabel("X (m)")
    axs[0, 0].grid(True)
    
    axs[0, 1].plot(t, Y)
    axs[0, 1].set_ylabel("Y (m)")
    axs[0, 1].grid(True)
    
    axs[0, 2].plot(t, Z)
    axs[0, 2].set_ylabel("Z (m)")
    axs[0, 2].grid(True)
    
    # Attitude angles
    axs[1, 0].plot(t, phi * 180.0/np.pi)
    axs[1, 0].set_ylabel("phi (deg)")
    axs[1, 0].grid(True)
    
    axs[1, 1].plot(t, theta * 180.0/np.pi)
    axs[1, 1].set_ylabel("theta (deg)")
    axs[1, 1].grid(True)
    
    axs[1, 2].plot(t, psi * 180.0/np.pi)
    axs[1, 2].set_ylabel("psi (deg)")
    axs[1, 2].grid(True)
    
    # Body velocities
    axs[2, 0].plot(t, u)
    axs[2, 0].set_ylabel("u (m/s)")
    axs[2, 0].grid(True)
    
    axs[2, 1].plot(t, v)
    axs[2, 1].set_ylabel("v (m/s)")
    axs[2, 1].grid(True)
    
    axs[2, 2].plot(t, w)
    axs[2, 2].set_ylabel("w (m/s)")
    axs[2, 2].grid(True)
    
    # Angular rates
    axs[3, 0].plot(t, p * 180.0/np.pi)
    axs[3, 0].set_ylabel("p (deg/s)")
    axs[3, 0].set_xlabel("time (s)")
    axs[3, 0].grid(True)
    
    axs[3, 1].plot(t, q * 180.0/np.pi)
    axs[3, 1].set_ylabel("q (deg/s)")
    axs[3, 1].set_xlabel("time (s)")
    axs[3, 1].grid(True)
    
    axs[3, 2].plot(t, r * 180.0/np.pi)
    axs[3, 2].set_ylabel("r (deg/s)")
    axs[3, 2].set_xlabel("time (s)")
    axs[3, 2].grid(True)

    plt.suptitle(f"Aircraft States - Trim: elevator={u_trim[0]*180/np.pi:.2f}°, throttle={u_trim[3]:.2f}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == "__main__":
    # create plane instance (adjust dt if you like)
    plane = FixedWing12DOFTrainer(dt=0.01)

    # desired trim speed (m/s)
    V_trim = 10

    # compute trim
    x_trim, u_trim, sol = plane.computeTrim(V_trim=V_trim)
    print("Trim solver success:", sol.success, sol.message)
    print("Trim state (partial): u, w, theta =",
          x_trim[6], x_trim[8], x_trim[4])
    print("Trim inputs (de, da, dr, throttle) =",
          u_trim[0], u_trim[1], u_trim[2], u_trim[3])
    
    # ===== EIGENVALUE ANALYSIS =====
    print("\n" + "="*60)
    print("PERFORMING EIGENVALUE ANALYSIS")
    print("="*60)
    
    eigenvals, eigenvecs, A, B = compute_eigenvalues(plane, x_trim, u_trim)
    
    # Plot eigenvalues
    plot_eigenvalues(eigenvals, f"Fixed-Wing Aircraft Eigenvalues (V_trim = {V_trim} m/s)")
    
    # Optional: Show the A matrix sparsity pattern
    plt.figure(figsize=(10, 8))
    plt.spy(A, markersize=3)
    plt.title("State Matrix A Sparsity Pattern")
    plt.xlabel("State Index")
    plt.ylabel("State Index") 
    # Add state labels
    state_labels = plane.state_names
    plt.xticks(range(len(state_labels)), state_labels, rotation=45)
    plt.yticks(range(len(state_labels)), state_labels)
    plt.tight_layout()
    plt.show()
    
    print(f"\nState vector ordering: {plane.state_names}")
    print(f"Input vector ordering: ['delta_e', 'delta_a', 'delta_r', 'throttle']")
    
    # ===== SIMULATION =====
    print("\n" + "="*60)
    print("PERFORMING SIMULATION")
    print("="*60)

    # short test: simulate from trimmed state and also apply a small elevator pulse at t=2s
    Tsim = 35.0
    dt = 0.01
    t, x_hist = simulate(plane, x_trim, u_trim, T=Tsim, dt=dt)

    # optional test perturbation: at t=2s apply elevator -4deg for 0.5s
    # We'll re-run simulation with perturbation for demonstration
    plane.state = x_trim.copy()
    x = x_trim.copy()
    nsteps = int(np.ceil(Tsim / dt))
    x_hist2 = np.zeros((nsteps + 1, plane.num_of_states))
    t2 = np.zeros(nsteps + 1)
    x_hist2[0] = x.copy()
    for k in tqdm(range(nsteps)):
        time = k * dt
        u_now = u_trim.copy()
        if 12.0 < time < 12.5:
            u_now[1] += -4.0 * np.pi/180.0  # elevator down pulse -4 deg
        x = plane.step(x, u_now, dt=dt)
        x_hist2[k+1] = x.copy()
        t2[k+1] = time + dt

    # plot results (perturbed run)
    plot_results(t2, x_hist2, u_trim)
