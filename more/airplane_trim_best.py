"""
test_fixedwing_trim_sim.py

Example: import FixedWing12DOFTrainer from your model_dynamics.py,
compute a trim for a desired airspeed, simulate, and plot results.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
import sys

# import the class from your model_dynamics module
from src.ergodic_exploration.my_erg_lib.model_dynamics import FixedWing12DOFTrainer

def trim_objective(vars, plane, V_trim):
    """
    vars: [w, theta, delta_e, throttle]
      w: body z-velocity
      theta: pitch angle (rad)
      delta_e: elevator (rad)
      throttle: 0..1
    plane: instance of FixedWing12DOFTrainer
    V_trim: desired airspeed (m/s) used for u (body-x)
    returns residuals [udot, wdot, qdot, u - V_trim]
    Note: class state ordering is:
      x = [X, Y, Z, phi, theta, psi, u, v, w, p, q, r]
    """
    w, theta, delta_e, throttle = vars
    # build state with symmetric (no lateral motion), no angular rates
    X = 0.0; Y = 0.0; Z = -0.0  # choose Z reference (your convention)
    phi = 0.0
    psi = 0.0
    u = V_trim
    v = 0.0
    p = 0.0; q = 0.0; r = 0.0

    x = np.array([X, Y, Z, phi, theta, psi, u, v, w, p, q, r], dtype=float)
    u_ctrl = np.array([delta_e, 0.0, 0.0, throttle])  # symmetric (ail/rud zero)

    # evaluate dynamics
    xdot = plane.f(x, u_ctrl)

    # residuals: udot = 0, wdot = 0, qdot = 0 (pitch accel), and u - V_trim = 0
    # xdot ordering in this implementation:
    # xdot[0:3] = pos_dot (Xdot,Ydot,Zdot)
    # xdot[3:6] = [phi_dot, theta_dot, psi_dot]
    # xdot[6:9] = [udot, vdot, wdot]
    # xdot[9:12] = [pdot, qdot, rdot]
    udot = xdot[6]
    wdot = xdot[8]
    qdot = xdot[10]
    # last residual enforces body-x speed equals V_trim (u - V_trim = 0)
    res = np.array([udot, wdot, qdot, u - V_trim])
    return res

def compute_trim(plane, V_trim=10.0):
    """
    Compute a symmetric trim for desired airspeed V_trim (m/s).
    Returns x_trim (state) and u_trim (controls).
    """
    # initial guess: small w, small pitch, small elevator, half throttle
    guess = np.array([0.0, 0.05, 0.0, 0.5])  # [w, theta, delta_e, throttle]
    sol = root(trim_objective, guess, args=(plane, V_trim), method='hybr', tol=1e-8)

    if not sol.success:
        print("Trim solver did not converge:", sol.message)
        # still return a best-effort guess
    w, theta, delta_e, throttle = sol.x
    u = V_trim
    x_trim = np.array([0.0, 0.0, 0.0,   # X, Y, Z
                       0.0, theta, 0.0,  # phi, theta, psi
                       u, 0.0, w,         # u, v, w
                       0.0, 0.0, 0.0])    # p, q, r
    u_trim = np.array([delta_e, 0.0, 0.0, np.clip(throttle, 0.0, 1.0)])
    return x_trim, u_trim, sol

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
    # choose variables to plot: airspeed (u), pitch (theta), elevator, altitude (Z), pitch rate q
    u = x_hist[:, 6]
    theta = x_hist[:, 4]
    Z = x_hist[:, 2]
    q = x_hist[:, 10]

    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    axs[0].plot(t, u)
    axs[0].set_ylabel("u (m/s)")
    axs[0].grid(True)

    axs[1].plot(t, theta * 180.0/np.pi)
    axs[1].set_ylabel("theta (deg)")
    axs[1].grid(True)

    axs[2].plot(t, Z)
    axs[2].set_ylabel("Z (m)")
    axs[2].set_xlabel("time (s)")
    axs[2].grid(True)

    plt.suptitle(f"Trim elevator={u_trim[0]*180/np.pi:.2f} deg, throttle={u_trim[3]:.2f}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def find_optimal_v_trim(plane, target_w=0.0, v_trim_start=19.0, search_range=2.0, num_points=50):
    """
    Search for V_trim around v_trim_start that results in x_trim[8] (w) closest to target_w.
    Returns the optimal V_trim and corresponding trim state.
    """
    v_trim_values = np.linspace(v_trim_start - search_range, v_trim_start + search_range, num_points)
    w_values = []
    trim_results = []
    
    print(f"Searching for V_trim around {v_trim_start} m/s that results in w ≈ {target_w}")
    print("V_trim (m/s) | w (x_trim[8]) | Pitch θ (deg) | Converged")
    print("-" * 55)
    
    for V_trim in v_trim_values:
        try:
            x_trim, u_trim, sol = compute_trim(plane, V_trim=V_trim)
            w = x_trim[8]
            theta_deg = x_trim[4] * 180.0 / np.pi  # Convert pitch angle to degrees
            w_values.append(w)
            trim_results.append((V_trim, x_trim, u_trim, sol))
            converged = "Yes" if sol.success else "No"
            print(f"{V_trim:8.3f}     | {w:9.6f}   | {theta_deg:9.3f}    | {converged}")
        except Exception as e:
            w_values.append(np.inf)
            trim_results.append((V_trim, None, None, None))
            print(f"{V_trim:8.3f}     | Error     | Error      | No")
    
    # Find the V_trim that gives w closest to target_w
    w_values = np.array(w_values)
    valid_indices = ~np.isinf(w_values)
    
    if not np.any(valid_indices):
        print("No valid solutions found!")
        return None, None, None, None
    
    valid_w = w_values[valid_indices]
    valid_indices_list = np.where(valid_indices)[0]
    
    # Find minimum absolute difference from target_w
    abs_diff = np.abs(valid_w - target_w)
    best_idx = valid_indices_list[np.argmin(abs_diff)]
    
    best_v_trim, best_x_trim, best_u_trim, best_sol = trim_results[best_idx]
    best_w = w_values[best_idx]
    
    print(f"\nOptimal result:")
    print(f"V_trim = {best_v_trim:.6f} m/s")
    print(f"w (x_trim[8]) = {best_w:.8f} m/s")
    print(f"Pitch angle θ = {best_x_trim[4]*180.0/np.pi:.6f}°")
    print(f"Difference from target: {abs(best_w - target_w):.8f}")
    
    return best_v_trim, best_x_trim, best_u_trim, best_sol

if __name__ == "__main__":
    # create plane instance (adjust dt if you like)
    plane = FixedWing12DOFTrainer(dt=0.01)

    # Search for V_trim around 19.0 m/s that results in zero w (x_trim[8])
    optimal_v_trim, x_trim, u_trim, sol = find_optimal_v_trim(
        plane, 
        target_w=0.0, 
        v_trim_start=18.714, 
        search_range=0.1,  # Refined search around the optimal point
        num_points=50
    )
    
    if optimal_v_trim is not None:
        print(f"\nFinal verification:")
        print(f"Using V_trim = {optimal_v_trim:.6f} m/s")
        print(f"x_trim[8] (w) = {x_trim[8]:.8f} m/s")
        
        # Show full trim state and controls
        print(f"\nComplete trim state:")
        print(f"Position: X={x_trim[0]:.3f}, Y={x_trim[1]:.3f}, Z={x_trim[2]:.3f}")
        print(f"Attitude: φ={x_trim[3]*180/np.pi:.3f}°, θ={x_trim[4]*180/np.pi:.3f}°, ψ={x_trim[5]*180/np.pi:.3f}°")
        print(f"Velocity: u={x_trim[6]:.3f}, v={x_trim[7]:.3f}, w={x_trim[8]:.6f}")
        print(f"Rates: p={x_trim[9]:.3f}, q={x_trim[10]:.3f}, r={x_trim[11]:.3f}")
        print(f"Controls: δe={u_trim[0]*180/np.pi:.3f}°, δa={u_trim[1]*180/np.pi:.3f}°, δr={u_trim[2]*180/np.pi:.3f}°, throttle={u_trim[3]:.3f}")
    
    sys.exit(0)

