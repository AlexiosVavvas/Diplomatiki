"""
test_fixedwing_trim_sim.py

Example: import FixedWing12DOFTrainer from your model_dynamics.py,
compute a trim for a desired airspeed, simulate, and plot results.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm

# import the class from your model_dynamics module
from src.ergodic_exploration.my_erg_lib.model_dynamics import FixedWing12DOFTrainer

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
    V_trim = 7.5

    # compute trim
    x_trim, u_trim, sol = plane.computeTrim(V_trim=V_trim)
    print("Trim solver success:", sol.success, sol.message)
    print("Trim state (partial): u, w, theta =",
          x_trim[6], x_trim[8], x_trim[4])
    print("Trim inputs (de, da, dr, throttle) =",
          u_trim[0], u_trim[1], u_trim[2], u_trim[3])
    

    # short test: simulate from trimmed state and also apply a small elevator pulse at t=2s
    Tsim = 30.0
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
        # if 2.0 < time < 2.5:
        #     u_now[0] += -4.0 * np.pi/180.0  # elevator down pulse -4 deg
        x = plane.step(x, u_now, dt=dt)
        x_hist2[k+1] = x.copy()
        t2[k+1] = time + dt

    # plot results (perturbed run)
    plot_results(t2, x_hist2, u_trim)
