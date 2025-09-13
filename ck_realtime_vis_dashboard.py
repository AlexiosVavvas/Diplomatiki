import numpy as np
import time
from my_erg_lib_old import basis
import matplotlib.pyplot as plt
import vis
import threading
from matplotlib.animation import FuncAnimation
import os
import shutil

Kmax = 4
L1 = 10 
L2 = 10

def phiExample(s, L1=1.0, L2=1.0):

    # Complex function with multiple peaks, valleys, and non-linearities
    x, y = s[0], s[1]
    
    # Multiple Gaussian bumps
    # Generate random bump positions within the L1, L2 boundaries
    bump_positions = [
        (0.3 * L1, 0.8 * L2),
        (0.7 * L1, 0.2 * L2),
        (0.15 * L1, 0.4 * L2),
        (0.85 * L1, 0.6 * L2)
    ]
    bump_heights = [5, 4, 3, 4.5]
    bump_widths = [0.7, 0.7, 15.2, 6.3]
    
    bumps = 0
    for i in range(len(bump_positions)):
        pos_x, pos_y = bump_positions[i]
        height = bump_heights[i]
        width = bump_widths[i]
        bumps += height * np.exp(-width * ((x-pos_x)**2 + (y-pos_y)**2))
    
    # Sinusoidal variations
    # waves = 2 * np.sin(8 * np.pi * x) * np.cos(6 * np.pi * y)
    
    # Polynomial trend
    # trend = (x - 0.4)**2 * (y - 0.6)**2 * 5
    
    # Sharp ridge
    # ridge = 3 * np.exp(-100 * (x - y)**2)
    
    # Combine all components
    # return 0.3 #+ waves + trend + ridge
    return bumps + 0.01 #+ waves + trend + ridge

# Function to be used for phi with specific L1 and L2 values
def phi_func(s):
    return phiExample(s, L1=10.0, L2=10.0)/43.8855 * 4
    # return phiExample(s, L1=10.0, L2=10.0)/72.8855 * 4

def visualiseCoefficients(ck):
    import matplotlib.pyplot as plt
    from matplotlib import cm

    k1 = np.linspace(0, Kmax, Kmax+1)
    k2 = np.linspace(0, Kmax, Kmax+1)
    K1, K2 = np.meshgrid(k1, k2)
    Z_ck = np.zeros((len(k1), len(k2)))
    Z_phik = np.zeros((len(k1), len(k2)))
    
    for i in range(len(k1)):
        for j in range(len(k2)):
            Z_ck[i, j] = ck[i, j]
            Z_phik[i, j] = base.calcPhikCoeff(int(k1[i]), int(k2[j]))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot Ck coefficients
    im1 = ax1.imshow(Z_ck, cmap=cm.viridis, origin='lower', 
                    extent=[0, Kmax, 0, Kmax], aspect='equal')
    ax1.set_title('Ck Coefficients')
    ax1.set_xlabel('k1')
    ax1.set_ylabel('k2')
    fig.colorbar(im1, ax=ax1, label='Ck Value')
    
    # Plot Phi_k coefficients
    im2 = ax2.imshow(Z_phik, cmap=cm.viridis, origin='lower', 
                    extent=[0, Kmax, 0, Kmax], aspect='equal')
    ax2.set_title('Phi_k Coefficients')
    ax2.set_xlabel('k1')
    ax2.set_ylabel('k2')
    fig.colorbar(im2, ax=ax2, label='Phi_k Value')
    
    plt.tight_layout()
    plt.show()

def plotPhi(phi_rec_from_ck, phi_rec_from_agent, all_traj=None, grid_res=50, clip_to_min_max=False):
    phi_original = base.phi

    x1 = np.linspace(0, L1, grid_res)
    x2 = np.linspace(0, L2, grid_res)

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
    
    # # Calculate the integrals using 2D trapezoidal rule
    # integral_original = np.trapz(np.trapz(Z_original, x2, axis=0), x1)
    # integral_agent_fourier = np.trapz(np.trapz(Z_agent_fourier_rec, x2, axis=0), x1)
    # integral_rec_from_ck = np.trapz(np.trapz(Z_rec_from_ck, x2, axis=0), x1)

    # print(f"Integral of original function: {integral_original:.6f}")
    # print(f"Integral of agent fourier reconstruction: {integral_agent_fourier:.6f}")
    # print(f"Integral of reconstruction from ck: {integral_rec_from_ck:.6f}")
        
    fig = plt.figure(figsize=(18, 6))
    
    ax1 = fig.add_subplot(131)
    im1 = ax1.imshow(Z_original, extent=(0, L1, 0, L2), origin='lower', cmap=cm.viridis)
    ax1.set_title('Original Function Φ')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_aspect('auto')
    plt.colorbar(im1, ax=ax1, label='Function Value')

    ax2 = fig.add_subplot(132)
    im2 = ax2.imshow(Z_agent_fourier_rec,
                     extent=(0, L1, 0, L2), origin='lower', cmap=cm.viridis)
    ax2.set_title(f'Fourier Reconstruction (Kmax = {Kmax})')
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
        im3 = ax3.imshow(Z_rec_from_ck, extent=(0, L1, 0, L2), 
                        origin='lower', cmap=cm.viridis, vmin=min_val, vmax=max_val)
    else:
        im3 = ax3.imshow(Z_rec_from_ck, extent=(0, L1, 0, L2), 
                        origin='lower', cmap=cm.viridis)
    ax3.set_title('Reconstructed from Ck')
    ax3.set_xlabel('x1')
    ax3.set_ylabel('x2')
    ax3.set_aspect('auto')
    # x and y lims to 0 -> L1 and 0 -> L2
    ax3.set_xlim(0, L1)
    ax3.set_ylim(0, L2)
    plt.colorbar(im3, ax=ax3, label='Function Value')
    
    if all_traj is not None:
        all_traj = np.array(all_traj)
        ax3.plot(all_traj[:, 0], all_traj[:, 1], 'k-', label='Trajectory')

    plt.tight_layout()

def calcCkCoeff(x_traj, ti, T, x_buffer=None, do_not_divide_integral_flag=False):
        '''
        Calculate the coefficients Ck for the trajectory x_traj from time ti to T.
            x_traj: Ergodic states trajectory only (x1, x2)
            ti:     Current Initial Time
            T:      Duration forward
        '''
        ck = np.zeros((Kmax+1, Kmax+1))
        
        # Append to the trajectory the buffer points at the beginning with the traj continueing from the last buffer poit
        if x_buffer is not None:
            x_traj = np.concatenate((x_buffer, x_traj), axis=0)
            # Lets calculate simulation time step (dt) assuming uniform time spacing
            dt = (T - ti) / len(x_traj)
            # How much time in the back do we go with the buffer?
            delta_t = len(x_buffer) * dt  # Ergodic memory time
        else:
            # If we dont play with a buffer, we dont need ergodic memory
            delta_t = 0            
        
        # Calculate time step (dt) assuming uniform time spacing
        n_points = len(x_traj)
        
        # Time points corresponding to trajectory points
        t_points = np.linspace(ti-delta_t, ti+T, n_points)
        
        for k1 in range(Kmax+1):
            for k2 in range(Kmax+1):
                hk = base.calcHk(k1, k2)

                # Vectorized Fk calculation
                cos_k1 = np.cos(k1*np.pi/L1*x_traj[:, 0])
                cos_k2 = np.cos(k2*np.pi/L2*x_traj[:, 1])
                
                # Evaluate Fk at each trajectory point
                fk_values = cos_k1 * cos_k2 / hk
                
                # Perform trapezoidal integration
                if do_not_divide_integral_flag:
                    ck[k1, k2] = np.trapz(fk_values, x=t_points)
                else:
                    ck[k1, k2] = np.trapz(fk_values, x=t_points) / (delta_t + T)

        
        return ck

def calcErgodicCost(ck):
        ergodic_cost = 0.0
        for k1 in range(Kmax+1):
            for k2 in range(Kmax+1):
                ergodic_cost += base.LamdaK_cache[(k1, k2)] * (ck[k1, k2] - base.calcPhikCoeff(k1, k2))**2
        ergodic_cost *= 30
        return ergodic_cost

base = basis.Basis(L1, L2, Kmax, phi_=phi_func, precalc_phik_coeff=True, num_gauss_points=22)
phi_rec = basis.ReconstructedPhi(base, precalc_phik=False)

def read_ck_from_file(file_path="logs/ck_values.txt"):
    """
    Read ck coefficients from file in format: k1,k2,ck_value
    Returns a numpy array of shape (Kmax+1, Kmax+1)
    """
    try:
        # Create a copy of the file to avoid reading while it's being written
        copy_path = file_path.replace('.txt', '_copy.txt')
        
        # Copy the file
        import shutil
        shutil.copy2(file_path, copy_path)
        
        # Read from the copy
        data = np.genfromtxt(copy_path, delimiter=',')
        ck = np.zeros((Kmax+1, Kmax+1))
        
        for row in data:
            k1, k2, ck_value = int(row[0]), int(row[1]), row[2]
            if k1 <= Kmax and k2 <= Kmax:
                ck[k1, k2] = ck_value
        
        # Clean up the copy file
        try:
            os.remove(copy_path)
        except:
            pass  # Don't fail if we can't remove the copy
        
        return ck
    except Exception as e:
        print(f"Error reading ck values from file: {e}")
        return None

class RealTimeVisualizer:
    def __init__(self, base, phi_rec, grid_res=50, update_interval=100, use_file_ck=False, ck_file_path="logs/ck_values.txt"):
        self.base = base
        self.phi_rec = phi_rec
        self.grid_res = grid_res
        self.update_interval = update_interval
        self.use_file_ck = use_file_ck
        self.ck_file_path = ck_file_path
        self.states_list = []
        self.last_file_size = 0
        self.last_ck_file_size = 0
        self.file_path = "logs/agent_state.txt"
        self.running = True
        self.T_now = 1
        self.current_ck = None
        
        # Pre-calculate static data
        self.x1 = np.linspace(0, L1, grid_res)
        self.x2 = np.linspace(0, L2, grid_res)
        self.setup_static_plots()
        
        # Start file monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_file)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def setup_static_plots(self):
        # Calculate static data once
        Z_original = np.zeros((len(self.x1), len(self.x2)))
        Z_agent_fourier_rec = np.zeros((len(self.x1), len(self.x2)))
        
        for i in range(len(self.x1)):
            for j in range(len(self.x2)):
                Z_original[j, i] = self.base.phi([self.x1[i], self.x2[j]])
                Z_agent_fourier_rec[j, i] = self.phi_rec([self.x1[i], self.x2[j]])
        
        # Create figure and static plots
        self.fig = plt.figure(figsize=(18, 6))
        
        # Original function plot (static)
        self.ax1 = self.fig.add_subplot(131)
        im1 = self.ax1.imshow(Z_original, extent=(0, L1, 0, L2), origin='lower', cmap='viridis')
        self.ax1.set_title('Original Function Φ')
        self.ax1.set_xlabel('x1')
        self.ax1.set_ylabel('x2')
        self.ax1.set_aspect('auto')
        plt.colorbar(im1, ax=self.ax1, label='Function Value')
        
        # Fourier reconstruction plot (static)
        self.ax2 = self.fig.add_subplot(132)
        im2 = self.ax2.imshow(Z_agent_fourier_rec, extent=(0, L1, 0, L2), origin='lower', cmap='viridis')
        self.ax2.set_title(f'Fourier Reconstruction (Kmax = {Kmax})')
        self.ax2.set_xlabel('x1')
        self.ax2.set_ylabel('x2')
        self.ax2.set_aspect('auto')
        plt.colorbar(im2, ax=self.ax2, label='Function Value')
        
        # Dynamic plot setup
        self.ax3 = self.fig.add_subplot(133)
        self.ax3.set_title('Real-time Reconstructed from Ck')
        self.ax3.set_xlabel('x1')
        self.ax3.set_ylabel('x2')
        self.ax3.set_aspect('auto')
        self.ax3.set_xlim(0, L1)
        self.ax3.set_ylim(0, L2)
        
        # Initialize empty plot
        self.Z_rec_from_ck = np.zeros((len(self.x1), len(self.x2)))
        self.im3 = self.ax3.imshow(self.Z_rec_from_ck, extent=(0, L1, 0, L2), 
                                  origin='lower', cmap='viridis')
        self.cbar3 = plt.colorbar(self.im3, ax=self.ax3, label='Function Value')
        
        # Trajectory line
        self.traj_line, = self.ax3.plot([], [], 'k-', linewidth=1, alpha=0.7, label='Trajectory')
        
        plt.tight_layout()
        
    def _monitor_file(self):
        """Monitor file for new data in a separate thread"""
        while self.running:
            try:
                if os.path.exists(self.file_path):
                    current_size = os.path.getsize(self.file_path)
                    if current_size > self.last_file_size:
                        self._read_new_data()
                        self.last_file_size = current_size
                
                # If using file-based ck, also monitor ck file
                if self.use_file_ck and os.path.exists(self.ck_file_path):
                    current_ck_size = os.path.getsize(self.ck_file_path)
                    if current_ck_size > self.last_ck_file_size:
                        self._read_ck_from_file()
                        self.last_ck_file_size = current_ck_size
                        
                time.sleep(0.05)  # Check every 50ms
            except Exception as e:
                print(f"Error monitoring file: {e}")
                time.sleep(0.1)
                
    def _read_ck_from_file(self):
        """Read ck values from file"""
        self.current_ck = read_ck_from_file(self.ck_file_path)
        
    def _read_new_data(self):
        """Read only new lines from the file"""
        try:
            # Read all current data
            new_data = np.genfromtxt(self.file_path)
            if len(new_data.shape) == 1:  # Single row
                new_data = new_data.reshape(1, -1)
            
            if len(new_data) > len(self.states_list):
                # Extract only position columns (assuming columns 1 and 2 are x, y)
                self.states_list = new_data[:, 1:3]
                self.T_now = new_data[-1, 0]
                
        except Exception as e:
            print(f"Error reading file: {e}")
    
    def update_plot(self, frame):
        """Update function for animation"""
        if len(self.states_list) < 2 and not self.use_file_ck:
            return [self.im3, self.traj_line]
            
        # Calculate or read Ck coefficients
        if self.use_file_ck:
            if self.current_ck is None:
                self.current_ck = read_ck_from_file(self.ck_file_path)
            if self.current_ck is None:
                return [self.im3, self.traj_line]
            ck_values = self.current_ck
        else:
            if len(self.states_list) < 2:
                return [self.im3, self.traj_line]
            ck_values = calcCkCoeff(self.states_list, ti=0, T=self.T_now)
        
        # Calculate and print ergodic cost
        print(f"Erg Cost: {calcErgodicCost(ck_values)}")

        # Create new phi reconstruction
        phi_rec_from_ck = basis.ReconstructedPhiFromCk(self.base, ck_values)
        
        # Update Z matrix
        for i in range(len(self.x1)):
            for j in range(len(self.x2)):
                self.Z_rec_from_ck[j, i] = phi_rec_from_ck([self.x1[i], self.x2[j]])
        
        # Update image data
        self.im3.set_array(self.Z_rec_from_ck)
        self.im3.set_clim(vmin=self.Z_rec_from_ck.min(), vmax=self.Z_rec_from_ck.max())
        
        # Update trajectory
        if len(self.states_list) > 0:
            self.traj_line.set_data(self.states_list[:, 0], self.states_list[:, 1])
        
        return [self.im3, self.traj_line]
    
    def start_animation(self):
        """Start the real-time animation"""
        self.ani = FuncAnimation(self.fig, self.update_plot, interval=self.update_interval, 
                               blit=True, cache_frame_data=False)
        plt.show()
        
    def stop(self):
        """Stop the visualization"""
        self.running = False
        if hasattr(self, 'ani'):
            self.ani.event_source.stop()

def start_realtime_visualization(use_file_ck=False):
    """Start real-time visualization"""
    visualizer = RealTimeVisualizer(base, phi_rec, grid_res=50, update_interval=200, use_file_ck=use_file_ck)
    
    try:
        visualizer.start_animation()
    except KeyboardInterrupt:
        print("Stopping real-time visualization...")
        visualizer.stop()

def get_user_choice():
    """Get user choice for ck calculation method"""
    while True:
        print("\nChoose ck calculation method:")
        print("1. Calculate from trajectory (default)")
        print("2. Read from file (logs/ck_values.txt)")
        
        choice = input("Enter your choice (1 or 2): ").strip()
        
        if choice == "" or choice == "1":
            return False  # Calculate from trajectory
        elif choice == "2":
            return True   # Read from file
        else:
            print("Invalid choice. Please enter 1 or 2.")

def main():
    """Main execution function"""
    use_file_ck = get_user_choice()
    
    if use_file_ck:
        print("Using ck values from file: logs/ck_values.txt")
        # Check if file exists
        if not os.path.exists("logs/ck_values.txt"):
            print("Warning: ck_values.txt file not found. Please ensure the file exists.")
            return
    else:
        print("Calculating ck values from trajectory")
    
    # Ask for visualization mode
    print("\nChoose visualization mode:")
    print("1. Real-time visualization")
    print("2. Static visualization")
    
    vis_choice = input("Enter your choice (1 or 2): ").strip()
    
    if vis_choice == "" or vis_choice == "1":
        start_realtime_visualization(use_file_ck=use_file_ck)
    else:
        # Static visualization
        if use_file_ck:
            ck_values = read_ck_from_file("logs/ck_values.txt")
            if ck_values is None:
                print("Failed to read ck values from file")
                return
        else:
            states_list = np.genfromtxt("logs/agent_state.txt")[:, 1:3]
            ck_values = calcCkCoeff(states_list, ti=0, T=0.5)
        
        phi_rec_from_ck = basis.ReconstructedPhiFromCk(base, ck_values)
        
        if not use_file_ck:
            states_list = np.genfromtxt("logs/agent_state.txt")[:, 1:3]
            plotPhi(phi_rec_from_ck=phi_rec_from_ck, phi_rec_from_agent=phi_rec, all_traj=states_list)
        else:
            plotPhi(phi_rec_from_ck=phi_rec_from_ck, phi_rec_from_agent=phi_rec)
        plt.show()

# Comment out the original execution and replace with main function call
# start_realtime_visualization()

# For real-time visualization, uncomment the line below:
# start_realtime_visualization()

# For static visualization (original behavior), uncomment the lines below:
# states_list = np.genfromtxt("logs/agent_state.txt")[:, 1:3]
# ck_values = calcCkCoeff(states_list, ti=0, T=0.5)
# phi_rec_from_ck = basis.ReconstructedPhiFromCk(base, ck_values)
# plotPhi(phi_rec_from_ck=phi_rec_from_ck, phi_rec_from_agent=phi_rec, all_traj=states_list)
# plt.show()

if __name__ == "__main__":
    main()