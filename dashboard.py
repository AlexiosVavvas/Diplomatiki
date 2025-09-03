import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
from matplotlib.widgets import Button
import os

class LiveDashboard:
    def __init__(self):
        # Create separate figure windows for each plot
        self.cbf_fig, self.cbf_ax = plt.subplots(1, 1, figsize=(8, 6))
        self.cbf_fig.suptitle('Control Barrier Function', fontsize=14)
        
        self.safe_control_fig, self.safe_control_ax = plt.subplots(1, 1, figsize=(8, 6))
        self.safe_control_fig.suptitle('Safe Control Inputs', fontsize=14)
        
        self.psi_fig, self.psi_ax = plt.subplots(1, 1, figsize=(8, 6))
        self.psi_fig.suptitle('PSI Components', fontsize=14)
        
        self.control_fig, self.control_ax = plt.subplots(1, 1, figsize=(8, 6))
        self.control_fig.suptitle('Control Inputs', fontsize=14)
        
        self.ergodic_fig, self.ergodic_ax = plt.subplots(1, 1, figsize=(8, 6))
        self.ergodic_fig.suptitle('Ergodic Cost', fontsize=14)
        
        self.traj_fig, self.traj_ax = plt.subplots(1, 1, figsize=(8, 6))
        self.traj_fig.suptitle('Agent Trajectory', fontsize=14)
        
        # Auto-refresh flag
        self.auto_refresh = True
        
        # Set up key press event on agent trajectory figure
        self.traj_fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Initialize animation
        self.anim = animation.FuncAnimation(
            self.cbf_fig, self.update_plots, interval=200, blit=False
        )
        
    def on_key_press(self, event):
        if event.key == 'e':
            print("Manual refresh triggered")
            self.update_plots(None)
        elif event.key == 'a':
            self.auto_refresh = not self.auto_refresh
            status = 'ON' if self.auto_refresh else 'OFF'
            print(f"Auto-refresh: {status}")
        elif event.key == 'c':
            print("Clearing all plots")
            self.clear_plots()
        elif event.key == 'q':
            print("Closing all windows")
            plt.close('all')
    
    def clear_plots(self):
        """Clear all plot lines while maintaining axes structure"""
        # Clear all axes
        self.cbf_ax.clear()
        self.safe_control_ax.clear()
        self.psi_ax.clear()
        self.traj_ax.clear()
        self.control_ax.clear()
        self.ergodic_ax.clear()
        
        # Reset titles and grids
        self.cbf_ax.set_title('Control Barrier Function')
        self.cbf_ax.grid(True)
        
        self.safe_control_ax.set_title('Safe Control Inputs')
        self.safe_control_ax.grid(True)
        
        self.psi_ax.set_title('PSI Components')
        self.psi_ax.grid(True)
        
        self.control_ax.set_title('Control Inputs')
        self.control_ax.grid(True)
        
        self.ergodic_ax.set_title('Ergodic Cost')
        self.ergodic_ax.grid(True)
        
        self.traj_ax.set_title('Agent Trajectory')
        self.traj_ax.set_xlim(0, 10)
        self.traj_ax.set_ylim(0, 10)
        self.traj_ax.grid(True)
        
        # Draw all figures
        for fig in [self.cbf_fig, self.safe_control_fig, self.psi_fig, 
                   self.traj_fig, self.control_fig, self.ergodic_fig]:
            fig.canvas.draw()
            
    def load_data(self, filename):
        try:
            if os.path.exists(filename):
                return np.loadtxt(filename)
            else:
                return None
        except:
            return None
            
    def update_plots(self, frame):
        if not self.auto_refresh and frame is not None:
            return
            
        # Clear all axes
        self.cbf_ax.clear()
        self.safe_control_ax.clear()
        self.psi_ax.clear()
        self.traj_ax.clear()
        self.control_ax.clear()
        self.ergodic_ax.clear()
            
        # Load data
        cbf_data = self.load_data('logs/cbf_log.txt')
        psi_data = self.load_data('logs/PSI.txt')
        agent_data = self.load_data('logs/agent_state.txt')
        obstacles_data = self.load_data('logs/obstacles_points.txt')
        ergodic_data = self.load_data('logs/ergodic_cost.txt')
        
        # Plot 1: CBF values
        if cbf_data is not None:
            self.cbf_ax.plot(cbf_data[:, 0], label='h', linewidth=2)
            self.cbf_ax.plot(cbf_data[:, 1], label='PSI', linewidth=2)
            self.cbf_ax.plot(cbf_data[:, 2], label='grad_h', linewidth=2)
            self.cbf_ax.set_title('Control Barrier Function')
            self.cbf_ax.legend()
            self.cbf_ax.grid(True)
            
        # Plot 2: Safe Control
        if cbf_data is not None:
            self.safe_control_ax.plot(cbf_data[:, 3], label='U_safe[0]', linewidth=2)
            self.safe_control_ax.plot(cbf_data[:, 4], label='U_safe[1]', linewidth=2)
            self.safe_control_ax.set_title('Safe Control Inputs')
            self.safe_control_ax.legend()
            self.safe_control_ax.grid(True)
            
        # Plot 3: PSI Components
        if psi_data is not None:
            self.psi_ax.plot(psi_data[:, 0], label='h_ddot', linewidth=2)
            self.psi_ax.plot(psi_data[:, 1], label='alpha_1*h_dot', linewidth=2)
            self.psi_ax.plot(psi_data[:, 2], label='alpha_2*h', linewidth=2)
            self.psi_ax.plot(psi_data[:, 3], label='PSI', linewidth=2)
            self.psi_ax.set_title('PSI Components')
            self.psi_ax.legend()
            self.psi_ax.grid(True)
            
        # Plot 4: Control Inputs
        if agent_data is not None and agent_data.shape[1] > 4:
            self.control_ax.plot(agent_data[:, 0], agent_data[:, 3], label='U1', linewidth=2)
            self.control_ax.plot(agent_data[:, 0], agent_data[:, 4], label='U2', linewidth=2)

            self.control_ax.set_title('Control Inputs')
            self.control_ax.legend()
            self.control_ax.grid(True)
            
        # Plot 5: Ergodic Cost
        if ergodic_data is not None:
            self.ergodic_ax.plot(ergodic_data[:, 0], ergodic_data[:, 1], 
                              label='Ergodic Cost', linewidth=2)
            # Scale binary 0-1 active flag to be of the same order of magnitude to plot together with ergodic cost
            self.ergodic_ax.plot(ergodic_data[:, 0], ergodic_data[:, 2] * ergodic_data[0, 1], 
                              label='Active Safe Control', linewidth=2)
            self.ergodic_ax.set_title('Ergodic Cost')
            self.ergodic_ax.legend()
            self.ergodic_ax.grid(True)
            
        # Plot 6: Agent Trajectory
        if agent_data is not None:
            # Red dot at the agent's current position
            self.traj_ax.scatter(agent_data[-1, 1], agent_data[-1, 2], s=100, c='red', label='Current Position', zorder=3)
            # Agent trajectory
            self.traj_ax.plot(agent_data[:, 1], agent_data[:, 2], linewidth=2, label='Agent Path', zorder=2)
            # Obstacles
            if obstacles_data is not None:
                self.traj_ax.scatter(obstacles_data[:, 0], obstacles_data[:, 1], 
                                   c='black', s=20, label='Obstacles', zorder=1)
            self.traj_ax.set_xlim(0, 10)
            self.traj_ax.set_ylim(0, 10)
            self.traj_ax.set_title('Agent Trajectory')
            self.traj_ax.legend()
            self.traj_ax.grid(True)

        # Draw all figures
        for fig in [self.cbf_fig, self.safe_control_fig, self.psi_fig, 
                   self.traj_fig, self.control_fig, self.ergodic_fig]:
            fig.canvas.draw()
        
    def show(self):
        plt.show()

if __name__ == "__main__":
    dashboard = LiveDashboard()
    print("Dashboard Controls:")
    print("- Press 'e' to manually refresh")
    print("- Press 'a' to toggle auto-refresh")
    print("- Press 'c' to clear all plots")
    print("- Press 'q' to quit")
    dashboard.show()