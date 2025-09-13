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
        self.traj_fig.suptitle('Agent Trajectories', fontsize=14)
        
        # Auto-refresh flag
        self.auto_refresh = True
        
        # Colors for different agents
        self.agent_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
        
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
        
        self.traj_ax.set_title('Agent Trajectories')
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
    
    def separate_agent_data(self, data, agent_column_idx):
        """Separate multi-agent data by agent index"""
        if data is None:
            return {}
        
        # Handle 1D data (single row)
        if data.ndim == 1:
            if len(data) > abs(agent_column_idx):
                agent_idx = int(data[agent_column_idx])
                return {agent_idx: data}
            else:
                return {0: data}  # Default to agent 0
        
        # Handle 2D data (multiple rows)
        if data.shape[1] <= abs(agent_column_idx):
            return {0: data}  # Default to agent 0 if no agent column
        
        agent_data = {}
        try:
            # Get unique agent indices from the specified column
            agent_indices = np.unique(data[:, agent_column_idx].astype(int))
            
            for agent_idx in agent_indices:
                # Filter data for this agent
                mask = data[:, agent_column_idx] == agent_idx
                agent_data[agent_idx] = data[mask]
        except (IndexError, ValueError):
            # Fallback: treat all data as agent 0
            agent_data[0] = data
        
        return agent_data
            
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
        
        # Separate data by agents (agent_state.txt and ergodic_cost.txt have agent indices in last column)
        agent_states_by_agent = self.separate_agent_data(agent_data, -1)  # Agent index in last column
        ergodic_data_by_agent = self.separate_agent_data(ergodic_data, -1)  # Agent index in last column
        
        # Plot 1: CBF values (assuming CBF is still single agent or first agent)
        # if cbf_data is not None:
        #     self.cbf_ax.plot(cbf_data[:, 0], label='h', linewidth=2)
        #     self.cbf_ax.plot(cbf_data[:, 1], label='PSI', linewidth=2)
        #     self.cbf_ax.plot(cbf_data[:, 2], label='grad_h', linewidth=2)
        #     self.cbf_ax.set_title('Control Barrier Function')
        #     self.cbf_ax.legend()
        #     self.cbf_ax.grid(True)
            
        # Plot 2: Safe Control (assuming CBF is still single agent or first agent)
        # if cbf_data is not None:
        #     self.safe_control_ax.plot(cbf_data[:, 3], label='U_safe[0]', linewidth=2)
        #     self.safe_control_ax.plot(cbf_data[:, 4], label='U_safe[1]', linewidth=2)
        #     self.safe_control_ax.set_title('Safe Control Inputs')
        #     self.safe_control_ax.legend()
        #     self.safe_control_ax.grid(True)
            
        # Plot 3: PSI Components (assuming PSI is still single agent or first agent)
        # if psi_data is not None:
        #     self.psi_ax.plot(psi_data[:, 0], label='h_ddot', linewidth=2)
        #     self.psi_ax.plot(psi_data[:, 1], label='alpha_1*h_dot', linewidth=2)
        #     self.psi_ax.plot(psi_data[:, 2], label='alpha_2*h', linewidth=2)
        #     self.psi_ax.plot(psi_data[:, 3], label='PSI', linewidth=2)
        #     self.psi_ax.set_title('PSI Components')
        #     self.psi_ax.legend()
        #     self.psi_ax.grid(True)
            
        # Plot 4: Control Inputs (multi-agent)
        # if agent_states_by_agent:
        #     for agent_idx, agent_states in agent_states_by_agent.items():
        #         if agent_states is not None and len(agent_states) > 0 and agent_states.shape[1] >= 5:  # time, x, y, u1, u2, (agent_idx)
        #             color = self.agent_colors[agent_idx % len(self.agent_colors)]
        #             self.control_ax.plot(agent_states[:, 0], agent_states[:, 3], 
        #                                label=f'U1 - Agent {agent_idx}', linewidth=2, color=color, linestyle='-')
        #             self.control_ax.plot(agent_states[:, 0], agent_states[:, 4], 
        #                                label=f'U2 - Agent {agent_idx}', linewidth=2, color=color, linestyle='--')

        # self.control_ax.set_title('Control Inputs (All Agents)')
        # self.control_ax.legend()
        # self.control_ax.grid(True)
        # self.control_ax.set_xlabel('Time [s]')
        # self.control_ax.set_ylabel('Control Values')
            
        # Plot 5: Ergodic Cost (multi-agent)
        if ergodic_data_by_agent:
            for agent_idx, ergodic_agent_data in ergodic_data_by_agent.items():
                if ergodic_agent_data is not None and len(ergodic_agent_data) > 0 and ergodic_agent_data.shape[1] >= 3:  # time, cost, active_flag, (agent_idx)
                    if agent_idx == -1:
                        # Special case for total ergodic cost
                        self.ergodic_ax.plot(ergodic_agent_data[:, 0], ergodic_agent_data[:, 1], 
                                           label='Total Ergodic Cost (Average CK)', linewidth=2, color='black', linestyle='-')
                    else:
                        # Individual agent ergodic costs
                        color = self.agent_colors[agent_idx % len(self.agent_colors)]
                        self.ergodic_ax.plot(ergodic_agent_data[:, 0], ergodic_agent_data[:, 1], 
                                           label=f'Ergodic Cost - Agent {agent_idx}', linewidth=2, color=color)
                        # Scale binary 0-1 active flag to be visible with ergodic cost
                        if len(ergodic_agent_data[:, 1]) > 0 and np.max(ergodic_agent_data[:, 1]) > 0:
                            scale_factor = np.max(ergodic_agent_data[:, 1])
                            self.ergodic_ax.plot(ergodic_agent_data[:, 0], ergodic_agent_data[:, 2] * scale_factor, 
                                               label=f'Active Safe Control - Agent {agent_idx}', linewidth=2, 
                                               color=color, linestyle=':', alpha=0.7)
        
        self.ergodic_ax.set_title('Ergodic Cost (Individual Agents + Total)')
        # self.ergodic_ax.legend()
        self.ergodic_ax.grid(True)
        self.ergodic_ax.set_xlabel('Time [s]')
        self.ergodic_ax.set_ylabel('Cost')
            
        # Plot 6: Agent Trajectories (multi-agent)
        if agent_states_by_agent:
            for agent_idx, agent_states in agent_states_by_agent.items():
                if agent_states is not None and len(agent_states) > 0 and agent_states.shape[1] >= 3:  # time, x, y, ...
                    color = self.agent_colors[agent_idx % len(self.agent_colors)]
                    
                    # Current position (colored dot for each agent)
                    self.traj_ax.scatter(agent_states[-1, 1], agent_states[-1, 2], s=100, 
                                       c=color, label=f'Agent {agent_idx} Current', zorder=3, marker='o')
                    
                    # Agent trajectory
                    self.traj_ax.plot(agent_states[:, 1], agent_states[:, 2], linewidth=2, 
                                    label=f'Agent {agent_idx} Path', color=color, zorder=2)

        # Obstacles (same for all agents)
        if obstacles_data is not None:
            self.traj_ax.scatter(obstacles_data[:, 0], obstacles_data[:, 1], 
                               c='black', s=20, label='Obstacles', zorder=1, alpha=0.7)
        
        self.traj_ax.set_xlim(0, 10)
        self.traj_ax.set_ylim(0, 10)
        self.traj_ax.set_title('Agent Trajectories')
        # self.traj_ax.legend()
        self.traj_ax.grid(True)
        self.traj_ax.set_xlabel('X Position')
        self.traj_ax.set_ylabel('Y Position')

        # Draw all figures
        for fig in [self.cbf_fig, self.safe_control_fig, self.psi_fig, 
                   self.traj_fig, self.control_fig, self.ergodic_fig]:
            fig.canvas.draw()
        
    def show(self):
        plt.show()

if __name__ == "__main__":
    dashboard = LiveDashboard()
    print("Multi-Agent Dashboard Controls:")
    print("- Press 'e' to manually refresh")
    print("- Press 'a' to toggle auto-refresh")
    print("- Press 'c' to clear all plots")
    print("- Press 'q' to quit")
    dashboard.show()