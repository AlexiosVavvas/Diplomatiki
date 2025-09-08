#!/usr/bin/env python3
"""
Ergodic Exploration Visualization Script

This script provides real-time and static visualization of ergodic exploration 
for specific agents in a ROS2 environment. It subscribes to CkTable messages
and visualizes the phi function reconstruction using ck_values_average_in_range.

Usage:
    python vis_ck_erg.py AGENT_ID [--mode {realtime,static}] [--plot-mode {all,ros-only}]

Examples:
    python vis_ck_erg.py 1                              # Real-time, all plots for agent 1
    python vis_ck_erg.py 2 --mode static                # Static, all plots for agent 2
    python vis_ck_erg.py 1 --plot-mode ros-only         # Real-time, ROS plot only for agent 1
    python vis_ck_erg.py 3 --mode static --plot-mode ros-only  # Static, ROS plot only for agent 3
"""

import numpy as np
import time
import matplotlib
# Set matplotlib backend before importing pyplot
try:
    matplotlib.use('Qt5Agg', force=True)  # Use Qt5Agg backend for better event handling
except ImportError:
    try:
        matplotlib.use('TkAgg', force=True)  # Fallback to TkAgg
    except ImportError:
        # Use default backend if both fail
        pass
import matplotlib.pyplot as plt
import threading
from matplotlib.animation import FuncAnimation
import argparse
import signal
import atexit

# ROS2 imports
import rclpy
from rclpy.node import Node
from my_interfaces.msg import CkTable

# Import the new ROS library
from my_erg_lib import basis

# Global shutdown manager
class GracefulShutdown:
    def __init__(self):
        self.shutdown_requested = threading.Event()
        self.cleanup_called = threading.Event()  # Prevent multiple cleanup calls
        self.active_visualizers = []
        self.active_nodes = []
        self.setup_signal_handlers()
        
    def setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self.cleanup)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n\nReceived signal {signum}. Shutting down gracefully...")
        self.shutdown_requested.set()
        self.cleanup()
        
    def register_visualizer(self, visualizer):
        """Register a visualizer for cleanup"""
        self.active_visualizers.append(visualizer)
        
    def register_node(self, node):
        """Register a ROS node for cleanup"""
        self.active_nodes.append(node)
        
    def cleanup(self):
        """Clean up all resources"""
        # Prevent multiple cleanup calls
        if self.cleanup_called.is_set():
            return
        self.cleanup_called.set()
        
        try:
            # Stop all visualizers
            for viz in self.active_visualizers:
                try:
                    if hasattr(viz, 'stop'):
                        viz.stop()
                except Exception as e:
                    # Silently handle visualizer cleanup errors to prevent cascade failures
                    pass
            
            # Close matplotlib windows
            try:
                plt.close('all')
            except Exception as e:
                # Silently handle matplotlib cleanup errors
                pass
            
            # Destroy ROS nodes
            for node in self.active_nodes:
                try:
                    if hasattr(node, 'destroy_node'):
                        node.destroy_node()
                except Exception as e:
                    # Silently handle node cleanup errors
                    pass
            
            # Shutdown ROS
            try:
                if rclpy.ok():
                    rclpy.shutdown()
            except Exception as e:
                # Silently handle ROS shutdown errors
                pass
                
            print("Cleanup complete. Goodbye!")
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
        finally:
            # Force exit if needed to prevent hanging
            import os
            try:
                # Give a moment for cleanup to complete
                time.sleep(0.1)
            except:
                pass

# Global shutdown manager instance
shutdown_manager = GracefulShutdown()

Kmax = 4
L1 = 10 
L2 = 10

from src.ergodic_exploration.ergodic_exploration.agent_node import create_phi_func

# Create the phi function to match agent_node.py
phi_func = create_phi_func(L1=10.0, L2=10.0)

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

def plotPhi(phi_rec_from_ck, phi_rec_from_agent, phi_rec_from_ros_ck, all_traj=None, grid_res=50, clip_to_min_max=False, ergodic_cost=None):
    phi_original = base.phi

    x1 = np.linspace(0, L1, grid_res)
    x2 = np.linspace(0, L2, grid_res)

    # Plot in a 1x3 matplotlib figure as heatmap colors
    import matplotlib.pyplot as plt
    from matplotlib import cm

    Z_original = np.zeros((len(x1), len(x2)))
    Z_agent_fourier_rec = np.zeros((len(x1), len(x2)))
    Z_rec_from_ros_ck = np.zeros((len(x1), len(x2)))

    for i in range(len(x1)):
        for j in range(len(x2)):
            Z_original[j, i] = phi_original([x1[i], x2[j]])
            Z_agent_fourier_rec[j, i] = phi_rec_from_agent([x1[i], x2[j]])
            Z_rec_from_ros_ck[j, i] = phi_rec_from_ros_ck([x1[i], x2[j]])
        
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
        im3 = ax3.imshow(Z_rec_from_ros_ck, extent=(0, L1, 0, L2), 
                        origin='lower', cmap=cm.viridis, vmin=min_val, vmax=max_val)
    else:
        im3 = ax3.imshow(Z_rec_from_ros_ck, extent=(0, L1, 0, L2), 
                        origin='lower', cmap=cm.viridis)
    ax3.set_title('Reconstructed from ROS Ck (ck_values_average_in_range)')
    ax3.set_xlabel('x1')
    ax3.set_ylabel('x2')
    ax3.set_aspect('auto')
    # x and y lims to 0 -> L1 and 0 -> L2
    ax3.set_xlim(0, L1)
    ax3.set_ylim(0, L2)
    plt.colorbar(im3, ax=ax3, label='Function Value')
    
    # Add ergodic cost as text annotation
    if ergodic_cost is not None:
        ax3.text(0.02, 0.98, f'Ergodic Cost: {ergodic_cost:.4f}', 
                transform=ax3.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                fontsize=10, verticalalignment='top')
    
    if all_traj is not None:
        all_traj = np.array(all_traj)
        ax3.plot(all_traj[:, 0], all_traj[:, 1], 'k-', label='Trajectory')

    plt.tight_layout()

def plot_ros_only(phi_rec_from_ros_ck, agent_id, grid_res=50, ergodic_cost=None):
    """Plot only the ROS Ck reconstruction in a single figure"""
    x1 = np.linspace(0, L1, grid_res)
    x2 = np.linspace(0, L2, grid_res)

    Z_rec_from_ros_ck = np.zeros((len(x1), len(x2)))

    for i in range(len(x1)):
        for j in range(len(x2)):
            Z_rec_from_ros_ck[j, i] = phi_rec_from_ros_ck([x1[i], x2[j]])
        
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    
    im = ax.imshow(Z_rec_from_ros_ck, extent=(0, L1, 0, L2), origin='lower', cmap='viridis')
    ax.set_title(f'Agent {agent_id} - ROS Ck (ck_values_average_in_range)')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_aspect('equal')
    ax.set_xlim(0, L1)
    ax.set_ylim(0, L2)
    plt.colorbar(im, ax=ax, label='Function Value')
    
    # Add ergodic cost as text annotation
    if ergodic_cost is not None:
        ax.text(0.02, 0.98, f'Ergodic Cost: {ergodic_cost:.4f}', 
                transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                fontsize=12, verticalalignment='top')
    
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

# base = basis.Basis(L1, L2, Kmax, phi_=lambda s: 1/100, precalc_phik_coeff=True, num_gauss_points=22)
base = basis.Basis(L1, L2, Kmax, phi_=phi_func, precalc_phik_coeff=True, num_gauss_points=22)
phi_rec = basis.ReconstructedPhi(base, precalc_phik=False)

class CkSubscriber(Node):
    """ROS subscriber to listen to CkTable messages from a specific agent"""
    
    def __init__(self, agent_id):
        super().__init__(f'ck_visualizer_{agent_id}')
        self.agent_id = agent_id
        self.latest_ck_values_average = None
        self.latest_ck_values = None
        self.ergodic_cost = 0
        self.erg_cost_reduction_perc = 0.0

        # Subscribe to the agent's ck topic
        self.subscription = self.create_subscription(
            CkTable,
            f'agent_{agent_id}/ck',
            self.ck_callback,
            10
        )
        
        self.get_logger().info(f'Subscribed to agent_{agent_id}/ck topic')
    
    def ck_callback(self, msg):
        """Callback to handle incoming CkTable messages"""
        try:
            # Reshape flattened arrays back to square matrices
            table_size = msg.table_size
            
            # Get ck_values_average_in_range
            if len(msg.ck_values_average_in_range) == table_size * table_size:
                self.latest_ck_values_average = np.array(msg.ck_values_average_in_range).reshape(table_size, table_size)
                self.get_logger().debug(f'Received ck_values_average_in_range from agent {self.agent_id}')
            
            # Also store regular ck_values for comparison
            if len(msg.ck_values) == table_size * table_size:
                self.latest_ck_values = np.array(msg.ck_values).reshape(table_size, table_size)
                
            self.ergodic_cost = msg.total_erg_cost_in_range
            self.erg_cost_reduction_perc = msg.erg_cost_reduction_perc

        except Exception as e:
            self.get_logger().error(f'Error processing CkTable message: {e}')

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

class RealTimeVisualizer:
    def __init__(self, base, phi_rec, agent_id, plot_mode='all', grid_res=50, update_interval=200):
        self.base = base
        self.phi_rec = phi_rec
        self.agent_id = agent_id
        self.plot_mode = plot_mode
        self.grid_res = grid_res
        self.update_interval = update_interval
        self.running = True
        self.ani = None  # Initialize animation object
        self.fig = None  # Initialize figure object
        
        # Initialize ROS subscriber
        self.ck_subscriber = CkSubscriber(agent_id)
        
        # Register with shutdown manager
        shutdown_manager.register_visualizer(self)
        shutdown_manager.register_node(self.ck_subscriber)
        
        # Pre-calculate static data
        self.x1 = np.linspace(0, L1, grid_res)
        self.x2 = np.linspace(0, L2, grid_res)
        self.setup_plots()
        
    def setup_plots(self):
        if self.plot_mode == 'ros-only':
            self.setup_ros_only_plot()
        else:
            self.setup_all_plots()
    
    def setup_ros_only_plot(self):
        """Setup single plot showing only ROS Ck visualization"""
        # Create figure with single subplot
        self.fig = plt.figure(self.agent_id, figsize=(8, 6))
        self.ax3 = self.fig.add_subplot(111)
        
        self.ax3.set_title(f'Agent {self.agent_id} - ROS Ck (ck_values_average_in_range)')
        self.ax3.set_xlabel('x1')
        self.ax3.set_ylabel('x2')
        self.ax3.set_aspect('equal')
        self.ax3.set_xlim(0, L1)
        self.ax3.set_ylim(0, L2)
        
        # Initialize empty plot
        self.Z_rec_from_ros_ck = np.zeros((len(self.x1), len(self.x2)))
        self.im3 = self.ax3.imshow(self.Z_rec_from_ros_ck, extent=(0, L1, 0, L2), 
                                  origin='lower', cmap='viridis')
        self.cbar3 = plt.colorbar(self.im3, ax=self.ax3, label='Function Value')
        
        # Initialize ergodic cost text (will be updated in animation)
        self.ergodic_cost_text = self.ax3.text(0.02, 0.98, 'Ergodic Cost: N/A', 
                                              transform=self.ax3.transAxes, 
                                              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                                              fontsize=12, verticalalignment='top')
        
        plt.tight_layout()
        
    def setup_all_plots(self):
        """Setup all three plots (original implementation)"""
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
        
        # Dynamic plot setup - ROS ck_values_average_in_range
        self.ax3 = self.fig.add_subplot(133)
        self.ax3.set_title(f'Real-time from Agent {self.agent_id} ROS Ck (ck_values_average_in_range)')
        self.ax3.set_xlabel('x1')
        self.ax3.set_ylabel('x2')
        self.ax3.set_aspect('auto')
        self.ax3.set_xlim(0, L1)
        self.ax3.set_ylim(0, L2)
        
        # Initialize empty plot
        self.Z_rec_from_ros_ck = np.zeros((len(self.x1), len(self.x2)))
        self.im3 = self.ax3.imshow(self.Z_rec_from_ros_ck, extent=(0, L1, 0, L2), 
                                  origin='lower', cmap='viridis')
        self.cbar3 = plt.colorbar(self.im3, ax=self.ax3, label='Function Value')
        
        # Initialize ergodic cost text (will be updated in animation)
        self.ergodic_cost_text = self.ax3.text(0.02, 0.98, 'Ergodic Cost: N/A', 
                                              transform=self.ax3.transAxes, 
                                              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                                              fontsize=10, verticalalignment='top')
        
        plt.tight_layout()
    
    def update_plot(self, frame):
        """Update function for animation"""
        # Check for shutdown request
        if shutdown_manager.shutdown_requested.is_set() or not self.running:
            self.stop()
            return [self.im3, self.ergodic_cost_text] if (hasattr(self, 'im3') and hasattr(self, 'ergodic_cost_text')) else []
            
        # Check if we have new ck data from ROS
        if self.ck_subscriber.latest_ck_values_average is None:
            return [self.im3, self.ergodic_cost_text] if (hasattr(self, 'im3') and hasattr(self, 'ergodic_cost_text')) else []
        
        # Use ck_values_average_in_range for visualization
        ck_values = self.ck_subscriber.latest_ck_values_average
        
        # Calculate ergodic cost
        try:
            ergodic_cost = self.ck_subscriber.ergodic_cost
            # print(f"Agent {self.agent_id} - Ergodic Cost (avg in range): {ergodic_cost:.4f}")
            # Update the text display on the plot
            if hasattr(self, 'ergodic_cost_text'):
                self.ergodic_cost_text.set_text(f'Ergodic Cost: {ergodic_cost:.4f} (-> {100 * self.ck_subscriber.erg_cost_reduction_perc:.2f}%)')
        except Exception as e:
            print(f"Error calculating ergodic cost: {e}")
            if hasattr(self, 'ergodic_cost_text'):
                self.ergodic_cost_text.set_text('Ergodic Cost: Error')
        
        # Create new phi reconstruction
        try:
            phi_rec_from_ros_ck = basis.ReconstructedPhiFromCk(self.base, ck_values)
            
            # Update Z matrix
            for i in range(len(self.x1)):
                for j in range(len(self.x2)):
                    self.Z_rec_from_ros_ck[j, i] = phi_rec_from_ros_ck([self.x1[i], self.x2[j]])
            
            # Update image data
            self.im3.set_array(self.Z_rec_from_ros_ck)
            self.im3.set_clim(vmin=self.Z_rec_from_ros_ck.min(), vmax=self.Z_rec_from_ros_ck.max())
            
        except Exception as e:
            print(f"Error updating plot: {e}")
        
        # Return both the image and text for animation updates
        return [self.im3, self.ergodic_cost_text] if (hasattr(self, 'im3') and self.im3 is not None and hasattr(self, 'ergodic_cost_text')) else []
    
    def on_key_press(self, event):
        """Handle keyboard events"""
        if event.key in ['q', 'Q', 'escape']:
            if self.running:  # Only trigger shutdown once
                print("\nKey pressed. Shutting down gracefully...")
                shutdown_manager.shutdown_requested.set()
                self.stop()
    
    def on_close(self, event):
        """Handle window close event"""
        if self.running:  # Only trigger shutdown once
            print("\nWindow closed. Shutting down gracefully...")
            shutdown_manager.shutdown_requested.set()
            self.stop()
    
    def start_animation(self):
        """Start the real-time animation"""
        try:
            # Set up event handlers
            if hasattr(self.fig, 'canvas') and self.fig.canvas is not None:
                self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
                self.fig.canvas.mpl_connect('close_event', self.on_close)
            
            # Add instruction text
            # if self.plot_mode == 'ros-only':
            #     self.fig.suptitle(f'Agent {self.agent_id} - ROS Ck Visualization (Press Q or Ctrl+C to exit)', 
            #                     fontsize=14, y=0.95)
            # else:
            #     self.fig.suptitle(f'Agent {self.agent_id} - Ergodic Exploration Visualization (Press Q or Ctrl+C to exit)', 
            #                     fontsize=14, y=0.95)
            
            # Create animation with better error handling
            self.ani = None
            try:
                self.ani = FuncAnimation(self.fig, self.update_plot, interval=self.update_interval, 
                                       blit=True, cache_frame_data=False, repeat=True)
                print("Animation started successfully. Use Ctrl+C or 'Q' to exit gracefully.")
                
                # Use plt.show() in a way that handles backend issues
                plt.show(block=True)
                
            except Exception as e:
                print(f"Error creating or running animation: {e}")
                self.stop()
                
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received.")
            self.stop()
        except Exception as e:
            print(f"Error in start_animation: {e}")
            self.stop()
        finally:
            # Ensure cleanup is called
            if self.running:
                self.stop()
        
    def stop(self):
        """Stop the visualization"""
        if not self.running:
            return  # Already stopped, avoid multiple calls
            
        self.running = False
        
        if hasattr(self, 'ani') and self.ani is not None:
            try:
                # Check if the animation has an event_source and it's not None
                if hasattr(self.ani, 'event_source') and self.ani.event_source is not None:
                    if hasattr(self.ani.event_source, 'stop'):
                        self.ani.event_source.stop()
                    else:
                        # Alternative: try to remove the animation
                        self.ani._stop()
                        
            except (AttributeError, RuntimeError) as e:
                # Silently handle common animation shutdown errors
                pass
            except Exception as e:
                print(f"Warning: Error stopping animation: {e}")
                
        # Close the figure if it exists
        if hasattr(self, 'fig') and self.fig is not None:
            try:
                plt.close(self.fig)
                self.fig = None
            except Exception as e:
                print(f"Warning: Error closing figure: {e}")
                
        # Set ani to None to prevent further calls
        self.ani = None

def start_realtime_visualization(agent_id, plot_mode='all'):
    """Start real-time visualization for specific agent"""
    
    # Initialize ROS
    rclpy.init()
    
    try:
        visualizer = RealTimeVisualizer(base, phi_rec, agent_id, plot_mode, grid_res=50, update_interval=500)
        
        # Start ROS spinning in a separate thread
        def ros_spin():
            try:
                while rclpy.ok() and not shutdown_manager.shutdown_requested.is_set():
                    rclpy.spin_once(visualizer.ck_subscriber, timeout_sec=0.1)
            except Exception as e:
                if not shutdown_manager.shutdown_requested.is_set():
                    print(f"ROS spinning error: {e}")
        
        ros_thread = threading.Thread(target=ros_spin, daemon=True)
        ros_thread.start()
        
        print(f"Starting real-time visualization for agent_{agent_id}")
        print("Press 'Q' key or Ctrl+C to exit gracefully")
        
        # Start the visualization
        visualizer.start_animation()
        
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received in main thread.")
    except Exception as e:
        print(f"Error in real-time visualization: {e}")
    finally:
        shutdown_manager.shutdown_requested.set()
        # Cleanup is handled by shutdown_manager

def start_static_visualization(agent_id, plot_mode='all'):
    """Start static visualization using latest ROS message from specific agent"""
    
    # Initialize ROS
    rclpy.init()
    
    try:
        # Create subscriber to get one message
        ck_subscriber = CkSubscriber(agent_id)
        shutdown_manager.register_node(ck_subscriber)
        
        print(f"Waiting for CkTable message from agent_{agent_id}...")
        print("Press Ctrl+C to cancel if needed")
        
        # Spin until we get a message
        timeout_count = 0
        while (ck_subscriber.latest_ck_values_average is None and 
               timeout_count < 50 and 
               not shutdown_manager.shutdown_requested.is_set()):
            rclpy.spin_once(ck_subscriber, timeout_sec=0.1)
            timeout_count += 1
        
        if shutdown_manager.shutdown_requested.is_set():
            print("Shutdown requested during message wait.")
            return
            
        if ck_subscriber.latest_ck_values_average is None:
            print(f"Timeout: No messages received from agent_{agent_id} after 5 seconds.")
            print("Troubleshooting steps:")
            print(f"  1. Check if agent_{agent_id} is running: ros2 node list | grep agent_{agent_id}")
            print(f"  2. Check if topic exists: ros2 topic list | grep agent_{agent_id}/ck")
            print(f"  3. Check message publishing: ros2 topic echo agent_{agent_id}/ck --max-msgs 1")
            return
        
        # Use the received ck_values_average_in_range for visualization
        ck_values_average = ck_subscriber.latest_ck_values_average
        phi_rec_from_ros_ck = basis.ReconstructedPhiFromCk(base, ck_values_average)
        
        print(f"Received ck data from agent_{agent_id}. Creating static visualization...")
        print("Close the window or press Ctrl+C to exit")
        
        if plot_mode == 'ros-only':
            # Show only the ROS Ck plot
            plot_ros_only(phi_rec_from_ros_ck, agent_id, ergodic_cost=ck_subscriber.ergodic_cost)
        else:
            # Show all three plots
            plotPhi(phi_rec_from_ck=phi_rec_from_ros_ck, 
                    phi_rec_from_agent=phi_rec, 
                    phi_rec_from_ros_ck=phi_rec_from_ros_ck,
                    ergodic_cost=ck_subscriber.ergodic_cost)

        # Set up graceful shutdown for static plot
        def on_key_press(event):
            if event.key in ['q', 'Q', 'escape']:
                print("\nKey pressed. Closing...")
                plt.close('all')
        
        def on_close(event):
            print("\nWindow closed.")
        
        # Connect event handlers
        fig = plt.gcf()
        fig.canvas.mpl_connect('key_press_event', on_key_press)
        fig.canvas.mpl_connect('close_event', on_close)
        fig.suptitle(f'Agent {agent_id} - Static Visualization (Press Q or close window to exit)', 
                    fontsize=14, y=0.95)
        
        plt.show()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error in static visualization: {e}")
    finally:
        shutdown_manager.shutdown_requested.set()
        # Cleanup handled by shutdown_manager

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Visualize ergodic exploration for specific agent')
    parser.add_argument('agent_id', type=int, help='Agent ID to visualize (e.g., 1, 2, 3...)')
    parser.add_argument('--mode', choices=['realtime', 'static'], default='realtime', 
                       help='Visualization mode (default: realtime)')
    parser.add_argument('--plot-mode', choices=['all', 'ros-only'], default='all',
                       help='Plot display mode: "all" shows all 3 plots, "ros-only" shows only the ROS Ck plot (default: all)')
    
    try:
        args = parser.parse_args()
        
        print(f"Starting visualization for agent_{args.agent_id}")
        print(f"Mode: {args.mode}")
        print(f"Plot mode: {args.plot_mode}")
        print("Make sure the agent is running and publishing CkTable messages!")
        print("\nGraceful shutdown: Press Ctrl+C or 'Q' key to exit")
        
        if args.mode == 'realtime':
            start_realtime_visualization(args.agent_id, args.plot_mode)
        else:
            start_static_visualization(args.agent_id, args.plot_mode)
            
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received in main.")
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        # Trigger cleanup
        shutdown_manager.shutdown_requested.set()
        print("Main function complete.")

# Add at the end of file
if __name__ == "__main__":
    main()