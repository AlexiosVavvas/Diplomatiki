#!/usr/bin/env python3
"""
Ergodic Exploration Visualization Script

This script provides real-time and static visualization of ergodic exploration 
for specific agents in a ROS2 environment. It subscribes to CkTable messages
and visualizes the phi function reconstruction using ck_values_average_in_range.

Usage:
    python vis_ck_erg.py AGENT_ID [--mode {realtime,static}] [--plot-mode {all,ros-only}] [--3d] [--color-range MIN MAX]

Examples:
    python vis_ck_erg.py 1                              # Real-time, all plots for agent 1
    python vis_ck_erg.py 2 --mode static                # Static, all plots for agent 2
    python vis_ck_erg.py 1 --plot-mode ros-only         # Real-time, ROS plot only for agent 1
    python vis_ck_erg.py 3 --mode static --plot-mode ros-only  # Static, ROS plot only for agent 3
    python vis_ck_erg.py 1 --3d                         # Real-time, 3D surface visualization for agent 1
    python vis_ck_erg.py 1 --3d --color-range 0 0.1     # Real-time, 3D surface with fixed color range and z-limits
"""

import numpy as np
import time
import matplotlib
# Set matplotlib backend before importing pyplot
# Try multiple backends in order of preference
backend_set = False
for backend in ['Qt5Agg', 'TkAgg', 'GTK3Agg', 'WXAgg']:
    try:
        matplotlib.use(backend, force=True)
        backend_set = True
        print(f"Using matplotlib backend: {backend}")
        break
    except (ImportError, ValueError):
        continue

if not backend_set:
    # Use default backend
    print("Using default matplotlib backend")
    
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
from src.ergodic_exploration.my_erg_lib.basis import Basis, ReconstructedPhi, ReconstructedPhiFromCk

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

# =============----------------------------
# Here used in phi calculations, not only visualization

Kmax = 4
L1_min = 0
L1_max = 1
L2_min = 0
L2_max = 1
L1_size = L1_max - L1_min
L2_size = L2_max - L2_min

# Color range settings for visualization
# When USE_FIXED_COLOR_RANGE is False, colors scale dynamically with data (original behavior)
# When True, uses fixed VISUALIZATION_VMIN/VMAX for consistent colorbar
USE_FIXED_COLOR_RANGE = False
USE_3D_PLOT = False
VISUALIZATION_VMIN = 0.0
VISUALIZATION_VMAX = 0.1

# Create the phi function to match agent_node.py
from src.ergodic_exploration.ergodic_exploration.agent_node import createPhiFunc
phi_func = createPhiFunc(L1_BOUNDS=[L1_min, L1_max], L2_BOUNDS=[L2_min, L2_max])
# phi_func = lambda s: 1/100  # Uniform distribution for testing

base = Basis(L1_BOUNDS=[L1_min, L1_max], L2_BOUNDS=[L2_min, L2_max], Kmax=Kmax, phi_=phi_func, precalc_phik_coeff=True, num_gauss_points=22)
base.phi_rec = ReconstructedPhi(base, precalc_phik=False)

# =============----------------------------

def calculate_phi_integral(phi_function, bounds=None, num_points=100):
    """
    Calculate the numerical integral of a phi function over its domain using Simpson's rule.
    
    Args:
        phi_function: Function to integrate, should accept [x1, x2] as input
        bounds: [L1_min, L1_max, L2_min, L2_max] or None to use global bounds
        num_points: Number of grid points per dimension for integration
    
    Returns:
        float: The numerical integral value
    """
    if bounds is None:
        bounds = [L1_min, L1_max, L2_min, L2_max]
    
    L1_min_local, L1_max_local, L2_min_local, L2_max_local = bounds
    
    # Create grid points
    x1_points = np.linspace(L1_min_local, L1_max_local, num_points)
    x2_points = np.linspace(L2_min_local, L2_max_local, num_points)
    
    # Calculate step sizes
    dx1 = (L1_max_local - L1_min_local) / (num_points - 1)
    dx2 = (L2_max_local - L2_min_local) / (num_points - 1)
    
    # Evaluate function on grid
    Z = np.zeros((num_points, num_points))
    for i, x1 in enumerate(x1_points):
        for j, x2 in enumerate(x2_points):
            try:
                Z[j, i] = phi_function([x1, x2])
            except Exception as e:
                print(f"Error evaluating phi function at [{x1}, {x2}]: {e}")
                Z[j, i] = 0.0
    
    # Numerical integration using trapezoidal rule (more robust than Simpson's for arbitrary grid sizes)
    integral = np.trapz([np.trapz(row, x1_points) for row in Z], x2_points)
    
    return integral

def plotPhi(phi_rec_from_ck, phi_rec_from_agent, phi_rec_from_ros_ck, all_traj=None, grid_res=50, clip_to_min_max=False, ergodic_cost=None):
    phi_original = base.phi

    x1 = np.linspace(base.L1_min, base.L1_max, grid_res)
    x2 = np.linspace(base.L2_min, base.L2_max, grid_res)

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
    im1 = ax1.imshow(Z_original, extent=(base.L1_min, base.L1_max, base.L2_min, base.L2_max), origin='lower', cmap=cm.viridis)
    ax1.set_title('Original Function Φ')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_aspect('auto')
    plt.colorbar(im1, ax=ax1, label='Function Value')

    ax2 = fig.add_subplot(132)
    im2 = ax2.imshow(Z_agent_fourier_rec,
                     extent=(base.L1_min, base.L1_max, base.L2_min, base.L2_max), origin='lower', cmap=cm.viridis)
    ax2.set_title(f'Fourier Reconstruction (Kmax = {base.Kmax})')
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
        im3 = ax3.imshow(Z_rec_from_ros_ck, extent=(base.L1_min, base.L1_max, base.L2_min, base.L2_max), 
                        origin='lower', cmap=cm.viridis, vmin=min_val, vmax=max_val)
    else:
        im3 = ax3.imshow(Z_rec_from_ros_ck, extent=(base.L1_min, base.L1_max, base.L2_min, base.L2_max), 
                        origin='lower', cmap=cm.viridis)
    ax3.set_title('Reconstructed from ROS Ck (ck_values_average_in_range)')
    ax3.set_xlabel('x1')
    ax3.set_ylabel('x2')
    ax3.set_aspect('auto')
    # x and y lims to base.L1_min -> base.L1_max and base.L2_min -> base.L2_max
    ax3.set_xlim(base.L1_min, base.L1_max)
    ax3.set_ylim(base.L2_min, base.L2_max)
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
    x1 = np.linspace(base.L1_min, base.L1_max, grid_res)
    x2 = np.linspace(base.L2_min, base.L2_max, grid_res)

    Z_rec_from_ros_ck = np.zeros((len(x1), len(x2)))

    for i in range(len(x1)):
        for j in range(len(x2)):
            Z_rec_from_ros_ck[j, i] = phi_rec_from_ros_ck([x1[i], x2[j]])
        
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    im = ax.imshow(Z_rec_from_ros_ck, extent=(base.L1_min, base.L1_max, base.L2_min, base.L2_max), origin='lower', cmap='viridis')
    ax.set_title(f'Agent {agent_id} - ROS Ck (ck_values_average_in_range)')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_aspect('equal')
    ax.set_xlim(base.L1_min, base.L1_max)
    ax.set_ylim(base.L2_min, base.L2_max)
    plt.colorbar(im, ax=ax, label='Function Value')
    
    # Add ergodic cost as text annotation
    if ergodic_cost is not None:
        ax.text(0.02, 0.98, f'Ergodic Cost: {ergodic_cost:.4f}', 
                transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                fontsize=12, verticalalignment='top')
    
    plt.tight_layout()


class CkSubscriber(Node):
    """ROS subscriber to listen to CkTable messages from a specific agent"""
    
    def __init__(self, agent_id):
        super().__init__(f'ck_visualizer_{agent_id}')
        self.agent_id = agent_id
        self.latest_ck_values_average = None
        self.latest_ck_values = None
        self.ergodic_cost = 0
        self.erg_cost_reduction_perc = 0.0
        self.l_bounds_initiated_flag = False  # Flag to check if l_bounds have been set
        self.bounds_changed_flag = False  # Flag to indicate bounds have changed

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
            if not self.l_bounds_initiated_flag:
                # Lets set l_bounds
                l_bounds = msg.l_bounds
                kmax = int(msg.table_size - 1)
                # Create the phi function to match agent_node.py
                phi_func = createPhiFunc(L1_BOUNDS=[l_bounds[0], l_bounds[1]], L2_BOUNDS=[l_bounds[2], l_bounds[3]])
                # phi_func = lambda s: 1/100  # Uniform distribution for testing

                base.__init__(L1_BOUNDS=[l_bounds[0], l_bounds[1]], L2_BOUNDS=[l_bounds[2], l_bounds[3]], Kmax=kmax, phi_=phi_func, precalc_phik_coeff=True, num_gauss_points=22)
                base.phi_rec = ReconstructedPhi(base, precalc_phik=False)
                print(f"Changing Bounds to : {l_bounds} \tand \tKmax : {kmax}")
                self.l_bounds_initiated_flag = True
                self.bounds_changed_flag = True  # Set flag to indicate bounds changed

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
        self.x1 = np.linspace(base.L1_min, base.L1_max, grid_res)
        self.x2 = np.linspace(base.L2_min, base.L2_max, grid_res)
        self.setup_plots()
        
    def setup_plots(self):
        if USE_3D_PLOT:
            if self.plot_mode == 'ros-only':
                self.setup_ros_only_plot_3d()
            else:
                self.setup_all_plots_3d()
        else:
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
        self.ax3.set_xlim(base.L1_min, base.L1_max)
        self.ax3.set_ylim(base.L2_min, base.L2_max)

        # Initialize empty plot
        self.Z_rec_from_ros_ck = np.zeros((len(self.x1), len(self.x2)))
        
        # Set color range based on configuration
        if USE_FIXED_COLOR_RANGE:
            self.vmin, self.vmax = VISUALIZATION_VMIN, VISUALIZATION_VMAX
            self.im3 = self.ax3.imshow(self.Z_rec_from_ros_ck, extent=(base.L1_min, base.L1_max, base.L2_min, base.L2_max), 
                                      origin='lower', cmap='viridis', vmin=self.vmin, vmax=self.vmax)
        else:
            # Use dynamic color range (original behavior)
            self.im3 = self.ax3.imshow(self.Z_rec_from_ros_ck, extent=(base.L1_min, base.L1_max, base.L2_min, base.L2_max), 
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
        im1 = self.ax1.imshow(Z_original, extent=(base.L1_min, base.L1_max, base.L2_min, base.L2_max), origin='lower', cmap='viridis')
        self.ax1.set_title('Original Function Φ')
        self.ax1.set_xlabel('x1')
        self.ax1.set_ylabel('x2')
        self.ax1.set_aspect('auto')
        plt.colorbar(im1, ax=self.ax1, label='Function Value')
        
        # Fourier reconstruction plot (static)
        self.ax2 = self.fig.add_subplot(132)
        im2 = self.ax2.imshow(Z_agent_fourier_rec, extent=(base.L1_min, base.L1_max, base.L2_min, base.L2_max), origin='lower', cmap='viridis')
        self.ax2.set_title(f'Fourier Reconstruction (Kmax = {base.Kmax})')
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
        self.ax3.set_xlim(base.L1_min, base.L1_max)
        self.ax3.set_ylim(base.L2_min, base.L2_max)

        # Initialize empty plot
        self.Z_rec_from_ros_ck = np.zeros((len(self.x1), len(self.x2)))
        
        # Set color range based on configuration
        if USE_FIXED_COLOR_RANGE:
            self.vmin, self.vmax = VISUALIZATION_VMIN, VISUALIZATION_VMAX
            self.im3 = self.ax3.imshow(self.Z_rec_from_ros_ck, extent=(base.L1_min, base.L1_max, base.L2_min, base.L2_max), 
                                      origin='lower', cmap='viridis', vmin=self.vmin, vmax=self.vmax)
        else:
            # Use dynamic color range (original behavior)
            self.im3 = self.ax3.imshow(self.Z_rec_from_ros_ck, extent=(base.L1_min, base.L1_max, base.L2_min, base.L2_max), 
                                      origin='lower', cmap='viridis')
        self.cbar3 = plt.colorbar(self.im3, ax=self.ax3, label='Function Value')
        
        # Initialize ergodic cost text (will be updated in animation)
        self.ergodic_cost_text = self.ax3.text(0.02, 0.98, 'Ergodic Cost: N/A', 
                                              transform=self.ax3.transAxes, 
                                              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                                              fontsize=10, verticalalignment='top')
        
        plt.tight_layout()
        
    def update_plot_bounds(self):
        """Update plot bounds and grid when base bounds change"""
        # Recalculate grid points with new bounds
        self.x1 = np.linspace(self.base.L1_min, self.base.L1_max, self.grid_res)
        self.x2 = np.linspace(self.base.L2_min, self.base.L2_max, self.grid_res)
        
        # Update Z matrix size
        self.Z_rec_from_ros_ck = np.zeros((len(self.x1), len(self.x2)))
        
        if self.plot_mode == 'ros-only':
            # Update axis limits
            self.ax3.set_xlim(self.base.L1_min, self.base.L1_max)
            self.ax3.set_ylim(self.base.L2_min, self.base.L2_max)
            
            # Update image extent
            if hasattr(self, 'im3'):
                self.im3.set_extent((self.base.L1_min, self.base.L1_max, self.base.L2_min, self.base.L2_max))
            
            # Force axis to recalculate ticks and labels
            self.ax3.relim()
            self.ax3.autoscale_view()
            
            # Manually set nice tick locations
            self.ax3.locator_params(axis='x', nbins=6)
            self.ax3.locator_params(axis='y', nbins=6)
            
        else:
            # Update all three plots for 'all' mode
            for ax in [self.ax1, self.ax2, self.ax3]:
                ax.set_xlim(self.base.L1_min, self.base.L1_max)
                ax.set_ylim(self.base.L2_min, self.base.L2_max)
                
                # Force each axis to recalculate ticks and labels
                ax.relim()
                ax.autoscale_view()
                ax.locator_params(axis='x', nbins=6)
                ax.locator_params(axis='y', nbins=6)
            
            # Update image extents for all plots
            if hasattr(self, 'im3'):
                self.im3.set_extent((self.base.L1_min, self.base.L1_max, self.base.L2_min, self.base.L2_max))
        
        # Force a redraw of the figure
        if hasattr(self, 'fig') and self.fig is not None:
            self.fig.canvas.draw_idle()
        
        print(f"Plot bounds updated to: x1=[{self.base.L1_min}, {self.base.L1_max}], x2=[{self.base.L2_min}, {self.base.L2_max}]")
    
    def setup_ros_only_plot_3d(self):
        """Setup single 3D surface plot showing only ROS Ck visualization"""
        # Create figure with 3D subplot
        self.fig = plt.figure(self.agent_id, figsize=(10, 8))
        self.ax3 = self.fig.add_subplot(111, projection='3d')
        
        self.ax3.set_title(f'Agent {self.agent_id} - ROS Ck 3D Surface (ck_values_average_in_range)')
        self.ax3.set_xlabel('x1')
        self.ax3.set_ylabel('x2')
        self.ax3.set_zlabel('Function Value')

        # Create meshgrid for 3D plotting
        self.X, self.Y = np.meshgrid(self.x1, self.x2)
        self.Z_rec_from_ros_ck = np.zeros((len(self.x1), len(self.x2)))
        
        # Create initial 3D surface
        if USE_FIXED_COLOR_RANGE:
            self.vmin, self.vmax = VISUALIZATION_VMIN, VISUALIZATION_VMAX
            self.surf3 = self.ax3.plot_surface(self.X, self.Y, self.Z_rec_from_ros_ck, 
                                              cmap='viridis', vmin=self.vmin, vmax=self.vmax,
                                              alpha=0.8, antialiased=True)
            # Set z-axis limits to match color range for consistent visualization
            self.ax3.set_zlim(self.vmin, self.vmax)
        else:
            self.surf3 = self.ax3.plot_surface(self.X, self.Y, self.Z_rec_from_ros_ck, 
                                              cmap='viridis', alpha=0.8, antialiased=True)
        
        # Add colorbar
        self.cbar3 = plt.colorbar(self.surf3, ax=self.ax3, label='Function Value', shrink=0.6)
        
        # Initialize ergodic cost text
        self.ergodic_cost_text = self.ax3.text2D(0.02, 0.98, 'Ergodic Cost: N/A', 
                                                 transform=self.ax3.transAxes, 
                                                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                                                 fontsize=12, verticalalignment='top')
        
        plt.tight_layout()
        
    def setup_all_plots_3d(self):
        """Setup all plots with 3D surface for ROS Ck visualization"""
        # Calculate static data for first two plots
        Z_original = np.zeros((len(self.x1), len(self.x2)))
        Z_agent_fourier_rec = np.zeros((len(self.x1), len(self.x2)))
        
        for i in range(len(self.x1)):
            for j in range(len(self.x2)):
                Z_original[j, i] = self.base.phi([self.x1[i], self.x2[j]])
                Z_agent_fourier_rec[j, i] = self.phi_rec([self.x1[i], self.x2[j]])
        
        # Create figure with mixed 2D and 3D plots
        self.fig = plt.figure(figsize=(20, 6))
        
        # Original function plot (2D)
        self.ax1 = self.fig.add_subplot(131)
        im1 = self.ax1.imshow(Z_original, extent=(base.L1_min, base.L1_max, base.L2_min, base.L2_max), origin='lower', cmap='viridis')
        self.ax1.set_title('Original Function Φ')
        self.ax1.set_xlabel('x1')
        self.ax1.set_ylabel('x2')
        self.ax1.set_aspect('auto')
        plt.colorbar(im1, ax=self.ax1, label='Function Value')
        
        # Fourier reconstruction plot (2D)
        self.ax2 = self.fig.add_subplot(132)
        im2 = self.ax2.imshow(Z_agent_fourier_rec, extent=(base.L1_min, base.L1_max, base.L2_min, base.L2_max), origin='lower', cmap='viridis')
        self.ax2.set_title(f'Fourier Reconstruction (Kmax = {base.Kmax})')
        self.ax2.set_xlabel('x1')
        self.ax2.set_ylabel('x2')
        self.ax2.set_aspect('auto')
        plt.colorbar(im2, ax=self.ax2, label='Function Value')
        
        # Dynamic 3D surface plot - ROS ck_values_average_in_range
        self.ax3 = self.fig.add_subplot(133, projection='3d')
        self.ax3.set_title(f'Real-time 3D Surface from Agent {self.agent_id}')
        self.ax3.set_xlabel('x1')
        self.ax3.set_ylabel('x2')
        self.ax3.set_zlabel('Function Value')

        # Create meshgrid and initialize 3D surface
        self.X, self.Y = np.meshgrid(self.x1, self.x2)
        self.Z_rec_from_ros_ck = np.zeros((len(self.x1), len(self.x2)))
        
        if USE_FIXED_COLOR_RANGE:
            self.vmin, self.vmax = VISUALIZATION_VMIN, VISUALIZATION_VMAX
            self.surf3 = self.ax3.plot_surface(self.X, self.Y, self.Z_rec_from_ros_ck, 
                                              cmap='viridis', vmin=self.vmin, vmax=self.vmax,
                                              alpha=0.8, antialiased=True)
            # Set z-axis limits to match color range for consistent visualization
            self.ax3.set_zlim(self.vmin, self.vmax)
        else:
            self.surf3 = self.ax3.plot_surface(self.X, self.Y, self.Z_rec_from_ros_ck, 
                                              cmap='viridis', alpha=0.8, antialiased=True)
        
        # Add colorbar
        self.cbar3 = plt.colorbar(self.surf3, ax=self.ax3, label='Function Value', shrink=0.6)
        
        # Initialize ergodic cost text
        self.ergodic_cost_text = self.ax3.text2D(0.02, 0.98, 'Ergodic Cost: N/A', 
                                                 transform=self.ax3.transAxes, 
                                                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                                                 fontsize=10, verticalalignment='top')
        
        plt.tight_layout()
    
    def update_plot(self, frame):
        """Update function for animation"""
        # Check for shutdown request
        if shutdown_manager.shutdown_requested.is_set() or not self.running:
            self.stop()
            # Return empty list for both cases since we're shutting down
            return []
        
        # Check if bounds have changed and update plot if needed
        if hasattr(self.ck_subscriber, 'bounds_changed_flag') and self.ck_subscriber.bounds_changed_flag:
            self.update_plot_bounds()
            self.ck_subscriber.bounds_changed_flag = False  # Reset flag
            
        # Check if we have new ck data from ROS
        if self.ck_subscriber.latest_ck_values_average is None:
            # No data available, return appropriate value based on blitting mode
            if USE_3D_PLOT:
                return []  # 3D mode uses blit=False
            else:
                return [self.im3, self.ergodic_cost_text] if (hasattr(self, 'im3') and hasattr(self, 'ergodic_cost_text')) else []
        
        # Use ck_values_average_in_range for visualization
        ck_values = self.ck_subscriber.latest_ck_values_average
        
        # Calculate ergodic cost
        try:
            ergodic_cost = self.ck_subscriber.ergodic_cost
            # Update the text display on the plot
            if hasattr(self, 'ergodic_cost_text'):
                self.ergodic_cost_text.set_text(f'Ergodic Cost: {ergodic_cost:.4f} (-> {100 * self.ck_subscriber.erg_cost_reduction_perc:.2f}%)')
        except Exception as e:
            print(f"Error calculating ergodic cost: {e}")
            if hasattr(self, 'ergodic_cost_text'):
                self.ergodic_cost_text.set_text('Ergodic Cost: Error')
        
        # Create new phi reconstruction
        try:
            phi_rec_from_ros_ck = ReconstructedPhiFromCk(self.base, ck_values)
            
            # Update Z matrix
            for i in range(len(self.x1)):
                for j in range(len(self.x2)):
                    self.Z_rec_from_ros_ck[j, i] = phi_rec_from_ros_ck([self.x1[i], self.x2[j]])
            
            if USE_3D_PLOT:
                # Update 3D surface
                self.ax3.clear()
                
                # Recreate the surface with updated data
                if USE_FIXED_COLOR_RANGE:
                    self.surf3 = self.ax3.plot_surface(self.X, self.Y, self.Z_rec_from_ros_ck, 
                                                      cmap='viridis', vmin=self.vmin, vmax=self.vmax,
                                                      alpha=0.8, antialiased=True)
                    # Set z-axis limits to match color range for consistent visualization
                    self.ax3.set_zlim(self.vmin, self.vmax)
                else:
                    self.surf3 = self.ax3.plot_surface(self.X, self.Y, self.Z_rec_from_ros_ck, 
                                                      cmap='viridis', alpha=0.8, antialiased=True)
                
                # Reset axes properties
                if self.plot_mode == 'ros-only':
                    self.ax3.set_title(f'Agent {self.agent_id} - ROS Ck 3D Surface (ck_values_average_in_range)')
                else:
                    self.ax3.set_title(f'Real-time 3D Surface from Agent {self.agent_id}')
                self.ax3.set_xlabel('x1')
                self.ax3.set_ylabel('x2')
                self.ax3.set_zlabel('Function Value')
                
            else:
                # Update 2D image data
                self.im3.set_array(self.Z_rec_from_ros_ck)
                
                # Apply color scaling based on configuration
                if not USE_FIXED_COLOR_RANGE:
                    # Dynamic color range (original behavior)
                    self.im3.set_clim(vmin=self.Z_rec_from_ros_ck.min(), vmax=self.Z_rec_from_ros_ck.max())
            
        except Exception as e:
            print(f"Error updating plot: {e}")
        
        # Return animation objects based on plot type and blitting mode
        if USE_3D_PLOT:
            # 3D mode uses blit=False, so return value is not needed
            return []
        else:
            # 2D mode uses blit=True, so return the artists that need to be redrawn
            return [self.im3, self.ergodic_cost_text] if (hasattr(self, 'im3') and self.im3 is not None and hasattr(self, 'ergodic_cost_text')) else []
    
    def on_key_press(self, event):
        """Handle keyboard events"""
        if event.key in ['q', 'Q', 'escape']:
            if self.running:  # Only trigger shutdown once
                print("\nKey pressed. Shutting down gracefully...")
                shutdown_manager.shutdown_requested.set()
                self.stop()
        elif event.key in ['i', 'I']:
            # Calculate and print the integral of the reconstructed function
            if (hasattr(self.ck_subscriber, 'latest_ck_values_average') and 
                self.ck_subscriber.latest_ck_values_average is not None):
                try:
                    print("\nCalculating integral of reconstructed function...")
                    # Create phi reconstruction from current ROS data
                    phi_rec_from_ros_ck = ReconstructedPhiFromCk(self.base, self.ck_subscriber.latest_ck_values_average)
                    
                    # Calculate the integral over the domain
                    bounds = [self.base.L1_min, self.base.L1_max, self.base.L2_min, self.base.L2_max]
                    integral_value = calculate_phi_integral(phi_rec_from_ros_ck, bounds=bounds, num_points=100)
                    
                    print(f"\n=== INTEGRAL CALCULATION ===")
                    print(f"Agent {self.agent_id} - Reconstructed Function Integral:")
                    print(f"Domain: x1 ∈ [{self.base.L1_min:.3f}, {self.base.L1_max:.3f}], x2 ∈ [{self.base.L2_min:.3f}, {self.base.L2_max:.3f}]")
                    print(f"Integral value: {integral_value:.6f}")
                    print(f"Domain area: {(self.base.L1_max - self.base.L1_min) * (self.base.L2_max - self.base.L2_min):.6f}")
                    print(f"Average function value: {integral_value / ((self.base.L1_max - self.base.L1_min) * (self.base.L2_max - self.base.L2_min)):.6f}")
                    print("============================\n")
                    
                except Exception as e:
                    print(f"\nError calculating integral: {e}\n")
            else:
                print("\nNo ROS data available for integral calculation. Please wait for data...\n")
    
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
                # Disable blitting for 3D plots as it causes AttributeError with Poly3DCollection
                use_blit = not USE_3D_PLOT
                self.ani = FuncAnimation(self.fig, self.update_plot, interval=self.update_interval, 
                                       blit=use_blit, cache_frame_data=False, repeat=True)
                if USE_3D_PLOT:
                    print("Animation started successfully (3D mode - blitting disabled). Use Ctrl+C or 'Q' to exit gracefully.")
                else:
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
        visualizer = RealTimeVisualizer(base, base.phi_rec, agent_id, plot_mode, grid_res=50, update_interval=500)
        
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
        print("Press 'I' key to calculate and print the integral of the reconstructed function")
        
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
        phi_rec_from_ros_ck = ReconstructedPhiFromCk(base, ck_values_average)
        
        print(f"Received ck data from agent_{agent_id}. Creating static visualization...")
        print("Close the window or press Ctrl+C to exit")
        print("Press 'I' key to calculate and print the integral of the reconstructed function")
        
        # Note: 3D plotting is currently only supported in real-time mode
        if USE_3D_PLOT:
            print("Warning: 3D plotting is currently only supported in real-time mode. Using 2D for static visualization.")
        
        if plot_mode == 'ros-only':
            # Show only the ROS Ck plot
            plot_ros_only(phi_rec_from_ros_ck, agent_id, ergodic_cost=ck_subscriber.ergodic_cost)
        else:
            # Show all three plots
            plotPhi(phi_rec_from_ck=phi_rec_from_ros_ck, 
                    phi_rec_from_agent=base.phi_rec, 
                    phi_rec_from_ros_ck=phi_rec_from_ros_ck,
                    ergodic_cost=ck_subscriber.ergodic_cost)

        # Set up graceful shutdown for static plot
        def on_key_press(event):
            if event.key in ['q', 'Q', 'escape']:
                print("\nKey pressed. Closing...")
                plt.close('all')
            elif event.key in ['i', 'I']:
                # Calculate and print the integral of the reconstructed function
                try:
                    print("\nCalculating integral of reconstructed function...")
                    # Calculate the integral over the domain
                    bounds = [base.L1_min, base.L1_max, base.L2_min, base.L2_max]
                    integral_value = calculate_phi_integral(phi_rec_from_ros_ck, bounds=bounds, num_points=100)
                    
                    print(f"\n=== INTEGRAL CALCULATION ===")
                    print(f"Agent {agent_id} - Reconstructed Function Integral:")
                    print(f"Domain: x1 ∈ [{base.L1_min:.3f}, {base.L1_max:.3f}], x2 ∈ [{base.L2_min:.3f}, {base.L2_max:.3f}]")
                    print(f"Integral value: {integral_value:.6f}")
                    print(f"Domain area: {(base.L1_max - base.L1_min) * (base.L2_max - base.L2_min):.6f}")
                    print(f"Average function value: {integral_value / ((base.L1_max - base.L1_min) * (base.L2_max - base.L2_min)):.6f}")
                    print("============================\n")
                    
                except Exception as e:
                    print(f"\nError calculating integral: {e}\n")
        
        def on_close(event):
            print("\nWindow closed.")
        
        # Connect event handlers
        fig = plt.gcf()
        fig.canvas.mpl_connect('key_press_event', on_key_press)
        fig.canvas.mpl_connect('close_event', on_close)
        fig.suptitle(f'Agent {agent_id} - Static Visualization (Press Q to exit, I for integral)', 
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
    parser.add_argument('--color-range', nargs=2, type=float, metavar=('MIN', 'MAX'),
                       help='Use fixed color range for consistent colorbar and 3D z-axis limits (default: dynamic auto-scaling). Example: --color-range 0 0.05')
    parser.add_argument('--3d', action='store_true',
                       help='Use 3D surface plot instead of 2D heatmap for visualization')
    
    try:
        args = parser.parse_args()
        
        # Update color range if specified
        if args.color_range:
            global USE_FIXED_COLOR_RANGE, VISUALIZATION_VMIN, VISUALIZATION_VMAX
            USE_FIXED_COLOR_RANGE = True
            VISUALIZATION_VMIN, VISUALIZATION_VMAX = args.color_range
            print(f"Using fixed color range: {VISUALIZATION_VMIN} to {VISUALIZATION_VMAX}")
        else:
            print("Using dynamic color range (default behavior)")
        
        # Update 3D plotting mode if specified
        if getattr(args, '3d', False):
            global USE_3D_PLOT
            USE_3D_PLOT = True
            print("Using 3D surface visualization")
        else:
            print("Using 2D heatmap visualization")
        
        print(f"Starting visualization for agent_{args.agent_id}")
        print(f"Mode: {args.mode}")
        print(f"Plot mode: {args.plot_mode}")
        print(f"Visualization: {'3D surface' if USE_3D_PLOT else '2D heatmap'}")
        if USE_FIXED_COLOR_RANGE:
            print(f"Color range: {VISUALIZATION_VMIN} to {VISUALIZATION_VMAX} (fixed)")
        else:
            print("Color range: dynamic (auto-scaling)")
        print("Make sure the agent is running and publishing CkTable messages!")
        print("\nVisualization Options:")
        print("  Default: 2D heatmap visualization")
        print("  --3d: 3D surface visualization")
        print("  --color-range MIN MAX: Fixed range (consistent colorbar and 3D z-limits)")
        print("  Default color: Dynamic scaling (colorbar changes with data)")
        print("\nGraceful shutdown: Press Ctrl+C or 'Q' key to exit")
        print("Integral calculation: Press 'I' key to print the integral of the reconstructed function")
        
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