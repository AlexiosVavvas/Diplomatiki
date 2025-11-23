import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
# import pandas as pd
from matplotlib.widgets import Button
from matplotlib.patches import Circle, Rectangle, Polygon
import os
import colorsys
import argparse
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from my_interfaces.msg import AgentData, MultipleObstacles, SingleObstacle, MultipleTargetEstimates, SingleTargetEstimate, CkTable  # type: ignore
from std_msgs.msg import Header
import threading
import time
import copy
import re
from matplotlib.patches import Ellipse

# Here used only to set axis limits for visualization
# Those are the initial bounds, they are about to change when agents arrive
L1_BOUNDS = [0.0, 1.0]
L2_BOUNDS = [0.0, 1.0]

VISUALISE_VIRTUAL_OBS = False # Whether to visualize virtual obstacles in the dashboard

# Thread-safe lock for bounds updates
BOUNDS_LOCK = threading.Lock()

# Global flag for blue first agent
FORCE_BLUE_FIRST = False

def generateAgentColor(agent_id, max_agents=10, force_blue_first=False):
    """Generate a distinct color for each agent using HSV color space."""
    # If force_blue_first is True and this is agent 1 (assuming 1 is the first agent), make it blue
    if force_blue_first and agent_id == 1:
        return (0.0, 0.0, 0.7)  # Pure blue
    
    # Use golden angle to distribute colors evenly around the color wheel
    hue = (agent_id * 137.5) % 360  # Golden angle: 360 * (3 - sqrt(5)) / 2
    saturation = 0.8
    value = 0.9
    
    # Convert HSV to RGB
    r, g, b = colorsys.hsv_to_rgb(hue / 360.0, saturation, value)
    
    return (r, g, b)  # Return as tuple for matplotlib

def getAgentColorRgb255(agent_id, max_agents=10, force_blue_first=False):
    """Get RGB color values in 0-255 range for an agent."""
    r, g, b = generateAgentColor(agent_id, max_agents, force_blue_first)
    return (int(r * 255), int(g * 255), int(b * 255))

def createColoredBox(r, g, b, text=""):
    """Create a colored box using ANSI escape codes."""
    # ANSI escape code for background color
    return f"\033[48;2;{r};{g};{b}m  \033[0m {text}"

def createColoredText(r, g, b, text):
    """Create colored text using ANSI escape codes."""
    # ANSI escape code for foreground color
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"

def updateGlobalBounds(l_bounds_list):
    """Update global L1_BOUNDS and L2_BOUNDS based on all agents' bounds"""
    global L1_BOUNDS, L2_BOUNDS
    
    if not l_bounds_list:
        return
    
    with BOUNDS_LOCK:
        # Initialize with first agent's bounds or keep current if no agents
        if len(l_bounds_list) > 0:
            # Find the minimum of all l1_min and l2_min, maximum of all l1_max and l2_max
            l1_mins = [bounds[0] for bounds in l_bounds_list if len(bounds) >= 4]
            l1_maxs = [bounds[1] for bounds in l_bounds_list if len(bounds) >= 4]
            l2_mins = [bounds[2] for bounds in l_bounds_list if len(bounds) >= 4]
            l2_maxs = [bounds[3] for bounds in l_bounds_list if len(bounds) >= 4]
            
            if l1_mins and l1_maxs and l2_mins and l2_maxs:
                new_l1_bounds = [min(l1_mins), max(l1_maxs)]
                new_l2_bounds = [min(l2_mins), max(l2_maxs)]
                
                # Only update if bounds have changed significantly (avoid tiny fluctuations)
                if (abs(new_l1_bounds[0] - L1_BOUNDS[0]) > 1e-6 or 
                    abs(new_l1_bounds[1] - L1_BOUNDS[1]) > 1e-6 or
                    abs(new_l2_bounds[0] - L2_BOUNDS[0]) > 1e-6 or 
                    abs(new_l2_bounds[1] - L2_BOUNDS[1]) > 1e-6):
                    
                    L1_BOUNDS = new_l1_bounds
                    L2_BOUNDS = new_l2_bounds
                    
                    print(f"Updated global bounds: L1=[{L1_BOUNDS[0]:.3f}, {L1_BOUNDS[1]:.3f}], L2=[{L2_BOUNDS[0]:.3f}, {L2_BOUNDS[1]:.3f}]")

class LiveDashboard(Node):
    def __init__(self, allowed_agents=None, max_plot_points=1000, ekf_agent=None, top_view_only=False, pos_inverted_agents=None):
        super().__init__('dashboard_node')
        
        # Store allowed agents filter
        self.allowed_agents = set(allowed_agents) if allowed_agents else None
        
        # Store EKF agent ID for target position estimation plot
        self.ekf_agent = ekf_agent
        
        # Store max plot points for visualization
        self.max_plot_points = max_plot_points
        
        # Store top view only flag
        self.top_view_only = top_view_only
        
        # Store position inverted agents (NED convention agents like airplanes)
        self.pos_inverted_agents = set(pos_inverted_agents) if pos_inverted_agents else set()
        
        # Create QoS profile for best effort communication
        # This allows for faster data transmission with potential message loss
        self.best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE
        )
        
        # Create separate figure windows for each plot
        self.traj_fig, self.traj_ax = plt.subplots(1, 1, figsize=(9, 7))
        self.traj_fig.suptitle('Agent Trajectories', fontsize=14)
        
        # Only create additional plots if not in top_view_only mode
        if not self.top_view_only:
            self.ergodic_fig, self.ergodic_ax = plt.subplots(1, 1, figsize=(10, 6))
            self.ergodic_fig.suptitle('Ergodic Cost', fontsize=14)
            
            self.control_fig, self.control_ax = plt.subplots(1, 1, figsize=(10, 6))
            self.control_fig.suptitle('Control Inputs', fontsize=14)
            
            self.delta_t_fig, self.delta_t_ax = plt.subplots(1, 1, figsize=(10, 6))
            self.delta_t_fig.suptitle('Delta T Timestamps', fontsize=14)
            
            # Create EKF plot if EKF agent is specified
            if self.ekf_agent is not None:
                self.ekf_fig, self.ekf_axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
                self.ekf_fig.suptitle(f'Agent {self.ekf_agent} - Target Position Estimates with 3σ Confidence Bands', fontsize=14)
            else:
                self.ekf_fig = None
                self.ekf_axes = None
        else:
            # Set all other plots to None in top_view_only mode
            self.ergodic_fig = None
            self.ergodic_ax = None
            self.control_fig = None
            self.control_ax = None
            self.delta_t_fig = None
            self.delta_t_ax = None
            self.ekf_fig = None
            self.ekf_axes = None
        
        # Auto-refresh flag
        self.auto_refresh = True
        self._shutdown_requested = False  # Flag to track shutdown state
        
        # Data storage for each agent (dynamic)
        self.agent_data = {}
        self.data_lock = threading.Lock()  # Thread safety lock
        
        # Obstacle data storage (shared across all agents)
        self.obstacles_data = {}
        self.obstacles_lock = threading.Lock()
        
        # Target estimates data storage (shared across all agents)
        self.target_estimates_data = {}
        self.target_estimates_lock = threading.Lock()
        
        # Historical target estimates data for EKF plotting
        self.target_estimates_history = {}
        self.target_estimates_history_lock = threading.Lock()
        
        # CK data storage for each agent
        self.ck_data = {}
        self.ck_lock = threading.Lock()
        
        # Store active subscribers and discovered agents
        self.subscribers = {}
        self.obstacle_subscribers = {}
        self.target_estimate_subscribers = {}
        self.ck_subscribers = {}
        self.discovered_agents = set()
        
        # Timer for periodic agent discovery
        self.discovery_timer = self.create_timer(2.0, self.discoverAgents)  # Check every 2 seconds
        
        # Set up key press event on agent trajectory figure
        self.traj_fig.canvas.mpl_connect('key_press_event', self.onKeyPress)
        
        # Initialize animation with faster update rate
        # Use traj_fig for animation in top_view_only mode, otherwise use delta_t_fig
        animation_fig = self.traj_fig if self.top_view_only else self.delta_t_fig
        self.anim = animation.FuncAnimation(
            animation_fig, self.updatePlots, interval=50, blit=False, cache_frame_data=False
        )
        
        # Perform initial discovery
        self.discoverAgents()
        
        # Log filtering information
        if self.allowed_agents:
            self.get_logger().info(f'Dashboard initialized with agent filter: {sorted(self.allowed_agents)} - discovering all agents but only processing filtered ones')
        else:
            self.get_logger().info('Dashboard initialized - discovering all agents dynamically')
            
        if self.pos_inverted_agents:
            self.get_logger().info(f'Position inversion enabled for agents (NED convention): {sorted(self.pos_inverted_agents)}')
            
        if self.ekf_agent is not None:
            self.get_logger().info(f'EKF target position plot enabled for agent {self.ekf_agent}')

    def displayAgentColors(self):
        """Display discovered agents with their color mappings"""
        # Filter agents if allowed_agents is set
        agents_to_display = self.discovered_agents
        if self.allowed_agents:
            agents_to_display = self.discovered_agents.intersection(self.allowed_agents)
        
        if not agents_to_display:
            return
            
        print("\n" + "="*60)
        if self.allowed_agents:
            print("🎨 Agent Color Mappings (Dashboard & RViz consistent) - FILTERED")
        else:
            print("🎨 Agent Color Mappings (Dashboard & RViz consistent)")
        print("="*60)
        
        for agent_id in sorted(agents_to_display):
            r, g, b = getAgentColorRgb255(agent_id, force_blue_first=FORCE_BLUE_FIRST)
            
            # Create colored box and text
            colored_box = createColoredBox(r, g, b)
            colored_agent_text = createColoredText(r, g, b, f"Agent {agent_id}")
            
            print(f"  {colored_box} {colored_agent_text}: RGB({r}, {g}, {b})")
        
        if self.allowed_agents:
            filtered_out = self.discovered_agents - self.allowed_agents
            if filtered_out:
                print(f"  Discovered but filtered out: {sorted(filtered_out)}")
        
        print("="*60)
        print("Colors will appear consistently in both dashboard plots and RViz visualization")
        print("="*60 + "\n")

    def discoverAgents(self):
        """Discover active agent nodes and create subscribers for them"""
        if self._shutdown_requested:
            return
            
        try:
            # Get list of all nodes
            node_names = self.get_node_names()
            
            # Look for agent nodes (assuming they follow pattern like 'agent_X' or contain 'agent')
            agent_pattern = re.compile(r'agent[_\-]?(\d+)', re.IGNORECASE)
            current_agents = set()
            
            for node_name in node_names:
                match = agent_pattern.search(node_name)
                if match:
                    agent_id = int(match.group(1))
                    current_agents.add(agent_id)
                    
                    # Create subscriber if this is a new agent (regardless of filtering)
                    # We discover all agents but only process data for filtered ones
                    if agent_id not in self.discovered_agents:
                        self.createAgentSubscriber(agent_id)
            
            # Remove subscribers for agents that are no longer active
            inactive_agents = self.discovered_agents - current_agents
            for agent_id in inactive_agents:
                self.removeAgentSubscriber(agent_id)
            
            # Update discovered agents
            if current_agents != self.discovered_agents:
                if self.allowed_agents:
                    filtered_current = current_agents.intersection(self.allowed_agents)
                    self.get_logger().info(f'Active agents: {sorted(current_agents)}, filtered: {sorted(filtered_current)}')
                else:
                    self.get_logger().info(f'Active agents: {sorted(current_agents)}')
                self.discovered_agents = current_agents
                
                # Display colored agent mappings
                self.displayAgentColors()
        except Exception as e:
            # Don't log errors if the node is being shut down
            if not self._shutdown_requested and self.get_logger() is not None:
                self.get_logger().warn(f'Error during agent discovery: {e}')

    def createAgentSubscriber(self, agent_id):
        """Create a subscriber for a specific agent"""
        if self._shutdown_requested:
            return
            
        with self.data_lock:
            # Calculate the offset for the new agent based on the maximum simulation time
            # of all currently active agents
            max_sim_time = 0.0
            for existing_agent_id, existing_data in self.agent_data.items():
                if len(existing_data['simulation_times']) > 0:
                    # Get the latest unified simulation time from existing agents
                    latest_unified_time = existing_data['simulation_times'][-1]
                    max_sim_time = max(max_sim_time, latest_unified_time)
            
            # Initialize data storage for new agent
            self.agent_data[agent_id] = {
                'timestamps': [],
                'simulation_times': [],
                'simulation_time_offset': max_sim_time,  # This is the offset to add to this agent's simulation time
                'states': [],
                'inputs': [],
                'ergodic_costs': [],
                'cbf_flags': [],
                'in_range_agents_ids': [],
                'delta_t_ts': []  # Store delta_t_ts from AgentData messages
            }
        
        with self.ck_lock:
            # Initialize CK data storage for new agent
                self.ck_data[agent_id] = {
                    'timestamps': [],
                    'ck_tables': [],
                    'table_size': None,
                    'total_erg_costs': [],
                    'l_bounds': []  # Store l_bounds for dynamic boundary updates
                }        # Create agent data subscriber
        subscriber = self.create_subscription(
            AgentData,
            f'agent_{agent_id}/data',
            lambda msg, aid=agent_id: self.agentDataCallback(msg, aid),
            self.best_effort_qos
        )
        self.subscribers[agent_id] = subscriber
        
        # Create obstacle data subscriber
        obstacle_subscriber = self.create_subscription(
            MultipleObstacles,
            f'agent_{agent_id}/known_obstacles',
            lambda msg, aid=agent_id: self.obstaclesCallback(msg, aid),
            self.best_effort_qos
        )
        self.obstacle_subscribers[agent_id] = obstacle_subscriber
        
        # Create target estimates subscriber
        target_subscriber = self.create_subscription(
            MultipleTargetEstimates,
            f'agent_{agent_id}/target_estimates',
            lambda msg, aid=agent_id: self.targetEstimatesCallback(msg, aid),
            self.best_effort_qos
        )
        self.target_estimate_subscribers[agent_id] = target_subscriber
        
        # Create CK data subscriber
        ck_subscriber = self.create_subscription(
            CkTable,
            f'agent_{agent_id}/ck',
            lambda msg, aid=agent_id: self.ckCallback(msg, aid),
            self.best_effort_qos
        )
        self.ck_subscribers[agent_id] = ck_subscriber
        
        # Log the offset being applied
        with self.data_lock:
            offset = self.agent_data[agent_id]['simulation_time_offset']
            self.get_logger().info(f'Created subscribers for agent {agent_id} with time offset: {offset:.3f}s')

    def removeAgentSubscriber(self, agent_id):
        """Remove subscriber and data for an inactive agent"""
        try:
            if agent_id in self.subscribers:
                # Destroy agent data subscriber
                try:
                    self.destroy_subscription(self.subscribers[agent_id])
                except Exception as e:
                    self.get_logger().warn(f'Error destroying agent subscriber for {agent_id}: {e}')
                del self.subscribers[agent_id]
                
            if agent_id in self.obstacle_subscribers:
                # Destroy obstacle subscriber
                try:
                    self.destroy_subscription(self.obstacle_subscribers[agent_id])
                except Exception as e:
                    self.get_logger().warn(f'Error destroying obstacle subscriber for {agent_id}: {e}')
                del self.obstacle_subscribers[agent_id]
                
                # Remove obstacle data for this agent
                with self.obstacles_lock:
                    if agent_id in self.obstacles_data:
                        del self.obstacles_data[agent_id]
            
            if agent_id in self.target_estimate_subscribers:
                # Destroy target estimates subscriber
                try:
                    self.destroy_subscription(self.target_estimate_subscribers[agent_id])
                except Exception as e:
                    self.get_logger().warn(f'Error destroying target estimate subscriber for {agent_id}: {e}')
                del self.target_estimate_subscribers[agent_id]
                
                # Remove target estimates data for this agent
                with self.target_estimates_lock:
                    if agent_id in self.target_estimates_data:
                        del self.target_estimates_data[agent_id]
                
                # Remove historical target estimates data for this agent
                with self.target_estimates_history_lock:
                    if agent_id in self.target_estimates_history:
                        del self.target_estimates_history[agent_id]
            
            if agent_id in self.ck_subscribers:
                # Destroy CK subscriber
                try:
                    self.destroy_subscription(self.ck_subscribers[agent_id])
                except Exception as e:
                    self.get_logger().warn(f'Error destroying CK subscriber for {agent_id}: {e}')
                del self.ck_subscribers[agent_id]
                
                # Remove CK data for this agent
                with self.ck_lock:
                    if agent_id in self.ck_data:
                        del self.ck_data[agent_id]
                        
                        # Update global bounds after removing this agent
                        all_remaining_bounds = []
                        for aid, ck_data in self.ck_data.items():
                            if len(ck_data.get('l_bounds', [])) > 0:
                                all_remaining_bounds.append(ck_data['l_bounds'][-1])
                        updateGlobalBounds(all_remaining_bounds)
                
                # Remove agent data
                with self.data_lock:
                    if agent_id in self.agent_data:
                        del self.agent_data[agent_id]
                
                self.get_logger().info(f'Removed subscribers for agent {agent_id}')
        except Exception as e:
            self.get_logger().error(f'Error removing subscribers for agent {agent_id}: {e}')

    def agentDataCallback(self, msg, agent_id):
        """Callback function for agent data messages"""
        if self._shutdown_requested:
            return
        
        # Skip processing if this agent is not in the allowed list
        if self.allowed_agents and agent_id not in self.allowed_agents:
            return
            
        # Extract timestamp from ROS header (convert from nanoseconds to seconds)
        ros_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        # Get simulation time from the message
        agent_sim_time = msg.simulation_time if hasattr(msg, 'simulation_time') else ros_time
        
        # Thread-safe data storage
        with self.data_lock:
            data = self.agent_data[agent_id]
            
            # Calculate the unified simulation time by adding the offset
            # This ensures all agents are on the same timeline
            unified_sim_time = agent_sim_time + data['simulation_time_offset']
            
            data['timestamps'].append(ros_time)
            data['simulation_times'].append(unified_sim_time)  # Store unified simulation time
            # Use numpy arrays directly - no need to convert to list
            data['states'].append(np.array(msg.states))
            data['inputs'].append(np.array(msg.inputs))
            data['ergodic_costs'].append(msg.ergodic_cost)
            data['cbf_flags'].append(msg.active_cbf_flag)
            data['in_range_agents_ids'].append(list(msg.in_range_agents_ids))  # Store communication connections
            
            # Store delta_t_ts if available
            if hasattr(msg, 'delta_t_ts'):
                data['delta_t_ts'].append(msg.delta_t_ts)
            else:
                # If not available, append 0.0 to maintain array consistency
                data['delta_t_ts'].append(0.0)

            # Keep only the last 300000 data points to avoid memory issues but show more history
            max_points = int(1e5)
            if len(data['timestamps']) > max_points:
                for key in data:
                    if key != 'simulation_time_offset':  # Don't truncate the offset value
                        data[key] = data[key][-max_points:]

    def obstaclesCallback(self, msg, agent_id):
        """Callback function for obstacle data messages"""
        if self._shutdown_requested:
            return
        
        # Skip processing if this agent is not in the allowed list
        if self.allowed_agents and agent_id not in self.allowed_agents:
            return
            
        with self.obstacles_lock:
            # Store obstacles from this agent (we'll use the first agent's obstacles for now)
            # In the future, you might want to merge obstacles from multiple agents
            self.obstacles_data[agent_id] = []
            
            for obs_msg in msg.obstacles:
                obstacle = {
                    'name': obs_msg.obs_name,
                    'type': obs_msg.obs_type,
                    'position': [obs_msg.position.x, obs_msg.position.y, obs_msg.position.z],
                    'dimensions': list(obs_msg.dimensions),
                    'kappa': obs_msg.kappa,
                    'rho0': obs_msg.rho0
                }
                # Append only if name not of the form "agent_{id}"
                if VISUALISE_VIRTUAL_OBS or not re.match(r'agent[_\-]?(\d+)', obstacle['name'], re.IGNORECASE):
                    self.obstacles_data[agent_id].append(obstacle)

    def targetEstimatesCallback(self, msg, agent_id):
        """Callback function for target estimates data messages"""
        if self._shutdown_requested:
            return
        
        # Skip processing if this agent is not in the allowed list
        if self.allowed_agents and agent_id not in self.allowed_agents:
            return
            
        # Get current simulation time for historical data
        current_time = 0.0
        with self.data_lock:
            if agent_id in self.agent_data and len(self.agent_data[agent_id]['simulation_times']) > 0:
                current_time = self.agent_data[agent_id]['simulation_times'][-1]
            
        with self.target_estimates_lock:
            # Store target estimates and ground truths from this agent
            self.target_estimates_data[agent_id] = {
                'estimates': [],
                'ground_truths': []
            }
            
            # Store target estimates with covariances
            for est_msg in msg.target_estimates:
                estimate = {
                    'target_id': est_msg.target_id,
                    'position': [est_msg.position.x, est_msg.position.y, est_msg.position.z],
                    'covariance': np.array(est_msg.covariance).reshape(3, 3)  # Reshape from flattened to 3x3
                }
                self.target_estimates_data[agent_id]['estimates'].append(estimate)
            
            # Store ground truth positions
            for gt_msg in msg.ground_truths:
                ground_truth = {
                    'target_id': gt_msg.target_id,
                    'position': [gt_msg.position.x, gt_msg.position.y, gt_msg.position.z]
                }
                self.target_estimates_data[agent_id]['ground_truths'].append(ground_truth)
        
        # Store historical data for EKF plotting (only if this is the EKF agent)
        if self.ekf_agent == agent_id:
            with self.target_estimates_history_lock:
                if agent_id not in self.target_estimates_history:
                    self.target_estimates_history[agent_id] = {}
                
                # Store estimates for each target
                for est_msg in msg.target_estimates:
                    target_id = est_msg.target_id
                    
                    if target_id not in self.target_estimates_history[agent_id]:
                        self.target_estimates_history[agent_id][target_id] = {
                            'times': [],
                            'positions': [],
                            'covariances': []
                        }
                    
                    # Append historical data
                    self.target_estimates_history[agent_id][target_id]['times'].append(current_time)
                    self.target_estimates_history[agent_id][target_id]['positions'].append(
                        [est_msg.position.x, est_msg.position.y, est_msg.position.z]
                    )
                    self.target_estimates_history[agent_id][target_id]['covariances'].append(
                        np.array(est_msg.covariance).reshape(3, 3)
                    )
                    
                    # Keep only recent history to avoid memory issues
                    max_history_points = self.max_plot_points
                    if len(self.target_estimates_history[agent_id][target_id]['times']) > max_history_points:
                        for key in ['times', 'positions', 'covariances']:
                            self.target_estimates_history[agent_id][target_id][key] = \
                                self.target_estimates_history[agent_id][target_id][key][-max_history_points:]
        
    def ckCallback(self, msg, agent_id):
        """Callback function for CK table data messages"""
        if self._shutdown_requested:
            return
        
        # Skip processing if this agent is not in the allowed list
        if self.allowed_agents and agent_id not in self.allowed_agents:
            return
            
        # sim_time = the last of the ones in the agent data
        sim_time = 0
        with self.data_lock:
            # Find the maximum last simulation time among all agents
            for aid, data in self.agent_data.items():
                if len(data['simulation_times']) > 0:
                    sim_time = max(data['simulation_times'][-1], sim_time)

        with self.ck_lock:
            # Initialize CK data for this agent if not exists
            if agent_id not in self.ck_data:
                self.ck_data[agent_id] = {
                    'timestamps': [],
                    'ck_tables': [],
                    'table_size': None,
                    'total_erg_costs': [],
                    'l_bounds': []
                }
            
            # Store CK table data
            self.ck_data[agent_id]['timestamps'].append(sim_time)
            self.ck_data[agent_id]['table_size'] = msg.table_size
            self.ck_data[agent_id]['total_erg_costs'].append(msg.total_erg_cost_in_range)
            
            # Store l_bounds if available
            if hasattr(msg, 'l_bounds') and len(msg.l_bounds) >= 4:
                self.ck_data[agent_id]['l_bounds'].append(list(msg.l_bounds))
                
                # Update global bounds with all current agent bounds
                all_current_bounds = []
                for aid, ck_data in self.ck_data.items():
                    if len(ck_data['l_bounds']) > 0:
                        # Use the latest bounds from each agent
                        all_current_bounds.append(ck_data['l_bounds'][-1])
                
                # Update global bounds
                # TODO: Maybe dont update every time, maybe once in a while if we see performance issues
                updateGlobalBounds(all_current_bounds)
            
            # Reshape the flattened array to a square matrix
            ck_table = np.array(msg.ck_values).reshape(msg.table_size, msg.table_size)
            self.ck_data[agent_id]['ck_tables'].append(ck_table.copy())

            # Keep only the last 300000 data points to avoid memory issues
            max_points = int(1e5)
            if len(self.ck_data[agent_id]['timestamps']) > max_points:
                for key in ['timestamps', 'ck_tables', 'total_erg_costs', 'l_bounds']:
                    if key in self.ck_data[agent_id]:
                        self.ck_data[agent_id][key] = self.ck_data[agent_id][key][-max_points:]
            
    def onKeyPress(self, event):
        if event.key == 'e':
            print("Manual refresh triggered")
            self.updatePlots(None)
        elif event.key == 'a':
            self.auto_refresh = not self.auto_refresh
            status = 'ON' if self.auto_refresh else 'OFF'
            print(f"Auto-refresh: {status}")
        elif event.key == 'c':
            print("Clearing all plots")
            self.clearPlots()
        elif event.key == 'q':
            print("Closing all windows")
            plt.close('all')
    
    def clearPlots(self):
        """Clear all plot lines while maintaining axes structure"""
        # Clear trajectory axis (always exists)
        self.traj_ax.clear()
        
        # Only clear other axes if they exist (not in top_view_only mode)
        if not self.top_view_only:
            self.control_ax.clear()
            self.ergodic_ax.clear()
            self.delta_t_ax.clear()
            
            # Clear EKF plot if it exists
            if self.ekf_axes is not None:
                for ax in self.ekf_axes:
                    ax.clear()
        
        # Reset trajectory plot
        # Use current dynamic bounds
        with BOUNDS_LOCK:
            current_l1_bounds = L1_BOUNDS.copy()
            current_l2_bounds = L2_BOUNDS.copy()
        
        self.traj_ax.set_title('Agent Trajectories')
        self.traj_ax.set_xlim(current_l1_bounds[0], current_l1_bounds[1])
        self.traj_ax.set_ylim(current_l2_bounds[0], current_l2_bounds[1])
        self.traj_ax.grid(True)
        
        # Only reset other plots if they exist
        if not self.top_view_only:
            self.control_ax.set_title('Control Inputs')
            self.control_ax.grid(True)
            
            self.ergodic_ax.set_title('Ergodic Cost & Total Ergodic Cost from CK Messages')
            self.ergodic_ax.grid(True)
            
            self.delta_t_ax.set_title('Delta T Timestamps')
            self.delta_t_ax.grid(True)
            
            # Reset EKF plot if it exists
            if self.ekf_axes is not None:
                self.ekf_axes[0].set_ylabel('X Position')
                self.ekf_axes[0].grid(True)
                self.ekf_axes[1].set_ylabel('Y Position')
                self.ekf_axes[1].grid(True)
                self.ekf_axes[2].set_xlabel('Time [s]')
                self.ekf_axes[2].set_ylabel('Z Position')
                self.ekf_axes[2].grid(True)
        
        # Draw all figures
        figures_to_draw = [self.traj_fig]
        if not self.top_view_only:
            figures_to_draw.extend([self.control_fig, self.ergodic_fig, self.delta_t_fig])
            if self.ekf_fig is not None:
                figures_to_draw.append(self.ekf_fig)
        
        for fig in figures_to_draw:
            fig.canvas.draw()
            
    def updatePlots(self, frame):
        if not self.auto_refresh and frame is not None:
            return
            
        # Clear trajectory axis (always exists)
        self.traj_ax.clear()
        
        # Only clear other axes if they exist (not in top_view_only mode)
        if not self.top_view_only:
            self.control_ax.clear()
            self.ergodic_ax.clear()
            self.delta_t_ax.clear()
            
            # Clear EKF plot if it exists
            if self.ekf_axes is not None:
                for ax in self.ekf_axes:
                    ax.clear()
        
        # Create a thread-safe shallow copy of the data structure, but reference arrays directly
        with self.data_lock:
            agent_data_refs = {}
            for agent_id, data in self.agent_data.items():
                # Skip agents not in the allowed list
                if self.allowed_agents and agent_id not in self.allowed_agents:
                    continue
                    
                # Only copy the structure, not the large arrays
                # Also limit to recent data points for performance
                max_plot_points = self.max_plot_points  # Use configurable limit
                agent_data_refs[agent_id] = {
                    'simulation_times': data['simulation_times'][-max_plot_points:],
                    'inputs': data['inputs'][-max_plot_points:],
                    'ergodic_costs': data['ergodic_costs'][-max_plot_points:],
                    'cbf_flags': data['cbf_flags'][-max_plot_points:],
                    'states': data['states'][-max_plot_points:],
                    'in_range_agents_ids': data['in_range_agents_ids'][-max_plot_points:],
                    'delta_t_ts': data['delta_t_ts'][-max_plot_points:]
                }
            
        # Plot 1: Control Inputs (multi-agent) - only if not in top_view_only mode
        if not self.top_view_only:
            for agent_id in sorted(agent_data_refs.keys()):
                data = agent_data_refs[agent_id]
                if len(data['simulation_times']) > 0 and len(data['inputs']) > 0:
                    color = generateAgentColor(agent_id, force_blue_first=FORCE_BLUE_FIRST)
                    
                    try:
                        # Use the unified simulation times (already calculated in callback)
                        # Convert to numpy array only once
                        if len(data['simulation_times']) > 0 and len(data['inputs']) > 0:
                            time_array = np.array(data['simulation_times'])
                            # Extract control inputs (assuming 2 inputs per agent)
                            inputs_array = np.vstack(data['inputs']) if data['inputs'] else np.array([])
                            
                            # Ensure arrays have consistent lengths
                            min_len = min(len(time_array), len(inputs_array)) if len(inputs_array) > 0 else 0
                            if min_len > 0:
                                time_array = time_array[:min_len]
                                inputs_array = inputs_array[:min_len]
                                
                                if inputs_array.size > 0 and inputs_array.ndim > 1 and inputs_array.shape[1] >= 2:
                                    self.control_ax.plot(time_array, inputs_array[:, 0], 
                                                       label=f'U1 - Agent {agent_id}', linewidth=2, 
                                                       color=color, linestyle='-')
                                    self.control_ax.plot(time_array, inputs_array[:, 1], 
                                                       label=f'U2 - Agent {agent_id}', linewidth=2, 
                                                       color=color, linestyle='--')
                    except (ValueError, IndexError) as e:
                        # Skip this agent if data is inconsistent
                        continue

            self.control_ax.set_title('Control Inputs (All Agents)')
            if self.control_ax.get_legend_handles_labels()[0]:  # Only add legend if there are plots
                self.control_ax.legend()
            self.control_ax.grid(True)
            self.control_ax.set_xlabel('Unified Simulation Time [s]')
            self.control_ax.set_ylabel('Control Values')
            
        # Plot 2: Ergodic Cost (multi-agent) + Total Ergodic Cost from Averaged CK - only if not in top_view_only mode
        if not self.top_view_only:
            max_time = 0.0  # Track the maximum time across all data sources
            
            # Plot individual agent ergodic costs
            for agent_id in sorted(agent_data_refs.keys()):
                data = agent_data_refs[agent_id]
                if len(data['simulation_times']) > 0 and len(data['ergodic_costs']) > 0:
                    color = generateAgentColor(agent_id, force_blue_first=FORCE_BLUE_FIRST)
                    
                    try:
                        # Use the unified simulation times (already calculated in callback)
                        if len(data['simulation_times']) > 0 and len(data['ergodic_costs']) > 0:
                            time_array = np.array(data['simulation_times'])
                            ergodic_costs = np.array(data['ergodic_costs'])
                            cbf_flags = np.array(data['cbf_flags'])
                            
                            # Ensure arrays have consistent lengths
                            min_len = min(len(time_array), len(ergodic_costs), len(cbf_flags))
                            if min_len > 0:
                                time_array = time_array[:min_len]
                                ergodic_costs = ergodic_costs[:min_len]
                                cbf_flags = cbf_flags[:min_len]
                                
                                # Update max time
                                if len(time_array) > 0:
                                    max_time = max(max_time, np.max(time_array))
                                
                                self.ergodic_ax.plot(time_array, ergodic_costs, 
                                                   label=f'Ergodic Cost - Agent {agent_id}', 
                                                   linewidth=2, color=color)
                                
                                # Scale CBF flags to be visible
                                if len(ergodic_costs) > 0 and np.max(ergodic_costs) > 0:
                                    scale_factor = np.max(ergodic_costs)
                                    self.ergodic_ax.plot(time_array, cbf_flags * scale_factor, 
                                                       label=f'Active CBF - Agent {agent_id}', 
                                                       linewidth=1, color=color, linestyle='--', alpha=0.85)
                    except (ValueError, IndexError) as e:
                        # Skip this agent if data is inconsistent
                        continue
            
            # Plot total_erg_cost from CK messages for each agent (dashed lines)
            with self.ck_lock:
                ck_data_refs = {}
                for agent_id, data in self.ck_data.items():
                    # Skip agents not in the allowed list
                    if self.allowed_agents and agent_id not in self.allowed_agents:
                        continue
                        
                    # Only reference the arrays we need, and limit points for performance
                    max_plot_points = self.max_plot_points  # Use configurable limit
                    ck_data_refs[agent_id] = {
                        'timestamps': data['timestamps'][-max_plot_points:],
                        'total_erg_costs': data['total_erg_costs'][-max_plot_points:]
                    }
            
            for agent_id in sorted(ck_data_refs.keys()):
                if agent_id in agent_data_refs:  # Only plot if agent is active
                    ck_data = ck_data_refs[agent_id]
                    if len(ck_data['timestamps']) > 0 and len(ck_data['total_erg_costs']) > 0:
                        color = generateAgentColor(agent_id, force_blue_first=FORCE_BLUE_FIRST)
                        
                        try:
                            time_array = np.array(ck_data['timestamps'])
                            total_erg_costs = np.array(ck_data['total_erg_costs'])
                            
                            # Ensure arrays have consistent lengths
                            min_len = min(len(time_array), len(total_erg_costs))
                            if min_len > 0:
                                time_array = time_array[:min_len]
                                total_erg_costs = total_erg_costs[:min_len]
                                
                                # Update max time
                                if len(time_array) > 0:
                                    max_time = max(max_time, np.max(time_array))
                                
                                self.ergodic_ax.plot(time_array, total_erg_costs, 
                                                   label=f'Total Erg Cost - Agent {agent_id}', 
                                                   linewidth=2, color=color, linestyle='--')
                        except (ValueError, IndexError) as e:
                            # Skip this agent if data is inconsistent
                            continue
            
            self.ergodic_ax.set_title('Ergodic Cost & Total Ergodic Cost from CK Messages')
            if self.ergodic_ax.get_legend_handles_labels()[0]:  # Only add legend if there are plots
                self.ergodic_ax.legend()
            self.ergodic_ax.grid(True)
            self.ergodic_ax.set_xlabel('Time [s]')
            self.ergodic_ax.set_ylabel('Cost')
            
            # Set x-axis limits based on maximum time found
            # if max_time > 0:
            #     self.ergodic_ax.set_xlim(0, max_time * 1.05)  # Add 5% padding
            
        # Plot 4: Delta T Timestamps (multi-agent) from AgentData messages - only if not in top_view_only mode
        if not self.top_view_only:
            # First, find the maximum sample count across all agents
            max_samples = 0
            valid_agents = []
            for agent_id in sorted(agent_data_refs.keys()):
                data = agent_data_refs[agent_id]
                if len(data['delta_t_ts']) > 0:
                    max_samples = max(max_samples, len(data['delta_t_ts']))
                    valid_agents.append(agent_id)
            
            # Plot each agent's delta_t_ts using the common x-axis range
            for agent_id in valid_agents:
                data = agent_data_refs[agent_id]
                color = generateAgentColor(agent_id, force_blue_first=FORCE_BLUE_FIRST)
                
                try:
                    delta_t_ts_array = np.array(data['delta_t_ts'])
                    
                    if len(delta_t_ts_array) > 0:
                        # Use sequential index starting from the end of the max range
                        # This aligns all agents to the "current time" (right side of plot)
                        start_idx = max_samples - len(delta_t_ts_array)
                        x_axis = np.arange(start_idx, max_samples)
                        
                        self.delta_t_ax.plot(x_axis, delta_t_ts_array, 
                                           label=f'Delta T - Agent {agent_id}', 
                                           linewidth=2, color=color)
                except (ValueError, IndexError) as e:
                        # Skip this agent if data is inconsistent
                        continue
            # Add horizontal black dashed line at y = 1
            self.delta_t_ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='Ideal Delta T = 1s')
            self.delta_t_ax.set_title('Delta T Timestamps from AgentData Messages')
            if self.delta_t_ax.get_legend_handles_labels()[0]:  # Only add legend if there are plots
                self.delta_t_ax.legend()
            self.delta_t_ax.grid(True)
            self.delta_t_ax.set_xlabel('Sample Index')
            self.delta_t_ax.set_ylabel('Delta T [s]')        # Plot 3: Agent Trajectories (multi-agent)
        with BOUNDS_LOCK:
            current_l1_bounds = L1_BOUNDS.copy()
            current_l2_bounds = L2_BOUNDS.copy()
        
        for agent_id in sorted(agent_data_refs.keys()):
            data = agent_data_refs[agent_id]
            if len(data['states']) > 0:
                color = generateAgentColor(agent_id, force_blue_first=FORCE_BLUE_FIRST)
                
                try:
                    # Extract positions (assuming first two states are x, y)
                    if len(data['states']) > 0:
                        states_array = np.vstack(data['states']) if data['states'] else np.array([])
                        if states_array.size > 0 and states_array.ndim > 1 and states_array.shape[1] >= 2:
                            # Check if this agent uses inverted position convention (NED like airplanes)
                            if agent_id in self.pos_inverted_agents:
                                # For NED convention agents (airplanes), swap X and Y
                                x_positions = states_array[:, 1]
                                y_positions = states_array[:, 0]
                            else:
                                # For standard agents (drones), dont swap X and Y
                                x_positions = states_array[:, 0]
                                y_positions = states_array[:, 1]
                            
                            # Current position
                            if len(x_positions) > 0:
                                self.traj_ax.scatter(x_positions[-1], y_positions[-1], s=100, 
                                                   color=color, label=f'Agent {agent_id} Current', 
                                                   zorder=3, marker='o')
                                
                                # Agent trajectory
                                # if agent_id == 5 or agent_id == 6:
                                    # self.traj_ax.plot(x_positions, y_positions, linewidth=1, 
                                                    # label=f'Agent {agent_id} Path', color=color, zorder=2, linestyle='--', alpha=0.8)
                                # else:
                                self.traj_ax.plot(x_positions, y_positions, linewidth=2, 
                                                label=f'Agent {agent_id} Path', color=color, zorder=2)
                                    
                except (ValueError, IndexError) as e:
                    # Skip this agent if data is inconsistent
                    continue
        
        # Use dynamically updated bounds
        self.traj_ax.set_xlim(current_l1_bounds[0], current_l1_bounds[1])
        self.traj_ax.set_ylim(current_l2_bounds[0], current_l2_bounds[1])
        self.traj_ax.set_title('Agent Trajectories')
        self.traj_ax.set_aspect('equal')
        
        # Draw communication lines between agents (before obstacles and targets for proper layering)
        self.drawCommunicationLines(self.traj_ax, agent_data_refs)
        
        # Draw obstacles on the trajectory plot
        self.drawObstacles(self.traj_ax)
        
        # Draw target estimates and ground truths on the trajectory plot
        self.drawTargetEstimates(self.traj_ax)
        
        # if self.traj_ax.get_legend_handles_labels()[0]:  # Only add legend if there are plots
        #     self.traj_ax.legend()
        self.traj_ax.grid(True)
        self.traj_ax.set_xlabel('X Position')
        self.traj_ax.set_ylabel('Y Position')

        # Plot EKF target position estimates with confidence bands for specified agent
        if not self.top_view_only:
            self.drawEKFPlot(agent_data_refs)

        # Draw all figures
        figures_to_draw = [self.traj_fig]
        if not self.top_view_only:
            figures_to_draw.extend([self.control_fig, self.ergodic_fig, self.delta_t_fig])
            if self.ekf_fig is not None:
                figures_to_draw.append(self.ekf_fig)
            
        for fig in figures_to_draw:
            fig.canvas.draw()
    
    def drawObstacles(self, ax):
        """Draw obstacles on the given axis"""
        with self.obstacles_lock:
            # TODO: Could calculate every few iterations to reduce computational load
            # Just get a shallow reference to avoid deep copying
            # obstacles_refs = dict(self.obstacles_data)

            # Combine obstacles from all agents into a single list
            obstacles_refs = {}
            combined_obstacles = []
            obstacle_names_seen = set()  # Track unique obstacle names to avoid duplicates
            
            for agent_id, agent_obstacles in self.obstacles_data.items():
                for obs in agent_obstacles:
                    # Use obstacle name as unique identifier, or position-based identifier if name not available
                    obs_name = obs.get('name')
                    if not obs_name:
                        obs_identifier = obs_name
                    else:
                        # Create position-based identifier from actual coordinates
                        x_pos = obs['position'][0]
                        y_pos = obs['position'][1]
                        obs_identifier = f"obs_{x_pos:.2f}_{y_pos:.2f}"
                    
                    # print(f"Agent {agent_id} obstacle: {obs_identifier}")
                    
                    # Only add if we haven't seen this obstacle before
                    if obs_identifier not in obstacle_names_seen:
                        combined_obstacles.append(obs)
                        obstacle_names_seen.add(obs_identifier)
                        # print(f"Added obstacle: {obs_identifier} from agent {agent_id}")

                # Create a single entry with all combined obstacles
                if combined_obstacles:
                    obstacles_refs['combined'] = combined_obstacles

        # Draw obstacles from any agent (we'll use the first available agent's obstacles)
        # In practice, obstacles should be the same across agents or you might want to merge them
        for agent_id, obstacles in obstacles_refs.items():
            for obs in obstacles:
                try:
                    if obs['type'] == 'circle':
                        if len(obs['dimensions']) >= 1:
                            radius = obs['dimensions'][0]
                            circle = Circle(
                                (obs['position'][0], obs['position'][1]), 
                                radius, 
                                color='black', 
                                alpha=0.5, 
                                linewidth=2,
                                fill=True
                            )
                            ax.add_patch(circle)
                            # Add obstacle name if available
                            # if obs['name']:
                            #     ax.text(obs['position'][0], obs['position'][1], obs['name'], 
                            #            ha='center', va='center', fontsize=8, color='white', weight='bold')
                    
                    elif obs['type'] == 'rectangle':
                        if len(obs['dimensions']) >= 2:
                            width = obs['dimensions'][0]
                            height = obs['dimensions'][1]
                            # Rectangle position is centered
                            rect = Rectangle(
                                (obs['position'][0] - width/2, obs['position'][1] - height/2), 
                                width, 
                                height,
                                color='black', 
                                alpha=0.5, 
                                linewidth=2,
                                fill=True
                            )
                            ax.add_patch(rect)
                            # Add obstacle name if available
                            # if obs['name']:
                            #     ax.text(obs['position'][0], obs['position'][1], obs['name'], 
                            #            ha='center', va='center', fontsize=8, color='white', weight='bold')
                    
                    elif obs['type'] == 'wall':
                        if len(obs['dimensions']) == 3:
                            # Wall represented by normal vector
                            normal_x = obs['dimensions'][0]
                            normal_y = obs['dimensions'][1]
                            wall_length = obs['dimensions'][2]
                            
                            # Create a line perpendicular to the normal through the given point
                            # For visualization, we'll draw a thick line segment

                            # Perpendicular direction to the normal
                            perp_x = -normal_y
                            perp_y = normal_x
                            
                            # Normalize perpendicular vector
                            perp_norm = np.sqrt(perp_x**2 + perp_y**2)
                            if perp_norm > 0:
                                perp_x /= perp_norm
                                perp_y /= perp_norm
                            
                            # Wall endpoints
                            x1 = obs['position'][0] - wall_length/2 * perp_x
                            y1 = obs['position'][1] - wall_length/2 * perp_y
                            x2 = obs['position'][0] + wall_length/2 * perp_x
                            y2 = obs['position'][1] + wall_length/2 * perp_y
                            
                            ax.plot([x1, x2], [y1, y2], color='black', linewidth=8, alpha=0.7, solid_capstyle='round')
                            
                            # Add obstacle name if available
                            # if obs['name']:
                            #     ax.text(obs['position'][0], obs['position'][1], obs['name'], 
                            #            ha='center', va='center', fontsize=8, color='white', weight='bold')
                
                except (IndexError, ValueError) as e:
                    # Skip malformed obstacle data
                    continue
            
            # Only draw obstacles from the first agent to avoid duplicates
            # (assuming all agents share the same obstacle map)
            break
    
    def drawTargetEstimates(self, ax):
        """Draw target estimates with ellipses and ground truth with black X marks"""
        with self.target_estimates_lock:
            # Just get shallow references to avoid deep copying, filtered by allowed agents
            target_data_refs = {}
            for agent_id, data in self.target_estimates_data.items():
                # Skip agents not in the allowed list
                if self.allowed_agents and agent_id not in self.allowed_agents:
                    continue
                target_data_refs[agent_id] = data
        
        # Draw target estimates and ground truths from each agent
        for agent_id, data in target_data_refs.items():
            color = generateAgentColor(agent_id, force_blue_first=FORCE_BLUE_FIRST)
            
            try:                
                # Draw ground truth positions as black X marks - only the nearest one to each estimate
                for estimate in data['estimates']:
                    est_pos = np.array(estimate['position'][:2])  # Only x, y coordinates
                    
                    if len(data['ground_truths']) > 0:
                        # Find the nearest ground truth to this estimate
                        min_distance = float('inf')
                        nearest_gt = None
                        
                        for gt in data['ground_truths']:
                            gt_pos = np.array(gt['position'][:2])  # Only x, y coordinates
                            distance = np.linalg.norm(est_pos - gt_pos)
                            if distance < min_distance:
                                min_distance = distance
                                nearest_gt = gt
                        
                        # Plot only the nearest ground truth
                        if nearest_gt is not None:
                            gt_pos = nearest_gt['position']
                            ax.scatter(gt_pos[0], gt_pos[1], color='black', s=50, marker='x', linewidth=1,
                                     label='Ground Truth' if estimate == data['estimates'][0] and agent_id == min(target_data_refs.keys()) else "",
                                     zorder=5)
                            
                # Draw target estimates as ellipses (uncertainty visualization)
                for estimate in data['estimates']:
                    pos = estimate['position']
                    cov = estimate['covariance']
                    
                    # Extract 2D covariance (x-y components only)
                    cov_2d = cov[:2, :2]
                    
                    # Calculate eigenvalues and eigenvectors for ellipse parameters
                    eigenvals, eigenvecs = np.linalg.eig(cov_2d)
                    
                    # Sort eigenvalues and eigenvectors
                    idx = np.argsort(eigenvals)[::-1]  # Sort in descending order
                    eigenvals = eigenvals[idx]
                    eigenvecs = eigenvecs[:, idx]
                    
                    # Calculate ellipse parameters (2-sigma confidence ellipse)
                    confidence_scale = 2.0  # 2-sigma confidence
                    width = 2 * confidence_scale * np.sqrt(eigenvals[0])
                    height = 2 * confidence_scale * np.sqrt(eigenvals[1])
                    
                    # Calculate rotation angle in degrees
                    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
                    
                    # Create and add ellipse
                    ellipse = Ellipse(
                        xy=(pos[0], pos[1]),
                        width=width,
                        height=height,
                        angle=angle,
                        facecolor=color,
                        alpha=0.3,
                        edgecolor=color,
                        linewidth=2
                    )
                    ax.add_patch(ellipse)
                    
                    # Add center point
                    ax.scatter(pos[0], pos[1], color=color, s=50, marker='x', 
                             label=f'Target Est. Agent {agent_id}' if estimate == data['estimates'][0] else "",
                             zorder=4)
                    
            except (ValueError, IndexError, np.linalg.LinAlgError) as e:
                # Skip this agent's target data if there are issues with the covariance matrix
                continue
        
    def drawCommunicationLines(self, ax, agent_data_refs):
        """Draw dotted communication lines between agents that can communicate with each other"""
        try:
            # Get current positions and communication connections for all agents
            agent_positions = {}
            agent_connections = {}
            
            # Extract current positions and in-range agents for each agent
            for agent_id, data in agent_data_refs.items():
                if len(data['states']) > 0 and len(data['in_range_agents_ids']) > 0:
                    # Get current position (last state)
                    current_state = data['states'][-1]
                    if len(current_state) >= 2:
                        agent_positions[agent_id] = (current_state[0], current_state[1])
                        
                        # Get current in-range agents (last communication data)
                        in_range_agents = data['in_range_agents_ids'][-1]
                        agent_connections[agent_id] = in_range_agents
            
            # Draw communication lines
            for agent_id, connections in agent_connections.items():
                if agent_id not in agent_positions:
                    continue
                    
                agent_pos = agent_positions[agent_id]
                agent_color = generateAgentColor(agent_id, force_blue_first=FORCE_BLUE_FIRST)
                
                # Draw dotted lines to each agent this agent can communicate with
                for connected_agent_id in connections:
                    if connected_agent_id in agent_positions:
                        connected_pos = agent_positions[connected_agent_id]
                        
                        # Draw dotted line from this agent to the connected agent
                        ax.plot([agent_pos[0], connected_pos[0]], 
                               [agent_pos[1], connected_pos[1]], 
                               color=agent_color, 
                               linestyle=':', 
                               linewidth=1.5, 
                               alpha=0.7,
                               zorder=1)  # Lower zorder so lines appear behind trajectories
                        
        except Exception as e:
            # Print error for debugging (can be removed in production)
            if hasattr(self, 'get_logger'):
                self.get_logger().warn(f'Error drawing communication lines: {e}')
            else:
                print(f"Error drawing communication lines: {e}")
            pass
    
    def drawEKFPlot(self, agent_data_refs):
        """Draw EKF target position estimates with 3-sigma confidence bands for the specified agent"""
        if self.ekf_agent is None or self.ekf_axes is None:
            return
            
        # Check if the EKF agent has historical target estimate data
        with self.target_estimates_history_lock:
            if self.ekf_agent not in self.target_estimates_history:
                return
                
            ekf_history = self.target_estimates_history[self.ekf_agent]
            
            if not ekf_history:
                return
        
        # Color palette for different targets
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        # Get current estimates and ground truth data for matching nearest ground truths
        current_estimates = {}
        current_ground_truths = {}
        with self.target_estimates_lock:
            if self.ekf_agent in self.target_estimates_data:
                # Get current estimates
                for est in self.target_estimates_data[self.ekf_agent].get('estimates', []):
                    current_estimates[est['target_id']] = est['position']
                    
                # Store all ground truths for nearest-neighbor matching
                all_ground_truths = self.target_estimates_data[self.ekf_agent].get('ground_truths', [])
        
        # For each target estimate, find the nearest ground truth (same logic as trajectory plot)
        visible_ground_truths = {}
        for target_id, est_pos in current_estimates.items():
            if len(all_ground_truths) > 0:
                est_pos_2d = np.array(est_pos[:2])  # Only x, y coordinates for matching
                min_distance = float('inf')
                nearest_gt = None
                
                for gt in all_ground_truths:
                    gt_pos_2d = np.array(gt['position'][:2])  # Only x, y coordinates
                    distance = np.linalg.norm(est_pos_2d - gt_pos_2d)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_gt = gt
                
                # Only store the nearest ground truth for this target
                if nearest_gt is not None:
                    visible_ground_truths[target_id] = nearest_gt['position']
        
        # Plot historical trajectories for each target
        for target_id, history in ekf_history.items():
            if len(history['times']) == 0:
                continue
                
            color = colors[target_id % len(colors)]
            
            try:
                # Convert to numpy arrays for easier handling
                times = np.array(history['times'])
                positions = np.array(history['positions'])
                covariances = np.array(history['covariances'])
                
                if len(times) == 0 or len(positions) == 0 or len(covariances) == 0:
                    continue
                
                # Ensure all arrays have the same length
                min_len = min(len(times), len(positions), len(covariances))
                if min_len == 0:
                    continue
                    
                times = times[:min_len]
                positions = positions[:min_len]
                covariances = covariances[:min_len]
                
                # Extract standard deviations from covariances
                sigmas_x = np.sqrt(covariances[:, 0, 0])  # X variance
                sigmas_y = np.sqrt(covariances[:, 1, 1])  # Y variance  
                sigmas_z = np.sqrt(covariances[:, 2, 2])  # Z variance
                
                # X position plot
                self.ekf_axes[0].plot(times, positions[:, 0], color=color, linewidth=2,
                                    label=f'Target {target_id} - Agent {self.ekf_agent}')
                self.ekf_axes[0].fill_between(times, 
                                            positions[:, 0] - 3 * sigmas_x, 
                                            positions[:, 0] + 3 * sigmas_x, 
                                            color=color, alpha=0.2)
                
                # Y position plot
                self.ekf_axes[1].plot(times, positions[:, 1], color=color, linewidth=2)
                self.ekf_axes[1].fill_between(times, 
                                            positions[:, 1] - 3 * sigmas_y, 
                                            positions[:, 1] + 3 * sigmas_y, 
                                            color=color, alpha=0.2)
                
                # Z position plot
                self.ekf_axes[2].plot(times, positions[:, 2], color=color, linewidth=2)
                self.ekf_axes[2].fill_between(times, 
                                            positions[:, 2] - 3 * sigmas_z, 
                                            positions[:, 2] + 3 * sigmas_z, 
                                            color=color, alpha=0.2)
                
                # Plot ground truth as horizontal dashed lines if available
                if target_id in visible_ground_truths:
                    gt_pos = visible_ground_truths[target_id]
                    self.ekf_axes[0].axhline(y=gt_pos[0], color=color, linestyle='--', 
                                           alpha=0.8, label=f'Real Target {target_id}')
                    self.ekf_axes[1].axhline(y=gt_pos[1], color=color, linestyle='--', alpha=0.8)
                    self.ekf_axes[2].axhline(y=gt_pos[2], color=color, linestyle='--', alpha=0.8)
                    
            except (ValueError, IndexError) as e:
                # Skip this target if data is inconsistent
                continue
        
        # Configure axes
        self.ekf_axes[0].set_ylabel('X Position')
        self.ekf_axes[0].grid(True)
        # if self.ekf_axes[0].get_legend_handles_labels()[0]:
        #     self.ekf_axes[0].legend()
        
        self.ekf_axes[1].set_ylabel('Y Position')
        self.ekf_axes[1].grid(True)
        
        self.ekf_axes[2].set_xlabel('Time [s]')
        self.ekf_axes[2].set_ylabel('Z Position') 
        self.ekf_axes[2].grid(True)
        
        # Set dynamic bounds if available
        with BOUNDS_LOCK:
            current_l1_bounds = L1_BOUNDS.copy()
            current_l2_bounds = L2_BOUNDS.copy()
            
        # Set Y limits based on workspace bounds (for X and Y plots)
        self.ekf_axes[0].set_ylim(current_l1_bounds[0], current_l1_bounds[1])
        self.ekf_axes[1].set_ylim(current_l2_bounds[0], current_l2_bounds[1])
    
    def cleanup(self):
        """Clean up all subscribers and resources"""
        self._shutdown_requested = True  # Set shutdown flag
        
        try:
            # Stop discovery timer
            if hasattr(self, 'discovery_timer'):
                self.discovery_timer.cancel()
            
            # Remove all agent subscribers
            agent_ids_to_remove = list(self.discovered_agents)
            for agent_id in agent_ids_to_remove:
                self.removeAgentSubscriber(agent_id)
            
            self.get_logger().info('Dashboard cleanup completed')
        except Exception as e:
            self.get_logger().error(f'Error during cleanup: {e}')

    def show(self):
        plt.show()

def rosSpinThread(dashboard, shutdown_event):
    """Thread function to spin the ROS node"""
    executor = MultiThreadedExecutor()
    executor.add_node(dashboard)
    try:
        while not shutdown_event.is_set():
            executor.spin_once(timeout_sec=0.1)
    except Exception as e:
        if not shutdown_event.is_set():
            print(f"ROS spinning error: {e}")
    finally:
        try:
            executor.shutdown()
        except Exception as e:
            print(f"Executor shutdown error: {e}")

import signal

def parseArguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='ROS2 Multi-Agent Dashboard')
    parser.add_argument('--agents', nargs='+', type=int, metavar='ID',
                        help='Specify agent IDs to filter (e.g., --agents 1 3 4). If not specified, all discovered agents will be shown.')
    parser.add_argument('--max-path-points', type=int, default=1000, metavar='NUM',
                        help='Maximum number of path points to display in visualization (default: 1000). Higher values show longer trails but may impact performance.')
    parser.add_argument('--ekf-agent', type=int, metavar='ID',
                        help='Specify agent ID to show EKF target position estimates with confidence bands. If not specified, EKF plot will not be shown.')
    parser.add_argument('--blue', action='store_true',
                        help='Force the first agent (agent 1) to be colored blue regardless of color methodology.')
    parser.add_argument('--top_view_only', action='store_true',
                        help='Display only the top-view trajectory plot, hiding all other plots (control inputs, ergodic cost, delta T, and EKF).')
    parser.add_argument('--pos_inverted', nargs='+', type=int, metavar='ID',
                        help='Specify agent IDs that use inverted position convention (NED for airplanes). For these agents, X and Y positions will NOT be swapped. Example: --pos_inverted 1 2')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parseArguments()
    
    # Set the global blue flag
    global FORCE_BLUE_FIRST
    FORCE_BLUE_FIRST = args.blue
    
    # Initialize ROS
    rclpy.init()

    # Create dashboard (agents will be discovered dynamically)
    dashboard = None
    
    try:
        dashboard = LiveDashboard(allowed_agents=args.agents, max_plot_points=args.max_path_points, 
                                 ekf_agent=args.ekf_agent, top_view_only=args.top_view_only,
                                 pos_inverted_agents=args.pos_inverted)

        # Shutdown event for clean exit
        shutdown_event = threading.Event()

        def handle_sigint(signum, frame):
            print("\nSIGINT received. Shutting down...")
            shutdown_event.set()
            plt.close('all')

        signal.signal(signal.SIGINT, handle_sigint)

        # Start ROS spinning in a separate thread
        ros_thread = threading.Thread(target=rosSpinThread, args=(dashboard, shutdown_event))
        ros_thread.daemon = True
        ros_thread.start()

        print("Multi-Agent Dashboard Controls (dynamic agent discovery):")
        print("- Press 'e' to manually refresh")
        print("- Press 'a' to toggle auto-refresh")
        print("- Press 'c' to clear all plots")
        print("- Press 'q' to quit")
        print("- Agents, obstacles, and target estimates will be automatically discovered from running nodes")
        print("- Target estimates shown as colored ellipses (2-sigma confidence), ground truth as black X marks")
        print("- Agent colors are consistent between dashboard and RViz (use same color generation)")
        
        if args.top_view_only:
            print("- TOP VIEW ONLY MODE: Only displaying agent trajectories plot")
        elif args.ekf_agent is not None:
            print(f"- EKF plot window: Shows target position estimates with 3σ confidence bands for Agent {args.ekf_agent}")
            print("- Five plot windows: Control Inputs, Ergodic Cost, Agent Trajectories, Delta T Timestamps, and EKF Target Positions")
        else:
            print("- Four plot windows: Control Inputs, Ergodic Cost, Agent Trajectories, and Delta T Timestamps")
            
        print(f"- Max path points displayed: {dashboard.max_plot_points} (change with --max-path-points)")
        
        if args.agents:
            print(f"- FILTERING: Only processing data for agents: {sorted(args.agents)}")
        else:
            print("- Processing data for all discovered agents")
            
        if args.blue:
            print("- BLUE OVERRIDE: Agent 1 will be colored blue regardless of color methodology")
        
        print("\nWaiting for agent discovery...")

        try:
            dashboard.show()
        except Exception as e:
            print(f"Exception during dashboard display: {e}")
        finally:
            shutdown_event.set()
            # Wait for ROS thread to finish
            if ros_thread.is_alive():
                ros_thread.join(timeout=2.0)
            
    except Exception as e:
        print(f"Exception during initialization: {e}")
    finally:
        if dashboard is not None:
            try:
                dashboard.cleanup()
                dashboard.destroy_node()
            except Exception as e:
                print(f"Error during dashboard cleanup: {e}")
        
        try:
            rclpy.shutdown()
        except Exception as e:
            print(f"Error during ROS shutdown: {e}")

if __name__ == "__main__":
    main()