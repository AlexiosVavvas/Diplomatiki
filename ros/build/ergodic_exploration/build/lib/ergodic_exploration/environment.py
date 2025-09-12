#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import re
import numpy as np
import math
from my_interfaces.msg import AgentData, MultipleObstacles, SingleObstacle, MultipleTargetEstimates, SingleTargetEstimate
from geometry_msgs.msg import Point, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from nav_msgs.msg import OccupancyGrid, MapMetaData, Path
from rcl_interfaces.srv import GetParameters
import colorsys
from my_erg_lib.basis import Basis, ReconstructedPhiFromCk

# ===================-----------------------------------------------

# Map Parameters
MAP_WIDTH = 20.0        # [m] - Usually 20 (Airplane 200)
MAP_HEIGHT = 20.0       # [m] - Usually 20 (Airplane 200)
MAP_RESOLUTION = 0.05   # [pixels/m] - Usually 0.05 (Airplane 1.0)
WALL_THICKNESS = 0.1    # [m] - Thickness of wall obstacles in the occupancy grid

# ===================-----------------------------------------------


class EnvironmentNode(Node):
    def __init__(self):
        super().__init__('environment')
        
        # Dictionary to store agent positions {agent_id: (x, y, z)}
        self.agent_positions = {}
        
        # Dictionary to store agent orientations {agent_id: (x, y, psi, uvel, omegavel)}
        # For boat agents: stores full state including orientation
        self.agent_states = {}
        
        # Dictionary to store agent model types {agent_id: model_type}
        self.agent_model_types = {}
        
        # Dictionary to store agent subscribers {agent_id: subscriber}
        self.agent_subscribers = {}
        
        # Dictionary to store obstacle subscribers {agent_id: subscriber}
        self.obstacle_subscribers = {}
        
        # Dictionary to store target estimate subscribers {agent_id: subscriber}
        self.target_estimate_subscribers = {}
        
        # Dictionary to store path publishers {agent_id: publisher}
        self.path_publishers = {}
        
        # Dictionary to store agent paths {agent_id: Path}
        self.agent_paths = {}
        
        # Set of currently discovered agent IDs
        self.discovered_agents = set()
        
        # Dictionary to store obstacles from all agents {agent_id: [obstacle_list]}
        self.agent_obstacles = {}
        
        # Dictionary to store target estimates from all agents {agent_id: {estimates: [...], ground_truths: [...]}}
        self.agent_target_estimates = {}
        
        # Publisher for agent markers
        self.marker_publisher = self.create_publisher(MarkerArray, 'agent_markers', 10)
        
        # Publisher for obstacle markers
        self.obstacle_marker_publisher = self.create_publisher(MarkerArray, 'obstacle_markers', 10)
        
        # Publisher for target estimate markers
        self.target_estimate_marker_publisher = self.create_publisher(MarkerArray, 'target_estimate_markers', 10)
        
        # Publisher for occupancy grid map
        self.map_publisher = self.create_publisher(OccupancyGrid, 'obstacle_map', 10)
        
        # Map parameters
        self.map_width = MAP_WIDTH             # Map width in meters (0 to 500)
        self.map_height = MAP_HEIGHT           # Map height in meters (0 to 500)
        self.map_resolution = MAP_RESOLUTION   # Grid resolution in meters/pixel (1 pixel per meter)
        self.grid_width = int(self.map_width / self.map_resolution)
        self.grid_height = int(self.map_height / self.map_resolution)
        
        # Timer to periodically discover new agents
        self.discovery_timer = self.create_timer(2.0, self.discover_agents)
        
        # Timer to publish markers at regular intervals
        self.marker_timer = self.create_timer(0.1, self.publish_agent_markers)  # 10 Hz
        
        # Timer to update and publish occupancy grid
        self.map_timer = self.create_timer(10.0, self.publish_occupancy_grid)  # Every 10 seconds
        
        # Timer to publish obstacle markers
        self.obstacle_marker_timer = self.create_timer(5.0, self.publish_obstacle_markers)  # Every 5 seconds
        
        # Timer to publish target estimate markers
        self.target_estimate_marker_timer = self.create_timer(0.5, self.publish_target_estimate_markers)  # Every 0.5 seconds

        self.get_logger().info(f'Environment node initialized with map size: {self.grid_width}x{self.grid_height} pixels')


    def discover_agents(self):
        """Discover active agent nodes and create subscribers for them"""
        # Get list of all nodes
        node_names = self.get_node_names()
        
        # Look for agent nodes (pattern: 'agent_X' where X is a number)
        agent_pattern = re.compile(r'agent[_\-]?(\d+)', re.IGNORECASE)
        current_agents = set()
        
        for node_name in node_names:
            match = agent_pattern.search(node_name)
            if match:
                agent_id = int(match.group(1))
                current_agents.add(agent_id)
                
                # Create subscriber if this is a new agent
                if agent_id not in self.discovered_agents:
                    self.create_agent_subscriber(agent_id)
        
        # Remove subscribers for agents that are no longer active
        inactive_agents = self.discovered_agents - current_agents
        for agent_id in inactive_agents:
            self.remove_agent_subscriber(agent_id)
        
        # Update discovered agents and log changes
        if current_agents != self.discovered_agents:
            newly_discovered = current_agents - self.discovered_agents
            newly_lost = self.discovered_agents - current_agents
            
            if newly_discovered:
                self.get_logger().info(f'Discovered new agents: {sorted(newly_discovered)}')
            if newly_lost:
                self.get_logger().info(f'Lost agents: {sorted(newly_lost)}')
                
            self.get_logger().info(f'Active agents: {sorted(current_agents)}')
            self.discovered_agents = current_agents

    def get_agent_model_type(self, agent_id):
        """Query the model_type parameter from an agent node"""
        try:
            # Try to get the parameter using a simple approach
            # We'll delay the parameter query and try it multiple times if needed
            import subprocess
            import time
            
            # Use ros2 param get command as a fallback to verify the approach
            try:
                result = subprocess.run([
                    'ros2', 'param', 'get', f'agent_{agent_id}', 'model_type'
                ], capture_output=True, text=True, timeout=5.0)
                
                if result.returncode == 0:
                    # Parse the output: "String value is: SimpleBoatSecondOrder"
                    output = result.stdout.strip()
                    if "String value is:" in output:
                        model_type = output.split("String value is:")[-1].strip()
                        self.agent_model_types[agent_id] = model_type
                        # self.get_logger().info(f'Agent {agent_id} model type: {model_type}')
                        return model_type
                else:
                    self.get_logger().warn(f'Failed to get model_type parameter for agent_{agent_id}: {result.stderr}')
                    
            except subprocess.TimeoutExpired:
                self.get_logger().warn(f'Timeout getting model_type for agent_{agent_id} via subprocess')
            except Exception as e:
                self.get_logger().warn(f'Subprocess error for agent_{agent_id}: {str(e)}')
                
        except Exception as e:
            self.get_logger().error(f'Error getting model_type for agent_{agent_id}: {str(e)}')
        
        return None

    def create_agent_subscriber(self, agent_id):
        """Create a subscriber for a specific agent's data topic"""
        try:
            # Get the agent's model type
            self.get_agent_model_type(agent_id)
            
            # Subscribe to agent data
            data_subscriber = self.create_subscription(
                AgentData,
                f'agent_{agent_id}/data',
                lambda msg, aid=agent_id: self.agent_data_callback(msg, aid),
                10
            )
            self.agent_subscribers[agent_id] = data_subscriber
            
            # Subscribe to agent obstacles
            obstacle_subscriber = self.create_subscription(
                MultipleObstacles,
                f'agent_{agent_id}/known_obstacles',
                lambda msg, aid=agent_id: self.obstacle_callback(msg, aid),
                10
            )
            self.obstacle_subscribers[agent_id] = obstacle_subscriber
            
            # Subscribe to agent target estimates
            target_estimate_subscriber = self.create_subscription(
                MultipleTargetEstimates,
                f'agent_{agent_id}/target_estimates',
                lambda msg, aid=agent_id: self.target_estimate_callback(msg, aid),
                10
            )
            self.target_estimate_subscribers[agent_id] = target_estimate_subscriber
            
            # Create path publisher for this agent
            path_publisher = self.create_publisher(
                Path,
                f'agent_{agent_id}/path',
                10
            )
            self.path_publishers[agent_id] = path_publisher
            
            # Initialize empty path for this agent
            self.agent_paths[agent_id] = Path()
            self.agent_paths[agent_id].header.frame_id = "map"
            self.agent_paths[agent_id].poses = []
            
            self.get_logger().info(f'Created subscribers and path publisher for agent_{agent_id}')
            
        except Exception as e:
            self.get_logger().error(f'Failed to create subscribers for agent_{agent_id}: {str(e)}')

    def remove_agent_subscriber(self, agent_id):
        """Remove subscriber and data for an inactive agent"""
        if agent_id in self.agent_subscribers:
            # Destroy the data subscriber
            self.destroy_subscription(self.agent_subscribers[agent_id])
            del self.agent_subscribers[agent_id]
            
        if agent_id in self.obstacle_subscribers:
            # Destroy the obstacle subscriber
            self.destroy_subscription(self.obstacle_subscribers[agent_id])
            del self.obstacle_subscribers[agent_id]
            
        if agent_id in self.target_estimate_subscribers:
            # Destroy the target estimate subscriber
            self.destroy_subscription(self.target_estimate_subscribers[agent_id])
            del self.target_estimate_subscribers[agent_id]
            
        if agent_id in self.path_publishers:
            # Destroy the path publisher
            self.destroy_publisher(self.path_publishers[agent_id])
            del self.path_publishers[agent_id]
            
        self.get_logger().info(f'Removed subscribers and path publisher for agent_{agent_id}')
        
        # Remove stored position data for this agent
        if agent_id in self.agent_positions:
            del self.agent_positions[agent_id]
            
        # Remove stored state data for this agent
        if agent_id in self.agent_states:
            del self.agent_states[agent_id]
            
        # Remove stored model type for this agent
        if agent_id in self.agent_model_types:
            del self.agent_model_types[agent_id]
            
        # Remove stored obstacle data for this agent
        if agent_id in self.agent_obstacles:
            del self.agent_obstacles[agent_id]
            
        # Remove stored target estimate data for this agent
        if agent_id in self.agent_target_estimates:
            del self.agent_target_estimates[agent_id]
            
        # Remove stored path data for this agent
        if agent_id in self.agent_paths:
            del self.agent_paths[agent_id]
            
        # Publish a delete marker for this agent
        self.publish_delete_marker(agent_id)

    def publish_delete_marker(self, agent_id):
        """Publish DELETE markers to remove agent visualization, path, and target estimates"""
        marker_array = MarkerArray()
        
        # Delete agent marker
        agent_marker = Marker()
        agent_marker.header.frame_id = "map"
        agent_marker.header.stamp = self.get_clock().now().to_msg()
        agent_marker.ns = "agents"
        agent_marker.id = agent_id
        agent_marker.action = Marker.DELETE
        marker_array.markers.append(agent_marker)
        
        # Delete target estimate markers for this agent
        # We need to delete a range of possible target estimate markers
        for marker_id in range(100):  # Assume max 100 target estimates per agent
            target_marker = Marker()
            target_marker.header.frame_id = "map"
            target_marker.header.stamp = self.get_clock().now().to_msg()
            target_marker.ns = "target_estimates"
            target_marker.id = marker_id  # This will delete markers globally, but that's okay
            target_marker.action = Marker.DELETE
            marker_array.markers.append(target_marker)
        
        # Publish delete markers
        self.marker_publisher.publish(marker_array)
        
        # Publish empty path to clear the path visualization
        if agent_id in self.path_publishers:
            empty_path = Path()
            empty_path.header.frame_id = "map"
            empty_path.header.stamp = self.get_clock().now().to_msg()
            empty_path.poses = []  # Empty path
            self.path_publishers[agent_id].publish(empty_path)

    def agent_data_callback(self, msg, agent_id):
        """Callback function to handle AgentData messages from agents"""
        try:
            # Extract position from the states (assuming first 2-3 states are x, y, z)
            if len(msg.states) >= 2:
                x = msg.states[0]
                y = msg.states[1]
                # z = msg.states[2] if len(msg.states) >= 3 else 0.0
                
                # Store agent position
                self.agent_positions[agent_id] = (x, y, 0)
                
                # Check if this is a boat agent and store full state
                model_type = self.agent_model_types.get(agent_id, None)
                # print(f" Model tpye: {model_type}")
                if model_type == "SimpleBoatSecondOrder" and len(msg.states) >= 5:
                    # For boat: states are [x, y, psi, uvel, omegavel]
                    psi = msg.states[2]  # Yaw angle
                    uvel = msg.states[3]  # Linear velocity
                    omegavel = msg.states[4]  # Angular velocity
                    
                    self.agent_states[agent_id] = {
                        'x': x,
                        'y': y, 
                        'psi': psi,
                        'uvel': uvel,
                        'omegavel': omegavel,
                        'is_boat': True,
                        'is_car': False
                    }
                    
                    # Debug logging for boat detection
                    # self.get_logger().info(f'Agent {agent_id} detected as boat - model_type: {model_type}, psi: {psi:.2f}')
                    
                elif model_type == "SimpleCarSecondOrder" and len(msg.states) >= 6:
                    # For car: states are [x, y, psi, u, delta, omega]
                    psi = msg.states[2]    # Yaw angle
                    u_speed = msg.states[3]    # Forward velocity
                    delta = msg.states[4]  # Steering angle
                    omega = msg.states[5]  # Angular velocity
                    
                    self.agent_states[agent_id] = {
                        'x': x,
                        'y': y, 
                        'psi': psi,
                        'u_speed': u_speed,
                        'delta': delta,
                        'omega': omega,
                        'is_boat': False,
                        'is_car': True,
                        'is_airplane': False
                    }
                    
                    # Debug logging for car detection
                    # self.get_logger().info(f'Agent {agent_id} detected as car - model_type: {model_type}, psi: {psi:.2f}')
                    
                elif model_type == "FixedWing12DOFTrainer" and len(msg.states) >= 12:
                    # For airplane: states are [X, Y, Z, phi, theta, psi, u, v, w, p, q, r]
                    z_pos = msg.states[2]    # Altitude
                    phi = msg.states[3]      # Roll angle
                    theta = msg.states[4]    # Pitch angle
                    psi = msg.states[5]      # Yaw angle
                    u_vel = msg.states[6]    # Body-frame x velocity
                    v_vel = msg.states[7]    # Body-frame y velocity
                    w_vel = msg.states[8]    # Body-frame z velocity
                    p_rate = msg.states[9]   # Roll rate
                    q_rate = msg.states[10]  # Pitch rate
                    r_rate = msg.states[11]  # Yaw rate
                    
                    self.agent_states[agent_id] = {
                        'x': x,
                        'y': y,
                        'z': z_pos,
                        'phi': phi,
                        'theta': theta, 
                        'psi': psi,
                        'u_vel': u_vel,
                        'v_vel': v_vel,
                        'w_vel': w_vel,
                        'p_rate': p_rate,
                        'q_rate': q_rate,
                        'r_rate': r_rate,
                        'is_boat': False,
                        'is_car': False,
                        'is_airplane': True
                    }
                    
                    # Update position with actual Z coordinate for airplanes
                    self.agent_positions[agent_id] = (x, y, z_pos)
                    
                    # Debug logging for airplane detection
                    # self.get_logger().info(f'Agent {agent_id} detected as airplane - model_type: {model_type}, psi: {psi:.2f}, altitude: {z_pos:.2f}')
                    
                else:
                    # For non-boat, non-car, non-airplane agents, store basic position info
                    self.agent_states[agent_id] = {
                        'x': x,
                        'y': y,
                        'psi': 0.0,  # No orientation
                        'is_boat': False,
                        'is_car': False,
                        'is_airplane': False
                    }
                    
                    # Debug logging for non-boat, non-car detection
                    # if model_type:
                    #     self.get_logger().info(f'Agent {agent_id} detected as non-boat/non-car - model_type: {model_type}')
                    # else:
                    #     self.get_logger().info(f'Agent {agent_id} model_type not yet determined')
                
                # Update and publish agent path
                # Determine z coordinate based on agent type
                z_coord = 0.0  # Default for ground vehicles
                if agent_id in self.agent_states and self.agent_states[agent_id].get('is_airplane', False):
                    z_coord = self.agent_states[agent_id].get('z', 0.0)
                
                self.update_agent_path(agent_id, x, y, msg.header, z_coord)
                
                # Log position updates (can be commented out to reduce verbosity)
                # self.get_logger().info(f'Agent {agent_id} position: ({x:.2f}, {y:.2f}, {z:.2f})')
            
        except Exception as e:
            self.get_logger().error(f'Error processing data from agent_{agent_id}: {str(e)}')

    def update_agent_path(self, agent_id, x, y, header, z=0.0):
        """Update and publish the path for a specific agent"""
        try:
            # Check if agent path exists, if not create it
            if agent_id not in self.agent_paths:
                self.agent_paths[agent_id] = Path()
                self.agent_paths[agent_id].header.frame_id = "map"
                self.agent_paths[agent_id].poses = []
            
            # Create a new PoseStamped for this position
            pose_stamped = PoseStamped()
            pose_stamped.header = header
            pose_stamped.header.frame_id = "map"  # Ensure frame_id is map
            
            # Set position (z coordinate will be 0.0 for ground vehicles, actual altitude for airplanes)
            pose_stamped.pose.position.x = x
            pose_stamped.pose.position.y = y
            pose_stamped.pose.position.z = z if (self.agent_states[agent_id].get('is_airplane', False) \
                                                 or self.agent_states[agent_id].get('is_boat', False) \
                                                 or self.agent_states[agent_id].get('is_car', False)) else 4.0
            
            # Set orientation (no rotation for now)
            pose_stamped.pose.orientation.x = 0.0
            pose_stamped.pose.orientation.y = 0.0
            pose_stamped.pose.orientation.z = 0.0
            pose_stamped.pose.orientation.w = 1.0
            
            # Add to path (with filtering to avoid too many close points)
            should_add = True
            if len(self.agent_paths[agent_id].poses) > 0:
                last_pose = self.agent_paths[agent_id].poses[-1]
                dx = x - last_pose.pose.position.x
                dy = y - last_pose.pose.position.y
                dz = z - last_pose.pose.position.z  # Include Z distance for airplanes
                distance = (dx*dx + dy*dy + dz*dz)**0.5
                
                # Only add if the agent moved at least 5cm in 3D space
                if distance < 0.05:
                    should_add = False
            
            if should_add:
                self.agent_paths[agent_id].poses.append(pose_stamped)
                
                # Update path header
                self.agent_paths[agent_id].header.stamp = header.stamp
                self.agent_paths[agent_id].header.frame_id = "map"
                
                # Publish the updated path
                if agent_id in self.path_publishers:
                    self.path_publishers[agent_id].publish(self.agent_paths[agent_id])
                
                # Optional: Limit path length to prevent memory issues (keep last 1000 points)
                if len(self.agent_paths[agent_id].poses) > 1000:
                    self.agent_paths[agent_id].poses = self.agent_paths[agent_id].poses[-1000:]
                
        except Exception as e:
            self.get_logger().error(f'Error updating path for agent_{agent_id}: {str(e)}')

    def obstacle_callback(self, msg, agent_id):
        """Callback function to handle obstacle data from agents"""
        try:
            obstacles = []
            for obs_msg in msg.obstacles:
                obstacle_info = {
                    'name': obs_msg.obs_name,
                    'type': obs_msg.obs_type,
                    'position': (obs_msg.position.x, obs_msg.position.y),
                    'dimensions': obs_msg.dimensions,
                    'kappa': obs_msg.kappa,
                    'rho0': obs_msg.rho0
                }
                # Append only if name not of the form "agent_{id}"
                if not re.match(r'agent[_\-]?(\d+)', obstacle_info['name'], re.IGNORECASE):
                    obstacles.append(obstacle_info)
            
            # Store obstacles for this agent
            self.agent_obstacles[agent_id] = obstacles
            
            # Log obstacle reception (can be commented out to reduce verbosity)
            # self.get_logger().info(f'Received {len(obstacles)} obstacles from agent_{agent_id}')
            
        except Exception as e:
            self.get_logger().error(f'Error processing obstacles from agent_{agent_id}: {str(e)}')

    def target_estimate_callback(self, msg, agent_id):
        """Callback function to handle target estimate data from agents"""
        try:
            estimates = []
            ground_truths = []
            
            # Process target estimates
            for est_msg in msg.target_estimates:
                estimate = {
                    'target_id': est_msg.target_id,
                    'position': (est_msg.position.x, est_msg.position.y, est_msg.position.z),
                    'covariance': est_msg.covariance
                }
                estimates.append(estimate)
            
            # Process ground truths
            for gt_msg in msg.ground_truths:
                ground_truth = {
                    'target_id': gt_msg.target_id,
                    'position': (gt_msg.position.x, gt_msg.position.y, gt_msg.position.z)
                }
                ground_truths.append(ground_truth)
            
            # Store target estimates for this agent
            self.agent_target_estimates[agent_id] = {
                'estimates': estimates,
                'ground_truths': ground_truths
            }
            
            # Log target estimate reception (can be commented out to reduce verbosity)
            # self.get_logger().info(f'Received {len(estimates)} target estimates from agent_{agent_id}')
            
        except Exception as e:
            self.get_logger().error(f'Error processing target estimates from agent_{agent_id}: {str(e)}')

    def world_to_grid(self, x, y):
        """Convert world coordinates to grid coordinates"""
        grid_x = int(x / self.map_resolution)
        grid_y = int(y / self.map_resolution)
        
        # Clamp to grid bounds
        grid_x = max(0, min(grid_x, self.grid_width - 1))
        grid_y = max(0, min(grid_y, self.grid_height - 1))
        
        return grid_x, grid_y

    def fill_circle_obstacle(self, grid, center_x, center_y, radius):
        """Fill a circular obstacle in the grid"""
        grid_radius = int(radius / self.map_resolution)
        center_grid_x, center_grid_y = self.world_to_grid(center_x, center_y)
        
        for dy in range(-grid_radius, grid_radius + 1):
            for dx in range(-grid_radius, grid_radius + 1):
                if dx*dx + dy*dy <= grid_radius*grid_radius:
                    grid_x = center_grid_x + dx
                    grid_y = center_grid_y + dy
                    
                    if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
                        grid_idx = grid_y * self.grid_width + grid_x
                        grid[grid_idx] = 100  # Occupied

    def fill_rectangle_obstacle(self, grid, center_x, center_y, width, height):
        """Fill a rectangular obstacle in the grid"""
        # Calculate rectangle bounds in world coordinates
        half_width = width / 2.0
        half_height = height / 2.0
        
        x_min = center_x - half_width
        x_max = center_x + half_width
        y_min = center_y - half_height
        y_max = center_y + half_height
        
        # Convert to grid coordinates
        grid_x_min, grid_y_min = self.world_to_grid(x_min, y_min)
        grid_x_max, grid_y_max = self.world_to_grid(x_max, y_max)
        
        for grid_y in range(grid_y_min, grid_y_max + 1):
            for grid_x in range(grid_x_min, grid_x_max + 1):
                if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
                    grid_idx = grid_y * self.grid_width + grid_x
                    grid[grid_idx] = 100  # Occupied

    def fill_wall_obstacle(self, grid, wall_x, wall_y, normal_x, normal_y, wall_length):
        """Fill a wall obstacle in the grid (simplified as a line)"""
        # For simplicity, treat wall as a thick line along the normal direction
        # This is a basic implementation - you might want to enhance this based on your wall model
        
        # Create a small rectangular obstacle perpendicular to the normal
        wall_thickness = WALL_THICKNESS  # 10cm thick wall
        wall_length = wall_length

        # Calculate rectangle dimensions based on normal
        if abs(normal_x) > abs(normal_y):  # More horizontal normal
            width = wall_thickness
            height = wall_length
        else:  # More vertical normal
            width = wall_length
            height = wall_thickness
            
        self.fill_rectangle_obstacle(grid, wall_x, wall_y, width, height)

    def publish_occupancy_grid(self):
        """Create and publish occupancy grid map with obstacles"""
        # Initialize grid with free space (0)
        grid = np.zeros(self.grid_width * self.grid_height, dtype=np.int8)
        
        # Collect all obstacles from all agents
        all_obstacles = []
        for agent_id, obstacles in self.agent_obstacles.items():
            all_obstacles.extend(obstacles)
        
        # Fill obstacles in grid
        for obstacle in all_obstacles:
            obs_type = obstacle['type']
            pos_x, pos_y = obstacle['position']
            dimensions = obstacle['dimensions']
            
            try:
                if obs_type == "circle" and len(dimensions) == 1:
                    radius = dimensions[0]
                    self.fill_circle_obstacle(grid, pos_x, pos_y, radius)
                    
                elif obs_type == "rectangle" and len(dimensions) == 2:
                    width = dimensions[0]
                    height = dimensions[1]
                    self.fill_rectangle_obstacle(grid, pos_x, pos_y, width, height)
                    
                elif obs_type == "wall" and len(dimensions) == 3:
                    normal_x = dimensions[0]
                    normal_y = dimensions[1]
                    wall_length = dimensions[2]
                    self.fill_wall_obstacle(grid, pos_x, pos_y, normal_x, normal_y, wall_length)
                    
            except Exception as e:
                self.get_logger().warn(f'Error processing obstacle {obstacle["name"]}: {str(e)}')
        
        # Create OccupancyGrid message
        occupancy_grid = OccupancyGrid()
        
        # Set header
        occupancy_grid.header.stamp = self.get_clock().now().to_msg()
        occupancy_grid.header.frame_id = "map"
        
        # Set map metadata
        occupancy_grid.info = MapMetaData()
        occupancy_grid.info.resolution = self.map_resolution
        occupancy_grid.info.width = self.grid_width
        occupancy_grid.info.height = self.grid_height
        occupancy_grid.info.map_load_time = self.get_clock().now().to_msg()
        
        # Set origin (bottom-left corner of the map)
        occupancy_grid.info.origin.position.x = 0.0
        occupancy_grid.info.origin.position.y = 0.0
        occupancy_grid.info.origin.position.z = 0.0
        occupancy_grid.info.origin.orientation.x = 0.0
        occupancy_grid.info.origin.orientation.y = 0.0
        occupancy_grid.info.origin.orientation.z = 0.0
        occupancy_grid.info.origin.orientation.w = 1.0
        
        # Set grid data
        occupancy_grid.data = grid.tolist()
        
        # Publish the occupancy grid
        self.map_publisher.publish(occupancy_grid)
        
        # Log publication (can be commented out to reduce verbosity)
        obstacle_count = len(all_obstacles)
        if obstacle_count > 0:
            self.get_logger().info(f'Published occupancy grid with {obstacle_count} obstacles')

    def publish_obstacle_markers(self):
        """Create and publish MarkerArray with obstacle visualization"""
        marker_array = MarkerArray()
        
        # Collect all obstacles from all agents
        all_obstacles = []
        for agent_id, obstacles in self.agent_obstacles.items():
            for obstacle in obstacles:
                # Add agent_id to obstacle info for unique marker IDs
                obstacle_with_agent = obstacle.copy()
                obstacle_with_agent['agent_id'] = agent_id
                all_obstacles.append(obstacle_with_agent)
        
        # Create markers for each obstacle
        marker_id = 0
        for obstacle in all_obstacles:
            try:
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "obstacles"
                marker.id = marker_id
                marker.action = Marker.ADD
                
                # Set position
                pos_x, pos_y = obstacle['position']
                marker.pose.position.x = float(pos_x)
                marker.pose.position.y = float(pos_y)
                marker.pose.position.z = 0.0
                
                # Set orientation (no rotation for now)
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.orientation.w = 1.0
                
                # Set color (semi-transparent black for obstacles)
                marker.color.r = 0.2
                marker.color.g = 0.2
                marker.color.b = 0.2
                marker.color.a = 1.0

                # Set lifetime
                marker.lifetime.sec = 0
                marker.lifetime.nanosec = 0
                
                # Set marker type and scale based on obstacle type
                obs_type = obstacle['type']
                dimensions = obstacle['dimensions']
                
                if obs_type == "circle" and len(dimensions) >= 1:
                    marker.type = Marker.CYLINDER
                    radius = dimensions[0]
                    marker.scale.x = float(radius * 2.0)  # Diameter
                    marker.scale.y = float(radius * 2.0)  # Diameter
                    marker.scale.z = 2.5  # Height for visualization
                    
                elif obs_type == "rectangle" and len(dimensions) >= 2:
                    marker.type = Marker.CUBE
                    width = dimensions[0]
                    height = dimensions[1]
                    marker.scale.x = float(width)
                    marker.scale.y = float(height)
                    marker.scale.z = 2.5  # Height for visualization

                    # Make them grey
                    marker.color.r = 0.5
                    marker.color.g = 0.5
                    marker.color.b = 0.5
                    marker.color.a = 1.0

                elif obs_type == "wall" and len(dimensions) >= 2:
                    marker.type = Marker.CUBE
                    normal_x = dimensions[0]
                    normal_y = dimensions[1]
                    
                    # Create a wall representation
                    wall_thickness = 0.1  # 10cm thick
                    wall_length = 10.0     # 10m long

                    if abs(normal_x) > abs(normal_y):  # More horizontal normal
                        marker.scale.x = wall_thickness
                        marker.scale.y = wall_length
                    else:  # More vertical normal
                        marker.scale.x = wall_length
                        marker.scale.y = wall_thickness
                    marker.scale.z = 1.0  # Height for visualization
                    
                    # Set different color for walls
                    marker.color.r = 0.3
                    marker.color.g = 0.3
                    marker.color.b = 0.3
                    marker.color.a = 1.0

                else:
                    # Default marker for unknown obstacle types
                    marker.type = Marker.SPHERE
                    marker.scale.x = 0.5
                    marker.scale.y = 0.5
                    marker.scale.z = 0.5
                    
                    # Set different color for unknown obstacles
                    marker.color.r = 0.8
                    marker.color.g = 0.8
                    marker.color.b = 0.2
                    marker.color.a = 0.7
                
                marker_array.markers.append(marker)
                marker_id += 1
                
            except Exception as e:
                self.get_logger().warn(f'Error creating marker for obstacle {obstacle.get("name", "unknown")}: {str(e)}')
        
        # Add delete markers for any obstacles that might have been removed
        # (This helps clean up old markers when obstacles are no longer present)
        current_marker_count = len(marker_array.markers)
        max_markers = 100  # Assume maximum of 100 obstacle markers
        
        for i in range(current_marker_count, max_markers):
            delete_marker = Marker()
            delete_marker.header.frame_id = "map"
            delete_marker.header.stamp = self.get_clock().now().to_msg()
            delete_marker.ns = "obstacles"
            delete_marker.id = i
            delete_marker.action = Marker.DELETE
            marker_array.markers.append(delete_marker)
        
        # Publish the marker array
        self.obstacle_marker_publisher.publish(marker_array)
        
        # Log publication (can be commented out to reduce verbosity)
        if len(all_obstacles) > 0:
            self.get_logger().info(f'Published {len(all_obstacles)} obstacle markers')
        elif len(all_obstacles) == 0 and hasattr(self, '_last_obstacle_count') and self._last_obstacle_count > 0:
            self.get_logger().info('Published empty obstacle marker array (clearing old obstacles)')
        
        # Remember the last obstacle count for logging
        self._last_obstacle_count = len(all_obstacles)

    def publish_target_estimate_markers(self):
        """Create and publish MarkerArray with target estimate visualization"""
        marker_array = MarkerArray()
        
        # Collect all target estimates from all agents
        marker_id = 0
        
        for agent_id, target_data in self.agent_target_estimates.items():
            if agent_id in self.discovered_agents:  # Only process active agents
                agent_color = self.get_agent_color(agent_id)
                
                # Create markers for target estimates (small X markers)
                for estimate in target_data.get('estimates', []):
                    try:
                        marker = Marker()
                        marker.header.frame_id = "map"
                        marker.header.stamp = self.get_clock().now().to_msg()
                        marker.ns = "target_estimates"
                        marker.id = marker_id
                        marker.action = Marker.ADD
                        marker.type = Marker.TEXT_VIEW_FACING
                        
                        # Set position from estimate
                        pos_x, pos_y, pos_z = estimate['position']
                        marker.pose.position.x = float(pos_x)
                        marker.pose.position.y = float(pos_y)
                        marker.pose.position.z = float(pos_z) + 0.1  # Raise slightly above ground
                        
                        # Set orientation (no rotation)
                        marker.pose.orientation.x = 0.0
                        marker.pose.orientation.y = 0.0
                        marker.pose.orientation.z = 0.0
                        marker.pose.orientation.w = 1.0
                        
                        # Set scale for text
                        marker.scale.z = 0.3  # Text height
                        
                        # Set color to match the agent that estimated it
                        marker.color = agent_color
                        
                        # Set the text to display as 'X'
                        marker.text = "X"
                        
                        # Set lifetime
                        marker.lifetime.sec = 0
                        marker.lifetime.nanosec = 0
                        
                        marker_array.markers.append(marker)
                        marker_id += 1
                        
                    except Exception as e:
                        self.get_logger().warn(f'Error creating target estimate marker for agent {agent_id}: {str(e)}')
                
                # Create markers for ground truth targets (black X markers)
                # Note: We'll only show ground truths from one agent to avoid duplicates
                if agent_id == min(self.discovered_agents):  # Use the lowest ID agent
                    for ground_truth in target_data.get('ground_truths', []):
                        try:
                            marker = Marker()
                            marker.header.frame_id = "map"
                            marker.header.stamp = self.get_clock().now().to_msg()
                            marker.ns = "ground_truths"
                            marker.id = marker_id
                            marker.action = Marker.ADD
                            marker.type = Marker.CUBE
                            
                            # Set position from ground truth
                            pos_x, pos_y, pos_z = ground_truth['position']
                            marker.pose.position.x = float(pos_x)
                            marker.pose.position.y = float(pos_y)
                            marker.pose.position.z = float(pos_z) + 0.1  # Raise slightly above ground
                            
                            # Set orientation (no rotation)
                            marker.pose.orientation.x = 0.0
                            marker.pose.orientation.y = 0.0
                            marker.pose.orientation.z = 0.0
                            marker.pose.orientation.w = 1.0
                            
                            # Set scale for small cube
                            marker.scale.x = 0.2  # Small cube size
                            marker.scale.y = 0.2  # Small cube size
                            marker.scale.z = 0.2  # Small cube size
                            
                            # Set color to red for ground truth
                            marker.color.r = 1.0
                            marker.color.g = 0.0
                            marker.color.b = 0.0
                            marker.color.a = 0.4
                            
                            # Set lifetime
                            marker.lifetime.sec = 0
                            marker.lifetime.nanosec = 0
                            
                            marker_array.markers.append(marker)
                            marker_id += 1
                            
                        except Exception as e:
                            self.get_logger().warn(f'Error creating ground truth marker for agent {agent_id}: {str(e)}')
        
        # Add delete markers for any target estimates that might have been removed
        current_marker_count = len(marker_array.markers)
        max_estimate_markers = 200  # Assume maximum of 200 target estimate markers
        
        # Delete old estimate markers
        for i in range(current_marker_count, max_estimate_markers):
            delete_marker = Marker()
            delete_marker.header.frame_id = "map"
            delete_marker.header.stamp = self.get_clock().now().to_msg()
            delete_marker.ns = "target_estimates"
            delete_marker.id = i
            delete_marker.action = Marker.DELETE
            marker_array.markers.append(delete_marker)
        
        # Delete old ground truth markers
        for i in range(current_marker_count, 50):  # Assume max 50 ground truth markers
            delete_marker = Marker()
            delete_marker.header.frame_id = "map"
            delete_marker.header.stamp = self.get_clock().now().to_msg()
            delete_marker.ns = "ground_truths"
            delete_marker.id = i
            delete_marker.action = Marker.DELETE
            marker_array.markers.append(delete_marker)
        
        # Publish the marker array
        self.target_estimate_marker_publisher.publish(marker_array)
        
        # Log publication (can be commented out to reduce verbosity)
        total_estimates = sum(len(data.get('estimates', [])) for data in self.agent_target_estimates.values())
        total_ground_truths = sum(len(data.get('ground_truths', [])) for data in self.agent_target_estimates.values())
        
        # if total_estimates > 0 or total_ground_truths > 0:
        #     self.get_logger().info(f'Published {total_estimates} target estimate markers and {total_ground_truths} ground truth markers')

    def get_agent_color(self, agent_id):
        """Generate a unique color for each agent based on their ID"""
        # Use HSV color space to generate distinct colors
        hue = (agent_id * 137.5) % 360  # Golden angle for good color distribution
        saturation = 0.8
        value = 0.9
        
        # Convert HSV to RGB
        r, g, b = colorsys.hsv_to_rgb(hue / 360.0, saturation, value)
        
        color = ColorRGBA()
        color.r = float(r)
        color.g = float(g)
        color.b = float(b)
        color.a = 0.8  # Slight transparency
        
        return color

    def yaw_to_quaternion(self, yaw):
        """Convert yaw angle to quaternion"""
        # Convert yaw to quaternion (rotation around z-axis)
        half_yaw = yaw * 0.5
        qz = math.sin(half_yaw)
        qw = math.cos(half_yaw)
        return 0.0, 0.0, qz, qw  # qx, qy, qz, qw

    def euler_to_quaternion(self, roll, pitch, yaw):
        """Convert Euler angles (roll, pitch, yaw) to quaternion"""
        # Convert Euler angles to quaternion
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        return qx, qy, qz, qw

    def publish_agent_markers(self):
        """Publish MarkerArray with cylinder markers for each active agent"""
        marker_array = MarkerArray()
        
        # Create a marker for each active agent with position data
        for agent_id in sorted(self.discovered_agents):
            if agent_id in self.agent_positions and agent_id in self.agent_states:
                position = self.agent_positions[agent_id]
                state = self.agent_states[agent_id]
                
                marker = Marker()
                marker.header.frame_id = "map"  # Change this to your desired frame
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "agents"
                marker.id = agent_id
                marker.action = Marker.ADD
                
                # Set position
                marker.pose.position.x = float(position[0])
                marker.pose.position.y = float(position[1])
                marker.pose.position.z = float(position[2]) + 0.5  # Raise marker slightly above ground
                
                # Debug logging
                is_boat = state.get('is_boat', False)
                is_car = state.get('is_car', False)
                is_airplane = state.get('is_airplane', False)
                model_type = self.agent_model_types.get(agent_id, 'Unknown')
                # self.get_logger().info(f'Creating marker for agent {agent_id}: model_type={model_type}, is_boat={is_boat}, is_car={is_car}, is_airplane={is_airplane}')
                
                # Check if this is a boat agent
                if is_boat:
                    # Use STL mesh marker for boats to show orientation
                    marker.type = Marker.MESH_RESOURCE
                    
                    # Use package relative path for the STL file
                    marker.mesh_resource = "package://ergodic_exploration/meshes/boat.stl"
                    
                    # Set mesh orientation based on yaw angle (psi)
                    psi = state['psi'] + np.pi
                    qx, qy, qz, qw = self.yaw_to_quaternion(psi)
                    marker.pose.orientation.x = qx
                    marker.pose.orientation.y = qy
                    marker.pose.orientation.z = qz
                    marker.pose.orientation.w = qw
                    
                    # Set mesh scale (start with small scale to ensure visibility)
                    marker.scale.x = 1.0  # Small scale to test visibility
                    marker.scale.y = 1.0  # Small scale to test visibility
                    marker.scale.z = 1.0  # Small scale to test visibility

                    # Adjust z position for mesh
                    marker.pose.position.z = float(position[2]) + 0.1
                    
                    # Enable mesh_use_embedded_materials if needed
                    marker.mesh_use_embedded_materials = False
                    
                    # Debug logging for mesh marker
                    # self.get_logger().info(f'Agent {agent_id}: Using BOAT MESH marker, mesh_resource={marker.mesh_resource}, scale={marker.scale.x}')
                    
                # Check if this is a car agent
                elif is_car:
                    # Use STL mesh marker for cars to show orientation
                    marker.type = Marker.MESH_RESOURCE
                    
                    # Use package relative path for the STL file
                    marker.mesh_resource = "package://ergodic_exploration/meshes/car.stl"
                    
                    # Set mesh orientation based on yaw angle (psi)
                    psi = state['psi'] + np.pi
                    qx, qy, qz, qw = self.yaw_to_quaternion(psi)
                    marker.pose.orientation.x = qx
                    marker.pose.orientation.y = qy
                    marker.pose.orientation.z = qz
                    marker.pose.orientation.w = qw
                    
                    # Set mesh scale (start with small scale to ensure visibility)
                    marker.scale.x = 1.0  # Small scale to test visibility
                    marker.scale.y = 1.0  # Small scale to test visibility
                    marker.scale.z = 1.0  # Small scale to test visibility

                    # Adjust z position for mesh
                    marker.pose.position.z = float(position[2]) + 0.1
                    
                    # Enable mesh_use_embedded_materials if needed
                    marker.mesh_use_embedded_materials = False
                    
                    # Debug logging for mesh marker
                    # self.get_logger().info(f'Agent {agent_id}: Using CAR MESH marker, mesh_resource={marker.mesh_resource}, scale={marker.scale.x}')
                    
                # Check if this is an airplane agent
                elif is_airplane:
                    # Use STL mesh marker for airplanes to show full 3D orientation
                    marker.type = Marker.MESH_RESOURCE
                    
                    # Use package relative path for the STL file
                    marker.mesh_resource = "package://ergodic_exploration/meshes/hermes.stl"
                    
                    # Set mesh orientation based on roll, pitch, and yaw angles
                    phi = state.get('phi', 0.0)      # Roll
                    phi = -phi
                    theta = state.get('theta', 0.0)  # Pitch
                    psi = state.get('psi', 0.0)      # Yaw
                    
                    # Convert Euler angles to quaternion
                    qx, qy, qz, qw = self.euler_to_quaternion(phi, theta, psi)
                    marker.pose.orientation.x = qx
                    marker.pose.orientation.y = qy
                    marker.pose.orientation.z = qz
                    marker.pose.orientation.w = qw
                    
                    # Set mesh scale
                    marker.scale.x = 1.0
                    marker.scale.y = 1.0
                    marker.scale.z = 1.0

                    # Use the actual Z position from the state
                    marker.pose.position.z = float(state.get('z', position[2]))

                    # Its difficult to follow the marker arround, so lets just fix it in the center
                    # Remove the following lines and the plane will go as expected
                    marker.pose.position.x = 0.0
                    marker.pose.position.y = 0.0
                    marker.pose.position.z = 0.0
                    
                    # Enable mesh_use_embedded_materials if needed
                    marker.mesh_use_embedded_materials = False
                    
                    # Debug logging for mesh marker
                    # self.get_logger().info(f'Agent {agent_id}: Using AIRPLANE MESH marker, mesh_resource={marker.mesh_resource}, altitude={marker.pose.position.z:.2f}')
                    
                else:
                    # Use drone mesh for non-boat, non-car, non-airplane agents
                    marker.type = Marker.MESH_RESOURCE
                    
                    # Use package relative path for the STL file
                    marker.mesh_resource = "package://ergodic_exploration/meshes/drone_small.stl"
                    
                    # Set orientation (no rotation)
                    marker.pose.orientation.x = 0.0
                    marker.pose.orientation.y = 0.0
                    marker.pose.orientation.z = 0.0
                    marker.pose.orientation.w = 1.0
                    
                    # Set scale (mesh dimensions)
                    marker.scale.x = 1.0
                    marker.scale.y = 1.0
                    marker.scale.z = 1.0
                    
                    # Adjust z position for mesh
                    marker.pose.position.z = float(position[2]) + 5.0
                    
                    # Enable mesh_use_embedded_materials if needed
                    marker.mesh_use_embedded_materials = False
                    
                    # self.get_logger().info(f'Agent {agent_id}: Using DRONE_SMALL MESH marker')
                
                # Set color based on agent ID
                marker.color = self.get_agent_color(agent_id)
                
                # Set lifetime (0 = forever until explicitly deleted)
                marker.lifetime.sec = 0
                marker.lifetime.nanosec = 0
                
                marker_array.markers.append(marker)
        
        # Publish the marker array
        self.marker_publisher.publish(marker_array)
        
        # Optional: Log when publishing (can be commented out to reduce verbosity)
        # if len(marker_array.markers) > 0:
        #     self.get_logger().info(f'Published {len(marker_array.markers)} agent markers')

    def get_agent_position(self, agent_id):
        """Get the current position of a specific agent"""
        return self.agent_positions.get(agent_id, None)

    def get_all_agent_positions(self):
        """Get positions of all known agents"""
        return self.agent_positions.copy()

    def get_active_agent_ids(self):
        """Get list of currently active agent IDs"""
        return sorted(list(self.discovered_agents))

    def get_agent_path(self, agent_id):
        """Get the current path of a specific agent"""
        return self.agent_paths.get(agent_id, None)

    def get_all_agent_paths(self):
        """Get paths of all known agents"""
        return self.agent_paths.copy()

    def print_agent_status(self):
        """Print current status of all agents (useful for debugging)"""
        if not self.discovered_agents:
            self.get_logger().info('No active agents found')
            return
            
        self.get_logger().info(f'Active agents: {sorted(self.discovered_agents)}')
        for agent_id in sorted(self.discovered_agents):
            pos = self.agent_positions.get(agent_id, 'No position data')
            model_type = self.agent_model_types.get(agent_id, 'Unknown')
            state = self.agent_states.get(agent_id, {})
            
            if isinstance(pos, tuple):
                position_str = f'({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})'
                if state.get('is_boat', False):
                    psi = state.get('psi', 0.0)
                    self.get_logger().info(f'  Agent {agent_id} [{model_type}]: {position_str}, yaw: {psi:.2f} rad ({math.degrees(psi):.1f}°)')
                elif state.get('is_car', False):
                    psi = state.get('psi', 0.0)
                    self.get_logger().info(f'  Agent {agent_id} [{model_type}]: {position_str}, yaw: {psi:.2f} rad ({math.degrees(psi):.1f}°)')
                elif state.get('is_airplane', False):
                    phi = state.get('phi', 0.0)
                    theta = state.get('theta', 0.0)
                    psi = state.get('psi', 0.0)
                    self.get_logger().info(f'  Agent {agent_id} [{model_type}]: {position_str}, roll: {phi:.2f}, pitch: {theta:.2f}, yaw: {psi:.2f} rad')
                else:
                    self.get_logger().info(f'  Agent {agent_id} [{model_type}]: {position_str}')
            else:
                self.get_logger().info(f'  Agent {agent_id} [{model_type}]: {pos}')


def main(args=None):
    rclpy.init(args=args)
    
    try:
        environment_node = EnvironmentNode()
        
        # Add a timer to periodically print status (optional)
        def status_callback():
            environment_node.print_agent_status()
        
        # Print status every 10 seconds
        status_timer = environment_node.create_timer(10.0, status_callback)
        
        rclpy.spin(environment_node)
        
    except KeyboardInterrupt:
        pass
    finally:
        if 'environment_node' in locals():
            environment_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
