#!/usr/bin/env python3
"""
TF Visualizer for Aircraft and Obstacles

This node subscribes to aircraft agent data and known obstacles,
then publishes TF transforms for visualization in RViz.

Topics subscribed:
  - /agent_{id}/data (AgentData): Aircraft state data
  - /agent_{id}/known_obstacles (MultipleObstacles): Known obstacles

TF frames published:
  - map (parent frame)
  - aircraft_{id} (aircraft position and orientation)
  - obstacle_{name} (obstacle center with orientation based on type)

Author: Test script for aircraft visualization
"""

import rclpy
from rclpy.node import Node
import math
import tf2_ros
from geometry_msgs.msg import TransformStamped, PoseStamped
from nav_msgs.msg import Path
from my_interfaces.msg import AgentData, MultipleObstacles


class TFVisualizerAirplane(Node):
    def __init__(self, agent_id=1):
        super().__init__('tf_visualizer_airplane')
        
        self.agent_id = agent_id
        
        # Aircraft state storage
        self.aircraft_state = None
        
        # Obstacles storage
        self.obstacles = {}
        
        # Path storage for trajectory history
        self.path = Path()
        self.path.header.frame_id = "map"
        self.path.poses = []
        self.max_path_length = 2000  # Maximum number of poses to keep
        self.min_distance_threshold = 0.5  # Minimum distance between path points [m]
        
        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # Path publisher for trajectory visualization
        self.path_pub = self.create_publisher(
            Path,
            f'/agent_{self.agent_id}/trajectory',
            10
        )
        
        # Subscribe to aircraft data
        self.data_sub = self.create_subscription(
            AgentData,
            f'/agent_{self.agent_id}/data',
            self.agent_data_callback,
            10
        )
        
        # Subscribe to known obstacles
        self.obstacles_sub = self.create_subscription(
            MultipleObstacles,
            f'/agent_{self.agent_id}/known_obstacles',
            self.obstacles_callback,
            10
        )
        
        # Timer to publish TF transforms at 50 Hz
        self.tf_timer = self.create_timer(0.02, self.publish_transforms)
        
        self.get_logger().info(f'TF Visualizer for Aircraft initialized')
        self.get_logger().info(f'Subscribing to /agent_{self.agent_id}/data')
        self.get_logger().info(f'Subscribing to /agent_{self.agent_id}/known_obstacles')
        self.get_logger().info(f'Publishing trajectory path to /agent_{self.agent_id}/trajectory')
    
    def agent_data_callback(self, msg):
        """Callback for aircraft data - expects 12-state fixed-wing model"""
        try:
            if len(msg.states) >= 12:
                # Fixed-wing aircraft states: [X, Y, Z, phi, theta, psi, u, v, w, p, q, r]
                self.aircraft_state = {
                    'x': msg.states[0],
                    'y': msg.states[1],
                    'z': msg.states[2],
                    'phi': msg.states[3],      # Roll
                    'theta': msg.states[4],    # Pitch
                    'psi': msg.states[5],      # Yaw
                    'u': msg.states[6],        # Body x velocity
                    'v': msg.states[7],        # Body y velocity
                    'w': msg.states[8],        # Body z velocity
                    'p': msg.states[9],        # Roll rate
                    'q': msg.states[10],       # Pitch rate
                    'r': msg.states[11]        # Yaw rate
                }
                
                # Update trajectory path
                self.update_path(msg.header, msg.states[0], msg.states[1], msg.states[2])
        except Exception as e:
            self.get_logger().error(f'Error processing aircraft data: {e}')
    
    def obstacles_callback(self, msg):
        """Callback for known obstacles"""
        try:
            self.obstacles = {}
            for obs in msg.obstacles:
                obs_name = obs.obs_name if obs.obs_name else f'obs_{len(self.obstacles)}'
                
                self.obstacles[obs_name] = {
                    'type': obs.obs_type,
                    'x': obs.position.x,
                    'y': obs.position.y,
                    'z': obs.position.z,
                    'dimensions': list(obs.dimensions),
                    'kappa': obs.kappa,
                    'rho0': obs.rho0
                }
            
            # Log obstacle count on first receive or when count changes
            # self.get_logger().info(f'Received {len(self.obstacles)} obstacles')
            
        except Exception as e:
            self.get_logger().error(f'Error processing obstacles: {e}')
    
    def publish_transforms(self):
        """Publish TF transforms for aircraft and obstacles"""
        current_time = self.get_clock().now().to_msg()
        
        # Publish aircraft transform
        if self.aircraft_state is not None:
            self.publish_aircraft_transform(current_time)
        
        # Publish obstacle transforms
        for obs_name, obs_data in self.obstacles.items():
            self.publish_obstacle_transform(obs_name, obs_data, current_time)
    
    def publish_aircraft_transform(self, stamp):
        """Publish TF transform for the aircraft"""
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = "map"
        t.child_frame_id = f"aircraft_{self.agent_id}"
        
        # Set translation
        t.transform.translation.x = float(self.aircraft_state['x'])
        t.transform.translation.y = float(self.aircraft_state['y'])
        t.transform.translation.z = float(self.aircraft_state['z'])
        
        # Set rotation from Euler angles (phi, theta, psi)
        phi = self.aircraft_state['phi']
        theta = self.aircraft_state['theta']
        psi = self.aircraft_state['psi']
        qx, qy, qz, qw = self.euler_to_quaternion(phi, theta, psi)
        
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        
        self.tf_broadcaster.sendTransform(t)
    
    def publish_obstacle_transform(self, obs_name, obs_data, stamp):
        """Publish TF transform for an obstacle"""
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = "map"
        # Clean the obstacle name for TF frame (replace spaces, special chars)
        clean_name = obs_name.replace(' ', '_').replace('-', '_')
        t.child_frame_id = f"obstacle_{clean_name}"
        
        # Set translation (obstacle center)
        t.transform.translation.x = float(obs_data['x'])
        t.transform.translation.y = float(obs_data['y'])
        t.transform.translation.z = float(obs_data['z'])
        
        # Set rotation based on obstacle type
        obs_type = obs_data['type']
        dims = obs_data['dimensions']
        
        if obs_type == 'wall' and len(dims) >= 2:
            # Wall: dimensions are [normal_x, normal_y, wall_length]
            # Orient the frame so X-axis points along the wall normal
            normal_x = dims[0]
            normal_y = dims[1]
            yaw = math.atan2(normal_y, normal_x)
            qx, qy, qz, qw = self.yaw_to_quaternion(yaw)
        elif obs_type == 'rectangle' and len(dims) >= 2:
            # Rectangle: dimensions are [width, height]
            # Use identity orientation (aligned with map axes)
            qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
        else:
            # Circle or other: no special orientation
            qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
        
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        
        self.tf_broadcaster.sendTransform(t)
    
    def update_path(self, header, x, y, z):
        """Update the trajectory path with new position"""
        # Check if we should add this point (distance filtering)
        should_add = True
        if len(self.path.poses) > 0:
            last_pose = self.path.poses[-1]
            dx = x - last_pose.pose.position.x
            dy = y - last_pose.pose.position.y
            dz = z - last_pose.pose.position.z
            distance = math.sqrt(dx*dx + dy*dy + dz*dz)
            
            if distance < self.min_distance_threshold:
                should_add = False
        
        if should_add:
            # Create new pose
            pose_stamped = PoseStamped()
            pose_stamped.header = header
            pose_stamped.header.frame_id = "map"
            pose_stamped.pose.position.x = x
            pose_stamped.pose.position.y = y
            pose_stamped.pose.position.z = z
            pose_stamped.pose.orientation.w = 1.0  # Identity orientation
            
            # Add to path
            self.path.poses.append(pose_stamped)
            
            # Limit path length
            if len(self.path.poses) > self.max_path_length:
                self.path.poses = self.path.poses[-self.max_path_length:]
        
        # Publish path
        self.path.header.stamp = self.get_clock().now().to_msg()
        self.path_pub.publish(self.path)
    
    def euler_to_quaternion(self, roll, pitch, yaw):
        """Convert Euler angles (roll, pitch, yaw) to quaternion"""
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
    
    def yaw_to_quaternion(self, yaw):
        """Convert yaw angle to quaternion (rotation around z-axis)"""
        half_yaw = yaw * 0.5
        qz = math.sin(half_yaw)
        qw = math.cos(half_yaw)
        return 0.0, 0.0, qz, qw


def main(args=None):
    rclpy.init(args=args)
    
    try:
        # Parse command line arguments for agent_id
        import argparse
        parser = argparse.ArgumentParser(description='TF Visualizer for Aircraft')
        parser.add_argument('--agent-id', type=int, default=1, 
                          help='Agent ID to subscribe to (default: 1)')
        
        # Parse known args to handle ROS2 remapping args
        parsed_args, _ = parser.parse_known_args()
        
        node = TFVisualizerAirplane(agent_id=parsed_args.agent_id)
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
