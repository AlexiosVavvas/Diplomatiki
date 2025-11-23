#!/usr/bin/env python3
"""
Aircraft Data Converter Node
Converts AgentData to AircraftData for fixed-wing aircraft state visualization.
"""

import rclpy
from rclpy.node import Node
import numpy as np

from my_interfaces.msg import AgentData, AircraftData


class AircraftDataConverter(Node):
    """Converts AgentData state vector to detailed AircraftData message."""
    
    def __init__(self):
        super().__init__('aircraft_data_converter')
        
        # Declare parameters
        self.declare_parameter('agent_id', 1)
        self.declare_parameter('subscribe_topic', '/agent_1/data')
        self.declare_parameter('publish_topic', '/agent_1/aircraft_data')
        
        # Get parameters
        agent_id = self.get_parameter('agent_id').value
        sub_topic = self.get_parameter('subscribe_topic').value
        pub_topic = self.get_parameter('publish_topic').value
        
        # Create subscriber and publisher
        self.subscription = self.create_subscription(
            AgentData,
            sub_topic,
            self.agent_data_callback,
            10
        )
        
        self.publisher = self.create_publisher(
            AircraftData,
            pub_topic,
            10
        )
        
        self.get_logger().info(
            f'Aircraft Data Converter started for agent {agent_id}'
        )
        self.get_logger().info(f'  Subscribing to: {sub_topic}')
        self.get_logger().info(f'  Publishing to: {pub_topic}')
    
    def agent_data_callback(self, msg: AgentData):
        """Convert AgentData to AircraftData."""
        # Check if we have enough states (12 for fixed-wing)
        if len(msg.states) < 12:
            self.get_logger().warn(
                f'Insufficient states: {len(msg.states)} < 12',
                throttle_duration_sec=5.0
            )
            return
        
        # Extract state vector [X, Y, Z, phi, theta, psi, u, v, w, p, q, r]
        x = np.array(msg.states[:12])
        
        # Create AircraftData message
        aircraft_msg = AircraftData()
        aircraft_msg.header = msg.header
        
        # Position (NED)
        aircraft_msg.north = x[0]
        aircraft_msg.east = x[1]
        aircraft_msg.down = x[2]
        aircraft_msg.altitude = -x[2]  # altitude is negative of down
        
        # Attitude
        phi, theta, psi = x[3], x[4], x[5]
        aircraft_msg.roll = phi
        aircraft_msg.pitch = theta
        aircraft_msg.yaw = psi
        aircraft_msg.roll_deg = float(np.rad2deg(phi))
        aircraft_msg.pitch_deg = float(np.rad2deg(theta))
        aircraft_msg.yaw_deg = float(np.rad2deg(psi))
        
        # Body velocities
        u, v, w = x[6], x[7], x[8]
        aircraft_msg.u_forward = u
        aircraft_msg.v_sideways = v
        aircraft_msg.w_downward = w
        aircraft_msg.airspeed = float(np.sqrt(u**2 + v**2 + w**2))
        
        # Angular rates
        p, q, r = x[9], x[10], x[11]
        aircraft_msg.p_roll_rate = p
        aircraft_msg.q_pitch_rate = q
        aircraft_msg.r_yaw_rate = r
        aircraft_msg.p_deg_s = float(np.rad2deg(p))
        aircraft_msg.q_deg_s = float(np.rad2deg(q))
        aircraft_msg.r_deg_s = float(np.rad2deg(r))
        
        # NED velocities (body to NED transformation)
        cpsi = np.cos(psi)
        spsi = np.sin(psi)
        cth = np.cos(theta)
        sth = np.sin(theta)
        cphi = np.cos(phi)
        sphi = np.sin(phi)
        
        R = np.array([
            [cpsi * cth, cpsi * sth * sphi - spsi * cphi, cpsi * sth * cphi + spsi * sphi],
            [spsi * cth, spsi * sth * sphi + cpsi * cphi, spsi * sth * cphi - cpsi * sphi],
            [-sth,       cth * sphi,                      cth * cphi]
        ])
        
        vel_body = np.array([u, v, w])
        vel_ned = R.dot(vel_body)
        
        aircraft_msg.v_north = float(vel_ned[0])
        aircraft_msg.v_east = float(vel_ned[1])
        aircraft_msg.v_down = float(vel_ned[2])
        aircraft_msg.climb_rate = float(-vel_ned[2])  # positive up
        aircraft_msg.ground_speed = float(np.sqrt(vel_ned[0]**2 + vel_ned[1]**2))
        
        # Aerodynamic angles
        V_safe = max(aircraft_msg.airspeed, 1e-3)
        alpha = np.arctan2(w, u)
        beta = np.arcsin(np.clip(v / V_safe, -0.99, 0.99))
        
        aircraft_msg.alpha = float(alpha)
        aircraft_msg.beta = float(beta)
        aircraft_msg.alpha_deg = float(np.rad2deg(alpha))
        aircraft_msg.beta_deg = float(np.rad2deg(beta))
        
        # Publish
        self.publisher.publish(aircraft_msg)


def main(args=None):
    rclpy.init(args=args)
    node = AircraftDataConverter()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
