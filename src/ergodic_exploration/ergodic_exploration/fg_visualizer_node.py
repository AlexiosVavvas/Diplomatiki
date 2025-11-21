#!/usr/bin/env python3
"""
FlightGear Visualizer Node for Fixed-Wing Drone

Subscribes to /agent_1/data (AgentData message) and visualizes the airplane state in FlightGear.
State ordering: [X, Y, Z, phi, theta, psi, u, v, w, p, q, r]
Inputs: [delta_e, delta_a, delta_r, throttle]
"""

import rclpy
from rclpy.node import Node
from my_interfaces.msg import AgentData
from flightgear_python.fg_if import FDMConnection
import numpy as np
import time
import threading


class FGVisualizerNode(Node):
    def __init__(self):
        super().__init__('fg_visualizer_node')
        
        # Declare parameters
        self.declare_parameter('fg_host', 'localhost')
        self.declare_parameter('fg_rx_port', 5501)  # Receive from FG
        self.declare_parameter('fg_tx_port', 5502)  # Send to FG
        self.declare_parameter('update_rate_hz', 30.0)  # FlightGear update rate
        
        self.fg_host = self.get_parameter('fg_host').value
        self.fg_rx_port = self.get_parameter('fg_rx_port').value
        self.fg_tx_port = self.get_parameter('fg_tx_port').value
        self.update_rate = self.get_parameter('update_rate_hz').value
        
        # State storage (thread-safe)
        self.state_lock = threading.Lock()
        self.latest_states = None
        self.latest_inputs = None
        
        # Subscribe to agent data
        self.subscription = self.create_subscription(
            AgentData,
            '/agent_1/data',
            self.agent_data_callback,
            10
        )
        
        # FlightGear connection
        self.fdm_conn = None
        self.fdm_event_pipe = None
        self.fg_connected = False
        
        # Start trying to connect immediately
        self.setup_flightgear()
        
        # Timer for sending to FlightGear at controlled rate
        self.timer = self.create_timer(1.0 / self.update_rate, self.update_flightgear)
        
        self.get_logger().info('FlightGear Visualizer Node initialized')
        self.get_logger().info(f'Listening to /agent_1/data and updating FlightGear at {self.update_rate} Hz')
    
    def setup_flightgear(self):
        """Setup FlightGear connection with infinite retry logic"""
        retry_delay = 3.0  # seconds
        attempt = 0
        
        while not self.fg_connected:
            attempt += 1
            try:
                self.get_logger().info(f'Attempting to connect to FlightGear (attempt {attempt})...')
                
                # Increase timeout to 30 seconds to wait for FlightGear to boot
                self.fdm_conn = FDMConnection(rx_timeout_s=30.0)
                
                # Receive from FG (its FDM out)
                self.fdm_event_pipe = self.fdm_conn.connect_rx(
                    self.fg_host, 
                    self.fg_rx_port, 
                    self.fdm_callback
                )
                
                # Send to FG (its FDM in)
                self.fdm_conn.connect_tx(self.fg_host, self.fg_tx_port)
                
                # Start RX/TX loop (this will block until first packet or timeout)
                self.fdm_conn.start()
                
                self.fg_connected = True
                self.get_logger().info(f'✓ Successfully connected to FlightGear at {self.fg_host}:{self.fg_tx_port}')
                self.get_logger().info(f'✓ Receiving from FlightGear on port {self.fg_rx_port}')
                return  # Success!
                
            except Exception as e:
                self.get_logger().warn(f'Connection attempt {attempt} failed: {e}')
                
                # Clean up failed connection
                if self.fdm_conn is not None:
                    try:
                        self.fdm_conn.stop()
                    except:
                        pass
                    self.fdm_conn = None
                    self.fdm_event_pipe = None
                
                self.get_logger().info(f'Retrying in {retry_delay} seconds...')
                time.sleep(retry_delay)
    
    def fdm_callback(self, fdm_data, event_pipe):
        """
        Callback for receiving FDM data from FlightGear.
        We'll modify the FDM data with our simulation state.
        
        NOTE: This runs in a SEPARATE PROCESS (multiprocessing), not just a thread!
        We need to receive data via event_pipe, not shared memory.
        """
        # Check if there's data from the parent process via the event pipe
        if event_pipe.child_poll():
            # Receive state data from main process
            states_copy, inputs_copy = event_pipe.child_recv()
            
            if states_copy is not None and len(states_copy) >= 12:
                # State ordering: [X, Y, Z, phi, theta, psi, u, v, w, p, q, r]
                X, Y, Z, phi, theta, psi, u, v, w, p, q, r = states_copy[:12]
                
                # Convert position to lat/lon/alt
                # Assuming X=North (m), Y=East (m), Z=altitude (m)
                # Simple flat-earth approximation for visualization
                # Starting from a reference point (can be made configurable)
                ref_lat = 37.6213  # degrees (default FG location)
                ref_lon = -122.3790  # degrees
                
                # Convert X (north) and Y (east) in meters to lat/lon degrees
                lat_rad = np.deg2rad(ref_lat) + (X / 6371000.0)  # Earth radius ~6371 km
                lon_rad = np.deg2rad(ref_lon) + (Y / (6371000.0 * np.cos(lat_rad)))
                
                fdm_data.lat_rad = lat_rad
                fdm_data.lon_rad = lon_rad
                fdm_data.alt_m = Z  # Z is altitude above ground
                fdm_data.agl_m = Z  # Above ground level
                
                # Attitude (Euler angles)
                fdm_data.phi_rad = -phi      # Roll (inverted)
                fdm_data.theta_rad = theta  # Pitch
                fdm_data.psi_rad = psi      # Yaw
                
                # Angular rates
                fdm_data.phidot_rad_per_s = p
                fdm_data.thetadot_rad_per_s = q
                fdm_data.psidot_rad_per_s = r
                
                # Body velocities
                fdm_data.v_body_u = u  # m/s
                fdm_data.v_body_v = v
                fdm_data.v_body_w = w
                
                # Convert body velocities to NED frame for display
                # Rotation matrix from body to NED
                cp, sp = np.cos(phi), np.sin(phi)
                ct, st = np.cos(theta), np.sin(theta)
                cy, sy = np.cos(psi), np.sin(psi)
                
                # Body to NED rotation
                v_north = (ct * cy * u + (sp * st * cy - cp * sy) * v + 
                          (cp * st * cy + sp * sy) * w)
                v_east = (ct * sy * u + (sp * st * sy + cp * cy) * v + 
                         (cp * st * sy - sp * cy) * w)
                v_down = (-st * u + sp * ct * v + cp * ct * w)
                
                # Convert m/s to ft/s for FlightGear
                fdm_data.v_north_ft_per_s = v_north * 3.28084
                fdm_data.v_east_ft_per_s = v_east * 3.28084
                fdm_data.v_down_ft_per_s = v_down * 3.28084
                

                # Control surfaces (if we have inputs)
                if inputs_copy is not None and len(inputs_copy) >= 4:
                    # Inputs: [delta_e, delta_a, delta_r, throttle]
                    delta_e, delta_a, delta_r, throttle = inputs_copy[:4]
                    
                    # Normalize to FlightGear's expected range [-1, 1] or [0, 1]
                    fdm_data.elevator = float(delta_e)
                    fdm_data.left_aileron = float(-delta_a)  # Aileron sign convention
                    fdm_data.right_aileron = float(delta_a)
                    fdm_data.rudder = float(delta_r)
                    
                    # Engine (throttle is typically [0, 1])
                    # You may need to adjust based on your throttle range
        
        return fdm_data
    
    def agent_data_callback(self, msg: AgentData):
        """Callback for receiving agent data"""
        with self.state_lock:
            if msg.num_of_states >= 12:
                self.latest_states = np.array(msg.states)
            if msg.num_of_inputs >= 4:
                self.latest_inputs = np.array(msg.inputs)
    
    def update_flightgear(self):
        """
        Timer callback to send latest state data to FlightGear process.
        Uses event_pipe to communicate across process boundaries.
        """
        if self.fdm_event_pipe is not None:
            try:
                with self.state_lock:
                    states_copy = self.latest_states
                    inputs_copy = self.latest_inputs
                
                # Send state data to the FlightGear callback process
                self.fdm_event_pipe.parent_send((states_copy, inputs_copy))
            except Exception as e:
                self.get_logger().error(f'Error sending to event pipe: {e}', throttle_duration_sec=5.0)
    
    def destroy_node(self):
        """Clean up FlightGear connection"""
        if self.fdm_conn is not None:
            try:
                self.fdm_conn.stop()
            except Exception as e:
                self.get_logger().error(f'Error stopping FlightGear connection: {e}')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = FGVisualizerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
