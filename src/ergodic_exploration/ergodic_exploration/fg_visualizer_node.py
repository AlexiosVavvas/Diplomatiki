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

        # Reference parameters
        self.declare_parameter('ref_lat', 63.9850)
        self.declare_parameter('ref_lon', -22.6548)
        self.declare_parameter('ref_alt', 0.0)

        self.ref_lat = self.get_parameter('ref_lat').value
        self.ref_lon = self.get_parameter('ref_lon').value
        self.ref_alt = self.get_parameter('ref_alt').value

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
                # In NED frame: X=North, Y=East, Z=Down
                X, Y, Z, phi, theta, psi, u, v, w, p, q, r = states_copy[:12]
                
                # Convert position to lat/lon/alt
                # X is North (affects latitude), Y is East (affects longitude)
                lat_rad = np.deg2rad(self.ref_lat) + (X / 6378137.0)  # X = North -> latitude
                lon_rad = np.deg2rad(self.ref_lon) + (Y / (6378137.0 * np.cos(lat_rad)))  # Y = East -> longitude

                fdm_data.lat_rad = lat_rad
                fdm_data.lon_rad = lon_rad
                fdm_data.alt_m = self.ref_alt - Z   # Altitude is negative of down
                fdm_data.agl_m = self.ref_alt - Z   # Above ground level approximation
                
                # Attitude (Euler angles)
                fdm_data.phi_rad = phi      # Roll 
                fdm_data.theta_rad = theta  # Pitch
                fdm_data.psi_rad = psi      # Yaw
                
                # Angular rates (body frame)
                fdm_data.phidot_rad_per_s = p
                fdm_data.thetadot_rad_per_s = q
                fdm_data.psidot_rad_per_s = r
                
                # ===== BODY VELOCITIES (PRIMARY) =====
                # These are the ACTUAL velocities FlightGear uses for visualization
                fdm_data.v_body_u = u * 3.28084  # ft/s forward
                fdm_data.v_body_v = v * 3.28084  # ft/s right
                fdm_data.v_body_w = w * 3.28084  # ft/s down
                
                # ===== NED VELOCITIES (for display/logging only) =====
                # FlightGear uses these for GPS/navigation displays, NOT for visual position
                # Calculate TRUE inertial velocities by differentiating position
                # (Not from body velocities! Those can have sideslip/wind)
                
                # Option 1: Use position derivatives (if available from your dynamics)
                # If you store Xdot, Ydot, Zdot from your integration:
                # v_north = Xdot  # m/s north (inertial)
                # v_east = Ydot   # m/s east (inertial)
                # v_down = Zdot   # m/s down (inertial)
                
                # Option 2: Compute from body velocities (less accurate with sideslip)
                # Rotation matrix from body to NED
                cp, sp = np.cos(phi), np.sin(phi)
                ct, st = np.cos(theta), np.sin(theta)
                cy, sy = np.cos(psi), np.sin(psi)
                
                # Body to NED rotation (for reference only - not used for position!)
                v_north_body_derived = (ct * cy * u + (sp * st * cy - cp * sy) * v + 
                        (cp * st * cy + sp * sy) * w)
                v_east_body_derived = (ct * sy * u + (sp * st * sy + cp * cy) * v + 
                        (cp * st * sy - sp * cy) * w)
                v_down_body_derived = (-st * u + sp * ct * v + cp * ct * w)
                
                # Convert m/s to ft/s for FlightGear
                # NOTE: These are for DISPLAY ONLY (speedometer, GPS, etc.)
                # FlightGear may integrate v_body_* for visual position!
                fdm_data.v_north_ft_per_s = v_north_body_derived * 3.28084
                fdm_data.v_east_ft_per_s = v_east_body_derived * 3.28084
                fdm_data.v_down_ft_per_s = v_down_body_derived * 3.28084
                

                # Control surfaces (if we have inputs)
                if inputs_copy is not None and len(inputs_copy) >= 4:
                    # Inputs: [delta_e, delta_a, delta_r, throttle]
                    delta_e, delta_a, delta_r, throttle = inputs_copy[:4]
                    
                    # Normalize to FlightGear's expected range [-1, 1] or [0, 1]
                    e_norm = np.clip(delta_e / (10 * np.pi / 180), -1.0, 1.0)  # Max deflection 25 degrees
                    a_norm = np.clip(delta_a / (10 * np.pi / 180), -1.0, 1.0)
                    r_norm = np.clip(delta_r / (10 * np.pi / 180), -1.0, 1.0)  # Rudder may have larger deflection
                    
                    # Set control surfaces
                    fdm_data.elevator = float(e_norm) 
                    fdm_data.left_aileron = float(a_norm)
                    fdm_data.right_aileron = float(a_norm)
                    fdm_data.rudder = float(r_norm)
        
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
