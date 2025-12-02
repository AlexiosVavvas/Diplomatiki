#!/usr/bin/env python3
"""
Teleoperation node for manually controlling a fixed-wing aircraft using Arduino joystick.

This node allows direct control of the aircraft control surfaces (elevator, ailerons, rudder)
and throttle via Arduino joystick input, bypassing the ergodic controller.

Joystick mapping (Arduino joystick channels):
  - Throttle stick:  Throttle control (channel 1)
  - Aileron stick:   Ailerons/roll control (channel 3)
  - Elevator stick:  Elevator/pitch control (channel 5)
  - Rudder stick:    Rudder/yaw control (channel 7)
  - Switch:          Reset to trim state (channel 9)

Author: Created for teleoperation testing
"""

import os
import numpy as np
import threading
import signal
import sys
import time

import rclpy
from rclpy.node import Node
from my_interfaces.msg import JoystickData
import argparse


# Global shutdown flag
shutdown_flag = threading.Event()


def signalHandler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\nReceived interrupt signal. Shutting down gracefully...")
    shutdown_flag.set()


# Set up signal handler
signal.signal(signal.SIGINT, signalHandler)


class TeleopAgent(Node):
    """ROS2 Node for teleoperation of fixed-wing aircraft"""
    
    def __init__(self, agent_id, dynamic_model, init_pos_3d, l1_bounds, l2_bounds, 
                 publish_freq=50):
        """
        Initialize teleoperation agent
        
        Args:
            agent_id: Unique identifier for this agent
            dynamic_model: Instance of FixedWing12DOFTrainer or similar
            init_pos_2d: Initial [x, y] position
            l1_bounds: [min, max] bounds for x dimension
            l2_bounds: [min, max] bounds for y dimension
            publish_freq: Hz frequency for publishing state data
        """
        super().__init__(f'teleop_agent_{agent_id}')
        
        self.agent_id = agent_id
        self.model = dynamic_model
        self.L1_BOUNDS = l1_bounds
        self.L2_BOUNDS = l2_bounds
        self.publish_freq = publish_freq
        
        # Control input state (will be updated by joystick)
        self.u_manual = self.model.u_trim.copy()  # Start at trim
        
        # Last switch state for edge detection
        self.last_switch_state = 0
        
        # ROS2 Subscribers - subscribe to joystick data
        self.joy_sub = self.create_subscription(
            JoystickData,
            'joystick',
            self.joystickCallback,
            10
        )
        
        # ROS2 Publishers (reusing AgentData message for visualization)
        from my_interfaces.msg import AgentData, MultipleObstacles, MultipleTargetEstimates, CkTable
        self.data_pub = self.create_publisher(
            AgentData,
            f'/agent_{self.agent_id}/data',
            10
        )
        
        # Add publishers for other topics expected by dashboard
        self.obstacles_pub = self.create_publisher(
            MultipleObstacles,
            f'agent_{self.agent_id}/known_obstacles',
            10
        )
        
        self.target_estimates_pub = self.create_publisher(
            MultipleTargetEstimates,
            f'agent_{self.agent_id}/target_estimates',
            10
        )
        
        self.ck_pub = self.create_publisher(
            CkTable,
            f'agent_{self.agent_id}/ck',
            10
        )
        
        # Timer for publishing data
        self.publish_timer = self.create_timer(
            1.0 / self.publish_freq,
            self.publishDataCallback
        )
        
        # System parameters as ROS parameters
        from rcl_interfaces.msg import ParameterDescriptor
        descriptor = ParameterDescriptor(read_only=True, description='Read only')
        self.declare_parameter('id', agent_id, descriptor=descriptor)
        self.declare_parameter('model_type', self.model.type, descriptor=descriptor)
        self.declare_parameter('mode', 'teleoperation', descriptor=descriptor)
        
        self.get_logger().info(f"Teleoperation agent {agent_id} initialized")
        self.get_logger().info(f"Model type: {self.model.type}")
        self.get_logger().info("Waiting for joystick input on 'joystick' topic...")
        self.get_logger().info("Joystick mapping:")
        self.get_logger().info("  - Throttle stick (ch 1): Throttle control")
        self.get_logger().info("  - Aileron stick (ch 3): Roll control")
        self.get_logger().info("  - Elevator stick (ch 5): Pitch control")
        self.get_logger().info("  - Rudder stick (ch 7): Yaw control")
        self.get_logger().info("  - Switch (ch 9): Reset to trim (toggle)")
        self.get_logger().info(f"Model input limits:")
        self.get_logger().info(f"  Elevator: [{self.model.input_limits[0, 0]:.3f}, {self.model.input_limits[0, 1]:.3f}]")
        self.get_logger().info(f"  Aileron:  [{self.model.input_limits[1, 0]:.3f}, {self.model.input_limits[1, 1]:.3f}]")
        self.get_logger().info(f"  Rudder:   [{self.model.input_limits[2, 0]:.3f}, {self.model.input_limits[2, 1]:.3f}]")
        self.get_logger().info(f"  Throttle: [{self.model.input_limits[3, 0]:.3f}, {self.model.input_limits[3, 1]:.3f}]")
        
        # Simulation state
        self.time_since_start = 0.0
        self.last_update_time = time.time()
    
    def mapToRange(self, normalized_value, min_val, max_val):
        """
        Map normalized value from [-1, 1] to [min_val, max_val]
        
        Args:
            normalized_value: Value in range [-1, 1]
            min_val: Minimum output value
            max_val: Maximum output value
            
        Returns:
            Mapped value in range [min_val, max_val]
        """
        # Clamp input to [-1, 1]
        normalized_value = np.clip(normalized_value, -1.0, 1.0)
        
        # Map from [-1, 1] to [min_val, max_val]
        return min_val + (normalized_value + 1.0) / 2.0 * (max_val - min_val)
    
    def joystickCallback(self, msg):
        """
        Process joystick input and update control commands
        
        Args:
            msg: JoystickData message with normalized values
        """
        # Map normalized joystick values [-1, 1] to model input limits
        # u = [delta_e, delta_a, delta_r, throttle]
        
        elevator_cmd = self.mapToRange(
            msg.elevator,
            self.model.input_limits[0, 0],
            self.model.input_limits[0, 1]
        )
        
        aileron_cmd = self.mapToRange(
            msg.aileron,
            self.model.input_limits[1, 0],
            self.model.input_limits[1, 1]
        )
        
        rudder_cmd = self.mapToRange(
            msg.rudder,
            self.model.input_limits[2, 0],
            self.model.input_limits[2, 1]
        )
        
        throttle_cmd = self.mapToRange(
            msg.throttle,
            self.model.input_limits[3, 0],
            self.model.input_limits[3, 1]
        )
        
        # Build control input vector
        self.u_manual = np.array([
            elevator_cmd,   # Elevator
            aileron_cmd,    # Ailerons
            rudder_cmd,     # Rudder
            throttle_cmd    # Throttle
        ])
        
        # Handle reset switch (trigger on rising edge: 0 -> 1)
        if msg.switch_state == 1 and self.last_switch_state == 0:
            self.resetToTrim()
        
        self.last_switch_state = msg.switch_state
    
    def resetToTrim(self):
        """Reset aircraft to trim state"""
        # Reset to initial state with trim velocity
        x0 = self.model.state.copy()
        x0[0:3] = [0, 0, -200]
        x0[3:6] = [0, 0.07, 0.053]  # Reset angles to trim
        x0[6:9] = [self.model.params['V_trim'], 0, 0]  # Reset velocities to trim
        x0[9:12] = [0, 0, 0]  # Reset angular rates
        
        self.model.state = x0
        self.u_manual = self.model.u_trim.copy()
        
        self.get_logger().info("Aircraft reset to trim state")
    
    def publishDataCallback(self):
        """Publish agent state data for visualization"""
        from my_interfaces.msg import AgentData, MultipleObstacles, MultipleTargetEstimates, CkTable
        from std_msgs.msg import Header
        
        msg = AgentData()
        
        # Header
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = f'agent_{self.agent_id}'
        
        # Simulation time
        msg.simulation_time = self.time_since_start
        
        # State and input dimensions
        msg.num_of_states = len(self.model.state)
        msg.num_of_inputs = len(self.u_manual)
        
        # State vector
        msg.states = self.model.state.tolist()
        
        # Control input
        msg.inputs = self.u_manual.tolist()
        
        # Additional info
        msg.ergodic_cost = 0.0  # Not applicable in teleop mode
        msg.active_cbf_flag = False
        msg.delta_t_ts = -1.0
        
        # No agents in range for teleop mode
        msg.in_range_agents_ids = []
        
        self.data_pub.publish(msg)
        
        # Publish empty messages for other topics to keep dashboard.py happy
        obstacles_msg = MultipleObstacles()
        obstacles_msg.obstacles = []
        self.obstacles_pub.publish(obstacles_msg)
        
        target_estimates_msg = MultipleTargetEstimates()
        target_estimates_msg.target_estimates = []
        target_estimates_msg.ground_truths = []
        self.target_estimates_pub.publish(target_estimates_msg)
        
        # Publish empty CK table
        ck_msg = CkTable()
        ck_msg.table_size = 0
        ck_msg.ck_values = []
        ck_msg.total_erg_cost_in_range = 0.0
        ck_msg.l_bounds = [self.L1_BOUNDS[0], self.L1_BOUNDS[1], 
                          self.L2_BOUNDS[0], self.L2_BOUNDS[1]]
        self.ck_pub.publish(ck_msg)
    
    def step(self):
        """Step the simulation forward one timestep"""
        # Apply manual control and step the model
        self.model.state = self.model.step(self.model.state, self.u_manual)
        
        # Update time
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.time_since_start += self.model.dt
        self.last_update_time = current_time
        
        return dt


def main(args=None):
    """Main entry point for teleoperation node"""
    from my_erg_lib.model_dynamics import FixedWing12DOFTrainer, FixedWing12DOFTrainerJAX
    from my_erg_lib.Utilities import loadAgentConfigFromYaml
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Teleoperation node for fixed-wing aircraft')
    parser.add_argument('--agent_id', type=int, required=True,
                       help='Agent ID to name the node')
    parser.add_argument('--init_pos', type=float, nargs=3, required=False,
                       default=[9, 3, 200], help='Initial position as [x, y, z]')
    parser.add_argument('--l_bounds', type=float, nargs=4, required=False,
                       default=[0, 10, 0, 10],
                       help='Bounds as [x_min, x_max, y_min, y_max]')
    parser.add_argument('--agent_config', type=str, required=False,
                       default='src/ergodic_exploration/agent_configs/fixed_wing_12dof_trainer.yaml',
                       help='Path to agent configuration YAML file')
    parser.add_argument('--model_type', type=str, required=False,
                       default='FixedWing12DOFTrainer',
                       help='Model type (FixedWing12DOFTrainer or FixedWing12DOFTrainerJAX)')
    parser.add_argument('--publish_freq', type=int, required=False,
                       default=50, help='Publishing frequency in Hz')
    
    parsed_args, ros_args = parser.parse_known_args()
    
    # Extract arguments
    AGENT_ID = parsed_args.agent_id
    INIT_POS_3D = np.array(parsed_args.init_pos)
    L1_BOUNDS = [parsed_args.l_bounds[0], parsed_args.l_bounds[1]]
    L2_BOUNDS = [parsed_args.l_bounds[2], parsed_args.l_bounds[3]]
    MODEL_TYPE = parsed_args.model_type
    PUBLISH_FREQ = parsed_args.publish_freq
    
    # Load agent configuration
    try:
        agent_config = loadAgentConfigFromYaml(parsed_args.agent_config, L1_BOUNDS, L2_BOUNDS)
        if agent_config is None:
            print(f"ERROR: Failed to load configuration from {parsed_args.agent_config}")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load agent configuration: {e}")
        sys.exit(1)
    
    # Extract dynamics configuration
    dynamics_config = agent_config['dynamics']
    dt = dynamics_config.get('dt', 0.01)
    v_trim = dynamics_config.get('v_trim', 15.0)
    use_linear_f = dynamics_config.get('use_linear_f', False)
    use_linear_fx_fu = dynamics_config.get('use_linear_fx_fu', False)
    
    # Initialize dynamic model
    print(f"Initializing {MODEL_TYPE} model for teleoperation...")
    print(f"  dt = {dt}, v_trim = {v_trim} m/s")
    print(f"  Initial position: {INIT_POS_3D}")
    
    # State: [X, Y, Z, phi, theta, psi, u, v, w, p, q, r]
    x0 = [INIT_POS_3D[0], INIT_POS_3D[1], INIT_POS_3D[2], 0, 0.07, 0.053, v_trim, 0, 0, 0, 0, 0]
    
    if MODEL_TYPE == "FixedWing12DOFTrainer":
        dynamic_model = FixedWing12DOFTrainer(
            dt=dt, x0=x0, v_trim=v_trim,
            use_linear_f=use_linear_f,
            use_linear_fx_fu=use_linear_fx_fu
        )
    elif MODEL_TYPE == "FixedWing12DOFTrainerJAX":
        dynamic_model = FixedWing12DOFTrainerJAX(
            dt=dt, x0=x0, v_trim=v_trim,
            use_linear_f=use_linear_f,
            use_linear_fx_fu=use_linear_fx_fu
        )
    else:
        print(f"ERROR: Unsupported model type: {MODEL_TYPE}")
        print("Teleoperation mode only supports FixedWing12DOFTrainer models")
        sys.exit(1)
    
    print(f"Model initialized. Trim inputs: {dynamic_model.u_trim}")
    print(f"Input limits: \n{dynamic_model.input_limits}")
    
    # Initialize ROS2
    rclpy.init(args=ros_args)
    
    # Create teleoperation agent node
    teleop_agent = TeleopAgent(
        agent_id=AGENT_ID,
        dynamic_model=dynamic_model,
        init_pos_3d=INIT_POS_3D,
        l1_bounds=L1_BOUNDS,
        l2_bounds=L2_BOUNDS,
        publish_freq=PUBLISH_FREQ
    )
    
    print("\n" + "="*60)
    print("TELEOPERATION MODE ACTIVE")
    print("="*60)
    print("Make sure the joystick_node is running:")
    print("  $ ros2 run ergodic_exploration joystick_node")
    print("\nTo view the aircraft in RViz, run:")
    print("  $ rviz2 -d <your_rviz_config.rviz>")
    print("\nJoystick Controls:")
    print("  - Throttle stick (ch 1): Throttle control")
    print("  - Aileron stick (ch 3):  Roll control")
    print("  - Elevator stick (ch 5): Pitch control")
    print("  - Rudder stick (ch 7):   Yaw control")
    print("  - Switch (ch 9):         Reset to trim")
    print("="*60 + "\n")
    
    # Simulation loop
    def simulationFunction():
        """Main simulation loop"""
        target_dt = dynamic_model.dt
        iteration = 0
        
        while not shutdown_flag.is_set():
            loop_start = time.time()
            
            # Step the simulation
            actual_dt = teleop_agent.step()
            
            # Log info periodically
            # if iteration % 500 == 0:
            #     state = dynamic_model.state
            #     u = teleop_agent.u_manual
            #     teleop_agent.get_logger().info(
            #         f"t={teleop_agent.time_since_start:.2f}s | "
            #         f"Pos: ({state[0]:.1f}, {state[1]:.1f}, {state[2]:.1f}) | "
            #         f"V: {np.linalg.norm(state[6:9]):.1f} m/s | "
            #         f"φ={np.rad2deg(state[3]):.1f}° θ={np.rad2deg(state[4]):.1f}° "
            #         f"ψ={np.rad2deg(state[5]):.1f}° | "
            #         f"Throttle: {u[3]:.2f}"
            #     )
            # iteration += 1
            
            
            # Sleep to maintain simulation rate
            loop_time = time.time() - loop_start
            sleep_time = max(0, target_dt - loop_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    # Start simulation in background thread
    sim_thread = threading.Thread(target=simulationFunction, daemon=True)
    sim_thread.start()
    
    # Spin ROS node
    try:
        while rclpy.ok() and not shutdown_flag.is_set():
            try:
                rclpy.spin_once(teleop_agent, timeout_sec=0.1)
            except rclpy.executors.ExternalShutdownException:
                teleop_agent.get_logger().warn("ROS external shutdown detected.")
                break
    except KeyboardInterrupt:
        teleop_agent.get_logger().warn("\nKeyboard interrupt received.")
    finally:
        # Signal shutdown
        shutdown_flag.set()
        
        # Wait for simulation thread
        if sim_thread.is_alive():
            teleop_agent.get_logger().info("Waiting for simulation thread to finish...")
            sim_thread.join(timeout=3.0)
        
        # Cleanup
        try:
            if rclpy.ok():
                teleop_agent.destroy_node()
                rclpy.shutdown()
        except Exception as e:
            print(f"Error during cleanup: {e}")
        
        print("Teleoperation node shutdown complete.")


if __name__ == "__main__":
    main()
