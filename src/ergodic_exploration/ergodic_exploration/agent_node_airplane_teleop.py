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
from my_interfaces.msg import JoystickData, ObsAvoidanceDebug
import argparse

# CBF obstacle avoidance imports
from my_erg_lib.obstacles import Obstacle, saveObstaclesToMemory
from my_erg_lib.cbf_qp_solver import solve_cbf_qp
from my_erg_lib.Utilities import loadObstaclesFromYaml


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
                 publish_freq=50, cbf_params=None):
        """
        Initialize teleoperation agent
        
        Args:
            agent_id: Unique identifier for this agent
            dynamic_model: Instance of FixedWing12DOFTrainer or similar
            init_pos_2d: Initial [x, y] position
            l1_bounds: [min, max] bounds for x dimension
            l2_bounds: [min, max] bounds for y dimension
            publish_freq: Hz frequency for publishing state data
            cbf_params: Dictionary of CBF safety filter parameters
        """
        super().__init__(f'teleop_agent_{agent_id}')
        
        self.agent_id = agent_id
        self.model = dynamic_model
        self.L1_BOUNDS = l1_bounds
        self.L2_BOUNDS = l2_bounds
        self.L1_min, self.L1_max = l1_bounds
        self.L2_min, self.L2_max = l2_bounds
        self.publish_freq = publish_freq
        
        # Control input state (will be updated by joystick)
        self.u_manual = self.model.u_trim.copy()  # Start at trim
        
        # Last switch state for edge detection
        self.last_switch_state = 0
        
        # Obstacle avoidance infrastructure
        self.obstacle_list = []
        
        # CBF safety filter parameters (with defaults)
        default_cbf_params = {
            'alpha_1': 0.1, 'alpha_2': 3.0, 'alpha_3': 15.0,
            'alpha_u': 50.0, 'cbf_Kp': 5.0, 'cbf_dt': 0.025,
            'alpha_max_deg': 8.0, 'alpha_aoa_1': 10.0, 'alpha_aoa_2': 15.0,
            'slack_penalty_aoa': 300.0, 'use_aoa_constraint': True,
            'delta_safe': 0.0, 'cbf_skip_iter': 5, 'relax_factor': 0.5
        }
        self.cbf_params = default_cbf_params
        if cbf_params is not None:
            self.cbf_params.update(cbf_params)
        
        # Previous control input for CBF rate constraints
        self.u_before = self.model.u_trim.copy()
        self.u_safe = np.zeros(self.model.num_of_inputs)
        self.active_cbf_flag = False
        
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
        
        # Publisher for CBF debug data
        self.obs_avoidance_debug_publisher = self.create_publisher(
            ObsAvoidanceDebug,
            f'agent_{self.agent_id}/obs_avoidance_debug',
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
    
    # ======= Obstacle Avoidance / CBF Methods =======
    def calcPotentialU(self, x):
        """
        Calculate the potential function U at a given state x.
        This function sums the potential contributions from all obstacles in the obstacle list.
        """
        if len(x) == 2:
            x = np.append(x, 0)
        U = 0
        for obs in self.obstacle_list:
            U += obs.U(x[:3])
        return U

    def calcPotentialUAndGradient(self, x):
        """
        Calculate both the potential function U and its gradient at a given state x.
        """
        if len(x) == 2:
            x = np.append(x, 0)
        U = 0
        grad_U = np.zeros(3)
        for obs in self.obstacle_list:
            U_obs, grad_U_obs = obs.UandGradU(x[:3])
            U += U_obs
            grad_U += grad_U_obs
        return U, grad_U

    def calcH(self, x, delta=0.0, u_value_precomputed=None):
        """
        Calculate h(x), the CBF (Control Barrier Function) value at a given state x.
        """
        if len(x) == 2:
            x = np.append(x, 0)
        U = self.calcPotentialU(x) if u_value_precomputed is None else u_value_precomputed
        h = 1 / (1 + U) - delta
        return h

    def calcHGradient(self, x, also_return_h_flag=False, output_convention='ENU'):
        """
        Calculate the gradient of h(x) at a given state x.
        ATTENTION: Returns vector only for positional dimensions initially.
        Input: ENU -> Output: ENU or NED
        """
        U, grad_U = self.calcPotentialUAndGradient(x)
        h_grad = -grad_U / (1 + U)**2

        # Append zeros for the other dimensions
        if self.model.num_of_states > 2:
            h_grad = np.append(h_grad, np.zeros(self.model.num_of_states - self.model.pos_dim))
        if output_convention == 'NED':
            h_grad_ned = h_grad.copy()
            h_grad_ned[0], h_grad_ned[1] = h_grad[1], h_grad[0]
            if len(h_grad_ned) > 2:
                h_grad_ned[2] = -h_grad[2]
            h_grad = h_grad_ned

        if also_return_h_flag:
            h = self.calcH(x, u_value_precomputed=U)
            return h, h_grad
        else:
            return h_grad

    def calcHessianH(self, x, epsilon=1e-3, output_convention='ENU'):
        """
        Calculate the Hessian of h(x) using finite differences.
        Input: ENU -> Output: ENU or NED
        """
        hessian_h = np.zeros((self.model.num_of_states, self.model.num_of_states))
        
        for i in range(self.model.pos_dim):
            for j in range(i, self.model.pos_dim):
                if i == j:
                    x_plus = x.copy()
                    x_plus[i] += epsilon
                    h_plus = self.calcH(x_plus)

                    x_minus = x.copy()
                    x_minus[i] -= epsilon
                    h_minus = self.calcH(x_minus)

                    hessian_h[i, i] = (h_plus - 2 * self.calcH(x) + h_minus) / (epsilon ** 2)
                else:
                    x_pp = x.copy(); x_pp[i] += epsilon; x_pp[j] += epsilon
                    x_pm = x.copy(); x_pm[i] += epsilon; x_pm[j] -= epsilon
                    x_mp = x.copy(); x_mp[i] -= epsilon; x_mp[j] += epsilon
                    x_mm = x.copy(); x_mm[i] -= epsilon; x_mm[j] -= epsilon

                    hessian_value = (self.calcH(x_pp) - self.calcH(x_pm) - self.calcH(x_mp) + self.calcH(x_mm)) / (4 * epsilon ** 2)
                    hessian_h[i, j] = hessian_value
                    hessian_h[j, i] = hessian_value

        if output_convention == 'NED':
            hess_h_ned = hessian_h.copy()
            hess_h_ned[[0, 1], :] = hessian_h[[1, 0], :]
            hess_h_ned[:, [0, 1]] = hess_h_ned[:, [1, 0]]
            if hess_h_ned.shape[0] > 2:
                hess_h_ned[2, :] = -hess_h_ned[2, :]
                hess_h_ned[:, 2] = -hess_h_ned[:, 2]
            return hess_h_ned

        return hessian_h

    def calcUsafe(self, x, udef_now, u_before):
        """
        Compute safe control input using CBF-QP.
        """
        p = self.cbf_params
        
        # Calculate CBF function h(x) and gradient
        h, grad_h = self.calcHGradient(self.model.position(x), also_return_h_flag=True, 
                                        output_convention=self.model.coord_convention)
        # Calculate CBF Hessian
        hess_h = self.calcHessianH(self.model.position(x), output_convention=self.model.coord_convention)

        # System Dynamics
        f = self.model.f(x, udef_now)
        if self.model.type == "FixedWing12DOFTrainer":
            f_x = self.model.f_x(x, udef_now, first_three_rows_only=True)
        else:
            f_x = self.model.f_x(x, udef_now)

        # Solve CBF-QP
        u_safe, h, h_dot, h_ddot, psi_2, L_G_psi2 = solve_cbf_qp(
            h=h,
            grad_h=grad_h,
            hess_h=hess_h,
            f=f,
            f_x=f_x,
            f_u=self.model.h(x),
            u_ref=udef_now,
            u_current=u_before,
            alpha_1=p['alpha_1'],
            alpha_2=p['alpha_2'],
            alpha_3=p['alpha_3'],
            alpha_u=p['alpha_u'],
            Kp=p['cbf_Kp'],
            dt=p['cbf_dt'],
            alpha_max=np.deg2rad(p['alpha_max_deg']),
            alpha_aoa_1=p['alpha_aoa_1'],
            alpha_aoa_2=p['alpha_aoa_2'],
            slack_penalty_aoa=p['slack_penalty_aoa'],
            x_state=x if p['use_aoa_constraint'] else None,
        )

        # Apply control limits
        u_safe[:3] = np.clip(u_safe[:3], self.model.input_limits[:3, 0], self.model.input_limits[:3, 1])
        u_safe[3] = np.clip(u_safe[3], -1, 1)

        # Publish debug information
        debug_msg = ObsAvoidanceDebug()
        debug_msg.psi = float(psi_2)
        debug_msg.h = float(h)
        debug_msg.hdot = float(h_dot)
        debug_msg.hddot = float(h_ddot)
        debug_msg.two_alpha_h_hdot = float(2 * p['alpha_1'] * h_dot)
        debug_msg.alpha2_h = float(p['alpha_2'] * h)
        debug_msg.beta = L_G_psi2.flatten().tolist()
        debug_msg.u_safe = u_safe.flatten().tolist()
        self.obs_avoidance_debug_publisher.publish(debug_msg)

        return u_safe

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
        msg.active_cbf_flag = self.active_cbf_flag
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
    
    def step(self, iteration=0):
        """Step the simulation forward one timestep with obstacle avoidance"""
        u = self.u_manual.copy()
        
        # Apply CBF safety filter if obstacles exist
        if len(self.obstacle_list) > 0:
            # Apply safety control every N iterations or if we were near an obstacle before
            skip_iter = self.cbf_params.get('cbf_skip_iter', 5)
            self.active_cbf_flag = bool(np.any(np.abs(self.u_safe) > 1e-4) or iteration % skip_iter == 0)
            
            if self.active_cbf_flag:
                self.u_safe = self.calcUsafe(self.model.state, u, self.u_before)
            else:
                self.u_safe = np.zeros(self.model.num_of_inputs)
            
            # Add safe control correction
            u += self.u_safe
            
            # Clip to input limits
            u = np.clip(u, self.model.input_limits[:, 0], self.model.input_limits[:, 1])
            
            # Smooth with previous control
            relax = self.cbf_params.get('relax_factor', 0.5)
            u = relax * u + (1 - relax) * self.u_before
        
        # Store for next iteration
        self.u_before = u.copy()
        
        # Step the model
        self.model.state = self.model.step(self.model.state, u)
        
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
    parser.add_argument('--obstacles_yaml', type=str, required=False,
                       default='None', help='Path to YAML file containing obstacle definitions')
    
    parsed_args, ros_args = parser.parse_known_args()
    
    # Extract arguments
    AGENT_ID = parsed_args.agent_id
    INIT_POS_3D = np.array(parsed_args.init_pos)
    L1_BOUNDS = [parsed_args.l_bounds[0], parsed_args.l_bounds[1]]
    L2_BOUNDS = [parsed_args.l_bounds[2], parsed_args.l_bounds[3]]
    MODEL_TYPE = parsed_args.model_type
    PUBLISH_FREQ = parsed_args.publish_freq
    OBSTACLES_YAML_PATH = parsed_args.obstacles_yaml
    
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
    
    # Extract CBF parameters from config if available
    cbf_params = None
    if 'control' in agent_config:
        control_config = agent_config['control']
        cbf_params = {
            'alpha_1': control_config.get('alpha_1', 0.1),
            'alpha_2': control_config.get('alpha_2', 3.0),
            'alpha_3': control_config.get('alpha_3', 15.0),
            'alpha_u': control_config.get('alpha_u', 50.0),
            'cbf_Kp': control_config.get('cbf_Kp', 5.0),
            'cbf_dt': control_config.get('cbf_dt', 0.025),
            'alpha_max_deg': control_config.get('alpha_max_deg', 8.0),
            'alpha_aoa_1': control_config.get('alpha_aoa_1', 10.0),
            'alpha_aoa_2': control_config.get('alpha_aoa_2', 15.0),
            'slack_penalty_aoa': control_config.get('slack_penalty_aoa', 300.0),
            'use_aoa_constraint': control_config.get('use_aoa_constraint', True),
            'delta_safe': control_config.get('delta_safe', 0.0),
            'cbf_skip_iter': control_config.get('cbf_skip_iter', 5),
            'relax_factor': control_config.get('relax_factor', 0.5),
            'kappa_wall': control_config.get('kappa_wall', 1.0),
            'rho_wall': control_config.get('rho_wall', 0.75),
            'kappa_obs': control_config.get('kappa_obs', 1.0),
            'rho_obs': control_config.get('rho_obs', 0.75),
        }
    
    # Initialize ROS2
    rclpy.init(args=ros_args)
    
    # Create teleoperation agent node
    teleop_agent = TeleopAgent(
        agent_id=AGENT_ID,
        dynamic_model=dynamic_model,
        init_pos_3d=INIT_POS_3D,
        l1_bounds=L1_BOUNDS,
        l2_bounds=L2_BOUNDS,
        publish_freq=PUBLISH_FREQ,
        cbf_params=cbf_params
    )
    
    # Load obstacles -------------------
    # Always load the default walls to keep us inside L bound domain
    KAPPA_WALL = cbf_params.get('kappa_wall', 1.0) if cbf_params else 1.0
    RHO_WALL = cbf_params.get('rho_wall', 0.75) if cbf_params else 0.75
    KAPPA_OBS = cbf_params.get('kappa_obs', 1.0) if cbf_params else 1.0
    RHO_OBS = cbf_params.get('rho_obs', 0.75) if cbf_params else 0.75
    
    obstacle_default_walls = loadObstaclesFromYaml(
        'src/ergodic_exploration/ergodic_exploration/default_walls.yaml', 
        L1_BOUNDS, L2_BOUNDS,
        kappa_obs=KAPPA_OBS, rho_obs=RHO_OBS,
        kappa_wall=KAPPA_WALL, rho_wall=RHO_WALL
    )
    saveObstaclesToMemory(teleop_agent, obs_list=obstacle_default_walls)
    
    # Load obstacles from custom YAML configuration file if available
    if OBSTACLES_YAML_PATH != "None":
        obstacles_from_yaml = loadObstaclesFromYaml(
            OBSTACLES_YAML_PATH, L1_BOUNDS, L2_BOUNDS,
            kappa_obs=KAPPA_OBS, rho_obs=RHO_OBS,
            kappa_wall=KAPPA_WALL, rho_wall=RHO_WALL
        )
        if obstacles_from_yaml:
            saveObstaclesToMemory(teleop_agent, obs_list=obstacles_from_yaml)
            print(f"Loaded {len(obstacles_from_yaml)} obstacles from {OBSTACLES_YAML_PATH}")
        else:
            print("Warning: No obstacles loaded from YAML file.")
    
    print(f"Total obstacles loaded: {len(teleop_agent.obstacle_list)}")
    
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
    print("="*60)
    if len(teleop_agent.obstacle_list) > 0:
        print(f"\nOBSTACLE AVOIDANCE ENABLED ({len(teleop_agent.obstacle_list)} obstacles)")
        print("CBF safety filter will modify control inputs to avoid obstacles.")
    else:
        print("\nNo obstacles loaded - flying without safety constraints.")
    print("="*60 + "\n")
    
    # Simulation loop
    def simulationFunction():
        """Main simulation loop"""
        target_dt = dynamic_model.dt
        iteration = 0
        
        while not shutdown_flag.is_set():
            loop_start = time.time()
            
            # Step the simulation with obstacle avoidance
            actual_dt = teleop_agent.step(iteration=iteration)
            
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
