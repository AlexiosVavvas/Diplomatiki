import numpy as np
from abc import ABC, abstractmethod

class DynamicsBase(ABC):
    """
    Base class for all dynamics models providing common functionality.
    """
    
    def __init__(self, dt=0.001, x0=None, num_of_states=None, num_of_inputs=None, state_names=None):
        self.dt = dt
        self.num_of_states = num_of_states
        self.num_of_inputs = num_of_inputs
        self.state_names = state_names if state_names is not None else [f"x{i}" for i in range(num_of_states)]
        self.reset(x0)
    
    def reset(self, state=None):
        """Reset the system to initial state."""
        if state is None:
            # random seed for reproducibility
            np.random.seed(0)
            self.state = np.random.uniform(0., 1., size=(self.num_of_states,))
        else:
            assert len(state) == self.num_of_states, f"Reset Input state must be of length: {self.num_of_states}."
            self.state = np.array(state.copy())
        return self.state.copy()
    
    @abstractmethod
    def f(self, x, u):
        """Continuous time dynamics - must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def f_x(self, x, u):
        """Jacobian of dynamics with respect to state - must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def f_u(self, x):
        """Jacobian of dynamics with respect to input - must be implemented by subclasses."""
        pass
    
    def g(self, x):
        """
        Affine Dynamics, States Part
        x' = f(x) = g(x) + h(x) u
        Default implementation assumes f(x, 0) = g(x)
        """
        return self.f(x, np.zeros((self.num_of_inputs,)))
    
    def h(self, x):
        """
        Affine Dynamics Control Part
        x' = f(x) = g(x) + h(x) u
        Default implementation returns f_u(x)
        """
        return self.f_u(x)
    
    def step(self, x, u, dt=None):
        """Euler integration step."""
        dt = self.dt if dt is None else dt
        self.state = self.state + self.f(self.state, u) * dt
        return self.state
    
    def simulateForward(self, x0, ti, udef=None, T=1.0, dt=None):
        """
        Simulate the system forward in time from ti -> ti+T
        """
        dt = self.dt if dt is None else dt
        t = ti
        x = x0.copy()
        x_traj = []
        u_traj = []
        t_traj = []
        
        # Check for callable udef
        assert callable(udef) or udef is None, "udef must be a callable function or None."

        # Reset the model with the initial state and simulate forward
        self.reset(x0)
        while t < ti + T:
            udef_ = udef(x, t) if callable(udef) else np.zeros((self.num_of_inputs,))
            x = self.step(x=x, u=udef_, dt=dt)
            x_traj.append(x.copy())
            u_traj.append(udef_.copy())
            t_traj.append(t)
            t += dt  # Increment time by the model's time step
        
        self.reset(x0)  # Reset the model to the initial state after simulation
        return np.array(x_traj), np.array(u_traj), np.array(t_traj)
    
    def convertForcesToInputs(self, F):
        """
        Convert forces to control inputs - default implementation for 2D forces.
        Override in subclasses for more complex conversions.
        """
        fx, fy = F
        return np.array([fx, fy])
    
    @property
    def ergodic_state(self):
        """Return the ergodic state (typically position). Default: first 2 elements."""
        return self.state[:2].copy()
    
    @property
    @abstractmethod
    def state_string(self):
        """String representation of current state - must be implemented by subclasses."""
        pass


class SingleIntegrator(DynamicsBase):
    '''
    Basic First Order Dynamics Model ----
    Model: 
        x1' = u1       -> x1 = x
        x2' = u2       -> x2 = y
    So, the state is:
        x = [x1, x2]    -> Ergodic state: xv = [x, y] = [x1, x2]
        x = [x, y]
        u = [u1, u2]
    '''
    def __init__(self, dt=0.001, x0=None):
        super().__init__(dt=dt, x0=x0, num_of_states=2, num_of_inputs=2, 
                        state_names=["x", "y"])

        self.type = "SingleIntegrator"
        self.A = np.array([
                [0., 0.],
                [0., 0.]
        ])# - np.diag([0,0,1,1]) * 0.25

        self.B = np.array([
                [1.0, 0.],
                [0., 1.0]
        ])
    
    def f(self, x, u):
        '''
        Continuous time dynamics
        '''
        return self.A @ x + self.B @ u

    def f_x(self, x, u):
        '''
        Jacobian of the dynamics with respect to x
        '''
        return self.A.copy()

    def f_u(self, x):
        '''
        Jacobian of the dynamics with respect to u
        '''
        return self.B.copy()

    def g(self, x):
        """
        Affine Dynamics, States Part
        x' = f(x) = g(x) + h(x) u
        """
        return self.A @ x 
    
    @property
    def state_string(self):
        return f"x: {self.state[0]:.2f}, y: {self.state[1]:.2f}"
    


class DoubleIntegrator(DynamicsBase):
    '''
    Basic Second Order Dynamics Model ----
    Model: 
        x1'' = u1       -> x1 = x  |  x3 = x'
        x2'' = u2       -> x2 = y  |  x4 = y'
    Or equivalently:
        x1' = x3
        x2' = x4
        x3' = u1
        x4' = u2
    So, the state is:
        x = [x1, x2, x3, x4]    -> Ergodic state: xv = [x, y] = [x1, x2]
        x = [x,  y,  x', y']
        u = [u1, u2]
    
    Note: By design of my code, the ergodic states should ALWAYS be the first two elements of the state vector.
    '''
    def __init__(self, mass=1, dt=0.001, x0=None, damping=0):
        super().__init__(dt=dt, x0=x0, num_of_states=4, num_of_inputs=2, 
                        state_names=["x", "y", "x'", "y'"])
        
        self.type = "DoubleIntegrator"
        self.m = mass
        self.b = damping
        # v' + b/m * v = u/m : First order LTI in velocity, so τ = m/b
        # For τ = 1s (velocity go to zero in 1 second) we can choose b = m/1

        self.A = np.array([
                [0., 0., 1.0,            0.            ],
                [0., 0., 0.,             1.0           ],
                [0., 0., -self.b/self.m, 0.            ],
                [0., 0., 0.,             -self.b/self.m]
        ])

        self.B = np.array([
                [0., 0.],
                [0., 0.],
                [1.0, 0.],
                [0., 1.0]
        ]) / self.m

    def f(self, x, u):
        '''
        Continuous time dynamics
        '''
        return self.A @ x + self.B @ u

    def f_x(self, x, u):
        '''
        Jacobian of the dynamics with respect to x
        '''
        return self.A.copy()

    def f_u(self, x):
        '''
        Jacobian of the dynamics with respect to u
        '''
        return self.B.copy()

    def g(self, x):
        """
        Affine Dynamics, States Part
        x' = f(x) = g(x) + h(x) u
        """
        return self.A @ x 

    @property
    def state_string(self):
        return f"x: {self.state[0]:.2f}, y: {self.state[1]:.2f}, | x': {self.state[2]:.2f}, y': {self.state[3]:.2f}"


from scipy.linalg import solve_continuous_are

class Quadcopter(DynamicsBase):
    '''
    Basic Quadcopter Dynamics Model ----
    Model:
        x1' = x7
        x2' = x8
        x3' = x9
        x4' = x10
        x5' = x11
        x6' = x12
        x7' = u1 * (sin(phi) * sin(psi) + cos(phi) * cos(psi) * sin(theta)) / m
        x8' = u1 * (cos(phi) * sin(theta) * sin(psi) - cos(psi) * sin(phi)) / m
        x9' = u1 * cos(theta) * cos(phi)/m  -  9.81
        x10' = u2 - damping * x10
        x11' = u3 - damping * x11
        x12' = u4 - damping * x12
    So, the state is:
        x = [x1, x2, x3, x4,  x5,    x6,  x7,  x8,  x9,  x10,    x11,      x12   ]    -> Ergodic state: xv = [x, y] = [x1, x2]
        x = [x,  y,  z,  psi, theta, phi, x',  y',  z',  psidot, thetadot, phidot]
        u = [u1, u2, u3, u4]
            u[0]: Total thrust force in the body z-direction
            u[1]: Yaw moment/torque   (controls rotation around z-axis)
            u[2]: Pitch moment/torque (controls rotation around y-axis)
            u[3]: Roll moment/torque  (controls rotation around x-axis)
    '''

    def __init__(self, dt=0.001, x0=None, mass=0.1, damping=0, Q=None, R=None, z_target=1, motor_limits=None, zero_out_states=None):
        super().__init__(dt=dt, x0=x0, num_of_states=12, num_of_inputs=4, 
                        state_names=["x", "y", "z", "ψ", "θ", "φ", "x'", "y'", "z'", "ψ'", "θ'", "φ'"])

        self.type = "Quadcopter"
        self.m = mass
        self.damping = damping
        self.A = np.zeros((self.num_of_states, self.num_of_states)) +  np.diag([1.0]*6, 6)
        self.B = np.zeros((self.num_of_states, self.num_of_inputs))
        self.z_target = z_target
        
        # Default state target
        self.state_target = np.array([0, 0, z_target, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        self._state_target = self.state_target.copy() # Temporary: Needed for obstacle controllers to have one to append velocity commands
        self._state_target_history_for_plotting = self.state_target.copy()
        self.state_target_modified = False
        self.f_command_to_controller = None

        # Lets now set the motor limits
        self.input_limits, self.motor_limits = self.convertMotorLimitsToInputLimits(motor_limits)

        # Dictionary of state_names and positions
        self._state_names_dict = {name: i for i, name in enumerate(self.state_names)}

        # LQR Control for stabilization ------------
        # Zeroed out are the states for which we dont care to implement LQR control (like position for an airplane etc)
        if zero_out_states is not None:
            assert isinstance(zero_out_states, list), "zero_out_states must be a list of state names."
            assert all(state in self.state_names for state in zero_out_states), f"zero_out_states must be a list of state names from: {self.state_names}."
        self.zero_out_states = zero_out_states
        #                                                    [x,    y,    z,   psi,  theta, phi,  x',   y',   z',  psidot, thetadot, phidot]
        self.Q = np.asarray(Q) if Q is not None else np.diag([0.01, 0.01, 100, 0.01, 0.1,   0.1,  0.1,  0.1,  1,  0.1,    0.1,      0.1])
        self.R = np.asarray(R) if R is not None else np.diag([1, 1, 1, 1]) # TODO: Maybe change R, since it doesnt refer to motor inputs, but to input_u. But with which mapping...?
        self.k_lqr = self._calculateLqrControlGain(self.Q, self.R)
        # Lets have also a Q for obstacle avoidance if nesessary [x,    y,    z,   psi,  theta, phi,  x',   y',   z',  psidot, thetadot, phidot]
        self.Q_obs = np.asarray(Q) if Q is not None else np.diag([0.01, 0.01, 100, 0.01, 0.1,   0.1,  150,  150,  1,  0.1,    0.1,      0.1])

    def reset(self, state=None):
        if state is None:
            # random seed for reproducibility
            np.random.seed(0)
            self.state = np.random.uniform(0., 1., size=(self.num_of_states,))
            self.state[4:] = 0
        else:
            assert len(state) == self.num_of_states, f"Reset Input state must be of length: {self.num_of_states}."
            self.state = np.array(state.copy())
        return self.state.copy()
    
    def rk4Step(self, f, x, dt, *args):
        """
        Fourth-order Runge-Kutta integration method
        """
        k1 = f(x, *args)
        k2 = f(x + 0.5*dt*k1, *args)
        k3 = f(x + 0.5*dt*k2, *args)
        k4 = f(x + dt*k3, *args)
        return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    def f(self, x, u):

        # Lets clip the inputs to the limits
        u = np.clip(u, self.input_limits[:, 0], self.input_limits[:, 1])

        psi = x[3]
        theta = x[4]
        phi = x[5]

        xddot = u[0] * (np.sin(phi) * np.sin(psi) + np.cos(phi) * np.cos(psi) * np.sin(theta)) / self.m
        yddot = u[0] * (np.cos(phi) * np.sin(theta) * np.sin(psi) - np.cos(psi) * np.sin(phi)) / self.m
        zddot = u[0] * np.cos(theta) * np.cos(phi)/self.m  -  9.81

        psiddot = u[1] - self.damping * x[9]
        thetaddot = u[2] - self.damping * x[10]
        phiddot = u[3] - self.damping * x[11]

        return np.array([
                x[6],
                x[7],
                x[8],
                x[9],
                x[10],
                x[11],
                xddot,
                yddot,
                zddot,
                psiddot,
                thetaddot,
                phiddot
            ])

    def f_x(self, x, u):
        # Lets clip the inputs to the limits
        u = np.clip(u, self.input_limits[:, 0], self.input_limits[:, 1])
        
        psi = x[3]
        theta = x[4]
        phi = x[5]
        # A = np.zeros((self.nX, self.nX)) +  np.diag([1.0]*6, 6)
        self.A[6,3] = u[0] * (np.cos(psi) * np.sin(phi) - np.cos(phi) * np.sin(theta)*np.sin(psi) )/self.m
        self.A[6,4] = u[0] * np.cos(theta) * np.cos(phi) * np.cos(psi) / self.m
        self.A[6,5] = u[0] * (-np.cos(psi) * np.sin(theta) * np.sin(phi) + np.cos(phi) * np.sin(psi))/self.m
        self.A[7,3] = u[0] * (np.cos(phi) * np.cos(psi)*np.sin(theta) + np.sin(phi)*np.sin(psi) )/self.m
        self.A[7,4] = u[0] * np.cos(theta) * np.cos(phi) * np.sin(psi) / self.m
        self.A[7,5] = u[0] * (-np.cos(phi) * np.cos(psi) - np.sin(theta) * np.sin(phi) * np.sin(psi))/self.m
        self.A[8,4] = -u[0] * np.cos(phi) * np.sin(theta) / self.m
        self.A[8,5] = -u[0] * np.cos(theta) * np.sin(phi) / self.m
        self.A[9,9] = -self.damping
        self.A[10,10] = -self.damping
        self.A[11,11] = -self.damping
        return self.A
    
    def f_u(self, x):
        psi = x[3]
        theta = x[4]
        phi = x[5]
        self.B[6,0] = (np.cos(phi) * np.cos(psi) * np.sin(theta) + np.sin(phi) * np.sin(psi) )/ self.m
        self.B[7,0] = (-np.cos(psi) * np.sin(phi) + np.cos(phi) * np.sin(theta) * np.sin(psi)) / self.m
        self.B[8,0] = np.cos(theta) * np.cos(phi) / self.m
        self.B[9,1] = 1.0
        self.B[10,2] = 1.0
        self.B[11,3] = 1.0
        return self.B

    def h(self, x):
        '''
        Affine Dynamics Control Part
        x' = f(x) = g(x) + h(x) u
        '''
        return self.f_u(x)

    # TODO: To verify if this is correct
    def g(self, x):
        """
        Affine Dynamics, States Part
        x' = f(x) = g(x) + h(x) u
        """
        return np.array([
            x[6],  # x'
            x[7],  # y'
            x[8],  # z'
            x[9],  # ψ'
            x[10], # θ'
            x[11],  # φ'
            0,
            0,
            -9.81,
            -self.damping * x[9],  # ψ'' = -damping * ψ'
            -self.damping * x[10], # θ'' = -damping * θ'
            -self.damping * x[11]  # φ'' = -damping * φ'
        ])

    def step(self, x, u, dt=None):
        dt = self.dt if dt is None else dt

        # Lets clip the inputs to the limits
        u = np.asarray(u)
        # m = self.convertInputToMotorCommands(u) # TODO: This leads to imbalance
        # m = np.clip(m, self.motor_limits[:, 0], self.motor_limits[:, 1])
        # u = self.convertMotorCommandsToInput(m)
        u = np.clip(u, self.input_limits[:, 0], self.input_limits[:, 1])
        
        return self.rk4Step(self.f, x, dt, *(u,))
    
    def _calculateLqrControlGain(self, Q, R):
        """
        Calculate the LQR control gain matrix K using the continuous-time algebraic Riccati equation.
        """
        assert Q.shape == (self.num_of_states, self.num_of_states), "Q must be a square matrix of size num_of_states."
        assert R.shape == (self.num_of_inputs, self.num_of_inputs), "R must be a square matrix of size num_of_inputs."
        
        # u_nom -> Thrust = Weight - Torque = 0
        u_nom = np.zeros((self.num_of_inputs,))
        u_nom[0] = self.m * 9.81

        self.f_x(self.state, u_nom)
        self.f_u(self.state)

        # Solve the continuous-time algebraic Riccati equation
        P = solve_continuous_are(self.A, self.B, Q, R)

        # Calculate the LQR gain
        K = np.linalg.inv(R) @ self.B.T @ P

        # Zero out the states that we dont care about
        if self.zero_out_states is not None:
            indices = [self._state_names_dict[state_name] for state_name in self.zero_out_states if state_name in self._state_names_dict]
            K[:, indices] = 0

        return K

    def calcLQRcontrol(self, x, t, state_target=None):
        """
        This is the Nominal Input we use to the ergodic controller
        """
        state_target = self._state_target.copy() if state_target is None else state_target

        # Reset the state target flag to let future controllers change it if nesessary
        self.state_target_modified = False
        self._state_target = self.state_target.copy()
        self._state_target_history_for_plotting = state_target.copy()

        # Calculate the control input
        u = -self.k_lqr @ (x - state_target)

        u[0] += self.m * 9.81 # Adjust thrust to maintain altitude

        # Lets clip the inputs to the limits
        u = np.clip(u, self.input_limits[:, 0], self.input_limits[:, 1])
        
        return u

    def convertInputToMotorCommands(self, u):
        """
        Convert abstract control inputs to individual motor commands
        
        Parameters:
        u[0]: Total thrust
        u[1]: Yaw torque
        u[2]: Pitch torque
        u[3]: Roll torque
        
        Returns:
        Array of 4 motor commands [m1, m2, m3, m4]
        """
        # Motor mixing matrix for X configuration
        # Assuming:
        # m1: front right
        # m2: front left
        # m3: rear left
        # m4: rear right
        
        # Extract control inputs
        thrust = u[0]  # Total thrust
        yaw = u[1]     # Yaw torque
        pitch = u[2]   # Pitch torque
        roll = u[3]    # Roll torque
        
        # Apply mixer matrix
        m1 = thrust/4 + yaw/4 + pitch/4 + roll/4   # Front right
        m2 = thrust/4 - yaw/4 + pitch/4 - roll/4   # Front left
        m3 = thrust/4 + yaw/4 - pitch/4 - roll/4   # Rear left
        m4 = thrust/4 - yaw/4 - pitch/4 + roll/4   # Rear right
        
        # Ensure no negative motor commands
        motors = np.maximum(0, np.array([m1, m2, m3, m4]))
        
        return motors
    
    def convertMotorCommandsToInput(self, motors):
        """
        Convert motor commands back to control inputs
        
        Parameters:
        motors: Array of 4 motor commands [m1, m2, m3, m4]
        
        Returns:
        Array of control inputs [u1, u2, u3, u4]
        """
        # Extract motor commands
        m1, m2, m3, m4 = motors
        
        # Apply inverse mixer matrix
        thrust = m1 + m2 + m3 + m4
        yaw = m1 - m2 + m3 - m4
        pitch = m1 + m2 - m3 - m4
        roll = m1 - m2 - m3 + m4
        
        return np.array([thrust, yaw, pitch, roll])

    def convertMotorLimitsToInputLimits(self, motor_limits=None):
        # TODO: The limits are not converted properly. A mapping of the limits cant be made, we need a convert -> clip -> convert approach
        if motor_limits is None:
            # Set infinite limits if not provided
            motor_limits = np.array([[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]])
        else:
            motor_limits = np.asarray(motor_limits)
            assert motor_limits.shape == (4, 2), "motor_limits should be a 4x2 array with [lower, upper] pairs for each motor."
            # Make sure every lower bound is less than the upper bound
            assert np.all(motor_limits[:, 0] < motor_limits[:, 1]), "Lower bounds must be less than upper bounds."

        m_min = motor_limits[:, 0]
        m_max = motor_limits[:, 1]

        is_any_max_inf = np.any(m_max == np.inf)
        is_any_min_inf = np.any(m_min == -np.inf)

        # Throttle limits
        # m1 + m2 + m3 + m4
        t_max = +np.inf if is_any_max_inf else m_max[0] + m_max[1] + m_max[2] + m_max[3]
        t_min = -np.inf if is_any_min_inf else m_min[0] + m_min[1] + m_min[2] + m_min[3]

        # Yaw limits
        # m1 - m2 + m3 - m4
        if m_max[0] == np.inf or m_max[2] == np.inf or m_min[1] == -np.inf or m_min[3] == -np.inf:
            y_max = np.inf
        else: 
            y_max = m_max[0] - m_min[1] + m_max[2] - m_min[3]
        if m_min[0] == -np.inf or m_min[2] == -np.inf or m_max[1] == np.inf or m_max[3] == np.inf:
            y_min = -np.inf
        else:
            y_min = m_min[0] - m_max[1] + m_min[2] - m_max[3]

        # Pitch limits
        # m1 + m2 - m3 - m4
        if m_max[0] == np.inf or m_max[1] == np.inf or m_min[2] == -np.inf or m_min[3] == -np.inf:
            p_max = np.inf
        else: 
            p_max = m_max[0] + m_max[1] - m_min[2] - m_min[3]
        if m_min[0] == -np.inf or m_min[1] == -np.inf or m_max[2] == np.inf or m_max[3] == np.inf:
            p_min = -np.inf
        else:
            p_min = m_min[0] + m_min[1] - m_max[2] - m_max[3]

        # Roll limits
        # m1 - m2 - m3 + m4
        if m_max[0] == np.inf or m_min[1] == -np.inf or m_min[2] == -np.inf or m_max[3] == np.inf:
            r_max = np.inf
        else: 
            r_max = m_max[0] - m_min[1] - m_min[2] + m_max[3]
        if m_min[0] == -np.inf or m_max[1] == np.inf or m_max[2] == np.inf or m_min[3] == -np.inf:
            r_min = -np.inf
        else:
            r_min = m_min[0] - m_max[1] - m_max[2] + m_min[3]

        u_limits = np.array([[t_min, t_max], [y_min, y_max], [p_min, p_max], [r_min, r_max]])
        # print limits
        print("Motor Limits: \n", motor_limits)
        print("Input Limits: \n", u_limits)
        return u_limits, motor_limits
    
    def convertForcesToInputs(self, F):
        # F -> (Fx, Fy)
        fx, fy = F

        psi = self.state[3]
        s = np.sin(psi)
        c = np.cos(psi)
        
        m1_pos = [-c+s, -c-s]
        m2_pos = [-c-s, +c-s]
        m3_pos = [+c-s, +c+s]
        m4_pos = [+c+s, +c-s]

        # Z = slope_x * x + slope_y * y
        # Z = - fx * x - fy * y

        m1 = - fx * m1_pos[0] - fy * m1_pos[1]
        m2 = - fx * m2_pos[0] - fy * m2_pos[1]
        m3 = - fx * m3_pos[0] - fy * m3_pos[1]
        m4 = - fx * m4_pos[0] - fy * m4_pos[1]

        return self.convertMotorCommandsToInput(np.array([m1, m2, m3, m4]))

    @property
    def state_string(self):
        return f"x: {self.state[0]:.2f}, y: {self.state[1]:.2f}, z: {self.state[2]:.2f}, | ψ: {self.state[3]*180/np.pi:.2f}, θ: {self.state[4]*180/np.pi:.2f}, φ: {self.state[5]*180/np.pi:.2f}, | x': {self.state[6]:.2f}, y': {self.state[7]:.2f}, z': {self.state[8]:.2f}, | ψ': {self.state[9]*180/np.pi:.2f}, θ': {self.state[10]*180/np.pi:.2f}, φ': {self.state[11]*180/np.pi:.2f} [angles -> DEG]"
    

class SimpleBoatSecondOrder(DynamicsBase):
    def __init__(
        self,
        dt=0.001,
        x0=None,
        m=3.0,                  # 3 kg: typical for small robotic hull
        Iz=0.25,                # kg*m^2 (order: m*L^2 / 12, with L≈0.6 m)
        d_v=5.0,                # increased surge drag, SI units (N/(m/s)^2)
        d_w=2.0,                # increased yaw drag, SI units (N*m/(rad/s)^2)
        k_delta=4.0,            # rudder effectiveness, in typical range for small boats
        max_allowed_rev_thr=0,  # Negative thrust is allowed. Max 0 means no reverse.
                                # If max =  2, we only allow reverse thrust up to 2 N
                                # If max = -2, we dont allow forward thrust to get below 2 N 
        rudder_priority=4       # Used to prioritize rudder over thrust when the latter is saturated (Cant reverse thrust in the corner)
    ):
        super().__init__(dt=dt, x0=x0, num_of_states=5, num_of_inputs=2, 
                        state_names=["x", "y", "psi", "v", "omega"])

        self.type = "SimpleBoatSecondOrder"
        self.m = m
        self.Iz = Iz
        self.d_v = d_v
        self.d_w = d_w
        self.k_delta = k_delta
        self.input_names = ["thrust", "rudder"]
        self.max_allowed_rev_thr = max_allowed_rev_thr
        self.rudder_priority = rudder_priority

    def f(self, x, u):
        x_, y_, psi, v, omega = x
        thrust, rudder = u
        dx = v * np.cos(psi)
        dy = v * np.sin(psi)
        dpsi = omega
        dv = (thrust - self.d_v * v * abs(v)) / self.m
        domega = (self.k_delta * v * rudder - self.d_w * omega * abs(omega)) / self.Iz
        return np.array([dx, dy, dpsi, dv, domega])
    
    def f_x(self, x, u):
        '''
        Jacobian of the dynamics with respect to x
        '''
        fx = np.zeros((self.num_of_states, self.num_of_states))
        x_, y_, psi, v, omega = x
        thrust, rudder = u
        fx[0, 2] = -v * np.sin(psi)  # ∂(dx)/∂(psi)
        fx[0, 3] = np.cos(psi)       # ∂(dx)/∂(v)
        fx[1, 2] = v * np.cos(psi)   # ∂(dy)/∂(psi)
        fx[1, 3] = np.sin(psi)       # ∂(dy)/∂(v)
        fx[2, 4] = 1.0               # ∂(dpsi)/∂(omega)
        fx[3, 3] = -2 * self.d_v * abs(v) / self.m  # ∂(dv)/∂(v)
        fx[4, 4] = -2 * self.d_w * abs(omega) / self.Iz  # ∂(domega)/∂(omega)
    
        return fx.copy()

    def f_u(self, x):
        '''
        Jacobian of the dynamics with respect to u
        '''
        fu = np.zeros((self.num_of_states, self.num_of_inputs))
        x_, y_, psi, v, omega = x
        fu[3, 0] = 1.0 / self.m          # ∂(dv)/∂(thrust)
        fu[4, 1] = self.k_delta * v / self.Iz  # ∂(domega)/∂(rudder)
        return fu.copy()

    @property
    def state_string(self):
        x, y, psi, v, omega = self.state
        return f"x={x:.2f} y={y:.2f} psi={psi:.2f} v={v:.2f} omega={omega:.2f}"


class SimpleCarSecondOrder(DynamicsBase):
    def __init__(
        self,
        dt=0.001,
        x0=None,
        m=8.0,                  # kg: typical small UGV mass
        L=0.9,                  # m: wheelbase
        b_v=1.0,                # viscous surge damping (N/(m/s))
        d_v=5.0,                # quadratic surge drag (N/(m/s)^2)
        k_delta=20.0,           # steering actuator bandwidth (1/s)
        k_steer=5.0,            # steering -> yaw torque gain (tuneable)
        Iz=0.8,                 # yaw inertia (kg*m^2) - tune to vehicle size
        d_r=1.0,                # yaw damping coefficient
        u_epsilon=1e-2,         # small speed floor for smoothing
        max_allowed_rev_thr=-1, # same semantics as in the boat class
        steer_priority=0.004
    ):
        # now 6 states: x, y, psi, u, delta, omega
        super().__init__(dt=dt, x0=x0, num_of_states=6, num_of_inputs=2,
                         state_names=["x", "y", "psi", "u", "delta", "omega"])

        self.type = "SimpleCarSecondOrder"
        self.m = m
        self.L = L
        self.b_v = b_v
        self.d_v = d_v
        self.k_delta = k_delta
        self.k_steer = k_steer
        self.Iz = Iz
        self.d_r = d_r
        self.u_epsilon = u_epsilon

        self.input_names = ["drive", "steer_cmd"]
        self.max_allowed_rev_thr = max_allowed_rev_thr
        self.steer_priority = steer_priority

    def f(self, x, u):
        """
        States: x = [x, y, psi, u, delta, omega]
        Inputs: u = [drive_force, steer_cmd]
        """
        x_pos, y_pos, psi, u_speed, delta, omega = x
        drive_force, steer_cmd = u

        # position kinematics
        dx = u_speed * np.cos(psi)
        dy = u_speed * np.sin(psi)

        # yaw kinematics now uses omega directly
        dpsi = omega

        # longitudinal dynamics: viscous + quadratic drag
        du = (drive_force - self.b_v * u_speed - self.d_v * u_speed * abs(u_speed)) / self.m

        # steering actuator (first-order)
        ddelta = -self.k_delta * (delta - steer_cmd)

        # yaw dynamics: steering (via delta) produces yaw torque proportional to u * delta
        # similar in spirit to boat's rudder torque: torque ≈ k_steer * u * delta
        domega = (self.k_steer * u_speed * delta - self.d_r * omega * abs(omega)) / self.Iz

        return np.array([dx, dy, dpsi, du, ddelta, domega])

    def f_x(self, x, u):
        """
        Jacobian of f w.r.t. state x (6x6 matrix)
        """
        fx = np.zeros((self.num_of_states, self.num_of_states))
        x_pos, y_pos, psi, u_speed, delta, omega = x
        drive_force, steer_cmd = u

        # ∂(dx)/∂psi and ∂(dx)/∂u
        fx[0, 2] = -u_speed * np.sin(psi)
        fx[0, 3] = np.cos(psi)

        # ∂(dy)/∂psi and ∂(dy)/∂u
        fx[1, 2] = u_speed * np.cos(psi)
        fx[1, 3] = np.sin(psi)

        # ∂(dpsi)/∂omega
        fx[2, 5] = 1.0

        # longitudinal accel derivative wrt u_speed: -(b_v + 2*d_v*abs(u))/m
        fx[3, 3] = -(self.b_v + 2.0 * self.d_v * abs(u_speed)) / self.m

        # steering actuator derivative wrt delta
        fx[4, 4] = -self.k_delta

        # yaw accel derivatives:
        # domega = (k_steer * u_speed * delta - d_r * omega * abs(omega)) / Iz
        # ∂domega/∂u_speed = (k_steer * delta) / Iz
        fx[5, 3] = (self.k_steer * delta) / max(self.Iz, 1e-9)
        # ∂domega/∂delta = (k_steer * u_speed) / Iz
        fx[5, 4] = (self.k_steer * u_speed) / max(self.Iz, 1e-9)
        # ∂domega/∂omega = -2 * d_r * abs(omega) / Iz
        fx[5, 5] = -2.0 * self.d_r * abs(omega) / max(self.Iz, 1e-9)

        return fx.copy()

    def f_u(self, x):
        """
        Jacobian of f w.r.t. inputs u (6x2 matrix)
        Inputs are [drive_force, steer_cmd]
        """
        fu = np.zeros((self.num_of_states, self.num_of_inputs))
        x_pos, y_pos, psi, u_speed, delta, omega = x

        # ∂(du)/∂(drive_force) = 1/m
        fu[3, 0] = 1.0 / self.m

        # ∂(ddelta)/∂(steer_cmd) = k_delta (steering actuator)
        fu[4, 1] = self.k_delta

        # steer_cmd affects domega only indirectly (via delta dynamics),
        # so no direct term for domega here. The solver/CBF will see influence
        # through f_x (domega/∂delta) and fu[4,1].

        return fu.copy()

    @property
    def state_string(self):
        x_pos, y_pos, psi, u_speed, delta, omega = self.state
        return f"x={x_pos:.2f} y={y_pos:.2f} psi={psi:.2f} u={u_speed:.2f} delta={delta:.2f} omega={omega:.2f}"

