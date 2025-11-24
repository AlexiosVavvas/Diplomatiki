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
    def f_u(self, x, u=None):
        """
        Jacobian of dynamics with respect to input - must be implemented by subclasses.
        
        Args:
            x: State vector
            u: Input vector at which to evaluate Jacobian (optional, defaults vary by implementation)
        """
        pass
    
    def g(self, x, u_ref=None):
        """
        Affine Dynamics, Drift Term
        x' = f(x,u) ≈ g(x) + h(x) u
        
        For better approximation around a reference input u_ref:
        g(x) = f(x, u_ref) - h(x) @ u_ref
        
        Args:
            x: State vector
            u_ref: Reference input (default: zeros, or trim if available)
        
        Returns:
            Drift vector g(x)
        """
        if u_ref is None:
            # Try to use trim input if available, otherwise zeros
            u_ref = getattr(self, 'u_trim', np.zeros((self.num_of_inputs,)))
        
        # g(x) = f(x, u_ref) - f_u(x, u_ref) @ u_ref
        # This ensures: f(x, u) ≈ g(x) + h(x) @ u
        f_at_ref = self.f(x, u_ref)
        h_x = self.h(x, u_ref)
        
        return f_at_ref - h_x @ u_ref
    
    def h(self, x, u_ref=None):
        """
        Affine Dynamics, Control Effectiveness Matrix
        x' = f(x,u) ≈ g(x) + h(x) u
        
        Returns the input Jacobian h(x) = df/du evaluated at reference input u_ref.
        
        Args:
            x: State vector
            u_ref: Reference input for Jacobian evaluation (default: zeros, or trim if available)
        
        Returns:
            Control effectiveness matrix h(x)
        """
        if u_ref is None:
            # Try to use trim input if available, otherwise zeros
            u_ref = getattr(self, 'u_trim', np.zeros((self.num_of_inputs,)))
        
        return self.f_u(x, u_ref)
    
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

    def f_u(self, x, u=None):
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

    def f_u(self, x, u=None):
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
    Quadcopter Dynamics Model (12-DOF) ----
    
    Mathematical Model:
    The quadcopter is modeled as a rigid body with 6 degrees of freedom (position + orientation)
    and their corresponding velocities, resulting in a 12-state system.
    
    State Vector:
        x = [x, y, z, ψ, θ, φ, x', y', z', ψ', θ', φ']
        
        Position states:    x, y, z         (inertial frame positions)
        Orientation states: ψ, θ, φ         (yaw, pitch, roll angles)
        Linear velocities:  x', y', z'      (inertial frame velocities)
        Angular velocities: ψ', θ', φ'      (body frame angular rates)
    
    Input Vector:
        u = [T, M_ψ, M_θ, M_φ]
        
        T:   Total thrust (upward force in body z-direction)
        M_ψ: Yaw moment
        M_θ: Pitch moment  
        M_φ: Roll moment
    
    Dynamics Equations:
    
    Position kinematics:
        x' = ẋ
        y' = ẏ  
        z' = ż
    
    Orientation kinematics:
        ψ' = ψ̇
        θ' = θ̇
        φ' = φ̇
    
    Translational dynamics (Newton's second law):
        mẍ = T * (sin(φ)sin(ψ) + cos(φ)cos(ψ)sin(θ))
        mÿ = T * (cos(φ)sin(θ)sin(ψ) - cos(ψ)sin(φ))
        mz̈ = T * cos(θ)cos(φ) - mg
    
    Rotational dynamics (with damping):
        ψ̈ = M_ψ - d*ψ̇
        θ̈ = M_θ - d*θ̇
        φ̈ = M_φ - d*φ̇
    
    Where:
        m: mass of quadcopter
        g: gravitational acceleration (9.81 m/s²)
        d: damping coefficient for angular rates
    
    Motor Mixing:
    The inputs u are related to individual motor thrusts [m1, m2, m3, m4] via:
        [T]   [1   1   1   1 ] [m1]
        [M_ψ] [1  -1   1  -1 ] [m2]
        [M_θ] [1   1  -1  -1 ] [m3]
        [M_φ] [1  -1  -1   1 ] [m4]
    
    Note: The ergodic state for path planning is xv = [x, y] (first two elements).
    '''
    def __init__(self, dt=0.001, x0=None, mass=0.1, damping=0.0, Q=None, R=None, z_target=1.0, motor_limits=None, zero_out_states=None):
        super().__init__(dt=dt, x0=x0, num_of_states=12, num_of_inputs=4,
                         state_names=["x", "y", "z", "ψ", "θ", "φ", "x'", "y'", "z'", "ψ'", "θ'", "φ'"])
        self.type = "Quadcopter"
        self.m = float(mass)
        self.damping = float(damping)
        self.z_target = z_target

        # base A: kinematic x_dot = v (positions depend on velocities)
        A_base = np.zeros((self.num_of_states, self.num_of_states))
        A_base[0, 6] = 1.0
        A_base[1, 7] = 1.0
        A_base[2, 8] = 1.0
        A_base[3, 9] = 1.0
        A_base[4, 10] = 1.0
        A_base[5, 11] = 1.0
        self._A_base = A_base  # keep base matrix

        # mixing matrix M: maps motor thrusts [m1,m2,m3,m4] -> inputs [thrust,yaw,pitch,roll]
        # rows: [T, Yaw, Pitch, Roll]
        self.M = np.array([
            [ 1.,  1.,  1.,  1.],
            [ 1., -1.,  1., -1.],
            [ 1.,  1., -1., -1.],
            [ 1., -1., -1.,  1.]
        ], dtype=float)
        # inverse (for motor commands from u)
        self.M_inv = 0.25 * np.array([
            [ 1.,  1.,  1.,  1.],
            [ 1., -1.,  1., -1.],
            [ 1.,  1., -1., -1.],
            [ 1., -1., -1.,  1.]
        ], dtype=float)

        # set motor limits and deduce input limits robustly
        self.input_limits, self.motor_limits = self._compute_input_limits_from_motor_limits(motor_limits)

        # LQR setup
        if zero_out_states is not None:
            assert isinstance(zero_out_states, list)
            assert all(s in self.state_names for s in zero_out_states)
        self.zero_out_states = zero_out_states

        self.Q = np.asarray(Q) if Q is not None else np.diag([0.01, 0.01, 100, 0.01, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1])
        self.R = np.asarray(R) if R is not None else np.diag([1.0, 1.0, 1.0, 1.0])
        # compute LQR gain using local linearization (no side-effects)
        self.k_lqr = self._calculateLqrControlGain(self.Q, self.R)

        # default targets
        self.state_target = np.array([0, 0, z_target, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        self._state_target = self.state_target.copy()
        self._state_target_history_for_plotting = self.state_target.copy()
        self.state_target_modified = False

    def _compute_input_limits_from_motor_limits(self, motor_limits):
        # default infinite motor limits
        if motor_limits is None:
            motor_limits = np.vstack([[-np.inf, np.inf]]*4)
        else:
            motor_limits = np.asarray(motor_limits, dtype=float)
            assert motor_limits.shape == (4, 2)
            assert np.all(motor_limits[:, 0] < motor_limits[:, 1])
        m_min = motor_limits[:, 0]
        m_max = motor_limits[:, 1]
        # For each input i, the min/max is found by taking for each motor j:
        # contribution = M[i,j] * (m_min[j] if M[i,j]>=0 else m_max[j])  (min)
        u_min = np.zeros(4)
        u_max = np.zeros(4)
        for i in range(4):
            mins = []
            maxs = []
            for j in range(4):
                a = self.M[i, j]
                if np.isposinf(m_max[j]) or np.isneginf(m_min[j]):
                    # if any bound infinite produce -/+inf correctly
                    if a > 0:
                        min_contrib = a * m_min[j]
                        max_contrib = a * m_max[j]
                    else:
                        min_contrib = a * m_max[j]
                        max_contrib = a * m_min[j]
                else:
                    if a >= 0:
                        min_contrib = a * m_min[j]
                        max_contrib = a * m_max[j]
                    else:
                        min_contrib = a * m_max[j]
                        max_contrib = a * m_min[j]
                mins.append(min_contrib)
                maxs.append(max_contrib)
            u_min[i] = np.sum(mins)
            u_max[i] = np.sum(maxs)
        u_limits = np.vstack([u_min, u_max]).T  # shape (4,2)
        return u_limits, motor_limits

    def f(self, x, u):
        # clip inputs to safe bounds
        u = np.clip(np.asarray(u, dtype=float), self.input_limits[:, 0], self.input_limits[:, 1])

        psi = x[3]; theta = x[4]; phi = x[5]

        # translational accelerations from total thrust (body z direction)
        xddot = u[0] * (np.sin(phi) * np.sin(psi) + np.cos(phi) * np.cos(psi) * np.sin(theta)) / self.m
        yddot = u[0] * (np.cos(phi) * np.sin(theta) * np.sin(psi) - np.cos(psi) * np.sin(phi)) / self.m
        zddot = u[0] * np.cos(theta) * np.cos(phi) / self.m - 9.81

        psiddot = u[1] - self.damping * x[9]
        thetaddot = u[2] - self.damping * x[10]
        phiddot = u[3] - self.damping * x[11]

        return np.array([
            x[6], x[7], x[8],      # pos derivatives
            x[9], x[10], x[11],    # angle derivatives
            xddot, yddot, zddot,   # linear accel
            psiddot, thetaddot, phiddot
        ], dtype=float)

    def f_x(self, x, u):
        """
        Return a fresh Jacobian A(x,u) (do NOT mutate internal matrices).
        """
        u = np.clip(np.asarray(u, dtype=float), self.input_limits[:, 0], self.input_limits[:, 1])
        psi = x[3]; theta = x[4]; phi = x[5]
        A = self._A_base.copy()  # start from base kinematic structure

        # fill partial derivatives for acceleration rows (indices 6..8)
        # derivatives of xddot, yddot, zddot w.r.t psi,theta,phi
        # note: these match the analytic terms in your original code
        A[6, 3] = u[0] * (np.cos(psi) * np.sin(phi) - np.cos(phi) * np.sin(theta) * np.sin(psi)) / self.m
        A[6, 4] = u[0] * np.cos(theta) * np.cos(phi) * np.cos(psi) / self.m
        A[6, 5] = u[0] * (-np.cos(psi) * np.sin(theta) * np.sin(phi) + np.cos(phi) * np.sin(psi)) / self.m

        A[7, 3] = u[0] * (np.cos(phi) * np.cos(psi) * np.sin(theta) + np.sin(phi) * np.sin(psi)) / self.m
        A[7, 4] = u[0] * np.cos(theta) * np.cos(phi) * np.sin(psi) / self.m
        A[7, 5] = u[0] * (-np.cos(phi) * np.cos(psi) - np.sin(theta) * np.sin(phi) * np.sin(psi)) / self.m

        A[8, 4] = -u[0] * np.cos(phi) * np.sin(theta) / self.m
        A[8, 5] = -u[0] * np.cos(theta) * np.sin(phi) / self.m

        # angular damping on rates (9..11)
        A[9, 9] = -self.damping
        A[10, 10] = -self.damping
        A[11, 11] = -self.damping

        return A

    def f_u(self, x, u=None):
        """
        Return fresh B(x) matrix (mapping u -> state derivatives).
        """
        psi = x[3]; theta = x[4]; phi = x[5]
        B = np.zeros((self.num_of_states, self.num_of_inputs), dtype=float)
        B[6, 0] = (np.cos(phi) * np.cos(psi) * np.sin(theta) + np.sin(phi) * np.sin(psi)) / self.m
        B[7, 0] = (-np.cos(psi) * np.sin(phi) + np.cos(phi) * np.sin(theta) * np.sin(psi)) / self.m
        B[8, 0] = np.cos(theta) * np.cos(phi) / self.m
        B[9, 1] = 1.0
        B[10, 2] = 1.0
        B[11, 3] = 1.0
        return B

    def h(self, x):
        return self.f_u(x)

    def rk4Step(self, f, x, dt, *f_args):
        """
        Classic 4th-order Runge-Kutta step.
        - f : function f(x, *f_args) -> xdot
        - x : current state (np.array)
        - dt: timestep (float)
        - f_args: additional args forwarded to f (e.g. u)
        Returns: x_next (np.array)
        """
        k1 = f(x, *f_args)
        k2 = f(x + 0.5 * dt * k1, *f_args)
        k3 = f(x + 0.5 * dt * k2, *f_args)
        k4 = f(x + dt * k3, *f_args)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
 
 
    def step(self, x, u, dt=None):
        dt = self.dt if dt is None else dt
        u = np.clip(np.asarray(u, dtype=float), self.input_limits[:, 0], self.input_limits[:, 1])
        return self.rk4Step(self.f, x, dt, *(u,))


    def _calculateLqrControlGain(self, Q, R):
        # compute linearization around current state and nominal hover u_nom
        u_nom = np.zeros((self.num_of_inputs,))
        u_nom[0] = self.m * 9.81  # hover thrust
        A_lin = self.f_x(self.state, u_nom)
        B_lin = self.f_u(self.state)
        P = solve_continuous_are(A_lin, B_lin, Q, R)
        K = np.linalg.inv(R) @ B_lin.T @ P
        # zero out undesired columns if requested
        if self.zero_out_states is not None:
            idxs = [self.state_names.index(s) for s in self.zero_out_states if s in self.state_names]
            if len(idxs):
                K[:, idxs] = 0.0
        return K

    def calcLQRcontrol(self, x, t, state_target=None):
        state_target = self._state_target.copy() if state_target is None else state_target
        u = -self.k_lqr @ (x - state_target)
        u[0] += self.m * 9.81
        u = np.clip(u, self.input_limits[:, 0], self.input_limits[:, 1])
        return u

    # convenient helpers for motor/input conversions using M, M_inv
    def convertInputToMotorCommands(self, u):
        u = np.asarray(u, dtype=float)
        motors = self.M_inv @ u
        # enforce motor limits
        motors = np.clip(motors, self.motor_limits[:, 0], self.motor_limits[:, 1])
        return motors

    def convertMotorCommandsToInput(self, motors):
        motors = np.asarray(motors, dtype=float)
        return self.M @ motors

    @property
    def state_string(self):
         return f"x: {self.state[0]:.2f}, y: {self.state[1]:.2f}, z: {self.state[2]:.2f}, | ψ: {self.state[3]*180/np.pi:.2f}, θ: {self.state[4]*180/np.pi:.2f}, φ: {self.state[5]*180/np.pi:.2f}"
    

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

    def f_u(self, x, u=None):
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

    def f_u(self, x, u=None):
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


from scipy.optimize import root
class FixedWing12DOFTrainer(DynamicsBase):
    """
    12-DOF rigid-body trainer airplane (starter parameters for ~1.5 m RC trainer).
    State ordering:
      x = [X, Y, Z, phi, theta, psi, u, v, w, p, q, r]
    Inputs:
      u = [delta_e, delta_a, delta_r, throttle]
    Uses rk4Step for integration (same style as Quadcopter).
    """
    def __init__(self, dt=0.001, x0=None, params=None, v_trim=10, use_linear_f=False, use_linear_fx_fu=False):
        # 12 states, 4 inputs
        super().__init__(dt=dt, x0=x0, num_of_states=12, num_of_inputs=4,
                         state_names=["X", "Y", "Z", "phi", "theta", "psi",
                                      "u", "v", "w", "p", "q", "r"])
        self.type = "FixedWing12DOFTrainer"

        # default parameter set (good starting point for 1.5 m trainer)
        default_params = {
            'm': 2,          # kg (def: 1.6)
            'S': 0.36,       # m^2
            'b': 1.50,       # m (span)
            'c': 0.2407,     # m (mean aerodynamic chord ~ S/b)
            'rho': 1.225,
            'Ix': 0.072,     # kg m^2 (estimate; replace with CAD)
            'Iy': 0.0255,
            'Iz': 0.0963,
            'Ixz': -6.12e-4,
            # longitudinal coefficients (linearized starter)
            'CL0': 0.20,
            'CL_alpha': 4.756,   # per rad
            'CL_q': 3.0,
            'CL_de': -1.0,       # elevator effectiveness (neg. if down elevator -> negative Cm_de)
            'CD0': 0.025,
            'k': 0.0639,         # induced drag factor
            'Cm_alpha': -0.8,
            'Cm_q': -8.0,
            'Cm_de': -1.2,
            # lateral-directional
            'C_ell_beta': -0.12,
            'C_ell_p': -0.26,
            'C_ell_r': 0.14,
            'Cn_beta': 0.25,
            'Cn_p': -0.022,
            'Cn_r': -0.35,
            'CY_beta': -0.02,
            # control-surface derivatives (starter guesses)
            'C_ell_da': 0.10,   # roll per aileron rad
            'C_ell_dr': 0.03,   # roll per rudder rad (def: 0.01)
            'Cn_da': -0.03,     # yaw per aileron rad (adverse yaw)
            'Cn_dr': -0.10,     # yaw per rudder rad (If we dont reverse Usafe, this should be positive)
            'CY_da': 0.0,       # side force per aileron rad
            'CY_dr': 0.12,      # side force per rudder rad
            'CY_p': 0.0,
            'CY_r': 0.0,
            # propulsion
            'T_max': 10.0,   # N (example static thrust)
            # input limits: [min, max] for [de, da, dr, throttle]
            'input_limits': np.array([[-0.4363, 0.4363],  # elevator ±25 deg
                                    [-0.4363, 0.4363],  # aileron ±25 deg
                                    [-0.4363, 0.4363],  # rudder ±25 deg
                                    [0.0, 1.0]]),       # throttle 0..1
            'V_trim': v_trim  # Desired trim speed [m/s]
        }

        self.params = default_params if params is None else {**default_params, **params}
        self.input_limits = self.params['input_limits'].copy()

        # Lets try and trim the plane at desired speed
        # Start with non linear before we linearise
        self.use_linear_model_for_f = False
        self.use_linear_model_for_fx_fu = False
        x_trim, u_trim, sol = self.computeTrim(V_trim=self.params['V_trim'])
        if not sol.success:
            print("ATTENTION: Trimming failed:", sol.message)
            input("Waiting for confirmation to continue... [Enter]")
        self.x_trim = x_trim
        self.u_trim = u_trim
        self.trim_sol_flags = sol

        # Reset part of state to trim point
        x_new = self.x_trim.copy()
        x_new[0] = self.state[0]  # X
        x_new[1] = self.state[1]  # Y
        x_new[2] = self.state[2]  # Z
        x_new[5] = self.state[5]  # psi
        self.reset(x_new)

        # Print initial position
        print(f"State \n{self.state_string}")

        # Linearise flight dynamics
        # x_dot = A x + B u
        self.A, self.B = self.linearizeAtTrimPoint(self.x_trim, self.u_trim)
        self.use_linear_model_for_f = use_linear_f
        self.use_linear_model_for_fx_fu = use_linear_fx_fu

    def _extract_params(self, params):
        """Extract parameters from dictionary for Jacobian computation."""
        m = params['m']
        S = params['S']
        b = params['b']
        c = params['c']
        rho = params['rho']
        Ix = params['Ix']
        Iy = params['Iy']
        Iz = params['Iz']
        Ixz = params['Ixz']
        CL0 = params['CL0']
        CL_alpha = params['CL_alpha']
        CL_q = params['CL_q']
        CL_de = params['CL_de']
        CD0 = params['CD0']
        k_drag = params.get('k_drag', params.get('k', 0.0639))
        Cm_alpha = params['Cm_alpha']
        Cm_q = params['Cm_q']
        Cm_de = params['Cm_de']
        CY_beta = params['CY_beta']
        CY_p = params['CY_p']
        CY_r = params['CY_r']
        CY_da = params['CY_da']
        CY_dr = params['CY_dr']
        C_ell_beta = params['C_ell_beta']
        C_ell_p = params['C_ell_p']
        C_ell_r = params['C_ell_r']
        C_ell_da = params['C_ell_da']
        C_ell_dr = params['C_ell_dr']
        Cn_beta = params['Cn_beta']
        Cn_p = params['Cn_p']
        Cn_r = params['Cn_r']
        Cn_da = params['Cn_da']
        Cn_dr = params['Cn_dr']
        T_max = params['T_max']
        g = 9.81

        return m, S, b, c, rho, Ix, Iy, Iz, Ixz, CL0, CL_alpha, CL_q, CL_de, CD0, k_drag, Cm_alpha, Cm_q, Cm_de, CY_beta, CY_p, CY_r, CY_da, CY_dr, C_ell_beta, C_ell_p, C_ell_r, C_ell_da, C_ell_dr, Cn_beta, Cn_p, Cn_r, Cn_da, Cn_dr, T_max, g

    def f(self, x, u):
        # u is an array [de, da, dr, throttle]
        u = np.asarray(u)
        # clip inputs
        u = np.clip(u, self.input_limits[:, 0], self.input_limits[:, 1])
        delta_e, delta_a, delta_r, throttle = u

        # If using linear model, just do that
        if self.use_linear_model_for_f:
            x_dot = self.A @ (x - self.x_trim) + self.B @ (u - self.u_trim)
            return x_dot

        # unpack state (match ordering)
        X, Y, Z, phi, theta, psi, ub, vb, wb, p, q, r = x
        p = float(p); q = float(q); r = float(r)  # ensure scalars

        # params (direct indexing)
        P = self.params
        m = P['m']; S = P['S']; b = P['b']; c = P['c']; rho = P['rho']
        Ix = P['Ix']; Iy = P['Iy']; Iz = P['Iz']; Ixz = P['Ixz']

        # airspeed and angles (protect small V)
        V = np.sqrt(max(ub*ub + vb*vb + wb*wb, 1e-6))
        V_safe = max(V, 1e-3)
        alpha = np.arctan2(wb, ub)   # angle of attack
        beta = np.arcsin(np.clip(vb / V_safe, -0.99, 0.99))

        qbar = 0.5 * rho * V**2

        # Longitudinal coefficients (linearized)
        CL = P['CL0'] + P['CL_alpha'] * alpha + P['CL_q'] * (c * q / (2.0 * V_safe)) + P['CL_de'] * delta_e
        CD = P['CD0'] + P['k'] * CL**2

        # side force: include rudder/aileron and small rate terms
        CY = (P['CY_beta'] * beta
            + P['CY_p'] * (b * p / (2.0 * V_safe))
            + P['CY_r'] * (b * r / (2.0 * V_safe))
            + P['CY_da'] * delta_a
            + P['CY_dr'] * delta_r)

        # aerodynamic forces (wind axes)
        L = qbar * S * CL
        D = qbar * S * CD
        Y_force = qbar * S * CY

        # transform wind-axis forces to body axes (rotate by alpha)
        X_aero = -D * np.cos(alpha) - L * np.sin(alpha)
        Z_aero = -D * np.sin(alpha) - L * np.cos(alpha)
        Y_aero = Y_force

        # Moments coefficients (include control-surface derivatives)
        Cl = (P['C_ell_beta'] * beta
            + P['C_ell_p'] * (b * p / (2.0 * V_safe))
            + P['C_ell_r'] * (b * r / (2.0 * V_safe))
            + P['C_ell_da'] * delta_a
            + P['C_ell_dr'] * delta_r)

        Cm = (P['Cm_alpha'] * alpha
            + P['Cm_q'] * (c * q / (2.0 * V_safe))
            + P['Cm_de'] * delta_e)

        Cn = (P['Cn_beta'] * beta
            + P['Cn_p'] * (b * p / (2.0 * V_safe))
            + P['Cn_r'] * (b * r / (2.0 * V_safe))
            + P['Cn_da'] * delta_a
            + P['Cn_dr'] * delta_r)

        L_aero = qbar * S * b * Cl
        M_aero = qbar * S * c * Cm
        N_aero = qbar * S * b * Cn

        # Propulsion: thrust along body x
        T = P['T_max'] * np.clip(throttle, 0.0, 1.0)
        X_prop = T; Y_prop = 0.0; Z_prop = 0.0

        # Gravity in body axes
        g = 9.81
        X_grav = -m * g * np.sin(theta)
        Y_grav = m * g * np.cos(theta) * np.sin(phi)
        Z_grav = m * g * np.cos(theta) * np.cos(phi)

        # Total forces
        X_tot = X_aero + X_prop + X_grav
        Y_tot = Y_aero + Y_prop + Y_grav
        Z_tot = Z_aero + Z_prop + Z_grav

        # Translational accelerations (body axes)
        udot = (X_tot / m) - q * wb + r * vb
        vdot = (Y_tot / m) - r * ub + p * wb
        wdot = (Z_tot / m) - p * vb + q * ub

        # rotational EoM: I * omega_dot + omega x I omega = M
        I = np.array([[Ix, 0.0, -Ixz],
                    [0.0, Iy, 0.0],
                    [-Ixz, 0.0, Iz]])
        omega = np.array([p, q, r])
        M_vec = np.array([L_aero, M_aero, N_aero])
        # solve for omega_dot
        omega_dot = np.linalg.solve(I, M_vec - np.cross(omega, I.dot(omega)))
        pdot, qdot, rdot = omega_dot

        # Euler kinematics (body rates -> euler angle rates)
        cphi = np.cos(phi); sphi = np.sin(phi)
        cth = np.cos(theta); sth = np.sin(theta)
        if abs(cth) < 1e-6:
            cth = 1e-6

        E = np.array([[1.0, sphi * np.tan(theta), cphi * np.tan(theta)],
                    [0.0, cphi, -sphi],
                    [0.0, sphi / cth, cphi / cth]])
        phi_dot, theta_dot, psi_dot = E.dot(omega)

        # inertial position derivative (body->inertial)
        cpsi = np.cos(psi); spsi = np.sin(psi)
        R = np.array([
            [cpsi * cth, cpsi * sth * sphi - spsi * cphi, cpsi * sth * cphi + spsi * sphi],
            [spsi * cth, spsi * sth * sphi + cpsi * cphi, spsi * sth * cphi - cpsi * sphi],
            [-sth,       cth * sphi,                     cth * cphi]
        ])
        pos_dot = R.dot(np.array([ub, vb, wb]))

        # assemble xdot in same ordering as state
        xdot = np.zeros(12)
        xdot[0:3] = pos_dot           # Xdot, Ydot, Zdot
        xdot[3:6] = [phi_dot, theta_dot, psi_dot]
        xdot[6:9] = [udot, vdot, wdot]
        xdot[9:12] = [pdot, qdot, rdot]

        return xdot

    # Finite Differences Jacobians (We use sympy analytical ones for now)
    def f_x_fd(self, x, u, eps=1e-6):
        # Use analytical Jacobian at trim point if we play with linear model
        if self.use_linear_model_for_fx_fu:
            return self.A.copy()

        # numerical Jacobian wrt state (finite differences)
        fx = np.zeros((self.num_of_states, self.num_of_states))
        f0 = self.f(x, u)
        for i in range(self.num_of_states):
            xp = x.copy()
            xp[i] += eps
            fp = self.f(xp, u)
            fx[:, i] = (fp - f0) / eps
        return fx

    # Finite Differences Jacobians (We use sympy analytical ones for now)
    def f_u_fd(self, x, eps=1e-6):
        # Use analytical Jacobian at trim point if we play with linear model
        if self.use_linear_model_for_fx_fu:
            return self.B.copy()
        
        # numerical Jacobian wrt inputs (finite differences)
        fu = np.zeros((self.num_of_states, self.num_of_inputs))
        u0 = np.zeros((self.num_of_inputs,))
        # to get a meaningful linearization we use mid inputs (neutral)
        # but here we simply compute around zero-throttle/zero-control baseline
        for j in range(self.num_of_inputs):
            up = u0.copy()
            up[j] += eps
            fp = self.f(x, up)
            f0 = self.f(x, u0)
            fu[:, j] = (fp - f0) / eps
        return fu
   
    def f_x_analytical(self, x, u, params):
        """
        Analytical Jacobians for FixedWing12DOFTrainer
        Generated by symbolic differentiation using SymPy

        This file contains hardcoded analytical expressions for:
        - f_x(x, u, params): State Jacobian (12x12)
        - f_u(x, u, params): Input Jacobian (12x4)

        These functions provide EXACT derivatives without numerical errors
        and are faster than finite differences or JAX for single evaluations.

        Date: 2025-11-01
        """

        """
        Analytical state Jacobian (df/dx) for fixed-wing aircraft.
        
        Args:
            x: State vector [X, Y, Z, phi, theta, psi, u, v, w, p, q, r]
            u: Input vector [delta_e, delta_a, delta_r, throttle]
            params: Dictionary of aircraft parameters
        
        Returns:
            A: State Jacobian matrix (12x12)
        """
        # Extract state variables
        X = x[0]
        Y = x[1]
        Z = x[2]
        phi = x[3]
        theta = x[4]
        psi = x[5]
        u_b = x[6]  # body velocity u (renamed to avoid conflict with input u)
        v_b = x[7]  # body velocity v
        w_b = x[8]  # body velocity w
        p = x[9]
        q = x[10]
        r = x[11]

        # Extract input variables
        delta_e = u[0]
        delta_a = u[1]
        delta_r = u[2]
        throttle = u[3]

        # Extract parameters
        m, S, b, c, rho, Ix, Iy, Iz, Ixz, CL0, CL_alpha, CL_q, CL_de, CD0, k_drag, Cm_alpha, Cm_q, Cm_de, CY_beta, CY_p, CY_r, CY_da, CY_dr, C_ell_beta, C_ell_p, C_ell_r, C_ell_da, C_ell_dr, Cn_beta, Cn_p, Cn_r, Cn_da, Cn_dr, T_max, g = self._extract_params(params)

        # Initialize Jacobian matrix
        A = np.zeros((12, 12))

        # Compute Jacobian elements (generated by SymPy)
        # Note: Only non-zero elements are computed for efficiency

        # Precompute common terms (speeds up calculations quite a lot, who would have thought hahaha)
        uvw_sq = u_b**2 + v_b**2 + w_b**2
        sq_uvw = np.sqrt(uvw_sq)
        sin_phi = np.sin(phi); cos_phi = np.cos(phi)
        sin_psi = np.sin(psi); cos_psi = np.cos(psi)
        sin_theta = np.sin(theta); cos_theta = np.cos(theta); tan_theta = np.tan(theta)

        A[0, 3] = v_b*(sin_phi*sin_psi + sin_theta*cos_phi*cos_psi) - w_b*(sin_phi*sin_theta*cos_psi - sin_psi*cos_phi)
        A[0, 4] = (-u_b*sin_theta + v_b*sin_phi*cos_theta + w_b*cos_phi*cos_theta)*cos_psi
        A[0, 5] = -u_b*sin_psi*cos_theta - v_b*(sin_phi*sin_psi*sin_theta + cos_phi*cos_psi) + w_b*(sin_phi*cos_psi - sin_psi*sin_theta*cos_phi)
        A[0, 6] = cos_psi*cos_theta
        A[0, 7] = sin_phi*sin_theta*cos_psi - sin_psi*cos_phi
        A[0, 8] = sin_phi*sin_psi + sin_theta*cos_phi*cos_psi
        A[1, 3] = -v_b*(sin_phi*cos_psi - sin_psi*sin_theta*cos_phi) - w_b*(sin_phi*sin_psi*sin_theta + cos_phi*cos_psi)
        A[1, 4] = (-u_b*sin_theta + v_b*sin_phi*cos_theta + w_b*cos_phi*cos_theta)*sin_psi
        A[1, 5] = u_b*cos_psi*cos_theta + v_b*(sin_phi*sin_theta*cos_psi - sin_psi*cos_phi) + w_b*(sin_phi*sin_psi + sin_theta*cos_phi*cos_psi)
        A[1, 6] = sin_psi*cos_theta
        A[1, 7] = sin_phi*sin_psi*sin_theta + cos_phi*cos_psi
        A[1, 8] = -sin_phi*cos_psi + sin_psi*sin_theta*cos_phi
        A[2, 3] = (v_b*cos_phi - w_b*sin_phi)*cos_theta
        A[2, 4] = -u_b*cos_theta - v_b*sin_phi*sin_theta - w_b*sin_theta*cos_phi
        A[2, 6] = -sin_theta
        A[2, 7] = sin_phi*cos_theta
        A[2, 8] = cos_phi*cos_theta
        A[3, 3] = (q*cos_phi - r*sin_phi)*tan_theta
        A[3, 4] = (q*sin_phi + r*cos_phi)/cos_theta**2
        A[3, 9] = 1
        A[3, 10] = sin_phi*tan_theta
        A[3, 11] = cos_phi*tan_theta
        A[4, 3] = -q*sin_phi - r*cos_phi
        A[4, 10] = cos_phi
        A[4, 11] = -sin_phi
        A[5, 3] = (q*cos_phi - r*sin_phi)/cos_theta
        A[5, 4] = (q*sin_phi + r*cos_phi)*sin_theta/cos_theta**2
        A[5, 10] = sin_phi/cos_theta
        A[5, 11] = cos_phi/cos_theta
        A[6, 4] = -g*cos_theta
        A[6, 6] = S*rho*(2*k_drag*(u_b**2 + w_b**2)**(7/2)*(2*CL_alpha*w_b*(uvw_sq)**(3/2) + CL_q*c*q*u_b**3)*(uvw_sq)**2*(2*CL_alpha*w_b*sq_uvw + CL_q*c*q*u_b + 2*u_b*(CL0 + CL_de*delta_e)*sq_uvw) - 4*u_b**2*w_b*(u_b**2 + w_b**2)**(7/2)*(uvw_sq)**(5/2)*(2*CL_alpha*w_b*sq_uvw + CL_q*c*q*u_b + 2*u_b*(CL0 + CL_de*delta_e)*sq_uvw) - 2*u_b**2*(u_b**2 + w_b**2)**(7/2)*(4*CD0*u_b**2*(uvw_sq) + k_drag*(2*CL_alpha*w_b*sq_uvw + CL_q*c*q*u_b + 2*u_b*(CL0 + CL_de*delta_e)*sq_uvw)**2)*(uvw_sq)**2 - w_b**2*(u_b**2 + w_b**2)**(5/2)*(uvw_sq)**3*(4*CD0*u_b**2*(uvw_sq) + k_drag*(2*CL_alpha*w_b*sq_uvw + CL_q*c*q*u_b + 2*u_b*(CL0 + CL_de*delta_e)*sq_uvw)**2 + 2*w_b*sq_uvw*(2*CL_alpha*w_b*sq_uvw + CL_q*c*q*u_b + 2*u_b*(CL0 + CL_de*delta_e)*sq_uvw)) + 2*w_b*(u_b**2 + w_b**2)**(7/2)*(2*CL_alpha*w_b*(uvw_sq)**(3/2) + CL_q*c*q*u_b**3)*(uvw_sq)**(5/2) + 2*w_b*(2*CL_alpha*w_b*sq_uvw + CL_q*c*q*u_b + 2*u_b*(CL0 + CL_de*delta_e)*sq_uvw)*(u_b**4 + u_b**2*v_b**2 + 2*u_b**2*w_b**2 + v_b**2*w_b**2 + w_b**4)**(7/2))*np.abs(u_b)/(8*m*u_b**3*(u_b**2 + w_b**2)**4*(uvw_sq)**3)
        A[6, 7] = (CL_q*S*c*k_drag*q*rho*u_b*v_b*(u_b**2 + w_b**2)**(3/2)*(uvw_sq)**2*(2*CL_alpha*w_b*sq_uvw + CL_q*c*q*u_b + 2*u_b*(CL0 + CL_de*delta_e)*sq_uvw)*np.abs(u_b) + CL_q*S*c*q*rho*u_b*v_b*w_b*(u_b**2 + w_b**2)**(3/2)*(uvw_sq)**(5/2)*np.abs(u_b) - 2*S*rho*v_b*w_b*(u_b**2 + w_b**2)**(3/2)*(uvw_sq)**(5/2)*(2*CL_alpha*w_b*sq_uvw + CL_q*c*q*u_b + 2*u_b*(CL0 + CL_de*delta_e)*sq_uvw)*np.abs(u_b) - S*rho*v_b*(u_b**2 + w_b**2)**(3/2)*(4*CD0*u_b**2*(uvw_sq) + k_drag*(2*CL_alpha*w_b*sq_uvw + CL_q*c*q*u_b + 2*u_b*(CL0 + CL_de*delta_e)*sq_uvw)**2)*(uvw_sq)**2*np.abs(u_b) + 4*m*r*u_b**2*(u_b**2 + w_b**2)**2*(uvw_sq)**3)/(4*m*u_b**2*(u_b**2 + w_b**2)**2*(uvw_sq)**3)
        A[6, 8] = (-S*rho*w_b*(u_b**2 + w_b**2)**(5/2)*(uvw_sq)*(2*CL_alpha*(uvw_sq)**(3/2) - CL_q*c*q*u_b*w_b + 2*w_b*(2*CL_alpha*w_b*sq_uvw + CL_q*c*q*u_b + 2*u_b*(CL0 + CL_de*delta_e)*sq_uvw))*np.abs(u_b)/4 + S*rho*w_b*(4*CD0*u_b**2*(uvw_sq) + k_drag*(2*CL_alpha*w_b*sq_uvw + CL_q*c*q*u_b + 2*u_b*(CL0 + CL_de*delta_e)*sq_uvw)**2 + 2*w_b*sq_uvw*(2*CL_alpha*w_b*sq_uvw + CL_q*c*q*u_b + 2*u_b*(CL0 + CL_de*delta_e)*sq_uvw))*(u_b**4 + u_b**2*v_b**2 + 2*u_b**2*w_b**2 + v_b**2*w_b**2 + w_b**4)**(3/2)*np.abs(u_b)/8 - S*rho*(u_b**2 + w_b**2)**(5/2)*(k_drag*(2*CL_alpha*(uvw_sq)**(3/2) - CL_q*c*q*u_b*w_b)*(2*CL_alpha*w_b*sq_uvw + CL_q*c*q*u_b + 2*u_b*(CL0 + CL_de*delta_e)*sq_uvw) + w_b*(4*CD0*u_b**2*(uvw_sq) + k_drag*(2*CL_alpha*w_b*sq_uvw + CL_q*c*q*u_b + 2*u_b*(CL0 + CL_de*delta_e)*sq_uvw)**2))*sq_uvw*np.abs(u_b)/4 - S*rho*(u_b**2 + w_b**2)**(5/2)*(uvw_sq)**2*(2*CL_alpha*w_b*sq_uvw + CL_q*c*q*u_b + 2*u_b*(CL0 + CL_de*delta_e)*sq_uvw)*np.abs(u_b)/4 - m*q*u_b**2*(u_b**2 + w_b**2)**3*(uvw_sq)**(3/2))/(m*u_b**2*(u_b**2 + w_b**2)**3*(uvw_sq)**(3/2))
        A[6, 10] = (-CL_q*S*c*k_drag*rho*(2*CL_alpha*w_b*sq_uvw + CL_q*c*q*u_b + 2*u_b*(CL0 + CL_de*delta_e)*sq_uvw)*np.abs(u_b) - CL_q*S*c*rho*w_b*sq_uvw*np.abs(u_b) - 4*m*u_b*w_b*np.sqrt(u_b**2 + w_b**2))/(4*m*u_b*np.sqrt(u_b**2 + w_b**2))
        A[6, 11] = v_b
        A[7, 3] = g*cos_phi*cos_theta
        A[7, 4] = -g*sin_phi*sin_theta
        A[7, 6] = (-S*rho*u_b*(2*CY_beta*v_b + CY_p*b*p + CY_r*b*r) + 2*S*rho*u_b*(2*CY_beta*v_b + CY_p*b*p + CY_r*b*r + 2*(CY_da*delta_a + CY_dr*delta_r)*sq_uvw) - 4*m*r*sq_uvw)/(4*m*sq_uvw)
        A[7, 7] = S*rho*(2*CY_beta*(uvw_sq) - v_b*(2*CY_beta*v_b + CY_p*b*p + CY_r*b*r) + 2*v_b*(2*CY_beta*v_b + CY_p*b*p + CY_r*b*r + 2*(CY_da*delta_a + CY_dr*delta_r)*sq_uvw))/(4*m*sq_uvw)
        A[7, 8] = (-S*rho*w_b*(2*CY_beta*v_b + CY_p*b*p + CY_r*b*r) + 2*S*rho*w_b*(2*CY_beta*v_b + CY_p*b*p + CY_r*b*r + 2*(CY_da*delta_a + CY_dr*delta_r)*sq_uvw) + 4*m*p*sq_uvw)/(4*m*sq_uvw)
        A[7, 9] = CY_p*S*b*rho*sq_uvw/(4*m) + w_b
        A[7, 11] = CY_r*S*b*rho*sq_uvw/(4*m) - u_b
        A[8, 3] = -g*sin_phi*cos_theta
        A[8, 4] = -g*sin_theta*cos_phi
        A[8, 6] = (2*S*k_drag*rho*w_b*(u_b**2 + w_b**2)**5*(2*CL_alpha*w_b*(uvw_sq)**(3/2) + CL_q*c*q*u_b**3)*(uvw_sq)**2*(2*CL_alpha*w_b*sq_uvw + CL_q*c*q*u_b + 2*u_b*(CL0 + CL_de*delta_e)*sq_uvw)*np.abs(u_b) - 4*S*rho*u_b**4*(u_b**2 + w_b**2)**5*(uvw_sq)**(5/2)*(2*CL_alpha*w_b*sq_uvw + CL_q*c*q*u_b + 2*u_b*(CL0 + CL_de*delta_e)*sq_uvw)*np.abs(u_b) - 2*S*rho*u_b**2*w_b**2*(u_b**2 + w_b**2)**4*(uvw_sq)**(7/2)*(2*CL_alpha*w_b*sq_uvw + CL_q*c*q*u_b + 2*u_b*(CL0 + CL_de*delta_e)*sq_uvw)*np.abs(u_b) - 2*S*rho*u_b**2*w_b*(u_b**2 + w_b**2)**5*(4*CD0*u_b**2*(uvw_sq) + k_drag*(2*CL_alpha*w_b*sq_uvw + CL_q*c*q*u_b + 2*u_b*(CL0 + CL_de*delta_e)*sq_uvw)**2)*(uvw_sq)**2*np.abs(u_b) + 2*S*rho*u_b**2*(u_b**2 + w_b**2)**5*(2*CL_alpha*w_b*(uvw_sq)**(3/2) + CL_q*c*q*u_b**3)*(uvw_sq)**(5/2)*np.abs(u_b) - S*rho*w_b**3*(u_b**2 + w_b**2)**4*(4*CD0*u_b**2*(uvw_sq) + k_drag*(2*CL_alpha*w_b*sq_uvw + CL_q*c*q*u_b + 2*u_b*(CL0 + CL_de*delta_e)*sq_uvw)**2)*(uvw_sq)**3*np.abs(u_b) + S*rho*w_b*(u_b**2 + w_b**2)**5*(4*CD0*u_b**2*(uvw_sq) + k_drag*(2*CL_alpha*w_b*sq_uvw + CL_q*c*q*u_b + 2*u_b*(CL0 + CL_de*delta_e)*sq_uvw)**2)*(uvw_sq)**3*np.abs(u_b) + 8*m*q*u_b**4*(u_b**2 + w_b**2)**(11/2)*(uvw_sq)**3)/(8*m*u_b**4*(u_b**2 + w_b**2)**(11/2)*(uvw_sq)**3)
        A[8, 7] = -p + (CL_q*S*c*k_drag*q*rho*v_b*w_b*(CL0 + CL_alpha*w_b/u_b + CL_de*delta_e + CL_q*c*q/(2*sq_uvw))/(2*u_b*np.sqrt(1 + w_b**2/u_b**2)*sq_uvw) + CL_q*S*c*q*rho*v_b/(4*np.sqrt(1 + w_b**2/u_b**2)*sq_uvw) - S*rho*v_b*(CL0 + CL_alpha*w_b/u_b + CL_de*delta_e + CL_q*c*q/(2*sq_uvw))/np.sqrt(1 + w_b**2/u_b**2) - S*rho*v_b*w_b*(CD0 + k_drag*(CL0 + CL_alpha*w_b/u_b + CL_de*delta_e + CL_q*c*q/(2*sq_uvw))**2)/(u_b*np.sqrt(1 + w_b**2/u_b**2)))/m
        A[8, 8] = S*rho*(2*u_b**2*w_b*(uvw_sq)**2*(2*CL_alpha*w_b*sq_uvw + CL_q*c*q*u_b + 2*u_b*(CL0 + CL_de*delta_e)*sq_uvw) - 2*u_b**2*(u_b**2 + w_b**2)*(uvw_sq)*(2*CL_alpha*(uvw_sq)**(3/2) - CL_q*c*q*u_b*w_b + 2*w_b*(2*CL_alpha*w_b*sq_uvw + CL_q*c*q*u_b + 2*u_b*(CL0 + CL_de*delta_e)*sq_uvw)) + w_b**2*(4*CD0*u_b**2*(uvw_sq) + k_drag*(2*CL_alpha*w_b*sq_uvw + CL_q*c*q*u_b + 2*u_b*(CL0 + CL_de*delta_e)*sq_uvw)**2)*(uvw_sq)**(3/2) - 2*w_b*(u_b**2 + w_b**2)*(k_drag*(2*CL_alpha*(uvw_sq)**(3/2) - CL_q*c*q*u_b*w_b)*(2*CL_alpha*w_b*sq_uvw + CL_q*c*q*u_b + 2*u_b*(CL0 + CL_de*delta_e)*sq_uvw) + w_b*(4*CD0*u_b**2*(uvw_sq) + k_drag*(2*CL_alpha*w_b*sq_uvw + CL_q*c*q*u_b + 2*u_b*(CL0 + CL_de*delta_e)*sq_uvw)**2))*sq_uvw - (u_b**2 + w_b**2)*(4*CD0*u_b**2*(uvw_sq) + k_drag*(2*CL_alpha*w_b*sq_uvw + CL_q*c*q*u_b + 2*u_b*(CL0 + CL_de*delta_e)*sq_uvw)**2)*(uvw_sq)**(3/2))*np.abs(u_b)/(8*m*u_b**3*(u_b**2 + w_b**2)**(3/2)*(uvw_sq)**(3/2))
        A[8, 9] = -v_b
        A[8, 10] = (-CL_q*S*c*k_drag*rho*w_b*np.sqrt(u_b**2 + w_b**2)*(2*CL_alpha*w_b*sq_uvw + CL_q*c*q*u_b + 2*u_b*(CL0 + CL_de*delta_e)*sq_uvw)*np.abs(u_b) - CL_q*S*c*rho*u_b**2*np.sqrt(u_b**4 + u_b**2*v_b**2 + 2*u_b**2*w_b**2 + v_b**2*w_b**2 + w_b**4)*np.abs(u_b) + 4*m*u_b**3*(u_b**2 + w_b**2))/(4*m*u_b**2*(u_b**2 + w_b**2))
        A[9, 6] = S*b*rho*u_b*(Ixz*(2*Cn_beta*v_b + Cn_p*b*p + Cn_r*b*r + 4*(Cn_da*delta_a + Cn_dr*delta_r)*sq_uvw) + Iz*(2*C_ell_beta*v_b + C_ell_p*b*p + C_ell_r*b*r + 4*(C_ell_da*delta_a + C_ell_dr*delta_r)*sq_uvw))/(4*(Ix*Iz - Ixz**2)*sq_uvw)
        A[9, 7] = S*b*rho*(Ixz*(2*Cn_beta*(uvw_sq) - v_b*(2*Cn_beta*v_b + Cn_p*b*p + Cn_r*b*r) + 2*v_b*(2*Cn_beta*v_b + Cn_p*b*p + Cn_r*b*r + 2*(Cn_da*delta_a + Cn_dr*delta_r)*sq_uvw)) + Iz*(2*C_ell_beta*(uvw_sq) - v_b*(2*C_ell_beta*v_b + C_ell_p*b*p + C_ell_r*b*r) + 2*v_b*(2*C_ell_beta*v_b + C_ell_p*b*p + C_ell_r*b*r + 2*(C_ell_da*delta_a + C_ell_dr*delta_r)*sq_uvw)))/(4*(Ix*Iz - Ixz**2)*sq_uvw)
        A[9, 8] = S*b*rho*w_b*(Ixz*(2*Cn_beta*v_b + Cn_p*b*p + Cn_r*b*r + 4*(Cn_da*delta_a + Cn_dr*delta_r)*sq_uvw) + Iz*(2*C_ell_beta*v_b + C_ell_p*b*p + C_ell_r*b*r + 4*(C_ell_da*delta_a + C_ell_dr*delta_r)*sq_uvw))/(4*(Ix*Iz - Ixz**2)*sq_uvw)
        A[9, 9] = (Ixz*(Cn_p*S*b**2*rho*sq_uvw + 4*Ix*q - 4*Iy*q) + Iz*(C_ell_p*S*b**2*rho*sq_uvw + 4*Ixz*q))/(4*(Ix*Iz - Ixz**2))
        A[9, 10] = (-Ixz*(-Ix*p + Ixz*r + Iy*p) + Iz*(Ixz*p + Iy*r - Iz*r))/(Ix*Iz - Ixz**2)
        A[9, 11] = (Ixz*(Cn_r*S*b**2*rho*sq_uvw - 4*Ixz*q) + Iz*(C_ell_r*S*b**2*rho*sq_uvw + 4*Iy*q - 4*Iz*q))/(4*(Ix*Iz - Ixz**2))
        A[10, 6] = Cm_alpha*S*c*rho*w_b/(2*Iy) - Cm_alpha*S*c*rho*v_b**2*w_b/(2*Iy*u_b**2) - Cm_alpha*S*c*rho*w_b**3/(2*Iy*u_b**2) + Cm_de*S*c*delta_e*rho*u_b/Iy + Cm_q*S*c**2*q*rho*u_b/(4*Iy*sq_uvw)
        A[10, 7] = Cm_alpha*S*c*rho*v_b*w_b/(Iy*u_b) + Cm_de*S*c*delta_e*rho*v_b/Iy + Cm_q*S*c**2*q*rho*v_b/(4*Iy*sq_uvw)
        A[10, 8] = Cm_alpha*S*c*rho*u_b/(2*Iy) + Cm_alpha*S*c*rho*v_b**2/(2*Iy*u_b) + 3*Cm_alpha*S*c*rho*w_b**2/(2*Iy*u_b) + Cm_de*S*c*delta_e*rho*w_b/Iy + Cm_q*S*c**2*q*rho*w_b/(4*Iy*sq_uvw)
        A[10, 9] = (-Ix*r - 2*Ixz*p + Iz*r)/Iy
        A[10, 10] = Cm_q*S*c**2*rho*sq_uvw/(4*Iy)
        A[10, 11] = (-Ix*p + 2*Ixz*r + Iz*p)/Iy
        A[11, 6] = S*b*rho*u_b*(Ix*(2*Cn_beta*v_b + Cn_p*b*p + Cn_r*b*r + 4*(Cn_da*delta_a + Cn_dr*delta_r)*sq_uvw) + Ixz*(2*C_ell_beta*v_b + C_ell_p*b*p + C_ell_r*b*r + 4*(C_ell_da*delta_a + C_ell_dr*delta_r)*sq_uvw))/(4*(Ix*Iz - Ixz**2)*sq_uvw)
        A[11, 7] = S*b*rho*(Ix*(2*Cn_beta*(uvw_sq) - v_b*(2*Cn_beta*v_b + Cn_p*b*p + Cn_r*b*r) + 2*v_b*(2*Cn_beta*v_b + Cn_p*b*p + Cn_r*b*r + 2*(Cn_da*delta_a + Cn_dr*delta_r)*sq_uvw)) + Ixz*(2*C_ell_beta*(uvw_sq) - v_b*(2*C_ell_beta*v_b + C_ell_p*b*p + C_ell_r*b*r) + 2*v_b*(2*C_ell_beta*v_b + C_ell_p*b*p + C_ell_r*b*r + 2*(C_ell_da*delta_a + C_ell_dr*delta_r)*sq_uvw)))/(4*(Ix*Iz - Ixz**2)*sq_uvw)
        A[11, 8] = S*b*rho*w_b*(Ix*(2*Cn_beta*v_b + Cn_p*b*p + Cn_r*b*r + 4*(Cn_da*delta_a + Cn_dr*delta_r)*sq_uvw) + Ixz*(2*C_ell_beta*v_b + C_ell_p*b*p + C_ell_r*b*r + 4*(C_ell_da*delta_a + C_ell_dr*delta_r)*sq_uvw))/(4*(Ix*Iz - Ixz**2)*sq_uvw)
        A[11, 9] = (Ix*(Cn_p*S*b**2*rho*sq_uvw + 4*Ix*q - 4*Iy*q) + Ixz*(C_ell_p*S*b**2*rho*sq_uvw + 4*Ixz*q))/(4*(Ix*Iz - Ixz**2))
        A[11, 10] = (-Ix*(-Ix*p + Ixz*r + Iy*p) + Ixz*(Ixz*p + Iy*r - Iz*r))/(Ix*Iz - Ixz**2)
        A[11, 11] = (Ix*(Cn_r*S*b**2*rho*sq_uvw - 4*Ixz*q) + Ixz*(C_ell_r*S*b**2*rho*sq_uvw + 4*Iy*q - 4*Iz*q))/(4*(Ix*Iz - Ixz**2))

        return A

    def f_u_analytical(self, x, u, params):
        """
        Analytical input Jacobian (df/du) for fixed-wing aircraft.
        
        Args:
            x: State vector [X, Y, Z, phi, theta, psi, u, v, w, p, q, r]
            u: Input vector [delta_e, delta_a, delta_r, throttle]
            params: Dictionary of aircraft parameters
        
        Returns:
            B: Input Jacobian matrix (12x4)
        """
        # Extract state variables
        X = x[0]
        Y = x[1]
        Z = x[2]
        phi = x[3]
        theta = x[4]
        psi = x[5]
        u_b = x[6]  # body velocity u (renamed to avoid conflict with input u)
        v_b = x[7]  # body velocity v
        w_b = x[8]  # body velocity w
        p = x[9]
        q = x[10]
        r = x[11]

        # Extract input variables
        u = np.zeros(4) if u is None else u
        delta_e = u[0]
        delta_a = u[1]
        delta_r = u[2]
        throttle = u[3]

        # Extract parameters
        m, S, b, c, rho, Ix, Iy, Iz, Ixz, CL0, CL_alpha, CL_q, CL_de, CD0, k_drag, Cm_alpha, Cm_q, Cm_de, CY_beta, CY_p, CY_r, CY_da, CY_dr, C_ell_beta, C_ell_p, C_ell_r, C_ell_da, C_ell_dr, Cn_beta, Cn_p, Cn_r, Cn_da, Cn_dr, T_max, g = self._extract_params(params)

        # Initialize Jacobian matrix
        B = np.zeros((12, 4))

        # Compute Jacobian elements (generated by SymPy)
        # Note: Only non-zero elements are computed for efficiency
        uvw_sq = u_b**2 + v_b**2 + w_b**2
        sq_uvw = np.sqrt(uvw_sq)

        B[6, 0] = CL_de*S*rho*(-k_drag*sq_uvw*(2*CL_alpha*w_b*sq_uvw + CL_q*c*q*u_b + 2*u_b*(CL0 + CL_de*delta_e)*sq_uvw) - w_b*(uvw_sq))*np.abs(u_b)/(2*m*u_b*np.sqrt(u_b**2 + w_b**2))
        B[6, 3] = T_max/m
        B[7, 1] = CY_da*S*rho*(uvw_sq)/(2*m)
        B[7, 2] = CY_dr*S*rho*(uvw_sq)/(2*m)
        B[8, 0] = CL_de*S*rho*(-k_drag*w_b*(2*CL_alpha*w_b*sq_uvw + CL_q*c*q*u_b + 2*u_b*(CL0 + CL_de*delta_e)*sq_uvw)*np.sqrt(u_b**4 + u_b**2*v_b**2 + 2*u_b**2*w_b**2 + v_b**2*w_b**2 + w_b**4) - u_b**2*np.sqrt(u_b**2 + w_b**2)*(uvw_sq))*np.abs(u_b)/(2*m*u_b**2*(u_b**2 + w_b**2))
        B[9, 1] = S*b*rho*(C_ell_da*Iz + Cn_da*Ixz)*(uvw_sq)/(2*(Ix*Iz - Ixz**2))
        B[9, 2] = S*b*rho*(C_ell_dr*Iz + Cn_dr*Ixz)*(uvw_sq)/(2*(Ix*Iz - Ixz**2))
        B[10, 0] = Cm_de*S*c*rho*(uvw_sq)/(2*Iy)
        B[11, 1] = S*b*rho*(C_ell_da*Ixz + Cn_da*Ix)*(uvw_sq)/(2*(Ix*Iz - Ixz**2))
        B[11, 2] = S*b*rho*(C_ell_dr*Ixz + Cn_dr*Ix)*(uvw_sq)/(2*(Ix*Iz - Ixz**2))

        return B

    def f_x(self, x, u):
        return self.f_x_analytical(x, u, self.params)
    
    def f_u(self, x, u=None):
        """
        Compute the input Jacobian df/du.
        
        Args:
            x: State vector
            u: Input vector. If None, uses trim input for better approximation
        
        Returns:
            Input Jacobian matrix (12x4)
        """
        if u is None:
            u = self.u_trim  # Use trim input for better linearization
        return self.f_u_analytical(x, u, self.params)
    
    def rk4Step(self, f, x, dt, *args):
        """
        Fourth-order Runge-Kutta integration method
        """
        k1 = f(x, *args)
        k2 = f(x + 0.5*dt*k1, *args)
        k3 = f(x + 0.5*dt*k2, *args)
        k4 = f(x + dt*k3, *args)
        return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


    def step(self, x, u, dt=None):
        # follow same pattern as other classes: clip inputs and RK4
        dt = self.dt if dt is None else dt
        u = np.asarray(u)
        u = np.clip(u, self.input_limits[:, 0], self.input_limits[:, 1])
        return self.rk4Step(self.f, x, dt, *(u,))


    def computeTrim(self, V_trim=10.0):
        """
        Compute a symmetric trim for desired airspeed V_trim (m/s).
        Returns x_trim (state) and u_trim (controls).
        """
        def _trimObjective(vars, plane, V_trim):
            """
            vars: [w, theta, delta_e, throttle]
            w: body z-velocity
            theta: pitch angle (rad)
            delta_e: elevator (rad)
            throttle: 0..1
            plane: instance of FixedWing12DOFTrainer
            V_trim: desired airspeed (m/s) used for u (body-x)
            returns residuals [udot, wdot, qdot, u - V_trim]
            Note: class state ordering is:
            x = [X, Y, Z, phi, theta, psi, u, v, w, p, q, r]
            """
            u, w, theta, delta_e, throttle = vars
            # build state with symmetric (no lateral motion), no angular rates
            X = 0.0; Y = 0.0; Z = -0.0  # choose Z reference (your convention)
            phi = 0.0
            psi = 0.0
            v = 0.0
            p = 0.0; q = 0.0; r = 0.0

            x = np.array([X, Y, Z, phi, theta, psi, u, v, w, p, q, r], dtype=float)
            u_ctrl = np.array([delta_e, 0.0, 0.0, throttle])  # symmetric (ail/rud zero)

            # evaluate dynamics
            xdot = plane.f(x, u_ctrl)

            # residuals: udot = 0, wdot = 0, qdot = 0 (pitch accel), Zdot=0, and u - V_trim = 0
            # xdot ordering in this implementation:
            # xdot[0:3] = pos_dot (Xdot,Ydot,Zdot)
            # xdot[3:6] = [phi_dot, theta_dot, psi_dot]
            # xdot[6:9] = [udot, vdot, wdot]
            # xdot[9:12] = [pdot, qdot, rdot]
            udot = xdot[6]
            wdot = xdot[8]
            qdot = xdot[10]
            Zdot = xdot[2]  # vertical speed should be zero for level flight
            airspeed = np.sqrt(u**2 + v**2 + w**2)
            # last residual enforces body-x speed equals V_trim (u - V_trim = 0)
            res = np.array([udot, wdot, qdot, Zdot, airspeed - V_trim])
            # res = np.array([udot, wdot, qdot, Zdot])
            # res = np.array([udot, wdot, qdot, u - V_trim])
            return res
        
        # initial guess: small w, small pitch, small elevator, half throttle
        guess = np.array([V_trim, 0.0, 0.05, 0.0, 0.5])  # [V_trim, w, theta, delta_e, throttle]
        sol = root(_trimObjective, guess, args=(self, V_trim), method='hybr', tol=1e-8)

        if not sol.success:
            print("Trim solver did not converge:", sol.message)
            # still return a best-effort guess
        u, w, theta, delta_e, throttle = sol.x
        x_trim = np.array([0.0, 0.0, 0.0,  # X, Y, Z
                        0.0, theta, 0.0,   # phi, theta, psi
                        u, 0.0, w,         # u, v, w
                        0.0, 0.0, 0.0])    # p, q, r
        u_trim = np.array([delta_e, 0.0, 0.0, np.clip(throttle, 0.0, 1.0)])

        # Print trim state message
        print("==============================================")
        print("Trim solver success:", sol.success, sol.message)
        print(f"Trim state (partial): u, w, theta = {x_trim[6]:.2f}, {x_trim[8]:.2f}, {180 / np.pi * x_trim[4]:.2f}")
        print(f"Trim inputs (de, throttle) = {180/np.pi*u_trim[0]:.2f}°, {u_trim[3]:.2%}")
        print("==============================================")

        return x_trim, u_trim, sol

    def linearizeAtTrimPoint(self, x_trim, u_trim, eps=1e-6):
        n = x_trim.size
        m = u_trim.size
        f0 = self.f(x_trim, u_trim)
        A = np.zeros((n,n))
        B = np.zeros((n,m))
        # A: df/dx
        for i in range(n):
            xp = x_trim.copy(); xp[i] += eps
            A[:, i] = (self.f(xp, u_trim) - f0) / eps
        # B: df/du
        for j in range(m):
            up = u_trim.copy(); up[j] += eps
            B[:, j] = (self.f(x_trim, up) - f0) / eps
        return A, B

    @property
    def state_string(self):
        X, Y, Z, phi, theta, psi, u, v, w, p, q, r = self.state
        return f"X:{X:.2f} Y:{Y:.2f} Z:{Z:.2f} || phi:{phi*180/np.pi:.1f}° theta:{theta*180/np.pi:.1f}° psi:{psi*180/np.pi:.1f}° || u:{u:.2f} v:{v:.2f} w:{w:.2f} || p:{p*180/np.pi:.1f} [°/s] q:{q*180/np.pi:.1f} [°/s] r:{r*180/np.pi:.1f} [°/s]"

    # Not used for now
    def getInertialVelocity(self, x=None):
        """
        Compute inertial velocity vector [Xdot, Ydot, Zdot] from state.
        
        Args:
            x: State vector [X, Y, Z, phi, theta, psi, u, v, w, p, q, r].
               If None, uses current state.
        
        Returns:
            vel_inertial: np.array([Xdot, Ydot, Zdot]) - inertial velocities (m/s)
        """
        if x is None:
            x = self.state
        
        # Extract orientation angles and body velocities
        phi, theta, psi = x[3], x[4], x[5]
        u_b, v_b, w_b = x[6], x[7], x[8]
        
        # Build rotation matrix from body to inertial frame
        cpsi = np.cos(psi); spsi = np.sin(psi)
        cth = np.cos(theta); sth = np.sin(theta)
        cphi = np.cos(phi); sphi = np.sin(phi)
        
        R = np.array([
            [cpsi * cth, cpsi * sth * sphi - spsi * cphi, cpsi * sth * cphi + spsi * sphi],
            [spsi * cth, spsi * sth * sphi + cpsi * cphi, spsi * sth * cphi - cpsi * sphi],
            [-sth,       cth * sphi,                     cth * cphi]
        ])
        
        # Transform body velocities to inertial frame
        vel_body = np.array([u_b, v_b, w_b])
        vel_inertial = R.dot(vel_body)
        
        return vel_inertial

    


import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

class FixedWing12DOFTrainerJAX(FixedWing12DOFTrainer):
    """
    JAX-enabled version of FixedWing12DOFTrainer using automatic differentiation.
    Provides exact analytical Jacobians via JAX's autodiff without manual derivation.
    
    Maintains the same API as FixedWing12DOFTrainer for interchangeable behavior.
    
    PERFORMANCE CHARACTERISTICS (from benchmarks):
    - f(x,u):    ~1.3x SLOWER than NumPy (array conversion overhead)
    - f_x(x,u):  ~5x FASTER than finite differences (exact autodiff)
    - f_u(x):    ~5x FASTER than finite differences (exact autodiff)
    
    WHEN TO USE:
    ✓ Computing Jacobians for optimization, control design, or sensitivity analysis
    ✓ Need exact analytical derivatives (no numerical errors)
    ✓ Batch processing multiple states/controls
    ✓ Extending to higher-order derivatives (Hessians, etc.)
    
    WHEN NOT TO USE:
    ✗ Real-time control loops where every microsecond counts
    ✗ Only need dynamics f(x,u), not Jacobians
    
    State ordering:
      x = [X, Y, Z, phi, theta, psi, u, v, w, p, q, r]
    Inputs:
      u = [delta_e, delta_a, delta_r, throttle]
      
    Example:
        >>> plane = FixedWing12DOFTrainerJAX(v_trim=15.0, dt=0.01)
        >>> xdot = plane.f(x, u)                # Dynamics (slower than NumPy)
        >>> A = plane.f_x(x, u)                 # Jacobian (5x faster!) (Compared to FD)
        >>> B = plane.f_u(x)                    # Jacobian (5x faster!) (Compared to FD)
    """
    
    def __init__(self, dt=0.001, x0=None, params=None, v_trim=10, use_linear_f=False, use_linear_fx_fu=False):
        """
        Initialize JAX-enabled fixed-wing trainer.
        
        Args:
            dt: Integration time step
            x0: Initial state (12-element array)
            params: Dictionary of aircraft parameters (overrides defaults)
            v_trim: Trim airspeed (m/s)
            use_linear_f: If True, use linearized dynamics for f()
            use_linear_fx_fu: If True, return trim-point Jacobians A, B
        """
        # We need to initialize JAX functions before calling parent __init__
        # because parent __init__ calls computeTrim() which calls self.f()
        # So we do a partial initialization first
        
        # Set up basic attributes needed for _f_jax_pure
        self.dt = dt
        self.num_of_states = 12
        self.num_of_inputs = 4
        
        # default parameter set (good starting point for 1.5 m trainer)
        default_params = {
            'm': 2,          # kg
            'S': 0.36,       # m^2
            'b': 1.50,       # m (span)
            'c': 0.2407,     # m (mean aerodynamic chord ~ S/b)
            'rho': 1.225,
            'Ix': 0.072,     # kg m^2 (estimate; replace with CAD)
            'Iy': 0.0255,
            'Iz': 0.0963,
            'Ixz': -6.12e-4,
            # longitudinal coefficients (linearized starter)
            'CL0': 0.20,
            'CL_alpha': 4.756,   # per rad
            'CL_q': 3.0,
            'CL_de': -1.0,       # elevator effectiveness (neg. if down elevator -> negative Cm_de)
            'CD0': 0.025,
            'k': 0.0639,         # induced drag factor
            'Cm_alpha': -0.8,
            'Cm_q': -8.0,
            'Cm_de': -1.2,
            # lateral-directional
            'C_ell_beta': -0.12,
            'C_ell_p': -0.26,
            'C_ell_r': 0.14,
            'Cn_beta': 0.25,
            'Cn_p': -0.022,
            'Cn_r': -0.35,
            'CY_beta': -0.02,
            # control-surface derivatives (starter guesses)
            'C_ell_da': 0.10,   # roll per aileron rad
            'C_ell_dr': 0.03,   # roll per rudder rad
            'Cn_da': -0.03,     # yaw per aileron rad (adverse yaw)
            'Cn_dr': -0.10,     # yaw per rudder rad (If we dont reverse Usafe, this should be positive)
            'CY_da': 0.0,       # side force per aileron rad
            'CY_dr': 0.12,      # side force per rudder rad
            'CY_p': 0.0,
            'CY_r': 0.0,
            # propulsion
            'T_max': 10.0,   # N (example static thrust)
            # input limits: [min, max] for [de, da, dr, throttle]
            'input_limits': np.array([[-0.4363, 0.4363],  # elevator ±25 deg
                                    [-0.4363, 0.4363],  # aileron ±25 deg
                                    [-0.4363, 0.4363],  # rudder ±25 deg
                                    [0.0, 1.0]]),       # throttle 0..1
            'V_trim': v_trim  # Desired trim speed [m/s]
        }

        self.params = default_params if params is None else {**default_params, **params}
        self.input_limits = self.params['input_limits'].copy()
        
        # Convert input limits to JAX array
        self.input_limits_jax = jnp.array(self.input_limits)
        
        # Create JIT-compiled versions of dynamics and Jacobians for performance
        # These MUST be created before calling parent __init__
        self._f_jax_compiled = jax.jit(self._f_jax_pure)
        self._f_x_jax_compiled = jax.jit(jax.jacfwd(self._f_jax_pure, argnums=0))
        self._f_u_jax_compiled = jax.jit(jax.jacfwd(self._f_jax_pure, argnums=1))
        
        # Call parent constructor
        # NOTE: Parent will call computeTrim() which uses self.f()
        # Our JAX f() will be used, but trim solver may converge to different solution
        super().__init__(dt=dt, x0=x0, params=params, v_trim=v_trim, 
                        use_linear_f=use_linear_f, use_linear_fx_fu=use_linear_fx_fu)
        
        # WORKAROUND for trim solver issue:
        # The JAX dynamics can cause scipy's root finder to converge to spurious solutions
        # (e.g., theta ≈ -2π instead of theta ≈ 0.15, with zero throttle)
        # Use the parent class's trimmer as a reference
        if abs(self.x_trim[4]) > 1.0:  # theta > 1 radian (57°) is suspicious
            print(f"[JAX WARNING] Trim theta={180/np.pi*self.x_trim[4]:.1f}° seems wrong, recomputing...")
            # Use parent's f() for trim
            parent_f = FixedWing12DOFTrainer.f
            saved_f = self.f
            self.f = lambda x, u: parent_f(self, x, u)
            self.x_trim, self.u_trim, self.trim_sol_flags = self.computeTrim(V_trim=v_trim)
            self.f = saved_f  # Restore JAX f()
            # Recompute linearization with corrected trim
            self.A, self.B = self.linearizeAtTrimPoint(self.x_trim, self.u_trim)
        
        self.type = "FixedWing12DOFTrainerJAX"
        print(f"[JAX] Initialized {self.type} with automatic differentiation")
        print(f"[JAX] Jacobians computed via autodiff (exact analytical derivatives)")
    
    def _f_jax_pure(self, x, u):
        """
        Pure JAX implementation of aircraft dynamics (no side effects).
        Uses jax.numpy for automatic differentiation compatibility.
        
        This is the core function that JAX will differentiate.
        
        Args:
            x: State vector (12,) [X, Y, Z, phi, theta, psi, u, v, w, p, q, r]
            u: Control vector (4,) [delta_e, delta_a, delta_r, throttle]
        
        Returns:
            xdot: State derivative (12,)
        """
        # Clip inputs to limits
        u = jnp.clip(u, self.input_limits_jax[:, 0], self.input_limits_jax[:, 1])
        delta_e, delta_a, delta_r, throttle = u
        
        # Unpack state
        X, Y, Z, phi, theta, psi, ub, vb, wb, p, q, r = x
        
        # Parameters (use dict access for clarity)
        P = self.params
        m = P['m']; S = P['S']; b = P['b']; c = P['c']; rho = P['rho']
        Ix = P['Ix']; Iy = P['Iy']; Iz = P['Iz']; Ixz = P['Ixz']
        
        # Airspeed and aerodynamic angles
        V_squared = ub*ub + vb*vb + wb*wb
        V_squared_safe = jnp.maximum(V_squared, 1e-6)
        V = jnp.sqrt(V_squared_safe)
        V_safe = jnp.maximum(V, 1e-3)
        
        alpha = jnp.arctan2(wb, ub)  # angle of attack
        beta = jnp.arcsin(jnp.clip(vb / V_safe, -0.99, 0.99))  # sideslip
        
        qbar = 0.5 * rho * V_squared
        
        # Longitudinal coefficients
        CL = (P['CL0'] + P['CL_alpha'] * alpha + 
              P['CL_q'] * (c * q / (2.0 * V_safe)) + 
              P['CL_de'] * delta_e)
        CD = P['CD0'] + P['k'] * CL**2
        
        # Side force coefficient
        CY = (P['CY_beta'] * beta +
              P['CY_p'] * (b * p / (2.0 * V_safe)) +
              P['CY_r'] * (b * r / (2.0 * V_safe)) +
              P['CY_da'] * delta_a +
              P['CY_dr'] * delta_r)
        
        # Aerodynamic forces (wind axes)
        L_aero = qbar * S * CL
        D_aero = qbar * S * CD
        Y_force = qbar * S * CY
        
        # Transform wind-axis forces to body axes (rotate by alpha)
        cos_alpha = jnp.cos(alpha)
        sin_alpha = jnp.sin(alpha)
        X_aero = -D_aero * cos_alpha - L_aero * sin_alpha
        Z_aero = -D_aero * sin_alpha - L_aero * cos_alpha
        Y_aero = Y_force
        
        # Moment coefficients
        Cl = (P['C_ell_beta'] * beta +
              P['C_ell_p'] * (b * p / (2.0 * V_safe)) +
              P['C_ell_r'] * (b * r / (2.0 * V_safe)) +
              P['C_ell_da'] * delta_a +
              P['C_ell_dr'] * delta_r)
        
        Cm = (P['Cm_alpha'] * alpha +
              P['Cm_q'] * (c * q / (2.0 * V_safe)) +
              P['Cm_de'] * delta_e)
        
        Cn = (P['Cn_beta'] * beta +
              P['Cn_p'] * (b * p / (2.0 * V_safe)) +
              P['Cn_r'] * (b * r / (2.0 * V_safe)) +
              P['Cn_da'] * delta_a +
              P['Cn_dr'] * delta_r)
        
        L_moment = qbar * S * b * Cl
        M_moment = qbar * S * c * Cm
        N_moment = qbar * S * b * Cn
        
        # Propulsion: thrust along body x-axis
        T = P['T_max'] * jnp.clip(throttle, 0.0, 1.0)
        X_prop = T
        Y_prop = 0.0
        Z_prop = 0.0
        
        # Gravity in body axes
        g = 9.81
        sin_theta = jnp.sin(theta)
        cos_theta = jnp.cos(theta)
        sin_phi = jnp.sin(phi)
        cos_phi = jnp.cos(phi)
        
        X_grav = -m * g * sin_theta
        Y_grav = m * g * cos_theta * sin_phi
        Z_grav = m * g * cos_theta * cos_phi
        
        # Total forces
        X_tot = X_aero + X_prop + X_grav
        Y_tot = Y_aero + Y_prop + Y_grav
        Z_tot = Z_aero + Z_prop + Z_grav
        
        # Translational accelerations (body axes)
        udot = (X_tot / m) - q * wb + r * vb
        vdot = (Y_tot / m) - r * ub + p * wb
        wdot = (Z_tot / m) - p * vb + q * ub
        
        # Rotational equations of motion: I * omega_dot + omega x (I * omega) = M
        # Using JAX-compatible matrix operations
        I = jnp.array([[Ix, 0.0, -Ixz],
                       [0.0, Iy, 0.0],
                       [-Ixz, 0.0, Iz]])
        omega = jnp.array([p, q, r])
        M_vec = jnp.array([L_moment, M_moment, N_moment])
        
        # omega_dot = I^-1 * (M - omega x (I * omega))
        I_omega = I @ omega
        cross_term = jnp.cross(omega, I_omega)
        omega_dot = jnp.linalg.solve(I, M_vec - cross_term)
        pdot, qdot, rdot = omega_dot
        
        # Euler kinematics (body rates -> euler angle rates)
        tan_theta = jnp.tan(theta)
        sec_theta = 1.0 / jnp.maximum(jnp.abs(cos_theta), 1e-6)
        sec_theta = jnp.where(cos_theta >= 0, sec_theta, -sec_theta)
        
        E = jnp.array([[1.0, sin_phi * tan_theta, cos_phi * tan_theta],
                       [0.0, cos_phi, -sin_phi],
                       [0.0, sin_phi * sec_theta, cos_phi * sec_theta]])
        euler_dot = E @ omega
        phi_dot, theta_dot, psi_dot = euler_dot
        
        # Inertial position derivative (body -> inertial)
        sin_psi = jnp.sin(psi)
        cos_psi = jnp.cos(psi)
        
        R = jnp.array([
            [cos_psi * cos_theta, 
             cos_psi * sin_theta * sin_phi - sin_psi * cos_phi,
             cos_psi * sin_theta * cos_phi + sin_psi * sin_phi],
            [sin_psi * cos_theta,
             sin_psi * sin_theta * sin_phi + cos_psi * cos_phi,
             sin_psi * sin_theta * cos_phi - cos_psi * sin_phi],
            [-sin_theta,
             cos_theta * sin_phi,
             cos_theta * cos_phi]
        ])
        vel_body = jnp.array([ub, vb, wb])
        pos_dot = R @ vel_body
        
        # Assemble state derivative
        xdot = jnp.concatenate([
            pos_dot,                           # [Xdot, Ydot, Zdot]
            euler_dot,                         # [phi_dot, theta_dot, psi_dot]
            jnp.array([udot, vdot, wdot]),    # [udot, vdot, wdot]
            omega_dot                          # [pdot, qdot, rdot]
        ])
        
        return xdot
    
    def f(self, x, u):
        """
        Dynamics function compatible with parent class API.
        
        If use_linear_model_for_f is True, uses linearized dynamics.
        Otherwise, uses JAX implementation for consistency with Jacobians.
        
        Args:
            x: State vector (numpy array)
            u: Control vector (numpy array)
        
        Returns:
            xdot: State derivative (numpy array)
        """
        # Clip inputs using numpy
        u = np.asarray(u)
        u = np.clip(u, self.input_limits[:, 0], self.input_limits[:, 1])
        
        # If using linear model, use the linearized dynamics
        if self.use_linear_model_for_f:
            x_dot = self.A @ (x - self.x_trim) + self.B @ (u - self.u_trim)
            return x_dot
        
        # Convert to JAX arrays, compute, and convert back to numpy
        x_jax = jnp.array(x)
        u_jax = jnp.array(u)
        xdot_jax = self._f_jax_compiled(x_jax, u_jax)
        
        return np.array(xdot_jax)
    
    def f_x(self, x, u, eps=1e-6):
        """
        Jacobian of dynamics with respect to state (df/dx).
        Computed using JAX automatic differentiation.
        
        Args:
            x: State vector (numpy array)
            u: Control vector (numpy array)
            eps: Unused (kept for API compatibility)
        
        Returns:
            A: Jacobian matrix (12 x 12) as numpy array
        """
        # If using linear model, return the trim-point Jacobian
        if self.use_linear_model_for_fx_fu:
            return self.A.copy()
        
        # Convert to JAX arrays
        x_jax = jnp.array(x)
        u_jax = jnp.array(u)
        
        # Compute Jacobian using JAX autodiff
        A_jax = self._f_x_jax_compiled(x_jax, u_jax)
        
        return np.array(A_jax)
    
    def f_u(self, x, eps=1e-6):
        """
        Jacobian of dynamics with respect to inputs (df/du).
        Computed using JAX automatic differentiation.
        
        Args:
            x: State vector (numpy array)
            eps: Unused (kept for API compatibility)
        
        Returns:
            B: Jacobian matrix (12 x 4) as numpy array
        """
        # If using linear model, return the trim-point Jacobian
        if self.use_linear_model_for_fx_fu:
            return self.B.copy()
        
        # Convert to JAX arrays
        x_jax = jnp.array(x)
        # For consistency with parent class, evaluate at zero control
        # (though ideally we'd evaluate at the actual control input)
        u_jax = jnp.zeros(self.num_of_inputs)
        
        # Compute Jacobian using JAX autodiff
        B_jax = self._f_u_jax_compiled(x_jax, u_jax)
        
        return np.array(B_jax)
    
    def f_jax(self, x, u):
        """
        Pure JAX dynamics function (NO numpy conversions).
        
        This method is designed for use within JAX transformations (grad, jit, vmap).
        It accepts JAX arrays and returns JAX arrays without any numpy conversions.
        
        Use this method when:
        - Computing gradients with jax.grad()
        - Inside JIT-compiled functions
        - Within vmap/pmap operations
        
        Do NOT use this method for:
        - Regular dynamics evaluation (use f() instead)
        - Integration with numpy-based code
        
        Args:
            x: State vector (JAX array, shape (12,))
            u: Control vector (JAX array, shape (4,))
        
        Returns:
            xdot: State derivative (JAX array, shape (12,))
        """
        return self._f_jax_compiled(x, u)
    
    def g_jax(self, x):
        """
        Pure JAX control matrix function (NO numpy conversions).
        
        Returns the control effectiveness matrix g(x) where xdot = f(x,u) = f(x,0) + g(x)*u.
        For control-affine systems, g(x) is the Jacobian df/du.
        
        This method is designed for use within JAX transformations.
        
        Args:
            x: State vector (JAX array, shape (12,))
        
        Returns:
            g: Control matrix (JAX array, shape (12, 4))
        """
        u_zero = jnp.zeros(self.num_of_inputs)
        return self._f_u_jax_compiled(x, u_zero)