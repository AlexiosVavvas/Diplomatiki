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

    def f_u(self, x):
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
            'm': 1.6,        # kg
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
            'C_ell_dr': 0.01,   # roll per rudder rad
            'Cn_da': -0.03,     # yaw per aileron rad (adverse yaw)
            'Cn_dr': -0.10,     # yaw per rudder rad
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

        # Linearise flight dynamics
        # x_dot = A x + B u
        self.A, self.B = self.linearizeAtTrimPoint(self.x_trim, self.u_trim)
        self.use_linear_model_for_f = use_linear_f
        self.use_linear_model_for_fx_fu = use_linear_fx_fu

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

    def f_x(self, x, u, eps=1e-6):
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

    def f_u(self, x, eps=1e-6):
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
            w, theta, delta_e, throttle = vars
            # build state with symmetric (no lateral motion), no angular rates
            X = 0.0; Y = 0.0; Z = -0.0  # choose Z reference (your convention)
            phi = 0.0
            psi = 0.0
            u = V_trim
            v = 0.0
            p = 0.0; q = 0.0; r = 0.0

            x = np.array([X, Y, Z, phi, theta, psi, u, v, w, p, q, r], dtype=float)
            u_ctrl = np.array([delta_e, 0.0, 0.0, throttle])  # symmetric (ail/rud zero)

            # evaluate dynamics
            xdot = plane.f(x, u_ctrl)

            # residuals: udot = 0, wdot = 0, qdot = 0 (pitch accel), and u - V_trim = 0
            # xdot ordering in this implementation:
            # xdot[0:3] = pos_dot (Xdot,Ydot,Zdot)
            # xdot[3:6] = [phi_dot, theta_dot, psi_dot]
            # xdot[6:9] = [udot, vdot, wdot]
            # xdot[9:12] = [pdot, qdot, rdot]
            udot = xdot[6]
            wdot = xdot[8]
            qdot = xdot[10]
            # last residual enforces body-x speed equals V_trim (u - V_trim = 0)
            res = np.array([udot, wdot, qdot, u - V_trim])
            return res
        
        # initial guess: small w, small pitch, small elevator, half throttle
        guess = np.array([0.0, 0.05, 0.0, 0.5])  # [w, theta, delta_e, throttle]
        sol = root(_trimObjective, guess, args=(self, V_trim), method='hybr', tol=1e-8)

        if not sol.success:
            print("Trim solver did not converge:", sol.message)
            # still return a best-effort guess
        w, theta, delta_e, throttle = sol.x
        u = V_trim
        x_trim = np.array([0.0, 0.0, 0.0,   # X, Y, Z
                        0.0, theta, 0.0,  # phi, theta, psi
                        u, 0.0, w,         # u, v, w
                        0.0, 0.0, 0.0])    # p, q, r
        u_trim = np.array([delta_e, 0.0, 0.0, np.clip(throttle, 0.0, 1.0)])

        # Print trim state message
        print("==============================================")
        print("Trim solver success:", sol.success, sol.message)
        print(f"Trim state (partial): u, w, theta = {x_trim[6]:.2f}, {x_trim[8]:.2f}, {180 / np.pi * x_trim[4]:.2f}")
        print(f"Trim inputs (de, da, dr, throttle) = {180/np.pi*u_trim[0]:.2f}°, {u_trim[1]:.2f}°, {u_trim[2]:.2f}°, {u_trim[3]:.2%}")
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
