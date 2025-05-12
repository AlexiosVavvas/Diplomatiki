import numpy as np

class SingleIntegrator():
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
        self.dt = dt
        self.num_of_states = 2
        self.num_of_inputs = 2

        self.A = np.array([
                [0., 0.],
                [0., 0.]
        ])# - np.diag([0,0,1,1]) * 0.25

        self.B = np.array([
                [1.0, 0.],
                [0., 1.0]
        ])

        self.reset(x0)

    def reset(self, state=None):
        if state is None:
            # random seed for reproducibility
            np.random.seed(0)
            self.state = np.random.uniform(0., 1., size=(self.num_of_states,))
        else:
            assert len(state) == self.num_of_states, f"Reset Input state must be of length: {self.num_of_states}."
            self.state = np.array(state.copy())
        return self.state.copy()
    
    def f(self, x, u):
        '''
        Continuous time dynamics
        '''
        return self.A @ x + self.B @ u

    def f_x(self, x):
        '''
        Jacobian of the dynamics with respect to x
        '''
        return self.A.copy()

    def f_u(self, x):
        '''
        Jacobian of the dynamics with respect to u
        '''
        return self.B.copy()

    def h(self, x):
        '''
        Affine Dynamics Control Part
        '''
        return self.f_u(x)


    def step(self, u):
        self.state = self.state + self.f(self.state, u) * self.dt
        return self.state
    
    @property
    def ergodic_state(self):
        return self.state.copy()
    


class DoubleIntegrator():
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
    def __init__(self, mass=1, dt=0.001, x0=None):
        
        self.dt = dt
        self.num_of_states = 4
        self.num_of_inputs = 2
        self.m = mass

        self.A = np.array([
                [0., 0., 1.0, 0.],
                [0., 0., 0., 1.0],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.]
        ])

        self.B = np.array([
                [0., 0.],
                [0., 0.],
                [1.0, 0.],
                [0., 1.0]
        ]) / self.m
        
        self.reset(x0)

    def reset(self, state=None):
        if state is None:
            # random seed for reproducibility
            np.random.seed(0)
            self.state = np.random.uniform(0., 1., size=(self.num_of_states,))
        else:
            assert len(state) == self.num_of_states, f"Reset Input state must be of length: {self.num_of_states}."
            self.state = np.array(state.copy())
        return self.state.copy()

    def f(self, x, u):
        '''
        Continuous time dynamics
        '''
        return self.A @ x + self.B @ u

    def f_x(self, x):
        '''
        Jacobian of the dynamics with respect to x
        '''
        return self.A.copy()

    def f_u(self, x):
        '''
        Jacobian of the dynamics with respect to u
        '''
        return self.B.copy()

    def h(self, x):
        '''
        Affine Dynamics Control Part
        '''
        return self.f_u(x)


    def step(self, u):
        self.state = self.state + self.f(self.state, u) * self.dt
        return self.state
    
    @property
    def ergodic_state(self):
        return self.state[:2].copy()


from scipy.linalg import solve_continuous_are
class Quadcopter():
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

    def __init__(self, dt=0.001, x0=None, mass=0.1, damping=0, Q=None, R=None):
        self.dt = dt
        self.num_of_states = 12
        self.num_of_inputs = 4
        self.m = mass
        self.damping = damping
        self.A = np.zeros((self.num_of_states, self.num_of_states)) +  np.diag([1.0]*6, 6)
        self.B = np.zeros((self.num_of_states, self.num_of_inputs))

        self.reset(x0)

        # LQR Control for stabilization
        #                                                    [x,   y,   z,   psi, theta, phi, x',  y',  z',  psidot, thetadot, phidot]
        self.Q = np.asarray(Q) if Q is not None else np.diag([0.1, 0.1, 100, 1,   1,     1,   10,  10,  10,  1,      1,        1])
        self.R = np.asarray(R) if R is not None else np.diag([1, 1, 1, 1])
        self.k_lqr = self._calculateLqrControlGain(self.Q, self.R)


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

    def fdx(self, x, u):
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
    
    def fdu(self, x, u):
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
        '''
        return self.f_u(x)

    def step(self, x0, u0):
        return self.rk4Step(self.f, x0, self.dt, *(u0,))
    

    def _calculateLqrControlGain(self, Q, R):
        """
        Calculate the LQR control gain matrix K using the continuous-time algebraic Riccati equation.
        """
        assert Q.shape == (self.num_of_states, self.num_of_states), "Q must be a square matrix of size num_of_states."
        assert R.shape == (self.num_of_inputs, self.num_of_inputs), "R must be a square matrix of size num_of_inputs."
        
        # u_nom -> Thrust = Weight - Torque = 0
        u_nom = np.zeros((self.num_of_inputs,))
        u_nom[0] = self.m * 9.81

        self.fdx(self.state, u_nom)
        self.fdu(self.state, u_nom)

        # Solve the continuous-time algebraic Riccati equation
        P = solve_continuous_are(self.A, self.B, Q, R)

        # Calculate the LQR gain
        K = np.linalg.inv(R) @ self.B.T @ P

        # Zero out the first two columns of K, we dont need to control x or y 
        K[:, :2] = 0
        # K[:, 6:8] = 0 # Uncomment if we dont care about controlling x'. y; 
        
        return K

    def calcLQRcontrol(self, x, t, z_target=1):
        
        # Define the target state
        state_target = np.array([0, 0, z_target, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        # Calculate the control input
        u = -self.k_lqr @ (x - state_target)

        u[0] += self.m * 9.81 # Adjust thrust to maintain altitude
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


    @property
    def ergodic_state(self):
        return self.state[:2].copy()