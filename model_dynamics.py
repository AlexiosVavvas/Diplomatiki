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
    def __init__(self, dt=0.001):
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

        self.state = self.reset()

    def reset(self, state=None):
        if state is None:
            # random seed for reproducibility
            np.random.seed(0)
            self.state = np.random.uniform(0., 1., size=(2,))
        else:
            assert len(state) == self.num_of_states, f"Reset Input state must be of length: {self.num_of_states}."
            self.state = state.copy()
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
    def __init__(self, dt=0.001):
        
        self.dt = dt
        self.num_of_states = 4
        self.num_of_inputs = 2

        self.A = np.array([
                [0., 0., 1.0-0.2, 0.],
                [0., 0., 0., 1.0-0.2],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.]
        ])# - np.diag([0,0,1,1]) * 0.25

        self.B = np.array([
                [0., 0.],
                [0., 0.],
                [1.0, 0.],
                [0., 1.0]
        ])

        self.state = self.reset()

    def reset(self, state=None):
        if state is None:
            # random seed for reproducibility
            np.random.seed(0)
            self.state = np.random.uniform(0., 1., size=(4,))
        else:
            assert len(state) == self.num_of_states, f"Reset Input state must be of length: {self.num_of_states}."
            self.state = state.copy()
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