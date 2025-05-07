import numpy as np

class SingleIntegrator():

    def __init__(self):
        self.dt = 0.001

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
    


class DoubleIntegrator():

    def __init__(self):
        self.dt = 0.001

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