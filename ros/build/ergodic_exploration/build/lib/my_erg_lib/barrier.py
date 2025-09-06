import numpy as np

# TODO: Remove barrier class, we dont need it anymore
class Barrier:
    '''
    This class prevents the agent from
    going outside the exploration space
    '''
    def __init__(self, L1, L2, pow_=2, weight=100, eps_=0.01):
        self.pow = pow_
        self.weight = weight
        self.eps = eps_

        self.space_top_lim = np.array([L1, L2])
        self.space_bot_lim = np.array([0, 0])

        assert pow_ >= 1, "Power must be greater than or equal to 1."

    def cost(self, x):
        '''
        Returns the actual cost of the barrier
        '''
        cost = 0.
        cost += np.sum((x > self.space_top_lim-self.eps) * (x - (self.space_top_lim-self.eps))**self.pow)
        cost += np.sum((x < self.space_bot_lim+self.eps) * (x - (self.space_bot_lim+self.eps))**self.pow)
        return self.weight * cost

    def dx(self, x):
        '''
        Returns the derivative of the barrier wrt to the exploration
        state for a trajectory of points with shape (N, 2)
        '''
        # Initialize zeros with same shape as input
        dx = np.zeros_like(x)
        
        # Apply conditions element-wise across all points in trajectory
        dx += self.pow * (x > (self.space_top_lim-self.eps)) * (x - (self.space_top_lim-self.eps))**(self.pow-1)
        dx += self.pow * (x < (self.space_bot_lim+self.eps)) * (x - (self.space_bot_lim+self.eps))**(self.pow-1)
        
        return self.weight * dx
