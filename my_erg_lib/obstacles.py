import numpy as np
from my_erg_lib.agent import Agent

class Obstacle():
    """
    Implements basic potential field obstacles
        - Circle
        - Rectangle
        - Wall
    The obstacles are defined in a 2D space (for now)
    """
    def __init__(self, pos, dimensions, obs_type, f_min, f_max, min_dist=1e-2, obs_name=None, e_max=1):
        # Save variables
        self.type = obs_type
        self.pos = np.asarray(pos)
        self.name_id = obs_name

        self.min_dist = min_dist
        self.k_obs = f_max * min_dist**2
        
        dimensions = np.asarray(dimensions) if isinstance(dimensions, (list, tuple, np.ndarray)) else np.array([dimensions])
        if obs_type == 'circle':
            assert dimensions.size == 1 and dimensions > 0, "Circle obstacle must have only one dimension (radius) > 0"
            self.r = dimensions[0]
            self.eps = -1 + 1/self.r*np.sqrt(self.k_obs/f_min)  # eps: % of the radius more
            assert self.eps <= e_max, f"FMIN is too small. FMIN = {(self.k_obs/((1+e_max)**2 * self.r**2)):.2e} \t[{obs_name} - eps={self.eps:.2f} > {e_max:.2f} (more than {e_max:.1%} of the radius)]"

        elif obs_type == 'rectangle':
            assert len(dimensions) == 2 and dimensions.all(), "Rectangle obstacle must have two dimensions (width, height) > 0"
            self.width = dimensions[0]
            self.height = dimensions[1]
            self.bottom_left = self.pos - np.array([self.width / 2, self.height / 2])
            self.eps_x = -1 + 1/self.width*np.sqrt(self.k_obs/f_min)    # eps: % of the width  more
            self.eps_y = -1 + 1/self.height*np.sqrt(self.k_obs/f_min)   # eps: % of the height more

            assert self.eps_x <= e_max, f"FMIN is too small. FMIN = {(self.k_obs/((1+e_max)**2 * self.width**2)):.2e} \t[{obs_name} - eps_x={self.eps_x:.2f} > {e_max:.2f} (more than {e_max:.1%} of the width)]"
            assert self.eps_y <= e_max, f"FMIN is too small. FMIN = {(self.k_obs/((1+e_max)**2 * self.height**2)):.2e} \t[{obs_name} - eps_y={self.eps_y:.2f} > {e_max:.2f} (more than {e_max:.1%} of the height)]"


        elif obs_type == 'wall':
            """ 
            Wall Object is an infinite line that restricts the agent to only the one side
            A wall object is defined by a point and a normal vector
            The normal vector is the direction of the permitted side
            The wall is defined by the equation: (x - p) . n = 0
            where: 
                - x is the point on the wall
                - p is the point defining the wall
                - n is the normal vector
            Equation: 
                (x - x0) nx + (y - y0) ny = 0
            Example:
                Horizontal wall: n = [0, 1], p = [x0, y0]
                Vertical wall:   n = [1, 0], p = [x0, y0]
            Parameters:
                - pos: point defining the wall
                - dimensions: normal vector of the wall
            """
            p = self.pos
            n = np.asarray(dimensions)
            assert n.size == 2 and np.linalg.norm(n) > 0, "Wall obstacle must have a normal vector of size 2 and non-zero length"
            n = n / np.linalg.norm(n)
            self.n = n

            self.eps = np.sqrt(self.k_obs/f_min) # eps: [m]
            assert self.eps <= e_max, f"FMIN is too small. FMIN = {self.k_obs/(e_max)**2:.2e} \t[{obs_name} - eps={self.eps:.2f} > {e_max:.2f} (more than {e_max:.3f}[m] from the wall)]"
            
        else:
            raise ValueError("Obstacle type must be either 'circle' or 'rectangle'")

        # Make sure we have the right format
        assert len(self.pos) == 2, "Obstacle position must be a 2D vector for now"

        # Debug print
        if self.type == 'circle':
            print(f"Obstacle: {obs_name} \t- k: {self.k_obs:.2e} \t- e: {self.eps:.2%} \t- type: {self.type} \t- Pos: {self.pos} \t- Dim: {dimensions}")
        elif self.type == 'wall':
            print(f"Obstacle: {obs_name} \t- k: {self.k_obs:.2e} \t- e: {self.eps:.2f}[m] \t- type: {self.type} \t- Pos: {self.pos} \t\t- Normal: {dimensions}")
        elif self.type == 'rectangle':
            e_ = np.array([self.eps_x, self.eps_y])
            print(f"Obstacle: {obs_name} \t- k: {self.k_obs:.2e} \t- e: ({e_[0]:.2e}, {e_[1]:.2e}) \t- type: {self.type} \t- Pos: {self.pos} \t- Dim: {dimensions}")

    def distanceToTheWall(self, x):
        """
        Returns the distance to the wall
        """
        if self.type == 'wall':
            # Wall equation: (x - p) . n = 0
            # Distance to the wall: d = (x - p) . n
            return np.dot(x - self.pos, self.n)
        
        else:
            raise ValueError("Distance to wall is only available for wall obstacles")
        

def ObstacleAvoidanceControllerGenerator(agent: Agent, obs_list):
    # Make sure the list is not empty
    assert len(obs_list) > 0, "Obstacle list is empty. Please provide a list of obstacles."
    # Make sure the obstacles are of type Obstacle
    for obstacle in obs_list:
        assert isinstance(obstacle, Obstacle), "Obstacle list must contain instances of the Obstacle class."
    
    # Lets append obstacles to the agent list
    for obstacle in obs_list:
        agent.obstacle_list.append(obstacle)

    def obstacle_avoidance_control(x, t):
        """
        Obstacle avoidance control function.
        This function calculates the control action to avoid obstacles.
        """
        # Forces in the global X, Y direction
        f = np.zeros((2,))

        # Iterate over all obstacles
        for obstacle in obs_list:
            # Check the type of the obstacle
            if obstacle.type == 'circle':
                # Calculate the distance to the obstacle
                dist = np.linalg.norm(x[:2] - obstacle.pos)
                dist = obstacle.min_dist if dist < obstacle.min_dist else dist
            
                if dist <= obstacle.r * (1 + obstacle.eps):
                    # If the agent is within the obstacle radius, apply a control action to move away from it
                    f += (x[:2] - obstacle.pos) / (dist**3) * obstacle.k_obs

            elif obstacle.type == 'rectangle':
                # Calculate the distance to the rectangle (X direction)
                dist_x_abs = np.abs(x[0] - obstacle.pos[0])
                dist_x_abs = obstacle.min_dist if dist_x_abs < obstacle.min_dist else dist_x_abs
                
                if dist_x_abs <= obstacle.width / 2 * (1 + obstacle.eps_x):
                    if x[0] < obstacle.pos[0]:
                        f[0] += -1 / (dist_x_abs**2) * obstacle.k_obs
                    else:
                        f[0] += +1 / (dist_x_abs**2) * obstacle.k_obs

                # Calculate the distance to the rectangle (Y direction)
                dist_y_abs = np.abs(x[1] - obstacle.pos[1])
                dist_y_abs = obstacle.min_dist if dist_y_abs < obstacle.min_dist else dist_y_abs

                if dist_y_abs <= obstacle.height / 2 * (1 + obstacle.eps_y):
                    if x[1] < obstacle.pos[1]:
                        f[1] += -1 / (dist_y_abs**2) * obstacle.k_obs
                    else:
                        f[1] += +1 / (dist_y_abs**2) * obstacle.k_obs

            elif obstacle.type == 'wall':
                # Calculate the distance to the wall
                dist = obstacle.distanceToTheWall(x[:2]) # dist can be negative
                dist = obstacle.min_dist if dist < obstacle.min_dist else dist # Make sure we have a positive distance
                
                if dist <= obstacle.eps:
                    # If the agent is within the wall distance, apply a control action to move away from it
                    dx = dist * obstacle.n

                    f += dx / (dist**3) * obstacle.k_obs
            
        # Lets translate Fx and Fy to the control space
        u = agent.model.convertForcesToInputs(f)
        # if f[0] != 0 or f[1] != 0 or u[0] != 0 or u[1] != 0:
        #     print(f"f = {f}, u = {u}")

        # Return control action
        return u
    
    return obstacle_avoidance_control