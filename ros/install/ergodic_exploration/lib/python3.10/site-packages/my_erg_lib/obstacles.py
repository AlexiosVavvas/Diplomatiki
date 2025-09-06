import numpy as np
from my_erg_lib.agent import Agent
import my_erg_lib.model_dynamics as model_dynamics

class Obstacle():
    """
    Implements basic potential field obstacles
        - Circle
        - Rectangle
        - Wall
    The obstacles are defined in a 2D space (for now)
    """
    def __init__(self, pos, dimensions, kappa, obs_type, rho0, r0=None, obs_name=None):
        """
        Parameters:
            - pos: position of the obstacle
            - dimensions: dimensions of the obstacle (radius, w-h, or normal vector according to the type)
            - obs_type: type of the obstacle (circle, rectangle, wall)
            - f_min: force applied
                o Circle: f @ x = r + eps
                o Rectangle: f @ x = w + eps_x or h + eps_y
                o Wall: f @ x = eps
            - obs_name: name of the obstacle (for debugging)
            # TODO: Complete here

        """
        # Save variables
        self.type = obs_type
        self.pos = np.asarray(pos)
        self.name_id = obs_name
        # Potential field parameters
        self.kappa = kappa
        self.rho0 = rho0 # Obstacle vicinity distance (measured from the border of the obstacle and on)

        # Function Placeholders, to be filled later
        self.rhoFunc = None
        self.gradRhoFunc = None

        dimensions = np.asarray(dimensions) if isinstance(dimensions, (list, tuple, np.ndarray)) else np.array([dimensions])
        if obs_type == 'circle':
            assert dimensions.size == 1 and dimensions > 0, "Circle obstacle must have only one dimension (radius) > 0"
            self.r = dimensions[0]
            self.r0 = r0 if r0 is not None else self.r  # r0 is the radius used to calculation of pot fields. Could be greater than r, for safety.

            # Lets define distance and gradient functions
            # Distance function: ρ(x) = ||x - pos|| - r0
            def _rhoFunc(x):
                return np.linalg.norm(x - self.pos) - self.r0
            self.rhoFunc = _rhoFunc

            # Gradient function: ∇ρ(x) = (x - pos) / ||x - pos||
            def _gradRhoFunc(x):
                return (x - self.pos) / np.linalg.norm(x - self.pos)
            self.gradRhoFunc = _gradRhoFunc

        elif obs_type == 'rectangle':
            assert len(dimensions) == 2 and dimensions.all(), "Rectangle obstacle must have two dimensions (width, height) > 0"
            self.width = dimensions[0]
            self.height = dimensions[1]
            self.bottom_left = self.pos - np.array([self.width / 2, self.height / 2])

            self.W = np.array([self.width/2, self.height/2])  # Half-width and half-height for easier calculations later
            
            # Lets define distance and gradient functions
            # Distance function
            def _rhoFunc(x):
                E = np.abs(x - self.pos) - self.W
                maxE = np.max(E)
                if maxE >= 0:
                    return maxE
                else:
                    return -np.min(-E)
            self.rhoFunc = _rhoFunc

            # Gradient function
            def _gradRhoFunc(x):
                E = np.abs(x - self.pos) - self.W
                if np.max(E) >= 0:
                    # outside or on: gradient points in the direction
                    # of the coordinate that attained the max
                    if E[0] >= E[1]:
                        return np.array([np.sign(x[0] - self.pos[0]), 0])
                    else:
                        return np.array([0, np.sign(x[1] - self.pos[1])])
                else:
                    # inside: gradient also points to the nearest side
                    G = - E
                    if G[0] <= G[1]:
                        return np.array([np.sign(x[0] - self.pos[0]), 0])
                    else:
                        return np.array([0, np.sign(x[1] - self.pos[1])])
            self.gradRhoFunc = _gradRhoFunc


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
            n = np.asarray(dimensions)
            assert n.size == 2 and np.linalg.norm(n) > 0, "Wall obstacle must have a normal vector of size 2 and non-zero length"
            n = n / np.linalg.norm(n)
            self.n = n

            # Lets define distance and gradient functions
            # Distance function: ρ(x) = (x - p) . n
            def _rhoFunc(x):
                return np.dot(x - self.pos, self.n)
            self.rhoFunc = _rhoFunc

            # Gradient function: ∇ρ(x) = n
            def _gradRhoFunc(x):
                return self.n
            self.gradRhoFunc = _gradRhoFunc

        else:
            raise ValueError("Obstacle type must be either 'circle', 'rectangle' or 'wall'")

        # Make sure we have the right format
        assert len(self.pos) == 2, "Obstacle position must be a 2D vector for now"

        # Debug print
        # if self.type == 'circle':
        #     print(f"Obstacle: {obs_name} \t- type: {self.type} \t- Pos: {self.pos} \t- Dim: {dimensions} \t- K: {self.kappa} \t- rho_0: {self.rho0} \t- R0: {self.r0}")
        # elif self.type == 'wall':
        #     print(f"Obstacle: {obs_name} \t- type: {self.type} \t- Pos: {self.pos} \t\t- Normal: {dimensions} \t- K: {self.kappa} \t- rho_0: {self.rho0}")
        # elif self.type == 'rectangle':
        #     print(f"Obstacle: {obs_name} \t- type: {self.type} \t- Pos: {self.pos} \t- Dim: {dimensions} \t- K: {self.kappa} \t- rho_0: {self.rho0}")

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
        
    def withinReach(self, x):
        """
        Check if the agent is within reach of the obstacle
        """
        assert len(x) == 2, f"{self.name_id}.withinReach(x): Obstacle avoidance is only available for 2D systems. Please provide a 2D state vector x"

        if self.type == 'circle':
            return np.linalg.norm(x[:2] - self.pos) <= self.r

        elif self.type == 'rectangle':
            # Check if the agent is within the rectangle
            return (np.abs(x[0] - self.pos[0]) <= self.width/2) and (np.abs(x[1] - self.pos[1]) <= self.height/2)

        elif self.type == 'wall':
            # Check if the agent is within the wall distance
            return self.distanceToTheWall(x[:2]) <= 0

        else:
            raise ValueError("Obstacle type must be either 'circle', 'rectangle' or 'wall'")

    def returnBoundaryPointsForPlotting(self, num_of_points=100):
        # Return a 2d array of points that define the boundary of the obstacle
        if self.type == 'circle':
            theta = np.linspace(0, 2 * np.pi, num_of_points)
            x = self.pos[0] + self.r * np.cos(theta)
            y = self.pos[1] + self.r * np.sin(theta)
            # Lets also put some points inside the circle uniformly using angle and radious
            r_steps = np.linspace(0, self.r, num_of_points // 2)
            theta_steps = np.linspace(0, 2 * np.pi, num_of_points // 2)
            x_inner = self.pos[0] + r_steps[:, None] * np.cos(theta_steps)
            y_inner = self.pos[1] + r_steps[:, None] * np.sin(theta_steps)
            # Combine outer and inner points
            x = np.concatenate((x, x_inner.flatten()))
            y = np.concatenate((y, y_inner.flatten()))
            return np.column_stack((x, y))

        elif self.type == 'rectangle':
            # Left wall using num_of_points
            x_left = np.linspace(self.bottom_left[0], self.bottom_left[0], num_of_points)
            y_left = np.linspace(self.bottom_left[1], self.bottom_left[1] + self.height, num_of_points)

            x_right = np.linspace(self.bottom_left[0] + self.width, self.bottom_left[0] + self.width, num_of_points)
            y_right = np.linspace(self.bottom_left[1], self.bottom_left[1] + self.height, num_of_points)

            x_top = np.linspace(self.bottom_left[0], self.bottom_left[0] + self.width, num_of_points)
            y_top = np.linspace(self.bottom_left[1] + self.height, self.bottom_left[1] + self.height, num_of_points)

            x_bottom = np.linspace(self.bottom_left[0], self.bottom_left[0] + self.width, num_of_points)
            y_bottom = np.linspace(self.bottom_left[1], self.bottom_left[1], num_of_points)

            # Lets put some points inside the rectangle uniformly
            # Calculate number of points based on rectangle dimensions
            width_points = max(num_of_points // 4, int(num_of_points * self.width / (self.width + self.height)))
            height_points = max(num_of_points // 4, int(num_of_points * self.height / (self.width + self.height)))
            
            x_inner = np.linspace(self.bottom_left[0] + 0.1, self.bottom_left[0] + self.width - 0.1, width_points)
            y_inner = np.linspace(self.bottom_left[1] + 0.1, self.bottom_left[1] + self.height - 0.1, height_points)
            x_inner, y_inner = np.meshgrid(x_inner, y_inner)
            x_inner = x_inner.flatten()
            y_inner = y_inner.flatten()

            # Combine all points
            x = np.concatenate((x_left, x_right, x_top, x_bottom, x_inner))
            y = np.concatenate((y_left, y_right, y_top, y_bottom, y_inner))
            return np.column_stack((x, y))
        
        elif self.type == 'wall':
            # return empty array
            return np.empty((0, 2))  # Wall has no boundary points to plot

    def U(self, x):
        """
        Potential function for the obstacle avoidance.
        """
        rho = self.rhoFunc(x[:2])
        if rho == 0:
            rho = 1e-6  # Avoid division by zero
        elif rho < 0:
            return np.inf  # If the agent is inside the obstacle, return infinity
        elif rho >= self.rho0:
            # U = 0 for rho >= rho0
            rho = self.rho0

        return 0.5 * self.kappa * (1 / rho - 1 / self.rho0) ** 2

    def gradU(self, x):
        """
        Gradient of the potential function for the obstacle avoidance.
        """
        rho = self.rhoFunc(x[:2])
        rho = rho if rho != 0 else 1e-6  # Avoid division by zero
        # Calculate ∇ρ
        if rho >= self.rho0:
            grad_rho = np.zeros_like(x[:2])
        else: 
            grad_rho = self.gradRhoFunc(x[:2])

        return -self.kappa / (rho**2) * (1 / rho - 1 / self.rho0) * grad_rho

def saveObstaclesToMemory(agent: Agent, obs_list):
    # Make sure the list is not empty
    assert len(obs_list) > 0, "Obstacle list is empty. Please provide a list of obstacles."

    # Make sure the obstacles are of type Obstacle
    for obstacle in obs_list:
        assert isinstance(obstacle, Obstacle), "Obstacle list must contain instances of the Obstacle class."

    # Lets append obstacles to the agent list
    for obstacle in obs_list:
        agent.obstacle_list.append(obstacle)
