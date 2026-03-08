import numpy as np
import my_erg_lib.model_dynamics as model_dynamics
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from my_erg_lib.agent import Agent

class Obstacle():
    """
    Implements basic potential field obstacles in 2D/3D:
        - Circle (2D)
        - Sphere (3D)
        - Rectangle (2D)
        - Wall / plane (3D)
    2D ones are infinite in the z direction.
    """
    def __init__(self, pos, dimensions, kappa, obs_type, rho0, r0=None, obs_name=None):
        """
        Parameters:
            - pos:        obstacle reference point (2D for circle/rectangle, 3D for sphere/wall)
            - dimensions: geometry descriptor
                o circle (2D):      radius
                o sphere (3D):      radius
                o rectangle (2D):   width, height
                o wall (3D):        normal vector; length only used for plotting wall extent
            - kappa:    potential field strength parameter
            - obs_type: obstacle type ('circle', 'sphere', 'rectangle', 'wall')
            - rho0:     obstacle vicinity distance measured from the obstacle boundary
            - r0:       optional safety radius for circle/sphere (defaults to radius when omitted)
            - obs_name: identifier for debugging/logging
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
            assert len(self.pos) == 2, "Circle obstacle position must be a 2D vector"
            self.r = dimensions[0]
            self.r0 = r0 if r0 is not None else self.r  # r0 is the radius used to calculation of pot fields. Could be greater than r, for safety.

            # Lets define distance and gradient functions
            # Distance function: ρ(x) = ||x - pos|| - r0
            def _rhoFunc(x):
                return np.linalg.norm(x[:2] - self.pos) - self.r0
            self.rhoFunc = _rhoFunc

            # Gradient function: ∇ρ(x) = (x - pos) / ||x - pos||
            def _gradRhoFunc(x):
                norm = np.linalg.norm(x[:2] - self.pos)
                if norm < 1e-8:  # Avoid division by zero
                    return np.zeros(3)
                return np.append((x[:2] - self.pos) / norm, 0)
            self.gradRhoFunc = _gradRhoFunc

        elif obs_type == 'sphere':
            assert dimensions.size == 1 and dimensions > 0, "Circle obstacle must have only one dimension (radius) > 0"
            assert len(self.pos) == 3, "Sphere obstacle position must be a 3D vector"
            self.r = dimensions[0]
            self.r0 = r0 if r0 is not None else self.r  # r0 is the radius used to calculation of pot fields. Could be greater than r, for safety.

            # Lets define distance and gradient functions
            # Distance function: ρ(x) = ||x - pos|| - r0
            def _rhoFunc(x):
                return np.linalg.norm(x[:3] - self.pos) - self.r0
            self.rhoFunc = _rhoFunc

            # Gradient function: ∇ρ(x) = (x - pos) / ||x - pos||
            def _gradRhoFunc(x):
                norm = np.linalg.norm(x[:3] - self.pos)
                if norm < 1e-8:  # Avoid division by zero
                    return np.zeros(3)
                return (x[:3] - self.pos) / norm
            self.gradRhoFunc = _gradRhoFunc

        elif obs_type == 'rectangle':
            assert len(dimensions) == 2 and dimensions.all(), "Rectangle obstacle must have two dimensions (width, height) > 0"
            assert len(self.pos) == 2, "Rectangle obstacle position must be a 2D vector"
            self.width = dimensions[0]
            self.height = dimensions[1]
            self.bottom_left = self.pos - np.array([self.width / 2, self.height / 2])

            self.W = np.array([self.width/2, self.height/2])  # Half-width and half-height for easier calculations later
            
            # Lets define distance and gradient functions
            # Distance function
            def _rhoFunc(x):
                E = np.abs(x[:2] - self.pos) - self.W
                maxE = np.max(E)
                if maxE >= 0:
                    return maxE
                else:
                    return -np.min(-E)
            self.rhoFunc = _rhoFunc

            # Gradient function
            def _gradRhoFunc(x):
                E = np.abs(x[:2] - self.pos) - self.W
                if np.max(E) >= 0:
                    # outside or on: gradient points in the direction
                    # of the coordinate that attained the max
                    if E[0] >= E[1]:
                        return np.array([np.sign(x[0] - self.pos[0]), 0, 0])
                    else:
                        return np.array([0, np.sign(x[1] - self.pos[1]), 0])
                else:
                    # inside: gradient also points to the nearest side
                    G = - E
                    if G[0] <= G[1]:
                        return np.array([np.sign(x[0] - self.pos[0]), 0, 0])
                    else:
                        return np.array([0, np.sign(x[1] - self.pos[1]), 0])
            self.gradRhoFunc = _gradRhoFunc


        elif obs_type == 'wall':
            """ 
            Wall object is an infinite plane that restricts the agent to one side.
            Defined by a point p on the plane and a normal vector n that points to
            the permitted half-space. In 3D the plane equation is: (x - p) . n = 0
            where:
                - x is any point on the plane
                - p is the point defining the plane
                - n is the plane normal (size 3)
            3D examples:
                - Horizontal plane at z = z0: n = [0, 0, 1], p = [x0, y0, z0]
                - Vertical plane:             n = [1, 0, 0], p = [x0, y0, z0]
            Parameters:
                - pos: point defining the plane
                - dimensions: normal vector of the plane
            """
            n = np.asarray(dimensions)
            assert n.size == 3 and np.linalg.norm(n) > 0, "Wall obstacle must have a normal vector of size 3 and non-zero length"
            assert len(self.pos) == 3, "Wall obstacle position must be a 3D vector"
            self.wall_length = np.linalg.norm(n[:2])  # For plotting purposes only
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
            raise ValueError("Obstacle type must be either 'circle', 'sphere', 'rectangle' or 'wall'")

        # Make sure we have the right format
        assert len(self.pos) <= 3, "Obstacle position must be a 2D or 3D vector for now"

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
            return np.dot(x[:3] - self.pos, self.n)
        
        else:
            raise ValueError("Distance to wall is only available for wall obstacles")
        
    def withinReach(self, x):
        """
        Check if the agent is within reach of the obstacle
        """
        if len(x) == 2:
            # append a zero z-coordinate for 2D states
            x = np.append(x, 0)
        assert len(x) == 3, "State x must be a 2D or 3D vector"

        if self.type == 'circle':
            return np.linalg.norm(x[:2] - self.pos) <= self.r
        
        elif self.type == 'sphere':
            return np.linalg.norm(x[:3] - self.pos) <= self.r

        elif self.type == 'rectangle':
            # Check if the agent is within the rectangle
            return (np.abs(x[0] - self.pos[0]) <= self.width/2) and (np.abs(x[1] - self.pos[1]) <= self.height/2)

        elif self.type == 'wall':
            # Check if the agent is within the wall distance
            return self.distanceToTheWall(x[:3]) <= 0

        else:
            raise ValueError("Obstacle type must be either 'circle', 'sphere', 'rectangle' or 'wall'")
        

    def U(self, x):
        """
        Potential function for the obstacle avoidance.
        """
        rho = self.rhoFunc(x[:3])

        # Quick distance check before calculation
        if rho >= self.rho0:
            # U = 0 for rho >= rho0
            return 0.0
        elif rho == 0:
            rho = 1e-6  # Avoid division by zero
        elif rho < 0:
            return np.inf  # If the agent is inside the obstacle, return infinity

        return 0.5 * self.kappa * (1 / rho - 1 / self.rho0) ** 2

    # def gradU(self, x):
    #     """
    #     Gradient of the potential function for the obstacle avoidance.
    #     """
    #     rho = self.rhoFunc(x[:2])
    #     rho = rho if rho != 0 else 1e-6  # Avoid division by zero
    #     # Calculate ∇ρ
    #     if rho >= self.rho0:
    #         return np.zeros(2) # ∇ρ = 0 -> ∇U = 0
    #     else: 
    #         grad_rho = self.gradRhoFunc(x[:2])

    #     return -self.kappa / (rho**2) * (1 / rho - 1 / self.rho0) * grad_rho
    
    def UandGradU(self, x):
        """
        Returns both the potential function and its gradient for the obstacle avoidance.
        More efficient than calling U() and gradU() separately as it computes rho only once.
        """
        rho = self.rhoFunc(x[:3])

        # Quick distance check before calculation
        if rho >= self.rho0:
            # U = 0 for rho >= rho0
            return 0.0, np.zeros(3) # ∇ρ = 0 -> ∇U = 0
        
        if rho <= 0:
            # Agent is inside the obstacle
            rho = 1e-6  # Avoid division by zero in gradient calculation
            grad_rho = self.gradRhoFunc(x[:3])
            return np.inf, grad_rho
        
        # Normal case: 0 < rho < rho0
        ratio = 1 / rho - 1 / self.rho0
        U = 0.5 * self.kappa * ratio * ratio # ratio ** 2 but avoids python overhead for exponentiation
        
        grad_rho = self.gradRhoFunc(x[:3])
        grad_U = -self.kappa / (rho * rho) * ratio * grad_rho

        return U, grad_U

def saveObstaclesToMemory(agent: "Agent", obs_list):
    # Make sure the list is not empty
    assert len(obs_list) > 0, "Obstacle list is empty. Please provide a list of obstacles."

    # Make sure the obstacles are of type Obstacle
    for obstacle in obs_list:
        assert isinstance(obstacle, Obstacle), "Obstacle list must contain instances of the Obstacle class."

    # Lets append obstacles to the agent list
    for obstacle in obs_list:
        agent.obstacle_list.append(obstacle)

def removeObstaclesFromMemory(agent: "Agent", obs_name_list):
    agent.obstacle_list = [obs for obs in agent.obstacle_list if obs.name_id not in obs_name_list]

def updateObstaclePositionInMemory(agent: "Agent", obs_name, new_pos):
    for obs in agent.obstacle_list:
        if obs.name_id == obs_name:
            obs.pos = np.asarray(new_pos)
            return
    # If we reach here, the obstacle was not found
    agent.get_logger().warning(f"Obstacle with name {obs_name} not found in memory. Cannot update position. (new_pos: {new_pos})")
