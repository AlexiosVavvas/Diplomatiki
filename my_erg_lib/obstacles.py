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
    def __init__(self, pos, dimensions, obs_type, f_max, f_min=None, min_dist=1e-2, obs_name=None, eps_meters=None, e_max=1):
        """
        Parameters:
            - pos: position of the obstacle
            - dimensions: dimensions of the obstacle (radius, w-h, or normal vector according to the type)
            - obs_type: type of the obstacle (circle, rectangle, wall)
            - f_min: force applied
                o Circle: f @ x = r + eps
                o Rectangle: f @ x = w + eps_x or h + eps_y
                o Wall: f @ x = eps
            - eps_meters: if given, it will be used instead of f_min to calculate the eps
            - f_max: maximum force applied @ x = min_dist
            - min_dist: minimum distance to the obstacle (min dist to avoid dividing by zero)
            - obs_name: name of the obstacle (for debugging)
            - e_max: maximum eps allowed (Used to calculate f_min)

        """
        # Save variables
        self.type = obs_type
        self.pos = np.asarray(pos)
        self.name_id = obs_name

        self.min_dist = min_dist
        self.k_obs = f_max * min_dist**2
        
        assert not (f_min == None and eps_meters is None), f"{obs_name}: Either a minimum force or a distance from the obstacle must be provided"
        assert not (f_min is not None and eps_meters is not None), f"{obs_name}: Only one of <f_min> or <eps_meters> can be provided at any time"

        dimensions = np.asarray(dimensions) if isinstance(dimensions, (list, tuple, np.ndarray)) else np.array([dimensions])
        if obs_type == 'circle':
            assert dimensions.size == 1 and dimensions > 0, "Circle obstacle must have only one dimension (radius) > 0"
            self.r = dimensions[0]

            # If given minimum force, lets calculate the distance we have it
            if f_min is not None:
                self.eps = -self.r + np.sqrt(self.k_obs/f_min)  # eps: [m] away from the wall
                if self.eps > e_max:
                    f"FMIN might be too small. FMIN > {self.k_obs/((self.eps + self.r)**2):.2e} \t[{obs_name} - eps={self.eps:.2f} > {e_max:.2f}]"
            # Otherwise lets go directly to the eps percentage and recalculate f_min there
            elif eps_meters is not None:
                # If the user has given us a number in meters, we dont care about fmin
                assert eps_meters > 0, f"eps_meters additional distance must be greater than 0 (obs_name={obs_name})"
                self.eps = eps_meters
                f_min = self.k_obs/((self.eps + self.r)**2)
            else:
                raise ValueError("Either a minimum force or a distance from the obstacle must be provided")

        elif obs_type == 'rectangle':
            assert len(dimensions) == 2 and dimensions.all(), "Rectangle obstacle must have two dimensions (width, height) > 0"
            self.width = dimensions[0]
            self.height = dimensions[1]
            self.bottom_left = self.pos - np.array([self.width / 2, self.height / 2])

            if f_min is not None:
                self.eps_x = -self.width/2  + np.sqrt(self.k_obs/f_min)    # eps: [m] away from the wall
                self.eps_y = -self.height/2 + np.sqrt(self.k_obs/f_min)   # eps: [m] away from the wall
            
                if self.eps_x > e_max:
                    print(f"{obs_name}: FMIN might be too small. FMIN > {(self.k_obs/((self.eps + self.width/2)**2)):.2e} \t[{obs_name} - eps_x={self.eps_x:.2f} > {e_max:.2f}")
                if self.eps_y > e_max: 
                    print(f"{obs_name}: FMIN might be too small. FMIN > {(self.k_obs/((self.eps + self.height/2)**2)):.2e} \t[{obs_name} - eps_y={self.eps_y:.2f} > {e_max:.2f}")
            
            elif eps_meters is not None:
                self.eps_x = eps_meters
                self.eps_y = eps_meters
                f_min_x = self.k_obs/((self.eps_x + self.width/2 )**2)
                f_min_y = self.k_obs/((self.eps_y + self.height/2)**2)
                f_min = max(f_min_x, f_min_y)

            else:
                raise ValueError("Either a minimum force or a distance from the obstacle must be provided")
            


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

            if f_min is not None:
                self.eps = np.sqrt(self.k_obs/f_min) # eps: [m] away from the wall
                if self.eps > e_max: 
                    print(f"{obs_name}: FMIN might be  too small. FMIN > {self.k_obs/(e_max)**2:.2e} \t[{obs_name} - eps={self.eps:.2f} > {e_max:.2f} (more than {e_max:.3f}[m] from the wall)]")
            
            elif eps_meters is not None:
                self.eps = eps_meters
                f_min = self.k_obs/(self.eps**2)
            
            else:
                raise ValueError("Either a minimum force or a distance from the obstacle must be provided")
            
        else:
            raise ValueError("Obstacle type must be either 'circle', 'rectangle' or 'wall'")

        # Make sure we have the right format
        assert len(self.pos) == 2, "Obstacle position must be a 2D vector for now"

        # Debug print
        if self.type == 'circle':
            print(f"Obstacle: {obs_name} \t- k: {self.k_obs:.2e} \t- e: {self.eps:.2f}[m] \t- type: {self.type} \t- Pos: {self.pos} \t- Dim: {dimensions} \t\t f_min: {f_min:.2e} ")
        elif self.type == 'wall':
            print(f"Obstacle: {obs_name} \t- k: {self.k_obs:.2e} \t- e: {self.eps:.2f}[m] \t- type: {self.type} \t- Pos: {self.pos} \t\t- Normal: {dimensions} \t f_min: {f_min:.2e} ")
        elif self.type == 'rectangle':
            e_ = np.array([self.eps_x, self.eps_y])
            print(f"Obstacle: {obs_name} \t- k: {self.k_obs:.2e} \t- e: ({e_[0]:.2f}[m], {e_[1]:.2f}[m]) \t- type: {self.type} \t- Pos: {self.pos} \t- Dim: {dimensions} \t f_min: {f_min:.2e} ")

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

def ObstacleAvoidanceControllerGenerator(agent: Agent, obs_list, func_name=None):
    # Make sure the list is not empty
    assert len(obs_list) > 0, "Obstacle list is empty. Please provide a list of obstacles."
    # Make sure the obstacles are of type Obstacle
    for obstacle in obs_list:
        assert isinstance(obstacle, Obstacle), "Obstacle list must contain instances of the Obstacle class."
    
    # Lets append obstacles to the agent list
    for obstacle in obs_list:
        agent.obstacle_list.append(obstacle)

    # If we are playing with a Quadcopter we have to calculate an additional LQR gain for obstacle avoidance
    if isinstance(agent.model, model_dynamics.Quadcopter):
        K_LQR_obs = agent.model._calculateLqrControlGain(agent.model.Q_obs, agent.model.R)
        K_LQR_def = agent.model.k_lqr.copy()

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
            
                if dist <= obstacle.r + obstacle.eps:
                    # If the agent is within the obstacle radius, apply a control action to move away from it
                    f += (x[:2] - obstacle.pos) / (dist**3) * obstacle.k_obs

            elif obstacle.type == 'rectangle':
                # Calculate the distance to the rectangle (X direction)
                dist_x_abs = np.abs(x[0] - obstacle.pos[0])
                dist_x_abs = obstacle.min_dist if dist_x_abs < obstacle.min_dist else dist_x_abs

                # Calculate the distance to the rectangle (Y direction)
                dist_y_abs = np.abs(x[1] - obstacle.pos[1])
                dist_y_abs = obstacle.min_dist if dist_y_abs < obstacle.min_dist else dist_y_abs

                within_obs_reach_x = (dist_x_abs <= obstacle.width/2  + obstacle.eps_x) and (np.abs(x[1] - obstacle.pos[1]) <= obstacle.height/2 + 0.5 * obstacle.height/2)
                within_obs_reach_y = (dist_y_abs <= obstacle.height/2 + obstacle.eps_y) and (np.abs(x[0] - obstacle.pos[0]) <= obstacle.width/2  + 0.5 * obstacle.width/2)
                # within_obs_reach_x = (dist_x_abs <= obstacle.width/2  + obstacle.eps_x) and (np.abs(x[1] - obstacle.pos[1]) <= obstacle.height/2)
                # within_obs_reach_y = (dist_y_abs <= obstacle.height/2 + obstacle.eps_y) and (np.abs(x[0] - obstacle.pos[0]) <= obstacle.width/2 )

                if within_obs_reach_x:
                    # Force in the X direction
                    if x[0] < obstacle.pos[0]:
                        f[0] += -1 / (dist_x_abs**2) * obstacle.k_obs
                    else:
                        f[0] += +1 / (dist_x_abs**2) * obstacle.k_obs

                if within_obs_reach_y:
                    # Force in the Y direction
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
        if isinstance(agent.model, model_dynamics.Quadcopter):
            # Forces show desired direction. In a quad we cant translate forces to inputs. We need to use them as LQR velocities target
            # rotate forces to the body frame
            sy = np.sin(agent.model.state[3])  # Rotate yaw to get from world frame to body frame # TODO: Is this correct?
            cy = np.cos(agent.model.state[3])  # Rotate yaw to get from world frame to body frame
            r_ = np.array([[cy, -sy], [sy, cy]])
            f = r_ @ f
            agent.model.f_command_to_controller = f

            # [x,  y,  z,  psi, theta, phi, x',  y',  z',  psidot, thetadot, phidot]
            if f[0] != 0 or f[1] != 0:
                # If we have to avoid an obstacle, we need better velocity tracking
                agent.model._state_target[6] += float(f[0])  # x'
                agent.model._state_target[7] += float(f[1])  # y'
                agent.model.k_lqr = K_LQR_obs
                agent.model.state_target_modified = True
            else: 
                # Otherwise, we can use the default LQR gains
                # If someone needs to avoid obstacles, we have to use the obstacle avoidance gains
                agent.model.k_lqr = K_LQR_obs if agent.model.state_target_modified else K_LQR_def
            
            u = np.zeros((agent.model.num_of_inputs,))
            
        else:
            # Single and Double integrators work fine with forces X and Y as inputs directly (scaled accordingly)
            u = agent.model.convertForcesToInputs(f)

        # Return control action
        return u
    
    # Lets set the function name
    if func_name is not None:
        assert isinstance(func_name, str), "Function name must be a string"
        obstacle_avoidance_control.__name__ = func_name
    else:
        obstacle_avoidance_control.__name__ = "None"

    return obstacle_avoidance_control