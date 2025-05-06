from basis import Basis
import numpy as np
from model_dynamics import SingleIntegrator

class Agent():
    def __init__(self, L1, L2, Kmax, dynamics_model: SingleIntegrator, phi=None, Ts=0.01, uNominal=None, T_horizon=1.0):
        self.L1 = L1
        self.L2 = L2
        self.Kmax = Kmax
        
        self.model = dynamics_model
        self.model.reset()
        self.Ts = Ts

        # Initialize the basis object
        self.basis = Basis(L1, L2, Kmax)

        # Set the target distribution (phi)
        self.setPhi(phi)  # Set the target distribution in the basis object

        # Initialise Actions
        self.ustar_actions = np.zeros(int(T_horizon / Ts), dtype=np.float64)
        


    def setPhi(self, phi=None):
        """Set the target distribution."""
        self._phi = phi if phi is not None else lambda s: 2  # Default to constant 0 function if not provided
        self.basis.phi = self._phi


    def simulateForward(self, x0, ti, udef=None, T=1.0):
        """
        Simulate the system forward in time using the dynamics model.'
        From ti -> ti+T
        """
        t = ti
        x = x0.copy()
        trajectory = [x.copy()]
        
        # Check for callable udef
        assert callable(udef) or udef is None, "udef must be a callable function or None."
        if udef != None:
            udef_ = udef(x0, t)
        else:
            # Default udef is a zero vector of the same size as the state
            udef_ = np.zeros((2,))

        # Reset the model with the initial state and simulate forward
        self.model.reset(x0)
        while t < ti + T:
            udef_ = udef(x, t) if callable(udef) else udef_
            x = self.model.step(udef_)
            trajectory.append(x.copy())
            t += self.model.dt  # Increment time by the model's time step
        
        self.model.reset(x0)  # Reset the model to the initial state after simulation
        return np.array(trajectory)
    
    
    def simulateAdjointBackward(self, x_traj, ck, T=1.0, Q=1, num_of_agents=1):
        '''
        Simulate the adjoint state to get rho(t)
        '''
        rho = np.zeros((len(x_traj), 2))

        for i in range(len(x_traj)-2, -1, -1):
            # Jacobian of the dynamics with respect to x
            f_x = self.model.f_x(x_traj[i])
            
            rho_dot = -np.dot(f_x.T, rho[i+1])  # TOCHECK: f_x.T
            for k1 in range(self.Kmax+1):
                for k2 in range(self.Kmax+1):
                    lamda_k = self.basis.LamdaK_cache[(k1, k2)]
                    hk = self.basis.calcHk(k1, k2)
                    phi_k = self.basis.calcPhikCoeff(k1, k2)
                    rho_dot += (-2 * Q / T / num_of_agents) * lamda_k * (ck[k1, k2] - phi_k) * self.basis.dFk_dx(x_traj[i], k1, k2, hk)

            # Update rho using the computed rho_dot
            rho[i] = rho[i+1] - rho_dot * self.model.dt 

            def plotRho(rho):
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 5))
                plt.plot(rho[:, 0], 'o-', label='Rho 1')
                plt.plot(rho[:, 1], 'o-', label='Rho 2')
                plt.title('Adjoint State Rho over Time')
                plt.xlabel('Time Step')
                plt.ylabel('Rho Value')
                plt.legend()
                plt.grid()
                plt.show()
            # plotRho(rho)

        return rho, np.linspace(0, T, len(x_traj))

    def visualiseColorForTraj(self, ck, x_traj):
        '''
        Visualise the color for the trajectory using ck coefficients
        '''
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from basis import ReconstructedPhiFromCk
        # Reconstruct the target distribution using the coefficients
        phi_from_ck = ReconstructedPhiFromCk(self.basis, ck)

        x1 = np.linspace(0, 1, 50)
        x2 = np.linspace(0, 1, 50)


        Z_reconstructed = np.zeros((len(x1), len(x2)))

        for i in range(len(x1)):
            for j in range(len(x2)):
                Z_reconstructed[i, j] = phi_from_ck([x1[i], x2[j]])
            
        plt.figure(figsize=(6, 6))
        plt.imshow(Z_reconstructed.T, extent=(0, 1, 0, 1), origin='lower', cmap=cm.viridis)
        plt.title(f'Reconstructed Function (Kmax={self.Kmax})')
        plt.xlabel('x1')
        plt.ylabel('x2')

        if x_traj is not None:
            plt.plot(x_traj[:, 0], x_traj[:, 1], 'r-', label='Trajectory')

        plt.tight_layout()
        plt.show()

    def updateActionsWindow(self, us, tau, lamda_duration, ti):
        for i in range(len(self.ustar_actions)):
            pass





