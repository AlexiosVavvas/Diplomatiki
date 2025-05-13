from my_erg_lib.basis import Basis
import numpy as np
from my_erg_lib.model_dynamics import SingleIntegrator, DoubleIntegrator, Quadcopter

class Agent():
    def __init__(self, L1, L2, Kmax, dynamics_model, phi=None, x0=None, agent_id=None):
        self.agent_id = agent_id

        # Space Parameters
        self.L1 = L1
        self.L2 = L2
        self.Kmax = Kmax
        
        # Connecting model dynamics
        self.model = dynamics_model
        self.model.reset(x0)

        # Initialize the basis object
        self.basis = Basis(L1, L2, Kmax, phi)


    def setPhi(self, phi=None):
        """Set the target distribution."""
        self._phi = phi if phi is not None else lambda s: 2  # Default to constant 0 function if not provided
        self.basis.phi = self._phi

    # Simulates default input and returns full state trajectory
    def simulateForward(self, x0, ti, udef=None, T=1.0, dt=None):
        """
        Simulate the system forward in time using the dynamics model.'
        From ti -> ti+T
        """
        dt = self.model.dt if dt is None else dt
        t = ti
        x = x0.copy()
        x_traj = []
        u_traj = []
        t_traj = []
        
        # Check for callable udef
        assert callable(udef) or udef is None, "udef must be a callable function or None."

        # Reset the model with the initial state and simulate forward
        self.model.reset(x0)
        while t < ti + T:
            udef_ = udef(x, t) if callable(udef) else np.zeros((self.model.num_of_inputs,))
            x = self.model.step(x=x, u=udef_, dt=dt)
            x_traj.append(x.copy())
            u_traj.append(udef_.copy())
            t_traj.append(t)
            t += dt  # Increment time by the model's time step
        
        self.model.reset(x0)  # Reset the model to the initial state after simulation
        return np.array(x_traj), np.array(u_traj), np.array(t_traj)
    
    
    def simulateAdjointBackward(self, x_traj, u_traj, t_traj, ck, T=1.0, Q=1, num_of_agents=1):
        '''
        Simulate the adjoint state to get rho(t)
        Integrating with simple Euler method Backwards from rho(ti+T) = 0
        '''
        dt = t_traj[1] - t_traj[0]  # Time step: Dt is not a variable here. We chose it to be the same as the forward pass
        rho = np.zeros((len(x_traj), self.model.num_of_states))

        # Integrating with simple Euler method Backwards from ρ(ti+T) = 0
        for i in range(len(x_traj)-2, -1, -1):
            # Jacobian of the dynamics with respect to x
            f_x = self.model.f_x(x=x_traj[i], u=u_traj[i])

            rho_dot = -np.dot(f_x.T, rho[i+1])
            
            for k1 in range(self.Kmax+1):
                for k2 in range(self.Kmax+1):
                    # Calculating summation terms
                    lamda_k = self.basis.LamdaK_cache[(k1, k2)]
                    hk = self.basis.calcHk(k1, k2)
                    ck_ = ck[k1, k2]
                    phi_k = self.basis.calcPhikCoeff(k1, k2)
                    dFdx = self.basis.dFk_dx(x_traj[i][:2], k1, k2, hk)
                    # TODO: Check: Since Fk(xv) the derivative lacks dimensions to reach x. So i think we should append 0s
                    dFdx = np.concatenate((dFdx, np.zeros((self.model.num_of_states - 2,))))
                    
                    # Adding to rho_dot(x[i], t[i])
                    rho_dot += (-2 * Q / T / num_of_agents) * lamda_k * (ck_ - phi_k) * dFdx
                    
                    # if we are epsilon close to the barrier, we need to add the barrier term	
                    eps = self.erg_c.barrier.eps
                    x1 = x_traj[i][0]; x1_max = self.erg_c.barrier.space_top_lim[0] - eps; x1_min = eps
                    x2 = x_traj[i][1]; x2_max = self.erg_c.barrier.space_top_lim[1] - eps; x2_min = eps
                    if x1 >= x1_max or x1 <= x1_min or x2 >= x2_max or x2 <= x2_min:
                        barr_dx = self.erg_c.barrier.dx(x_traj[i][:2])
                    else: 
                        barr_dx = np.zeros((2,))
                    # However we need to append 0s to the non ergodic dimensions before adding to rho_dot
                    barr_dx = np.concatenate((barr_dx, np.zeros((self.model.num_of_states - 2,))))
                    rho_dot -= barr_dx

            # Update rho using the computed rho_dot
            rho[i] = rho[i+1] - rho_dot * dt 

        return rho, t_traj

    def withinBounds(self, x):
        '''
        Check if the state is within the bounds of the system
        '''
        # Check if the 2 first ergodic dimension are within the bounds L1, L2
        if x[0] < 0 or x[0] > self.L1 or x[1] < 0 or x[1] > self.L2:
            print(f"--> ATTENTION: State out of bounds: {x}")

        # Check if model is quadcopter
        if isinstance(self.model, Quadcopter):
            # Check if the 3rd dimension is within the bounds
            z = self.model.state[2]
            if z < 0 or z > self.model.z_target * 20:
                print(f"--> Quad is getting out of hand in the Z dim: {z:.2f} m")


    def visualiseColorForTraj(self, ck, x_traj):
        '''
        Visualise the color for the trajectory using ck coefficients
        '''
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from my_erg_lib.basis import ReconstructedPhiFromCk
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

    def visualiseCoefficients(self, ck):
        import matplotlib.pyplot as plt
        from matplotlib import cm

        k1 = np.linspace(0, self.Kmax, self.Kmax+1)
        k2 = np.linspace(0, self.Kmax, self.Kmax+1)
        K1, K2 = np.meshgrid(k1, k2)
        Z_ck = np.zeros((len(k1), len(k2)))
        Z_phik = np.zeros((len(k1), len(k2)))
        
        for i in range(len(k1)):
            for j in range(len(k2)):
                Z_ck[i, j] = ck[i, j]
                Z_phik[i, j] = self.basis.calcPhikCoeff(int(k1[i]), int(k2[j]))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot Ck coefficients
        im1 = ax1.imshow(Z_ck, cmap=cm.viridis, origin='lower', 
                        extent=[0, self.Kmax, 0, self.Kmax], aspect='equal')
        ax1.set_title('Ck Coefficients')
        ax1.set_xlabel('k1')
        ax1.set_ylabel('k2')
        fig.colorbar(im1, ax=ax1, label='Ck Value')
        
        # Plot Phi_k coefficients
        im2 = ax2.imshow(Z_phik, cmap=cm.viridis, origin='lower', 
                        extent=[0, self.Kmax, 0, self.Kmax], aspect='equal')
        ax2.set_title('Phi_k Coefficients')
        ax2.set_xlabel('k1')
        ax2.set_ylabel('k2')
        fig.colorbar(im2, ax=ax2, label='Phi_k Value')
        
        plt.tight_layout()
        plt.show()


