from basis import Basis
import numpy as np
from model_dynamics import SingleIntegrator, DoubleIntegrator

class Agent():
    def __init__(self, L1, L2, Kmax, dynamics_model, phi=None):
        self.L1 = L1
        self.L2 = L2
        self.Kmax = Kmax
        
        self.model = dynamics_model
        self.model.reset()

        # Initialize the basis object
        self.basis = Basis(L1, L2, Kmax, phi)


    def setPhi(self, phi=None):
        """Set the target distribution."""
        self._phi = phi if phi is not None else lambda s: 2  # Default to constant 0 function if not provided
        self.basis.phi = self._phi

    # Simulates default input and returns full state trajectory
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
            udef_ = np.zeros((self.model.num_of_inputs,))

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
        rho = np.zeros((len(x_traj), self.model.num_of_states))

        for i in range(len(x_traj)-2, -1, -1):
            # Jacobian of the dynamics with respect to x
            f_x = self.model.f_x(x_traj[i])
            # print(f"f_x: {f_x}")

            # self.erg_c.calcErgodicCost(ck)
            # self.visualiseCoefficients(ck)

            rho_dot = -np.dot(f_x.T, rho[i+1])  # TOCHECK: f_x.T
            for k1 in range(self.Kmax+1):
                for k2 in range(self.Kmax+1):
                    lamda_k = self.basis.LamdaK_cache[(k1, k2)]
                    hk = self.basis.calcHk(k1, k2)
                    ck_ = ck[k1, k2]
                    phi_k = self.basis.calcPhikCoeff(k1, k2)
                    dFdx = self.basis.dFk_dx(x_traj[i][:2], k1, k2, hk)
                    # TODO: Check: Since Fk(xv) the derivative lacks dimensions to reach x. So i think we should append 0s
                    dFdx = np.concatenate((dFdx, np.zeros((self.model.num_of_states - 2,))))
                    rho_dot += (-2 * Q / T / num_of_agents) * lamda_k * (ck_ - phi_k) * dFdx
                    # if dFdx[0] > 0.1 or dFdx[1] > 0.1 or ck_ > 0.1 or phi_k > 0.1:
                    #     print(f"dFdx: {dFdx[0]:.1f} - {dFdx[1]:.1f} \t@ (k1, k2): ({k1}, {k2}) \t ck: {ck[k1, k2]} \t phi_k: {phi_k} \t hk: {hk:.1f} \t rho_dot: {rho_dot}")
            # self.visualiseCoefficients(ck)
            # input("Press Enter to continue...")
                    # Print 0 instead of the number if it's smaller than 1e-4
            #         rho_dot_str = "0" if abs(rho_dot).all() < 1e-4 else str(rho_dot)
            #         ck_str = "0" if abs(ck[k1, k2]) < 1e-4 else str(ck[k1, k2])
            #         phi_k_str = "0" if abs(phi_k) < 1e-4 else str(phi_k)
            #         hk_str = "0" if abs(hk) < 1e-4 else str(hk)
            #         lamda_k_str = "0" if abs(lamda_k) < 1e-4 else str(lamda_k)
            #         basis = "0" if abs(self.basis.dFk_dx(x_traj[i], k1, k2, hk)).all() < 1e-2 else str(dFdx)
            #         print(f"rho_dot: {rho_dot_str}\t ck[{k1}, {k2}]: {ck_str}\t phi_k: {phi_k_str}\t hk: {hk_str}\t lamda_k: {lamda_k_str}\t basis: {basis}")
            # print(f"rho_dot: {rho_dot}\t dt: {self.model.dt}\n\n")
            # Update rho using the computed rho_dot
            rho[i] = rho[i+1] - rho_dot * self.model.dt 
            # input("Press Enter to continue...")
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
            # plotRho(x_traj)
        # input("Press Enter to continue...")
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
