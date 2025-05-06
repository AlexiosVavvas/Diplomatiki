import numpy as np

class DecentralisedErgodicController():
    def __init__(self, agent, phi=None, num_of_agents=1, R=np.eye(2), Q = 1, uNominal=None,
                 T = 1):

        self.agent = agent
        self.num_of_agents = num_of_agents
        self.R = R
        self.Q = Q
        self.uNominal = uNominal
        self.T = T

        self.agent.setPhi(phi)  # Set the target distribution in the basis object

    def calcNextAction(self, ti):
        """
        Calculate the next action based on the current state and the target distribution.
        """

        # Simulate Trajectory Forward
        traj = self.agent.simulateForward(x0=self.agent.model.state, ti=ti, udef=self.uNominal, T=self.T)

        # Calc Ck Coefficients
        ck = self.agent.basis.calcCkCoeff(traj, ti=ti, T=self.T)

        # Simulate Adjoint Backward to get rho(t)
        rho, _ = self.agent.simulateAdjointBackward(traj, ck, T=self.T, Q=self.Q, num_of_agents=self.num_of_agents)
        dt = self.T / len(traj)

        # Evaluate Ustar
        ustar = np.zeros((len(traj), 2))
        Rinv = np.linalg.inv(self.R)
        for i in range(len(traj)):
            ustar[i] = -Rinv @ self.agent.model.h(traj[i]).T @ rho[i]
            
            if self.uNominal is not None:
                ustar[i] += self.uNominal(traj[i], ti + i * dt)
        
        # Calculate Application Time
        tau, Jtau = self.calcApplicationTime(ustar, rho, traj, ti, self.T)
        assert Jtau != 0, "Jtau is zero, which is not expected."

        # Determine Control Duration
        lamda_duration = 0.01

        us = ustar[int((tau - ti) / self.agent.model.dt)]

        # So we have the triplet:
        # (u, τ, λ) = (us, tau, lamda_duration)

        def plotUs(us):
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.plot(us[:, 0], 'o-', label='Control Action')
            plt.title('Control Action over Time')
            plt.xlabel('Control 1')
            plt.ylabel('Control 2')
            plt.legend()
            plt.grid()
            plt.show()
        
        # normalize each us to be between -1 and 1 using the max value of us
        # ustar[:, 0] = ustar[:, 0] / np.max(np.abs(ustar[:, 0])) if np.max(np.abs(ustar[:, 0])) != 0 else ustar[:, 0]
        # ustar[:, 1] = ustar[:, 1] / np.max(np.abs(ustar[:, 1])) if np.max(np.abs(ustar[:, 1])) != 0 else ustar[:, 1]
        # ustar[:, 0] *= 20
        # ustar[:, 1] *= 20
        # plotUs(ustar)


        return us, tau, lamda_duration



    def calcApplicationTime(self, ustar, rho, x_traj, ti, T):

        # Calculate the cost function Jt for a given tau
        def Jt(tau):
            i = int((tau - ti) / self.agent.model.dt)
            x = x_traj[i]
            us = ustar[i]
            udef = self.uNominal(x, tau) if self.uNominal is not None else np.zeros((2,))
            
            Jt_value = rho[i].T @ (self.agent.model.f(x, us) - self.agent.model.f(x, udef))
            return Jt_value

        def plotJt(t, Jt):
            print("Plotting Jt(t)")
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm

            plt.figure(figsize=(10, 5))
            plt.plot(t, Jt, label='Jt(t)', color='blue')
            plt.axvline(x=ti, color='red', linestyle='--', label='ti')
            plt.axvline(x=ti+T, color='green', linestyle='--', label='ti+T')
            plt.title('Cost Function Jt over Time')
            plt.xlabel('Time (t)')
            plt.ylabel('Jt(t)')
            plt.legend()
            plt.grid()
            plt.show()            


        # TODO: Do gradient descent
        # Generate time points for evaluation
        t = np.linspace(ti, ti+T, len(x_traj))
        Jt_values = np.array([Jt(tau) for tau in t])
        # plotJt(t, Jt_values)

        # Find time that minimizes Jt (argmin Jt)
        min_idx = np.argmin(Jt_values)
        optimal_tau = t[min_idx]
        
        return optimal_tau, Jt(optimal_tau)
        


        
    def calcErgodicCost(self, ck):
        ergodic_cost = 0.0
        for k1 in range(self.agent.Kmax+1):
            for k2 in range(self.agent.Kmax+1):
                ergodic_cost += (ck[k1, k2] - self.agent.basis.calcPhikCoeff(k1, k2))**2
        ergodic_cost *= self.Q
        return ergodic_cost
