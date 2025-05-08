import numpy as np
from my_erg_lib.replay_buffer import ReplayBufferFIFO
from my_erg_lib.barrier import Barrier

class DecentralisedErgodicController():
    def __init__(self, agent, num_of_agents=1, R=np.eye(2), Q = 1, 
                 uNominal=None, uLimits=None,
                 T_horizon = 0.3, T_sampling=0.01, deltaT_erg=0.9,
                 barrier_weight=100, barrier_eps=0.01):

        self.agent = agent
        self.num_of_agents = num_of_agents

        self.T = T_horizon
        self.Ts = T_sampling
        self.deltaT_erg = deltaT_erg
        
        self.R = R
        self.Q = Q
        self.uNominal = uNominal
        if uLimits is None:
            self.uLimits = np.array([[-np.inf, np.inf] for _ in range(agent.model.num_of_inputs)])
        else:
            assert len(uLimits) == agent.model.num_of_inputs, "uLimits should contain [lower, upper] pairs for every control (num_of_inputs)."
            self.uLimits = np.array(uLimits)
        self.barrier = Barrier(L1=agent.L1, L2=agent.L2, pow_=2, weight=barrier_weight, eps_=barrier_eps)
        
        # Make sure everything is in the right format
        assert self.agent.model.dt < T_sampling < T_horizon, "T_sampling must be between dt and T_horizon."
        assert R.shape[0] == R.shape[1] == agent.model.num_of_inputs, "R must be a square matrix of size (num_of_inputs, num_of_inputs)"
        if uNominal is not None:
            assert callable(uNominal), "uNominal must be a callable function."
            assert uNominal(agent.model.state, 0).shape[0] == agent.model.num_of_inputs, "uNominal must return a vector of size (num_of_inputs,)"

        # Variable to store action if available (non zero if calculated and within sample space)
        self.ustar_mask = np.zeros((int(self.Ts/self.agent.model.dt), agent.model.num_of_inputs))
        # Variable to store past states for better Ck calculation (using Δte)
        self.past_states_buffer = ReplayBufferFIFO(capacity=int(self.deltaT_erg/self.agent.model.dt), element_size=(2,)) # size = 2, cause we only care about 2 ergodic dimensions

    def calcNextActionTriplet(self, ti):
        """
        Calculate the next action based on the current state and the target distribution.
        """

        # Simulate Trajectory Forward
        traj = self.agent.simulateForward(x0=self.agent.model.state, ti=ti, udef=self.uNominal, T=self.T)
        erg_traj = traj[:, :2] # Only take the ergodic dimensions
        
        # Calc Ck Coefficients
        ck = self.agent.basis.calcCkCoeff(erg_traj, x_buffer=self.past_states_buffer.get() ,ti=ti, T=self.T)
        erg_cost = self.calcErgodicCost(ck) 

        # Simulate Adjoint Backward to get rho(t)
        rho, _ = self.agent.simulateAdjointBackward(traj, ck, T=self.T, Q=self.Q, num_of_agents=self.num_of_agents)
        dt = self.T / len(traj) # TODO: Can i eliminate this calculation of dt? Take it from a definition?

        # Evaluate Ustar
        ustar = np.zeros((len(traj), self.agent.model.num_of_inputs))
        Rinv = np.linalg.inv(self.R)
        for i in range(len(traj)):
            ustar[i] = -Rinv @ self.agent.model.h(traj[i]).T @ rho[i]
            # print(f"rho[{i}]: {rho[i]}, \nustar[{i}]: {ustar[i]}\nh[{i}]: {self.agent.model.h(traj[i])}\n\n")

            if self.uNominal is not None:
                ustar[i] += self.uNominal(traj[i], ti + i * dt)
        
        # Calculate Application Time
        tau, Jtau = self.calcApplicationTime(ustar, rho, traj, ti, self.T)
        assert Jtau < 0, "Jtau is Non Negative, which is not expected."
        # print(f"tau-ti: {(tau-ti)/self.agent.model.dt} \t Jtau: {Jtau}")
        
        # Determine Control Duration
        # TODO: Implement a better way to determine the control duration
        lamda_duration = self.Ts * 1.5
        # lamda_duration = self.agent.model.dt * np.random.randint(1, int(self.Ts / self.agent.model.dt))

        # Keep the approprate control from t=tau
        us = ustar[int((tau - ti) / self.agent.model.dt)]

        # So we have the triplet:
        # (u, τ, λ) = (us, tau, lamda_duration)

        def plotUs(us):
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.plot(us[:, 0], 'o-', label='Control Action 1')
            plt.plot(us[:, 0], 'x-', label='Control Action 2')
            plt.title('Control Action over Time')
            plt.xlabel('Control 1')
            plt.ylabel('Control 2')
            plt.legend()
            plt.grid()
            plt.show()
        
        # Saturate Control to given limits
        us = np.clip(us, self.uLimits[:, 0], self.uLimits[:, 1])

        # plotUs(ustar)
        # plotUs(rho)
        # print(f"us: {us} \t tau-ti: {tau-ti}")
        return us, tau, lamda_duration, erg_cost



    def calcApplicationTime(self, ustar, rho, x_traj, ti, T):

        # Calculate the cost function Jt for a given tau
        def Jt(tau):
            i = int((tau - ti) / self.agent.model.dt)
            x = x_traj[i]
            us = ustar[i]
            udef = self.uNominal(x, tau) if self.uNominal is not None else np.zeros((self.agent.model.num_of_inputs,))
            
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


        # TODO: Do gradient descent or something faster / clever?
        # TODO: Make sure we dont always choose the first value - Seems like we do
        # Generate time points for evaluation
        t = np.linspace(ti, ti+T, len(x_traj))
        Jt_values = np.array([Jt(tau) for tau in t])
        # plotJt(t, Jt_values)

        # Find time that minimizes Jt (argmin Jt)
        min_idx = np.argmin(Jt_values)
        optimal_tau = t[min_idx]
        # optimal_tau = ti
        
        return optimal_tau, Jt(optimal_tau)
        

    def updateActionMask(self, ti, us, tau, lamda_duration):
        # Reset previous calculations
        self.ustar_mask[:] = 0
        # Calculate starting index
        i_start = int((tau - ti) / self.agent.model.dt)
        # Check wether tau is within the current timestep (τ ε [ti, ti + Ts])
        if i_start >= 0 and i_start < len(self.ustar_mask):
            # Calculate end index based on duration
            i_end = int((tau + lamda_duration - ti) / self.agent.model.dt)
            i_end = min(i_end, len(self.ustar_mask))
            # Save control to variable mask            
            for j in range(i_start, i_end):
                self.ustar_mask[j] = us


        
    def calcErgodicCost(self, ck):
        ergodic_cost = 0.0
        for k1 in range(self.agent.Kmax+1):
            for k2 in range(self.agent.Kmax+1):
                ergodic_cost += (ck[k1, k2] - self.agent.basis.calcPhikCoeff(k1, k2))**2
        ergodic_cost *= self.Q
        # print(f"Ergodic Cost: {ergodic_cost}")
        return ergodic_cost

