import numpy as np
from my_erg_lib.replay_buffer import ReplayBufferFIFO
from my_erg_lib.barrier import Barrier

class DecentralisedErgodicController():
    def __init__(self, agent, num_of_agents=1,  
                 uNominal=None, uLimits=None, R=np.eye(2), Q = 1,
                 T_horizon = 0.3, T_sampling=0.01, deltaT_erg=0.9,
                 barrier_weight=100, barrier_eps=0.01):

        # Connect the agent
        # Make sure agent is of type Agent from my_erg_lib
        from my_erg_lib.agent import Agent
        assert isinstance(agent, Agent), "agent must be an instance of the Agent class from my_erg_lib."
        self.agent = agent
        self.num_of_agents = num_of_agents

        # Time Parameters
        self.T = T_horizon
        self.Ts = T_sampling
        self.deltaT_erg = deltaT_erg
        
        # Control Parameters
        self.R = R
        self.Rinv = np.linalg.inv(self.R)
        self.Q = Q
        # If uNominal is not provided, set it to a zero function
        self.uNominal = uNominal if uNominal is not None else lambda x, t: np.zeros((agent.model.num_of_inputs,))
        # uLimits is the ergodic CutOff. It doesnt have to do with the physical limits of the model
        if uLimits is None:
            # Set infinite limits if not provided
            self.uLimits = np.array([[-np.inf, np.inf] for _ in range(agent.model.num_of_inputs)])
        else:
            assert len(uLimits) == agent.model.num_of_inputs, "uLimits should contain [lower, upper] pairs for every control (num_of_inputs)."
            self.uLimits = np.array(uLimits)
        # Set barrier to avoid going outside the exploration space
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
        self.past_states_buffer = ReplayBufferFIFO(capacity=int(self.deltaT_erg/self.agent.model.dt), element_size=(2,), init_content=[self.agent.model.state[:2]]) # size = 2, cause we only care about 2 ergodic dimensions

    def calcNextActionTriplet(self, ti):
        """
        Calculate the next action based on the current state and the target distribution.
        Returns the ergodic control triplet: 
            (u, τ, λ) = (us, tau, lamda_duration)
        where:
            - us: the ergodic control action to be applied
            - tau: the time at which the control action should start (τ ε [ti, ti + Ts])
            - lamda_duration: the duration for which the control action should be applied (λ ε [0, Ts])
        """

        # Simulate Trajectory Forward
        traj = self.agent.simulateForward(x0=self.agent.model.state, ti=ti, udef=self.uNominal, T=self.T)
        erg_traj = traj[:, :2] # Only take the ergodic dimensions
        
        # Calc Ck Coefficients
        ck = self.agent.basis.calcCkCoeff(erg_traj, x_buffer=self.past_states_buffer.get() ,ti=ti, T=self.T)
        erg_cost = self.calcErgodicCost(ck) 

        # Simulate Adjoint Backward to get rho(t)
        rho, _ = self.agent.simulateAdjointBackward(traj, ck, T=self.T, Q=self.Q, num_of_agents=self.num_of_agents, ti=ti)
        dt = self.T / len(traj) # TODO: Can i eliminate this calculation of dt? Take it from a definition?

        # Evaluate Ustar
        ustar = np.zeros((len(traj), self.agent.model.num_of_inputs))
        for i in range(len(traj)):
            ustar[i] = -self.Rinv @ self.agent.model.h(traj[i]).T @ rho[i]

            # Add nominal control if available
            ustar[i] += self.uNominal(traj[i], ti + i * dt)
        
        # Calculate Application Time
        tau, Jtau = self.calcApplicationTime(ustar, rho, traj, ti, self.T)
        assert Jtau < 0, "Jtau is Non Negative, which is not expected."
        
        # Determine Control Duration
        # TODO: Implement a better way to determine the control duration
        lamda_duration = self.Ts * 1.5

        # Keep the approprate control from t=tau
        us = ustar[int((tau - ti) / self.agent.model.dt)]

        # So we have the triplet:
        # (u, τ, λ) = (us, tau, lamda_duration)

        # Saturate Control to given limits
        us = np.clip(us, self.uLimits[:, 0], self.uLimits[:, 1])

        return us, tau, lamda_duration, erg_cost



    def calcApplicationTime(self, ustar, rho, x_traj, ti, T):

        # Calculate the cost function Jt for a given tau
        def Jt(tau):
            i = int((tau - ti) / self.agent.model.dt)
            x = x_traj[i]
            us = ustar[i]
            udef = self.uNominal(x, tau)
            
            Jt_value = rho[i].T @ (self.agent.model.f(x, us) - self.agent.model.f(x, udef))
            # Make sure Jt is a scalar number, otherwise something went wrong
            assert type(Jt_value) == np.float64, f"Jt is not a scalar number, but {type(Jt_value)} (Jt = {Jt_value}). Check the calculation of Jt."
            return Jt_value


        # TODO: Do gradient descent or something faster / clever?
        # TODO: Make sure we dont always choose the first value - Seems like we do
        # Generate time points for evaluation
        t = np.linspace(ti, ti+T, len(x_traj))
        Jt_values = np.array([Jt(tau) for tau in t])

        # Find time that minimizes Jt (argmin Jt)
        min_idx = np.argmin(Jt_values)
        optimal_tau = t[min_idx]
        
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
        return ergodic_cost

