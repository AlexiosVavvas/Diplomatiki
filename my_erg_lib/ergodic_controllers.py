import numpy as np
from my_erg_lib.replay_buffer import ReplayBufferFIFO
from my_erg_lib.barrier import Barrier
import vis

class DecentralisedErgodicController():
    def __init__(self, agent, num_of_agents=1,  
                 uNominal=None, uLimits=None, R=None, Q = 1,
                 T_horizon = 0.3, T_sampling=0.01, deltaT_erg=0.9,
                 barrier_weight=100, barrier_eps=0.01, barrier_pow=2):

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
        self.R = R if R is not None else np.eye(agent.model.num_of_inputs)
        self.Rinv = np.linalg.inv(self.R)
        self.Q = Q
        # If uNominal is not provided, set it to a zero function
        self.uNominal = NominalFunction(agent.model.num_of_inputs, agent.model.num_of_states, func=uNominal)

        # uLimits is the ergodic CutOff. It doesnt have to do with the physical limits of the model
        if uLimits is None:
            # Set infinite limits if not provided
            self.uLimits = np.array([[-np.inf, np.inf] for _ in range(agent.model.num_of_inputs)])
        else:
            assert len(uLimits) == agent.model.num_of_inputs, "uLimits should contain [lower, upper] pairs for every control (num_of_inputs)."
            self.uLimits = np.asarray(uLimits)

        # Set barrier to avoid going outside the exploration space
        self.barrier = Barrier(L1=agent.L1, L2=agent.L2, weight=barrier_weight, eps_=barrier_eps, pow_=barrier_pow)
        
        # Make sure everything is in the right format
        assert self.agent.model.dt < T_sampling < T_horizon, "T_sampling must be between dt and T_horizon."
        assert self.R.shape[0] == self.R.shape[1] == agent.model.num_of_inputs, "R must be a square matrix of size (num_of_inputs, num_of_inputs)"
        if uNominal is not None:
            assert callable(uNominal), "uNominal must be a callable function."
            assert uNominal(agent.model.state, 0).shape[0] == agent.model.num_of_inputs, "uNominal must return a vector of size (num_of_inputs,)"

        # Variable to store action if available (non zero if calculated and within sample space)
        self.ustar_mask = np.zeros((int(self.Ts/self.agent.model.dt), agent.model.num_of_inputs))
        # Variable to store past states for better Ck calculation (using Δte)
        self.past_states_buffer = ReplayBufferFIFO(capacity=int(self.deltaT_erg/self.agent.model.dt), element_size=(2,), init_content=[self.agent.model.state[:2]]) # size = 2, cause we only care about 2 ergodic dimensions


    def simulateAdjointBackward(self, x_traj, u_traj, t_traj, ck, T=1.0, Q=1, num_of_agents=1):
        '''
        Simulate the adjoint state to get rho(t)
        Integrating with simple Euler method Backwards from rho(ti+T) = 0
        '''
        dt = t_traj[1] - t_traj[0]  # Time step: Dt is not a variable here. We chose it to be the same as the forward pass
        rho = np.zeros((len(x_traj), self.agent.model.num_of_states))

        # Integrating with simple Euler method Backwards from ρ(ti+T) = 0
        for i in range(len(x_traj)-2, -1, -1):
            # Jacobian of the dynamics with respect to x
            f_x = self.agent.model.f_x(x=x_traj[i], u=u_traj[i])

            rho_dot = -np.dot(f_x.T, rho[i+1])
            
            for k1 in range(self.agent.Kmax+1):
                for k2 in range(self.agent.Kmax+1):
                    # Calculating summation terms
                    lamda_k = self.agent.basis.LamdaK_cache[(k1, k2)]
                    hk = self.agent.basis.calcHk(k1, k2)
                    ck_ = ck[k1, k2]
                    phi_k = self.agent.basis.calcPhikCoeff(k1, k2)
                    dFdx = self.agent.basis.dFk_dx(x_traj[i][:2], k1, k2, hk)
                    # TODO: Check: Since Fk(xv) the derivative lacks dimensions to reach x. So i think we should append 0s
                    dFdx = np.concatenate((dFdx, np.zeros((self.agent.model.num_of_states - 2,))))
                    
                    # Adding to rho_dot(x[i], t[i])
                    rho_dot += (-2 * Q / T / num_of_agents) * lamda_k * (ck_ - phi_k) * dFdx
                    
                    # if we are epsilon close to the barrier, we need to add the barrier term	
                    eps = self.agent.erg_c.barrier.eps
                    x1 = x_traj[i][0]; x1_max = self.agent.erg_c.barrier.space_top_lim[0] - eps; x1_min = eps
                    x2 = x_traj[i][1]; x2_max = self.agent.erg_c.barrier.space_top_lim[1] - eps; x2_min = eps
                    if x1 >= x1_max or x1 <= x1_min or x2 >= x2_max or x2 <= x2_min:
                        barr_dx = self.agent.erg_c.barrier.dx(x_traj[i][:2])
                    else: 
                        barr_dx = np.zeros((2,))
                    # However we need to append 0s to the non ergodic dimensions before adding to rho_dot
                    barr_dx = np.concatenate((barr_dx, np.zeros((self.agent.model.num_of_states - 2,))))
                    rho_dot -= barr_dx

            # Update rho using the computed rho_dot
            rho[i] = rho[i+1] - rho_dot * dt 

        return rho, t_traj


    def calcNextActionTriplet(self, ti, prediction_dt=None):
        """
        Calculate the next action based on the current state and the target distribution.
        Returns the ergodic control triplet: 
            (u, τ, λ) = (us, tau, lamda_duration)
        where:
            - us: the ergodic control action to be applied
            - tau: the time at which the control action should start (τ ε [ti, ti + Ts])
            - lamda_duration: the duration for which the control action should be applied (λ ε [0, Ts])
        """
        # Set default prediction_dt to model_dt if not provided
        prediction_dt = self.agent.model.dt if prediction_dt is None else prediction_dt

        # Simulate Trajectory Forward using prediction dt
        x_traj, u_traj, t_traj = self.agent.model.simulateForward(x0=self.agent.model.state, ti=ti, udef=self.uNominal, T=self.T, dt=prediction_dt)
        erg_traj = x_traj[:, :2] # Save seperately the ergodic dimensions
        
        # Calc Ck Coefficients
        ck = self.agent.basis.calcCkCoeff(erg_traj, x_buffer=self.past_states_buffer.get() ,ti=ti, T=self.T)
        erg_cost = self.calcErgodicCost(ck) 

        # Simulate Adjoint Backward to get rho(t)
        rho, _ = self.simulateAdjointBackward(x_traj, u_traj, t_traj, ck, T=self.T, Q=self.Q, num_of_agents=self.num_of_agents)
        # vis.simplePlot(x=t_traj - ti, y=rho, 
        #            title="Time [s]", y_label="Rho Values", y_type="np.array",
        #            x_lim=None, y_lim=None,
        #            T_SHOW=0.1, fig_num=0)

        # Evaluate Ustar
        ustar = np.zeros((len(x_traj), self.agent.model.num_of_inputs))
        for i in range(len(x_traj)):
            ustar[i] = -self.Rinv @ self.agent.model.h(x_traj[i]).T @ rho[i]

            # Add nominal control if available
            ustar[i] += self.uNominal(x_traj[i], ti + i * prediction_dt)
        
        # Calculate Application Time
        tau, Jtau = self.calcApplicationTime(ustar, rho, x_traj, t_traj, ti, self.T)
        assert Jtau < 0, "Jtau is Non Negative, which is not expected."
        
        # Determine Control Duration
        lamda_duration = self.calcLambdaDuration() # Default: 0.1 * Ts

        # Keep the approprate control from t=tau
        us = ustar[int((tau - ti) / prediction_dt)]

        # So we have the triplet:
        # (u, τ, λ) = (us, tau, lamda_duration)

        # Saturate Control to given limits
        us = np.clip(us, self.uLimits[:, 0], self.uLimits[:, 1])

        return us, tau, lamda_duration, erg_cost



    def calcApplicationTime(self, ustar, rho, x_traj, t_traj, ti, T):

        # Calculate the cost function Jt for a given tau
        def Jt(t, x, us, rho):
            udef = self.uNominal(x, t)
            
            Jt_value = rho.T @ (self.agent.model.f(x, us) - self.agent.model.f(x, udef))
            
            # Make sure Jt is a scalar number, otherwise something went wrong
            assert type(Jt_value) == np.float64, f"Jt is not a scalar number, but {type(Jt_value)} (Jt = {Jt_value}). Check the calculation of Jt."
            return Jt_value

        # TODO: Do gradient descent or something faster / clever? - Could change the time step here from t_traj to go faster
        # TODO: Make sure we dont always choose the first value - Seems like we do
        Jt_values = np.array([Jt(t_traj[i], x_traj[i], ustar[i], rho[i]) for i in range(len(t_traj))])
        # vis.simplePlot(x=(t_traj - ti)/self.Ts, y=[Jt_values], 
        #            title="Jt_values", y_label="Jt", y_type="list",
        #            x_lim=None, y_lim=None,
        #            T_SHOW=0.1, fig_num=1)

        # Find time that minimizes Jt (argmin Jt)
        min_idx = np.argmin(Jt_values)
        optimal_tau = t_traj[min_idx]
        optimal_Jt = Jt(t_traj[min_idx], x_traj[min_idx], ustar[min_idx], rho[min_idx])
        
        return optimal_tau, optimal_Jt
        
    def calcLambdaDuration(self):
        # TODO: Implement a better way to determine the control duration
        """
        Je(x(u*)) - Je(x(unom)) = ΔJe ~= pJe_pλ|τ * λ 
        Also ΔJe < Ce 
        We need to find the max value of λ that satisfies this condition
        We start with a big value and halve it until the condition is met.
        """
        lamda = self.Ts * 0.2
        return lamda

    def updateActionMask(self, ti, us, tau, lamda_duration):
        # Reset previous calculations
        self.ustar_mask[:] = 0
        # Calculate starting index
        i_start = int((tau - ti) / self.agent.model.dt)
        # Check wether tau is within the current timestep (τ ε [ti, ti + Ts])
        if i_start >= 0 and i_start < len(self.ustar_mask):
            # Calculate end index based on duration
            i_end = int((tau + lamda_duration - ti) / self.agent.model.dt) + 1
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


# Default controller class
class NominalFunction():
    def __init__(self, num_of_inputs, num_of_states, func=None):
        # Store them to use them for validation of the dimensions later
        self.num_of_inputs = num_of_inputs
        self.num_of_states = num_of_states
        # Append the function given
        func_ = func if func is not None else lambda x, t: np.zeros((num_of_inputs,))
        self.additional_functions = [func_] # Store additional functions to be added
        # Variables to remember
        self.uNom_was_None_at_the_beg = func is None
        self.have_already_removed_the_zero_control = False

    def __call__(self, x, t):
        result = 0
        for func in self.additional_functions:
            result += func(x, t)
        return result
    
    def __iadd__(self, other):
        # Add the new function to the list of additional functions
        # Lets make sure proper function form
        assert callable(other), "Functions appended to nominal control must be callable."
        x0 = np.zeros((self.num_of_states,))
        u0 = other(x0, 0)
        assert u0.shape[0] == self.num_of_inputs, f"Functions appended to nominal control must return a vector of size ({self.num_of_inputs},)"

        # Finally lets append the function to the list
        self.additional_functions.append(other)

        # If the original uNominal was None, we need to remove the zero control from the list. No need for it anymore.
        if self.uNom_was_None_at_the_beg and not self.have_already_removed_the_zero_control:
            # Remove the zero control from the list of additional functions
            self.additional_functions = self.additional_functions[1:]
            self.have_already_removed_the_zero_control = True

        # Return self
        return self
    