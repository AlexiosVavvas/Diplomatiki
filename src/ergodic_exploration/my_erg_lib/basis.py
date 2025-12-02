import numpy as np
from scipy.integrate import nquad
import time

class Basis():
    """
    Ergodic exploration basis function class for 2D domains.
    
    Implements Fourier basis functions for ergodic coverage and target distribution 
    approximation using cosine basis functions over a rectangular domain.
    """

    def __init__(self, L1_BOUNDS, L2_BOUNDS, Kmax, phi_=None, precalc_hk_coeff=True, precalc_phik_coeff=True, integration_method='gauss', num_gauss_points=30):
        """
        Initialize the Basis class for ergodic exploration.
        
        Parameters:
            - L1_BOUNDS (list/tuple):    Domain bounds for first dimension [min, max]
            - L2_BOUNDS (list/tuple):    Domain bounds for second dimension [min, max]  
            - Kmax (int):                Maximum number of basis functions in each dimension (>=1)
            - phi_ (callable, optional): Target distribution function phi(s) where s=[x1,x2]. 
                                         Defaults to constant function if None
            - precalc_hk_coeff (bool):   Whether to precalculate hk normalization coefficients
            - precalc_phik_coeff (bool): Whether to precalculate phi_k coefficients for target distribution
            - integration_method (str):  Integration method for phi_k calculation ('gauss' or 'nquad')
            - num_gauss_points (int):    Number of Gauss-Legendre quadrature points for integration (>=1)
        
        Raises:
            AssertionError: If bounds are invalid, Kmax < 1, or integration parameters are invalid
        """
        
        # Assert bounds are valid
        assert isinstance(L1_BOUNDS, (list, tuple)) and len(L1_BOUNDS) == 2, "L1_BOUNDS must be a list or tuple of 2 elements [min, max]."
        assert isinstance(L2_BOUNDS, (list, tuple)) and len(L2_BOUNDS) == 2, "L2_BOUNDS must be a list or tuple of 2 elements [min, max]."
        assert L1_BOUNDS[0] < L1_BOUNDS[1], "L1_BOUNDS are not valid. Lower bound must be less than upper bound."
        assert L2_BOUNDS[0] < L2_BOUNDS[1], "L2_BOUNDS are not valid. Lower bound must be less than upper bound."

        self.L1_min = L1_BOUNDS[0]
        self.L1_max = L1_BOUNDS[1]
        self.L2_min = L2_BOUNDS[0]
        self.L2_max = L2_BOUNDS[1]
        
        # Calculate domain sizes
        self.L1_size = self.L1_max - self.L1_min
        self.L2_size = self.L2_max - self.L2_min
        
        self.Kmax = Kmax
        self.ck = np.zeros((Kmax+1, Kmax+1))         # Last computed ck for sharing with others
        self.ck_bar_old = np.zeros((Kmax+1, Kmax+1)) # Cumulative Ck values from previous iterations (used for infinite buffer + avoiding infinite integration)
        assert Kmax >= 1, "Kmax must be greater than or equal to 1."
        
        # Dictionary to store precomputed values
        self.hk_cache = {}
        self.phi_coeff_cache = {}
        self.LamdaK_cache = {}
        
        # Integration method and parameters for phik calculation
        self.integration_method = integration_method   # Integration method for calculating phi coefficients integral
        self.num_gauss_points = num_gauss_points       # Number of Gauss-Legendre points for integration
        assert integration_method in ['gauss', 'nquad'], "Method must be either 'gauss' or 'nquad'."
        assert num_gauss_points >= 1, "Number of Gauss points must be greater than or equal to 1."

        # Target Distribution (Phi = f(s) where s -> s[0], s[1])
        self._phi = None
        if phi_ is not None:
            assert callable(phi_), "phi must be a callable function."
            self.phi = phi_
        else:
            # Default to constant 1 function if not provided
            self.phi = lambda s: 1
            # pass # TODO: Check, something is wrong here

        # Precalculate hk for all k1, k2 pairs
        if precalc_hk_coeff:
            self.precalcAllHk()

        # Precalculate LamdaK for all k1, k2 pairs
        self.precalcAllLamdaK()

        # Precalculate PhiK for all k1, k2 pairs
        if precalc_phik_coeff:
            self.precalcAllPhiK()


    # Basis Functions ---------------------------------------------------------
    # xv: [x1, x2] (2D point) - Ergodic dimensions
    def Fk(self, xv, k1, k2, hk):
        # Normalize position to [0, domain_size] by subtracting the minimum bound
        x_norm = [(xv[0] - self.L1_min), (xv[1] - self.L2_min)]
        Fk = np.cos(k1*np.pi*x_norm[0]/self.L1_size) * np.cos(k2*np.pi*x_norm[1]/self.L2_size) / hk
        return Fk
    
    def dFk_dx(self, xv, k1, k2, hk):
        # Normalize position to [0, domain_size] by subtracting the minimum bound
        x_norm = [(xv[0] - self.L1_min), (xv[1] - self.L2_min)]
        Fk_x = np.zeros((2,))
        Fk_x[0] = -np.sin(k1*np.pi*x_norm[0]/self.L1_size) * np.cos(k2*np.pi*x_norm[1]/self.L2_size) / hk * (k1*np.pi/self.L1_size)
        Fk_x[1] = -np.cos(k1*np.pi*x_norm[0]/self.L1_size) * np.sin(k2*np.pi*x_norm[1]/self.L2_size) / hk * (k2*np.pi/self.L2_size)
        return Fk_x

    # Precalculations ---------------------------------------------------------
    # Precompute hk for all k1, k2 pairs
    def precalcAllHk(self):
        for k1 in range(self.Kmax+1):
            for k2 in range(self.Kmax+1):
                self.calcHk(k1, k2)

    # Precompute LamdaK for all k1, k2 pairs
    # LamdaK = (1 + |k|^2)^(-(v+1)/2) where v = 2 (Num of Ergodic Dimensions)
    def precalcAllLamdaK(self):
        v_ = 2 # Num of Ergodic Dimensions
        for k1 in range(self.Kmax+1):
            for k2 in range(self.Kmax+1):
                abs_k_sq = k1**2 + k2**2
                lamda_k_ = (1 + abs_k_sq) ** (-(v_+1)/2)
                self.LamdaK_cache[(k1, k2)] = lamda_k_

    # Precompute PhiK
    def precalcAllPhiK(self):
        print("Precalculating PhiK coefficients...", end='', flush=True)
        for k1 in range(self.Kmax+1):
            for k2 in range(self.Kmax+1):
                t_ = time.time()
                self.calcPhikCoeff(k1, k2)
        print(" Done")
    
    # Main Coefficients Calculation ---------------------------------------------
    def calcHk(self, k1, k2):
        # Check if the value is already computed
        if (k1, k2) in self.hk_cache:
            return self.hk_cache[(k1, k2)]

        # Use domain sizes instead of max values
        L1 = self.L1_size
        L2 = self.L2_size

        # Calculation of hk based on their paper (Something is Wrong - Implementation Fault??)
        # if k1==0 and k2==0:
        #     hk = L1 * L2
        # elif k1==0 and k2!=0:
        #     hk = L2 * (2*k2*L1*np.pi + L2*np.sin(2*k2*L1*np.pi/L2)) / (4 * L2 * np.pi)
        # elif k1!=0 and k2==0:
        #     hk = L1 * (2*k1*L2*np.pi + L1*np.sin(2*k1*L2*np.pi/L1)) / (4 * L1 * np.pi)
        # else:
        #     hk = (2*k2*L1*np.pi + L2*np.sin(2*k2*L1*np.pi/L2)) * (2*k1*L2*np.pi + L1*np.sin(2*k1*L2*np.pi/L1)) / (4 * L1 * L2)
        #     hk /= 16 * k1 * k2 * np.pi**2

        # hk = np.sqrt(hk)  # Take the square root of the integral

        # Correct Calculation
        def alpha(k,L):
            return np.sqrt(1/L) if k==0 else np.sqrt(2/L)
        hk = 1 / (alpha(k1, L1) * alpha(k2, L2) )

        # add to dictionary
        self.hk_cache[(k1, k2)] = hk

        return hk

    def calcPhikCoeff(self, k1, k2, save_to_cache=True):

        assert self._phi != None, "Target distribution phi is not set."

        # Check if the value is already computed
        if (k1, k2) in self.phi_coeff_cache:
            return self.phi_coeff_cache[(k1, k2)]

        t_ = time.time()
        hk = self.calcHk(k1, k2)

        if self.integration_method == 'gauss':
            # Get Gauss-Legendre quadrature points and weights
            x1_points, x1_weights = np.polynomial.legendre.leggauss(self.num_gauss_points)
            x2_points, x2_weights = np.polynomial.legendre.leggauss(self.num_gauss_points)
            
            # Transform from [-1,1] to [L1_min,L1_max] and [L2_min,L2_max]
            x1_points = 0.5 * self.L1_size * (x1_points + 1) + self.L1_min
            x2_points = 0.5 * self.L2_size * (x2_points + 1) + self.L2_min
            x1_weights = 0.5 * self.L1_size * x1_weights
            x2_weights = 0.5 * self.L2_size * x2_weights
            
            # Compute the integral
            # TODO: Maybe vectorize this part
            result = 0.0
            for i in range(self.num_gauss_points):
                for j in range(self.num_gauss_points):
                    x1, x2 = x1_points[i], x2_points[j]
                    result += x1_weights[i] * x2_weights[j] * self._phi([x1, x2]) * self.Fk([x1, x2], k1, k2, hk)
            
            phi_k = result
        elif self.integration_method == 'nquad':
            # Use nquad for numerical integration over the actual bounds
            phi_k, _ = nquad(lambda x1, x2: self._phi([x1, x2]) * self.Fk([x1, x2], k1, k2, hk),
                [[self.L1_min, self.L1_max], [self.L2_min, self.L2_max]])
        else:
            raise ValueError("Invalid method. Use 'gauss' or 'nquad'.")

        if save_to_cache:
            # Save the computed value to the cache
            self.phi_coeff_cache[(k1, k2)] = phi_k

        if time.time() - t_ > 0.1:
            print(f"Phi Coefficient calculated for k1={k1}, k2={k2}." + (f" \t [{time.time()-t_:.2f} s]" if time.time()-t_ > 0.1 else ""))
        return phi_k

    # Recursively calculate the coefficients Ck
    def calcCkCoeffRecursive(self, x_traj, ti, T, ts, t0_erg, x_buffer, update_ck_old=True):
        # Calculate integral from ti -> ti + T
        ck_int_forward  = self.calcCkCoeff(x_traj, ti, T, do_not_divide_integral_flag=True)
        # Calculate integral from ti-ts -> ti
        ck_int_backward = self.calcCkCoeff(x_buffer, ti-ts, ts, do_not_divide_integral_flag=True)

        ck_bar_new = (ti - ts + T - t0_erg) / (ti + T - t0_erg) * self.ck_bar_old + 1/(ti + T - t0_erg) * ck_int_backward
        if update_ck_old:
            self.ck_bar_old = ck_bar_new

        # Calculate final ck
        ck = ck_bar_new + 1/(ti + T - t0_erg) * ck_int_forward
        
        # Save to memory
        self.ck = ck

        return ck.copy()

    def calcCkCoeff(self, x_traj, ti, T, x_buffer=None, do_not_divide_integral_flag=False):
        '''
        Calculate the coefficients Ck for the trajectory x_traj from time ti to T.
            x_traj: Ergodic states trajectory only (x1, x2)
            ti:     Current Initial Time
            T:      Duration forward
        '''
        ck = np.zeros((self.Kmax+1, self.Kmax+1))
        
        # Append to the trajectory the buffer points at the beginning with the traj continueing from the last buffer poit
        if x_buffer is not None:
            x_traj = np.concatenate((x_buffer, x_traj), axis=0)
            # Lets calculate simulation time step (dt) assuming uniform time spacing
            dt = (T - ti) / len(x_traj)
            # How much time in the back do we go with the buffer?
            delta_t = len(x_buffer) * dt  # Ergodic memory time
        else:
            # If we dont play with a buffer, we dont need ergodic memory
            delta_t = 0            
        
        # Calculate time step (dt) assuming uniform time spacing
        n_points = len(x_traj)
        
        # Time points corresponding to trajectory points
        t_points = np.linspace(ti-delta_t, ti+T, n_points)
        
        for k1 in range(self.Kmax+1):
            for k2 in range(self.Kmax+1):
                hk = self.calcHk(k1, k2)

                # Normalize trajectory points relative to domain bounds
                x_norm = x_traj - np.array([self.L1_min, self.L2_min])
                
                # Vectorized Fk calculation
                cos_k1 = np.cos(k1*np.pi/self.L1_size*x_norm[:, 0])
                cos_k2 = np.cos(k2*np.pi/self.L2_size*x_norm[:, 1])
                
                # Evaluate Fk at each trajectory point
                fk_values = cos_k1 * cos_k2 / hk
                
                # Perform trapezoidal integration
                if do_not_divide_integral_flag:
                    ck[k1, k2] = np.trapz(fk_values, x=t_points)
                else:
                    ck[k1, k2] = np.trapz(fk_values, x=t_points) / (delta_t + T)
        
        # Save to memory
        self.ck = ck
        
        return ck.copy()


    # Other Properties and Functions -----------------------------------------
    @property
    def phi(self):
        return self._phi
    
    @phi.setter
    def phi(self, new_phi):
        '''
        Set the target distribution phi when someone calls object.phi = new_phi 
        '''
        assert callable(new_phi), "phi must be a callable function."
        # Change Phi Target Distribution
        self._phi = new_phi
        # print("Setting new PHI")
        # Clear the cache for phi coefficients since the target distribution has changed
        self.phi_coeff_cache.clear()


    def copy(self):
        '''
        Create a copy of the current object with the same parameters and target distribution.
        '''
        new_basis = Basis(L1_BOUNDS=[self.L1_min, self.L1_max], L2_BOUNDS=[self.L2_min, self.L2_max], Kmax=self.Kmax, precalc_hk_coeff=False, phi_=self._phi, precalc_phik_coeff=False, num_gauss_points=self.num_gauss_points, integration_method=self.integration_method)

        new_basis.hk_cache = self.hk_cache.copy()
        new_basis.phi_coeff_cache = self.phi_coeff_cache.copy()
        # print("Coefficients copied.")
        
        return new_basis
    


class ReconstructedPhi():

    def __init__(self, base: Basis, precalc_phik=True):
        self.base = base.copy()

        # Precalculate coefficients at start
        if precalc_phik:
            self.precalcAllPhikCoeff()

    def precalcAllPhikCoeff(self):
        for k1 in range(self.base.Kmax+1):
            for k2 in range(self.base.Kmax+1):
                self.base.calcPhikCoeff(k1, k2)

    def __call__(self, *args, **kwds):
        result = 0

        for k1 in range(self.base.Kmax+1):
            for k2 in range(self.base.Kmax+1):
                result += self.base.calcPhikCoeff(k1, k2) * self.base.Fk(args[0], k1, k2, self.base.calcHk(k1, k2))
        
        return result
    

class ReconstructedPhiFromCk():
    def __init__(self, base: Basis, ck):
        self.base = base.copy()
        self.ck = ck.copy()

    def __call__(self, *args, **kwds):
        result = 0

        for k1 in range(self.base.Kmax+1):
            for k2 in range(self.base.Kmax+1):
                result += self.ck[k1, k2] * self.base.Fk(args[0], k1, k2, self.base.calcHk(k1, k2))
        
        return result