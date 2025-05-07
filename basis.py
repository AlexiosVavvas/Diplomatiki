import numpy as np
from scipy.integrate import nquad

class Basis():

    def __init__(self, L1, L2, Kmax, phi_=None, precalc_hk_coeff=True, precalc_phik_coeff=True):
        self.L1 = L1
        self.L2 = L2
        self.Kmax = Kmax
        
        # Dictionary to store precomputed values
        self.hk_cache = {}
        self.phi_coeff_cache = {}
        self.LamdaK_cache = {}

        # Target Distribution (Phi = f(s) where s -> s[0], s[1])
        self._phi = None
        if phi_ is not None:
            assert callable(phi_), "phi must be a callable function."
            self.phi = phi_

        # Precalculate hk for all k1, k2 pairs
        if precalc_hk_coeff:
            self.precalcAllHk()

        # Precalculate LamdaK for all k1, k2 pairs
        self.precalcAllLamdaK()

        # Precalculate PhiK for all k1, k2 pairs
        if precalc_phik_coeff:
            self.precalcAllPhiK()


    def calcHk(self, k1, k2):
        # Check if the value is already computed
        if (k1, k2) in self.hk_cache:
            return self.hk_cache[(k1, k2)]
        
        L1 = self.L1; L2 = self.L2
        if k1==0 and k2==0:
            hk = L1 * L2
        elif k1==0 and k2!=0:
            hk = L2 * (2*k2*L1*np.pi + L2*np.sin(2*k2*L1*np.pi/L2)) / (4 * L2 * np.pi)
        elif k1!=0 and k2==0:
            hk = L1 * (2*k1*L2*np.pi + L1*np.sin(2*k1*L2*np.pi/L1)) / (4 * L1 * np.pi)
        else:
            hk = (2*k2*L1*np.pi + L2*np.sin(2*k2*L1*np.pi/L2)) * (2*k1*L2*np.pi + L1*np.sin(2*k1*L2*np.pi/L1)) / (4 * L1 * L2)
            hk /= 16 * k1 * k2 * np.pi**2

        hk = np.sqrt(hk)  # Take the square root of the integral

        # add to dictionary
        self.hk_cache[(k1, k2)] = hk

        return hk
    

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
        for k1 in range(self.Kmax+1):
            for k2 in range(self.Kmax+1):
                self.calcPhikCoeff(k1, k2)

    def Fk(self, xv, k1, k2, hk):
        Fk = np.cos(k1*np.pi/self.L1*xv[0]) * np.cos(k2*np.pi/self.L2*xv[1]) / hk
        return Fk
    
    def dFk_dx(self, xv, k1, k2, hk):
        Fk_x = np.zeros((2,))
        Fk_x[0] = -np.sin(k1*np.pi/self.L1*xv[0]) * np.cos(k2*np.pi/self.L2*xv[1]) / hk * (k1*np.pi/self.L1)
        Fk_x[1] = -np.cos(k1*np.pi/self.L1*xv[0]) * np.sin(k2*np.pi/self.L2*xv[1]) / hk * (k2*np.pi/self.L2)
        return Fk_x
        
    def calcPhikCoeff(self, k1, k2):
        assert self._phi != None, "Target distribution phi is not set."

        # Check if the value is already computed
        if (k1, k2) in self.phi_coeff_cache:
            # print(f"----> Phi Coefficients already calculated for k1={k1}, k2={k2}.")
            return self.phi_coeff_cache[(k1, k2)]

        hk = self.calcHk(k1, k2)
        phi_k, _ = nquad(lambda x1, x2: self._phi([x1, x2]) * self.Fk([x1, x2], k1, k2, hk),
                [[0, self.L1], [0, self.L2]])

        self.phi_coeff_cache[(k1, k2)] = phi_k
        print(f"Phi Coefficients calculated for k1={k1}, k2={k2}.")

        return phi_k
    
    # TODO: Recursively calculate the coefficients Ck
    def calcCkCoeff(self, x_traj, ti, T, x_buffer=None):
        '''
        Calculate the coefficients Ck for the trajectory x_traj from time ti to T.
            x_traj: array of shape (n_points, 2) containing trajectory points
            ti: initial time
            T: Duration
        '''
        ck = np.zeros((self.Kmax+1, self.Kmax+1))
        
        # Append to the trajectory the buffer points at the beginning with the traj continueing from the last buffer poit
        if x_buffer is not None:
            x_traj = np.concatenate((x_buffer, x_traj), axis=0)
        
        # Calculate time step (dt) assuming uniform time spacing
        n_points = len(x_traj)
        
        # Time points corresponding to trajectory points
        t_points = np.linspace(ti, ti+T, n_points)
        
        for k1 in range(self.Kmax+1):
            for k2 in range(self.Kmax+1):
                hk = self.calcHk(k1, k2)
                
                # Evaluate Fk at each trajectory point
                fk_values = np.array([self.Fk(x, k1, k2, hk) for x in x_traj])
                
                # Perform trapezoidal integration
                ck[k1, k2] = np.trapz(fk_values, x=t_points) / T
        
        return ck


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
        print("Setting new PHI")
        # Clear the cache for phi coefficients since the target distribution has changed
        self.phi_coeff_cache.clear()


    def copy(self):
        '''
        Create a copy of the current object with the same parameters and target distribution.
        '''
        new_basis = Basis(self.L1, self.L2, self.Kmax, precalc_hk_coeff=False, phi_=self._phi, precalc_phik_coeff=False)
        
        new_basis.hk_cache = self.hk_cache.copy()
        new_basis.phi_coeff_cache = self.phi_coeff_cache.copy()
        print("Coefficients copied.")
        
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
                lamda_k = self.base.LamdaK_cache[(k1, k2)]
                lamda_k = 1
                result += lamda_k * self.ck[k1, k2] * self.base.Fk(args[0], k1, k2, self.base.calcHk(k1, k2))
        
        return result