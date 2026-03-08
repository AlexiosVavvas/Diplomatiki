"""
CBF-QP Solver Module

Implements High-Order Control Barrier Functions (HOCBF) with State Augmentation.
Uses the augmented dynamics approach where:
  - Augmented state: ξ = [x^T, u^T]^T ∈ R^16
  - New decision variable: v ∈ R^4 (actuator rate, i.e., du/dt)
  
The safety constraint has relative degree 3 in the augmented system.
Additional state constraints (e.g., AoA) have relative degree 2.

Reference: README_stateAugmentation.md
"""

import numpy as np
import cvxopt

cvxopt.solvers.options['show_progress'] = False


def solve_cbf_qp(h, grad_h, hess_h, f, f_x, f_u, u_ref, u_current, 
                 alpha_1=0.1, alpha_2=3.0, alpha_3=15.0, 
                 u_limits=None, alpha_u=50.0,
                 x_state=None, 
                 alpha_max=np.deg2rad(8), alpha_aoa_1=10.0, alpha_aoa_2=15.0,
                 slack_penalty_aoa=300.0, Kp=5.0, dt=0.025):
    """
    Solves the HOCBF-QP for the augmented system with relative degree 3 safety constraint.
    
    Args:
        h:                  Barrier function value ψ₀ = h(x)
        grad_h:             Gradient ∇ₓh (12,)
        hess_h:             Hessian Hₕ (12, 12)
        f:                  Drift dynamics f(x,u) (12,)
        f_x:                Jacobian ∂f/∂x (12, 12)
        f_u:                Jacobian ∂f/∂u (12, 4)
        u_ref:              Reference control input
        u_current:          Current control input (part of augmented state)
        alpha_1, alpha_2, alpha_3:  Class-K function coefficients for HOCBF
        u_limits:           Actuator bounds (n_inputs, 2) with [:, 0]=min, [:, 1]=max
        alpha_u:            Aggressiveness for input limit constraints
        x_state:            Full state for additional constraints (e.g., AoA)
        alpha_max:          Maximum angle of attack limit
        alpha_aoa_1, alpha_aoa_2:   Class-K coefficients for AoA constraint
        slack_penalty_aoa:  Penalty for AoA slack variable (higher = harder constraint)
        Kp:                 Proportional gain for nominal control rate
        dt:                 Integration timestep
        
    Returns:
        u_safe_correction:  Safe control correction (u_optimal - u_ref)
        h:                  Barrier function value ψ₀ = h(x)
        h_dot:              First time derivative of h
        h_ddot:             Second time derivative of h (drift part)
        psi_2:              Second order barrier function value
        L_G_psi2:           Control coefficient matrix
    """
    f = f.astype(np.float64)
    f_u = f_u.astype(np.float64)
    hess_h = hess_h.astype(np.float64)
    grad_h = grad_h.astype(np.float64)
    u_current = u_current.astype(np.float64)
    u_ref = u_ref.astype(np.float64)
    
    n_inputs = u_ref.shape[0]
    use_aoa_slack = x_state is not None

    # =========================================================================
    # HOCBF Sequence for Safety Constraint (Relative Degree 3)
    # =========================================================================
    
    # ψ₀ = h(x)
    psi_0 = h
    
    # ψ₁ = ∇ₓh·f + α₁·h
    h_dot = grad_h.T @ f
    psi_1 = h_dot + alpha_1 * psi_0
    
    # ψ₂ = f^T·Hₕ·f + ∇ₓh·fₓ·f + (α₁+α₂)·ḣ + α₁α₂·h
    grad_h_fx = grad_h.T @ f_x
    h_ddot_drift = f.T @ hess_h @ f + grad_h_fx @ f
    psi_2 = h_ddot_drift + (alpha_1 + alpha_2) * h_dot + alpha_1 * alpha_2 * psi_0
    
    # Control coefficient: L_G ψ₂ = (2f^T·Hₕ + ∇ₓh·fₓ)·fᵤ
    L_G_psi2 = (2 * f.T @ hess_h + grad_h_fx) @ f_u
    
    # Drift term: L_F ψ₂ (jerk contribution from state dynamics)
    # L_F ψ₂ = 3f^T·Hₕ·fₓ·f + ∇ₓh·fₓ·fₓ·f + (α₁+α₂)(f^T·Hₕ·f + ∇ₓh·fₓ·f) + α₁α₂·∇ₓh·f
    L_F_psi2 = (3 * f.T @ hess_h @ f_x @ f + 
                grad_h_fx @ f_x @ f + 
                (alpha_1 + alpha_2) * h_ddot_drift + 
                alpha_1 * alpha_2 * h_dot)
    
    # Safety constraint: ψ₃ = L_F ψ₂ + L_G ψ₂·ν + α₃·ψ₂ ≥ 0
    # Rewritten as: -L_G ψ₂·ν ≤ L_F ψ₂ + α₃·ψ₂
    A_safe = -L_G_psi2.reshape(1, -1)
    b_safe = (L_F_psi2 + alpha_3 * psi_2).reshape(-1, 1)

    # =========================================================================
    # Angle of Attack Constraint (Relative Degree 2)
    # h_α = α_max - arctan(w/u) ≥ 0
    # =========================================================================
    
    A_aoa, b_aoa = None, None
    if x_state is not None:
        u_body, w_body = x_state[6], x_state[8]
        denom = u_body**2 + w_body**2 + 1e-8
        
        h_aoa = alpha_max - np.arctan2(w_body, u_body)
        
        # ∇ₓh_α: non-zero only at velocity indices (6, 8)
        grad_h_aoa = np.zeros(12)
        grad_h_aoa[6] = w_body / denom
        grad_h_aoa[8] = -u_body / denom
        
        h_dot_aoa = grad_h_aoa @ f
        
        # Hessian H_α: 2x2 block at indices (6,8)
        hess_h_aoa = np.zeros((12, 12))
        hess_h_aoa[6, 6] = 2 * u_body * w_body / (denom**2)
        hess_h_aoa[6, 8] = (w_body**2 - u_body**2) / (denom**2)
        hess_h_aoa[8, 6] = hess_h_aoa[6, 8]
        hess_h_aoa[8, 8] = -2 * u_body * w_body / (denom**2)
        
        # Relative degree 2: A = -∇ₓh·fᵤ, b = ∇ₓh·fₓ·f + f^T·Hₕ·f + (α₁+α₂)ḣ + α₁α₂h
        A_aoa_base = -(grad_h_aoa @ f_u).reshape(1, -1)
        b_aoa_val = (grad_h_aoa @ f_x @ f + 
                     f.T @ hess_h_aoa @ f + 
                     (alpha_aoa_1 + alpha_aoa_2) * h_dot_aoa + 
                     alpha_aoa_1 * alpha_aoa_2 * h_aoa)
        
        # Soft constraint with slack: A·ν - s ≤ b
        A_aoa = np.hstack([A_aoa_base, np.array([[-1.0]])])
        b_aoa = np.array([[b_aoa_val]])

    # =========================================================================
    # QP Cost Function: min (1/2)||ν - ν_nom||² + (ρ/2)s²
    # =========================================================================
    
    nu_nom = -Kp * (u_current - u_ref)
    
    if use_aoa_slack:
        P_mat = np.zeros((n_inputs + 1, n_inputs + 1))
        P_mat[:n_inputs, :n_inputs] = 2.0 * np.eye(n_inputs)
        P_mat[n_inputs, n_inputs] = 2.0 * slack_penalty_aoa
        
        q_vec = np.zeros(n_inputs + 1)
        q_vec[:n_inputs] = -2.0 * nu_nom
    else:
        P_mat = 2.0 * np.eye(n_inputs)
        q_vec = -2.0 * nu_nom
    
    P = cvxopt.matrix(P_mat)
    q = cvxopt.matrix(q_vec)

    # =========================================================================
    # Assemble Inequality Constraints: A·z ≤ b
    # =========================================================================
    
    # Safety constraint (extend with zero column for slack if needed)
    if use_aoa_slack:
        A_total = np.hstack([A_safe, np.zeros((1, 1))])
    else:
        A_total = A_safe
    b_total = b_safe

    # Input limits as CBF: I·ν ≤ αᵤ(u_max - u), -I·ν ≤ αᵤ(u - u_min)
    if u_limits is not None:
        u_min, u_max = u_limits[:, 0], u_limits[:, 1]
        
        A_upper = np.eye(n_inputs)
        A_lower = -np.eye(n_inputs)
        b_upper = (alpha_u * (u_max - u_current)).reshape(-1, 1)
        b_lower = (alpha_u * (u_current - u_min)).reshape(-1, 1)
        
        if use_aoa_slack:
            A_upper = np.hstack([A_upper, np.zeros((n_inputs, 1))])
            A_lower = np.hstack([A_lower, np.zeros((n_inputs, 1))])
        
        A_total = np.vstack([A_total, A_upper, A_lower])
        b_total = np.vstack([b_total, b_upper, b_lower])
        
    # AoA soft constraint + slack non-negativity (s ≥ 0)
    if A_aoa is not None:
        A_total = np.vstack([A_total, A_aoa])
        b_total = np.vstack([b_total, b_aoa])
        
        slack_noneg = np.zeros((1, n_inputs + 1))
        slack_noneg[0, n_inputs] = -1.0
        A_total = np.vstack([A_total, slack_noneg])
        b_total = np.vstack([b_total, np.array([[0.0]])])
        
    A = cvxopt.matrix(A_total)
    b = cvxopt.matrix(b_total)

    # =========================================================================
    # Solve QP and Extract Solution
    # =========================================================================
    
    try:
        sol = cvxopt.solvers.qp(P, q, A, b)
        
        if sol['status'] == 'optimal':
            sol_array = np.array(sol['x']).flatten()
            nu_optimal = sol_array[:n_inputs]
            
            if use_aoa_slack and sol_array[n_inputs] > 1e-4:
                print(f"CBF-QP: AoA slack active: s = {sol_array[n_inputs]:.4f}")
            
            # Integrate: u_new = u_current + ν·dt
            u_optimal = u_current + nu_optimal * dt
            u_safe_correction = u_optimal - u_ref
        else:
            print("CBF-QP Solver Warning: Solver did not find optimal solution.")
            u_safe_correction = np.zeros_like(u_ref)

    except ValueError:
        print("CBF-QP Solver Error: Exception during QP solving.")
        u_safe_correction = np.zeros_like(u_ref)

    return u_safe_correction, psi_0, h_dot, h_ddot_drift, psi_2, L_G_psi2



















def solve_cbf_qp_old(h, grad_h, hess_h, f, f_x, f_u, u_ref, alpha_1=1.0, alpha_2=1.0):
    """
    Analytical CBF solution for relative degree 2 (no QP solver).
    Kept for timing comparison with the full QP-based approach.
    
    Args:
        h:			Barrier function value
        grad_h:		Gradient ∇ₓh (12,)
        hess_h:		Hessian Hₕ (12, 12)
        f:			Drift dynamics f(x,u) (12,)
        f_x:		Jacobian ∂f/∂x (12, 12)
        f_u:		Jacobian ∂f/∂u (12, 4)
        u_ref:		Reference control input
        alpha_1, alpha_2:	Class-K function coefficients
        
    Returns:
        u_safe:		Safe control correction
        psi:		Constraint value (ḧ + 2α₁ḣ + α₂h)
        h_ddot:		Second time derivative of h
        h_dot:		First time derivative of h
        L_G_h:		Control coefficient
    
    """
    h_dot = grad_h.T @ f    
    h_ddot = f.T @ hess_h @ f + grad_h.T @ f_x @ f
    
    psi = h_ddot + 2 * alpha_1 * h_dot + alpha_2 * h
    L_G_h = (f.T @ hess_h + grad_h.T @ f_x) @ f_u
    
    if psi >= 0:
        u_safe = np.zeros_like(u_ref)      
    else:
        if np.linalg.norm(L_G_h) < 1e-6:
            u_safe = np.zeros_like(u_ref)
        else:
            u_safe = -L_G_h.T / (np.linalg.norm(L_G_h)**2) * psi

    return u_safe, psi, h_ddot, h_dot, L_G_h


# Example call (for timing comparison):
# u_safe_old, psi_old, h_ddot_old, h_dot_old, L_G_old = solve_cbf_qp_old(
#     h=h,
#     grad_h=grad_h,
#     hess_h=hess_h,
#     f=f,
#     f_x=f_x,
#     f_u=model.h(x),  # This is f_u from the model
#     u_ref=udef_now,
#     alpha_1=1.0,
#     alpha_2=1.0
# )