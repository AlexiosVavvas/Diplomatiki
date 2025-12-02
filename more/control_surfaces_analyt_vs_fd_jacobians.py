#!/usr/bin/env python3
"""
Simple script to validate control surface Jacobians by comparing
analytical derivatives vs finite differences for the airplane model.
"""

import numpy as np
import sys
sys.path.append('/home/avavvas/dipl/src/ergodic_exploration')
from my_erg_lib.model_dynamics import FixedWing12DOFTrainer
# from src.ergodic_exploration.my_erg_lib.model_dynamics import FixedWing12DOFTrainer


def main():
    # Initialize airplane at default trim speed
    airplane = FixedWing12DOFTrainer(dt=0.01, v_trim=16.0)
    
    # Use trim state as test point
    x_new = airplane.x_trim.copy()
    u0 = airplane.u_trim.copy()
    
    # Get analytical Jacobian
    B_analytical = airplane.f_u(x_new)
    B_analytical[:, :3] *= np.pi / 180.0  # convert from per-rad to per-deg for control surfaces
    
    # Finite difference parameters
    delta_deg = 1e-3  # small perturbation in degrees for control surfaces
    delta_rad = delta_deg * np.pi / 180.0  # convert to radians for actual perturbation
    delta_throttle = 1e-3  # small perturbation for throttle (unitless)

    # Angular Accelerations Table
    print("\n" + "="*80)
    print("CONTROL JACOBIAN VALIDATION (Angular Accelerations)")
    print("="*80)
    print(f"{'Control':<12} {'Method':<15} {'Δp_dot':<12} {'Δq_dot':<12} {'Δr_dot':<12}")
    print("-"*80)
    
    # Elevator
    d = airplane.f(x_new, u0 + np.array([delta_rad,0,0,0])) - airplane.f(x_new, u0)

    print(f"{'delta_e':<12} {'Finite Diff':<15} {d[9]/delta_deg:>11.3f} {d[10]/delta_deg:>11.3f} {d[11]/delta_deg:>11.3f}")
    print(f"{'':12} {'Analytical':<15} {B_analytical[9,0]:>11.3f} {B_analytical[10,0]:>11.3f} {B_analytical[11,0]:>11.3f}")
    print(f"{'':12} {'Error':<15} {abs(d[9]/delta_deg - B_analytical[9,0]):>11.3e} {abs(d[10]/delta_deg - B_analytical[10,0]):>11.3e} {abs(d[11]/delta_deg - B_analytical[11,0]):>11.3e}")
    print("-"*80)
    
    # Aileron
    d = airplane.f(x_new, u0 + np.array([0,delta_rad,0,0])) - airplane.f(x_new, u0)
    print(f"{'delta_a':<12} {'Finite Diff':<15} {d[9]/delta_deg:>11.3f} {d[10]/delta_deg:>11.3f} {d[11]/delta_deg:>11.3f}")
    print(f"{'':12} {'Analytical':<15} {B_analytical[9,1]:>11.3f} {B_analytical[10,1]:>11.3f} {B_analytical[11,1]:>11.3f}")
    print(f"{'':12} {'Error':<15} {abs(d[9]/delta_deg - B_analytical[9,1]):>11.3e} {abs(d[10]/delta_deg - B_analytical[10,1]):>11.3e} {abs(d[11]/delta_deg - B_analytical[11,1]):>11.3e}")
    print("-"*80)
    
    # Rudder
    d = airplane.f(x_new, u0 + np.array([0,0,delta_rad,0])) - airplane.f(x_new, u0)
    print(f"{'delta_r':<12} {'Finite Diff':<15} {d[9]/delta_deg:>11.3f} {d[10]/delta_deg:>11.3f} {d[11]/delta_deg:>11.3f}")
    print(f"{'':12} {'Analytical':<15} {B_analytical[9,2]:>11.3f} {B_analytical[10,2]:>11.3f} {B_analytical[11,2]:>11.3f}")
    print(f"{'':12} {'Error':<15} {abs(d[9]/delta_deg - B_analytical[9,2]):>11.3e} {abs(d[10]/delta_deg - B_analytical[10,2]):>11.3e} {abs(d[11]/delta_deg - B_analytical[11,2]):>11.3e}")
    print("="*80 + "\n")

    # Linear Accelerations Table
    print("\n" + "="*80)
    print("CONTROL JACOBIAN VALIDATION (Linear Accelerations)")
    print("="*80)
    print(f"{'Control':<12} {'Method':<15} {'Δu_dot':<12} {'Δv_dot':<12} {'Δw_dot':<12}")
    print("-"*80)

    # Elevator
    d = airplane.f(x_new, u0 + np.array([delta_rad,0,0,0])) - airplane.f(x_new, u0)
    print(f"{'delta_e':<12} {'Finite Diff':<15} {d[6]/delta_deg:>11.3f} {d[7]/delta_deg:>11.3f} {d[8]/delta_deg:>11.3f}")
    print(f"{'':12} {'Analytical':<15} {B_analytical[6,0]:>11.3f} {B_analytical[7,0]:>11.3f} {B_analytical[8,0]:>11.3f}")
    print(f"{'':12} {'Error':<15} {abs(d[6]/delta_deg - B_analytical[6,0]):>11.3e} {abs(d[7]/delta_deg - B_analytical[7,0]):>11.3e} {abs(d[8]/delta_deg - B_analytical[8,0]):>11.3e}")
    print("-"*80)

    # Aileron
    d = airplane.f(x_new, u0 + np.array([0,delta_rad,0,0])) - airplane.f(x_new, u0)
    print(f"{'delta_a':<12} {'Finite Diff':<15} {d[6]/delta_deg:>11.3f} {d[7]/delta_deg:>11.3f} {d[8]/delta_deg:>11.3f}")
    print(f"{'':12} {'Analytical':<15} {B_analytical[6,1]:>11.3f} {B_analytical[7,1]:>11.3f} {B_analytical[8,1]:>11.3f}")
    print(f"{'':12} {'Error':<15} {abs(d[6]/delta_deg - B_analytical[6,1]):>11.3e} {abs(d[7]/delta_deg - B_analytical[7,1]):>11.3e} {abs(d[8]/delta_deg - B_analytical[8,1]):>11.3e}")
    print("-"*80)

    # Rudder
    d = airplane.f(x_new, u0 + np.array([0,0,delta_rad,0])) - airplane.f(x_new, u0)
    print(f"{'delta_r':<12} {'Finite Diff':<15} {d[6]/delta_deg:>11.3f} {d[7]/delta_deg:>11.3f} {d[8]/delta_deg:>11.3f}")
    print(f"{'':12} {'Analytical':<15} {B_analytical[6,2]:>11.3f} {B_analytical[7,2]:>11.3f} {B_analytical[8,2]:>11.3f}")
    print(f"{'':12} {'Error':<15} {abs(d[6]/delta_deg - B_analytical[6,2]):>11.3e} {abs(d[7]/delta_deg - B_analytical[7,2]):>11.3e} {abs(d[8]/delta_deg - B_analytical[8,2]):>11.3e}")
    print("-"*80)

    # Throttle
    d = airplane.f(x_new, u0 + np.array([0,0,0,delta_throttle])) - airplane.f(x_new, u0)
    print(f"{'throttle':<12} {'Finite Diff':<15} {d[6]/delta_throttle:>11.3f} {d[7]/delta_throttle:>11.3f} {d[8]/delta_throttle:>11.3f}")
    print(f"{'':12} {'Analytical':<15} {B_analytical[6,3]:>11.3f} {B_analytical[7,3]:>11.3f} {B_analytical[8,3]:>11.3f}")
    print(f"{'':12} {'Error':<15} {abs(d[6]/delta_throttle - B_analytical[6,3]):>11.3e} {abs(d[7]/delta_throttle - B_analytical[7,3]):>11.3e} {abs(d[8]/delta_throttle - B_analytical[8,3]):>11.3e}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
