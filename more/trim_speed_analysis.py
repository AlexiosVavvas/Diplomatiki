#!/usr/bin/env python3
"""
Trim Speed Analysis Tool
========================
Analyzes trim conditions at various airspeeds to help choose optimal cruise speed.
Computes trim at multiple speeds above stall and compares performance metrics.

Usage:
    python trim_speed_analysis.py
"""
import sys
sys.path.append('/home/avavvas/dipl/src/ergodic_exploration')

import numpy as np
import matplotlib.pyplot as plt
from my_erg_lib.model_dynamics import FixedWing12DOFTrainer # type: ignore

def analyze_trim_speeds():
    """
    Compute trim at various speeds and compare performance metrics.
    """
    
    # Create a reference aircraft to get parameters
    plane_ref = FixedWing12DOFTrainer(dt=0.001, v_trim=10.0)
    P = plane_ref.params
    
    # Extract aircraft parameters from model
    m = P['m']           # kg
    S = P['S']           # m^2
    rho = P['rho']       # kg/m^3
    g = 9.81             # m/s^2
    Weight = m * g       # N
    
    # Extract aerodynamic parameters from model
    CL0 = P['CL0']
    CL_alpha = P['CL_alpha']
    CD0 = P['CD0']
    k = P['k']
    T_max = P['T_max']   # N
    
    # Estimate stall speed (assuming max alpha ~15-17 degrees before stall)
    alpha_max = 15 * np.pi / 180  # rad
    CL_max = CL0 + CL_alpha * alpha_max
    V_stall = np.sqrt(2 * Weight / (rho * S * CL_max))
    
    print("=" * 70)
    print("TRIM SPEED ANALYSIS")
    print("=" * 70)
    print(f"Aircraft: m={m} kg, S={S} m², Wing loading={Weight/S:.1f} N/m²")
    print(f"Estimated stall speed: V_stall = {V_stall:.2f} m/s")
    print(f"Minimum safe speed (1.2×V_stall): {1.2*V_stall:.2f} m/s")
    print("=" * 70)
    print()
    
    # Test speeds: 1.2×, 1.5×, 2×, 2.5×, 3× stall speed
    speed_multiples = np.linspace(1.2, 3.5, 20)
    test_speeds = [mult * V_stall for mult in speed_multiples]
    
    # Storage for results
    results = []
    
    print(f"{'V (m/s)':<9} {'α (°)':<8} {'θ (°)':<8} {'CL':<8} {'CD':<8} "
          f"{'L/D':<8} {'Throttle':<10} {'δe (°)':<8} {'Power (W)':<10} {'Status':<15}")
    print("-" * 110)
    
    for V_trim in test_speeds:
        # Create aircraft model
        plane = FixedWing12DOFTrainer(dt=0.001, v_trim=V_trim)
        
        # Compute trim
        try:
            x_trim, u_trim, sol = plane.computeTrim(V_trim=V_trim)
            
            if not sol.success or np.linalg.norm(sol.fun) > 1e-6:
                print(f"{V_trim:9.2f} {'FAILED':^50s} Trim solver did not converge")
                continue
            
            # Extract trim state
            u, v, w = x_trim[6], x_trim[7], x_trim[8]
            theta = x_trim[4]
            delta_e, throttle = u_trim[0], u_trim[3]
            
            # Compute aerodynamic parameters
            V = np.sqrt(u**2 + v**2 + w**2)
            alpha = np.arctan2(w, u)
            qbar = 0.5 * P['rho'] * V**2
            
            CL = P['CL0'] + P['CL_alpha'] * alpha
            CD = P['CD0'] + P['k'] * CL**2
            LD_ratio = CL / CD
            
            # Compute forces
            Lift = qbar * P['S'] * CL
            Drag = qbar * P['S'] * CD
            Thrust = P['T_max'] * throttle
            
            # Power required
            Power = Thrust * V  # Watts
            
            # Force balance check
            Lift_vertical = Lift * np.cos(alpha)
            lift_error = abs(Lift_vertical - Weight) / Weight * 100
            thrust_error = abs(Thrust - Drag) / max(Thrust, 0.1) * 100
            
            # Status
            if V < 1.2 * V_stall:
                status = "⚠️ TOO SLOW"
            elif throttle > 0.8:
                status = "⚠️ HIGH THROTTLE"
            elif alpha > 0.20:  # ~11 degrees
                status = "⚠️ HIGH AoA"
            elif lift_error > 2 or thrust_error > 10:
                status = "⚠️ POOR BALANCE"
            else:
                status = "✓ GOOD"
            
            # Store results
            results.append({
                'V': V,
                'alpha': alpha,
                'theta': theta,
                'CL': CL,
                'CD': CD,
                'LD': LD_ratio,
                'throttle': throttle,
                'delta_e': delta_e,
                'Power': Power,
                'Lift': Lift,
                'Drag': Drag,
                'Thrust': Thrust,
                'lift_error': lift_error,
                'thrust_error': thrust_error,
                'status': status
            })
            
            # Print row
            print(f"{V:9.2f} {alpha*180/np.pi:8.2f} {theta*180/np.pi:8.2f} "
                  f"{CL:8.3f} {CD:8.4f} {LD_ratio:8.2f} {throttle:10.1%} "
                  f"{delta_e*180/np.pi:8.2f} {Power:10.1f} {status:<15}")
            
        except Exception as e:
            print(f"{V_trim:9.2f} {'ERROR':^50s} {str(e)[:50]}")
    
    print("-" * 110)
    print()
    
    if not results:
        print("ERROR: No successful trim solutions found!")
        return
    
    # Find optimal speeds
    print("=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    
    # Best L/D (most efficient)
    best_efficiency = max(results, key=lambda x: x['LD'])
    print(f"\n🏆 BEST EFFICIENCY (max L/D):")
    print(f"   Speed: {best_efficiency['V']:.2f} m/s")
    print(f"   L/D: {best_efficiency['LD']:.2f}")
    print(f"   Throttle: {best_efficiency['throttle']:.1%}")
    print(f"   Power: {best_efficiency['Power']:.1f} W")
    print(f"   AoA: {best_efficiency['alpha']*180/np.pi:.2f}°")
    
    # Minimum power
    min_power = min(results, key=lambda x: x['Power'])
    print(f"\n⚡ MINIMUM POWER:")
    print(f"   Speed: {min_power['V']:.2f} m/s")
    print(f"   Power: {min_power['Power']:.1f} W")
    print(f"   Throttle: {min_power['throttle']:.1%}")
    print(f"   L/D: {min_power['LD']:.2f}")
    print(f"   AoA: {min_power['alpha']*180/np.pi:.2f}°")
    
    # Recommended cruise (balance of efficiency and speed)
    # Look for speeds with good L/D (>80% of max) and reasonable speed
    good_efficiency = [r for r in results if r['LD'] > 0.8 * best_efficiency['LD']]
    if good_efficiency:
        recommended = max(good_efficiency, key=lambda x: x['V'])
        print(f"\n✈️  RECOMMENDED CRUISE (high speed + good efficiency):")
        print(f"   Speed: {recommended['V']:.2f} m/s")
        print(f"   L/D: {recommended['LD']:.2f} ({recommended['LD']/best_efficiency['LD']*100:.0f}% of best)")
        print(f"   Throttle: {recommended['throttle']:.1%}")
        print(f"   Power: {recommended['Power']:.1f} W")
        print(f"   AoA: {recommended['alpha']*180/np.pi:.2f}°")
    
    print("\n" + "=" * 70)
    
    # Plot results
    plot_trim_analysis(results, V_stall)
    
    return results


def plot_trim_analysis(results, V_stall):
    """
    Create plots comparing trim conditions at different speeds.
    """
    if not results:
        return
    
    # Extract data
    speeds = [r['V'] for r in results]
    alphas = [r['alpha'] * 180/np.pi for r in results]
    thetas = [r['theta'] * 180/np.pi for r in results]
    CLs = [r['CL'] for r in results]
    CDs = [r['CD'] for r in results]
    LDs = [r['LD'] for r in results]
    throttles = [r['throttle'] * 100 for r in results]
    powers = [r['Power'] for r in results]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Trim Speed Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Angle of Attack
    ax = axes[0, 0]
    ax.plot(speeds, alphas, 'b.-', linewidth=2, markersize=8)
    ax.axvline(V_stall, color='r', linestyle='--', alpha=0.5, label='Stall speed')
    ax.axvline(1.2 * V_stall, color='orange', linestyle='--', alpha=0.5, label='Min safe (1.2×)')
    ax.set_xlabel('Airspeed (m/s)')
    ax.set_ylabel('Angle of Attack (°)')
    ax.set_title('Trim Angle of Attack')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: Lift and Drag Coefficients
    ax = axes[0, 1]
    ax.plot(speeds, CLs, 'b.-', linewidth=2, markersize=8, label='CL')
    ax.plot(speeds, CDs, 'r.-', linewidth=2, markersize=8, label='CD')
    ax.axvline(V_stall, color='r', linestyle='--', alpha=0.3)
    ax.axvline(1.2 * V_stall, color='orange', linestyle='--', alpha=0.3)
    ax.set_xlabel('Airspeed (m/s)')
    ax.set_ylabel('Coefficient')
    ax.set_title('Lift & Drag Coefficients')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 3: L/D Ratio
    ax = axes[0, 2]
    ax.plot(speeds, LDs, 'g.-', linewidth=2, markersize=8)
    ax.axvline(V_stall, color='r', linestyle='--', alpha=0.3)
    ax.axvline(1.2 * V_stall, color='orange', linestyle='--', alpha=0.3)
    max_LD_idx = LDs.index(max(LDs))
    ax.axvline(speeds[max_LD_idx], color='g', linestyle=':', alpha=0.5, label=f'Best L/D: {max(LDs):.1f}')
    ax.set_xlabel('Airspeed (m/s)')
    ax.set_ylabel('L/D Ratio')
    ax.set_title('Aerodynamic Efficiency (L/D)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 4: Throttle Setting
    ax = axes[1, 0]
    ax.plot(speeds, throttles, 'm.-', linewidth=2, markersize=8)
    ax.axvline(V_stall, color='r', linestyle='--', alpha=0.3)
    ax.axvline(1.2 * V_stall, color='orange', linestyle='--', alpha=0.3)
    ax.axhline(50, color='gray', linestyle=':', alpha=0.5, label='50% throttle')
    ax.set_xlabel('Airspeed (m/s)')
    ax.set_ylabel('Throttle (%)')
    ax.set_title('Trim Throttle Setting')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 5: Power Required
    ax = axes[1, 1]
    ax.plot(speeds, powers, 'c.-', linewidth=2, markersize=8)
    ax.axvline(V_stall, color='r', linestyle='--', alpha=0.3)
    ax.axvline(1.2 * V_stall, color='orange', linestyle='--', alpha=0.3)
    min_power_idx = powers.index(min(powers))
    ax.axvline(speeds[min_power_idx], color='c', linestyle=':', alpha=0.5, label=f'Min power: {min(powers):.0f}W')
    ax.set_xlabel('Airspeed (m/s)')
    ax.set_ylabel('Power (W)')
    ax.set_title('Power Required')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 6: Pitch Attitude
    ax = axes[1, 2]
    ax.plot(speeds, thetas, 'k.-', linewidth=2, markersize=8)
    ax.axvline(V_stall, color='r', linestyle='--', alpha=0.3)
    ax.axvline(1.2 * V_stall, color='orange', linestyle='--', alpha=0.3)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5, label='Level')
    ax.set_xlabel('Airspeed (m/s)')
    ax.set_ylabel('Pitch Angle (°)')
    ax.set_title('Trim Pitch Attitude')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    # Save figure
    filename = '/home/avavvas/dipl/trim_speed_analysis.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n📊 Plot saved to: {filename}")
    
    plt.show()


if __name__ == "__main__":
    try:
        results = analyze_trim_speeds()
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
