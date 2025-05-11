# cd one dir back
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# -----------------------------------------------------------------------------------
from my_erg_lib.basis import Basis
import numpy as np
import time

def phiExample(s, L1=1.0, L2=1.0):
    # Complex function with multiple peaks, valleys, and non-linearities
    x, y = s[0], s[1]
    
    # Multiple Gaussian bumps
    # Generate random bump positions within the L1, L2 boundaries
    bump_positions = [
        (0.2 * L1, 0.3 * L2), 
        (0.7 * L1, 0.8 * L2), 
        (0.5 * L1, 0.1 * L2), 
        (0.9 * L1, 0.5 * L2)
    ]
    bump_heights = [3, 4, 2, 5]
    bump_widths = [30, 40, 25, 35]
    
    bumps = 0
    for i in range(len(bump_positions)):
        pos_x, pos_y = bump_positions[i]
        height = bump_heights[i]
        width = bump_widths[i]
        bumps += height * np.exp(-width * ((x-pos_x)**2 + (y-pos_y)**2))
    
    # Sinusoidal variations
    waves = 2 * np.sin(8 * np.pi * x) * np.cos(6 * np.pi * y)
    
    # Polynomial trend
    trend = (x - 0.4)**2 * (y - 0.6)**2 * 5
    
    # Sharp ridge
    ridge = 3 * np.exp(-100 * (x - y)**2)
    
    # Combine all components
    return bumps + 2 + waves + trend + ridge

# Function to be used for phi with specific L1 and L2 values
def phi_func1(s):
    return phiExample(s, L1=2.0, L2=2.0)
def phi_func2(s):
    return phiExample(s, L1=2.0, L2=2.0)

# Test different numbers of Gauss points
gauss_points_to_test = [2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 55, 60, 70, 80, 90, 100]
max_diff_results = []
time_diff_results = []

# Reference basis with nquad (high accuracy)
start_time_nquad = time.time()
b_nquad = Basis(L1=2.0, L2=2.0, Kmax=3, phi_=phi_func2, integration_method='nquad')
nquad_time = time.time() - start_time_nquad

k1 = np.linspace(0, 3, 4)
k2 = np.linspace(0, 3, 4)

# Calculate results for each number of Gauss points
for num_points in gauss_points_to_test:
    # Create basis with current number of Gauss points and measure time
    start_time_gauss = time.time()
    b_gauss = Basis(L1=2.0, L2=2.0, Kmax=3, phi_=phi_func1, 
                    integration_method='gauss', num_gauss_points=num_points)
    gauss_time = time.time() - start_time_gauss
    
    # Calculate max percentage difference for accuracy
    max_diff_perc = 0
    for i in range(len(k1)):
        for j in range(len(k2)):
            nquad_val = b_nquad.phi_coeff_cache[(k1[i], k2[j])]
            gauss_val = b_gauss.phi_coeff_cache[(k1[i], k2[j])]
            
            if abs(nquad_val) > 1e-10:  # Avoid division by zero
                diff_perc = abs(nquad_val - gauss_val) / abs(nquad_val) * 100
            else:
                diff_perc = 0
                
            max_diff_perc = max(max_diff_perc, diff_perc)
    
    # Calculate time difference percentage
    how_many_times_faster = nquad_time / gauss_time - 1
    # time_diff_perc = (nquad_time - gauss_time) / nquad_time * 100
    
    max_diff_results.append(max_diff_perc)
    time_diff_results.append(how_many_times_faster)
    
    print(f"Number of Gauss points: {num_points}, Max difference: {max_diff_perc:.4f}f, Time difference: {how_many_times_faster:.2f}, Time: {gauss_time:.4f} seconds")

# Plot the results
import matplotlib.pyplot as plt

# Plot for accuracy comparison
plt.figure(figsize=(10, 6))
plt.plot(gauss_points_to_test, max_diff_results, 'o-', linewidth=2)
plt.xlabel('Number of Gauss Integration Points')
plt.ylabel('Maximum Percentage Difference (%)')
plt.title('Error vs Number of Gauss Integration Points')
plt.grid(True)
plt.yscale('log')  # Log scale often better shows convergence
plt.axhline(y=0.1, color='r', linestyle='--', label='0.1% Error Threshold')
plt.legend()
plt.tight_layout()
# plt.savefig('gauss_accuracy_comparison.png')

# Plot for time comparison
plt.figure(figsize=(10, 6))
plt.plot(gauss_points_to_test, time_diff_results, 'o-', linewidth=2, color='green')
plt.xlabel('Number of Gauss Integration Points')
plt.ylabel('How many times faster than NQuad')
plt.title('Time Saved vs Number of Gauss Integration Points')
plt.grid(True)
plt.axhline(y=0, color='r', linestyle='--', label='No Time Difference')
plt.legend()
plt.tight_layout()
# plt.savefig('gauss_time_comparison.png')

plt.show()

# Also create a detailed comparison for the best number of points
# Find the best number of points closest to the 0.1% threshold
threshold = 0.1
differences_from_threshold = [abs(diff - threshold) for diff in max_diff_results]
best_num_points = gauss_points_to_test[differences_from_threshold.index(min(differences_from_threshold))]
print(f"\nDetailed comparison with best number of points ({best_num_points}):")
print(f"NQuad computation time: {nquad_time:.4f} seconds")
print(f"Gauss computation time with {best_num_points} points: {nquad_time * (1 - time_diff_results[gauss_points_to_test.index(best_num_points)]/100):.4f} seconds")
print(f"Time saved: {time_diff_results[gauss_points_to_test.index(best_num_points)]:.2f}%")

b_gauss = Basis(L1=2.0, L2=2.0, Kmax=3, phi_=phi_func1, 
                integration_method='gauss', num_gauss_points=best_num_points)

for i in range(len(k1)):
    for j in range(len(k2)):
        nquad_val = b_nquad.phi_coeff_cache[(k1[i], k2[j])]
        gauss_val = b_gauss.phi_coeff_cache[(k1[i], k2[j])]
        
        if abs(nquad_val) > 1e-10:
            diff_perc = abs(nquad_val - gauss_val) / abs(nquad_val) * 100
        else:
            diff_perc = 0
            
        print(f"({k1[i]}, {k2[j]}) = {nquad_val:.6f} (NQuad), {gauss_val:.6f} (Gauss) \t --> Diff: {diff_perc:.6f}%")
