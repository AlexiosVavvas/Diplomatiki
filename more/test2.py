# cd one dir back
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# -----------------------------------------------------------------------------------
import numpy as np
from scipy.integrate import nquad
import matplotlib.pyplot as plt
from matplotlib import cm

L1 = 1; L2 = 1
Kmax = 5

# Dictionary to store precomputed values
hk_cache = {}
def calcHk(k1, k2):
    if (k1, k2) in hk_cache:
        return hk_cache[(k1, k2)]
    
    if k1==0 and k2==0:
        hk = L1 * L2
        hk = np.sqrt(hk)
    elif k1==0 and k2!=0:
        hk = L2 * (2*k2*L1*np.pi + L2*np.sin(2*k2*L1*np.pi/L2)) / (4 * L2 * np.pi)
        hk = np.sqrt(hk)
    elif k1!=0 and k2==0:
        hk = L1 * (2*k1*L2*np.pi + L1*np.sin(2*k1*L2*np.pi/L1)) / (4 * L1 * np.pi)
        hk = np.sqrt(hk)
    else:
        hk = (2*k2*L1*np.pi + L2*np.sin(2*k2*L1*np.pi/L2)) * (2*k1*L2*np.pi + L1*np.sin(2*k1*L2*np.pi/L1)) / (4 * L1 * L2)
        hk /= 16 * k1 * k2 * np.pi**2
        hk = np.sqrt(hk)  # Take the square root of the integral

    # add to dictionary
    hk_cache[(k1, k2)] = hk

    return hk

    # def calcHk(k1, k2, n_points=100):
    #     # Check if the value is already computed
    #     if (k1, k2) in hk_cache:
    #         return hk_cache[(k1, k2)]
        
    #     # Use more efficient numerical integration with numpy's meshgrid and trapz
    #     x1 = np.linspace(0, L1, n_points)
    #     x2 = np.linspace(0, L2, n_points)
        
    #     X1, X2 = np.meshgrid(x1, x2)
    #     f_values = (np.cos(k1*np.pi/L1*X1))**2 * (np.cos(k2*np.pi/L2*X2))**2
        
    #     # Integrate using trapezoidal rule
    #     hk = np.trapz(np.trapz(f_values, x2, axis=0), x1)
    #     hk = np.sqrt(hk)  # Take the square root of the integral

    #     # Store the computed value
    #     hk_cache[(k1, k2)] = hk

    #     return hk

def Fk(s, k1, k2, hk):
    Fk = np.cos(k1*np.pi/L1*s[0]) * np.cos(k2*np.pi/L2*s[1]) / hk
    return Fk

def phi(s):
    # Complex function with multiple peaks, valleys, and non-linearities
    x, y = s[0], s[1]
    
    # Multiple Gaussian bumps
    bumps = 3 * np.exp(-30 * ((x-0.2)**2 + (y-0.3)**2)) + \
            4 * np.exp(-40 * ((x-0.7)**2 + (y-0.8)**2)) + \
            2 * np.exp(-25 * ((x-0.5)**2 + (y-0.1)**2)) + \
            5 * np.exp(-35 * ((x-0.9)**2 + (y-0.5)**2))
    
    # Sinusoidal variations
    waves = 2 * np.sin(8 * np.pi * x) * np.cos(6 * np.pi * y)
    
    # Polynomial trend
    trend = (x - 0.4)**2 * (y - 0.6)**2 * 5
    
    # Sharp ridge
    ridge = 3 * np.exp(-100 * (x - y)**2)
    
    # Combine all components
    return bumps + waves + trend + ridge

# Dictionary to store precomputed values
phi_coeff_cache = {}
def returnCoeff(k1, k2):
    # Check if the value is already computed
    if (k1, k2) in phi_coeff_cache:
        return phi_coeff_cache[(k1, k2)]
    
    hk = calcHk(k1, k2)
    phi_k, _ = nquad(lambda x1, x2: phi([x1, x2]) * Fk([x1, x2], k1, k2, hk),
               [[0, L1], [0, L2]])
    
    # Store the computed value
    phi_coeff_cache[(k1, k2)] = phi_k

    return phi_k

def returnPhiNew(s, Kmax = 5):
    phiNew = 0
    for k1 in range(0, Kmax+1):
        for k2 in range(0, Kmax+1):
            phiNew += returnCoeff(k1, k2) * Fk(s, k1, k2, calcHk(k1, k2))
    return phiNew



# =========================-----------------------------------------------
# # Generate Grid to test
N_points = 50
x_i = np.linspace(0, L1, N_points)
y_i = np.linspace(0, L2, N_points)
x_i = np.array(x_i)
y_i = np.array(y_i)

phi_old = np.zeros((N_points, N_points))
phi_new = np.zeros((N_points, N_points))

for i in range(N_points):
    for j in range(N_points):
        phi_old[i, j] = phi([x_i[i], y_i[j]])
        phi_new[i, j] = returnPhiNew([x_i[i], y_i[j]], Kmax=Kmax)



# Print the coefficients dictionary in a readable format
print("Fourier coefficients (k1, k2) -> value:")
for (k1, k2), value in sorted(phi_coeff_cache.items()):
    print(f"({k1}, {k2}): {value:.6f}")


# Figures 
plt.figure(1, figsize=(12, 5))

# Create a 1x2 subplot grid
plt.subplot(1, 2, 1)
plt.imshow(phi_old, cmap=cm.viridis, extent=[0, L1, 0, L2])
plt.colorbar(label='phi_old')
plt.title('Old phi')

plt.subplot(1, 2, 2)
plt.imshow(phi_new, cmap=cm.viridis, extent=[0, L1, 0, L2])
plt.colorbar(label='phi_new')
plt.title('New phi')

plt.tight_layout()  # Adjust spacing between subplots








# Lets plot the coefficients with imshow to see what makes a difference (phi_coeff_dict(k1, k2))
k1_ = np.arange(0, Kmax+1)
k2_ = np.arange(0, Kmax+1)
phi_coeff = np.zeros((len(k1_), len(k2_)))
for i in range(len(k1_)):
    for j in range(len(k2_)):
        phi_coeff[i, j] = returnCoeff(k1_[i], k2_[j])

plt.figure(3, figsize=(8, 6))
# Create meshgrid for pcolormesh
k1_mesh, k2_mesh = np.meshgrid(np.arange(Kmax+2)-0.5, np.arange(Kmax+2)-0.5)
# Use pcolormesh for clearer value placement
plt.pcolormesh(k1_mesh, k2_mesh, np.abs(phi_coeff), cmap=cm.viridis)
plt.colorbar(label='|phi_coeff|')
plt.title('Coefficient Magnitude by Mode (k1, k2)')
plt.xlabel('k1')
plt.ylabel('k2')
plt.xticks(np.arange(0, Kmax+1))
plt.yticks(np.arange(0, Kmax+1))
# Add text annotations with values
for i in range(Kmax+1):
    for j in range(Kmax+1):
        plt.text(i, j, f'{np.abs(phi_coeff[i,j]):.2f}', 
                 ha='center', va='center', color='white' if np.abs(phi_coeff[i,j]) > np.max(np.abs(phi_coeff))/2 else 'black')
plt.grid(False)








# Lets do a 3d plot surface of phi_new and phi_old in the same x1, x2 axes
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

X1, X2 = np.meshgrid(x_i, y_i)
# Calculate phi_old and phi_new for the grid
phi_old_surface = np.zeros_like(X1)
phi_new_surface = np.zeros_like(X1)
for i in range(N_points):
    for j in range(N_points):
        phi_old_surface[i, j] = phi_old[i, j]
        phi_new_surface[i, j] = phi_new[i, j]

ax.plot_surface(X1, X2, phi_old_surface, cmap=cm.viridis, alpha=0.5, label='phi_old')
ax.plot_surface(X1, X2, phi_new_surface, cmap=cm.Reds, alpha=0.5, label='phi_new')

plt.title('3D Surface Plot of phi_old and phi_new')
plt.xlabel('x1')
plt.ylabel('x2')



# =========================-----------------------------------------------
def xv(t, speed_deg_p_sec = 80, center=(0.5, 0.5), radius=0.3, spiral_out_speed=0.1):
    # Convert speed from degrees per second to radians per second
    speed_rad_p_sec = np.deg2rad(speed_deg_p_sec)
    
    # Calculate spiral coordinates
    # Start small and grow with time
    current_radius = np.minimum(radius, spiral_out_speed * t)
    
    # Spiral equations
    x = center[0] + current_radius * np.cos(speed_rad_p_sec * t)
    y = center[1] + current_radius * np.sin(speed_rad_p_sec * t)
    
    return x, y

t0 = 0      # Initial Time
T = 5         # [Sec] Horizon
t_final = T # [Sec] Final time

time = np.linspace(0, t_final, 100) # Time vector

ck_coeff_dict = {}
def calcCk(k1, k2):
    # Check if the value is already computed
    if (k1, k2) in ck_coeff_dict:
        return ck_coeff_dict[(k1, k2)]
    
    # Integrate using nquad Fk(x(t), y(t)) dt
    ck, _ = nquad(lambda t: Fk([*xv(t)], k1, k2, calcHk(k1, k2)), 
               [[t0, t0+T]])
    ck /= T

    # Store the computed value
    ck_coeff_dict[(k1, k2)] = ck

    return ck

k1_c = np.arange(0, Kmax+1)
k2_c = np.arange(0, Kmax+1)
ck_coeff = np.zeros((len(k1_c), len(k2_c)))

for i in range(len(k1_c)):
    for j in range(len(k2_c)):
        ck_coeff[i, j] = calcCk(k1_c[i], k2_c[j])

# Reconstruct Cnew 
def returnCnew(x, y):
    Cnew = 0
    for k1 in range(0, Kmax+1):
        for k2 in range(0, Kmax+1):
            Cnew += calcCk(k1, k2) * Fk([x, y], k1, k2, calcHk(k1, k2))
    return Cnew

# Lets Plot
# Create a 2D heatmap plot of Cnew with trajectory overlay
plt.figure(figsize=(10, 8))

# Calculate Cnew for the grid
Cnew_surface = np.zeros((N_points, N_points))
for i in range(N_points):
    for j in range(N_points):
        Cnew_surface[i, j] = returnCnew(x_i[i], y_i[j])

# Create the heatmap
plt.imshow(Cnew_surface.T, origin='lower', extent=[0, L1, 0, L2], cmap=cm.viridis, aspect='auto')
plt.colorbar(label='Cnew')

# Overlay the trajectory
x_traj, y_traj = xv(time)
plt.plot(x_traj, y_traj, 'r-', linewidth=2, label='Trajectory')

# Add labels and title
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('2D Heatmap of Cnew with Trajectory')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
