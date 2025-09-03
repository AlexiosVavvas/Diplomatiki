import numpy as np
import matplotlib.pyplot as plt

# space is 0->1, 0->1
x_obs = np.array([0.5, 0.5])
Dobs = 0.1

rho_0 = 0.3
kappa = 2/100

def U(rho):
    return np.where(rho <= rho_0, 0.5*kappa*(1/rho - 1/rho_0)**2, 0)
    
# Force is - grad U, calculated analytically
def F_magnitude(rho):
    return np.where(rho <= rho_0, kappa*(1/rho - 1/rho_0)/(rho**2), 0)







# for every point in the grid, calculate the force and plot it in 2d
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
rho_grid = np.sqrt((X - x_obs[0])**2 + (Y - x_obs[1])**2) - Dobs
U_grid = U(rho_grid)

# Calculate force components
F_mag = F_magnitude(rho_grid)
# Direction vector from obstacle to point
dx = X - x_obs[0]
dy = Y - x_obs[1]
dist_to_obs = np.sqrt(dx**2 + dy**2)
# Normalize direction (avoid division by zero)
dx_norm = np.where(dist_to_obs > 0, dx/dist_to_obs, 0)
dy_norm = np.where(dist_to_obs > 0, dy/dist_to_obs, 0)
# Force components (repulsive, pointing away from obstacle)
Fx = F_mag * dx_norm
Fy = F_mag * dy_norm
F_grid = np.sqrt(Fx**2 + Fy**2)

# Plot the result, in 1 2x1 grid for both the U and the F
fig, axs = plt.subplots(2, 1, figsize=(6, 7))
U_grid = np.clip(U_grid, -100, 100)  # Cap U values above 100, to 100

im1 = axs[0].imshow(U_grid, extent=(0, 1, 0, 1), origin='lower', cmap='viridis', aspect='auto')
# Draw a red dashed circle around x_obs, with radious d_obs and another with obs d_obs + d0
circle1 = plt.Circle(x_obs, Dobs, color='red', fill=False, linestyle='--')
circle2 = plt.Circle(x_obs, Dobs + rho_0, color='red', fill=False, linestyle='--')
axs[0].add_artist(circle1)
axs[0].add_artist(circle2)
axs[0].set_title('Potential U(d)')
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')
plt.colorbar(im1, ax=axs[0], label='Potential U')

# Cap Fgrid values above 100, to 100
F_grid = np.clip(F_grid, -100, 100)

im2 = axs[1].imshow(F_grid, extent=(0, 1, 0, 1), origin='lower', cmap='plasma', aspect='auto')
axs[1].set_title('Force Magnitude F(d)')
circle1 = plt.Circle(x_obs, Dobs, color='red', fill=False, linestyle='--')
circle2 = plt.Circle(x_obs, Dobs + rho_0, color='red', fill=False, linestyle='--')
axs[1].add_artist(circle1)
axs[1].add_artist(circle2)
axs[1].set_xlabel('X')
axs[1].set_ylabel('Y')
plt.colorbar(im2, ax=axs[1], label='Force Magnitude')


# I need also another 2d plot with the section at x = x_obs[0] and varying y (2d 2x1 grid)
fig, axs = plt.subplots(2, 1, figsize=(6, 7))
y_section = np.linspace(0, 1, 100)
U_section = U(np.abs(y_section - x_obs[1]) - Dobs)
F_section = F_magnitude(np.abs(y_section - x_obs[1]) - Dobs)
U_section = np.clip(U_section, -100, 100)  # Cap U values above 100, to 100
F_section = np.clip(F_section, -100, 100)  # Cap F values above 100, to 100

axs[0].plot(y_section, U_section, label='Potential U(d)')
axs[0].axvline(x_obs[1] + Dobs, color='red', linestyle='--', label='x_obs + d_obs = {:.1f}'.format(x_obs[1] + Dobs))
axs[0].axvline(x_obs[1] - Dobs, color='red', linestyle='--', label='x_obs - d_obs = {:.1f}'.format(x_obs[1] - Dobs))
axs[0].axvline(x_obs[1] + Dobs + rho_0, color='red', linestyle='--', label='x_obs + d_obs + rho_0 = {:.1f}'.format(x_obs[1] + Dobs + rho_0))
axs[0].axvline(x_obs[1] - Dobs - rho_0, color='red', linestyle='--', label='x_obs - d_obs - rho_0 = {:.1f}'.format(x_obs[1] - Dobs - rho_0))
axs[0].set_title('Potential U(d) Section')
axs[0].set_xlabel('Y')
axs[0].set_ylabel('Potential U')
axs[0].grid()

axs[1].plot(y_section, F_section, label='Force F(d)', color='orange')   
axs[1].axvline(x_obs[1] + Dobs, color='red', linestyle='--', label='x_obs + d_obs = {:.1f}'.format(x_obs[1] + Dobs))
axs[1].axvline(x_obs[1] - Dobs, color='red', linestyle='--', label='x_obs - d_obs = {:.1f}'.format(x_obs[1] - Dobs))
axs[1].axvline(x_obs[1] + Dobs + rho_0, color='red', linestyle='--', label='x_obs + d_obs + rho_0 = {:.1f}'.format(x_obs[1] + Dobs + rho_0))
axs[1].axvline(x_obs[1] - Dobs - rho_0, color='red', linestyle='--', label='x_obs - d_obs + rho_0 = {:.1f}'.format(x_obs[1] - Dobs - rho_0))
axs[1].set_title('Force F(d) Section')
axs[1].set_xlabel('Y')
axs[1].set_ylabel('Force F')
axs[1].grid()
plt.tight_layout()
plt.show()

