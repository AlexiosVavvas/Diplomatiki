import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

x0 = 0.5
# Initial values for parameters
D0_init = 0.15
rho_0_init = 0.2
kappa_init = 0.25
delta_init = 0

x = np.linspace(0, 1, 500)

def U(x, D0, rho_0, kappa):
    return np.where(np.abs(x - x0) <= D0 + rho_0,
                    0.5*kappa*(1/(np.abs(x-x0) - D0) - 1/rho_0)**2,
                    0)
    # Lets use a logarithmic potential for better behavior
    # return np.where(np.abs(x - x0) <= D0,
    #                 -kappa * np.log10((np.abs(x - x0) - D0) / D0),
    #                 0)
    
def F(x, D0, rho_0, kappa, delta):
    return 1 / (1 + U(x, D0, rho_0, kappa)) - delta

# Create figure with space for sliders
fig = plt.figure(figsize=(8, 7))
gs = fig.add_gridspec(3, 2, height_ratios=[2, 2, 0.8], hspace=0.3, wspace=0.3, 
                      top=0.92, bottom=0.15, left=0.08, right=0.95)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, :])

# Initial plots
line1, = ax1.plot(x, U(x, D0_init, rho_0_init, kappa_init), label='Potential U(d)')
vlines1 = [
    ax1.axvline(x0 + D0_init, color='red', linestyle='--'),
    ax1.axvline(x0 - D0_init, color='red', linestyle='--'),
    ax1.axvline(x0 + D0_init + rho_0_init, color='blue', linestyle='--'),
    ax1.axvline(x0 - D0_init - rho_0_init, color='blue', linestyle='--')
]
ax1.set_title('Potential U(d)')
ax1.set_xlabel('Distance d')
ax1.set_ylabel('Potential U')
ax1.grid()
ax1.legend()

line2, = ax2.plot(x, F(x, D0_init, rho_0_init, kappa_init, delta_init), label='Force F(d)', color='orange')
vlines2 = [
    ax2.axvline(x0 + D0_init, color='red', linestyle='--'),
    ax2.axvline(x0 - D0_init, color='red', linestyle='--'),
    ax2.axvline(x0 + D0_init + rho_0_init, color='blue', linestyle='--'),
    ax2.axvline(x0 - D0_init - rho_0_init, color='blue', linestyle='--')
]
ax2.set_title('Force F(d)')
ax2.set_xlabel('Distance d')
ax2.set_ylabel('Force F')
ax2.grid()
ax2.legend()

# Create sliders with better positioning
ax_D0 = fig.add_subplot(gs[2, 0])
ax_rho_0 = fig.add_subplot(gs[2, 1])
ax_kappa = plt.axes([0.1, 0.05, 0.35, 0.025])
ax_delta = plt.axes([0.55, 0.05, 0.35, 0.025])

slider_D0 = Slider(ax_D0, 'D0', 0, 0.3, valinit=D0_init)
slider_rho_0 = Slider(ax_rho_0, 'rho_0', 0, 0.3, valinit=rho_0_init)
slider_kappa = Slider(ax_kappa, 'kappa', 0, 1, valinit=kappa_init)
slider_delta = Slider(ax_delta, 'delta', 0, 0.2, valinit=delta_init)

def update(val):
    D0 = slider_D0.val
    rho_0 = slider_rho_0.val
    kappa = slider_kappa.val
    delta = slider_delta.val
    
    # Update potential plot
    line1.set_ydata(U(x, D0, rho_0, kappa))
    ax1.relim()
    ax1.autoscale_view()
    
    # Update force plot
    line2.set_ydata(F(x, D0, rho_0, kappa, delta))
    ax2.relim()
    ax2.autoscale_view()
    
    # Update vertical lines
    positions = [x0 + D0, x0 - D0, x0 + D0 + rho_0, x0 - D0 - rho_0]
    for i, pos in enumerate(positions):
        vlines1[i].set_xdata([pos, pos])
        vlines2[i].set_xdata([pos, pos])
    
    fig.canvas.draw_idle()

# Connect sliders to update function
slider_D0.on_changed(update)
slider_rho_0.on_changed(update)
slider_kappa.on_changed(update)
slider_delta.on_changed(update)

plt.show()

