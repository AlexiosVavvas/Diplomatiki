import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def calculate_force_field(angle_deg):
    # Convert angle to radians and create rotated p vector
    angle_rad = np.radians(angle_deg)
    p_magnitude = np.linalg.norm(p_original)
    p = p_magnitude * np.array([np.cos(angle_rad), np.sin(angle_rad)])
    
    # Initialize force components
    Fx = np.zeros_like(X)
    Fy = np.zeros_like(Y)
    
    for i in range(len(x)):
        for j in range(len(y)):
            rj = np.array([X[j, i], Y[j, i]])
            if (abs(rj[0] - r0[0]) > obs_radious or abs(rj[1] - r0[1]) > obs_radious) and (abs(rj[0] - r0[0]) < 2 * obs_radious and abs(rj[1] - r0[1]) < 2 * obs_radious):
            # if np.linalg.norm(rj - r0) > obs_radious:
                r = rj - r0
                if p.T @ r >= 0:
                    F = lamda * np.dot(p, r) * r - p * np.dot(r, r)
                    Fx[j, i] = F[0]
                    Fy[j, i] = F[1]
                else:
                    F = - p * np.dot(r, r)
                    Fx[j, i] = F[0]
                    Fy[j, i] = F[1]
            else:
                Fx[j, i] = 0
                Fy[j, i] = 0
    
    return Fx, Fy, p

def update_plot(val):
    angle_deg = slider_angle.val
    scale = slider_scale.val
    
    Fx, Fy, p = calculate_force_field(angle_deg)
    
    # Clear and redraw
    ax.clear()
    ax.quiver(X, Y, Fx, Fy, angles='xy', scale_units='xy', scale=scale)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Vector Field Visualization')
    ax.grid(True)
    ax.set_aspect('equal')
    
    # # Mark the center point and obstacle
    # circle = plt.Circle(r0, obs_radious, color='red', fill=False, linewidth=2)
    # ax.add_patch(circle)
    # ax.plot(r0[0], r0[1], 'ro', markersize=8, label='Center')

    # Rectangle for obstacle
    rect = plt.Rectangle((r0[0] - obs_radious, r0[1] - obs_radious), 2 * obs_radious, 2 * obs_radious, 
                         color='red', fill=False, linewidth=2)
    ax.add_patch(rect)

    
    # Draw the rotated p vector
    ax.quiver(r0[0], r0[1], p[0], p[1], angles='xy', scale_units='xy', scale=5, color='orange', label='p vector')
    ax.legend()
    
    plt.draw()

# Initial parameters
r0 = np.array([.5, .5])
lamda = 1
p_original = -1*np.array([1, 1])  # Store original p vector
# p_original = -np.array([1, 1])  # Store original p vector
obs_radious = 0.2

x = np.linspace(0, 1, 20)
y = np.linspace(0, 1, 20)
X, Y = np.meshgrid(x, y)

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 10))
plt.subplots_adjust(bottom=0.2)

# Create sliders
ax_angle_slider = plt.axes([0.2, 0.08, 0.6, 0.03])
slider_angle = Slider(ax_angle_slider, 'Angle (deg)', 0, 360, valinit=225, valfmt='%0.0f°')

ax_scale_slider = plt.axes([0.2, 0.03, 0.6, 0.03])
slider_scale = Slider(ax_scale_slider, 'Scale', 1, 20, valinit=10, valfmt='%0.0f')

# Initial plot
update_plot(None)

# Connect sliders to update function
slider_angle.on_changed(update_plot)
slider_scale.on_changed(update_plot)

plt.show()