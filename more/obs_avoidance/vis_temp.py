import numpy as np
import matplotlib.pyplot as plt

def phiExample(s, L1=1.0, L2=1.0):
    # Complex function with multiple peaks, valleys, and non-linearities
    x, y = s[0], s[1]
    
    # Multiple Gaussian bumps
    # Generate random bump positions within the L1, L2 boundaries
    bump_positions = [
        (0.3 * L1, 0.8 * L2),
        (0.7 * L1, 0.2 * L2),
        (0.15 * L1, 0.4 * L2),
        (0.85 * L1, 0.6 * L2)
    ]
    bump_heights = [5, 4, 3, 4.5]
    bump_widths = [0.7, 0.7, 15.2, 6.3]
    
    bumps = 0
    for i in range(len(bump_positions)):
        pos_x, pos_y = bump_positions[i]
        height = bump_heights[i]
        width = bump_widths[i]
        bumps += height * np.exp(-width * ((x-pos_x)**2 + (y-pos_y)**2))
    
    return bumps

# Function to be used for phi with specific L1 and L2 values
def phi_func(s):
    return phiExample(s, L1=10.0, L2=10.0)/42.885 * 4

# Create visualization
def visualize_function():
    # Create grid
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x, y)
    
    # Calculate function values
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = phi_func([X[i, j], Y[i, j]])
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(contour, label='Function Value')
    ax.contour(X, Y, Z, levels=20, colors='black', alpha=0.3, linewidths=0.5)
    
    # Add hover functionality
    def on_hover(event):
        if event.inaxes == ax:
            x_coord = event.xdata
            y_coord = event.ydata
            if x_coord is not None and y_coord is not None:
                z_value = phi_func([x_coord, y_coord])
                ax.set_title(f'2D Visualization of phi_func - X: {x_coord:.2f}, Y: {y_coord:.2f}, Z: {z_value:.4f}')
                fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect('motion_notify_event', on_hover)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('2D Visualization of phi_func')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    plt.show()

# Run visualization
if __name__ == "__main__":
    visualize_function()