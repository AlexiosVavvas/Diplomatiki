import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

x0 = np.array([0, 0])
# W = np.array([1, 1])
W = np.array([0.25, 0.7])

def infNorm(x):
    return np.max(np.abs(x))
def pNorm(x, p):
    return np.sum(np.abs(x) ** p) ** (1 / p)

R = 0.5
def rho(x):
    # return pNorm(R/W * (x - x0), 5) - R
    return infNorm(1/ W * (x - x0)) - 1

def rhoReal(x):
    # Calcute real distance to the wall of rectangle centered at x0 with width 2W[0] and height 2W[1]
    if np.max([np.abs(x-x0) - W]) >= 0:
        return np.max(np.abs(x-x0)-W)
    else:
        return -np.min(W - np.abs(x-x0))
    
def analyticGrad(x):
    E1  = np.abs(x[0] - x0[0]) - W[0]
    E2  = np.abs(x[1] - x0[1]) - W[1]

    if np.max([E1, E2]) >= 0:
        # outside or on: gradient points in the direction
        # of the coordinate that attained the max
        if E1 >= E2:
            grad = np.array([np.sign(x[0] - x0[0]), 0])
        else:
            grad = np.array([0, np.sign(x[1] - x0[1])])
    else:
        # inside: gradient also points to the nearest side
        G1 = W[0] - np.abs(x[0] - x0[0])
        G2 = W[1] - np.abs(x[1] - x0[1])
        if G1 <= G2:
            grad = np.array([np.sign(x[0] - x0[0]), 0])
        else:
            grad = np.array([0, np.sign(x[1] - x0[1])])
    return np.array(grad)

def finiteDiffGrad(x, h=1e-3):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = np.copy(x)
        x_plus[i] += h
        x_minus = np.copy(x)
        x_minus[i] -= h
        grad[i] = (rhoReal(x_plus) - rhoReal(x_minus)) / (2 * h)
    return grad

# Lets plot it
x = np.linspace(-1.5, 1.5, 50)
y = np.linspace(-1.5, 1.5, 50)
z = np.zeros((len(x), len(y)))

for i, x_ in enumerate(x):
    for j, y_ in enumerate(y):
        z[i, j] = rhoReal(np.array([x_, y_])) #/ rhoReal(np.array([x_, y_]))

fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(z.T, extent=(x[0], x[-1], y[0], y[-1]), origin='lower', cmap='viridis', aspect='auto')
# Draw a rectangle with center x0 and widht, height = 2*W
rect = Rectangle(
    (x0[0] - W[0], x0[1] - W[1]), 2 * W[0], 2 * W[1],
    linewidth=1, edgecolor='red', facecolor='none', linestyle='--'
)
ax.add_patch(rect)
# Draw the p=3 norm contour at x0 center with radius R
# contour = ax.contour(x, y, z.T, levels=[0], colors='red')
# Draw circle with radius R centered at x0
# ax.set_aspect('equal', adjustable='box')
# circle = plt.Circle(x0, R, color='blue', fill=False, linestyle='--')
# ax.add_artist(circle)
ax.set_title('Inf-Norm Contour Plot')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.grid()
# plt.show()

# Lets see it in 3d as well
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)
for i, x_ in enumerate(x):
    for j, y_ in enumerate(y):
        Z[i, j] = rhoReal(np.array([x_, y_])) #/ rhoReal(np.array([x_, y_]))
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax.set_title('Inf-Norm Surface Plot')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('rho')
plt.show()


# Lets visualise and compare the gradients
# Test multiple points and visualize gradient fields
test_points = np.array([
    [0.5, 0.5],    # inside rectangle
    [-0.3, 0.2],   # inside rectangle
    [0.4, 0.8],    # outside rectangle
    [-0.6, -0.4],  # outside rectangle
    [0.25, 0.7],   # on boundary (corner)
    [0.0, 0.7],    # on boundary (edge)
    [0.25, 0.0],   # on boundary (edge)
    [0.8, 0.3],    # outside rectangle
    [-0.1, -0.9],  # outside rectangle
])

print("Gradient Comparison Tests:")
print("=" * 60)
for i, x_test in enumerate(test_points):
    grad_analytic = analyticGrad(x_test)
    grad_finite_diff = finiteDiffGrad(x_test)
    rho_val = rhoReal(x_test)
    rho_real_val = rhoReal(x_test)
    
    print(f"Point {i+1}: {x_test}")
    print(f"  rho value: {rho_val:.6f}")
    print(f"  rhoReal value: {rho_real_val:.6f}")
    print(f"  Analytic Gradient: {grad_analytic}")
    print(f"  Finite Diff Gradient: {grad_finite_diff}")
    print(f"  Difference: {grad_analytic - grad_finite_diff}")
    print(f"  Norm of difference: {np.linalg.norm(grad_analytic - grad_finite_diff):.8f}")
    print("-" * 40)

# Visualize gradient fields
x_viz = np.linspace(-2.0, 2.0, 60)
y_viz = np.linspace(-2.0, 2.0, 60)
X, Y = np.meshgrid(x_viz, y_viz)

# Calculate gradients at each grid point
grad_x_analytic = np.zeros_like(X)
grad_y_analytic = np.zeros_like(Y)
grad_x_finite = np.zeros_like(X)
grad_y_finite = np.zeros_like(Y)
rho_values = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        point = np.array([X[i, j], Y[i, j]])
        grad_a = analyticGrad(point)
        grad_f = finiteDiffGrad(point)
        
        grad_x_analytic[i, j] = grad_a[0]
        grad_y_analytic[i, j] = grad_a[1]
        grad_x_finite[i, j] = grad_f[0]
        grad_y_finite[i, j] = grad_f[1]
        rho_values[i, j] = rhoReal(point)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Analytic gradient field
axes[0, 0].quiver(X, Y, grad_x_analytic, grad_y_analytic, alpha=0.7)
axes[0, 0].contour(X, Y, rho_values, levels=[0], colors='red', linewidths=2)
rect1 = Rectangle((x0[0] - W[0], x0[1] - W[1]), 2 * W[0], 2 * W[1],
                  linewidth=2, edgecolor='blue', facecolor='none', linestyle='--')
axes[0, 0].add_patch(rect1)
axes[0, 0].set_title('Analytic Gradient Field')
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('y')
axes[0, 0].grid(True)
axes[0, 0].set_aspect('equal')

# Finite difference gradient field
axes[0, 1].quiver(X, Y, grad_x_finite, grad_y_finite, alpha=0.7)
axes[0, 1].contour(X, Y, rho_values, levels=[0], colors='red', linewidths=2)
rect2 = Rectangle((x0[0] - W[0], x0[1] - W[1]), 2 * W[0], 2 * W[1],
                  linewidth=2, edgecolor='blue', facecolor='none', linestyle='--')
axes[0, 1].add_patch(rect2)
axes[0, 1].set_title('Finite Difference Gradient Field')
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('y')
axes[0, 1].grid(True)
axes[0, 1].set_aspect('equal')

# Difference field
diff_x = grad_x_analytic - grad_x_finite
diff_y = grad_y_analytic - grad_y_finite
diff_magnitude = np.sqrt(diff_x**2 + diff_y**2)

im = axes[1, 0].imshow(diff_magnitude, extent=[-2, 2, -2, 2], origin='lower', 
                       cmap='hot', aspect='equal')
axes[1, 0].contour(X, Y, rho_values, levels=[0], colors='cyan', linewidths=2)
rect3 = Rectangle((x0[0] - W[0], x0[1] - W[1]), 2 * W[0], 2 * W[1],
                  linewidth=2, edgecolor='blue', facecolor='none', linestyle='--')
axes[1, 0].add_patch(rect3)
axes[1, 0].set_title('Gradient Difference Magnitude')
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('y')
plt.colorbar(im, ax=axes[1, 0])

# rho function values
im2 = axes[1, 1].imshow(rho_values, extent=[-2, 2, -2, 2], origin='lower', 
                        cmap='viridis', aspect='equal')
axes[1, 1].contour(X, Y, rho_values, levels=[0], colors='red', linewidths=2)
rect4 = Rectangle((x0[0] - W[0], x0[1] - W[1]), 2 * W[0], 2 * W[1],
                  linewidth=2, edgecolor='white', facecolor='none', linestyle='--')
axes[1, 1].add_patch(rect4)
axes[1, 1].set_title('Rho Function Values')
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('y')
plt.colorbar(im2, ax=axes[1, 1])

# Add test points to all plots
for ax in axes.flat:
    ax.scatter(test_points[:, 0], test_points[:, 1], c='red', s=50, marker='x', linewidths=2)

plt.tight_layout()
plt.show()

# Test gradient accuracy statistics
all_diffs = []
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        point = np.array([X[i, j], Y[i, j]])
        grad_a = analyticGrad(point)
        grad_f = finiteDiffGrad(point)
        diff_norm = np.linalg.norm(grad_a - grad_f)
        all_diffs.append(diff_norm)

all_diffs = np.array(all_diffs)
print(f"\nGradient Accuracy Statistics:")
print(f"Mean difference: {np.mean(all_diffs):.8f}")
print(f"Max difference: {np.max(all_diffs):.8f}")
print(f"Min difference: {np.min(all_diffs):.8f}")
print(f"Std difference: {np.std(all_diffs):.8f}")



# Lets check the potential

x = np.linspace(-1.5, 1.5, 1000)

z = np.zeros_like(x)
h = np.zeros_like(x)
g = np.zeros_like(x)

def U(rho, rho0=0.2):
    if rho <= rho0:
        return 0.5 * 1 * (1/rho - 1/rho0)**2
    else:
        return 0

def Ugrad(x, h=1e-3):
    grad = 0
    x_plus = np.copy(x)
    x_plus += h
    x_minus = np.copy(x)
    x_minus -= h
    grad = (U(x_plus) - U(x_minus)) / (2 * h)
    return grad


for i in range(len(x)):
    z_ = rhoReal(np.array([x0[0], x[i]]))
    # z[i] = z_
    z[i] = U(z_, rho0=0.2)
    h[i] = 1/(1+z[i]) - 0.01
    g[i] = Ugrad(z_)

plt.figure()
plt.plot(x, z)
plt.title("Z")
plt.grid()

plt.figure()
plt.plot(x, h)
plt.title("H")
plt.grid()

plt.figure()
plt.plot(x, g)
plt.grid()
plt.title("G")
plt.show()