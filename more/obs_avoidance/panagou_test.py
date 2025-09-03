import numpy as np
import matplotlib.pyplot as plt

r0 = np.array([.5, .5])
lamda = 1
p = -np.array([1, 1])
obs_radious = 0.1

x = np.linspace(0, 1, 20)  # Reduced density for better visualization
y = np.linspace(0, 1, 20)
X, Y = np.meshgrid(x, y)

# Initialize force components
Fx = np.zeros_like(X)
Fy = np.zeros_like(Y)

for i in range(len(x)):
    for j in range(len(y)):
        rj = np.array([X[j, i], Y[j, i]])
        if np.linalg.norm(rj - r0) > obs_radious:
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

# Create the plot
plt.figure(figsize=(10, 8))
plt.quiver(X, Y, Fx, Fy, angles='xy', scale_units='xy', scale=10)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Vector Field Visualization')
plt.grid(True)
plt.axis('equal')

# Mark the center point and obstacle
circle = plt.Circle(r0, obs_radious, color='red', fill=False, linewidth=2)
plt.gca().add_patch(circle)
plt.plot(r0[0], r0[1], 'ro', markersize=8, label='Center')

# I want to see the pj vector visualised as well
plt.quiver(r0[0], r0[1], p[0], p[1], angles='xy', scale_units='xy', scale=5, color='orange', label='p vector')

plt.legend()
plt.show()