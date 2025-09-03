import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2, 2, 1000)

# Parameters ----------------------
k = 0.1
rho = 0.3

delta = 0
inf_value = -delta + (2 * rho**2)/(k + 2*rho**2)
print(f"inf_value = {inf_value:.3f}")
delta += inf_value
# delta = 0.15

# Main Calculation ----------------
# u = k/2 * (1/(-x))**2
u = np.zeros_like(x)
for i, x_ in enumerate(x):
    if x_ >= 0:
        u[i] = np.inf
    elif x_ > -rho:
        u[i] = k/2 * (1/(-x_) - 1/rho)**2
    elif x_ < -rho:
        u[i] = 0

h = 1 / (1 + u) - delta
grad_u = np.gradient(u)
grad_h = np.gradient(h)

# Plotting Results ----------------
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

ax1.plot(x, u, label='u(x)')
ax1.set_title('Potential Field U(x)')
ax1.legend()
ax1.grid()
ax1.set_ylim(-10, 100)

ax2.plot(x, grad_u)
# Fill green the positive and red the negative
negative_mask = grad_u < 0
ax2.fill_between(x, grad_u, 0, where=negative_mask, color='red', alpha=0.5, label='Negative values')
positive_mask = grad_u >= 0
ax2.fill_between(x, grad_u, 0, where=positive_mask, color='green', alpha=0.5, label='Positive values')
ax2.set_title('Gradient of u(x)')
ax2.axvline(x=0, color='black', linestyle='--', label='x = 0', alpha=0.5)
ax2.axvline(x=-rho, color='black', linestyle='--', label='x = -rho', alpha=0.5)
ax2.grid()
ax2.legend()
# Lim y axis to +-0.2
ax2.set_ylim(-0.2, 0.2)

ax3.plot(x, h, label='h(x)')
# Highlight negative parts in red with alpha=0.5
negative_mask = h < 0
ax3.fill_between(x, h, 0, where=negative_mask, color='red', alpha=0.5, label='Negative values')
ax3.axvline(x=0, color='black', linestyle='--', label='x = 0', alpha=0.5)
ax3.axvline(x=-rho, color='black', linestyle='--', label='x = -rho', alpha=0.5)
ax3.set_title('h(x) with Negative Values Highlighted')
ax3.grid()
ax3.legend()

ax4.plot(x, grad_h)
# Fill green the positive and red the negative
negative_mask = grad_h < 0
ax4.fill_between(x, grad_h, 0, where=negative_mask, color='red', alpha=0.5, label='Negative values')
positive_mask = grad_h >= 0
ax4.fill_between(x, grad_h, 0, where=positive_mask, color='green', alpha=0.5, label='Positive values')
ax4.set_title('Gradient of h(x)')
ax4.axvline(x=0, color='black', linestyle='--', label='x = 0', alpha=0.5)
ax4.axvline(x=-rho, color='black', linestyle='--', label='x = -rho', alpha=0.5)
ax4.grid()
ax4.legend()

plt.tight_layout()
plt.show()
