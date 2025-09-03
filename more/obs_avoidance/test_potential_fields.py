import numpy as np
import matplotlib.pyplot as plt

D0 = 0.25
C = 0.1
lamda = -D0/np.log(C)
kappa = 0.25
print(f"3% F0 at d = {-lamda*np.log(0.01):.2f} -> {-lamda*np.log(0.01)/D0:.1f} D0")
rho_0 = 0.2

d = np.linspace(0, 1, 500)

def U(d):
    return 0.5*kappa*(1/d - 1/rho_0)**2
    # return kappa * np.exp(-d / lamda)
    
# Force is - grad U
def F(d):
    return 1 / (1 + U(d)) - 0.742268
    # return -np.gradient(U(d), d)

# Plot the result, in 1 2x1 grid for bothe the u and the f
fig, axs = plt.subplots(2, 1, figsize=(8, 6))
axs[0].plot(d, U(d), label='Potential U(d)')
# Vertical line at D0
axs[0].axvline(D0, color='red', linestyle='--', label='D0 = {:.1f}'.format(D0))
axs[0].set_title('Potential U(d)')
axs[0].set_xlabel('Distance d')
axs[0].set_ylabel('Potential U')
axs[0].grid()
axs[0].legend()


axs[1].plot(d, F(d), label='Force F(d)', color='orange')
# Vertical line at D0
axs[1].axvline(D0, color='red', linestyle='--', label='D0 = {:.1f}'.format(D0))
axs[1].set_title('Force F(d)')
axs[1].set_xlabel('Distance d')
axs[1].set_ylabel('Force F')
axs[1].grid()
axs[1].legend()
plt.tight_layout()
plt.show()

