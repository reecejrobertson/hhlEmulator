import numpy as np

u0 = np.array([-1/np.sqrt(2), -1/np.sqrt(2)], dtype=float)
u1 = np.array([-1/np.sqrt(2), 1/np.sqrt(2)], dtype=float)
l0 = 2/3.
l1 = 4/3.
b0 = -1/np.sqrt(2)
b1 = 1/np.sqrt(2)
c = 2/3.

a0 = np.array([np.sqrt(1 - c**2/l0**2), c/l0])
a1 = np.array([np.sqrt(1 - c**2/l1**2), c/l1])
psi = np.kron(b0 * u0, a0) + np.kron(b1 * u1, a1)

# x = np.array([
#     (1/2) * np.sqrt(7/16) - (1/2) * np.sqrt(55/64),
#     3/8 - 3/16,
#     (1/2) * np.sqrt(7/16) + (1/2) * np.sqrt(55/64),
#     3/8 + 3/16,
# ], dtype=float)

print(psi)
print(psi**2)