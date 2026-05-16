import numpy as np
from pcam_model import PCAMModel
from data import make_patterns

# Real PCAM H
X = make_patterns()
model = PCAMModel(X)
a_star = model.find_equilibrium(X[0])
H = model.hessian(a_star)

base_eigs = np.linalg.eigvalsh(H)
base_spread = base_eigs.max() / base_eigs.min()
print(f"Base spread: {base_spread:.2f}")

def compute_spread(pi):
    pi_sqrt = np.sqrt(np.clip(pi, 1e-12, 10.0))
    S = pi_sqrt[:, None] * H * pi_sqrt[None, :]
    eigs = np.linalg.eigvalsh(S)
    return eigs.max() / eigs.min()

print(f"Spread pi=I: {compute_spread(np.ones(64)):.2f}")
print(f"Spread pi=|x|: {compute_spread(np.abs(X[0]) + 0.01):.2f}")
print(f"Spread pi=x^2: {compute_spread(X[0]**2 + 0.01):.2f}")
print(f"Spread pi=diag(H^-1): {compute_spread(np.diag(np.linalg.inv(H))):.2f}")
print(f"Spread pi=1/diag(H): {compute_spread(1.0 / np.diag(H)):.2f}")

# Find best pi by optimization?
# No time.

