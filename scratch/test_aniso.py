import numpy as np
from pcam_model import PCAMModel
from data import make_patterns

X = make_patterns()
model = PCAMModel(X)

# Evaluate spread reduction for the first pattern
idx = 0
pattern = X[idx]
a_star = model.find_equilibrium(pattern)
H = model.hessian(a_star)

# Baseline spread
base_eigs = np.linalg.eigvalsh(H)
print(f"H min eig: {base_eigs.min():.4f}, max eig: {base_eigs.max():.4f}")
base_spread = base_eigs.max() / base_eigs.min()
print(f"Baseline spread: {base_spread:.2f}")

# Function to compute spread for a given pi
def get_spread(pi):
    pi = model.clip_and_normalise(pi)
    pi_sqrt = np.sqrt(np.clip(pi, 1e-12, None))
    S = (pi_sqrt[:, None] * H) * pi_sqrt[None, :]
    eigs = np.linalg.eigvalsh(0.5 * (S + S.T))
    eigs = eigs[eigs > 1e-9]
    return eigs.max() / eigs.min()

print(f"Spread with pi=1: {get_spread(np.ones(64)):.2f}")
print(f"Spread with pi=diag(H^-1): {get_spread(np.diag(np.linalg.inv(H))):.2f}")
print(f"Spread with pi=1/diag(H): {get_spread(1.0 / np.diag(H)):.2f}")
print(f"Spread with pi=|pattern|: {get_spread(np.abs(pattern) + 0.01):.2f}")
print(f"Spread with pi=pattern^2: {get_spread(pattern**2 + 0.01):.2f}")

# Try to find optimal pi via scipy minimize
from scipy.optimize import minimize

def obj(pi):
    pi_sqrt = np.sqrt(np.clip(pi, 1e-12, None))
    S = (pi_sqrt[:, None] * H) * pi_sqrt[None, :]
    eigs = np.linalg.eigvalsh(0.5 * (S + S.T))
    eigs = eigs[eigs > 1e-9]
    return eigs.max() / eigs.min()

res = minimize(
    obj,
    x0=np.ones(64),
    bounds=[(0.1, 10.0)] * 64,
    options={'maxiter': 100}
)
print(f"Optimal spread (via optimization): {res.fun:.2f}")
if res.fun < base_spread / 2:
    print("Optimization found a good pi! Max/Min pi:", np.max(res.x), np.min(res.x))
else:
    print("Optimization could NOT significantly reduce spread.")
