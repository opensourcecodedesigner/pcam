import numpy as np
from pcam_model import PCAMModel, build_default_R
from data import make_patterns
from scipy.optimize import minimize

seed = 101
X = make_patterns(seed=seed)
R = build_default_R(seed=seed)
model = PCAMModel(X, R=R)
a_star = model.find_equilibrium(X[0])
H = model.hessian(a_star)

def get_spread(pi_raw):
    pi = model.clip_and_normalise(pi_raw)
    pi_sqrt = np.sqrt(pi)
    S = (pi_sqrt[:, None] * H) * pi_sqrt[None, :]
    eigs = np.linalg.eigvalsh(0.5 * (S + S.T))
    eigs = eigs[eigs > 1e-9]
    return eigs.max() / eigs.min()

print("Base spread:", get_spread(np.ones(64)))

res = minimize(get_spread, np.ones(64), method='L-BFGS-B', bounds=[(0.1, 10.0)]*64, options={'maxiter': 1000})
print("Optimized spread:", res.fun)
