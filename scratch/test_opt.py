import numpy as np
from pcam_model import PCAMModel
from data import make_patterns

X = make_patterns()
model = PCAMModel(X)

def test_spread(k):
    a_star = model.eta * (np.linalg.inv(model.R) @ X[k])
    z_star = model.beta * (X @ a_star)
    s_star = np.exp(z_star - z_star.max())
    s_star = s_star / s_star.sum()
    
    # User's exact H formula
    H_user = model.R - (model.eta * model.beta) * (X.T @ np.diag(s_star) @ X)
    
    H_diag = np.diag(H_user)
    pi_geom = 1.0 / np.sqrt(np.abs(H_diag) + 1e-8)
    pi_geom = pi_geom / np.mean(pi_geom)
    pi_geom = np.clip(pi_geom, 0.1, 10.0)
    
    # True H
    a_true = model.find_equilibrium(X[k])
    H_true = model.hessian(a_true)
    
    # Calculate spread
    pi = model.clip_and_normalise(pi_geom)
    pi_sqrt = np.sqrt(pi)
    S = (pi_sqrt[:, None] * H_true) * pi_sqrt[None, :]
    eigs = np.linalg.eigvalsh(0.5 * (S + S.T))
    eigs = eigs[eigs > 1e-9]
    return eigs.max() / eigs.min()

print("Spread with user's formula:", test_spread(0))

