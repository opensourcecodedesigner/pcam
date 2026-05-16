import numpy as np
from pcam_model import PCAMModel, build_default_R
from data import make_patterns

seed = 101
X = make_patterns(seed=seed)
R = build_default_R(seed=seed)
model = PCAMModel(X, R=R)

# 1. Optimize pi for pattern 0
k = 0
a_star = model.find_equilibrium(X[k])
H = model.hessian(a_star)

pi = np.ones(64)
lr = 0.5
for step in range(200):
    pi_clipped = np.clip(pi, 0.1, 10.0)
    pi_clipped = pi_clipped / np.mean(pi_clipped)
    
    pi_sqrt = np.sqrt(pi_clipped)
    S = (pi_sqrt[:, None] * H) * pi_sqrt[None, :]
    vals, vecs = np.linalg.eigh(0.5 * (S + S.T))
    
    valid_idx = vals > 1e-9
    vals = vals[valid_idx]
    vecs = vecs[:, valid_idx]
    
    l_max = vals[-1]
    l_min = vals[0]
    v_max = vecs[:, -1]
    v_min = vecs[:, 0]
    
    grad = (v_max**2 - v_min**2) / pi_clipped
    pi = pi_clipped - lr * grad

pi_geom = np.clip(pi, 0.1, 10.0)
pi_geom = pi_geom / np.mean(pi_geom)

# 2. Test retrieval of pattern 0 with 85% noise
rng = np.random.default_rng(42)
q = X[k].copy()
mask = rng.random(64) < 0.85
q[mask] = 0.0
q += rng.standard_normal(64) * 0.05
q /= np.linalg.norm(q)

# Run dynamics with pi_geom
a_final = model.run(q, pi_geom)
pred_idx = model.classify(a_final)
print(f"Retrieval with GD pi_geom: Pattern {pred_idx} (True was {k})")

# Run dynamics with pi=ones
a_final_ones = model.run(q, np.ones(64))
pred_idx_ones = model.classify(a_final_ones)
print(f"Retrieval with PI=ones: Pattern {pred_idx_ones}")
