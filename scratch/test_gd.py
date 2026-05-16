import numpy as np
from pcam_model import PCAMModel, build_default_R
from data import make_patterns

seeds_to_test = [7, 13, 31, 97, 211]

for seed in seeds_to_test:
    X = make_patterns(seed=seed)
    R = build_default_R(seed=seed)
    model = PCAMModel(X, R=R)
    a_star = model.find_equilibrium(X[0])
    H = model.hessian(a_star)

    pi = np.ones(64)
    lr = 0.5
    
    # Calculate base spread
    pi_sqrt_base = np.sqrt(np.ones(64))
    S_base = (pi_sqrt_base[:, None] * H) * pi_sqrt_base[None, :]
    vals_base = np.linalg.eigvalsh(0.5 * (S_base + S_base.T))
    vals_base = vals_base[vals_base > 1e-9]
    base_spread = vals_base[-1] / vals_base[0]

    for step in range(500):
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
        
    opt_spread = l_max / l_min
    reduction = base_spread / opt_spread
    print(f"Seed {seed:>3}: Base Spread {base_spread:7.2f} -> Opt Spread {opt_spread:7.2f} | Reduction: {reduction:.2f}x")
