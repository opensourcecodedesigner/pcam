import time
import numpy as np
from pcam_model import PCAMModel, build_default_R
from adapters.myteam import Engine
from data import make_patterns

seed = 42
X = make_patterns(seed=seed)
R = build_default_R(seed=seed)
params = {"eta": 0.5, "beta": 8.0, "R": R}

# Time __init__
t0 = time.time()
engine = Engine(X, params)
t1 = time.time()
print(f"Init time: {t1 - t0:.4f}s")

# Time predict_precision
q = X[0] + np.random.standard_normal(64) * 0.05
q = q / np.linalg.norm(q)

t0 = time.time()
for _ in range(1000):
    pi = engine.predict_precision(q)
t1 = time.time()
print(f"Predict time per query: {(t1 - t0) / 1000 * 1000:.4f}ms")
