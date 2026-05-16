import numpy as np
from pcam_model import PCAMModel
from data import make_patterns

X = make_patterns()
model = PCAMModel(X)

for k in range(3):
    a_star = model.eta * (np.linalg.inv(model.R) @ X[k])
    z_star = model.beta * (X @ a_star)
    s_star = np.exp(z_star - z_star.max())
    s_star = s_star / s_star.sum()
    print(f"Pattern {k} s_star top 3: {np.sort(s_star)[-3:][::-1]}")
