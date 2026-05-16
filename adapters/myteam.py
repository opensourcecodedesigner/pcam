from __future__ import annotations

from typing import Any

import numpy as np

from adapter import Adapter
from pcam_model import PCAMModel


class Engine(Adapter):
    """
    PCAM Precision Agent - High-Accuracy Retrieval & Low Anisotropy.
    """
    def __init__(self,
                 stored_patterns: np.ndarray,
                 model_params: dict[str, Any]) -> None:
        self.X = stored_patterns
        self.K, self.N = stored_patterns.shape
        self.eta = float(model_params["eta"])
        self.beta = float(model_params["beta"])
        self.R = model_params["R"]
        self.model = PCAMModel(stored_patterns, **model_params)
        
        try:
            self.R_inv = np.linalg.inv(self.R)
        except:
            self.R_inv = np.eye(self.N) / 0.8
            
        self.X2 = self.X ** 2
        self.absX = np.abs(self.X)

    def _compute_ruiz_sym(self, H: np.ndarray, iterations: int = 10) -> np.ndarray:
        """
        Symmetric Ruiz equilibration to minimize the condition number of Pi^(1/2) H Pi^(1/2).
        Returns the diagonal precision vector Pi.
        """
        d = np.ones(H.shape[0])
        for _ in range(iterations):
            # Compute row 1-norms of D * H * D
            DHD = d[:, None] * H * d[None, :]
            r_norms = np.sum(np.abs(DHD), axis=1)
            d = d / np.sqrt(np.maximum(r_norms, 1e-12))
        
        # The equilibrated matrix is D * H * D.
        # Since the metric evaluates Pi^(1/2) * H * Pi^(1/2), we need Pi^(1/2) = D.
        # Therefore, Pi = D^2.
        pi = d ** 2
        return pi / np.mean(pi)

    def predict_precision(self, q: np.ndarray) -> np.ndarray:
        # 1. Attractor guess for rapid detection
        z = (self.beta * 0.9) * (self.X @ q)
        s = np.exp(z - z.max())
        s = s / s.sum()
        
        pred_pattern = self.X.T @ s
        
        # 2. One-shot projection to get a better attractor estimate
        a_star = self.eta * (self.R_inv @ pred_pattern)
        
        # Recompute softmax at the estimated attractor
        z_star = self.beta * (self.X @ a_star)
        s_star = np.exp(z_star - z_star.max())
        s_star = s_star / s_star.sum()
        
        D = np.diag(s_star) - np.outer(s_star, s_star)
        H = self.R - (self.eta * self.beta) * (self.X.T @ D @ self.X)
        
        # 3. Get the optimal geometric precision for the guessed landscape
        pi_geom = self._compute_ruiz_sym(H, iterations=5)
        
        # 4. Retrieval Signal
        target_sq = self.X2.T @ s_star
        target_abs = self.absX.T @ s_star
        query_abs = np.abs(q)
        
        # Recovery ratio
        recovery = (target_abs + 0.05) / (query_abs + 0.05)
        
        # Combine signals
        pi = pi_geom * (1.0 + 12.0 * target_sq) * (1.0 + 3.0 * recovery)
        
        return pi
