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
            
        self.pi_geoms = np.zeros((self.K, self.N))
        
        for k in range(self.K):
            a_star = self.model.find_equilibrium(self.X[k])
            
            z_star = self.beta * (self.X @ a_star)
            s_star = np.exp(z_star - z_star.max())
            s_star = s_star / s_star.sum()
            
            D = np.diag(s_star) - np.outer(s_star, s_star)
            H = self.R - (self.eta * self.beta) * (self.X.T @ D @ self.X)
            H = 0.5 * (H + H.T)  # Ensure exact symmetry like PCAMModel
            
            # Simple geometric precision baseline
            H_diag = np.diag(H)
            pi_geom = 1.0 / np.sqrt(np.abs(H_diag) + 1e-8)
            pi_geom = pi_geom / np.mean(pi_geom)
            self.pi_geoms[k] = np.clip(pi_geom, 0.1, 10.0)

    def predict_precision(self, q: np.ndarray) -> np.ndarray:
        # Cosine similarity
        cosines = self.X @ q
        top3_idx = np.argsort(cosines)[-3:]
        
        # Softmax weights based on distance
        weights = np.exp(10.0 * cosines[top3_idx])
        weights /= weights.sum()
        
        pi_ensemble = np.zeros(self.N)
        query_abs = np.abs(q)
        
        for i, k in enumerate(top3_idx):
            pi_geom = self.pi_geoms[k].copy()
            target_sq = self.X[k] ** 2
            target_abs = np.abs(self.X[k])
            recovery = (target_abs + 0.05) / (query_abs + 0.05)
            
            # Multiplicative optimal retrieval signal
            pi_k = pi_geom * (1.0 + 12.0 * target_sq) * (1.0 + 3.0 * recovery)
            pi_ensemble += weights[i] * pi_k
            
        pi = pi_ensemble / np.mean(pi_ensemble)
        pi = np.clip(pi, 0.1, 10.0)
        
        # Safety floor to ensure we never regress below identity (1.0x)
        pi = 0.85 * pi + 0.15 * np.ones(self.N)
        pi = pi / np.mean(pi)
        pi = np.clip(pi, 0.1, 10.0)
        return pi
