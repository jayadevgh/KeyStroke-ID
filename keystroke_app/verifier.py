from typing import Optional, Tuple

import numpy as np
from scipy.spatial.distance import mahalanobis

from .config import DEFAULT_DISTANCE_THRESHOLD


class Verifier:
    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.inv_cov: Optional[np.ndarray] = None
        self.distance_threshold: float = float(DEFAULT_DISTANCE_THRESHOLD)
        self._n_enrollment_runs: int = 0

    @property
    def n_enrollment_runs(self) -> int:
        return self._n_enrollment_runs

    def has_reference(self) -> bool:
        return self.mean is not None and self.inv_cov is not None

    def clear(self):
        self.mean = None
        self.inv_cov = None
        self._n_enrollment_runs = 0

    def fit(self, X: np.ndarray):
        if X.ndim != 2:
            raise ValueError("Enrollment data must be a 2D array shaped (n_runs, feature_dim).")
        n_runs, feature_dim = X.shape
        if n_runs < 3:
            raise ValueError("Need at least 3 enrollment runs to compute a stable covariance matrix.")

        mean = X.mean(axis=0).astype(np.float32)
        cov = np.cov(X.T)
        cov = np.asarray(cov, dtype=np.float32)
        if cov.ndim == 0:  # handle single-feature edge case
            cov = cov.reshape(1, 1)
        regularization = np.eye(feature_dim, dtype=np.float32) * 1e-4
        cov = cov + regularization
        inv_cov = np.linalg.inv(cov).astype(np.float32)

        self.mean = mean
        self.inv_cov = inv_cov
        self._n_enrollment_runs = n_runs

    def score(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.has_reference():
            raise ValueError("Verifier has not been fitted; enroll first.")
        if X.ndim != 2:
            raise ValueError("Test data must be a 2D array shaped (n_runs, feature_dim).")
        if X.shape[1] != int(self.mean.shape[0]):  # type: ignore[arg-type]
            raise ValueError(f"Feature dimension mismatch: got {X.shape[1]}, expected {self.mean.shape[0]}")

        distances = np.array(
            [float(mahalanobis(sample, self.mean, self.inv_cov)) for sample in X],
            dtype=np.float32,
        )
        inlier = distances <= self.distance_threshold
        return distances, inlier
