from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .config import DEFAULT_DISTANCE_THRESHOLD
from .storage import load_session_data
from .verifier import Verifier


@dataclass
class ProfileEntry:
    name: str
    dataset_path: Path
    verifier: Verifier


class ProfileManager:
    def __init__(self):
        self._profiles: Dict[str, ProfileEntry] = {}

    def load_profile(self, profile_name: str, path: Path):
        data = load_session_data(path)
        if not data.enrollment_samples:
            raise ValueError("Dataset has no enrollment runs to build a profile.")

        verifier = Verifier()
        X = np.stack(data.enrollment_samples, axis=0)
        verifier.fit(X)

        self._profiles[profile_name] = ProfileEntry(name=profile_name, dataset_path=path, verifier=verifier)

    def identify(
        self, X: np.ndarray, unknown_threshold: float = DEFAULT_DISTANCE_THRESHOLD
    ) -> Tuple[Optional[str], float, Dict[str, float]]:
        if not self._profiles:
            raise ValueError("No profiles loaded. Use load_profile() first.")
        if X.ndim != 2 or X.size == 0:
            raise ValueError("Test matrix must be 2D and non-empty.")

        distances: Dict[str, float] = {}
        best_name: Optional[str] = None
        best_distance = float("inf")

        for name, entry in self._profiles.items():
            dists, _ = entry.verifier.score(X)
            avg_distance = float(dists.mean())
            distances[name] = avg_distance
            if avg_distance < best_distance:
                best_distance = avg_distance
                best_name = name

        if best_distance > unknown_threshold:
            best_name = None

        return best_name, best_distance, distances

    def list_profiles(self):
        return list(self._profiles.keys())

    def clear(self):
        self._profiles.clear()

    def expected_feature_dim(self) -> Optional[int]:
        if not self._profiles:
            return None
        first = next(iter(self._profiles.values()))
        if first.verifier.mean is None:
            return None
        return int(first.verifier.mean.shape[0])

    def iter_profiles(self):
        return list(self._profiles.values())
