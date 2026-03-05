"""
WelfordNormalizer — pure NumPy online mean/variance estimator.

Kept in its own module so it can be imported without pulling in
MuJoCo or any other heavy simulation dependency.
"""

import numpy as np


class WelfordNormalizer:
    """
    Incremental (Welford) mean and variance estimator for observation normalisation.

    After each call to ``update`` the internal statistics are updated.
    ``normalize`` standardises to approximately zero mean, unit variance.
    """

    def __init__(self, shape: int) -> None:
        self.n    = 0
        self.mean = np.zeros(shape, dtype=np.float64)
        self.M2   = np.zeros(shape, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        """Update running statistics with a new sample."""
        self.n += 1
        delta      = x - self.mean
        self.mean += delta / self.n
        delta2     = x - self.mean
        self.M2   += delta * delta2

    @property
    def variance(self) -> np.ndarray:
        if self.n < 2:
            return np.ones_like(self.mean)
        return self.M2 / (self.n - 1)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.variance + 1e-8)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def save(self, path: str) -> None:
        np.savez(path, n=np.array(self.n), mean=self.mean, M2=self.M2)

    def load(self, path: str) -> None:
        data      = np.load(path)
        self.n    = int(data["n"])
        self.mean = data["mean"].astype(np.float64)
        self.M2   = data["M2"].astype(np.float64)
