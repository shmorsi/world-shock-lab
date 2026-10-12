"""Data-generating processes with known impulse responses, for testing estimators.

A structural VAR(p): Y_t = A_1 Y_{t-1} + ... + A_p Y_{t-p} + B ε_t, with ε_t ~ N(0, I).
The response of variable i at horizon h to a one-standard-deviation shock j is (Ψ_h B)_{ij},
where Ψ_0 = I and Ψ_h = Σ_{l=1..min(h,p)} A_l Ψ_{h-l}.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StructuralVAR:
    A: tuple[np.ndarray, ...]
    B: np.ndarray

    @property
    def n(self) -> int:
        return int(self.B.shape[0])

    @property
    def p(self) -> int:
        return len(self.A)

    def companion_radius(self) -> float:
        n, p = self.n, self.p
        top = np.hstack(self.A)
        bottom = np.hstack([np.eye(n * (p - 1)), np.zeros((n * (p - 1), n))]) if p > 1 \
            else np.zeros((0, n))
        comp = np.vstack([top, bottom])
        return float(np.max(np.abs(np.linalg.eigvals(comp))))

    def irf(self, horizons: int) -> np.ndarray:
        """Array (H+1, n, n): [h, i, j] = response of variable i to shock j at horizon h."""
        n = self.n
        psi = [np.eye(n)]
        for h in range(1, horizons + 1):
            psi.append(sum((self.A[lag - 1] @ psi[h - lag]
                            for lag in range(1, min(h, self.p) + 1)), np.zeros((n, n))))
        return np.stack([m @ self.B for m in psi])

    def simulate(self, T: int, rng: np.random.Generator,
                 burn: int = 500) -> tuple[np.ndarray, np.ndarray]:
        """-> (Y of shape (T, n), ε of shape (T, n))."""
        if self.companion_radius() >= 1:
            raise ValueError("VAR is not stable")
        n, p = self.n, self.p
        eps = rng.standard_normal((T + burn, n))
        Y = np.zeros((T + burn, n))
        for t in range(p, T + burn):
            Y[t] = sum(self.A[lag - 1] @ Y[t - lag] for lag in range(1, p + 1)) + self.B @ eps[t]
        return Y[burn:], eps[burn:]


def textbook_var() -> StructuralVAR:
    """Persistent 3-variable VAR(2) (policy rate, activity, prices); shock 0 is 'monetary'.

    Companion spectral radius about 0.9: persistent enough that naive inference struggles,
    as with monthly macro data.
    """
    A1 = np.array([[0.70, 0.10, 0.05],
                   [-0.15, 0.60, 0.05],
                   [0.05, 0.10, 0.70]])
    A2 = np.array([[0.15, 0.00, 0.00],
                   [-0.05, 0.15, 0.00],
                   [0.00, 0.05, 0.15]])
    B = np.array([[0.50, 0.00, 0.00],
                  [-0.10, 0.80, 0.00],
                  [-0.05, 0.20, 0.40]])
    return StructuralVAR(A=(A1, A2), B=B)
