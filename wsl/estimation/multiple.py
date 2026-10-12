"""Multiple-testing control: Benjamini-Hochberg false discovery rate."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def bh_qvalues(p: NDArray[np.float64]) -> NDArray[np.float64]:
    """Benjamini-Hochberg adjusted p-values (q-values); reject where q <= q*.

    q_(i) = min_{j >= i} p_(j) · m / j, capped at 1, returned in the input order.
    """
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out


def bh_reject(p: NDArray[np.float64], q: float = 0.10) -> NDArray[np.bool_]:
    return bh_qvalues(p) <= q
