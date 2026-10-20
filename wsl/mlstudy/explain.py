"""Exact grouped Shapley values, and the asset-embedding cluster check.

Five groups or fewer, so every coalition is evaluated. The values satisfy the
efficiency identity: baseline prediction plus the sum of group values equals the
prediction, up to numerical error.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from wsl.ml.dataset import ASSET_NAMES, STATE_COLS
from wsl.ml.models import FAMILY_DUMMIES
from wsl.mlstudy.panels import SHOCK_NAMES


def group_index(feature_names: list[str], use_path: bool) -> list[tuple[str, str, np.ndarray]]:
    """(name, kind, column indices). ``kind`` is 'cols' or 'path'."""
    def idx(cols: list[str]) -> np.ndarray:
        return np.array([feature_names.index(c) for c in cols], dtype=int)

    groups = [
        ("surprise", "cols", idx(SHOCK_NAMES)),
        ("family", "cols", idx(list(FAMILY_DUMMIES))),
        ("macro_state", "cols", idx(list(STATE_COLS))),
        ("asset_vol", "cols", idx(["asset_vol"])),
    ]
    if use_path:
        groups.append(("return_path", "path", np.array([], dtype=int)))
    return groups


def grouped_shap(
    predict: Callable[[np.ndarray, np.ndarray], np.ndarray],
    X: np.ndarray,
    path: np.ndarray,
    baseline_x: np.ndarray,
    baseline_path: np.ndarray,
    groups: list[tuple[str, str, np.ndarray]],
) -> tuple[np.ndarray, float]:
    """Shapley values per row and group, and the largest efficiency gap.

    ``predict(X, path) -> (n,)``. Baselines have the same shape as ``X`` and ``path``.
    """
    n = len(X)
    g = len(groups)
    nmask = 1 << g
    preds = np.empty((nmask, n))
    for mask in range(nmask):
        xm = baseline_x.copy()
        pm = baseline_path.copy()
        for j, (_name, kind, cols) in enumerate(groups):
            if mask & (1 << j):
                if kind == "path":
                    pm = path.copy()
                else:
                    xm[:, cols] = X[:, cols]
        preds[mask] = predict(xm, pm)
    shap = np.zeros((n, g))
    for j in range(g):
        others = [m for m in range(nmask) if (m & (1 << j)) == 0]
        for mask in others:
            k = int(mask.bit_count())
            # |S|!(g-|S|-1)! / g!  = 1 / (g * C(g-1, |S|))
            weight = 1.0 / (g * _binom(g - 1, k))
            shap[:, j] += weight * (preds[mask | (1 << j)] - preds[mask])
    gap = float(np.max(np.abs(shap.sum(axis=1) + preds[0] - preds[(1 << g) - 1])))
    return shap, gap


def _binom(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 1
    c = 1
    for i in range(k):
        c = c * (n - i) // (i + 1)
    return max(c, 1)


def embedding_agreement(
    vectors: np.ndarray, groups: dict[str, list[str]], permutations: int, seed: int,
) -> dict[str, object]:
    """k-means against the pre-specified asset classes. Permutation p-value of the ARI."""
    label_of = {a: i for i, (_name, members) in enumerate(groups.items()) for a in members}
    order = [a for a in ASSET_NAMES if a in label_of]
    # vectors are in ASSET_NAMES order.
    labels = np.array([label_of[a] for a in ASSET_NAMES])
    km = KMeans(n_clusters=len(groups), n_init=20, random_state=seed).fit(vectors)
    ari = float(adjusted_rand_score(labels, km.labels_))
    rng = np.random.default_rng(seed)
    null = np.empty(permutations)
    for i in range(permutations):
        null[i] = adjusted_rand_score(rng.permutation(labels), km.labels_)
    p = float((1 + np.sum(null >= ari - 1e-12)) / (permutations + 1))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    cosine = (vectors / norms) @ (vectors / norms).T
    return {
        "assets": list(ASSET_NAMES),
        "groups": [int(x) for x in labels],
        "clusters": [int(x) for x in km.labels_],
        "ari": ari,
        "ari_permutation_p": p,
        "cosine": cosine.round(4).tolist(),
        "order_note": order,
    }
