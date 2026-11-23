"""The out-of-domain check: does the classifier's reading of a Bank of Canada statement say anything
about the Bank's next decision?

The classifier was trained on FOMC sentences only. Each Bank of Canada statement in the lab's
corpus is split into sentences, its first paragraph (which announces today's decision) dropped,
and scored: stance = (hawkish - dovish) / sentences, from -1 (all dovish) to +1 (all hawkish).
The next scheduled decision's change in basis points (a hold is 0) comes from
data/events/boc_fad.csv. Reduced form: a correlation, not a causal estimate.
"""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import pandas as pd

from wsl.paths import EVENTS_DIR, ROOT
from wsl.stance.study import RESULTS, Log, LoraClassifier, protocol, provenance

STATEMENTS = ROOT / "data" / "rag" / "statements.jsonl"
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def sentences(paragraphs: list[str]) -> list[str]:
    out: list[str] = []
    for p in paragraphs[1:]:
        out += [s.strip() for s in _SPLIT.split(p) if len(s.strip()) > 20]
    return out


def stance_index(labels: np.ndarray) -> float:
    """(hawkish - dovish) / n, labels coded 0 dovish, 1 hawkish, 2 neutral."""
    labels = np.asarray(labels)
    return float(((labels == 1).sum() - (labels == 0).sum()) / max(len(labels), 1))


def next_decisions(events: pd.DataFrame) -> pd.DataFrame:
    """Scheduled decisions with the change at the following scheduled decision (bp; hold is 0)."""
    e = events[events["scheduled"]].copy().sort_values("date").reset_index(drop=True)
    e["change"] = np.where(e["title_direction"] == "hold", 0.0, e["change_bp"].astype(float))
    e["next_change"] = e["change"].shift(-1)
    e["next_date"] = e["date"].shift(-1)
    return e[["date", "change", "next_change", "next_date"]].dropna()


def spearman_ci(
    x: np.ndarray, y: np.ndarray, draws: int = 10000, seed: int = 0, level: float = 0.90
) -> dict[str, float]:
    from scipy.stats import spearmanr

    rng = np.random.default_rng(seed)
    n = len(x)
    rho = float(spearmanr(x, y).statistic)
    b = np.empty(draws)
    for k in range(draws):
        i = rng.integers(0, n, n)
        b[k] = spearmanr(x[i], y[i]).statistic
    lo, hi = np.nanquantile(b, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"rho": rho, "lo": float(lo), "hi": float(hi)}


def run(log: Log = print) -> dict[str, Any]:
    import statsmodels.api as sm

    dev = json.loads((RESULTS / "dev.json").read_text())
    spec = protocol()["models"]["lora_roberta"]
    clf = LoraClassifier.load(
        ROOT / "data" / "raw" / "stance" / "adapter", dev["lora_roberta"]["learning_rate"], spec
    )
    rows = [json.loads(line) for line in STATEMENTS.read_text().splitlines()]
    boc = [r for r in rows if r["bank"] == "boc"]
    scored = []
    for r in boc:
        sents = sentences(r["paragraphs"])
        if not sents:
            continue
        lab = clf.predict_proba(sents).argmax(1)
        scored.append(
            {
                "date": r["date"],
                "doc_id": r["doc_id"],
                "sentences": len(sents),
                "stance": stance_index(lab),
                "hawkish": int((lab == 1).sum()),
                "dovish": int((lab == 0).sum()),
            }
        )
    st = pd.DataFrame(scored)
    ev = next_decisions(pd.read_csv(EVENTS_DIR / "boc_fad.csv"))
    df = st.merge(ev, on="date", how="inner")
    ci = spearman_ci(df["stance"].to_numpy(), df["next_change"].to_numpy())
    X = sm.add_constant(df[["stance", "change"]])
    ols = sm.OLS(df["next_change"], X).fit(cov_type="HC1")
    out = {
        "statements_scored": len(st),
        "pairs": len(df),
        "first": df["date"].min(),
        "last": df["date"].max(),
        "spearman": {
            **ci,
            "verdict": "informative" if ci["lo"] > 0 else "not detectably informative",
        },
        "spearman_with_current_change": float(df["stance"].corr(df["change"], method="spearman")),
        "ols": {
            "coef_stance": float(ols.params["stance"]),
            "t_stance": float(ols.tvalues["stance"]),
            "coef_current": float(ols.params["change"]),
            "t_current": float(ols.tvalues["change"]),
            "r2": float(ols.rsquared),
        },
        "mean_stance_before": {
            k: float(df.loc[df["next_change"].pipe(f), "stance"].mean())
            for k, f in (
                ("hike", lambda s: s > 0),
                ("hold", lambda s: s == 0),
                ("cut", lambda s: s < 0),
            )
        },
        "label": "reduced form",
        "provenance": provenance(),
    }
    st.to_csv(RESULTS / "boc_stance.csv", index=False)
    (RESULTS / "external.json").write_text(json.dumps(out, indent=1))
    log(
        f"BoC stance vs next decision: Spearman {ci['rho']:.3f} "
        f"[{ci['lo']:.3f}, {ci['hi']:.3f}] over {len(df)} statements"
    )
    return out
