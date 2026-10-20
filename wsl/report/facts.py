"""Every number the generated documents use, read from committed result files."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from wsl.paths import ROOT

Z90 = 1.6448536
RESULTS = ROOT / "results"
CATALOG = ROOT / "web" / "public" / "data" / "catalog.json"
MINUS = "−"


@dataclass
class Fmt:
    """Formats numbers and remembers every string it produced (for the no-invented-numbers
    test)."""

    emitted: set[str] = field(default_factory=set)

    def _keep(self, s: str) -> str:
        self.emitted.add(s)
        return s

    def num(self, v: float, d: int = 2, signed: bool = False) -> str:
        s = f"{abs(v):,.{d}f}"
        if float(s.replace(",", "")) == 0:  # never "−0.00"
            v = 0.0
        if signed:
            s = ("+" if v > 0 else MINUS if v < 0 else "") + s
        elif v < 0:
            s = MINUS + s
        return self._keep(s)

    def words(self, k: int, zero: str = "none") -> str:
        """A count in a sentence: "none" for zero, otherwise the number."""
        return zero if k == 0 else self.count(k)

    def count(self, v: float) -> str:
        return self._keep(f"{round(v):,d}")

    def pct(self, share: float, d: int = 0) -> str:
        return self._keep(f"{100 * share:.{d}f}") + "%"

    def band(self, lo: float, hi: float, d: int = 2, signed: bool = True) -> str:
        return f"{self.num(lo, d, signed)} to {self.num(hi, d, signed)}"

    def year(self, y: int | str) -> str:
        return self._keep(str(y))

    def date(self, iso: str) -> str:
        """An ISO date taken from a data file (e.g. a forecast's decision date)."""
        return self._keep(iso[:10])

    def tokens(self) -> set[str]:
        """Numeric tokens (as the test extracts them) of everything emitted."""
        return {t for s in self.emitted for t in NUMBER.findall(s)}


NUMBER = re.compile(r"[+−-]?\d+(?:,\d{3})*(?:\.\d+)?")


def _load(path: Path) -> Any:
    return json.loads(path.read_text())


def _final_eval() -> dict[str, Any]:
    runs = sorted((RESULTS / "final_eval").glob("*.json"))
    doc: dict[str, Any] = _load(runs[-1])
    doc["_file"] = runs[-1].name
    doc["_runs"] = len(runs)
    return doc


def _short(k: str) -> str:
    return k.split(":")[0]


def _study() -> dict[str, Any]:
    """ml-study-v1. Separate from the M5 block above it in the documents."""
    doc = _load(RESULTS / "mlstudy" / "final.json")
    val = _load(RESULTS / "mlstudy" / "validation.json")
    mk, tk = doc["chosen"]["mean"], doc["chosen"]["tail"]
    mean_params = json.loads(mk.split(":", 1)[1])
    tail_params = json.loads(tk.split(":", 1)[1])
    mean, tail = doc["mean"], doc["tail"]
    book = doc["portfolio"]["difference_ml_minus_gauss"]
    return {
        "n_test": doc["n_test"], "years": doc["test_years"],
        "train_events": doc["counts"]["train_events"],
        "val_events": doc["counts"]["validation_events"],
        "channels": int(mean_params["channels"]),
        "weight_decay": float(mean_params["weight_decay"]),
        "depth": int(tail_params["max_depth"]),
        "mean_ml": mean["pooled_scaled_mae"][mk],
        "mean_lp": mean["pooled_scaled_mae"]["lp"],
        "mean_zero": mean["pooled_scaled_mae"]["zero"],
        "mean_diff": mean["difference_ml_minus_lp"]["point"],
        "mean_ci": mean["difference_ml_minus_lp"]["ci90"],
        "mean_verdict": mean["verdict"],
        "static_diff": mean["static_refit"]["difference_ml_minus_lp"]["point"],
        "static_ci": mean["static_refit"]["difference_ml_minus_lp"]["ci90"],
        "static_verdict": mean["static_refit"]["verdict"],
        "static_ml": mean["static_refit"]["pooled_scaled_mae"][mk],
        "static_lp": mean["static_refit"]["pooled_scaled_mae"]["lp"],
        "tail_ml": tail["pooled_scaled_pinball"][tk],
        "tail_gauss": tail["pooled_scaled_pinball"]["gauss_lp"],
        "tail_hist": tail["pooled_scaled_pinball"]["hist"],
        "tail_diff": tail["difference_ml_minus_gauss"]["point"],
        "tail_ci": tail["difference_ml_minus_gauss"]["ci90"],
        "tail_verdict": tail["verdict"],
        "hit_ml": tail["hit_rate_05"][tk],
        "hit_gauss": tail["hit_rate_05"]["gauss_lp"],
        "hit_hist": tail["hit_rate_05"]["hist"],
        "cov_tail": tail["coverage_90"][tk],
        "cov_gauss": tail["coverage_90"]["gauss_lp"],
        "cov_mean": mean["coverage90"][mk],
        "cov_lp": mean["coverage90"]["lp"],
        "fz_ml": tail["fissler_ziegel"][tk],
        "fz_gauss": tail["fissler_ziegel"]["gauss_lp"],
        "fz_hist": tail["fissler_ziegel"]["hist"],
        "shap_surprise": doc["shap"]["share"]["surprise"],
        "shap_state": doc["shap"]["share"]["macro_state"],
        "shap_family": doc["shap"]["share"]["family"],
        "shap_path": doc["shap"]["share"]["return_path"],
        "shap_vol": doc["shap"]["share"]["asset_vol"],
        "ari": doc["embeddings"]["ari"],
        "ari_p": doc["embeddings"]["ari_permutation_p"],
        "port_diff": book["point"], "port_ci": book["ci90"],
        "port_ml": doc["portfolio"]["pooled_scaled_pinball"][tk],
        "port_gauss": doc["portfolio"]["pooled_scaled_pinball"]["gauss_lp"],
        "matched": doc["inputs"]["matched_pin"], "revised": doc["inputs"]["revised"],
        "val_lp": val["pooled_scaled_mae"]["lp"],
        "val_mean": val["pooled_scaled_mae"][mk],
        "val_gauss": val["pooled_scaled_pinball"]["gauss_lp"],
        "val_tail": val["pooled_scaled_pinball"][tk],
        "stress_sd": doc["stress"]["size_sd"],
        "spy_q50_zero": doc["stress"]["assets"]["SPY"]["q50_zero"],
        "spy_q50_shock": doc["stress"]["assets"]["SPY"]["q50_shock"],
        "spy_q05_shock": doc["stress"]["assets"]["SPY"]["q05_shock"],
    }


def facts() -> dict[str, Any]:
    """Plain numbers (not yet formatted), grouped by topic."""
    canada = {r["outcome"]: r for r in _load(RESULTS / "m4" / "canada.json")["results"]}

    def fed(o: str, h: str) -> tuple[float, float, float, int]:
        m, se, n = canada[o]["fed"][h]
        return m, m - Z90 * se, m + Z90 * se, n

    def boc(o: str, h: str) -> tuple[float, float, float, int]:
        m, se, n = canada[o]["boc"][h]
        return m, m - Z90 * se, m + Z90 * se, n

    def rs(o: str, h: str) -> tuple[float, float, float]:
        p, lo, hi = canada[o]["boc_rigobon_sack"][h]
        return p, lo, hi

    fin = _final_eval()
    mae = {_short(k): v for k, v in fin["pooled_scaled_mae"].items()}
    m3 = _load(RESULTS / "m3" / "replication.json")
    main = [c for c in m3["claims"] if not c["id"].endswith("r")]
    published = [c for c in main if not c["source"].startswith("Project brief")]
    checks = [c for c in main if c["source"].startswith("Project brief")]
    substantive = [c for c in published if not c["mechanical"]]
    placebo = m3["placebo"]
    bp = _load(RESULTS / "m3" / "boc_proxy.json")
    sd = _load(RESULTS / "m4" / "state_dependence.json")["results"]
    pvals = [c[k] for c in sd for k in ("p_joint", "p_headline")]
    qvals = [c[k] for c in sd for k in ("q_joint", "q_headline")]
    rob = _load(RESULTS / "m4" / "robustness.json")["results"]
    net = _load(RESULTS / "m4" / "network.json")["results"]
    exp = _load(RESULTS / "m4" / "exporters.json")["results"]
    vv = _load(RESULTS / "m4" / "var_vs_lp.json")["results"]
    bs = vv["BS_MPS_ORTH"]
    ip = next(r for r in bs["lp_iv"]["US_IP"]["rows"] if r["h"] == 12)
    sim = _load(RESULTS / "simulation.json")
    ev_cfg = yaml.safe_load((ROOT / "config" / "evaluation.yaml").read_text())
    live_cfg = yaml.safe_load((ROOT / "config" / "live.yaml").read_text())
    forecasts = sorted((ROOT / "live" / "forecasts").glob("*.json"))
    scores = sorted((ROOT / "live" / "scores").glob("*.json"))
    exp_h0 = next(r for r in exp["panel"] if r["horizon"] == 0)

    return {
        "canada": {
            "tsx": {"fed0": fed("TSX.tr", "0"), "boc0": boc("TSX.tr", "0"),
                    "rs0": rs("TSX.tr", "0"), "fed5": fed("TSX.tr", "5"),
                    "boc5": boc("TSX.tr", "5"), "rs5": rs("TSX.tr", "5")},
            "banks": {"fed0": fed("CA_BANKS_EW6", "0"), "boc0": boc("CA_BANKS_EW6", "0"),
                      "rs0": rs("CA_BANKS_EW6", "0")},
            "xeg": {"fed0": fed("XEG.tr", "0"), "boc0": boc("XEG.tr", "0"),
                    "rs0": rs("XEG.tr", "0")},
            "ca10": {"fed0": fed("CA10Y", "0"), "boc0": boc("CA10Y", "0"),
                     "rs0": rs("CA10Y", "0")},
            "ca2": {"fed0": fed("CA2Y", "0")},
            "cad": {"fed0": fed("USDCAD", "0"), "boc0": boc("USDCAD", "0"),
                    "rs0": rs("USDCAD", "0")},
            "fed_f_min": min(r["fed_first_stage_f"] for r in canada.values()),
        },
        "study": _study(),
        "ml": {
            "file": fin["_file"], "runs": fin["_runs"], "protocol": fin["protocol_tag"],
            "n_test": fin["test_events_primary"], "years": fin["test_years"],
            "mae": mae, "diff": fin["difference_ml_minus_econ"]["point"],
            "ci": fin["difference_ml_minus_econ"]["ci90"], "verdict": fin["verdict"],
            "static_diff": fin["static_refit"]["difference_ml_minus_econ"]["point"],
            "static_ci": fin["static_refit"]["difference_ml_minus_econ"]["ci90"],
            "importance_surprise": max(v["surprise components"] for v in
                                       fin["attributions"].values()
                                       if isinstance(v, dict) and "surprise components" in v)
            if isinstance(fin.get("attributions"), dict) else None,
        },
        "replication": {
            "published": len(published),
            "replicated": sum(c["verdict"] == "replicated" for c in published),
            "same_sign": sum(c["verdict"].startswith("same") for c in published),
            "opposite": sum(c["verdict"].startswith("opposite") for c in published),
            "substantive": len(substantive),
            "substantive_ok": sum(c["verdict"] == "replicated" for c in substantive),
            "checks": len(checks), "checks_ok": sum(c["verdict"] == "replicated" for c in checks),
            "placebo_draws": 1000,
            "placebo_max_p": max(v["p_value"] for k, v in placebo.items()
                                 if k.split("_")[0] in ("JK1", "JK2", "KZ1")),
        },
        "boc_proxy": {"n": bp["n_events"], "ratio": bp["variance_ratio"]["ratio"],
                      "ratio_ci": bp["variance_ratio"]["ci90"],
                      "purged": bp["variance_ratio_purged"]["ratio"],
                      "purged_ci": bp["variance_ratio_purged"]["ci90"]},
        "state": {"cells": len(sd), "tests": len(pvals),
                  "raw": sum(p <= 0.10 for p in pvals), "expected": 0.10 * len(pvals),
                  "survive": sum(q <= 0.10 for q in qvals), "q": 0.10, "pvals": pvals},
        "robust": {"n": len(rob), "robust": sum(r["verdict"] == "robust" for r in rob)},
        "network": {"tested": len(net), "edges": sum(bool(e["edge"]) for e in net)},
        # exporters.json stores (estimate, s.e.) and, for the difference, (estimate, s.e., p).
        "oil": {"diff": (exp_h0["difference"][0],
                         exp_h0["difference"][0] - Z90 * exp_h0["difference"][1],
                         exp_h0["difference"][0] + Z90 * exp_h0["difference"][1],
                         exp_h0["difference"][2])},
        "var": {"f": bs["first_stage_f"], "lp_f": bs["lp_first_stage_f"], "T": bs["T"],
                "sample": bs["sample"], "ip12": (ip["coef"], ip["lo90"], ip["hi90"]),
                "ip12_ar": ip.get("ar90")},
        "sim": sim,
        "config": {"train_end": ev_cfg["splits"]["train"]["end"][:4],
                   "val": (ev_cfg["splits"]["validation"]["start"][:4],
                           ev_cfg["splits"]["validation"]["end"][:4]),
                   "test": (ev_cfg["splits"]["test"]["start"][:4],
                            ev_cfg["splits"]["test"]["end"][:4]),
                   "grid": live_cfg["grid_bp"], "horizons": live_cfg["horizons"]},
        "live": {"forecasts": [_load(p) for p in forecasts], "scores": [_load(p) for p in scores]},
    }
