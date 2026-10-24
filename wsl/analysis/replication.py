"""Replication of published qualitative findings ("paper vs ours", docs/REPLICATION.md).

Each claim names its source (paraphrased from the paper's abstract, verified 2026-10-12),
the test we run, and the sign we expect. Verdict rules, fixed before the first real run:
  replicated     expected sign and the 90% band excludes zero
  same sign      expected sign, band includes zero
  opposite       wrong sign and the 90% band excludes zero
  opposite (ns)  wrong sign, band includes zero
Claims marked `mechanical` are close to true by construction (e.g. the Fed surprise is
measured in a 30-minute window and the daily window contains it); they check plumbing,
not economics, and are flagged as such.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from wsl.analysis.eventstudy import event_irf, monthly_irf
from wsl.estimation.irf import IRF
from wsl.shocks import events as ev

Family = Literal["fomc", "bauer_swanson", "opec", "monthly"]

JK = "Jarociński & Karadi (2020), AEJ: Macroeconomics 12(2)"
KZ = "Känzig (2021), AER 111(4)"
BS = "Bauer & Swanson (2023), NBER Macroeconomics Annual 37"
BRIEF = "Project brief (expected sign; no specific paper claim verified here)"


@dataclass(frozen=True)
class Claim:
    id: str
    source: str
    claim: str
    family: Family
    shock: str
    outcome: str
    horizon: int
    expected: int  # +1 or -1
    per: float = 1.0
    controls: tuple[str, ...] = ()
    mechanical: bool = False
    test: str = ""


CLAIMS: list[Claim] = [
    Claim("JK1", JK, "A pure monetary policy tightening raises interest rates", "fomc",
          "mp_bp", "US2Y", 0, +1, per=10, controls=("cbi_bp",), mechanical=True,
          test="US 2-year yield, day 0, per 10 bp MP shock"),
    Claim("JK2", JK, "A pure monetary policy tightening lowers stock prices", "fomc",
          "mp_bp", "SPY.tr", 0, -1, per=10, controls=("cbi_bp",), mechanical=True,
          test="S&P 500 (SPY total return), day 0"),
    Claim("JK3", JK, "A central bank information shock raises interest rates", "fomc",
          "cbi_bp", "US2Y", 0, +1, per=10, controls=("mp_bp",), mechanical=True,
          test="US 2-year yield, day 0, per 10 bp CBI shock"),
    Claim("JK4", JK, "A central bank information shock raises stock prices", "fomc",
          "cbi_bp", "SPY.tr", 0, +1, per=10, controls=("mp_bp",), mechanical=True,
          test="SPY total return, day 0"),
    Claim("JK5", JK, "The policy shock's effect on rates persists beyond the day", "fomc",
          "mp_bp", "US2Y", 5, +1, per=10, controls=("cbi_bp",),
          test="US 2-year yield, cumulative to day 5"),
    Claim("FX1", BRIEF, "A contractionary Fed surprise strengthens the US dollar", "fomc",
          "pc1_bp", "USD_BROAD", 0, +1, per=10,
          test="Broad dollar index (H.10, noon fix: day 0 is the day after a 14:00 FOMC)"),
    Claim("FX2", BRIEF, "A contractionary Fed surprise weakens the Canadian dollar", "fomc",
          "pc1_bp", "USDCAD", 0, +1, per=10, test="USD/CAD (CAD per USD), noon fix, day 0"),
    Claim("BS1", BS, "Asset-price effects survive orthogonalization: yields", "bauer_swanson",
          "mps_orth_bp", "US2Y", 0, +1, per=10, mechanical=True,
          test="US 2-year yield on MPS_ORTH (compare with BS1r)"),
    Claim("BS1r", BS, "Reference: same test on the raw MPS", "bauer_swanson",
          "mps_bp", "US2Y", 0, +1, per=10, mechanical=True, test="US 2-year yield on MPS"),
    Claim("BS2", BS, "Asset-price effects survive orthogonalization: stocks", "bauer_swanson",
          "mps_orth_bp", "SPY.tr", 0, -1, per=10, test="SPY on MPS_ORTH (compare with BS2r)"),
    Claim("BS2r", BS, "Reference: same test on the raw MPS", "bauer_swanson",
          "mps_bp", "SPY.tr", 0, -1, per=10, test="SPY on MPS"),
    Claim("KZ1", KZ, "Negative oil supply news raises oil prices immediately", "opec",
          "pc_pct", "WTI", 0, +1, mechanical=True,
          test="WTI spot, day 0, per 1% surprise in oil futures"),
    Claim("KZ2", KZ, "Inflation expectations rise", "opec", "pc_pct", "US_BE5Y_PD", 5, +1,
          test="US 5-year breakeven (public-domain inputs), cumulative to day 5, bp"),
    Claim("KZ3", KZ, "The dollar weakens", "opec", "pc_pct", "USD_BROAD", 5, -1,
          test="Broad dollar index, cumulative to day 5 (paper: monthly VAR)"),
    Claim("KZ4", KZ, "US prices rise", "monthly", "KAENZIG_OIL_NEWS", "US_CPI_SA", 12, +1,
          test="US CPI (SA), monthly LP, cumulative to month 12, per unit of the news shock"),
    Claim("KZ5", KZ, "US economic activity falls", "monthly", "KAENZIG_OIL_NEWS", "US_IP",
          12, -1, test="US industrial production, monthly LP, month 12"),
    Claim("BS3", BS, "Macro effects with the orthogonalized surprise: activity falls",
          "monthly", "BS_MPS_ORTH", "US_IP", 18, -1, per=0.1,
          test="US industrial production, monthly LP, month 18, per 10 bp (compare BS3r)"),
    Claim("BS3r", BS, "Reference: same test on the raw MPS", "monthly", "BS_MPS", "US_IP", 18,
          -1, per=0.1, test="US industrial production on raw MPS, month 18"),
]


@dataclass
class ClaimResult:
    claim: Claim
    irf: IRF

    @property
    def est(self) -> float:
        return float(self.irf.coef[self.claim.horizon])

    @property
    def band90(self) -> tuple[float, float]:
        lo, hi = self.irf.band(0.90)
        return float(lo[self.claim.horizon]), float(hi[self.claim.horizon])

    @property
    def n(self) -> int:
        return int(self.irf.nobs[self.claim.horizon])

    @property
    def verdict(self) -> str:
        lo, hi = self.band90
        sig = lo > 0 or hi < 0
        right = np.sign(self.est) == self.claim.expected
        if right:
            return "replicated" if sig else "same sign"
        return "opposite" if sig else "opposite (ns)"


def _events(family: Family) -> pd.DataFrame:
    if family == "fomc":
        return ev.fomc().dropna(subset=["mp_bp", "cbi_bp"])
    if family == "bauer_swanson":
        return ev.bauer_swanson()
    return ev.opec()


def run_claim(c: Claim) -> ClaimResult:
    if c.family == "monthly":
        irf = monthly_irf(c.shock, c.outcome, per=c.per)
    else:
        irf = event_irf(_events(c.family), c.shock, c.outcome, other_shocks=c.controls,
                        per=c.per)
    return ClaimResult(c, irf)


def run_all() -> list[ClaimResult]:
    return [run_claim(c) for c in CLAIMS]


def snapshot(irf: IRF, horizons: tuple[int, ...]) -> list[dict[str, float]]:
    lo, hi = irf.band(0.90)
    return [{"h": h, "coef": float(irf.coef[h]), "lo90": float(lo[h]), "hi90": float(hi[h]),
             "n": int(irf.nobs[h])} for h in horizons if h < len(irf.coef)]

