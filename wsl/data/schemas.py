"""pandera schemas for every table the build writes. A failed check stops the build."""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

from wsl.data.sources.base import STATUS_INVALID, STATUS_MISSING, STATUS_OK


def _ok_rows_have_values(df: pd.DataFrame) -> bool:
    return bool(df.loc[df["status"] == STATUS_OK, "value"].notna().all())


def _missing_rows_are_nan(df: pd.DataFrame) -> bool:
    flagged = df["status"].isin([STATUS_MISSING, STATUS_INVALID])
    return bool(df.loc[flagged, "value"].isna().all())


OBS = DataFrameSchema(
    {
        "series_id": Column(str),
        "date": Column(pa.DateTime),
        "value": Column(float, nullable=True),
        "status": Column(str, Check.isin([STATUS_OK, STATUS_MISSING, STATUS_INVALID])),
    },
    unique=["series_id", "date"],
    checks=[Check(_ok_rows_have_values, error="status ok but value missing"),
            Check(_missing_rows_are_nan, error="status missing but value present")],
    strict=True,
)

SERIES = DataFrameSchema(
    {
        "series_id": Column(str, unique=True),
        "parent_id": Column(str),
        "source": Column(str),
        "freq": Column(str, Check.isin(["D", "M", "A"])),
        "policy": Column(str, Check.isin(["C", "D", "L"])),
        "sha256": Column(str, Check.str_matches(r"^[0-9a-f]{64}$")),
        "url": Column(str, Check.str_startswith("https://")),
        "retrieved_at": Column(str),
    },
    strict=False,
)

FOMC = DataFrameSchema(
    {
        "ts": Column(pd.DatetimeTZDtype(tz="America/New_York"), unique=True),
        "date": Column(pa.DateTime),
        "pc1": Column(float),
        "MP_median": Column(float, nullable=True),
        "CBI_median": Column(float, nullable=True),
    },
    strict=False,
)

OPEC = DataFrameSchema(
    {
        "date": Column(pa.DateTime, unique=True),
        "Front": Column(float),
        "pc_fullsample": Column(float),
        "weekend": Column(bool),
    },
    strict=False,
)

ECB = DataFrameSchema(
    {"date": Column(pa.DateTime, unique=True), "pc1": Column(float)},
    strict=False,
)

BOC_FAD = DataFrameSchema(
    {
        "date": Column(pa.DateTime, unique=True),
        "scheduled": Column(bool),
        "decision": Column(str, Check.isin(["hike", "cut", "hold"]), nullable=True),
        "change_bp": Column(float, nullable=True),
        "url": Column(str, Check.str_startswith("https://www.bankofcanada.ca/")),
    },
    strict=False,
)

MONTHLY_SHOCKS = DataFrameSchema(
    {
        "shock_id": Column(str),
        "month": Column(pa.DateTime, Check(lambda s: (s.dt.day == 1).all())),
        "value": Column(float, nullable=True),
        "source": Column(str),
    },
    unique=["shock_id", "month"],
    strict=True,
)
