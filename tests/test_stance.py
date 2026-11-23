"""Policy stance: the parts that need no model download (index, event join, bootstrap, gate)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from wsl.stance import study
from wsl.stance.external import next_decisions, sentences, stance_index


def test_stance_index_range_and_sign():
    assert stance_index(np.array([1, 1, 2, 0])) == pytest.approx(0.25)
    assert stance_index(np.array([0, 0])) == -1.0
    assert stance_index(np.array([])) == 0.0


def test_sentences_drop_the_announcement_paragraph():
    paras = [
        "The Bank today raised its policy rate to 5%.",
        "Inflation remains well above target. Further increases may be needed. Ok.",
    ]
    s = sentences(paras)
    assert s == ["Inflation remains well above target.", "Further increases may be needed."]


def test_next_decision_uses_scheduled_rows_and_holds_are_zero():
    ev = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-02-01", "2020-02-15", "2020-03-01"],
            "scheduled": [True, True, False, True],
            "title_direction": ["hold", "cut", "cut", "hike"],
            "change_bp": [np.nan, -25.0, -50.0, 25.0],
        }
    )
    n = next_decisions(ev)
    assert n["next_change"].tolist() == [-25.0, 25.0]
    assert n["change"].tolist() == [0.0, -25.0]


def test_bootstrap_difference_and_verdict():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 400)
    good = np.where(rng.random(400) < 0.9, y, (y + 1) % 3)
    bad = np.where(rng.random(400) < 0.5, y, (y + 1) % 3)
    ci = study.bootstrap_diff(y, good, bad, draws=500)
    assert ci["lo"] > 0 and study.verdict(ci) == "helps"
    assert study.verdict({"lo": -0.1, "hi": 0.1}) == "no detectable difference"


def test_gate_refuses_a_second_scoring(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "RESULTS", tmp_path)
    (tmp_path / "test.json").write_text("{}")
    monkeypatch.setattr(study, "_git", lambda *a: type("R", (), {"returncode": 0, "stdout": ""})())
    with pytest.raises(study.GateError, match="scored once"):
        study.check_gate()


def test_protocol_fixes_the_choices():
    p = study.protocol()
    assert p["data"]["revision"] and p["models"]["lora_roberta"]["base_revision"]
    assert p["primary_comparison"]["interval"].startswith("paired bootstrap")
