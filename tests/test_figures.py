"""The print SVGs stay what the generator already writes. A dark copy moves no point."""

from __future__ import annotations

import re
from pathlib import Path

from wsl.mlstudy.report import _figure, _load
from wsl.paths import ROOT
from wsl.report.docs import facts
from wsl.report.figures import canada_lens

FIG = ROOT / "docs" / "figures"


def _structure(svg: str) -> str:
    svg = re.sub(r'font-family="[^"]*"', 'font-family="_"', svg)
    svg = re.sub(r'fill="#[0-9A-Fa-f]{6}"', 'fill="_"', svg)
    svg = re.sub(r'stroke="#[0-9A-Fa-f]{6}"', 'stroke="_"', svg)
    return svg


def _same_drawing(light: Path, dark: Path, committed: Path) -> None:
    light_txt, dark_txt = light.read_text(), dark.read_text()
    assert light_txt == committed.read_text()
    assert _structure(light_txt) == _structure(dark_txt)
    assert 'fill="#000000"' in dark_txt
    assert "#ffffff" not in dark_txt


def test_canada_lens_dark_palette_keeps_the_print_drawing(tmp_path: Path) -> None:
    light, dark = tmp_path / "fig1.svg", tmp_path / "fig1-dark.svg"
    canada_lens(facts(), light)
    canada_lens(facts(), dark, dark=True)
    _same_drawing(light, dark, FIG / "fig1_canada_lens.svg")


def test_ml_figure_dark_palette_keeps_the_print_drawing(tmp_path: Path) -> None:
    light, dark = tmp_path / "ml.svg", tmp_path / "ml-dark.svg"
    doc = _load()
    _figure(doc, light)
    _figure(doc, dark, dark=True)
    _same_drawing(light, dark, FIG / "fig_ml_study.svg")
