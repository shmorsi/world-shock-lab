"""SVG figures for the research note, drawn from the same facts as the text.

Plain SVG (no plotting library): one hue per entity (Fed blue, BoC orange, the palette the
web app uses), one axis per panel, recessive grid, direct labels, 90% bands as whiskers or
light fills and 68% as darker fills."""

from __future__ import annotations

import json
from collections.abc import Sequence
from html import escape
from pathlib import Path
from typing import Any

from wsl.report.facts import CATALOG, MINUS, Z90

BLUE, ORANGE, INK, INK2, MUTED, GRID, LINE = ("#2a78d6", "#eb6834", "#121722", "#485164",
                                              "#5d6677", "#e3e7ee", "#c9d0dc")
FONT = "IBM Plex Sans, Helvetica Neue, Arial, sans-serif"


def _fmt(v: float, d: int) -> str:
    if abs(v) < 0.5 * 10 ** -d:
        return "0"
    s = f"{abs(v):.{d}f}"
    return ("+" if v > 0 else MINUS) + s


def _ticks(lo: float, hi: float, n: int = 5) -> tuple[list[float], int]:
    import math

    span = (hi - lo) or 1.0
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = float(mag * next(m for m in (1, 2, 5, 10) if raw / mag <= m))
    first = math.ceil(lo / step - 1e-9) * step
    out = []
    v = first
    while v <= hi + 1e-9:
        out.append(round(v, 10))
        v += step
    return out, max(0, -math.floor(math.log10(step) + 1e-9))


class Svg:
    def __init__(self, w: int, h: int, title: str) -> None:
        self.w, self.h = w, h
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
                      f'width="{w}" height="{h}" role="img" aria-label="{escape(title)}" '
                      f'font-family="{FONT}">',
                      f"<title>{escape(title)}</title>",
                      f'<rect width="{w}" height="{h}" fill="#ffffff"/>']

    def text(self, x: float, y: float, s: str, size: float = 12, fill: str = INK,
             anchor: str = "start", weight: int = 400) -> None:
        self.parts.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
                          f'text-anchor="{anchor}" font-weight="{weight}">{escape(s)}</text>')

    def line(self, x1: float, y1: float, x2: float, y2: float, stroke: str = GRID,
             width: float = 1, dash: str | None = None) -> None:
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                          f'stroke="{stroke}" stroke-width="{width}"{d}/>')

    def raw(self, s: str) -> None:
        self.parts.append(s)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join([*self.parts, "</svg>"]) + "\n")
        return path


def canada_lens(f: dict[str, Any], path: Path) -> Path:
    """Dot-and-whisker: Fed vs BoC (event LP and Rigobon-Sack), announcement day."""
    c = f["canada"]
    panels = [
        ("Canadian equities, % on the announcement day", "%", [
            ("S&P/TSX", c["tsx"]), ("Big-6 banks", c["banks"]), ("Energy (XEG)", c["xeg"])]),
        ("Canada 10-year yield, bp", "bp", [("GoC 10-year", c["ca10"])]),
        ("Canadian dollar: CAD per USD, % (up = weaker CAD)", "%", [("USD/CAD", c["cad"])]),
    ]
    row_h, top, left, right = 26, 56, 132, 28
    heights = [34 + len(rows) * 3 * row_h + 30 for _, _, rows in panels]
    svg = Svg(760, top + sum(heights) + 10,
              "Fed vs Bank of Canada: effect on Canadian assets of a surprise that raises the "
              "bank's own 2-year yield by 10 bp")
    svg.text(16, 22, "A 10 bp surprise in each bank's own 2-year yield: effect on Canadian assets",
             14, INK, weight=600)
    keys = [("fed0", "Fed (US 2y instrumented by JK)", BLUE, True),
            ("boc0", "BoC, event LP", ORANGE, True),
            ("rs0", "BoC, Rigobon–Sack", ORANGE, False)]
    lx = 16.0
    for _, lab, col, filled in keys:
        fill = col if filled else "#ffffff"
        svg.raw(f'<circle cx="{lx + 5}" cy="40" r="5" fill="{fill}" stroke="{col}" '
                'stroke-width="2"/>')
        svg.text(lx + 15, 44, lab, 11.5, INK2)
        lx += 16 + 7.2 * len(lab) + 18
    y0 = top
    for (title, _units, rows), ph in zip(panels, heights, strict=True):
        vals = [v for _, r in rows for k, _, _, _ in keys
                for v in (r[k][1], r[k][2])] + [0.0]
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * 0.08
        ticks, d = _ticks(lo - pad, hi + pad, 6)
        lo, hi = min(lo - pad, ticks[0]), max(hi + pad, ticks[-1])

        def x(v: float, lo: float = lo, hi: float = hi) -> float:
            return left + (v - lo) / (hi - lo) * (svg.w - left - right)

        svg.text(16, y0 + 18, title, 12.5, INK, weight=600)
        plot_top, plot_bot = y0 + 30, y0 + 30 + len(rows) * 3 * row_h
        for t in ticks:
            svg.line(x(t), plot_top, x(t), plot_bot, GRID)
            svg.text(x(t), plot_bot + 16, _fmt(t, d), 10.5, MUTED, "middle")
        svg.line(x(0), plot_top, x(0), plot_bot, LINE, 1.5)
        yy = plot_top
        for name, r in rows:
            svg.text(16, yy + 1.5 * row_h + 4, name, 12, INK2)
            for i, (k, _lab, col, filled) in enumerate(keys):
                m, a, b = r[k][0], r[k][1], r[k][2]
                cy = yy + (i + 0.5) * row_h
                svg.line(x(a), cy, x(b), cy, col, 2)
                fill = col if filled else "#ffffff"
                svg.raw(f'<circle cx="{x(m):.1f}" cy="{cy:.1f}" r="5" fill="{fill}" '
                        f'stroke="{col}" stroke-width="2"/>')
            yy += 3 * row_h
            svg.line(left, yy, svg.w - right, yy, GRID)
        y0 += ph
    return svg.save(path)


def fed_irfs(path: Path) -> Path:
    """Small multiples: responses to a 10 bp Fed policy surprise, days 0-20, 68/90% bands."""
    cat = json.loads(CATALOG.read_text())
    names = {o["id"]: o for o in cat["outcomes"]}
    panels = [("US2Y", "US 2-year yield, bp"), ("CA2Y", "Canada 2-year yield, bp"),
              ("TSX.tr", "S&P/TSX, %"), ("USDCAD", "CAD per USD, % (up = weaker CAD)")]
    pw, ph, gap, top = 360, 190, 24, 40
    svg = Svg(2 * pw + 3 * gap, top + 2 * ph + 2 * gap,
              "Responses to a 10 bp Fed policy surprise over 20 trading days")
    svg.text(gap, 24, "Responses to a 10 bp Fed policy surprise (Jarociński–Karadi), "
             "trading days 0–20", 14, INK, weight=600)
    for i, (sid, title) in enumerate(panels):
        r = cat["irfs"]["fed_mp"][sid]
        coef, se = r["coef"], r["se"]
        lo90 = [m - Z90 * s for m, s in zip(coef, se, strict=True)]
        hi90 = [m + Z90 * s for m, s in zip(coef, se, strict=True)]
        lo68 = [m - 0.9944579 * s for m, s in zip(coef, se, strict=True)]
        hi68 = [m + 0.9944579 * s for m, s in zip(coef, se, strict=True)]
        ox = gap + (i % 2) * (pw + gap)
        oy = top + (i // 2) * (ph + gap)
        L, R, T, B = ox + 44, ox + pw - 8, oy + 26, oy + ph - 26
        vals = [*lo90, *hi90, 0.0]
        ticks, d = _ticks(min(vals), max(vals), 4)
        lo, hi = min(min(vals), ticks[0]), max(max(vals), ticks[-1])
        H = len(coef) - 1

        def X(h: float, L: float = L, R: float = R, H: int = H) -> float:
            return L + h / H * (R - L)

        def Y(v: float, T: float = T, B: float = B, lo: float = lo, hi: float = hi) -> float:
            return B - (v - lo) / (hi - lo) * (B - T)

        svg.text(ox, oy + 14, title, 12.5, INK, weight=600)
        for t in ticks:
            svg.line(L, Y(t), R, Y(t), GRID)
            svg.text(L - 6, Y(t) + 4, _fmt(t, d), 10.5, MUTED, "end")
        svg.line(L, Y(0), R, Y(0), LINE, 1.5)
        for h in range(0, H + 1, 5):
            svg.text(X(h), B + 15, str(h), 10.5, MUTED, "middle")

        def area(a: Sequence[float], b: Sequence[float]) -> str:
            pts = [f"{X(h):.1f},{Y(v):.1f}" for h, v in enumerate(b)]
            pts += [f"{X(h):.1f},{Y(v):.1f}" for h, v in reversed(list(enumerate(a)))]
            return " ".join(pts)

        svg.raw(f'<polygon points="{area(lo90, hi90)}" fill="{BLUE}" fill-opacity="0.14"/>')
        svg.raw(f'<polygon points="{area(lo68, hi68)}" fill="{BLUE}" fill-opacity="0.28"/>')
        line = " ".join(f"{X(h):.1f},{Y(v):.1f}" for h, v in enumerate(coef))
        svg.raw(f'<polyline points="{line}" fill="none" stroke="{BLUE}" stroke-width="2"/>')
        _ = names[sid]
    return svg.save(path)


def ml_benchmark(f: dict[str, Any], path: Path) -> Path:
    """Pooled scaled error per model on the held-out test set (lower is better)."""
    order = [("zero", "No move (B0)"), ("lp", "Linear LP (B1)"),
             ("state", "State-dependent LP (B2)"), ("ridge", "Ridge (B3)"),
             ("gbm", "Gradient boosting (B4)"), ("mlp", "Neural net (MLP)")]
    mae = f["ml"]["mae"]
    row_h, top, left, right = 30, 52, 190, 60
    svg = Svg(760, top + len(order) * row_h + 44,
              "Pooled scaled error of each model on the held-out test announcements")
    svg.text(16, 22, "Held-out test announcements: pooled scaled error (lower is better)", 14, INK,
             weight=600)
    svg.text(16, 40, "1.00 = the training-period error of predicting no move", 11.5, MUTED)
    vals = [float(v) for v in mae.values()]
    lo, hi = min(vals) - 0.02, max(vals) + 0.03
    ticks, d = _ticks(lo, hi, 5)
    lo, hi = min(lo, ticks[0]), max(hi, ticks[-1])

    def x(v: float) -> float:
        return left + (v - lo) / (hi - lo) * (svg.w - left - right)

    bot = top + len(order) * row_h
    for t in ticks:
        svg.line(x(t), top, x(t), bot, GRID)
        svg.text(x(t), bot + 16, f"{t:.{d}f}", 10.5, MUTED, "middle")
    svg.line(x(mae["zero"]), top, x(mae["zero"]), bot, MUTED, 1.2, "4 3")
    for i, (k, lab) in enumerate(order):
        cy = top + (i + 0.5) * row_h
        col = BLUE if k == "lp" else ORANGE if k == "mlp" else INK2
        svg.text(16, cy + 4, lab, 12, INK2)
        svg.line(left, cy, x(mae[k]), cy, GRID, 1)
        svg.raw(f'<circle cx="{x(mae[k]):.1f}" cy="{cy:.1f}" r="6" fill="{col}"/>')
        svg.text(x(mae[k]) + 10, cy + 4, f"{mae[k]:.3f}", 11, INK2)
    return svg.save(path)


def pvalues(f: dict[str, Any], path: Path) -> Path:
    """Histogram of the pre-registered regime tests' p-values against the uniform."""
    p = f["state"]["pvals"]
    bins = [sum(1 for v in p if b / 10 <= v < (b + 1) / 10 or (b == 9 and v == 1.0))
            for b in range(10)]
    expected = len(p) / 10
    w, h, left, top, bot = 760, 300, 56, 52, 250
    svg = Svg(w, h, "Distribution of the p-values of the pre-registered regime tests")
    svg.text(16, 22, f"{len(p)} pre-registered regime tests: p-values vs what chance alone gives",
             14, INK, weight=600)
    svg.text(16, 40, "Bars: tests per p-value bin. Dashed: expected count if no regime "
                     "mattered.", 11.5, MUTED)
    ymax = max(max(bins), expected) * 1.15
    ticks, _ = _ticks(0, ymax, 4)
    ymax = max(ymax, ticks[-1])
    R = w - 24

    def Y(v: float) -> float:
        return bot - v / ymax * (bot - top)

    for t in ticks:
        svg.line(left, Y(t), R, Y(t), GRID)
        svg.text(left - 6, Y(t) + 4, f"{t:.0f}", 10.5, MUTED, "end")
    bw = (R - left) / 10
    for b, n in enumerate(bins):
        x0 = left + b * bw + 2
        col = ORANGE if b == 0 else BLUE
        svg.raw(f'<rect x="{x0:.1f}" y="{Y(n):.1f}" width="{bw - 4:.1f}" '
                f'height="{bot - Y(n):.1f}" fill="{col}" rx="2"/>')
        svg.text(x0 + (bw - 4) / 2, Y(n) - 5, str(n), 11, INK2, "middle")
        svg.text(left + b * bw, bot + 16, f"{b / 10:.1f}", 10.5, MUTED, "middle")
    svg.text(R, bot + 16, "1.0", 10.5, MUTED, "middle")
    svg.line(left, Y(expected), R, Y(expected), INK2, 1.5, "6 4")
    svg.text((left + R) / 2, bot + 36, "p-value (wild cluster bootstrap)", 11.5, INK2, "middle")
    return svg.save(path)
