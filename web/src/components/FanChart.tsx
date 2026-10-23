import { scaleLinear } from "d3-scale";
import { area, line } from "d3-shape";
import { useCallback, useRef, useState } from "react";
import { signed } from "../format";
import { TableToggle } from "./ui";

export interface Series {
  name: string;
  color: string; // a CSS variable, e.g. "var(--s1)"
  mean: number[];
  lo90?: number[];
  hi90?: number[];
  lo68?: number[];
  hi68?: number[];
  dashed?: boolean;
  points?: { h: number; v: number }[]; // markers only (e.g. ML-implied responses)
}

/** Width of an element, tracked with a ResizeObserver; attach the returned callback ref. */
export function useWidth(initial = 640): [(el: HTMLElement | null) => void, number] {
  const [w, setW] = useState(initial);
  const ro = useRef<ResizeObserver | null>(null);
  const ref = useCallback(
    (el: HTMLElement | null) => {
      ro.current?.disconnect();
      if (!el) return;
      const obs = new ResizeObserver(() => setW(el.clientWidth || initial));
      obs.observe(el);
      ro.current = obs;
      setW(el.clientWidth || initial);
    },
    [initial],
  );
  return [ref, w];
}

function niceTicks(lo: number, hi: number, n = 5): number[] {
  const span = hi - lo || 1;
  const raw = span / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const e = raw / mag;
  const step = mag * (e >= 7.5 ? 10 : e >= 3.5 ? 5 : e >= 1.5 ? 2 : 1);
  const out: number[] = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

export function FanChart({
  series,
  horizons,
  units,
  xLabel = "Trading days after the announcement",
  height,
  summary,
  compact = false,
}: {
  series: Series[];
  horizons: number[];
  units: string;
  xLabel?: string;
  height?: number;
  summary: string; // read by screen readers
  compact?: boolean;
}) {
  const [ref, W] = useWidth();
  const [hover, setHover] = useState<number | null>(null);
  const H = height ?? (compact ? 150 : Math.round(Math.max(240, Math.min(360, W * 0.56))));
  const M = compact ? { l: 40, r: 8, t: 8, b: 22 } : { l: 56, r: 12, t: 12, b: 44 };
  const vals = series.flatMap((s) => [...s.mean, ...(s.lo90 ?? []), ...(s.hi90 ?? []), ...(s.points ?? []).map((p) => p.v), 0]).filter(Number.isFinite);
  let lo = Math.min(...vals);
  let hi = Math.max(...vals);
  const pad = (hi - lo || 1) * 0.06;
  lo -= pad;
  hi += pad;
  const yt = niceTicks(lo, hi, compact ? 3 : 5);
  lo = Math.min(lo, yt[0]);
  hi = Math.max(hi, yt[yt.length - 1]);
  const hmax = horizons[horizons.length - 1];
  const x = scaleLinear().domain([0, hmax]).range([M.l, W - M.r]);
  const y = scaleLinear().domain([lo, hi]).range([H - M.b, M.t]);
  const mk = (arr: number[]) => line<number>().x((_, i) => x(horizons[i])).y((v) => y(v))(arr) ?? "";
  const bandPath = (a: number[], b: number[]) =>
    area<number>()
      .x((_, i) => x(horizons[i]))
      .y0((_, i) => y(a[i]))
      .y1((v) => y(v))(b) ?? "";
  const xt = horizons.filter((h) => h % (W < 480 || compact ? 5 : 2) === 0);
  // One precision for the whole axis, taken from the tick step (so 0, +2, +4 or 0.0, +0.5, +1.0).
  const step = yt.length > 1 ? yt[1] - yt[0] : 1;
  const digits = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
  const fmt = (v: number) => (Math.abs(v) < step / 2 ? "0" : signed(v, digits));

  return (
    <figure className="chart" ref={ref} style={{ margin: 0 }}>
      <svg viewBox={`0 0 ${W} ${H}`} height={H} role="img" aria-label={summary}>
        <title>{summary}</title>
        {yt.map((v) => (
          <g key={v}>
            <line x1={M.l} x2={W - M.r} y1={y(v)} y2={y(v)} style={{ stroke: "var(--line)" }} />
            <text x={M.l - 6} y={y(v) + 4} textAnchor="end">
              {fmt(v)}
            </text>
          </g>
        ))}
        <line x1={M.l} x2={W - M.r} y1={y(0)} y2={y(0)} style={{ stroke: "var(--line-2)" }} strokeWidth={1.5} />
        {xt.map((h) => (
          <text key={h} x={x(h)} y={H - M.b + 16} textAnchor="middle">
            {h}
          </text>
        ))}
        {!compact && (
          <>
            <text className="axis-title" x={(M.l + W - M.r) / 2} y={H - 6} textAnchor="middle">
              {xLabel}
            </text>
            <text className="axis-title" transform={`translate(12 ${(M.t + H - M.b) / 2}) rotate(-90)`} textAnchor="middle">
              {units}
            </text>
          </>
        )}
        {series.map((s, i) =>
          s.lo90 && s.hi90 ? (
            <path key={`b90-${i}`} d={bandPath(s.lo90, s.hi90)} style={{ fill: i === 0 ? "var(--band90)" : "none", stroke: i === 0 ? "none" : s.color, strokeOpacity: 0.5 }} strokeDasharray={i === 0 ? undefined : "2 3"} />
          ) : null,
        )}
        {series.map((s, i) =>
          i === 0 && s.lo68 && s.hi68 ? <path key={`b68-${i}`} d={bandPath(s.lo68, s.hi68)} style={{ fill: "var(--band68)" }} /> : null,
        )}
        {series.map((s, i) =>
          s.mean.length ? (
            <path key={`l-${i}`} d={mk(s.mean)} style={{ fill: "none", stroke: s.color }} strokeWidth={2} strokeDasharray={s.dashed ? "6 4" : undefined} />
          ) : null,
        )}
        {series.map((s, i) =>
          (s.points ?? []).map((p) => (
            <rect key={`p-${i}-${p.h}`} x={x(p.h) - 5} y={y(p.v) - 5} width={10} height={10} transform={`rotate(45 ${x(p.h)} ${y(p.v)})`} style={{ fill: s.color, stroke: "var(--surface)" }} strokeWidth={2} />
          )),
        )}
        {hover !== null && <line x1={x(hover)} x2={x(hover)} y1={M.t} y2={H - M.b} style={{ stroke: "var(--ink-2)" }} strokeDasharray="2 3" />}
        <rect
          x={M.l}
          y={M.t}
          width={Math.max(0, W - M.l - M.r)}
          height={Math.max(0, H - M.t - M.b)}
          fill="transparent"
          onMouseMove={(e) => {
            const rect = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
            const h = Math.round(x.invert(e.clientX - rect.left));
            setHover(Math.max(0, Math.min(hmax, h)));
          }}
          onMouseLeave={() => setHover(null)}
        />
      </svg>
      {hover !== null && (
        <div className="tip" style={{ left: Math.min(x(hover) + 12, W - 200), top: M.t }}>
          <div className="t">h = {hover}</div>
          {series.map((s) => {
            const i = horizons.indexOf(hover);
            const v = s.mean[i];
            const p = s.points?.find((q) => q.h === hover);
            if (!Number.isFinite(v) && !p) return null;
            return (
              <div key={s.name}>
                <div className="row">
                  <span>{s.name}</span>
                  <span>{signed(Number.isFinite(v) ? v : p!.v)}</span>
                </div>
                {s.lo90 && Number.isFinite(v) && (
                  <div className="row muted">
                    <span>90% band</span>
                    <span>
                      {signed(s.lo90[i])} to {signed(s.hi90![i])}
                    </span>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
      {!compact && (
        <TableToggle summary="Show this chart as a table">
          <table>
            <thead>
              <tr>
                <th>h</th>
                {series.map((s) => (
                  <th key={s.name}>{s.name}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {horizons.map((h, i) => (
                <tr key={h}>
                  <td>{h}</td>
                  {series.map((s) => {
                    const p = s.points?.find((q) => q.h === h);
                    const v = s.mean[i];
                    return (
                      <td key={s.name} className="num">
                        {Number.isFinite(v) ? signed(v) : p ? signed(p.v) : ""}
                        {s.lo90 && Number.isFinite(v) ? ` [${signed(s.lo90[i])}, ${signed(s.hi90![i])}]` : ""}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </TableToggle>
      )}
    </figure>
  );
}
