import { geoCentroid, geoCircle, geoDistance, geoGraticule10, geoNaturalEarth1, geoOrthographic, geoPath } from "d3-geo";
import { useEffect, useMemo, useRef, useState } from "react";
import type { KeyboardEvent as ReactKeyboardEvent, PointerEvent as ReactPointerEvent } from "react";
import { feature } from "topojson-client";
import type { Topology } from "topojson-specification";
import type { FeatureCollection, Geometry } from "geojson";
import world from "world-atlas/countries-110m.json";
import { useAll } from "../data";
import { shockNoun, signed } from "../format";
import type { Catalog, Meta } from "../types";
import { Chip, Loading, Panel, Seg, Select, TableToggle } from "../components/ui";
import { FanChart, useWidth } from "../components/FanChart";
import { Z68, Z90 } from "../engine/engine";

type D = { catalog: Catalog; meta: Meta };
type Measure = "equity" | "fx";
type View = "globe" | "flat";
type LonLat = [number, number];

const allCountries = feature(world as unknown as Topology, (world as unknown as Topology).objects.countries) as unknown as FeatureCollection<Geometry, { name: string }>;
// Antarctica (ISO 010) has no markets and takes a fifth of the flat map's height; leave it out there.
const flatCountries: typeof allCountries = { ...allCountries, features: allCountries.features.filter((f) => String((f as { id?: string }).id) !== "010") };
const idOf = (f: unknown) => String((f as { id?: string | number }).id ?? "");

// Where each kind of shock starts. The two news indices measure coverage worldwide, so they have no single source.
const ORIGINS: Record<string, { name: string; at: LonLat } | undefined> = {
  fomc: { name: "Washington", at: [-77.04, 38.9] },
  boc: { name: "Ottawa", at: [-75.7, 45.42] },
  opec: { name: "Vienna, where OPEC meets", at: [16.37, 48.21] },
};
// The five kinds of shock, one button each; the full list (three readings of a Fed surprise) is in the menu beside them.
const QUICK = [
  { value: "fed_mp", text: "Fed" },
  { value: "boc", text: "Bank of Canada" },
  { value: "opec", text: "Oil" },
  { value: "gpr", text: "Geopolitics" },
  { value: "tpu", text: "Trade policy" },
];
// A marker sits where the country's markets are, not at the middle of every territory it owns.
const MARKER: Record<string, LonLat> = { USA: [-96, 39], FRA: [2.4, 46.6], NOR: [9.5, 61.2], GBR: [-1.8, 53.6], CAN: [-98, 56] };
const LAST = 20; // the last trading day in the estimated responses
const ease = (k: number) => 1 - Math.pow(1 - k, 3);

export default function MapView() {
  const d = useAll<D>(["catalog", "meta"]);
  const [shock, setShock] = useState("fed_mp");
  const [h, setH] = useState(0); // trading days after the announcement; fractional only while the days are playing
  const [measure, setMeasure] = useState<Measure>("equity");
  const [view, setView] = useState<View>("globe");
  const [picked, setPicked] = useState("CAN");
  const [rot, setRot] = useState<LonLat>([39, -24]); // the globe's rotation: minus the longitude and latitude facing the reader
  const [wave, setWave] = useState(0); // how far the shock's front has travelled, 0 to 1 (an illustration, not a measurement)
  const [playing, setPlaying] = useState(false);
  const [hover, setHover] = useState<{ iso3: string; x: number; y: number } | null>(null);
  const [ref, width] = useWidth();
  const run = useRef({ play: 0, wave: 0, turn: 0 });
  const drag = useRef<{ x: number; y: number; moved: boolean } | null>(null);
  const rotNow = useRef(rot);
  rotNow.current = rot;
  const still = useMemo(() => typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches, []);

  /** Run `step` with k from 0 to 1 over `ms`; a newer run on the same channel replaces it. */
  const animate = (channel: "play" | "wave" | "turn", ms: number, step: (k: number) => void, done?: () => void) => {
    const token = ++run.current[channel];
    const t0 = performance.now();
    const frame = (now: number) => {
      if (run.current[channel] !== token) return;
      const k = Math.min(1, (now - t0) / ms);
      step(k);
      if (k < 1) requestAnimationFrame(frame);
      else done?.();
    };
    requestAnimationFrame(frame);
  };
  const turnTo = (target: LonLat) => {
    if (still) return setRot(target);
    const from = rotNow.current;
    const dl = ((((target[0] - from[0]) % 360) + 540) % 360) - 180; // the short way round
    animate("turn", 900, (k) => setRot([from[0] + dl * ease(k), from[1] + (target[1] - from[1]) * ease(k)]));
  };
  const sendWave = () => {
    if (still) return setWave(1);
    animate("wave", 2600, (k) => setWave(ease(k)));
  };
  const family = d && typeof d !== "string" ? d.catalog.shocks.find((x) => x.id === shock)?.family ?? "" : "";
  const origin = ORIGINS[family];

  // A new shock: turn the globe to face where it starts, and send the front out again.
  useEffect(() => {
    if (!family) return;
    turnTo([-((origin ? origin.at[0] : -20) + 38), -24]);
    sendWave();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [shock, family]);
  useEffect(() => {
    const live = run.current;
    return () => {
      live.play += 1;
      live.wave += 1;
      live.turn += 1;
    };
  }, []);

  if (!d || typeof d === "string") return <Loading error={typeof d === "string" ? d : null} />;
  const { catalog, meta } = d;
  const s = catalog.shocks.find((x) => x.id === shock)!;
  const sizeText =
    s.unit === "sd"
      ? `a one standard deviation ${shockNoun(s.name)}`
      : s.family === "opec"
        ? `${shockNoun(s.name)} that moves oil futures by +${s.per}%`
        : `a +${s.per} ${s.unit} ${shockNoun(s.name)}`;
  const day = Math.round(h);

  // Response of each country's asset at day `at` (between two days while playing). FX is the local currency's move vs USD (+ = appreciates).
  const value = (iso3: string, at: number) => {
    const c = meta.countries[iso3];
    const id = measure === "equity" ? c.equity : c.fx;
    const r = catalog.irfs[shock][id];
    if (!r) return null;
    const sign = measure === "fx" ? c.fx_sign : 1;
    const i0 = Math.min(LAST, Math.floor(at)), i1 = Math.min(LAST, i0 + 1), f = at - i0;
    return { m: sign * (r.coef[i0] * (1 - f) + r.coef[i1] * f), se: r.se[i0] * (1 - f) + r.se[i1] * f, n: r.n[i0], id };
  };
  const rows = Object.entries(meta.countries).map(([iso3, c]) => ({ iso3, c, v: value(iso3, h) }));
  // One colour scale for all 21 days, so a country's colour changes only because its response does.
  const maxAbs = Math.max(
    1e-6,
    ...Object.keys(meta.countries).flatMap((iso3) => catalog.horizons.map((hh) => Math.abs(value(iso3, hh)?.m ?? 0))),
  );
  const tone = (v: number) => (v >= 0 ? "var(--pos)" : "var(--neg)");
  const fillFor = (v: number) => ({ fill: tone(v), fillOpacity: 0.14 + 0.86 * Math.min(1, Math.abs(v) / maxAbs) });
  const byN3 = new Map(rows.map((r) => [r.c.iso_n3, r]));
  const sel = rows.find((r) => r.iso3 === picked)!;
  const irf = catalog.irfs[shock][measure === "equity" ? sel.c.equity : sel.c.fx];
  const sign = measure === "fx" ? sel.c.fx_sign : 1;
  const withData = rows.filter((r) => r.v);
  const ranked = [...withData].sort((a, b) => b.v!.m - a.v!.m);
  const mapLabel = `${view === "globe" ? "Globe" : "World map"} of ${measure} responses to ${s.name}, ${day} trading day${day === 1 ? "" : "s"} after the announcement; ${withData.length} countries with data. The list beside it and the table below give every value.`;
  const series = irf
    ? [
        {
          name: `${sel.c.name}, ${measure === "equity" ? "equity (USD)" : "currency vs USD"}`,
          color: "var(--s1)",
          mean: irf.coef.map((v) => sign * v),
          lo90: irf.coef.map((v, i) => sign * v - Z90 * irf.se[i]),
          hi90: irf.coef.map((v, i) => sign * v + Z90 * irf.se[i]),
          lo68: irf.coef.map((v, i) => sign * v - Z68 * irf.se[i]),
          hi68: irf.coef.map((v, i) => sign * v + Z68 * irf.se[i]),
        },
      ]
    : [];

  const play = () => {
    if (playing) {
      run.current.play += 1;
      setPlaying(false);
      setH(Math.round(h));
      return;
    }
    setPlaying(true);
    sendWave();
    const from = h >= LAST ? 0 : Math.round(h);
    if (still) {
      // No easing for readers who asked for less motion: one day at a time.
      animate("play", 420 * (LAST - from), (k) => setH(Math.round(from + (LAST - from) * k)), () => setPlaying(false));
      return;
    }
    animate("play", 340 * (LAST - from), (k) => setH(from + (LAST - from) * k), () => {
      setH(LAST);
      setPlaying(false);
    });
  };
  const pick = (iso3: string) => {
    setPicked(iso3);
    const at = markerAt(iso3);
    if (view === "globe" && at && geoDistance(at, [-rot[0], -rot[1]]) > 1.15) turnTo([-at[0], Math.max(-50, Math.min(50, -at[1] * 0.6))]);
  };

  // ---- the globe
  const S = Math.max(280, Math.min(width, 640));
  const R = S / 2 - 16;
  const globe = geoOrthographic().scale(R).translate([S / 2, S / 2]).rotate([rot[0], rot[1]]).clipAngle(90);
  const gpath = geoPath(globe);
  const facing: LonLat = [-rot[0], -rot[1]];
  const seen = (at: LonLat) => geoDistance(at, facing) < Math.PI / 2 - 0.02;
  const centroid = new Map(allCountries.features.map((f) => [idOf(f), geoCentroid(f) as LonLat]));
  const markerAt = (iso3: string): LonLat | undefined => MARKER[iso3] ?? centroid.get(meta.countries[iso3].iso_n3);
  const front = origin && wave > 0 && wave < 1 ? { type: "LineString" as const, coordinates: geoCircle().center(origin.at).radius(wave * 178)().coordinates[0] } : null;

  const onDown = (e: ReactPointerEvent<SVGSVGElement>) => {
    drag.current = { x: e.clientX, y: e.clientY, moved: false };
    e.currentTarget.setPointerCapture(e.pointerId);
    run.current.turn += 1;
  };
  const onMove = (e: ReactPointerEvent<SVGSVGElement>) => {
    const g = drag.current;
    if (!g) return;
    const dx = e.clientX - g.x, dy = e.clientY - g.y;
    if (!g.moved && Math.hypot(dx, dy) < 4) return;
    g.moved = true;
    g.x = e.clientX;
    g.y = e.clientY;
    const k = 58 / R; // a drag across the radius turns the globe about a sixth of the way round
    setRot(([l, p]) => [l + dx * k, Math.max(-70, Math.min(70, p - dy * k))]);
    setHover(null);
  };
  const onUp = () => {
    const g = drag.current;
    setTimeout(() => {
      if (drag.current === g) drag.current = null;
    }, 0); // after the click that ends a drag has been ignored
  };
  const onKey = (e: ReactKeyboardEvent<HTMLDivElement>) => {
    const step: Record<string, LonLat> = { ArrowLeft: [14, 0], ArrowRight: [-14, 0], ArrowUp: [0, -10], ArrowDown: [0, 10] };
    const by = step[e.key];
    if (!by) return;
    e.preventDefault();
    run.current.turn += 1;
    setRot(([l, p]) => [l + by[0], Math.max(-70, Math.min(70, p + by[1]))]);
  };
  const clickCountry = (iso3: string | undefined) => {
    if (drag.current?.moved) return;
    if (iso3) pick(iso3);
  };
  const tip = hover ? rows.find((r) => r.iso3 === hover.iso3) : null;
  /** Show a country's value beside the pointer (positions are within the map's own box). */
  const point = (iso3: string, e: ReactPointerEvent<SVGElement>) => {
    if (drag.current?.moved) return;
    const box = e.currentTarget.closest(".globe-stage")?.getBoundingClientRect();
    if (box) setHover({ iso3, x: e.clientX - box.left, y: e.clientY - box.top });
  };

  // ---- the flat map
  const H = Math.round(width * 0.44);
  const flat = geoPath(geoNaturalEarth1().fitSize([width, H], flatCountries));

  return (
    <Panel eyebrow="View 1 · World Reaction Map" title={`How markets react: ${s.name}`} chip={<Chip label={s.label} />} id="map-h">
      <p className="lede">
        Response to {sizeText}, {day === 0 ? "on the day of the announcement" : `${day} trading day${day === 1 ? "" : "s"} after the announcement`}. Equity is each country's US-listed iShares
        ETF (in US dollars, so it mixes stock and currency moves); currency is the local currency against the US dollar, positive when it
        strengthens. Press play to watch the 20 days after the news, turn the globe, and click a country for its full response with uncertainty bands.
      </p>
      <div className="controls">
        <Seg label="Shock" value={QUICK.some((q) => q.value === shock) ? shock : ""} onChange={(v) => v && setShock(v)} options={QUICK} />
        <Select id="map-shock" label="All shocks" value={shock} onChange={setShock}>
          {catalog.shocks.map((x) => (
            <option key={x.id} value={x.id}>
              {x.name}
            </option>
          ))}
        </Select>
        <Seg label="Measure" value={measure} onChange={setMeasure} options={[{ value: "equity", text: "Equity" }, { value: "fx", text: "Currency" }]} />
        <Seg label="View" value={view} onChange={setView} options={[{ value: "globe", text: "Globe" }, { value: "flat", text: "Flat map" }]} />
      </div>
      <div className="controls days">
        <button type="button" className="btn play" onClick={play} aria-pressed={playing}>
          {playing ? "Pause" : h >= LAST ? "Replay the 20 days" : "Play the 20 days"}
        </button>
        <div className="ctl map-horizon">
          <label htmlFor="map-day">Horizon: day {day}</label>
          <input
            id="map-day"
            type="range"
            min={0}
            max={LAST}
            value={day}
            onChange={(e) => {
              run.current.play += 1;
              setPlaying(false);
              setH(Number(e.target.value));
            }}
          />
        </div>
      </div>

      <div className={`world ${view}${playing ? " is-playing" : ""}`}>
        <div
          className="chart globe-stage"
          ref={ref}
          tabIndex={view === "globe" ? 0 : undefined}
          role={view === "globe" ? "group" : undefined}
          aria-label={view === "globe" ? "Globe. Drag it, or use the arrow keys, to turn it." : undefined}
          onKeyDown={view === "globe" ? onKey : undefined}
        >
          {view === "globe" ? (
            <svg
              viewBox={`0 0 ${S} ${S}`}
              width={S}
              height={S}
              role="img"
              aria-label={mapLabel}
              className="globe"
              onPointerDown={onDown}
              onPointerMove={onMove}
              onPointerUp={onUp}
              onPointerCancel={onUp}
              onPointerLeave={() => setHover(null)}
            >
              <title>{mapLabel}</title>
              <defs>
                <radialGradient id="g-sea" cx="38%" cy="32%" r="78%">
                  <stop offset="0%" className="sea-0" />
                  <stop offset="100%" className="sea-1" />
                </radialGradient>
                <radialGradient id="g-shade" cx="36%" cy="30%" r="80%">
                  <stop offset="55%" stopColor="#000" stopOpacity="0" />
                  <stop offset="100%" stopColor="#000" stopOpacity="0.5" />
                </radialGradient>
                <filter id="g-glow" x="-20%" y="-20%" width="140%" height="140%">
                  <feGaussianBlur stdDeviation="9" />
                </filter>
              </defs>
              <circle cx={S / 2} cy={S / 2} r={R + 3} className="air" filter="url(#g-glow)" />
              <circle cx={S / 2} cy={S / 2} r={R} fill="url(#g-sea)" />
              <path d={gpath(geoGraticule10()) ?? ""} className="graticule" />
              {allCountries.features.map((f, i) => {
                const r = byN3.get(idOf(f));
                return (
                  <path
                    key={i}
                    d={gpath(f) ?? ""}
                    className={`land${r?.v ? " has-data" : ""}${r?.iso3 === picked ? " is-picked" : ""}`}
                    style={r?.v ? fillFor(r.v.m) : undefined}
                    onClick={() => clickCountry(r?.v ? r.iso3 : undefined)}
                    onPointerMove={(e) => r?.v && point(r.iso3, e)}
                    onPointerLeave={() => setHover(null)}
                  />
                );
              })}
              <circle cx={S / 2} cy={S / 2} r={R} fill="url(#g-shade)" pointerEvents="none" />
              {origin &&
                withData.map((r) => {
                  const at = markerAt(r.iso3);
                  if (!at) return null;
                  const size = Math.min(1, Math.abs(r.v!.m) / maxAbs);
                  return <path key={r.iso3} d={gpath({ type: "LineString", coordinates: [origin.at, at] }) ?? ""} className="route" style={{ stroke: tone(r.v!.m), strokeOpacity: 0.18 + 0.6 * size, strokeWidth: 0.8 + 1.6 * size }} />;
                })}
              {front && <path d={gpath(front) ?? ""} className="front" style={{ strokeOpacity: 0.85 * (1 - wave) }} />}
              {withData.map((r) => {
                const at = markerAt(r.iso3);
                if (!at || !seen(at)) return null;
                const [x, y] = globe(at) ?? [0, 0];
                const size = Math.min(1, Math.abs(r.v!.m) / maxAbs);
                return (
                  <g key={r.iso3} className={`mark${r.iso3 === picked ? " is-picked" : ""}`} transform={`translate(${x.toFixed(1)} ${y.toFixed(1)})`} onClick={() => clickCountry(r.iso3)} onPointerMove={(e) => point(r.iso3, e)} onPointerLeave={() => setHover(null)}>
                    <circle r={4 + 13 * size} style={{ fill: tone(r.v!.m) }} className="halo" />
                    <circle r={2.6 + 2.4 * size} style={{ fill: tone(r.v!.m) }} className="dot" />
                    {r.iso3 === picked && (
                      <text x={12 + 8 * size} y={4} className="mark-name">
                        {r.c.name} {signed(r.v!.m)}%
                      </text>
                    )}
                  </g>
                );
              })}
              {origin && seen(origin.at) && (
                <g className="source" transform={`translate(${(globe(origin.at) ?? [0, 0]).map((v) => v.toFixed(1)).join(" ")})`}>
                  <circle r={5} className="pulse" />
                  <circle r={5} className="pulse late" />
                  <circle r={4.5} className="core" />
                  <text x={10} y={-9} className="mark-name">
                    {origin.name}
                  </text>
                </g>
              )}
            </svg>
          ) : (
            <svg viewBox={`0 0 ${width} ${H}`} height={H} role="img" aria-label={mapLabel} className="flatmap">
              <title>{mapLabel}</title>
              <defs>
                <pattern id="nodata" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                  <rect width="4" height="4" style={{ fill: "var(--nodata)" }} />
                  <line x1="0" y1="0" x2="0" y2="4" style={{ stroke: "var(--line-2)" }} strokeWidth={1} />
                </pattern>
              </defs>
              {flatCountries.features.map((f, i) => {
                const r = byN3.get(idOf(f));
                const style = r?.v ? fillFor(r.v.m) : { fill: "url(#nodata)", fillOpacity: 1 };
                return (
                  <path
                    key={i}
                    d={flat(f) ?? ""}
                    className="flat-land"
                    style={{ ...style, stroke: "var(--ink-2)", strokeOpacity: 0.25, cursor: r?.v ? "pointer" : "default" }}
                    strokeWidth={r?.iso3 === picked ? 2 : 0.5}
                    onClick={() => r?.v && pick(r.iso3)}
                  >
                    <title>{r?.v ? `${r.c.name}: ${signed(r.v.m)}%` : f.properties.name}</title>
                  </path>
                );
              })}
            </svg>
          )}
          {view === "globe" && tip?.v && hover && (
            <div className="tip" style={{ left: hover.x, top: hover.y }} role="presentation">
              <b>{tip.c.name}</b>
              <span>
                {signed(tip.v.m)}% <i>90%: {signed(tip.v.m - Z90 * tip.v.se)} to {signed(tip.v.m + Z90 * tip.v.se)}</i>
              </span>
            </div>
          )}
        </div>

        <div className="movers" aria-label={`Every country's ${measure} response on day ${day}, largest rise first`}>
          <h3>
            Day {day} <span className="muted">· largest rise to largest fall</span>
          </h3>
          <ol>
            {ranked.map((r) => {
              const w = (50 * Math.min(1, Math.abs(r.v!.m) / maxAbs)).toFixed(2);
              return (
                <li key={r.iso3} className={r.iso3 === picked ? "sel" : ""}>
                  <button type="button" className="rowbtn" onClick={() => pick(r.iso3)}>
                    {r.c.name}
                  </button>
                  <span className="bar" aria-hidden="true">
                    <i style={r.v!.m >= 0 ? { left: "50%", width: `${w}%`, background: "var(--pos)" } : { right: "50%", width: `${w}%`, background: "var(--neg)" }} />
                  </span>
                  <span className="num">{signed(r.v!.m)}%</span>
                </li>
              );
            })}
          </ol>
        </div>
      </div>

      <div className="legend" aria-hidden="true">
        <span>
          {signed(-maxAbs)}% <span className="ramp" /> {signed(maxAbs)}%
          <span className="muted"> ({measure === "equity" ? "falls · rises" : "local currency weakens · strengthens"}; strongest colour = largest move in the 20 days)</span>
        </span>
        <span>
          <i className="hatch" />
          {measure === "equity" ? "no US-listed country ETF in the data" : "no currency series in the data"}
        </span>
        {view === "globe" && (
          <span className="muted">
            {origin
              ? `Lines run from ${origin.name.split(",")[0]} to each market; their colour and weight are the measured response. The ring that spreads when a shock is chosen is an illustration, not a measurement.`
              : "This shock is a jump in a worldwide news index, so it has no single source on the globe."}
          </span>
        )}
      </div>
      <div className="side">
        <div>
          <h3>{sel.c.name}</h3>
          {series.length ? (
            <FanChart series={series} horizons={catalog.horizons} units="% (100 × log change)" summary={`Response of ${sel.c.name} over 20 trading days.`} />
          ) : (
            <p className="muted">No data for this country.</p>
          )}
        </div>
        <div>
          <TableToggle summary="All countries as a table">
            <table>
              <thead>
                <tr>
                  <th>Country</th>
                  <th>Response, %</th>
                  <th>90% band</th>
                </tr>
              </thead>
              <tbody>
                {ranked.map((r) => (
                  <tr key={r.iso3} className={r.iso3 === picked ? "sel" : ""}>
                    <td>
                      <button type="button" className="rowbtn" onClick={() => pick(r.iso3)}>
                        {r.c.name}
                      </button>
                    </td>
                    <td className="num">{signed(r.v!.m)}</td>
                    <td className="num">
                      {signed(r.v!.m - Z90 * r.v!.se)} to {signed(r.v!.m + Z90 * r.v!.se)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </TableToggle>
        </div>
      </div>
    </Panel>
  );
}
