import { geoNaturalEarth1, geoPath } from "d3-geo";
import { useState } from "react";
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

const allCountries = feature(world as unknown as Topology, (world as unknown as Topology).objects.countries) as unknown as FeatureCollection<Geometry, { name: string }>;
// Antarctica (ISO 010) has no markets and takes a fifth of the height; leave it out.
const countries: typeof allCountries = { ...allCountries, features: allCountries.features.filter((f) => String((f as { id?: string }).id) !== "010") };

export default function MapView() {
  const d = useAll<D>(["catalog", "meta"]);
  const [shock, setShock] = useState("fed_mp");
  const [h, setH] = useState(0);
  const [measure, setMeasure] = useState<Measure>("equity");
  const [picked, setPicked] = useState("CAN");
  const [ref, width] = useWidth();
  if (!d || typeof d === "string") return <Loading error={typeof d === "string" ? d : null} />;
  const { catalog, meta } = d;
  const s = catalog.shocks.find((x) => x.id === shock)!;
  const sizeText =
    s.unit === "sd"
      ? `a one standard deviation ${shockNoun(s.name)}`
      : s.family === "opec"
        ? `${shockNoun(s.name)} that moves oil futures by +${s.per}%`
        : `a +${s.per} ${s.unit} ${shockNoun(s.name)}`;

  // Response of each country's asset; FX is expressed as the local currency's move vs USD (+ = appreciates).
  const value = (iso3: string) => {
    const c = meta.countries[iso3];
    const id = measure === "equity" ? c.equity : c.fx;
    const r = catalog.irfs[shock][id];
    if (!r) return null;
    const sign = measure === "fx" ? c.fx_sign : 1;
    return { m: sign * r.coef[h], se: r.se[h], n: r.n[h], id };
  };
  const rows = Object.entries(meta.countries).map(([iso3, c]) => ({ iso3, c, v: value(iso3) }));
  const maxAbs = Math.max(...rows.map((r) => (r.v ? Math.abs(r.v.m) : 0)), 1e-6);
  // Diverging fill that follows the theme: hue gives the direction, opacity the size.
  const fillFor = (v: number) => ({ fill: v >= 0 ? "var(--pos)" : "var(--neg)", fillOpacity: 0.12 + 0.88 * (Math.abs(v) / maxAbs) });
  const byN3 = new Map(rows.map((r) => [r.c.iso_n3, r]));
  const H = Math.round(width * 0.44);
  const projection = geoNaturalEarth1().fitSize([width, H], countries);
  const path = geoPath(projection);
  const sel = rows.find((r) => r.iso3 === picked)!;
  const irf = catalog.irfs[shock][measure === "equity" ? sel.c.equity : sel.c.fx];
  const sign = measure === "fx" ? sel.c.fx_sign : 1;
  const mapLabel = `World map of ${measure} responses to ${s.name}; ${rows.filter((r) => r.v).length} countries with data. The table below lists every value.`;
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

  return (
    <Panel eyebrow="View 1 · World Reaction Map" title={`How markets react: ${s.name}`} chip={<Chip label={s.label} />} id="map-h">
      <p className="lede">
        Response to {sizeText}, {h === 0 ? "on the day of the announcement" : `${h} trading day${h === 1 ? "" : "s"} after the announcement`}. Equity is each country's US-listed iShares
        ETF (in US dollars, so it mixes stock and currency moves); currency is the local currency against the US dollar, positive when it
        strengthens. Click a country, or use the table, for its full response with uncertainty bands.
      </p>
      <div className="controls">
        <Select id="map-shock" label="Shock" value={shock} onChange={setShock}>
          {catalog.shocks.map((x) => (
            <option key={x.id} value={x.id}>
              {x.name}
            </option>
          ))}
        </Select>
        <Seg label="Measure" value={measure} onChange={setMeasure} options={[{ value: "equity", text: "Equity" }, { value: "fx", text: "Currency" }]} />
        <div className="ctl" style={{ minWidth: 220 }}>
          <label htmlFor="map-h">Horizon: day {h}</label>
          <input id="map-h" type="range" min={0} max={20} value={h} onChange={(e) => setH(Number(e.target.value))} />
        </div>
      </div>
      <div className="chart" ref={ref} style={{ background: "var(--surface-2)" }}>
        <svg viewBox={`0 0 ${width} ${H}`} height={H} role="img" aria-label={mapLabel}>
          <title>{mapLabel}</title>
          <defs>
            <pattern id="nodata" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <rect width="4" height="4" style={{ fill: "var(--nodata)" }} />
              <line x1="0" y1="0" x2="0" y2="4" style={{ stroke: "var(--line-2)" }} strokeWidth={1} />
            </pattern>
          </defs>
          {countries.features.map((f, i) => {
            const r = byN3.get(String((f as { id?: string | number }).id ?? ""));
            const style = r?.v ? fillFor(r.v.m) : { fill: "url(#nodata)", fillOpacity: 1 };
            return (
              <path
                key={i}
                d={path(f) ?? ""}
                style={{ ...style, stroke: "var(--ink-2)", strokeOpacity: 0.25, cursor: r?.v ? "pointer" : "default" }}
                strokeWidth={r?.iso3 === picked ? 2 : 0.5}
                onClick={() => r?.v && setPicked(r.iso3)}
              >
                <title>{r?.v ? `${r.c.name}: ${signed(r.v.m)}%` : f.properties.name}</title>
              </path>
            );
          })}
        </svg>
      </div>
      <div className="legend" aria-hidden="true">
        <span>
          {signed(-maxAbs)}% <span className="ramp" /> {signed(maxAbs)}%
          <span className="muted"> ({measure === "equity" ? "falls · rises" : "local currency weakens · strengthens"}; strongest colour = largest move in the map)</span>
        </span>
        <span>
          <i className="hatch" />
          {measure === "equity" ? "no US-listed country ETF in the data" : "no currency series in the data"}
        </span>
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
                {rows
                  .filter((r) => r.v)
                  .sort((a, b) => b.v!.m - a.v!.m)
                  .map((r) => (
                    <tr key={r.iso3} className={r.iso3 === picked ? "sel" : ""}>
                      <td>
                        <button type="button" className="rowbtn" onClick={() => setPicked(r.iso3)}>
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
