import { useMemo, useState } from "react";
import { useAll } from "../data";
import { signed } from "../format";
import { useScenario } from "../scenario";
import type { Catalog, Shock } from "../types";
import { Chip, Loading, Panel } from "../components/ui";
import { FanChart } from "../components/FanChart";
import { compose, inSd } from "../engine/engine";

type D = { catalog: Catalog };
const CANADA = ["CA2Y", "CA10Y", "USDCAD", "TSX.tr", "CA_BANKS_EW6", "XEG.tr"];
const HINT: Record<string, string> = { USDCAD: "Up = a weaker Canadian dollar", CA2Y: "Up = higher yield", CA10Y: "Up = higher yield" };
const PRESETS: { name: string; sizes: Record<string, number> }[] = [
  { name: "Hawkish Fed (+10 bp)", sizes: { fed_mp: 10 } },
  { name: "Fed hike + oil supply cut", sizes: { fed_mp: 10, opec: 3 } },
  { name: "Surprise BoC cut (−10 bp)", sizes: { boc: -10 } },
  { name: "Trade-policy shock (TPU +3 s.d.)", sizes: { tpu: 3 } },
  { name: "Geopolitical flare-up + oil", sizes: { gpr: 3, opec: 2 } },
];

function percentile(s: Shock, v: number): string {
  const d = s.distribution;
  if (v <= d.p01) return "below the 1st percentile";
  if (v >= d.p99) return "above the 99th percentile";
  if (v <= d.p05) return "1st–5th percentile";
  if (v >= d.p95) return "95th–99th percentile";
  return "within the central 90%";
}

function pathOf(irfs: Catalog["irfs"], shocks: Shock[], sizes: Record<string, number>, id: string) {
  // The combined response over every horizon (for the Canada dashboard charts).
  const H = irfs[shocks[0].id][id].coef.length;
  return Array.from({ length: H }, (_, h) => compose({ shocks, outcomes: [{ id } as never], irfs }, sizes, h).outcomes[0]);
}

export default function ScenarioView() {
  const d = useAll<D>(["catalog"]);
  const { scenario, setScenario } = useScenario();
  const [showAll, setShowAll] = useState(false);
  const result = useMemo(() => (d && typeof d !== "string" ? compose(d.catalog, scenario.sizes, scenario.horizon) : null), [d, scenario]);
  if (!d || typeof d === "string" || !result) return <Loading error={typeof d === "string" ? d : null} />;
  const { catalog } = d;
  const outcomes = new Map(catalog.outcomes.map((o) => [o.id, o]));
  const set = (id: string, v: number) => setScenario({ ...scenario, sizes: { ...scenario.sizes, [id]: v } });
  const ranked = [...result.outcomes].filter((r) => r.se > 0).sort((a, b) => Math.abs(b.mean / b.se) - Math.abs(a.mean / a.se));
  const shown = showAll ? ranked : ranked.slice(0, 12);

  return (
    <>
      <Panel eyebrow="View 3 · Scenario Composer" title="Build a scenario" chip={<Chip label="hypothetical_scenario" />} id="sc-h">
        <p className="lede">
          Size each shock in its own units; the readout shows where that size sits in history. Responses are added, with 90% bands from the
          estimates' uncertainty. Evidence grade of this scenario: <Chip label={result.evidence} />
        </p>
        <div className="controls" role="group" aria-label="Presets">
          {PRESETS.map((p) => (
            <button key={p.name} type="button" className="btn ghost" onClick={() => setScenario({ ...scenario, sizes: p.sizes })}>
              {p.name}
            </button>
          ))}
          <button type="button" className="btn ghost" onClick={() => setScenario({ ...scenario, sizes: {} })}>
            Clear
          </button>
        </div>
        <div>
          {catalog.shocks.map((s) => {
            const v = scenario.sizes[s.id] ?? 0;
            // Symmetric around zero (the tick marks "off"), wide enough for every observed size.
            const hi = Math.ceil(Math.max(Math.abs(s.distribution.min), Math.abs(s.distribution.max), 3 * s.distribution.sd) * 1.25);
            const lo = -hi;
            const step = s.unit === "%" || s.unit === "sd" ? 0.1 : 1;
            return (
              <div className="shockrow" key={s.id}>
                <div>
                  <label htmlFor={`sz-${s.id}`}>
                    <strong>{s.name}</strong>
                  </label>
                  <div className="muted" style={{ fontSize: 12.5 }}>
                    {v === 0 ? "off" : `${signed(inSd(catalog, s.id, v), 1)} s.d. · ${percentile(s, v)}`} · <Chip label={s.label} />
                  </div>
                </div>
                <div className="slider">
                  <input id={`sz-${s.id}-r`} aria-label={`${s.name} size slider`} type="range" min={lo} max={hi} step={step} value={v} onChange={(e) => set(s.id, Number(e.target.value))} />
                </div>
                <div>
                  <input id={`sz-${s.id}`} type="number" step={step} value={v} onChange={(e) => set(s.id, Number(e.target.value) || 0)} style={{ width: "100%" }} aria-describedby={`unit-${s.id}`} />
                  <span id={`unit-${s.id}`} className="muted" style={{ fontSize: 12 }}>
                    {s.unit}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
        <div className="ctl" style={{ maxWidth: 360 }}>
          <label htmlFor="sc-h-range">Horizon: {scenario.horizon} trading days after the announcement</label>
          <input id="sc-h-range" type="range" min={0} max={20} value={scenario.horizon} onChange={(e) => setScenario({ ...scenario, horizon: Number(e.target.value) })} />
        </div>
        {result.warnings.length > 0 && (
          <div className="warn" role="note" aria-live="polite">
            {result.warnings.map((w) => (
              <span key={w}>{w}</span>
            ))}
          </div>
        )}
        <div>
          <a className="btn" href="#/brief">
            Open the strategist brief for this scenario
          </a>
        </div>
      </Panel>

      {result.outcomes.length > 0 && (
        <>
          <Panel eyebrow="Canada dashboard" title={`Canadian assets, ${scenario.horizon} trading days after`} id="ca-h">
            <div className="cards">
              {CANADA.map((id) => {
                const o = outcomes.get(id)!;
                const r = result.outcomes.find((x) => x.id === id)!;
                const path = pathOf(catalog.irfs, catalog.shocks, scenario.sizes, id);
                return (
                  <div className="card" key={id}>
                    <div className="eyebrow">{o.name}</div>
                    {HINT[id] && <div className="hint">{HINT[id]}</div>}
                    <div className="stat">
                      <span className="big">
                        {signed(r.mean)} <span style={{ fontSize: 14 }}>{o.units}</span>
                      </span>
                      <span className="muted num" style={{ fontSize: 12.5 }}>
                        90%: {signed(r.lo90)} to {signed(r.hi90)}
                      </span>
                    </div>
                    <FanChart compact series={[{ name: o.name, color: "var(--s1)", mean: path.map((p) => p.mean), lo90: path.map((p) => p.lo90), hi90: path.map((p) => p.hi90), lo68: path.map((p) => p.lo68), hi68: path.map((p) => p.hi68) }]} horizons={catalog.horizons} units={o.units} summary={`${o.name}: ${signed(r.mean)} ${o.units} at day ${scenario.horizon}.`} />
                  </div>
                );
              })}
            </div>
          </Panel>
          <Panel eyebrow="All assets" title="What moves most" id="mv-h">
            <p className="muted">Ranked by signal-to-noise (estimate divided by its standard error). Bands are 90%.</p>
            <div className="tablewrap">
              <table>
                <thead>
                  <tr>
                    <th>Asset</th>
                    <th>Response</th>
                    <th>90% band</th>
                    <th className="txt">Units</th>
                  </tr>
                </thead>
                <tbody>
                  {shown.map((r) => {
                    const o = outcomes.get(r.id)!;
                    return (
                      <tr key={r.id}>
                        <td>{o.name}</td>
                        <td className="num">{signed(r.mean)}</td>
                        <td className="num">
                          {signed(r.lo90)} to {signed(r.hi90)}
                        </td>
                        <td className="txt">{o.units}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            {ranked.length > 12 && (
              <button type="button" className="btn ghost" onClick={() => setShowAll(!showAll)}>
                {showAll ? "Show top 12" : `Show all ${ranked.length}`}
              </button>
            )}
          </Panel>
        </>
      )}
    </>
  );
}
