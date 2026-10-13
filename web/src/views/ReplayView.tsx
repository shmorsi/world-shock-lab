import { useState } from "react";
import { useAll } from "../data";
import { signed, stamp } from "../format";
import type { FinalEval, ReplayEvent } from "../types";
import { Chip, Loading, Panel, Seg } from "../components/ui";

type D = { replay: { replay: ReplayEvent[]; replay_assets: string[]; final: Pick<FinalEval, "generated_at" | "protocol_tag" | "test_years" | "pooled_scaled_mae" | "per_cell" | "difference_ml_minus_econ" | "verdict"> } };
const MODEL_TEXT: Record<string, string> = { zero: "No move", lp: "Linear LP", state: "State LP", ridge: "Ridge", gbm: "Boosting", mlp: "Neural net" };
const FAMILY_TEXT: Record<string, string> = { fomc: "FOMC", boc: "Bank of Canada", opec: "OPEC" };
const name = (k: string) => MODEL_TEXT[k.split(":")[0]] ?? k;

export default function ReplayView() {
  const d = useAll<D>(["replay"]);
  const [family, setFamily] = useState("fomc");
  const [picked, setPicked] = useState<string | null>(null);
  if (!d || typeof d === "string") return <Loading error={typeof d === "string" ? d : null} />;
  const fin = { ...d.replay.final, replay: d.replay.replay, replay_assets: d.replay.replay_assets };
  const events = fin.replay.filter((e) => e.family === family);
  const ev = events.find((e) => e.event === picked) ?? events[events.length - 1];
  const models = Object.keys(fin.pooled_scaled_mae);
  const cells = ev ? Object.entries(ev.cells) : [];

  return (
    <>
      <Panel eyebrow="View 4 · Event Replay" title="Held-out announcements: predicted vs what happened" chip={<Chip label="model_forecast" />} id="rp-h">
        <p className="lede">
          Every announcement from {Math.min(...fin.test_years)} to {Math.max(...fin.test_years)} was held out while the models were built and
          tuned, then scored once ({stamp(fin.generated_at)}, protocol {fin.protocol_tag.replace("protocol-", "")}). Predictions use only information available before each
          announcement plus its own surprise; models are refit each year on earlier data only.
        </p>
        <div className="controls">
          <Seg label="Announcements" value={family} onChange={(f) => { setFamily(f); setPicked(null); }} options={["fomc", "boc", "opec"].map((f) => ({ value: f, text: FAMILY_TEXT[f] }))} />
          <div className="ctl">
            <label htmlFor="rp-ev">Event</label>
            <select id="rp-ev" value={ev?.event ?? ""} onChange={(e) => setPicked(e.target.value)}>
              {events.map((e) => (
                <option key={e.event} value={e.event}>
                  {e.ts.slice(0, 10)}
                  {e.same_day_fomc ? " (same day as FOMC; not scored)" : ""}
                </option>
              ))}
            </select>
          </div>
        </div>
        {ev && cells.length > 0 ? (
          <div className="tablewrap">
            <table>
              <caption className="muted" style={{ textAlign: "left", paddingBottom: 6 }}>
                {FAMILY_TEXT[ev.family]} announcement of {ev.ts.slice(0, 10)}. h1 = the first observation after the announcement; h5 = the fifth.
              </caption>
              <thead>
                <tr>
                  <th>Asset · horizon</th>
                  <th>What happened</th>
                  {models.map((m) => (
                    <th key={m}>{name(m)}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {cells.map(([cell, c]) => (
                  <tr key={cell}>
                    <td>{cell.replace("|", " · ")}</td>
                    <td className="num">
                      <strong>{signed(c.actual)}</strong>
                    </td>
                    {models.map((m) => (
                      <td key={m} className="num" title={`error ${signed(c.pred[m] - c.actual)}`}>
                        {signed(c.pred[m])}
                        <span className="muted"> ({signed(c.pred[m] - c.actual)})</span>
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="muted">No event-level values to show for this selection.</p>
        )}
        <p className="muted">
          Shown per event for public-domain series only ({fin.replay_assets.join(", ")}); the licensed prices (Yahoo equities, CanDeal bond
          yields) appear only in the aggregate scores below. Values in brackets are prediction minus outcome.
        </p>
      </Panel>
      <Panel eyebrow="All held-out events" title={`Verdict: ${fin.verdict}`} id="rp-v">
        <p>
          Pooled scaled error: each asset's mean absolute error divided by its average absolute move in the training years, averaged over 8
          assets × 2 horizons (lower is better; "no move" always predicts zero, so it has no direction to get right). Neural net minus the linear
          model: <span className="num">{signed(fin.difference_ml_minus_econ.point, 3)}</span>, 90% CI{" "}
          <span className="num">
            {signed(fin.difference_ml_minus_econ.ci90[0], 3)} to {signed(fin.difference_ml_minus_econ.ci90[1], 3)}
          </span>
          .
        </p>
        <div className="tablewrap">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Pooled scaled error</th>
                <th>90% interval coverage</th>
                <th>Direction right (large moves)</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => {
                const pc = Object.values(fin.per_cell[m]);
                const avg = (k: "coverage90" | "directional_accuracy") => pc.reduce((a, c) => a + c[k], 0) / pc.length;
                return (
                  <tr key={m}>
                    <td>{name(m)}</td>
                    <td className="num">{fin.pooled_scaled_mae[m].toFixed(3)}</td>
                    <td className="num">{(100 * avg("coverage90")).toFixed(0)}%</td>
                    <td className="num">{m.startsWith("zero") ? "n/a" : `${(100 * avg("directional_accuracy")).toFixed(0)}%`}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Panel>
    </>
  );
}
