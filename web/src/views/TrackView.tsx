import { useData } from "../data";
import { signed } from "../format";
import type { LiveEntry } from "../types";
import { Chip, Loading, Panel } from "../components/ui";

const OUT_TEXT: Record<string, string> = { CA2Y: "Canada 2-year (bp)", USDCAD: "USD/CAD (%)", CA_BANKS_EW6: "Canadian banks (%)" };

export default function TrackView() {
  const d = useData<{ entries: LiveEntry[] }>("live");
  if (!d.data) return <Loading error={d.error} />;
  const forecasts = d.data.entries.filter((e) => e.kind === "forecast");
  const scores = new Map(d.data.entries.filter((e) => e.kind === "score").map((e) => [e.event_id, e]));
  return (
    <Panel eyebrow="Live, pre-registered" title="Track record" chip={<Chip label="model_forecast" />} id="tr-h">
      <p className="lede">
        Before each Fed and Bank of Canada decision, a conditional forecast is published and committed with a timestamp: if the surprise is
        −10, 0 or +10 bp (measured as the day's 2-year yield move), here is the expected move in Canadian assets. After the decision the
        forecast is scored against what happened. Past forecasts are never edited.
      </p>
      {forecasts.length === 0 && <p className="muted">No forecasts published yet.</p>}
      {forecasts.map((f) => {
        const s = scores.get(f.event_id);
        return (
          <section key={f.event_id} className="card">
            <h3>
              {f.bank.toUpperCase()} decision of {f.decision_date} {f.collision ? "· same day as the other central bank (flagged)" : ""}
            </h3>
            <p className="muted">
              Published {f.published_at} · surprise measured by {f.proxy}
            </p>
            <div className="tablewrap">
              <table>
                <thead>
                  <tr>
                    <th>Outcome · horizon</th>
                    <th>If −10 bp</th>
                    <th>If 0</th>
                    <th>If +10 bp</th>
                    <th>Realized</th>
                  </tr>
                </thead>
                <tbody>
                  {Array.from(new Set((f.grid ?? []).map((g) => `${g.outcome}|${g.h}`))).map((key) => {
                    const [out, h] = key.split("|");
                    const at = (bp: number) => f.grid!.find((g) => g.outcome === out && String(g.h) === h && g.surprise_bp === bp)!;
                    const sc = s?.scores?.find((x) => x.outcome === out && String(x.h) === h);
                    return (
                      <tr key={key}>
                        <td>
                          {OUT_TEXT[out] ?? out} · day {h}
                        </td>
                        {[-10, 0, 10].map((bp) => (
                          <td key={bp} className="num">
                            {signed(at(bp).mean)} <span className="muted">[{signed(at(bp).lo90)}, {signed(at(bp).hi90)}]</span>
                          </td>
                        ))}
                        <td className="num">
                          {sc ? `${signed(sc.realized)} (forecast ${signed(sc.forecast)}; ${sc.inside90 ? "inside" : "outside"} 90%)` : "pending"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            {s && <p className="muted">Realized surprise proxy: {signed(s.realized_proxy_bp ?? NaN)} bp.</p>}
          </section>
        );
      })}
    </Panel>
  );
}
