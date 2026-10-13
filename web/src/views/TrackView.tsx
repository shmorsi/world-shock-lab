import { useData } from "../data";
import { signed, stamp } from "../format";
import type { LiveEntry, LiveScoreCell } from "../types";
import { Chip, Loading, Panel } from "../components/ui";

const OUT_TEXT: Record<string, string> = { CA2Y: "Canada 2-year yield (bp)", USDCAD: "CAD per USD (%)", CA_BANKS_EW6: "Big-6 Canadian banks (%)" };
const BANK_TEXT: Record<string, string> = { fomc: "Fed (FOMC)", boc: "Bank of Canada" };
const GRID = [-10, 0, 10];

function isoStamp(iso?: string): string {
  return iso ? stamp(iso.replace(/[-:]/g, "").replace(/\.\d+/, "")) : "";
}

function summarize(scores: LiveEntry[]) {
  const cells = scores.flatMap((s) => (s.scores ?? []).filter((c) => !c.mechanical));
  const share = (f: (c: LiveScoreCell) => boolean) => (cells.length ? cells.filter(f).length / cells.length : NaN);
  return { decisions: scores.length, cells: cells.length, in68: share((c) => c.inside68), in90: share((c) => c.inside90), beat: share((c) => c.beats_no_change) };
}

function ScoreText({ c }: { c?: LiveScoreCell }) {
  if (!c) return <span className="muted">pending</span>;
  if (c.mechanical) return <span className="muted">equals the proxy; not scored</span>;
  const band = `${c.inside90 ? "inside" : "outside"} 90%, ${c.inside68 ? "inside" : "outside"} 68%`;
  return (
    <span>
      {c.realized !== undefined && c.forecast !== undefined ? (
        <>
          <span className="num">{signed(c.realized)}</span> vs forecast <span className="num">{signed(c.forecast)}</span>;{" "}
        </>
      ) : null}
      {band}; {c.beats_no_change ? "beat" : "did not beat"} no-change
    </span>
  );
}

export default function TrackView() {
  const d = useData<{ entries: LiveEntry[] }>("live");
  if (!d.data) return <Loading error={d.error} />;
  const forecasts = d.data.entries.filter((e) => e.kind === "forecast").sort((a, b) => a.decision_date.localeCompare(b.decision_date) || a.bank.localeCompare(b.bank));
  const scores = d.data.entries.filter((e) => e.kind === "score");
  const scoreOf = (id: string, k: number) => scores.find((s) => s.event_id === id && s.score === k);
  const groups = [
    { name: "Clean days, score 1", rows: scores.filter((s) => s.score === 1 && !s.collision) },
    { name: "Collision days, score 1", rows: scores.filter((s) => s.score === 1 && s.collision) },
    { name: "Score 2 (JK intraday surprise)", rows: scores.filter((s) => s.score === 2) },
  ];
  const pct = (v: number) => (Number.isFinite(v) ? `${Math.round(100 * v)}%` : "–");

  return (
    <>
      <Panel eyebrow="Live, pre-registered" title="Track record" chip={<Chip label="model_forecast" />} id="tr-h">
        <p className="lede">
          Before each Fed and Bank of Canada decision a conditional forecast is published and committed with a timestamp: if the
          announcement moves the bank's own 2-year yield by −10, 0 or +10 bp that day, here is the expected move in Canadian assets, with
          68% and 90% ranges for the outcome. After the decision it is scored against what happened. Published files are never edited.
        </p>
        <div className="tablewrap">
          <table>
            <thead>
              <tr>
                <th>Scored so far</th>
                <th>Decisions</th>
                <th>Cells</th>
                <th>Inside 68%</th>
                <th>Inside 90%</th>
                <th>Beat no-change</th>
              </tr>
            </thead>
            <tbody>
              {groups.map((g) => {
                const s = summarize(g.rows);
                return (
                  <tr key={g.name}>
                    <td>{g.name}</td>
                    <td className="num">{s.decisions}</td>
                    <td className="num">{s.cells}</td>
                    <td className="num">{pct(s.in68)}</td>
                    <td className="num">{pct(s.in90)}</td>
                    <td className="num">{pct(s.beat)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="muted">
          With few decisions scored, these rates are noisy: ten scored decisions give 50 cells, and a 90% range would be expected to miss
          about five of them. Collision days (both banks announce) are summarized separately because daily data cannot separate the two
          decisions. Score 1 uses the day's 2-year yield move; score 2 (Fed only) re-scores with the 30-minute futures surprise once
          Jarociński &amp; Karadi publish it.
        </p>
      </Panel>

      {forecasts.length === 0 && (
        <Panel title="No forecasts published yet" id="tr-none">
          <p className="muted">Forecasts appear here ten days before each decision.</p>
        </Panel>
      )}
      {forecasts.map((f) => {
        const s1 = scoreOf(f.event_id, 1);
        const s2 = scoreOf(f.event_id, 2);
        const keys = Array.from(new Set((f.grid ?? []).map((g) => `${g.outcome}|${g.h}`)));
        return (
          <Panel
            key={f.event_id}
            eyebrow={`${f.collision ? "Collision day · " : ""}published ${isoStamp(f.published_at)}${f.hours_before ? `, ${Math.round(f.hours_before / 24)} days before` : ""}`}
            title={`${BANK_TEXT[f.bank] ?? f.bank} decision, ${f.decision_date}`}
            id={`tr-${f.event_id}`}
          >
            <p className="muted">
              Surprise measured as the {f.proxy_text}. h1 is the first observation after the announcement, h5 the fifth. {f.collision_note}
            </p>
            <div className="tablewrap">
              <table>
                <thead>
                  <tr>
                    <th>Outcome · horizon</th>
                    {GRID.map((bp) => (
                      <th key={bp}>If {bp > 0 ? `+${bp}` : bp === 0 ? "0" : `−${-bp}`} bp</th>
                    ))}
                    <th className="txt">Score 1</th>
                    {f.bank === "fomc" && <th className="txt">Score 2</th>}
                  </tr>
                </thead>
                <tbody>
                  {keys.map((key) => {
                    const [out, h] = key.split("|");
                    const at = (bp: number) => f.grid!.find((g) => g.outcome === out && String(g.h) === h && g.surprise_bp === bp)!;
                    const cell = (s?: LiveEntry) => s?.scores?.find((x) => x.outcome === out && String(x.h) === h);
                    return (
                      <tr key={key}>
                        <td>
                          {OUT_TEXT[out] ?? out} · h{h}
                        </td>
                        {GRID.map((bp) => (
                          <td key={bp} className="num">
                            {signed(at(bp).mean)} <span className="muted">[{signed(at(bp).lo90)}, {signed(at(bp).hi90)}]</span>
                          </td>
                        ))}
                        <td className="txt">
                          <ScoreText c={cell(s1)} />
                        </td>
                        {f.bank === "fomc" && (
                          <td className="txt">
                            <ScoreText c={cell(s2)} />
                          </td>
                        )}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <p className="muted">
              Brackets: 90% range for the outcome.
              {s1 && s1.realized_proxy_bp !== null && s1.realized_proxy_bp !== undefined && ` Realized surprise (score 1): ${signed(s1.realized_proxy_bp, 1)} bp.`}
              {s2 && s2.realized_proxy_bp !== null && s2.realized_proxy_bp !== undefined && ` JK surprise (score 2): ${signed(s2.realized_proxy_bp, 1)} bp.`}
              {s1?.restricted_note ? ` ${s1.restricted_note}` : ""} Realized values are shown only for public-domain series; for the
              Canadian yield and bank stocks, only whether each outcome fell inside the ranges.
            </p>
          </Panel>
        );
      })}
    </>
  );
}
