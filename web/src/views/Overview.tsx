import { useAll } from "../data";
import { band, signed } from "../format";
import type { M4Data } from "../types";
import { Chip, Loading, Panel } from "../components/ui";
import { Z90 } from "../engine/engine";

type Summary = {
  canada: M4Data["canada"];
  state_dependence: M4Data["state_dependence"];
  claims: { id: string; verdict: string; source: string; mechanical: boolean }[];
  ml: { verdict: string; test_events_primary: number; test_years: number[]; difference_ml_minus_econ: { point: number; ci90: [number, number] } };
};
type D = { summary: Summary };

export default function Overview() {
  const d = useAll<D>(["summary"]);
  if (!d || typeof d === "string") return <Loading error={typeof d === "string" ? d : null} />;
  const { summary } = d;
  const m4 = summary;
  const m3 = summary;
  const ml = { final: summary.ml };
  const can = (out: string) => m4.canada.find((c) => c.outcome === out)!;
  const tsx = can("TSX.tr");
  const ca10 = can("CA10Y");
  const fedB = (c: typeof tsx) => ({ m: c.fed["0"][0], lo: c.fed["0"][0] - Z90 * c.fed["0"][1], hi: c.fed["0"][0] + Z90 * c.fed["0"][1] });
  const bocB = (c: typeof tsx) => ({ m: c.boc["0"][0], lo: c.boc["0"][0] - Z90 * c.boc["0"][1], hi: c.boc["0"][0] + Z90 * c.boc["0"][1] });
  const tsxFed = fedB(tsx);
  const tsxBoc = bocB(tsx);
  const tsxRs = tsx.boc_rigobon_sack["0"];
  // Published findings only: reference rows ("r") and the project's own expected-sign checks are left out.
  const claims = m3.claims.filter((c) => !c.id.endsWith("r") && !c.source.startsWith("Project brief"));
  const substantive = claims.filter((c) => !c.mechanical);
  const substantiveOk = substantive.filter((c) => c.verdict === "replicated").length;
  const replicated = claims.filter((c) => c.verdict === "replicated").length;
  const opposite = claims.filter((c) => c.verdict.startsWith("opposite")).length;
  const sameSign = claims.filter((c) => c.verdict.startsWith("same")).length;
  const fin = ml.final;
  const sdCells = m4.state_dependence;
  const surviving = sdCells.filter((c) => Math.min(c.q_joint, c.q_headline) <= 0.1).length;

  return (
    <>
      <section className="panel" aria-labelledby="hero">
        <div className="eyebrow">Research lab · identified shocks, with a Canada lens</div>
        <h1 id="hero">How the world, and Canada, reacts to economic shocks</h1>
        <p className="lede">
          Fed and Bank of Canada rate surprises, oil supply news, and geopolitical and trade-policy risk, measured with high-frequency
          identification and local projections, then turned into scenarios a strategist can use. Every estimate is labelled for the kind
          of evidence it is, and the results that disappoint are reported alongside the ones that do not.
        </p>
        <div>
          <a className="btn" href="#/scenario">
            Build a scenario
          </a>{" "}
          <a className="btn ghost" href="#/map">
            See the world map
          </a>
        </div>
      </section>

      <div className="grid3">
        <div className="card">
          <div className="eyebrow">Finding 1 · Canada lens</div>
          <h3>Canadian stocks react more to the Fed than to the Bank of Canada</h3>
          <p>
            When each central bank's surprise lifts its own 2-year yield by 10 bp, the S&amp;P/TSX moves{" "}
            <span className="num">{signed(tsxFed.m)}%</span> on the day of a Fed move (90% band{" "}
            <span className="num">{band(tsxFed.lo, tsxFed.hi)}</span>) and <span className="num">{signed(tsxBoc.m)}%</span> after a BoC
            move (<span className="num">{band(tsxBoc.lo, tsxBoc.hi)}</span>; <span className="num">{signed(tsxRs[0])}%</span> with the
            Rigobon–Sack estimator). The Canadian 10-year yield reacts about equally:{" "}
            <span className="num">{signed(ca10.fed["0"][0])}</span> vs <span className="num">{signed(ca10.boc["0"][0])}</span> bp.
          </p>
          <Chip label="identified_causal" />
        </div>
        <div className="card">
          <div className="eyebrow">Finding 2 · does ML help?</div>
          <h3>{fin.verdict}</h3>
          <p>
            On {fin.test_events_primary} held-out announcements ({Math.min(...fin.test_years)}–{Math.max(...fin.test_years)}), scored once
            under a pre-registered protocol, the neural network's error minus the linear model's is{" "}
            <span className="num">{signed(fin.difference_ml_minus_econ.point, 3)}</span> (90% CI{" "}
            <span className="num">{band(fin.difference_ml_minus_econ.ci90[0], fin.difference_ml_minus_econ.ci90[1], 3)}</span>).
          </p>
          <Chip label="model_forecast" />
          <p>
            <a href="#/models">A second study</a> asks the question again with pooled networks and a left-tail model. It does not replace this
            result.
          </p>
        </div>
        <div className="card">
          <div className="eyebrow">Finding 3 · credibility</div>
          <h3>
            {replicated} of {claims.length} published findings replicate; {surviving === 0 ? "no regime effect survives" : `${surviving} regime effect${surviving === 1 ? " survives" : "s survive"}`}
          </h3>
          <p>
            Of the {substantive.length} that are not true almost by construction, {substantiveOk} replicate.{" "}
            {sameSign > 0 && `${sameSign} more have the expected sign but bands that include zero; `}
            {opposite === 0 ? "none" : opposite} come{opposite === 1 ? "s" : ""} out with the opposite sign. Of {sdCells.length} pre-registered
            regime tests, {surviving === 0 ? "none survives" : `${surviving} survive`} false-discovery control: with this many announcements, the
            same shock does not measurably matter more in volatile or high-inflation times.
          </p>
          <Chip label="identified_causal" />
        </div>
      </div>

      <Panel eyebrow="Read this first" title="What this is, and what it is not" id="isnot">
        <div className="grid2">
          <div>
            <h3>It is</h3>
            <ul>
              <li>Causal estimates where identification supports them (high-frequency Fed surprises, OPEC announcements, a validated BoC proxy).</li>
              <li>Reduced-form associations for geopolitical and trade-policy risk, labelled as such.</li>
              <li>A pre-registered, once-only test of whether machine learning adds anything (it does not, here).</li>
              <li>Scenarios that add estimated responses, with uncertainty, explicit linearity and out-of-range warnings.</li>
            </ul>
          </div>
          <div>
            <h3>It is not</h3>
            <ul>
              <li>A forecast of what markets will do: scenarios are conditional and hypothetical.</li>
              <li>Investment advice.</li>
              <li>Evidence that effects differ by regime: the pre-registered tests found nothing that survives correction.</li>
              <li>Reliable for monthly macro effects: those instruments are weak (see Methodology).</li>
            </ul>
          </div>
        </div>
      </Panel>
    </>
  );
}
