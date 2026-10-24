import type { ReactNode } from "react";
import { useAll } from "../data";
import { band, pct, signed } from "../format";
import type { LiveEntry, M4Data } from "../types";
import { Chip, Panel } from "../components/ui";
import { Z90 } from "../engine/engine";

type Summary = {
  canada: M4Data["canada"];
  state_dependence: M4Data["state_dependence"];
  claims: { id: string; verdict: string; source: string; mechanical: boolean }[];
  ml: {
    verdict: string;
    test_events_primary: number;
    test_years: number[];
    difference_ml_minus_econ: { point: number; ci90: [number, number] };
  };
};
type Diff = { point: number; ci90: [number, number] };
type Study = {
  n_test: number;
  test_years: number[];
  chosen: { mean: string; tail: string };
  mean: { difference: Diff; verdict: string };
  tail: { difference: Diff; verdict: string; hit_rate_05: Record<string, number> };
};
type D = { summary: Summary; mlstudy: Study; live: { entries: LiveEntry[] } };

const BANK: Record<string, string> = { fomc: "Fed", boc: "Bank of Canada" };

function Result({
  n,
  href,
  kicker,
  figure,
  detail,
}: {
  n: string;
  href: string;
  kicker: string;
  figure: string;
  detail: string;
}) {
  return (
    <li>
      <a href={href}>
        <span className="kicker">
          {n} · {kicker}
        </span>
        <span className="fig num">{figure}</span>
        <span className="fig-label">{detail}</span>
      </a>
    </li>
  );
}

export default function Overview() {
  const d = useAll<D>(["summary", "mlstudy", "live"]);
  const error = typeof d === "string" ? d : null;
  const ready = d && typeof d !== "string" ? d : null;

  let index: ReactNode = <p className="hero-status">{error ? `Could not load results: ${error}` : "Loading results…"}</p>;
  let below: ReactNode = null;

  if (ready) {
    const { summary, mlstudy, live } = ready;
    const can = (out: string) => summary.canada.find((c) => c.outcome === out)!;
    const tsx = can("TSX.tr");
    const ca10 = can("CA10Y");
    const fedB = (c: typeof tsx) => ({ m: c.fed["0"][0], lo: c.fed["0"][0] - Z90 * c.fed["0"][1], hi: c.fed["0"][0] + Z90 * c.fed["0"][1] });
    const bocB = (c: typeof tsx) => ({ m: c.boc["0"][0], lo: c.boc["0"][0] - Z90 * c.boc["0"][1], hi: c.boc["0"][0] + Z90 * c.boc["0"][1] });
    const tsxFed = fedB(tsx);
    const tsxBoc = bocB(tsx);
    const tsxRs = tsx.boc_rigobon_sack["0"];
    const claims = summary.claims.filter((c) => !c.id.endsWith("r") && !c.source.startsWith("Project brief"));
    const substantive = claims.filter((c) => !c.mechanical);
    const substantiveOk = substantive.filter((c) => c.verdict === "replicated").length;
    const replicated = claims.filter((c) => c.verdict === "replicated").length;
    const opposite = claims.filter((c) => c.verdict.startsWith("opposite")).length;
    const sameSign = claims.filter((c) => c.verdict.startsWith("same")).length;
    const fin = summary.ml;
    const sdCells = summary.state_dependence;
    const surviving = sdCells.filter((c) => Math.min(c.q_joint, c.q_headline) <= 0.1).length;
    const mean = mlstudy.mean.difference;
    const tail = mlstudy.tail.difference;
    const tailKey = Object.keys(mlstudy.tail.hit_rate_05).find((k) => k !== "hist" && k !== "gauss_lp") ?? "";
    const hit = mlstudy.tail.hit_rate_05[tailKey];
    const years = `${Math.min(...mlstudy.test_years)}–${Math.max(...mlstudy.test_years)}`;
    const forecasts = live.entries.filter((e) => e.kind === "forecast");
    const scored = live.entries.filter((e) => e.kind === "score" && e.score === 1);
    const dates = [...new Set(forecasts.map((e) => e.decision_date))].sort();
    const banks = [...new Set(forecasts.map((e) => BANK[e.bank] ?? e.bank))].sort(
      (a, b) => (a === "Fed" ? 0 : 1) - (b === "Fed" ? 0 : 1),
    );
    const liveFig = forecasts.length === 0 ? "None yet" : scored.length === 0 ? "Pending" : String(scored.length);
    const liveDetail =
      forecasts.length === 0
        ? "No forecast has been published yet."
        : scored.length === 0
          ? `${banks.join(" and ")} forecasts for ${dates.join(", ")} are committed. No decision has been scored yet.`
          : `${scored.length} scored so far. A track record takes years.`;

    index = (
      <ol className="result-index" aria-label="Key results">
        <Result
          n="01"
          href="#finding-canada"
          kicker="S&P/TSX · Fed"
          figure={`${signed(tsxFed.m)}%`}
          detail={`90% band ${band(tsxFed.lo, tsxFed.hi)}. Scaled to a 10 bp rise in the US 2-year yield.`}
        />
        <Result
          n="02"
          href="#finding-canada"
          kicker="S&P/TSX · Bank of Canada"
          figure={`${signed(tsxBoc.m)}%`}
          detail={`90% band ${band(tsxBoc.lo, tsxBoc.hi)}. Rigobon–Sack ${signed(tsxRs[0])}%.`}
        />
        <Result
          n="03"
          href="#/models"
          kicker={`${mlstudy.chosen.mean} − linear`}
          figure={signed(mean.point, 3)}
          detail={`${mlstudy.mean.verdict}. 90% CI ${band(mean.ci90[0], mean.ci90[1], 3)}.`}
        />
        <Result
          n="04"
          href="#/models"
          kicker="5% tail · boosting − Gaussian"
          figure={signed(tail.point, 3)}
          detail={`${mlstudy.tail.verdict}. Quantile crossed ${pct(hit, 1)} of the time.`}
        />
        <Result
          n="05"
          href="#finding-trust"
          kicker="Published findings"
          figure={`${replicated} of ${claims.length}`}
          detail={`Sign and 90% band. ${substantiveOk} of ${substantive.length} are not true almost by construction.`}
        />
        <Result n="06" href="#/track" kicker="Live forecasts" figure={liveFig} detail={liveDetail} />
      </ol>
    );

    below = (
      <div className="frame">
        <div className="grid3">
          <div className="card" id="finding-canada">
            <div className="eyebrow">Canada lens</div>
            <h2>Stocks move more with the Fed than with the Bank of Canada</h2>
            <p>
              When each surprise lifts its own 2-year yield by 10 bp, the S&amp;P/TSX moves <span className="num">{signed(tsxFed.m)}%</span>{" "}
              on a Fed day (90% band <span className="num">{band(tsxFed.lo, tsxFed.hi)}</span>) and{" "}
              <span className="num">{signed(tsxBoc.m)}%</span> after the Bank of Canada (
              <span className="num">{band(tsxBoc.lo, tsxBoc.hi)}</span>; Rigobon–Sack <span className="num">{signed(tsxRs[0])}%</span>
              ). The Canadian 10-year yield reacts about equally: <span className="num">{signed(ca10.fed["0"][0])}</span> bp versus{" "}
              <span className="num">{signed(ca10.boc["0"][0])}</span> bp.
            </p>
            <Chip label="identified_causal" />
          </div>
          <div className="card" id="finding-models">
            <div className="eyebrow">Does a network help?</div>
            <h2>{mlstudy.mean.verdict}</h2>
            <p>
              On <span className="num">{mlstudy.n_test}</span> held-out announcements ({years}), the {mlstudy.chosen.mean} minus the linear
              projection is <span className="num">{signed(mean.point, 3)}</span> (90% CI{" "}
              <span className="num">{band(mean.ci90[0], mean.ci90[1], 3)}</span>). {mlstudy.chosen.tail} minus the Gaussian on the 5% pinball loss is{" "}
              <span className="num">{signed(tail.point, 3)}</span>: {mlstudy.tail.verdict}. The first benchmark, a per-asset network on the
              same <span className="num">{fin.test_events_primary}</span> announcements, is{" "}
              <span className="num">{signed(fin.difference_ml_minus_econ.point, 3)}</span> ({fin.verdict}).
            </p>
            <Chip label="model_forecast" />
            <p>
              <a href="#/models">The second study</a> is on the models page. It does not replace the first benchmark.
            </p>
          </div>
          <div className="card" id="finding-trust">
            <div className="eyebrow">Credibility</div>
            <h2>
              {replicated} of {claims.length} published findings replicate
              {surviving === 0 ? ". No regime effect survives" : `. ${surviving} regime effect${surviving === 1 ? " survives" : "s survive"}`}
            </h2>
            <p>
              Of the {substantive.length} that are not true almost by construction, {substantiveOk} replicate.{" "}
              {sameSign > 0 && `${sameSign} more have the expected sign but bands that include zero. `}
              {opposite === 0 ? "None comes" : `${opposite} come${opposite === 1 ? "s" : ""}`} out with the opposite sign. Of {sdCells.length}{" "}
              pre-registered regime tests, {surviving === 0 ? "none survives" : `${surviving} survive`} false-discovery control.
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
                <li>Two pre-registered, once-only tests of machine learning against a linear projection: no gain on the average move, and a gain on the 5% tail pinball loss only.</li>
                <li>Scenarios that add estimated responses, with uncertainty, and explicit warnings when a size is out of range or effects are assumed to add.</li>
                <li>Live forecasts for each Fed and Bank of Canada decision, published before the decision and scored afterwards.</li>
              </ul>
            </div>
            <div>
              <h3>It is not</h3>
              <ul>
                <li>A forecast of what markets will do. Scenarios are conditional and hypothetical.</li>
                <li>Investment advice.</li>
                <li>Evidence that effects differ by regime. The pre-registered tests found nothing that survives correction.</li>
                <li>Reliable for monthly macro effects. Those instruments are weak.</li>
              </ul>
            </div>
          </div>
        </Panel>
      </div>
    );
  }

  return (
    <>
      <section className="hero" aria-labelledby="hero">
        <div className="hero-in">
          <div className="hero-main">
            <p className="eyebrow">Identified shocks · Canada lens</p>
            <h1 id="hero">
              World
              <br />
              Shock Lab
            </h1>
            <p className="tagline">How Canada's markets move when the Fed, the Bank of Canada, or OPEC surprises.</p>
            <p className="hero-meta">
              Every estimate is labelled for the claim it can support. The machine-learning tests were registered before the test set was
              opened. A scenario can only print a number the estimates contain.
            </p>
            <div className="actions">
              <a className="btn" href="#/scenario">
                Build a scenario
              </a>
              <a className="btn ghost" href="#/map">
                World map
              </a>
              <a className="btn ghost" href="#/models">
                Models
              </a>
            </div>
          </div>
          {index}
        </div>
      </section>
      {below}
    </>
  );
}
