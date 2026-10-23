import { useAll } from "../data";
import { band, pct, plain, signed } from "../format";
import { Chip, Loading, Panel } from "../components/ui";

type Scores = Record<string, number>;
type Diff = { point: number; ci90: [number, number] };
type Study = {
  protocol_tag: string;
  n_test: number;
  test_years: number[];
  chosen: { mean: string; tail: string };
  mean: { scores: Scores; difference: Diff; verdict: string; coverage90: Scores; static: Diff };
  tail: {
    scores: Scores;
    difference: Diff;
    verdict: string;
    hit_rate_05: Scores;
    coverage_90: Scores;
    fissler_ziegel: Scores;
  };
  shap_share: Record<string, number>;
  embeddings: { ari: number; p: number };
  portfolio: { pooled_scaled_pinball: Scores; difference_ml_minus_gauss: Diff };
  stress: {
    size_sd: number;
    assets: Record<string, { q50_zero: number; q50_shock: number; q05_zero: number; q05_shock: number }>;
  };
  inputs: { matched_pin: number; revised: number };
};

const MEAN_ROWS: [string, string][] = [
  ["zero", "Predict zero"],
  ["lp", "Linear projection"],
];
const TAIL_ROWS: [string, string][] = [
  ["hist", "Historical quantile"],
  ["gauss_lp", "Gaussian linear projection"],
];

function row(label: string, score: number, diff?: Diff) {
  return (
    <tr key={label}>
      <td>{label}</td>
      <td className="num">{plain(score, 3)}</td>
      <td className="num">{diff ? signed(diff.point, 3) : "–"}</td>
      <td className="num">{diff ? band(diff.ci90[0], diff.ci90[1], 3) : "–"}</td>
    </tr>
  );
}

export default function ModelsView() {
  const d = useAll<{ mlstudy: Study }>(["mlstudy"]);
  if (!d || typeof d === "string") return <Loading error={typeof d === "string" ? d : null} />;
  const s = d.mlstudy;
  const meanKey = Object.keys(s.mean.scores).find((k) => k !== "zero" && k !== "lp") ?? "";
  const tailKey = Object.keys(s.tail.scores).find((k) => k !== "hist" && k !== "gauss_lp") ?? "";
  const years = `${Math.min(...s.test_years)}–${Math.max(...s.test_years)}`;
  return (
    <>
      <section className="panel" aria-labelledby="models-h">
        <div className="eyebrow">Second study · {s.protocol_tag}</div>
        <h1 id="models-h">Pooled networks and the 5% tail</h1>
        <p className="lede">
          Same announcements and the same calendar split as the first machine-learning test, which stays on the overview. This page is a
          separate pre-registered comparison. Negative differences mean lower loss than the baseline.
        </p>
        <Chip label="model_forecast" />
      </section>

      <Panel eyebrow="Decision" title={`Held-out announcements, ${years}`} id="study-decision">
        <p>
          {s.n_test} test announcements. The mean model is the {s.chosen.mean}; the tail model is {s.chosen.tail}. The rule adds value only
          when the whole 90% interval of selected minus baseline lies below zero.
        </p>
        <div className="tablewrap">
          <table>
            <caption>Pooled scaled loss on the test set. The scale is the training-split loss of a naive forecast.</caption>
            <thead>
              <tr>
                <th>Model</th>
                <th>Score</th>
                <th>Minus baseline</th>
                <th>90% CI</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <th colSpan={4}>Mean response, scaled MAE. Verdict: {s.mean.verdict}</th>
              </tr>
              {MEAN_ROWS.map(([k, lab]) => row(lab, s.mean.scores[k]))}
              {row(s.chosen.mean, s.mean.scores[meanKey], s.mean.difference)}
              <tr>
                <th colSpan={4}>5% tail, scaled pinball. Verdict: {s.tail.verdict}</th>
              </tr>
              {TAIL_ROWS.map(([k, lab]) => row(lab, s.tail.scores[k]))}
              {row(s.chosen.tail, s.tail.scores[tailKey], s.tail.difference)}
            </tbody>
          </table>
        </div>
        <p>
          A single fit of the mean model through the validation window, which cannot change the verdict, differs from the linear projection
          by <span className="num">{signed(s.mean.static.point, 3)}</span> (90% CI{" "}
          <span className="num">{band(s.mean.static.ci90[0], s.mean.static.ci90[1], 3)}</span>). 90% interval coverage of the mean model is{" "}
          <span className="num">{pct(s.mean.coverage90.lp, 1)}</span> for the linear projection and{" "}
          <span className="num">{pct(s.mean.coverage90[meanKey], 1)}</span> for the network. The 5% quantile is crossed{" "}
          <span className="num">{pct(s.tail.hit_rate_05[tailKey], 1)}</span> of the time (
          <span className="num">{pct(s.tail.hit_rate_05.gauss_lp, 1)}</span> for the Gaussian), and the tail model's central interval covers{" "}
          <span className="num">{pct(s.tail.coverage_90[tailKey], 1)}</span> of outcomes (
          <span className="num">{pct(s.tail.coverage_90.gauss_lp, 1)}</span> for the Gaussian).
        </p>
      </Panel>

      <Panel eyebrow="What the network uses" title="Grouped Shapley shares, horizon 1" id="study-shap" chip={<Chip label="model_forecast" />}>
        <div className="tablewrap">
          <table>
            <caption>Share of mean absolute grouped Shapley value on the test announcements.</caption>
            <thead>
              <tr>
                <th>Group</th>
                <th>Share</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(s.shap_share).map(([k, v]) => (
                <tr key={k}>
                  <td>{k.replace(/_/g, " ")}</td>
                  <td className="num">{pct(v, 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p>
          Asset embeddings versus the pre-specified groups: adjusted Rand index <span className="num">{plain(s.embeddings.ari, 2)}</span>,
          permutation p <span className="num">{plain(s.embeddings.p, 2)}</span>.
        </p>
      </Panel>

      <Panel
        eyebrow="Secondary"
        title="Equity book and a hypothetical Fed surprise"
        id="study-stress"
        chip={<Chip label="hypothetical_scenario" />}
      >
        <p>
          The book’s scaled 5% pinball difference versus the Gaussian projection is{" "}
          <span className="num">{signed(s.portfolio.difference_ml_minus_gauss.point, 3)}</span> (90% CI{" "}
          <span className="num">
            {band(s.portfolio.difference_ml_minus_gauss.ci90[0], s.portfolio.difference_ml_minus_gauss.ci90[1], 3)}
          </span>
          ). It does not change the verdict. The stress row is a one-standard-deviation two-year futures surprise (
          <span className="num">{plain(s.stress.size_sd, 2)}</span> in the training sample), other components set by their training
          co-movement, state at the training median. It is not a test-set score.
        </p>
        <div className="tablewrap">
          <table>
            <caption>Model fit through 2018. Median and 5% quantile of the one-day move, zero surprise versus the hypothetical shock.</caption>
            <thead>
              <tr>
                <th>Asset</th>
                <th>Median, zero</th>
                <th>Median, shock</th>
                <th>5%, zero</th>
                <th>5%, shock</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(s.stress.assets).map(([asset, r]) => (
                <tr key={asset}>
                  <td>{asset}</td>
                  <td className="num">{signed(r.q50_zero, 2)}</td>
                  <td className="num">{signed(r.q50_shock, 2)}</td>
                  <td className="num">{signed(r.q05_zero, 2)}</td>
                  <td className="num">{signed(r.q05_shock, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="muted">
          Data vintage: {s.inputs.matched_pin} artifacts matched the committed pins and {s.inputs.revised} series were revised by the source.
        </p>
      </Panel>
    </>
  );
}
