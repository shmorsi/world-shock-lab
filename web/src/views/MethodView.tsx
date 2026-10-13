import { useAll } from "../data";
import { signed } from "../format";
import type { Claim, M4Data } from "../types";
import { Chip, Loading, Panel } from "../components/ui";

type D = { m3: { claims: Claim[] }; m4: M4Data };

export default function MethodView() {
  const d = useAll<D>(["m3", "m4"]);
  if (!d || typeof d === "string") return <Loading error={typeof d === "string" ? d : null} />;
  const robust = d.m4.robustness.filter((r) => r.verdict === "robust").length;
  const main = d.m3.claims.filter((c) => !c.id.endsWith("r"));
  const published = main.filter((c) => !c.source.startsWith("Project brief"));
  const checks = main.filter((c) => c.source.startsWith("Project brief"));
  return (
    <>
      <Panel eyebrow="View 6 · Methodology" title="How the estimates are made, in plain English" id="me-h">
        <div className="grid2">
          <div>
            <h3>Identification: where cause and effect come from</h3>
            <p>
              Markets move for many reasons at once. To isolate a central bank's effect, the Fed surprise is measured in a 30-minute window
              around each announcement using rate futures (Jarociński &amp; Karadi), so almost nothing else moves it. Oil supply news comes
              from oil-futures moves on OPEC announcement days (Känzig). For the Bank of Canada no free intraday data exists, so the surprise
              is the change in the Canadian 2-year yield on announcement days: a daily-window proxy, tested before use.
            </p>
          </div>
          <div>
            <h3>Local projections: how a response is traced</h3>
            <p>
              For each horizon h, one regression asks how much an asset moved from just before the announcement to h days after, per unit
              of surprise. Each asset's window uses its own closing time: the 2 p.m. Fed statement shows up in noon-fixed exchange rates a
              day later than in 4 p.m. stock closes. Bands are 68% and 90%, with standard errors validated by simulation.
            </p>
          </div>
          <div>
            <h3>Four kinds of evidence, always labelled</h3>
            <p>
              <Chip label="identified_causal" /> responses to high-frequency or validated surprises. <Chip label="reduced_form" /> geopolitical
              and trade-policy risk, which are measures of news intensity, not shocks. <Chip label="model_forecast" /> machine-learning and
              live forecasts. <Chip label="hypothetical_scenario" /> anything built in the Composer, which adds responses together.
            </p>
          </div>
          <div>
            <h3>Pre-registration: deciding the tests before seeing results</h3>
            <p>
              Time splits, metrics and the rule for "ML adds value" were fixed and tagged in git before any model was fitted; the 2019–2026
              test set was scored exactly once. Regime tests, the Canada comparison and robustness rules were also fixed before estimation,
              and false-discovery control is applied to whole families of tests.
            </p>
          </div>
        </div>
      </Panel>
      <Panel eyebrow="Checks" title="What was tested before trusting the numbers" id="me-c">
        <ul>
          <li>Simulation: on data with known answers, the estimators recover them and their bands have close to nominal coverage.</li>
          <li>
            Replication: {published.filter((c) => c.verdict === "replicated").length} of {published.length} published findings reproduce, and{" "}
            {checks.filter((c) => c.verdict === "replicated").length} of {checks.length} expected-sign checks on currencies hold (table below;
            "mechanical" marks claims that are close to true by construction).
          </li>
          <li>Placebo: the same surprises assigned to random non-announcement days produce responses indistinguishable from zero.</li>
          <li>
            Robustness: {robust} of {d.m4.robustness.length} headline responses keep their sign and significance across samples, controls and
            estimators.
          </li>
        </ul>
        <div className="tablewrap">
          <table>
            <thead>
              <tr>
                <th>Claim</th>
                <th className="txt">Source</th>
                <th>Ours (90% band)</th>
                <th className="txt">Verdict</th>
              </tr>
            </thead>
            <tbody>
              {d.m3.claims.map((c) => (
                <tr key={c.id}>
                  <td>
                    {c.claim}
                    {c.mechanical ? " (mechanical)" : ""}
                  </td>
                  <td className="txt">{c.source}</td>
                  <td className="num">
                    {signed(c.estimate)} [{signed(c.band90[0])}, {signed(c.band90[1])}]
                  </td>
                  <td className="txt">{c.verdict}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
      <Panel eyebrow="Limits" title="What this cannot tell you" id="me-l">
        <ul>
          <li>Responses are averages over past announcements; the next one can differ, and scenarios assume effects add up linearly.</li>
          <li>Monthly macro effects rest on weak instruments (first-stage F below 10); treat them as suggestive.</li>
          <li>The Bank of Canada proxy still carries some common North American rate news; Rigobon–Sack estimates are shown alongside.</li>
          <li>Country equity responses use US-listed ETFs in US dollars, which combine stock and currency moves.</li>
          <li>No regime difference survives multiple-testing correction; small effects of this kind would be invisible at these sample sizes.</li>
        </ul>
      </Panel>
    </>
  );
}
