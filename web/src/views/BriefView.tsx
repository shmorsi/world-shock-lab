import { useMemo, useState } from "react";
import { useAll } from "../data";
import { useScenario } from "../scenario";
import type { Catalog, M4Data } from "../types";
import { Chip, Loading, Panel } from "../components/ui";
import { compose } from "../engine/engine";
import { buildBrief, briefText, render } from "../brief/brief";

type D = { catalog: Catalog; m4: M4Data };

export default function BriefView() {
  const d = useAll<D>(["catalog", "m4"]);
  const { scenario } = useScenario();
  const [copied, setCopied] = useState<string | null>(null);
  const brief = useMemo(() => {
    if (!d || typeof d === "string") return null;
    const result = compose(d.catalog, scenario.sizes, scenario.horizon);
    return { result, brief: buildBrief(d.catalog, scenario.sizes, result, d.m4.robustness) };
  }, [d, scenario]);
  if (!d || typeof d === "string" || !brief) return <Loading error={typeof d === "string" ? d : null} />;
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(briefText(brief.brief));
      setCopied("Copied to the clipboard.");
    } catch {
      setCopied("Copying is blocked here; select the text and copy it instead.");
    }
  };
  return (
    <Panel eyebrow="View 5 · Strategist Brief" title="One-page brief for the current scenario" chip={<Chip label="hypothetical_scenario" />} id="br-h">
      <p className="muted">
        Written by templates from the scenario's numbers only: every figure below comes from the result files, and a test fails if the
        brief ever contains a number that is not in the data. Change the scenario in the{" "}
        <a href="#/scenario">Scenario Composer</a>.
      </p>
      <article className="brief" aria-live="polite">
        {brief.brief.sections.map((s) => (
          <section key={s.heading}>
            <h3>{s.heading}</h3>
            {s.paragraphs.length > 1 ? (
              <ul>
                {s.paragraphs.map((p, i) => (
                  <li key={i}>{render(p)}</li>
                ))}
              </ul>
            ) : (
              <p>{render(s.paragraphs[0])}</p>
            )}
          </section>
        ))}
      </article>
      <div className="controls">
        <button type="button" className="btn" onClick={copy}>
          Copy brief as text
        </button>
        {copied && <span className="muted">{copied}</span>}
      </div>
    </Panel>
  );
}
