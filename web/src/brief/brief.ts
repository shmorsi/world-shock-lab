// The Strategist Brief: a one-page summary generated from a scenario's numbers by templates.
//
// It cannot introduce numbers of its own. A paragraph is a list of parts:
//   text  wording written here; must contain no digits (enforced by brief.test.ts)
//   num   a number taken from the data, recorded with the value it was formatted from
//   name  a name taken from the data files (asset and shock names may contain digits)
// No LLM is involved.
import type { Catalog, Outcome, RobustRow } from "../types";
import { BAND_LEVEL, inSd, type ScenarioResult } from "../engine/engine";
import { shockNoun } from "../format";

export type Part = { kind: "text"; text: string } | { kind: "num"; text: string; value: number } | { kind: "name"; text: string };
export interface Section {
  heading: string;
  paragraphs: Part[][];
}
export interface Brief {
  title: string;
  sections: Section[];
}

const CANADA = ["CA2Y", "CA10Y", "USDCAD", "TSX.tr", "CA_BANKS_EW6", "XEG.tr"];
const LABEL_TEXT: Record<string, string> = {
  identified_causal: "identified causal estimates",
  reduced_form: "reduced-form associations (not causal)",
  model_forecast: "model forecasts",
  hypothetical_scenario: "a hypothetical scenario",
};
const WORDS = ["zero", "one", "two", "three", "four", "five"];

const t = (text: string): Part => ({ kind: "text", text });
const n = (value: number, digits = 2, signed = true): Part => {
  const abs = Math.abs(value).toFixed(digits);
  const sign = signed ? (value > 0 ? "+" : value < 0 ? "−" : "") : value < 0 ? "−" : "";
  return { kind: "num", text: `${sign}${abs}`, value };
};
const nm = (text: string): Part => ({ kind: "name", text });

export function render(parts: Part[]): string {
  return parts.map((p) => p.text).join("");
}

function line(o: Outcome, r: ScenarioResult["outcomes"][number]): Part[] {
  return [nm(o.name), t(": "), n(r.mean), nm(` ${o.units}`), t(" ("), n(BAND_LEVEL, 0, false), t("% range "), n(r.lo90), t(" to "), n(r.hi90), t(")")];
}

export function buildBrief(
  catalog: Pick<Catalog, "shocks" | "outcomes" | "irfs">,
  sizes: Record<string, number>,
  result: ScenarioResult,
  robustness: RobustRow[],
): Brief {
  const outcomes = new Map(catalog.outcomes.map((o) => [o.id, o]));
  const shocks = new Map(catalog.shocks.map((s) => [s.id, s]));
  const used = Object.entries(sizes).filter(([, v]) => v !== 0);
  if (used.length === 0 || result.outcomes.length === 0) {
    return { title: "Strategist brief", sections: [{ heading: "No scenario", paragraphs: [[t("Set at least one shock in the Scenario Composer to generate a brief.")]] }] };
  }

  const scenario: Part[] = [t("Scenario: ")];
  used.forEach(([k, v], i) => {
    const s = shocks.get(k)!;
    if (i > 0) scenario.push(t(i === used.length - 1 ? " and " : ", "));
    scenario.push(n(v, Number.isInteger(v) ? 0 : 1), nm(` ${s.unit} `), nm(shockNoun(s.name)), t(" ("), n(inSd(catalog, k, v), 1, true), t(" standard deviations)"));
  });
  scenario.push(t(", read "), n(result.horizon, 0, false), t(" trading days after the announcement."));
  const evidence: Part[] = [t("Evidence: the responses combined here are "), t(LABEL_TEXT[result.evidence]), t(". Combining them makes the whole a hypothetical scenario, not a forecast.")];

  const ranked = result.outcomes
    .filter((r) => r.se > 0 && (r.lo90 > 0 || r.hi90 < 0))
    .sort((a, b) => Math.abs(b.mean / b.se) - Math.abs(a.mean / a.se))
    .slice(0, 5);
  const movers: Part[][] = ranked.length
    ? ranked.map((r) => line(outcomes.get(r.id)!, r))
    : [[t("No asset's "), n(BAND_LEVEL, 0, false), t("% range excludes zero at this horizon.")]];

  const canada: Part[][] = CANADA.filter((id) => outcomes.has(id)).map((id) => line(outcomes.get(id)!, result.outcomes.find((r) => r.id === id)!));

  const sig = result.outcomes.filter((r) => r.lo90 > 0 || r.hi90 < 0).length;
  const confidence: Part[] = [n(sig, 0, false), t(" of "), n(result.outcomes.length, 0, false), t(" assets have a "), n(BAND_LEVEL, 0, false), t("% range that excludes zero. Ranges reflect estimation uncertainty only, not the uncertainty of the world itself.")];

  const families = new Set(used.map(([k]) => shocks.get(k)!.family));
  const risks: Part[][] = result.warnings.map((w) => [nm(w)]);
  const fragile = robustness.filter((r) => r.verdict === "fragile" && families.has(r.family) && outcomes.has(r.outcome));
  if (fragile.length) {
    risks.push([
      t("Fragile across specifications (sign or significance changes): "),
      ...fragile.flatMap((r, i) => [...(i ? [t("; ")] : []), nm(outcomes.get(r.outcome)!.name), t(" at day "), n(r.h, 0, false)]),
      t("."),
    ]);
  }
  if (families.has("boc"))
    risks.push([t("The Bank of Canada shock is a daily-window proxy that still carries some common North American rate news; for Canadian equities the alternative Rigobon-Sack estimate is larger in magnitude.")]);
  if (families.has("index"))
    risks.push([t("Geopolitical and trade-policy jumps are associations with news intensity, not identified shocks.")]);
  risks.push([t("Not investment advice. Past announcement responses may not describe the next one.")]);

  const count = WORDS[Math.min(ranked.length, 5)];
  return {
    title: "Strategist brief",
    sections: [
      { heading: "Scenario", paragraphs: [scenario, evidence] },
      { heading: `What moves most (top ${count} by signal-to-noise)`, paragraphs: movers },
      { heading: "Canada dashboard", paragraphs: canada },
      { heading: "Confidence", paragraphs: [confidence] },
      { heading: "Key risks and caveats", paragraphs: risks },
    ],
  };
}

export function briefText(b: Brief): string {
  return [b.title, ...b.sections.flatMap((s) => [s.heading, ...s.paragraphs.map(render)])].join("\n");
}
