// Scenario composition. A port of wsl/scenarios/engine.py, checked against fixtures that the
// Python implementation writes (engine.test.ts). Keep the two in step.
import type { Catalog, Label } from "../types";

export const LABEL_ORDER: Label[] = ["identified_causal", "reduced_form", "model_forecast", "hypothetical_scenario"];
export const Z68 = 0.9944579;
export const Z90 = 1.6448536;
/** Coverage, in per cent, of the bands built with Z90: the one fixed constant the brief may print. */
export const BAND_LEVEL = 90;
export const LINEARITY =
  "Combined by adding each shock's estimated response. This assumes effects are linear and do not interact; large or simultaneous shocks may behave differently.";

export interface OutcomeResult {
  id: string;
  mean: number;
  se: number;
  lo68: number;
  hi68: number;
  lo90: number;
  hi90: number;
}

export interface ScenarioResult {
  horizon: number;
  evidence: Label;
  warnings: string[];
  outcomes: OutcomeResult[];
}

type CatalogLike = Pick<Catalog, "shocks" | "outcomes" | "irfs">;

function gFormat(v: number): string {
  // Python's "{:g}" for the sizes users enter (integers and short decimals).
  return Number.isInteger(v) ? String(v) : String(Number(v.toPrecision(6)));
}

export function compose(catalog: CatalogLike, sizes: Record<string, number>, horizon: number): ScenarioResult {
  const shocks = new Map(catalog.shocks.map((s) => [s.id, s]));
  const used = Object.entries(sizes).filter(([, v]) => v !== 0);
  if (used.length === 0) return { horizon, evidence: "hypothetical_scenario", warnings: [], outcomes: [] };
  const evidence = used
    .map(([k]) => shocks.get(k)!.label)
    .reduce((a, b) => (LABEL_ORDER.indexOf(b) > LABEL_ORDER.indexOf(a) ? b : a));
  const warnings = [LINEARITY];
  for (const [k, v] of used) {
    const s = shocks.get(k)!;
    const { min, max } = s.distribution;
    if (v > max || v < min) {
      warnings.push(
        `${s.name}: a size of ${gFormat(v)} ${s.unit} is outside the range ever observed (${min.toFixed(1)} to ${max.toFixed(1)} ${s.unit}); the estimate is an extrapolation.`,
      );
    }
  }
  const usedMap = new Map(used);
  const outcomes = catalog.outcomes.map((o) => {
    let mean = 0;
    let variance = 0;
    for (const [k, v] of used) {
      const r = catalog.irfs[k][o.id];
      const w = v / shocks.get(k)!.per;
      mean += w * r.coef[horizon];
      variance += (w * r.se[horizon]) ** 2;
    }
    if (usedMap.has("fed_mp") && usedMap.has("fed_cbi")) {
      const wa = usedMap.get("fed_cbi")! / shocks.get("fed_cbi")!.per;
      const wb = usedMap.get("fed_mp")! / shocks.get("fed_mp")!.per;
      variance += 2 * wa * wb * catalog.irfs["fed_cbi"][o.id].cov_partner[horizon];
    }
    const se = Math.sqrt(Math.max(variance, 0));
    return { id: o.id, mean, se, lo68: mean - Z68 * se, hi68: mean + Z68 * se, lo90: mean - Z90 * se, hi90: mean + Z90 * se };
  });
  return { horizon, evidence, warnings, outcomes };
}

/** Where a size sits in the shock's historical distribution, in standard deviations. */
export function inSd(catalog: CatalogLike, shockId: string, size: number): number {
  const s = catalog.shocks.find((x) => x.id === shockId)!;
  return s.unit === "sd" ? size : size / s.distribution.sd;
}
