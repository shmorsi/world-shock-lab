// Number formatting used across the app (one place, so tables, charts and text agree).
export function signed(v: number, digits = 2): string {
  if (!Number.isFinite(v)) return "–";
  const s = Math.abs(v).toFixed(digits);
  return v > 0 ? `+${s}` : v < 0 ? `−${s}` : s;
}

export function plain(v: number, digits = 2): string {
  return Number.isFinite(v) ? v.toFixed(digits).replace("-", "−") : "–";
}

export function band(lo: number, hi: number, digits = 2): string {
  return `${signed(lo, digits)} to ${signed(hi, digits)}`;
}

export function pct(v: number, digits = 0): string {
  return `${(100 * v).toFixed(digits)}%`;
}

export const LABEL_TEXT: Record<string, string> = {
  identified_causal: "Identified causal",
  reduced_form: "Reduced-form association",
  model_forecast: "Model forecast",
  hypothetical_scenario: "Hypothetical scenario",
};

/** A shock's name inside a sentence: lower-case unless it starts with a proper noun. */
export function shockNoun(name: string): string {
  return /^(Fed|Bank of Canada|OPEC)\b/.test(name) ? name : name.charAt(0).toLowerCase() + name.slice(1);
}

/** "20261012T233454Z" → "2026-10-12 23:34 UTC". */
export function stamp(ts: string): string {
  const m = ts.match(/^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/);
  return m ? `${m[1]}-${m[2]}-${m[3]} ${m[4]}:${m[5]} UTC` : ts;
}
