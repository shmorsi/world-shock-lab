// Shapes of the JSON files written by `uv run wsl export site` (wsl/export/site.py).

export type Label = "identified_causal" | "reduced_form" | "model_forecast" | "hypothetical_scenario";

export interface Distribution {
  n: number;
  sd: number;
  min: number;
  max: number;
  p01: number;
  p05: number;
  p50: number;
  p95: number;
  p99: number;
  per: number;
}

export interface Shock {
  id: string;
  name: string;
  family: string;
  column: string;
  per: number;
  unit: string;
  label: Label;
  description: string;
  distribution: Distribution;
}

export interface Outcome {
  id: string;
  name: string;
  units: string;
  country: string;
  asset_class: string;
  policy: string;
}

export interface Irf {
  coef: number[];
  se: number[];
  n: number[];
  cov_partner: number[];
  units: string;
}

export interface Provenance {
  generated_at: string;
  git_sha: string;
  data_built_at: string;
  evaluation_config_sha256: string;
  manifest_sha256: string;
}

export interface Catalog {
  provenance: Provenance;
  horizons: number[];
  shocks: Shock[];
  outcomes: Outcome[];
  irfs: Record<string, Record<string, Irf>>;
}

export interface Country {
  iso_n3: string;
  name: string;
  equity: string;
  fx: string;
  fx_sign: number;
}

export interface Meta {
  provenance: Provenance;
  built_at: string;
  git_sha: string;
  public: boolean;
  countries: Record<string, Country>;
  labels: Record<Label, string>;
}

export type RegimeIrfs = Record<string, Record<string, Record<string, Record<string, Irf>>>>;

export interface CellMetrics {
  n: number;
  mae: number;
  scaled_mae: number;
  oos_r2_vs_zero: number;
  directional_accuracy: number;
  coverage68: number;
  coverage90: number;
  interval_score90_scaled: number;
}

export interface ReplayCell {
  actual: number;
  pred: Record<string, number>;
  half90: Record<string, number>;
}

export interface ReplayEvent {
  event: string;
  family: string;
  ts: string;
  same_day_fomc: boolean;
  cells: Record<string, ReplayCell>;
}

export interface FinalEval {
  generated_at: string;
  protocol_tag: string;
  git_sha: string;
  candidates: { econometric: string; ml: string };
  test_events_primary: number;
  test_years: number[];
  pooled_scaled_mae: Record<string, number>;
  per_cell: Record<string, Record<string, CellMetrics>>;
  difference_ml_minus_econ: { point: number; ci90: [number, number] };
  verdict: string;
  attributions: Record<string, Record<string, number>>;
  replay: ReplayEvent[];
  replay_assets: string[];
}

export interface Implied {
  typical_surprise: Record<string, Record<string, number>>;
  responses: Record<string, Record<string, Record<string, Record<string, { mean: number; sd_across_states: number }>>>>;
}

export interface MlData {
  final: FinalEval;
  implied: Implied;
}

export interface VarLp {
  variables: string[];
  sample: [string, string];
  T: number;
  first_stage_f: number;
  lp_first_stage_f: number;
  bvar: { irf: number[][]; lo90: number[][]; hi90: number[][]; lo68: number[][]; hi68: number[][] };
  lp_iv: Record<string, { rows: { h: number; coef: number; se: number; lo68: number; hi68: number; lo90: number; hi90: number }[] }>;
}

export interface RobustRow {
  family: string;
  outcome: string;
  h: number;
  variants: Record<string, [number, number, number]>;
  verdict: "robust" | "fragile";
}

export interface M4Data {
  var_vs_lp: Record<string, VarLp>;
  canada: { outcome: string; units: string; fed: Record<string, [number, number, number]>; boc: Record<string, [number, number, number]>; boc_rigobon_sack: Record<string, [number, number, number]>; fed_first_stage_f: number }[];
  robustness: RobustRow[];
  state_dependence: { family: string; outcome: string; regime: string; q_joint: number; q_headline: number; p_joint: number; p_headline: number }[];
}

export interface Claim {
  id: string;
  source: string;
  claim: string;
  test: string;
  expected_sign: number;
  mechanical: boolean;
  verdict: string;
  estimate: number;
  band90: [number, number];
  n: number;
}

export interface LiveEntry {
  kind: "forecast" | "score";
  event_id: string;
  bank: string;
  decision_date: string;
  published_at: string;
  collision: boolean;
  proxy: string;
  grid?: { surprise_bp: number; outcome: string; h: number; mean: number; lo68: number; hi68: number; lo90: number; hi90: number }[];
  realized_proxy_bp?: number;
  scores?: { outcome: string; h: number; forecast: number; realized: number; abs_error: number; inside68: boolean; inside90: boolean }[];
}
