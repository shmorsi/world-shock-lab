import { useState } from "react";
import { useAll, useData } from "../data";
import { signed } from "../format";
import type { Catalog, M4Data, MlData, RegimeIrfs } from "../types";
import { Chip, Loading, Panel, Seg, Select } from "../components/ui";
import { FanChart, type Series } from "../components/FanChart";
import { Z68, Z90 } from "../engine/engine";

type D = { catalog: Catalog; ml: MlData };
const NO_REGIMES: RegimeIrfs = {};
const ML_ASSET: Record<string, string> = {
  US2Y: "US2Y", CA2Y: "CA2Y", USDCAD: "USDCAD", "SPY.tr": "SPY", "TSX.tr": "TSX", CA_BANKS_EW6: "CA_BANKS_EW6", "XEG.tr": "XEG", WTI: "WTI",
};
const ML_FAMILY: Record<string, string> = { fed_pc1: "fomc", boc: "boc", opec: "opec" };
const GROUP_COLORS = ["var(--s2)", "var(--s3)", "var(--neg)"];
const REGIME_TEXT: Record<string, string> = { cycle: "Policy cycle", high_inflation: "Inflation", high_vix: "VIX" };
const FAMILY_OF: Record<string, string> = { fed_mp: "fomc", fed_cbi: "fomc", fed_pc1: "fomc", boc: "boc", opec: "opec" };

function fan(name: string, color: string, coef: number[], se: number[], bands = true): Series {
  return {
    name,
    color,
    mean: coef,
    lo90: coef.map((v, i) => v - Z90 * se[i]),
    hi90: coef.map((v, i) => v + Z90 * se[i]),
    ...(bands ? { lo68: coef.map((v, i) => v - Z68 * se[i]), hi68: coef.map((v, i) => v + Z68 * se[i]) } : {}),
  };
}

function Daily({ d, shock, setShock }: { d: D; shock: string; setShock: (s: string) => void }) {
  const { catalog, ml } = d;
  // Secondary files, fetched once the main chart is on screen.
  const regimes = { regimes: useData<{ regimes: RegimeIrfs }>("regimes").data?.regimes ?? NO_REGIMES };
  const robustness = useData<M4Data>("m4").data?.robustness ?? [];
  const [outcome, setOutcome] = useState("CA2Y");
  const [regime, setRegime] = useState("none");
  const s = catalog.shocks.find((x) => x.id === shock)!;
  const o = catalog.outcomes.find((x) => x.id === outcome)!;
  const irf = catalog.irfs[shock][outcome];
  const series: Series[] = [fan("All announcements", "var(--s1)", irf.coef, irf.se)];
  const reg = regimes.regimes[shock]?.[regime];
  if (regime !== "none" && reg) {
    Object.entries(reg).forEach(([g, outs], i) => {
      const r = outs[outcome];
      if (r) series.push({ ...fan(`${g} (N ${r.n[0]})`, GROUP_COLORS[i % 3], r.coef, r.se, false), lo68: undefined, hi68: undefined });
    });
  }
  const fam = ML_FAMILY[shock];
  const asset = ML_ASSET[outcome];
  const imp = fam && asset ? ml.implied.responses[fam]?.[asset] : undefined;
  if (imp) {
    const pts = (m: string) => [
      { h: 0, v: imp.h1[m].mean },
      { h: 4, v: imp.h5[m].mean },
    ];
    series.push({ name: "MLP (model-implied)", color: "var(--s2)", mean: [], points: pts("mlp") });
    series.push({ name: "Gradient boosting (model-implied)", color: "var(--s3)", mean: [], points: pts("gbm") });
  }
  const regimeOptions = Object.keys(regimes.regimes[shock] ?? {});
  const groups = Array.from(new Set(catalog.outcomes.map((x) => x.asset_class)));
  const rob = robustness.filter((r) => r.family === FAMILY_OF[shock] && r.outcome === outcome && shock !== "fed_pc1" && shock !== "fed_cbi");
  return (
    <>
      <div className="controls">
        <Select id="ex-shock" label="Shock" value={shock} onChange={(v) => { setShock(v); if (!regimes.regimes[v]?.[regime]) setRegime("none"); }}>
          {catalog.shocks.map((x) => (
            <option key={x.id} value={x.id}>
              {x.name} ({x.unit === "sd" ? "per 1 s.d." : `per ${x.per} ${x.unit}`})
            </option>
          ))}
        </Select>
        <Select id="ex-out" label="Asset" value={outcome} onChange={setOutcome}>
          {groups.map((g) => (
            <optgroup key={g} label={g.replace(/_/g, " ")}>
              {catalog.outcomes.filter((x) => x.asset_class === g).map((x) => (
                <option key={x.id} value={x.id}>
                  {x.name}
                </option>
              ))}
            </optgroup>
          ))}
        </Select>
        <Seg
          label="Compare regimes"
          value={regime}
          onChange={setRegime}
          options={[{ value: "none", text: "None" }, ...regimeOptions.map((r) => ({ value: r, text: REGIME_TEXT[r] ?? r }))]}
          disabled={(v) => v !== "none" && !regimes.regimes[shock]?.[v]?.[Object.keys(regimes.regimes[shock][v])[0]]?.[outcome]}
        />
      </div>
      <p className="muted">
        {s.description} {s.family === "index"
          ? `Local projection on every trading day (N = ${irf.n[0]}), with Newey–West bands.`
          : `Event-sample local projection over N = ${irf.n[0]} announcements, with HC3 bands.`} {regime !== "none" && "Regime lines are separate estimates within each regime; the pre-registered tests of whether they differ found nothing that survives false-discovery control (Methodology)."}
        {imp && " Diamonds: what the fitted ML models imply for a typical surprise of this kind, at the first and fifth trading day (labelled model forecast)."}
      </p>
      <div className="legend" aria-hidden="true">
        {series.map((x) => (
          <span key={x.name}>
            <i style={{ background: x.color }} />
            {x.name}
          </span>
        ))}
      </div>
      <FanChart series={series} horizons={catalog.horizons} units={o.units === "bp" ? "basis points" : "% (100 × log change)"} summary={`Response of ${o.name} to ${s.name} over 20 trading days: ${signed(irf.coef[0])} on day 0, ${signed(irf.coef[20])} on day 20.`} />
      {rob.length > 0 && (
        <p className="muted">
          Robustness (pre-registered variants):{" "}
          {rob.map((r) => `day ${r.h} ${r.verdict}`).join("; ")}.
        </p>
      )}
      {imp && <Chip label="model_forecast" />}
    </>
  );
}

function Monthly() {
  const m4 = useData<M4Data>("m4");
  if (!m4.data) return <Loading error={m4.error} />;
  return <MonthlyChart m4={m4.data} />;
}

function MonthlyChart({ m4 }: { m4: M4Data }) {
  const instruments = Object.keys(m4.var_vs_lp);
  const [inst, setInst] = useState(instruments[0]);
  const v = m4.var_vs_lp[inst];
  const [variable, setVariable] = useState(v.variables[1]);
  const i = v.variables.indexOf(variable);
  const hs = v.bvar.irf.map((_, h) => h);
  const lp = v.lp_iv[variable].rows;
  const series: Series[] = [
    { name: "Proxy BVAR", color: "var(--s1)", mean: v.bvar.irf.map((r) => r[i]), lo90: v.bvar.lo90.map((r) => r[i]), hi90: v.bvar.hi90.map((r) => r[i]), lo68: v.bvar.lo68.map((r) => r[i]), hi68: v.bvar.hi68.map((r) => r[i]) },
    { name: "LP-IV", color: "var(--s2)", mean: lp.map((r) => r.coef), lo90: lp.map((r) => r.lo90), hi90: lp.map((r) => r.hi90) },
  ];
  const weak = Math.max(v.first_stage_f, v.lp_first_stage_f) < 10;
  return (
    <>
      <div className="controls">
        <Seg label="Instrument" value={inst} onChange={(x) => setInst(x)} options={instruments.map((x) => ({ value: x, text: x === "BS_MPS_ORTH" ? "Bauer–Swanson (orthogonalized)" : "Jarociński–Karadi MP" }))} />
        <Select id="ex-var" label="Variable" value={variable} onChange={setVariable}>
          {v.variables.map((x) => (
            <option key={x} value={x}>
              {x}
            </option>
          ))}
        </Select>
      </div>
      {weak && (
        <div className="warn" role="note">
          <strong>Weak instrument.</strong> First-stage F {v.first_stage_f.toFixed(1)} (VAR) and {v.lp_first_stage_f.toFixed(1)} (LP-IV), below the usual
          threshold of 10. Read these monthly responses as suggestive; the weak-instrument-robust sets are in docs/RESULTS.md.
        </div>
      )}
      <div className="legend" aria-hidden="true">
        <span>
          <i style={{ background: "var(--s1)" }} />
          Proxy BVAR (bands from a moving-block bootstrap; too narrow in simulation)
        </span>
        <span>
          <i style={{ background: "var(--s2)" }} />
          LP-IV (dotted 90% band)
        </span>
      </div>
      <FanChart series={series} horizons={hs} xLabel="Months after the shock" units={variable.endsWith("Y") ? "percentage points" : "% (100 × log)"} summary={`Monthly response of ${variable} to a US policy shock raising the 1-year yield by 25 bp; BVAR and LP-IV estimates over 24 months.`} />
      <p className="muted">
        Sample {v.sample[0]} to {v.sample[1]} ({v.T} months). Shock: US monetary policy raising the 1-year yield 25 bp on impact.
      </p>
    </>
  );
}

export default function ExplorerView() {
  const d = useAll<D>(["catalog", "ml"]);
  const [tab, setTab] = useState<"daily" | "monthly">("daily");
  const [shock, setShock] = useState("fed_mp");
  if (!d || typeof d === "string") return <Loading error={typeof d === "string" ? d : null} />;
  const label = tab === "daily" ? d.catalog.shocks.find((x) => x.id === shock)!.label : "identified_causal";
  return (
    <Panel eyebrow="View 2 · Impulse Response Explorer" title="How a shock propagates, by method and regime" chip={<Chip label={label} />} id="ex-h">
      <Seg label="Data" value={tab} onChange={setTab} options={[{ value: "daily", text: "Daily markets" }, { value: "monthly", text: "Monthly macro (VAR vs LP)" }]} />
      {tab === "daily" ? <Daily d={d} shock={shock} setShock={setShock} /> : <Monthly />}
    </Panel>
  );
}
