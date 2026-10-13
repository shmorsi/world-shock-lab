// The brief cannot contain a number that is not in the data. Checked on hundreds of random
// scenarios over the real catalog and robustness results.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import type { Catalog, M4Data } from "../types";
import { BAND_LEVEL, compose, inSd } from "../engine/engine";
import { buildBrief, briefText, type Part } from "./brief";

const data = (f: string) => JSON.parse(readFileSync(resolve(__dirname, `../../public/data/${f}`), "utf8"));
const catalog: Catalog = data("catalog.json");
const m4: M4Data = data("m4.json");

function rng(seed: number) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32);
}

function randomScenario(r: () => number): Record<string, number> {
  const sizes: Record<string, number> = {};
  for (const s of catalog.shocks) {
    if (r() < 0.35) {
      const v = s.distribution.min + r() * (s.distribution.max - s.distribution.min) * 1.6;
      sizes[s.id] = Math.round(v * 10) / 10;
    }
  }
  return sizes;
}

/** Every value a number in the brief may legitimately come from. */
function allowedValues(sizes: Record<string, number>, h: number, result: ReturnType<typeof compose>): number[] {
  const vals: number[] = [BAND_LEVEL, h, result.outcomes.length, result.outcomes.filter((o) => o.lo90 > 0 || o.hi90 < 0).length];
  for (const [k, v] of Object.entries(sizes)) vals.push(v, inSd(catalog, k, v));
  for (const o of result.outcomes) vals.push(o.mean, o.lo90, o.hi90);
  for (const r of m4.robustness) vals.push(r.h);
  return vals;
}

describe("strategist brief", () => {
  it("never writes a digit outside data-derived numbers and names", () => {
    const r = rng(42);
    let checked = 0;
    for (let i = 0; i < 300; i++) {
      const sizes = randomScenario(r);
      const h = Math.floor(r() * 21);
      const result = compose(catalog, sizes, h);
      const brief = buildBrief(catalog, sizes, result, m4.robustness);
      const parts: Part[] = brief.sections.flatMap((s) => s.paragraphs.flat());
      for (const p of parts) {
        if (p.kind === "text") expect(p.text).not.toMatch(/\d/); // templates have no numbers
      }
      const allowed = allowedValues(sizes, h, result);
      for (const p of parts) {
        if (p.kind !== "num") continue;
        expect(allowed.some((v) => v === p.value)).toBe(true); // came from the data
        const shown = Number(p.text.replace("−", "-").replace("+", ""));
        const digits = (p.text.split(".")[1] ?? "").length;
        expect(Math.abs(shown - p.value)).toBeLessThanOrEqual(0.5 * 10 ** -digits + 1e-12);
        checked++;
      }
      // Warnings come from the engine; their numbers must be the sizes or observed ranges.
      const ranges = catalog.shocks.flatMap((s) => [s.distribution.min, s.distribution.max]);
      for (const p of parts) {
        if (p.kind !== "name" || !p.text.includes("outside the range")) continue;
        for (const m of p.text.match(/-?\d+(?:\.\d+)?/g) ?? []) {
          const v = Number(m);
          const ok = Object.values(sizes).some((s) => Math.abs(s - v) < 1e-9) ||
            ranges.some((x) => Math.abs(Number(x.toFixed(1)) - v) < 1e-9);
          expect(ok).toBe(true);
        }
      }
      expect(briefText(brief).length).toBeGreaterThan(0);
    }
    expect(checked).toBeGreaterThan(1000);
  });

  it("explains an empty scenario instead of inventing one", () => {
    const b = buildBrief(catalog, {}, compose(catalog, {}, 0), m4.robustness);
    expect(briefText(b)).toMatch(/Set at least one shock/);
  });
});
