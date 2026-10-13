// The TypeScript engine must reproduce the Python reference (wsl/scenarios/engine.py) on the
// fixtures written by tests/test_scenarios.py.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { compose } from "./engine";

const fixtures = JSON.parse(readFileSync(resolve(__dirname, "../../../tests/fixtures/scenario_cases.json"), "utf8"));

describe("scenario engine parity with Python", () => {
  for (const c of fixtures.cases) {
    it(`sizes ${JSON.stringify(c.sizes)} at h=${c.horizon}`, () => {
      const r = compose(fixtures.catalog, c.sizes, c.horizon);
      expect(r.evidence).toBe(c.evidence);
      expect(r.warnings.length).toBe(c.n_warnings);
      r.outcomes.forEach((o, i) => {
        const p = c.outcomes[i];
        for (const k of ["mean", "se", "lo68", "hi68", "lo90", "hi90"] as const) {
          expect(o[k]).toBeCloseTo(p[k], 10);
        }
      });
    });
  }
});
