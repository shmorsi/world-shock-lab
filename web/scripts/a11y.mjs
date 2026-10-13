// Accessibility check: runs axe-core (WCAG 2.1 A/AA rules) on every view, in light and dark
// themes, against a running build (default: `npx vite preview` on port 4173).
// Usage: node scripts/a11y.mjs [base-url]
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { chromium } from "playwright";

const require = createRequire(import.meta.url);
const axeSource = readFileSync(require.resolve("axe-core/axe.min.js"), "utf8");
const base = process.argv[2] ?? "http://localhost:4173/";
const routes = ["", "map", "explorer", "scenario", "replay", "brief", "method", "track"];

const browser = await chromium.launch();
let failures = 0;
for (const scheme of ["light", "dark"]) {
  const page = await (await browser.newContext({ colorScheme: scheme })).newPage();
  for (const r of routes) {
    await page.goto(`${base}#/${r}`);
    await page.waitForLoadState("networkidle");
    await page.addScriptTag({ content: axeSource });
    const res = await page.evaluate(() =>
      // eslint-disable-next-line no-undef
      axe.run(document, { runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"] } }),
    );
    for (const v of res.violations) {
      failures += v.nodes.length;
      console.log(`${scheme} #/${r}: ${v.id} (${v.impact}) x${v.nodes.length}: ${v.help}`);
      for (const n of v.nodes.slice(0, 3)) console.log(`    ${n.target.join(" ")}  ${n.failureSummary?.split("\n")[1] ?? ""}`);
    }
  }
}
await browser.close();
console.log(failures ? `${failures} accessibility violations` : "No WCAG 2.1 A/AA violations found by axe-core.");
process.exit(failures ? 1 : 0);
