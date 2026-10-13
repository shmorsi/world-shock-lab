// Load-time check: median time until each view's main content is on screen, unthrottled and on
// a slow phone profile (1.6 Mbps, 150 ms round trip, CPU slowed 4x: Lighthouse's "slow 4G").
// Runs against a built site (default: `npx vite preview` on port 4173).
// Usage: node scripts/measure.mjs [base-url] [--write]   (--write saves results/web_perf.json)
import { writeFileSync } from "node:fs";
import { chromium } from "playwright";

const args = process.argv.slice(2);
const base = args.find((a) => !a.startsWith("--")) ?? "http://localhost:4173/";
const routes = [
  ["", "text=Finding 1"],
  ["map", "svg path"],
  ["explorer", "text=Show this chart as a table"],
  ["scenario", "text=Canada dashboard"],
  ["replay", "text=Verdict"],
];
const SLOW = { latency: 150, downloadThroughput: (1.6 * 1024 * 1024) / 8, uploadThroughput: (750 * 1024) / 8 };

const browser = await chromium.launch();
const rows = [];
const record = [];
let worst = 0;
for (const [route, selector] of routes) {
  const row = [`#/${route}`];
  for (const slow of [false, true]) {
    const times = [];
    for (let k = 0; k < 3; k++) {
      const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true });
      const page = await ctx.newPage();
      if (slow) {
        const cdp = await ctx.newCDPSession(page);
        await cdp.send("Network.enable");
        await cdp.send("Network.emulateNetworkConditions", { offline: false, ...SLOW });
        await cdp.send("Emulation.setCPUThrottlingRate", { rate: 4 });
      }
      await page.goto(base + "#/" + route);
      await page.waitForSelector(selector, { timeout: 30000 });
      times.push(await page.evaluate(() => performance.now()));
      await ctx.close();
    }
    const median = times.sort((a, b) => a - b)[1] / 1000;
    if (slow) worst = Math.max(worst, median);
    row.push(`${median.toFixed(2)} s`);
    record.push({ view: `#/${route}`, profile: slow ? "slow_4g_4x_cpu" : "unthrottled", median_s: +median.toFixed(3) });
  }
  rows.push(row);
}
await browser.close();
console.log("| View | Unthrottled | Slow 4G, 4x CPU |\n|---|---|---|");
for (const r of rows) console.log(`| ${r.join(" | ")} |`);
console.log(`\nSlowest view on the slow profile: ${worst.toFixed(2)} s (target: under 3 s).`);
if (args.includes("--write")) {
  const out = new URL("../../results/web_perf.json", import.meta.url);
  const doc = { measured_at: new Date().toISOString().slice(0, 19) + "Z", profile_slow: "1.6 Mbps down, 750 kbps up, 150 ms RTT, CPU 4x slower", viewport: "390x844 mobile", runs_per_cell: 3, target_s: 3, slowest_slow_s: +worst.toFixed(3), rows: record };
  writeFileSync(out, JSON.stringify(doc, null, 1) + "\n");
  console.log(`wrote ${out.pathname}`);
}
process.exit(worst < 3 ? 0 : 1);
