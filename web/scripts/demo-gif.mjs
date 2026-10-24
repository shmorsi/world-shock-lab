// docs/img/demo.gif: a scripted walk through the built site (default: `npx vite preview`).
// Usage: node scripts/demo-gif.mjs [base-url]
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import gifenc from "gifenc";
import { PNG } from "pngjs";
import { chromium } from "playwright";

const { GIFEncoder, quantize, applyPalette } = gifenc;
const base = process.argv[2] ?? "http://localhost:4173/";
const out = resolve(dirname(fileURLToPath(import.meta.url)), "../../docs/img/demo.gif");
const W = 1100;
const H = 680;

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 1, colorScheme: "dark" });
const gif = GIFEncoder();
let frames = 0;

async function shot(holdMs) {
  const png = PNG.sync.read(await page.screenshot({ type: "png" }));
  const palette = quantize(png.data, 256);
  gif.writeFrame(applyPalette(png.data, palette), png.width, png.height, { palette, delay: holdMs });
  frames++;
}

async function go(route, waitFor) {
  await page.goto(`${base}#/${route}`);
  await page.waitForSelector(waitFor);
  await page.waitForTimeout(400);
}

// 1. The landing page and its three findings.
await go("", "text=Finding 1");
await shot(2600);
await page.mouse.wheel(0, 330);
await page.waitForTimeout(300);
await shot(2600);

// 2. Build a scenario: a hawkish Fed surprise plus an oil supply cut.
await go("scenario", "text=Build a scenario");
await page.evaluate(() => localStorage.clear());
await go("scenario", "text=Build a scenario");
await shot(1400);
await page.getByRole("button", { name: "Fed hike + oil supply cut" }).click();
await page.waitForTimeout(300);
await shot(2000);
// Move the horizon from day 5 to day 10, one step at a time.
const horizon = page.locator("#sc-h-range");
await horizon.focus();
for (let i = 0; i < 5; i++) {
  await page.keyboard.press("ArrowRight");
  await page.waitForTimeout(80);
  await shot(220);
}
await shot(1200);
// 3. The Canada dashboard.
await page.evaluate(() => {
  document.getElementById("ca-h")?.scrollIntoView({ block: "start" });
  window.scrollBy(0, -80); // clear the sticky top bar
});
await page.waitForTimeout(300);
await shot(3200);

// 4. The world map: oil supply news, currencies.
await go("map", "svg path");
await page.selectOption("#map-shock", "opec");
await page.waitForTimeout(300);
await shot(1600);
await page.getByRole("button", { name: "Currency" }).click();
await page.waitForTimeout(300);
await shot(2400);

// 5. The strategist brief for the scenario built above.
await go("brief", "text=One-page brief");
await shot(3400);

gif.finish();
mkdirSync(dirname(out), { recursive: true });
writeFileSync(out, gif.bytes());
await browser.close();
console.log(`wrote ${out} (${frames} frames, ${(gif.bytes().length / 1024 / 1024).toFixed(1)} MB)`);
