// Rasterise a dark-theme SVG (from the figure generator) to a README PNG.
// GitHub does not display a repository SVG in a README. Usage:
//   node scripts/readme-png.mjs <svg> <png>
import { readFileSync } from "node:fs";
import { chromium } from "playwright";

const [svgPath, pngPath] = process.argv.slice(2);
if (!svgPath || !pngPath) {
  console.error("usage: node scripts/readme-png.mjs <svg> <png>");
  process.exit(1);
}
const svg = readFileSync(svgPath, "utf8");
const width = Number(svg.match(/\bwidth="(\d+)"/)?.[1]);
const height = Number(svg.match(/\bheight="(\d+)"/)?.[1]);
if (!width || !height) {
  console.error("SVG is missing width and height");
  process.exit(1);
}
const face = (file, family, weight) =>
  `@font-face{font-family:"${family}";font-weight:${weight};font-style:normal;src:url("${file.href}") format("woff2")}`;
const font = (pkg, name) => new URL(`../node_modules/@fontsource/${pkg}/files/${name}`, import.meta.url);
const css = [
  face(font("space-grotesk", "space-grotesk-latin-400-normal.woff2"), "Space Grotesk", 400),
  face(font("space-grotesk", "space-grotesk-latin-500-normal.woff2"), "Space Grotesk", 500),
  face(font("space-grotesk", "space-grotesk-latin-600-normal.woff2"), "Space Grotesk", 600),
  face(font("inter", "inter-latin-400-normal.woff2"), "Inter", 400),
].join("");
const html = `<!doctype html><html><head><style>${css} html,body{margin:0;background:#07080c}</style></head><body>${svg}</body></html>`;

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 2 });
await page.setContent(html, { waitUntil: "load" });
await page.locator("svg").screenshot({ path: pngPath });
await browser.close();
console.log(`wrote ${pngPath} (${width * 2}×${height * 2})`);
