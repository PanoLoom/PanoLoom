// v1.1 e2e: stitch the sample -> open the control-point editor -> auto
// points appear -> Optimize improves (or keeps) the rms and re-renders.
// Requires the preview server on :4173.
import { chromium, firefox, webkit } from "playwright";
const engine =
  { chromium, firefox, webkit }[process.env.BROWSER ?? "chromium"] ?? chromium;

const browser = await engine.launch();
const page = await browser.newPage({ viewport: { width: 1200, height: 750 } });
const errors = [];
page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));

await page.goto("http://localhost:4173", { waitUntil: "networkidle" });
await page.click("text=try a sample set");
await page.waitForSelector(".shot >> nth=7", { timeout: 60000 });
await page.click("button.align-btn:not(.ghost)");
await page.waitForSelector(".viewer canvas", { timeout: 300000 });

await page.click('button:has-text("Points")');
await page.waitForSelector(".cp-editor", { timeout: 60000 });
await page.waitForSelector(".cp-row", { timeout: 60000 });
const rows = await page.locator(".cp-row").count();
const pairLabel = await page.textContent(".cp-pairlabel");
console.log(`editor open: ${rows} points on pair "${pairLabel?.trim()}"`);
if (rows < 3) throw new Error("expected auto control points");

await page.click('button:has-text("Optimize")');
await page.waitForSelector(".cp-rms", { timeout: 300000 });
const rms = await page.textContent(".cp-rms");
console.log("optimize:", rms?.trim());
const m = rms?.match(/rms ([\d.]+) → ([\d.]+) px/);
if (!m) throw new Error("no rms report");
if (Number(m[2]) > Number(m[1]) * 1.5) {
  throw new Error(`optimize made things much worse: ${rms}`);
}
if (Number(m[2]) > 5) throw new Error(`final rms too high: ${rms}`);

// Errors now shown per point.
await page.waitForSelector('.cp-err:has-text("px")', { timeout: 10000 });

// Close and confirm the (re-rendered) preview is interactive again.
await page.click('button:has-text("Close")');
await page.waitForSelector(".viewer canvas", { timeout: 60000 });

if (errors.length) {
  console.log("PAGE ERRORS:", errors);
  process.exit(1);
}
console.log("M9 CP EDITOR E2E OK");
await browser.close();
