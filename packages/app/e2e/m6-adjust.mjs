// M6 e2e: the live orientation preview (PSV sphereCorrection) must match
// the baked orient() result, and removing a shot resets the workflow.
// Requires the preview server on :4173 (uses the bundled sample).
import { chromium, firefox, webkit } from "playwright";
const engine =
  { chromium, firefox, webkit }[process.env.BROWSER ?? "chromium"] ?? chromium;
import { PNG } from "pngjs";

const browser = await engine.launch();
const page = await browser.newPage({ viewport: { width: 900, height: 600 } });
const errors = [];
page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));

await page.goto("http://localhost:4173", { waitUntil: "networkidle" });
await page.click("text=try a sample set");
await page.waitForSelector(".shot >> nth=7", { timeout: 60000 });
await page.click("button.align-btn:not(.ghost)");
await page.waitForSelector(".viewer canvas", { timeout: 300000 });
await page.waitForTimeout(800);

// Combined rotation: live preview vs baked result.
await page.click('button:has-text("Adjust")');
for (const [axis, v] of [
  ["yaw", 60],
  ["pitch", 20],
  ["roll", 15],
]) {
  await page.locator(`.adjust-row:has-text("${axis}") input`).fill(String(v));
}
await page.waitForTimeout(600);
const live = await page.locator(".viewer canvas").first().screenshot();

await page.click('button:has-text("Apply")');
await page.waitForTimeout(500);
await page.waitForSelector(".viewer canvas", { timeout: 300000 });
await page.waitForTimeout(1200);
const baked = await page.locator(".viewer canvas").first().screenshot();

const a = PNG.sync.read(live);
const b = PNG.sync.read(baked);
let sum = 0;
for (let i = 0; i < a.data.length; i += 4) {
  sum +=
    Math.abs(a.data[i] - b.data[i]) +
    Math.abs(a.data[i + 1] - b.data[i + 1]) +
    Math.abs(a.data[i + 2] - b.data[i + 2]);
}
const mad = sum / ((a.data.length / 4) * 3);
console.log(`live vs baked mean abs diff: ${mad.toFixed(2)}`);
if (mad > 5) throw new Error("live orientation preview diverges from bake");

// Removing a shot invalidates the preview and returns to the loaded state.
await page.click(".shot >> nth=0 >> .shot-remove");
await page.waitForSelector("text=7 shots", { timeout: 10000 });
await page.waitForSelector("text=7 shots on the loom", { timeout: 10000 });
console.log("shot removed, back to loaded state");

if (errors.length) {
  console.log("PAGE ERRORS:", errors);
  process.exit(1);
}
console.log("M6 ADJUST E2E OK");
await browser.close();
