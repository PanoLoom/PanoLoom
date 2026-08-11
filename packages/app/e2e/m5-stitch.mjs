// M5 e2e: import ring JPEGs -> Align & Preview -> 360 viewer appears.
// Requires the preview server on :4173 and the local test dataset.
import { chromium, firefox, webkit } from "playwright";
const engine =
  { chromium, firefox, webkit }[process.env.BROWSER ?? "chromium"] ?? chromium;
import { existsSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");
// Prefer the generated test set; fall back to the bundled sample (CI).
const setDir = [
  path.join(root, "tools/testdata/generated/ring_kloppenheim_06"),
  path.join(root, "packages/app/public/samples/ring"),
].find(existsSync);
if (!setDir) {
  console.log("SKIP: no test dataset");
  process.exit(0);
}
const jpegs = readdirSync(setDir)
  .filter((f) => f.endsWith(".jpg"))
  .sort()
  .map((f) => path.join(setDir, f));

const browser = await engine.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
const errors = [];
page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));

await page.goto("http://localhost:4173", { waitUntil: "networkidle" });
await page.waitForSelector("text=engine ready", { timeout: 30000 });

await page.setInputFiles('input[type="file"]', jpegs);
await page.waitForSelector(`text=${jpegs.length} shots`, { timeout: 30000 });
console.log("imported", jpegs.length, "shots");

const t0 = Date.now();
await page.click("button.align-btn");
await page.waitForSelector(".viewer canvas", { timeout: 180000 });
console.log(`aligned + previewed in ${((Date.now() - t0) / 1000).toFixed(1)}s`);

await page.waitForTimeout(1200); // let PSV render a frame
await page.screenshot({
  path: new URL("./m5-stitch.png", import.meta.url).pathname,
});
if (errors.length) {
  console.log("PAGE ERRORS:", errors);
  process.exit(1);
}
console.log("M5 E2E OK");
await browser.close();
