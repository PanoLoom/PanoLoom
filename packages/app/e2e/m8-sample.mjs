// M8 e2e: "try a sample set" -> import bundled shots -> align -> viewer.
// Requires the preview server on :4173 (samples ship in public/).
import { chromium, firefox, webkit } from "playwright";
const engine =
  { chromium, firefox, webkit }[process.env.BROWSER ?? "chromium"] ?? chromium;

const browser = await engine.launch();
const page = await browser.newPage();
const errors = [];
page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));

await page.goto("http://localhost:4173", { waitUntil: "networkidle" });
await page.waitForSelector("text=engine ready", { timeout: 30000 });

await page.click("text=try a sample set");
await page.waitForSelector("text=8 shots", { timeout: 60000 });
console.log("sample imported");

await page.click("button.align-btn:not(.ghost)");
await page.waitForSelector(".viewer canvas", { timeout: 300000 });
console.log("sample stitched");

if (errors.length) {
  console.log("PAGE ERRORS:", errors);
  process.exit(1);
}
console.log("M8 SAMPLE E2E OK");
await browser.close();
