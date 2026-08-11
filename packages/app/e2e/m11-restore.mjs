// Session restore e2e: stitch the sample -> reload the tab -> "Restore
// last session" brings the whole project back without re-aligning.
import { chromium, firefox, webkit } from "playwright";
const engine =
  { chromium, firefox, webkit }[process.env.BROWSER ?? "chromium"] ?? chromium;

const browser = await engine.launch();
const context = await browser.newContext(); // one context = shared IndexedDB
const page = await context.newPage();
const errors = [];
page.on("pageerror", (e) => {
  errors.push(`pageerror: ${e.message}`);
  console.log("pageerror:", e.message);
});

await page.goto("http://localhost:4173", { waitUntil: "networkidle" });
await page.click("text=try a sample set");
await page.waitForSelector(".shot >> nth=7", { timeout: 60000 });
await page.click("button.align-btn:not(.ghost)");
await page.waitForSelector(".viewer canvas", { timeout: 300000 });
// Give the debounced autosave time to write.
await page.waitForTimeout(2500);
console.log("stitched + autosaved");

await page.reload({ waitUntil: "networkidle" });
await page.waitForSelector("text=Restore last session", { timeout: 30000 });
const offer = await page.textContent(".resume-note");
console.log("offer:", offer?.replace(/\s+/g, " ").trim());
if (!offer?.includes("8 shots") || !offer.includes("aligned")) {
  throw new Error("restore offer incomplete");
}

const t0 = Date.now();
await page.click('button:has-text("Restore")');
await page.waitForSelector(".viewer canvas", { timeout: 120000 });
console.log(
  `restored in ${((Date.now() - t0) / 1000).toFixed(1)}s (no align);`,
  "name:",
  (await page.textContent(".project-name"))?.trim(),
);

if (errors.length) {
  console.log("PAGE ERRORS:", errors);
  process.exit(1);
}
console.log("M11 RESTORE E2E OK");
await browser.close();
