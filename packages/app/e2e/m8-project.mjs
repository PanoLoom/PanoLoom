// M8 e2e: stitch -> Save Project -> fresh page -> open .panoproj ->
// re-select photos -> preview restores WITHOUT re-aligning.
// Requires the preview server on :4173.
import { chromium } from "playwright";
import { existsSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "../../..");
const setDir =
  process.argv[2] ??
  [
    path.join(root, "tools/testdata/generated/ring_kloppenheim_06"),
    path.join(root, "packages/app/public/samples/ring"),
  ].find(existsSync);
if (!setDir || !existsSync(setDir)) {
  console.log("SKIP: test dataset not present");
  process.exit(0);
}
const jpegs = readdirSync(setDir)
  .filter((f) => /\.jpe?g$/i.test(f))
  .sort()
  .map((f) => path.join(setDir, f));

const browser = await chromium.launch();
const page = await browser.newPage();
const errors = [];
page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
await page.addInitScript(() => {
  delete window.showSaveFilePicker;
});

await page.goto("http://localhost:4173", { waitUntil: "networkidle" });
await page.waitForSelector("text=engine ready", { timeout: 30000 });
await page.setInputFiles('input[type="file"]', jpegs);
await page.waitForSelector(`text=${jpegs.length} shots`, { timeout: 60000 });
await page.click("button.align-btn:not(.ghost)");
await page.waitForSelector(".viewer canvas", { timeout: 300000 });
console.log("stitched", jpegs.length, "shots");

// The project name derives from the file names and is click-to-rename;
// it becomes the saved file name.
const derived = await page.textContent(".project-name");
console.log("derived name:", derived?.trim());
if (!derived?.includes("img")) throw new Error(`unexpected name ${derived}`);
await page.click(".project-name");
await page.fill(".project-name-input", "my panorama");
await page.keyboard.press("Enter");

const downloadP = page.waitForEvent("download", { timeout: 30000 });
await page.click('button:has-text("Save Project")');
const download = await downloadP;
if (download.suggestedFilename() !== "my panorama.panoproj") {
  throw new Error(`saved as ${download.suggestedFilename()}`);
}
const projPath = path.join(here, "m8-project.panoproj");
await download.saveAs(projPath);
console.log("saved project as", download.suggestedFilename());

// Fresh page: open the project, re-select the photos, expect instant preview.
await page.goto("http://localhost:4173", { waitUntil: "networkidle" });
await page.waitForSelector("text=engine ready", { timeout: 30000 });
await page.setInputFiles('input[type="file"]', projPath);
await page.waitForSelector(`text=Select this project's ${jpegs.length} photos`, {
  timeout: 10000,
});
console.log("project parsed, selecting photos");

const t0 = Date.now();
await page.setInputFiles('input[type="file"]', jpegs);
await page.waitForSelector(".viewer canvas", { timeout: 120000 });
const dt = (Date.now() - t0) / 1000;
console.log(`restored preview in ${dt.toFixed(1)}s (no align)`);

// The loaded project file's name becomes the project name.
const restored = await page.textContent(".project-name");
if (!restored?.includes("m8-project")) {
  throw new Error(`project name not restored: ${restored}`);
}

if (errors.length) {
  console.log("PAGE ERRORS:", errors);
  process.exit(1);
}
console.log("M8 PROJECT E2E OK");
await browser.close();
