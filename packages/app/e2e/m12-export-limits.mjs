// M12 e2e: the export size ceiling reaches the UI, and exporting at it works.
//
// A large panorama cannot be composed at full resolution in a 4 GB address
// space (137 shots at 12MP wants 50113x25057 = 4.7 GB). The engine reports
// the widest canvas it can hold; this checks that number reaches the
// dropdown and that choosing it still produces a valid JPEG.
import { chromium, firefox, webkit } from "playwright";
const engine =
  { chromium, firefox, webkit }[process.env.BROWSER ?? "chromium"] ?? chromium;
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "../../..");
const setDir = [
  path.join(root, "tools/testdata/generated/ring_kloppenheim_06"),
  path.join(root, "packages/app/public/samples/ring"),
].find(existsSync);
if (!setDir) {
  console.log("SKIP: no test dataset");
  process.exit(0);
}
const jpegs = readdirSync(setDir)
  .filter((f) => /\.jpe?g$/i.test(f))
  .sort()
  .map((f) => path.join(setDir, f));

const browser = await engine.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
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
await page.waitForSelector(".viewer canvas", { timeout: 600000 });

const options = await page
  .locator("select.export-size option")
  .evaluateAll((els) => els.map((e) => ({ value: Number(e.value), text: e.textContent.trim() })));
console.log("export sizes:", options.map((o) => o.text).join(" | "));

const ceiling = options.find((o) => /largest/i.test(o.text));
if (!ceiling) throw new Error("engine export ceiling never reached the dropdown");
if (!(ceiling.value > 4096 && ceiling.value < 65535))
  throw new Error(`implausible ceiling ${ceiling.value}`);
console.log("ceiling option:", ceiling.value, "px");

// Exporting at the ceiling must still yield a real JPEG. The sample's own
// native resolution is far below it, so this caps at native and stays fast.
await page.selectOption("select.export-size", String(ceiling.value));
const downloadP = page.waitForEvent("download", { timeout: 900000 });
await page.click('button:has-text("Export JPEG")');
const out = path.join(here, "m12-export-limits.jpg");
await (await downloadP).saveAs(out);
const buf = readFileSync(out);
if (buf[0] !== 0xff || buf[1] !== 0xd8) throw new Error("not a JPEG");
console.log(`exported at ceiling -> ${(buf.length / 1e6).toFixed(2)} MB`);

const visible = await page.locator(".error").allTextContents().catch(() => []);
if (visible.filter(Boolean).length) throw new Error(`UI error: ${visible}`);
if (errors.length) {
  console.log("PAGE ERRORS:", errors);
  process.exit(1);
}
console.log("M12 E2E OK");
await browser.close();
