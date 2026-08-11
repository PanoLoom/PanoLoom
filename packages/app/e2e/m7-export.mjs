// M7 e2e: import JPEGs -> align -> export full-res 360 JPEG with GPano XMP.
// Requires the preview server on :4173. Usage:
//   node e2e/m7-export.mjs [setDir] [exportWidth]
import { chromium } from "playwright";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "../../..");
const setDir =
  process.argv[2] ??
  path.join(root, "tools/testdata/generated/ring_kloppenheim_06");
const exportWidth = process.argv[3] ?? "65535";
if (!existsSync(setDir)) {
  console.log("SKIP: test dataset not present");
  process.exit(0);
}
const jpegs = readdirSync(setDir)
  .filter((f) => /\.(jpe?g)$/i.test(f))
  .sort()
  .map((f) => path.join(setDir, f));

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
const errors = [];
page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
// Force the anchor-download fallback so Playwright can capture the file.
await page.addInitScript(() => {
  delete window.showSaveFilePicker;
});

await page.goto("http://localhost:4173", { waitUntil: "networkidle" });
await page.waitForSelector("text=engine ready", { timeout: 30000 });

await page.setInputFiles('input[type="file"]', jpegs);
await page.waitForSelector(`text=${jpegs.length} shots`, { timeout: 60000 });
console.log("imported", jpegs.length, "shots");

let t0 = Date.now();
await page.click("button.align-btn:not(.ghost)");
await page.waitForSelector(".viewer canvas", { timeout: 600000 });
console.log(`aligned + previewed in ${((Date.now() - t0) / 1000).toFixed(1)}s`);

await page.selectOption("select.export-size", exportWidth);
t0 = Date.now();
const downloadP = page.waitForEvent("download", { timeout: 1800000 });
await page.click("button.align-btn.ghost");
// Progress: poll the status line while the export runs.
const poll = setInterval(async () => {
  try {
    const s = await page.textContent(".bar-status");
    if (s?.includes("export") || s?.includes("encoding")) console.log("  ", s.trim());
  } catch {
    /* page busy */
  }
}, 15000);
const download = await downloadP;
clearInterval(poll);
const out = path.join(here, "m7-export.jpg");
await download.saveAs(out);
console.log(`exported in ${((Date.now() - t0) / 1000).toFixed(1)}s -> ${out}`);

// Validate: JPEG magic, dimensions from SOF marker, GPano XMP present.
const buf = readFileSync(out);
if (buf[0] !== 0xff || buf[1] !== 0xd8) throw new Error("not a JPEG (no SOI)");
let w = 0;
let h = 0;
for (let i = 2; i < buf.length - 9; ) {
  if (buf[i] !== 0xff) break;
  const marker = buf[i + 1];
  if (marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc) {
    h = buf.readUInt16BE(i + 5);
    w = buf.readUInt16BE(i + 7);
    break;
  }
  i += 2 + buf.readUInt16BE(i + 2);
}
const xmp = buf.includes(Buffer.from("GPano:FullPanoWidthPixels"));
const equi = buf.includes(Buffer.from("equirectangular"));
console.log(`jpeg ${w}x${h}, ${(buf.length / 1e6).toFixed(1)} MB, GPano=${xmp}, equirect=${equi}`);
if (!xmp || !equi) throw new Error("GPano XMP missing");
if (w !== 2 * h) throw new Error(`not 2:1 (${w}x${h})`);

if (errors.length) {
  console.log("PAGE ERRORS:", errors);
  process.exit(1);
}
console.log("M7 E2E OK");
await browser.close();
