// v1.2 e2e: painting an AVOID mask on a shot changes the composited
// preview (the seam routes around the masked region).
import { chromium, firefox, webkit } from "playwright";
const engine =
  { chromium, firefox, webkit }[process.env.BROWSER ?? "chromium"] ?? chromium;
import { PNG } from "pngjs";

const browser = await engine.launch();
const page = await browser.newPage({ viewport: { width: 1100, height: 700 } });
const errors = [];
page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));

// Screenshot the panorama at four yaws so a seam change anywhere on the
// ring is caught regardless of the default view.
async function sweep() {
  const shots = [];
  for (const yaw of [0, Math.PI / 2, Math.PI, (3 * Math.PI) / 2]) {
    await page.evaluate(
      (y) => window.__psv?.rotate({ yaw: y, pitch: 0 }),
      yaw,
    );
    await page.waitForTimeout(400);
    shots.push(await page.locator(".viewer canvas").first().screenshot());
  }
  return shots;
}

await page.goto("http://localhost:4173", { waitUntil: "networkidle" });
await page.click("text=try a sample set");
await page.waitForSelector(".shot >> nth=7", { timeout: 60000 });
await page.click("button.align-btn:not(.ghost)");
await page.waitForSelector(".viewer canvas", { timeout: 300000 });
await page.waitForTimeout(1000);
const before = await sweep();

// Paint a broad avoid stroke across the middle of the first shot.
await page.click('button:has-text("Mask")');
await page.waitForSelector(".mask-main canvas", { timeout: 30000 });
await page.click('button:has-text("avoid")');
const box = await page.locator(".mask-main canvas").boundingBox();
const cy = box.y + box.height / 2;
await page.mouse.move(box.x + box.width * 0.25, cy);
await page.mouse.down();
for (let i = 0; i <= 20; i++) {
  await page.mouse.move(box.x + box.width * (0.25 + (0.5 * i) / 20), cy + Math.sin(i) * 20);
}
await page.mouse.up();
console.log("painted avoid stroke");

await page.click('button:has-text("Apply")');
await page.waitForSelector('button:has-text("Rendering…")', { timeout: 30000 });
await page.waitForSelector('button:has-text("Rendering…")', {
  state: "detached",
  timeout: 300000,
});
await page.click('button:has-text("Close")');
await page.waitForSelector(".viewer canvas", { timeout: 300000 });
await page.waitForTimeout(1200);
const after = await sweep();

let worst = 0;
for (let k = 0; k < before.length; k++) {
  const a = PNG.sync.read(before[k]);
  const b = PNG.sync.read(after[k]);
  let changed = 0;
  const n = Math.min(a.data.length, b.data.length);
  for (let i = 0; i < n; i += 4) {
    const d =
      Math.abs(a.data[i] - b.data[i]) +
      Math.abs(a.data[i + 1] - b.data[i + 1]) +
      Math.abs(a.data[i + 2] - b.data[i + 2]);
    if (d > 30) changed++;
  }
  worst = Math.max(worst, (100 * changed) / (n / 4));
}
console.log(`max pixels changed by mask across views: ${worst.toFixed(2)}%`);
if (worst < 0.25) throw new Error("mask had no visible effect on the seam");

if (errors.length) {
  console.log("PAGE ERRORS:", errors);
  process.exit(1);
}
console.log("M10 MASK E2E OK");
await browser.close();
