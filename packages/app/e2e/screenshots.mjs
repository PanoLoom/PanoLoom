// Captures the README screenshots into docs/images/ using the bundled
// sample set. Requires the preview server on :4173.
//   node e2e/screenshots.mjs
import { chromium } from "playwright";
import { mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const out = path.resolve(here, "../../../docs/images");
mkdirSync(out, { recursive: true });

const browser = await chromium.launch();
const context = await browser.newContext({
  viewport: { width: 1440, height: 900 },
  deviceScaleFactor: 1.5, // crisp images at README width
});
const page = await context.newPage();

const shot = (name, opts = {}) =>
  page.screenshot({
    path: path.join(out, name),
    ...(name.endsWith(".jpg") ? { type: "jpeg", quality: 86 } : {}),
    ...opts,
  });

await page.goto("http://localhost:4173", { waitUntil: "networkidle" });
await page.click("text=try a sample set");
await page.waitForSelector(".shot >> nth=7", { timeout: 60000 });
await page.click("button.align-btn:not(.ghost)");
await page.waitForSelector(".viewer canvas", { timeout: 300000 });
await page.waitForTimeout(1500);

// 1. Hero: the stitched panorama in the 360° viewer.
await shot("hero.jpg");
console.log("hero.jpg");

// 2. Adjust panel with a live tilt.
await page.click('button:has-text("Adjust")');
await page.locator('.adjust-row:has-text("roll") input').fill("8");
await page.waitForTimeout(600);
await shot("adjust.jpg");
await page.locator('.adjust-row:has-text("roll") input').fill("0");
await page.click('button:has-text("Adjust")'); // close
console.log("adjust.jpg");

// 3. Control-point editor after an optimize (errors populated).
await page.click('button:has-text("Points")');
await page.waitForSelector(".cp-row", { timeout: 120000 });
await page.click('button:has-text("Optimize")');
await page.waitForSelector(".cp-rms", { timeout: 300000 });
await page.waitForTimeout(800);
await shot("points.jpg");
await page.click('button:has-text("Close")');
console.log("points.jpg");

// 4. Mask editor with painted strokes.
await page.waitForSelector(".viewer canvas", { timeout: 60000 });
await page.click('button:has-text("Mask")');
await page.waitForSelector(".mask-main canvas", { timeout: 30000 });
await page.waitForTimeout(1500);
await page.click('button:has-text("avoid")');
const box = await page.locator(".mask-main canvas").boundingBox();
await page.mouse.move(box.x + box.width * 0.3, box.y + box.height * 0.3);
await page.mouse.down();
await page.mouse.move(box.x + box.width * 0.62, box.y + box.height * 0.34, {
  steps: 25,
});
await page.mouse.up();
await page.click('button:has-text("prefer")');
await page.mouse.move(box.x + box.width * 0.4, box.y + box.height * 0.66);
await page.mouse.down();
await page.mouse.move(box.x + box.width * 0.58, box.y + box.height * 0.64, {
  steps: 15,
});
await page.mouse.up();
await page.waitForTimeout(400);
await shot("mask.jpg");
await page.click('button:has-text("Close")');
console.log("mask.jpg");

// 5. Restore banner after a reload (autosave needs a moment first).
await page.waitForTimeout(2500);
await page.reload({ waitUntil: "networkidle" });
await page.waitForSelector("text=Restore last session", { timeout: 30000 });
await shot("restore.png", {
  clip: { x: 300, y: 60, width: 840, height: 180 },
});
console.log("restore.png");

await browser.close();
console.log("screenshots ->", out);
