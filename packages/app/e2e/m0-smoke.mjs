import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 720, height: 560 } });
const errors = [];
page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
page.on("console", (m) => m.type() === "error" && errors.push(`console: ${m.text()}`));

await page.goto("http://localhost:4173", { waitUntil: "networkidle" });
await page.waitForSelector("li", { timeout: 15000 });

const checks = await page.$$eval("li", (lis) =>
  lis.map((li) => li.textContent?.trim().replace(/\s+/g, " ")),
);
console.log("CHECKS:");
for (const c of checks) console.log(" ", c);
console.log("crossOriginIsolated (main thread):", await page.evaluate(() => crossOriginIsolated));
if (errors.length) console.log("PAGE ERRORS:", errors);

await page.screenshot({ path: new URL("./m0-smoke.png", import.meta.url).pathname });
await browser.close();

const allGreen = checks.length >= 5 && checks.every((c) => c?.startsWith("✓"));
console.log(allGreen ? "BROWSER SMOKE OK" : "BROWSER SMOKE FAILED");
process.exit(allGreen && errors.length === 0 ? 0 : 1);
