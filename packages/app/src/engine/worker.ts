/// <reference lib="webworker" />
/**
 * Engine worker. Loads the wasm module (built by `pnpm build:wasm` into
 * ./pkg) and answers the M0 smoke test: version + a pixel round trip.
 */
import init, { engine_version, smoke_grayscale } from "./pkg/panoloom.js";
import wasmUrl from "./pkg/panoloom_bg.wasm?url";
import type { EngineSmokeReport } from "./client";

// wasm SIMD feature probe (validates a tiny module using v128 ops).
const SIMD_PROBE = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1,
  8, 0, 65, 0, 253, 15, 253, 98, 11,
]);

self.onmessage = async () => {
  const report: EngineSmokeReport = {
    engineVersion: "unknown",
    grayscaleOk: false,
    crossOriginIsolated: self.crossOriginIsolated === true,
    simdSupported: WebAssembly.validate(SIMD_PROBE),
    threadsAvailable:
      self.crossOriginIsolated === true && typeof SharedArrayBuffer !== "undefined",
  };
  try {
    await init({ module_or_path: wasmUrl });
    report.engineVersion = engine_version();
    // 2x1 image: pure red, pure green — expect OpenCV luma 76, 150.
    const gray = smoke_grayscale(2, 1, new Uint8Array([255, 0, 0, 255, 0, 255, 0, 255]));
    report.grayscaleOk = gray.length === 2 && gray[0] === 76 && gray[1] === 150;
  } catch (err) {
    report.error = err instanceof Error ? err.message : String(err);
  }
  self.postMessage(report);
};
