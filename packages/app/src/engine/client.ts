/**
 * Main-thread client for the engine worker. For M0 this is a single
 * smoke-test round trip; the real typed API (Comlink) lands with M3+.
 */

export interface EngineSmokeReport {
  engineVersion: string;
  grayscaleOk: boolean;
  crossOriginIsolated: boolean;
  simdSupported: boolean;
  threadsAvailable: boolean;
  error?: string;
}

export function runEngineSmokeTest(): Promise<EngineSmokeReport> {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL("./worker.ts", import.meta.url), {
      type: "module",
    });
    worker.onmessage = (e: MessageEvent<EngineSmokeReport>) => {
      resolve(e.data);
      worker.terminate();
    };
    worker.onerror = (e) => {
      reject(new Error(e.message));
      worker.terminate();
    };
    worker.postMessage("smoke");
  });
}
