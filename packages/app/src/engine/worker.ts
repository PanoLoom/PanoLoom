/// <reference lib="webworker" />
/**
 * Engine worker: owns the wasm Engine instance. Pixels arrive/leave as
 * transferable ArrayBuffers; one in-flight request at a time per op.
 *
 * Two engine builds exist: pkg (single-thread) and pkg-mt (rayon over a
 * Web Worker pool via SharedArrayBuffer). The mt build needs cross-origin
 * isolation; when available we load it and spin up one rayon worker per
 * core. Both expose the identical Engine API.
 */
import type { Engine } from "./pkg/panoloom.js";
import type { WorkerRequest, WorkerResponse } from "./protocol";

let engine: Engine | null = null;

function post(msg: WorkerResponse, transfer: Transferable[] = []) {
  (self as unknown as Worker).postMessage(msg, transfer);
}

async function boot(): Promise<{ version: string; threads: number }> {
  if (typeof SharedArrayBuffer !== "undefined" && self.crossOriginIsolated) {
    try {
      const mod = await import("./pkg-mt/panoloom.js");
      const wasmUrl = (await import("./pkg-mt/panoloom_bg.wasm?url")).default;
      await mod.default({ module_or_path: wasmUrl });
      const threads = Math.min(navigator.hardwareConcurrency || 4, 16);
      await mod.initThreadPool(threads);
      engine = new mod.Engine() as unknown as Engine;
      return { version: mod.engine_version(), threads };
    } catch (err) {
      console.warn("mt engine failed to start, using single-thread:", err);
      engine = null;
    }
  }
  const mod = await import("./pkg/panoloom.js");
  const wasmUrl = (await import("./pkg/panoloom_bg.wasm?url")).default;
  await mod.default({ module_or_path: wasmUrl });
  engine = new mod.Engine();
  return { version: mod.engine_version(), threads: 0 };
}

self.onmessage = async (e: MessageEvent<WorkerRequest>) => {
  const msg = e.data;
  try {
    switch (msg.type) {
      case "init": {
        const { version, threads } = await boot();
        post({ type: "ready", version, threads });
        break;
      }
      case "addImage": {
        engine!.add_image(
          msg.id,
          new Uint8Array(msg.rgba),
          msg.width,
          msg.height,
          msg.posePrior ? Float64Array.from(msg.posePrior) : undefined,
        );
        post({ type: "imageAdded", id: msg.id });
        break;
      }
      case "removeImage": {
        engine!.remove_image(msg.id);
        post({ type: "imageRemoved", id: msg.id });
        break;
      }
      case "align": {
        const t0 = performance.now();
        const result = JSON.parse(engine!.align());
        console.log(`[engine] align: ${(performance.now() - t0).toFixed(0)}ms`);
        post({ type: "aligned", result });
        break;
      }
      case "exportAlignment": {
        post({ type: "alignmentExported", alignment: engine!.export_alignment() });
        break;
      }
      case "importAlignment": {
        const result = JSON.parse(engine!.import_alignment(msg.alignment));
        post({ type: "aligned", result });
        break;
      }
      case "preview": {
        const t0 = performance.now();
        const p = engine!.render_preview(msg.maxWidth);
        console.log(`[engine] preview: ${(performance.now() - t0).toFixed(0)}ms`);
        const rgba = p.take_rgba();
        const buf = rgba.buffer as ArrayBuffer;
        post(
          {
            type: "previewReady",
            rgba: buf,
            width: p.width,
            height: p.height,
          },
          [buf],
        );
        p.free();
        break;
      }
      case "beginExport": {
        const plan = engine!.begin_export(
          msg.targetWidth,
          Uint32Array.from(msg.fullSizes.map((s) => s.id)),
          Uint32Array.from(msg.fullSizes.map((s) => s.width)),
          Uint32Array.from(msg.fullSizes.map((s) => s.height)),
        );
        post({ type: "exportPlanned", plan: JSON.parse(plan) });
        break;
      }
      case "exportSetImage": {
        engine!.export_set_image(
          msg.id,
          new Uint8Array(msg.rgba),
          msg.width,
          msg.height,
        );
        post({ type: "exportImageSet", id: msg.id });
        break;
      }
      case "exportDropImage": {
        engine!.export_drop_image(msg.id);
        post({ type: "exportImageDropped", id: msg.id });
        break;
      }
      case "exportBand": {
        engine!.export_band(msg.band);
        post({ type: "bandDone", band: msg.band });
        break;
      }
      case "finishExport": {
        const r = engine!.finish_export(msg.quality);
        const jpeg = r.take_jpeg();
        const buf = jpeg.buffer as ArrayBuffer;
        post(
          { type: "exportDone", jpeg: buf, width: r.width, height: r.height },
          [buf],
        );
        r.free();
        break;
      }
    }
  } catch (err) {
    post({
      type: "error",
      op: msg.type,
      message: err instanceof Error ? err.message : String(err),
    });
  }
};
