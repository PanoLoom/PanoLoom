/// <reference lib="webworker" />
/**
 * Engine worker: owns the wasm Engine instance. Pixels arrive/leave as
 * transferable ArrayBuffers; one in-flight request at a time per op.
 */
import init, { Engine, engine_version } from "./pkg/panoloom.js";
import wasmUrl from "./pkg/panoloom_bg.wasm?url";
import type { WorkerRequest, WorkerResponse } from "./protocol";

let engine: Engine | null = null;

function post(msg: WorkerResponse, transfer: Transferable[] = []) {
  (self as unknown as Worker).postMessage(msg, transfer);
}

self.onmessage = async (e: MessageEvent<WorkerRequest>) => {
  const msg = e.data;
  try {
    switch (msg.type) {
      case "init": {
        await init({ module_or_path: wasmUrl });
        engine = new Engine();
        post({ type: "ready", version: engine_version() });
        break;
      }
      case "addImage": {
        engine!.add_image(
          msg.id,
          new Uint8Array(msg.rgba),
          msg.width,
          msg.height,
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
        const result = JSON.parse(engine!.align());
        post({ type: "aligned", result });
        break;
      }
      case "preview": {
        const p = engine!.render_preview(msg.maxWidth);
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
    }
  } catch (err) {
    post({
      type: "error",
      op: msg.type,
      message: err instanceof Error ? err.message : String(err),
    });
  }
};
