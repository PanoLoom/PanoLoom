/**
 * Typed async facade over the engine worker. One worker per project;
 * requests are serialized (the engine is single-threaded anyway).
 */
import type { AlignResult, WorkerRequest, WorkerResponse } from "./protocol";

/** Signatures of an engine left unusable by a wasm trap. */
const POISONED =
  /recursive use of an object|unreachable|memory access out of bounds|null pointer passed to rust/i;

const ENGINE_DIED =
  "the engine ran out of memory and was restarted — try a smaller export size";

type Pending = {
  resolve: (msg: WorkerResponse) => void;
  reject: (err: Error) => void;
};

export class EngineClient {
  private worker: Worker;
  private queue: Promise<unknown> = Promise.resolve();
  private pending: Pending | null = null;
  version = "";
  /** rayon pool size; 0 = single-threaded engine. */
  threads = 0;
  /** Widest panorama the engine can compose — a large set cannot reach
   *  full resolution in a 4 GB address space. */
  maxExportWidth = 65535;
  /** Fired on an UNCAUGHT worker error (wasm panic/OOM) — the engine is
   *  gone; the app should replace this client and re-import its shots. */
  onFatal: ((message: string) => void) | null = null;
  /** Fired as a long call (align/preview) moves between engine stages, so
   *  the UI can show which one is running rather than an opaque spinner. */
  onProgress: ((stage: string) => void) | null = null;

  constructor() {
    this.worker = new Worker(new URL("./worker.ts", import.meta.url), {
      type: "module",
    });
    this.worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
      // Progress is out-of-band: it arrives DURING a request, so it must not
      // settle the pending promise.
      if (e.data.type === "progress") {
        this.onProgress?.(e.data.stage);
        return;
      }
      const p = this.pending;
      this.pending = null;
      if (!p) return;
      if (e.data.type === "error") {
        // A wasm trap (out of memory, unreachable) aborts rather than
        // unwinds, so the borrow guard on the engine object is never
        // released. Every later call then fails with "recursive use of an
        // object" — a message about the wreckage, not the crash. Treat it
        // as fatal so the engine is rebuilt instead of the user chasing it.
        if (POISONED.test(e.data.message)) {
          p.reject(new Error(ENGINE_DIED));
          this.onFatal?.(ENGINE_DIED);
          return;
        }
        p.reject(new Error(e.data.message));
      } else {
        p.resolve(e.data);
      }
    };
    this.worker.onerror = (e) => {
      const message = e.message || "engine crashed";
      this.pending?.reject(new Error(message));
      this.pending = null;
      this.onFatal?.(message);
    };
  }

  private send(
    msg: WorkerRequest,
    transfer: Transferable[] = [],
  ): Promise<WorkerResponse> {
    const run = () =>
      new Promise<WorkerResponse>((resolve, reject) => {
        this.pending = { resolve, reject };
        this.worker.postMessage(msg, transfer);
      });
    const next = this.queue.then(run, run);
    this.queue = next.catch(() => {});
    return next;
  }

  async init(maxThreads?: number): Promise<string> {
    const r = await this.send({ type: "init", maxThreads });
    if (r.type !== "ready") throw new Error("unexpected response");
    this.version = r.version;
    this.threads = r.threads;
    this.maxExportWidth = r.maxExportWidth;
    return r.version;
  }

  async addImage(
    id: number,
    rgba: ArrayBuffer,
    width: number,
    height: number,
    posePrior: [number, number, number] | null = null,
  ): Promise<void> {
    await this.send({ type: "addImage", id, rgba, width, height, posePrior }, [
      rgba,
    ]);
  }

  async removeImage(id: number): Promise<void> {
    await this.send({ type: "removeImage", id });
  }

  async align(): Promise<AlignResult> {
    const r = await this.send({ type: "align" });
    if (r.type !== "aligned") throw new Error("unexpected response");
    return r.result;
  }

  /** Rotates the whole panorama (row-major 3x3, pano frame). */
  async orient(r: number[]): Promise<void> {
    const resp = await this.send({ type: "orient", r });
    if (resp.type !== "oriented") throw new Error("unexpected response");
  }

  /** Painted seam mask (0 none / 1 exclude / 2 prefer, registration dims). */
  async setMask(
    id: number,
    mask: ArrayBuffer,
    width: number,
    height: number,
  ): Promise<void> {
    const r = await this.send({ type: "setMask", id, mask, width, height }, [
      mask,
    ]);
    if (r.type !== "maskSet") throw new Error("unexpected response");
  }

  async clearMask(id: number): Promise<void> {
    const r = await this.send({ type: "clearMask", id });
    if (r.type !== "maskSet") throw new Error("unexpected response");
  }

  /** Feature-derived control points (registration coords). */
  async autoControlPoints(
    maxPerPair: number,
  ): Promise<import("./protocol").EngineControlPoint[]> {
    const r = await this.send({ type: "autoControlPoints", maxPerPair });
    if (r.type !== "controlPoints") throw new Error("unexpected response");
    return r.cps;
  }

  /** Optimize the alignment against control points; mutates engine state. */
  async optimizeCps(
    cps: import("./protocol").EngineControlPoint[],
    flags: import("./protocol").OptimizeFlags,
  ): Promise<import("./protocol").OptimizeReport> {
    const r = await this.send({ type: "optimizeCps", cps, flags });
    if (r.type !== "optimized") throw new Error("unexpected response");
    return r.report;
  }

  /** Frees an in-progress export session. */
  async cancelExport(): Promise<void> {
    const r = await this.send({ type: "cancelExport" });
    if (r.type !== "exportCancelled") throw new Error("unexpected response");
  }

  /** Exact-round-trip alignment JSON for project save. */
  async exportAlignment(): Promise<string> {
    const r = await this.send({ type: "exportAlignment" });
    if (r.type !== "alignmentExported") throw new Error("unexpected response");
    return r.alignment;
  }

  /** Restores a saved alignment; images must be re-added first. */
  async importAlignment(alignment: string): Promise<AlignResult> {
    const r = await this.send({ type: "importAlignment", alignment });
    if (r.type !== "aligned") throw new Error("unexpected response");
    return r.result;
  }

  async renderPreview(
    maxWidth: number,
  ): Promise<{ rgba: ArrayBuffer; width: number; height: number }> {
    const r = await this.send({ type: "preview", maxWidth });
    if (r.type !== "previewReady") throw new Error("unexpected response");
    return { rgba: r.rgba, width: r.width, height: r.height };
  }

  async beginExport(
    targetWidth: number,
    fullSizes: { id: number; width: number; height: number }[],
  ): Promise<import("./protocol").ExportPlan> {
    const r = await this.send({ type: "beginExport", targetWidth, fullSizes });
    if (r.type !== "exportPlanned") throw new Error("unexpected response");
    return r.plan;
  }

  async exportSetImage(
    id: number,
    rgba: ArrayBuffer,
    width: number,
    height: number,
  ): Promise<void> {
    await this.send({ type: "exportSetImage", id, rgba, width, height }, [
      rgba,
    ]);
  }

  async exportDropImage(id: number): Promise<void> {
    await this.send({ type: "exportDropImage", id });
  }

  async exportBand(band: number): Promise<void> {
    await this.send({ type: "exportBand", band });
  }

  async finishExport(quality: number): Promise<{
    jpeg: ArrayBuffer;
    width: number;
    height: number;
    left: number;
    top: number;
    fullWidth: number;
    fullHeight: number;
  }> {
    const r = await this.send({ type: "finishExport", quality });
    if (r.type !== "exportDone") throw new Error("unexpected response");
    const { type: _t, ...rest } = r;
    return rest;
  }

  dispose() {
    this.worker.terminate();
  }
}
