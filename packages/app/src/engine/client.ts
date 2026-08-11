/**
 * Typed async facade over the engine worker. One worker per project;
 * requests are serialized (the engine is single-threaded anyway).
 */
import type { AlignResult, WorkerRequest, WorkerResponse } from "./protocol";

type Pending = {
  resolve: (msg: WorkerResponse) => void;
  reject: (err: Error) => void;
};

export class EngineClient {
  private worker: Worker;
  private queue: Promise<unknown> = Promise.resolve();
  private pending: Pending | null = null;
  version = "";

  constructor() {
    this.worker = new Worker(new URL("./worker.ts", import.meta.url), {
      type: "module",
    });
    this.worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
      const p = this.pending;
      this.pending = null;
      if (!p) return;
      if (e.data.type === "error") {
        p.reject(new Error(e.data.message));
      } else {
        p.resolve(e.data);
      }
    };
    this.worker.onerror = (e) => {
      this.pending?.reject(new Error(e.message));
      this.pending = null;
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

  async init(): Promise<string> {
    const r = await this.send({ type: "init" });
    if (r.type !== "ready") throw new Error("unexpected response");
    this.version = r.version;
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

  async finishExport(
    quality: number,
  ): Promise<{ jpeg: ArrayBuffer; width: number; height: number }> {
    const r = await this.send({ type: "finishExport", quality });
    if (r.type !== "exportDone") throw new Error("unexpected response");
    return { jpeg: r.jpeg, width: r.width, height: r.height };
  }

  dispose() {
    this.worker.terminate();
  }
}
