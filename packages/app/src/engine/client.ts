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

  dispose() {
    this.worker.terminate();
  }
}
