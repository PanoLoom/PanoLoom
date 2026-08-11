/** Messages between the UI and the engine worker. */

export type WorkerRequest =
  | { type: "init" }
  | {
      type: "addImage";
      id: number;
      rgba: ArrayBuffer;
      width: number;
      height: number;
      posePrior: [number, number, number] | null;
    }
  | { type: "removeImage"; id: number }
  | { type: "align" }
  | { type: "preview"; maxWidth: number }
  | {
      type: "beginExport";
      targetWidth: number;
      fullSizes: { id: number; width: number; height: number }[];
    }
  | {
      type: "exportSetImage";
      id: number;
      rgba: ArrayBuffer;
      width: number;
      height: number;
    }
  | { type: "exportDropImage"; id: number }
  | { type: "exportBand"; band: number }
  | { type: "finishExport"; quality: number };

export interface ExportPlan {
  width: number;
  height: number;
  bands: { y0: number; y1: number; needed: number[] }[];
}

export interface AlignResult {
  aligned: number[];
  /** Placed via shooting-rig pose metadata (too few features to match). */
  rescued: number[];
  dropped: number[];
  warpedImageScale: number;
}

export type WorkerResponse =
  | { type: "ready"; version: string }
  | { type: "imageAdded"; id: number }
  | { type: "imageRemoved"; id: number }
  | { type: "aligned"; result: AlignResult }
  | {
      type: "previewReady";
      rgba: ArrayBuffer;
      width: number;
      height: number;
    }
  | { type: "exportPlanned"; plan: ExportPlan }
  | { type: "exportImageSet"; id: number }
  | { type: "exportImageDropped"; id: number }
  | { type: "bandDone"; band: number }
  | {
      type: "exportDone";
      jpeg: ArrayBuffer;
      width: number;
      height: number;
    }
  | { type: "error"; op: WorkerRequest["type"]; message: string };
