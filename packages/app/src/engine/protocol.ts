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
  | { type: "orient"; r: number[] }
  | { type: "exportAlignment" }
  | { type: "importAlignment"; alignment: string }
  | { type: "preview"; maxWidth: number }
  | { type: "cancelExport" }
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

/** width/height (and left/top) describe the coverage CROP the JPEG will
 *  span; fullWidth/fullHeight are the full 2:1 sphere it sits on. */
export interface ExportPlan {
  width: number;
  height: number;
  left: number;
  top: number;
  fullWidth: number;
  fullHeight: number;
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
  | { type: "ready"; version: string; threads: number }
  | { type: "imageAdded"; id: number }
  | { type: "imageRemoved"; id: number }
  | { type: "aligned"; result: AlignResult }
  | { type: "oriented" }
  | { type: "alignmentExported"; alignment: string }
  | { type: "exportCancelled" }
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
      left: number;
      top: number;
      fullWidth: number;
      fullHeight: number;
    }
  | { type: "error"; op: WorkerRequest["type"]; message: string };
