/** Messages between the UI and the engine worker. */

export type WorkerRequest =
  | { type: "init" }
  | {
      type: "addImage";
      id: number;
      rgba: ArrayBuffer;
      width: number;
      height: number;
    }
  | { type: "removeImage"; id: number }
  | { type: "align" }
  | { type: "preview"; maxWidth: number };

export interface AlignResult {
  aligned: number[];
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
  | { type: "error"; op: WorkerRequest["type"]; message: string };
