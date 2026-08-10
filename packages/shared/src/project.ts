/**
 * The `.panoproj` project document.
 *
 * MUST stay in sync with the serde structs in
 * `crates/panoloom-core/src/project.rs` — that file is the source of truth
 * for the wire format (camelCase JSON).
 */

export const PROJECT_FORMAT_VERSION = 1;

export type LensType = "rectilinear" | "fisheyeCircular" | "fisheyeFullframe";

export interface ExifInfo {
  focalLengthMm?: number | null;
  focalLength35mm?: number | null;
  make?: string | null;
  model?: string | null;
}

/** PanoTools-style lens model. */
export interface Lens {
  lensType: LensType;
  hfovDeg: number;
  a: number;
  b: number;
  c: number;
  d: number;
  e: number;
}

export interface Pose {
  yaw: number;
  pitch: number;
  roll: number;
}

export interface Photometric {
  /** Exposure offset in EV relative to the anchor image. */
  ev: number;
  wbR: number;
  wbB: number;
}

export interface ImageEntry {
  id: number;
  fileName: string;
  width: number;
  height: number;
  exif?: ExifInfo | null;
  lens: Lens;
  pose: Pose;
  photometric: Photometric;
  stackId?: number | null;
}

export type ControlPointKind = "normal" | "verticalLine" | "horizontalLine";

/** Coordinates are always in ORIGINAL image pixel space. */
export interface ControlPoint {
  id: number;
  imgA: number;
  imgB: number;
  xA: number;
  yA: number;
  xB: number;
  yB: number;
  kind: ControlPointKind;
  errorPx?: number | null;
}

export interface OptimizerSettings {
  optimizeYawPitchRoll: boolean;
  optimizeHfov: boolean;
  optimizeDistortion: boolean;
  optimizeShift: boolean;
}

export type Projection =
  | "equirectangular"
  | "cylindrical"
  | "rectilinear"
  | "stereographic"
  | "mercator";

export interface CropRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface PanoramaSettings {
  projection: Projection;
  yaw: number;
  pitch: number;
  roll: number;
  hfovDeg: number;
  vfovDeg: number;
  width: number;
  height: number;
  crop?: CropRect | null;
}

export interface Project {
  version: number;
  images: ImageEntry[];
  controlPoints: ControlPoint[];
  optimizer: OptimizerSettings;
  panorama: PanoramaSettings;
}
