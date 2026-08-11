/**
 * .panoproj save/load. The document follows the shared schema
 * (@panoloom/shared, mirrored by panoloom-core's project.rs); the
 * `panoloom.alignment` extension carries the engine's exact camera state
 * (serde_json round-trips every float), so loading restores the alignment
 * bit-for-bit without re-running registration.
 */
import {
  PROJECT_FORMAT_VERSION,
  rotationToPose,
  type ImageEntry,
  type Mat3,
  type Project,
} from "@panoloom/shared";

interface EngineCamera {
  focal: number;
  aspect: number;
  ppx: number;
  ppy: number;
  r: Mat3;
}

interface EngineAlignment {
  images: { id: number; camera: EngineCamera; rescued: boolean }[];
  dropped: number[];
  warpedImageScale: number;
}

export interface PanoloomProject extends Project {
  panoloom: {
    engineVersion: string;
    /** Registration scale the images were decoded at when saved. */
    workScale: number;
    alignment: EngineAlignment;
  };
}

export interface ShotMeta {
  id: number;
  fileName: string;
  fullWidth: number;
  fullHeight: number;
  focalLength35mm: number | null;
}

export function buildProject(
  shots: ShotMeta[],
  alignmentJson: string,
  workScale: number,
  engineVersion: string,
): string {
  const alignment = JSON.parse(alignmentJson) as EngineAlignment;
  const byId = new Map(alignment.images.map((ai) => [ai.id, ai]));

  const images: ImageEntry[] = shots.map((s) => {
    const ai = byId.get(s.id);
    const pose = ai
      ? rotationToPose(ai.camera.r)
      : { yaw: 0, pitch: 0, roll: 0 };
    // focal is in registration-scale pixels; hfov is scale-invariant.
    const hfovDeg = ai
      ? (2 *
          Math.atan2(s.fullWidth * workScale * 0.5, ai.camera.focal) *
          180) /
        Math.PI
      : 0;
    return {
      id: s.id,
      fileName: s.fileName,
      width: s.fullWidth,
      height: s.fullHeight,
      exif: s.focalLength35mm ? { focalLength35mm: s.focalLength35mm } : null,
      lens: { lensType: "rectilinear", hfovDeg, a: 0, b: 0, c: 0, d: 0, e: 0 },
      pose,
      photometric: { ev: 0, wbR: 1, wbB: 1 },
    };
  });

  const doc: PanoloomProject = {
    version: PROJECT_FORMAT_VERSION,
    images,
    controlPoints: [],
    optimizer: {
      optimizeYawPitchRoll: true,
      optimizeHfov: false,
      optimizeDistortion: false,
      optimizeShift: false,
    },
    panorama: {
      projection: "equirectangular",
      yaw: 0,
      pitch: 0,
      roll: 0,
      hfovDeg: 360,
      vfovDeg: 180,
      width: 0,
      height: 0,
    },
    panoloom: {
      engineVersion,
      workScale,
      alignment,
    },
  };
  return JSON.stringify(doc, null, 2);
}

export interface ParsedProject {
  /** Files to ask the user for, in project order. */
  entries: { id: number; fileName: string; width: number; height: number }[];
  alignmentJson: string;
  workScale: number;
}

export function parseProject(text: string): ParsedProject {
  let doc: PanoloomProject;
  try {
    doc = JSON.parse(text) as PanoloomProject;
  } catch {
    throw new Error("not a valid .panoproj file (bad JSON)");
  }
  if (doc.version !== PROJECT_FORMAT_VERSION) {
    throw new Error(`unsupported project version ${doc.version}`);
  }
  if (!doc.panoloom?.alignment || !doc.panoloom.workScale) {
    throw new Error("project has no saved alignment");
  }
  if (!Array.isArray(doc.images) || doc.images.length === 0) {
    throw new Error("project lists no images");
  }
  return {
    entries: doc.images.map((im) => ({
      id: im.id,
      fileName: im.fileName,
      width: im.width,
      height: im.height,
    })),
    alignmentJson: JSON.stringify(doc.panoloom.alignment),
    workScale: doc.panoloom.workScale,
  };
}
