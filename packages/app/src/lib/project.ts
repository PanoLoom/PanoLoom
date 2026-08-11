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
    /** Painted seam masks, RLE [value, count, ...] at registration dims. */
    masks?: { id: number; width: number; height: number; rle: number[] }[];
  };
}

export function rleEncode(data: Uint8Array): number[] {
  const out: number[] = [];
  let i = 0;
  while (i < data.length) {
    const v = data[i]!;
    let n = 1;
    while (i + n < data.length && data[i + n] === v) n++;
    out.push(v, n);
    i += n;
  }
  return out;
}

export function rleDecode(rle: number[], length: number): Uint8Array {
  const out = new Uint8Array(length);
  let p = 0;
  for (let i = 0; i + 1 < rle.length; i += 2) {
    out.fill(rle[i]!, p, Math.min(length, p + rle[i + 1]!));
    p += rle[i + 1]!;
  }
  return out;
}

export interface ShotMeta {
  id: number;
  fileName: string;
  fullWidth: number;
  fullHeight: number;
  focalLength35mm: number | null;
}

export interface CpLike {
  id: number;
  imgA: number;
  imgB: number;
  xA: number;
  yA: number;
  xB: number;
  yB: number;
  errorPx?: number | null;
}

export function buildProject(
  shots: ShotMeta[],
  alignmentJson: string,
  workScale: number,
  engineVersion: string,
  cps: CpLike[] = [],
  masks: Map<number, Uint8Array> = new Map(),
  maskDims: Map<number, { width: number; height: number }> = new Map(),
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
    // .panoproj stores CPs in ORIGINAL image coordinates.
    controlPoints: cps.map((cp) => ({
      id: cp.id,
      imgA: cp.imgA,
      imgB: cp.imgB,
      xA: cp.xA / workScale,
      yA: cp.yA / workScale,
      xB: cp.xB / workScale,
      yB: cp.yB / workScale,
      kind: "normal" as const,
      errorPx: cp.errorPx ?? null,
    })),
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
      masks: [...masks.entries()]
        .filter(([, m]) => m.some((v) => v !== 0))
        .flatMap(([id, m]) => {
          const dims = maskDims.get(id);
          return dims
            ? [{ id, width: dims.width, height: dims.height, rle: rleEncode(m) }]
            : [];
        }),
    },
  };
  return JSON.stringify(doc, null, 2);
}

export interface ParsedProject {
  /** Files to ask the user for, in project order. */
  entries: { id: number; fileName: string; width: number; height: number }[];
  alignmentJson: string;
  workScale: number;
  /** Control points converted to REGISTRATION coordinates. */
  cps: CpLike[];
  /** Painted seam masks at registration dims. */
  masks: { id: number; width: number; height: number; data: Uint8Array }[];
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
  const ws = doc.panoloom.workScale;
  return {
    entries: doc.images.map((im) => ({
      id: im.id,
      fileName: im.fileName,
      width: im.width,
      height: im.height,
    })),
    alignmentJson: JSON.stringify(doc.panoloom.alignment),
    workScale: ws,
    cps: (doc.controlPoints ?? []).map((cp) => ({
      id: cp.id,
      imgA: cp.imgA,
      imgB: cp.imgB,
      xA: cp.xA * ws,
      yA: cp.yA * ws,
      xB: cp.xB * ws,
      yB: cp.yB * ws,
      errorPx: cp.errorPx ?? null,
    })),
    masks: (doc.panoloom.masks ?? []).map((m) => ({
      id: m.id,
      width: m.width,
      height: m.height,
      data: rleDecode(m.rle, m.width * m.height),
    })),
  };
}
