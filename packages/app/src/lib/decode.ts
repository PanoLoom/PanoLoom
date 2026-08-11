/**
 * File decoding: browser-native JPEG/PNG decode, downscale to registration
 * scale (stitcher semantics: sqrt(0.6 MP / area), latched from the first
 * image), thumbnail, and EXIF focal length.
 */
import ExifReader from "exifreader";

export interface DecodedImage {
  id: number;
  fileName: string;
  fullWidth: number;
  fullHeight: number;
  /** Registration-scale pixels, ready for the engine. */
  rgba: ArrayBuffer;
  width: number;
  height: number;
  thumbnailUrl: string;
  focalLength35mm: number | null;
}

const REGISTR_MEGAPIXELS = 0.6e6;
let nextId = 1;

/** Work scale is latched from the FIRST image (stitcher.cpp:424-453). */
export function workScaleFor(width: number, height: number): number {
  return Math.min(1.0, Math.sqrt(REGISTR_MEGAPIXELS / (width * height)));
}

export async function decodeFile(
  file: File,
  workScale: number | null,
): Promise<DecodedImage> {
  const id = nextId++;
  const buf = await file.arrayBuffer();

  let focalLength35mm: number | null = null;
  try {
    const tags = ExifReader.load(buf);
    const f35 = tags.FocalLengthIn35mmFilm?.value;
    if (typeof f35 === "number" && f35 > 0) focalLength35mm = f35;
  } catch {
    // EXIF is best-effort.
  }

  const full = await createImageBitmap(new Blob([buf]));
  const scale = workScale ?? workScaleFor(full.width, full.height);
  const w = Math.max(2, Math.round(full.width * scale));
  const h = Math.max(2, Math.round(full.height * scale));

  const canvas = new OffscreenCanvas(w, h);
  const ctx = canvas.getContext("2d")!;
  ctx.drawImage(full, 0, 0, w, h);
  const pixels = ctx.getImageData(0, 0, w, h);

  // Thumbnail (fixed height 88px).
  const tw = Math.round((full.width / full.height) * 88);
  const tc = new OffscreenCanvas(tw, 88);
  tc.getContext("2d")!.drawImage(full, 0, 0, tw, 88);
  const thumbBlob = await tc.convertToBlob({ type: "image/jpeg", quality: 0.8 });
  const thumbnailUrl = URL.createObjectURL(thumbBlob);

  const result: DecodedImage = {
    id,
    fileName: file.name,
    fullWidth: full.width,
    fullHeight: full.height,
    rgba: pixels.data.buffer as ArrayBuffer,
    width: w,
    height: h,
    thumbnailUrl,
    focalLength35mm,
  };
  full.close();
  return result;
}
