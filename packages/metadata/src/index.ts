/**
 * @panoloom/metadata — EXIF reading, 360° XMP injection, and encoder glue.
 *
 * Planned surface (implemented in milestone M7, stubs to fix the API shape):
 *  - readExif(file): extract focal length / make / model via ExifReader
 *  - injectGPano(jpeg, opts): splice a Google Photo Sphere XMP APP1 segment
 *    into an equirectangular JPEG so viewers recognize it as a 360° photo
 *  - encodeJpeg(rgba, w, h, quality): mozjpeg (jSquash) from raw RGBA
 */

export interface GPanoOptions {
  fullPanoWidthPixels: number;
  fullPanoHeightPixels: number;
  croppedAreaLeftPixels: number;
  croppedAreaTopPixels: number;
  croppedAreaImageWidthPixels: number;
  croppedAreaImageHeightPixels: number;
}

export function injectGPano(_jpeg: Uint8Array, _opts: GPanoOptions): Uint8Array {
  throw new Error("not implemented until M7 (export milestone)");
}
