/**
 * @panoloom/metadata — EXIF reading, 360° XMP injection, and encoder glue.
 *
 * Surface:
 *  - injectGPano(jpeg, opts): splice a Google Photo Sphere XMP APP1 segment
 *    into an equirectangular JPEG so viewers recognize it as a 360° photo
 *
 * Planned (milestone M7):
 *  - readExif(file): extract focal length / make / model via ExifReader
 *  - encodeJpeg(rgba, w, h, quality): mozjpeg (jSquash) from raw RGBA
 */

export {
  injectGPano,
  buildGPanoXmpPacket,
  listJpegSegments,
  GPANO_NS,
  XMP_HEADER,
} from "./gpano";
export type { GPanoOptions, JpegSegment } from "./gpano";
