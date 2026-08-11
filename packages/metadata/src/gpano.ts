/**
 * GPano (Google Photo Sphere) XMP injection for equirectangular JPEGs.
 *
 * Spec: https://developers.google.com/streetview/spherical-metadata
 * The XMP packet lives in a JPEG APP1 segment whose payload begins with the
 * null-terminated namespace header "http://ns.adobe.com/xap/1.0/\0".
 */

/** GPano XML namespace URI (also used to detect existing GPano XMP segments). */
export const GPANO_NS = "http://ns.google.com/photos/1.0/panorama/";

/** APP1 payload prefix identifying an XMP packet (includes the NUL terminator). */
export const XMP_HEADER = "http://ns.adobe.com/xap/1.0/\0";

export interface GPanoOptions {
  fullPanoWidthPixels: number;
  fullPanoHeightPixels: number;
  croppedAreaLeftPixels: number;
  croppedAreaTopPixels: number;
  croppedAreaImageWidthPixels: number;
  croppedAreaImageHeightPixels: number;
  /** Defaults to "equirectangular" — the only value Google products support. */
  projectionType?: string;
  /** Defaults to true. */
  usePanoramaViewer?: boolean;
}

/** A parsed JPEG marker segment (header-area segments only, i.e. before SOS). */
export interface JpegSegment {
  /** Second marker byte, e.g. 0xe0 for APP0, 0xe1 for APP1, 0xda for SOS. */
  marker: number;
  /** Offset of the 0xFF marker prefix. */
  start: number;
  /** Offset one past the end of the segment (start of the next segment). */
  end: number;
}

const SOI = 0xd8;
const EOI = 0xd9;
const SOS = 0xda;
const APP0 = 0xe0;
const APP1 = 0xe1;

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder();

/**
 * Walk the marker segments of a JPEG header, starting after SOI and stopping
 * at SOS/EOI (entropy-coded scan data is not parsed). Throws on malformed or
 * truncated input. The terminating SOS/EOI marker itself is not included.
 */
export function listJpegSegments(jpeg: Uint8Array): JpegSegment[] {
  if (jpeg.length < 2 || jpeg[0] !== 0xff || jpeg[1] !== SOI) {
    throw new Error("injectGPano: input is not a JPEG (missing SOI marker 0xFFD8)");
  }
  const segments: JpegSegment[] = [];
  let pos = 2;
  while (pos < jpeg.length) {
    const start = pos;
    if (jpeg[pos] !== 0xff) {
      throw new Error(`injectGPano: malformed JPEG (expected marker at offset ${pos})`);
    }
    // Skip optional 0xFF fill bytes before the marker code.
    while (pos < jpeg.length && jpeg[pos] === 0xff) pos++;
    if (pos >= jpeg.length) {
      throw new Error("injectGPano: truncated JPEG (dangling 0xFF at end of file)");
    }
    const marker = jpeg[pos]!;
    pos++;
    if (marker === SOS || marker === EOI) {
      return segments; // header ends; scan data (or nothing) follows
    }
    if (marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) {
      // Standalone markers (TEM, RSTn) carry no length field.
      segments.push({ marker, start, end: pos });
      continue;
    }
    if (pos + 2 > jpeg.length) {
      throw new Error("injectGPano: truncated JPEG (segment length field cut off)");
    }
    const length = (jpeg[pos]! << 8) | jpeg[pos + 1]!; // includes the 2 length bytes
    if (length < 2 || pos + length > jpeg.length) {
      throw new Error(`injectGPano: malformed JPEG (bad segment length at offset ${pos})`);
    }
    pos += length;
    segments.push({ marker, start, end: pos });
  }
  throw new Error("injectGPano: truncated JPEG (no SOS/EOI marker found)");
}

/** True if the segment is an APP1 XMP packet that declares the GPano namespace. */
function isGPanoXmpSegment(jpeg: Uint8Array, seg: JpegSegment): boolean {
  if (seg.marker !== APP1) return false;
  // Payload starts after 0xFF 0xE1 + 2 length bytes. `seg.start` may include
  // fill bytes, so locate the marker code from the end of the 0xFF run.
  let p = seg.start;
  while (jpeg[p] === 0xff) p++;
  const payloadStart = p + 3; // marker code + 2 length bytes
  const payload = textDecoder.decode(jpeg.subarray(payloadStart, seg.end));
  return payload.startsWith(XMP_HEADER) && payload.includes(GPANO_NS);
}

function escapeXmlAttr(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function requireNonNegativeInt(name: string, value: number): number {
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(`injectGPano: ${name} must be a non-negative integer (got ${value})`);
  }
  return value;
}

/** Build the XMP packet (attribute-form RDF, as in the Photo Sphere spec examples). */
export function buildGPanoXmpPacket(opts: GPanoOptions): string {
  const fullW = requireNonNegativeInt("fullPanoWidthPixels", opts.fullPanoWidthPixels);
  const fullH = requireNonNegativeInt("fullPanoHeightPixels", opts.fullPanoHeightPixels);
  const cropW = requireNonNegativeInt(
    "croppedAreaImageWidthPixels",
    opts.croppedAreaImageWidthPixels,
  );
  const cropH = requireNonNegativeInt(
    "croppedAreaImageHeightPixels",
    opts.croppedAreaImageHeightPixels,
  );
  const left = requireNonNegativeInt("croppedAreaLeftPixels", opts.croppedAreaLeftPixels);
  const top = requireNonNegativeInt("croppedAreaTopPixels", opts.croppedAreaTopPixels);
  if (fullW <= 0 || fullH <= 0 || cropW <= 0 || cropH <= 0) {
    throw new Error("injectGPano: pano dimensions must be positive");
  }
  if (left + cropW > fullW || top + cropH > fullH) {
    throw new Error("injectGPano: cropped area extends outside the full pano dimensions");
  }
  const projection = escapeXmlAttr(opts.projectionType ?? "equirectangular");
  const useViewer = (opts.usePanoramaViewer ?? true) ? "True" : "False";

  return (
    '<?xpacket begin="\uFEFF" id="W5M0MpCehiHzreSzNTczkc9d"?>\n' +
    '<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="PanoLoom">\n' +
    '  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n' +
    '    <rdf:Description rdf:about=""\n' +
    `        xmlns:GPano="${GPANO_NS}"\n` +
    `      GPano:ProjectionType="${projection}"\n` +
    `      GPano:UsePanoramaViewer="${useViewer}"\n` +
    `      GPano:FullPanoWidthPixels="${fullW}"\n` +
    `      GPano:FullPanoHeightPixels="${fullH}"\n` +
    `      GPano:CroppedAreaImageWidthPixels="${cropW}"\n` +
    `      GPano:CroppedAreaImageHeightPixels="${cropH}"\n` +
    `      GPano:CroppedAreaLeftPixels="${left}"\n` +
    `      GPano:CroppedAreaTopPixels="${top}"/>\n` +
    "  </rdf:RDF>\n" +
    "</x:xmpmeta>\n" +
    '<?xpacket end="w"?>'
  );
}

/**
 * Splice a Google Photo Sphere (GPano) XMP APP1 segment into a JPEG.
 *
 * Insertion point: immediately after the leading run of APP0/APP1 segments
 * that follows SOI (so after JFIF APP0 and after an Exif APP1 if present,
 * which the Exif spec requires to come first). If the file starts with SOI
 * followed directly by non-APP segments, the XMP goes right after SOI.
 *
 * Idempotent: any existing APP1 XMP segment declaring the GPano namespace is
 * removed, so re-injecting replaces rather than duplicates. All other bytes
 * are preserved verbatim.
 */
export function injectGPano(jpeg: Uint8Array, opts: GPanoOptions): Uint8Array {
  const segments = listJpegSegments(jpeg); // also validates SOI + structure

  const packetBytes = textEncoder.encode(buildGPanoXmpPacket(opts));
  const headerBytes = textEncoder.encode(XMP_HEADER);
  const payloadLength = headerBytes.length + packetBytes.length;
  if (payloadLength + 2 > 0xffff) {
    throw new Error("injectGPano: XMP payload exceeds the APP1 segment size limit");
  }
  const app1 = new Uint8Array(4 + payloadLength);
  app1[0] = 0xff;
  app1[1] = APP1;
  app1[2] = (payloadLength + 2) >> 8;
  app1[3] = (payloadLength + 2) & 0xff;
  app1.set(headerBytes, 4);
  app1.set(packetBytes, 4 + headerBytes.length);

  const parts: Uint8Array[] = [jpeg.subarray(0, 2)]; // SOI
  let inserted = false;
  for (const seg of segments) {
    if (!inserted && seg.marker !== APP0 && seg.marker !== APP1) {
      parts.push(app1);
      inserted = true;
    }
    if (isGPanoXmpSegment(jpeg, seg)) continue; // drop the old GPano XMP
    parts.push(jpeg.subarray(seg.start, seg.end));
  }
  if (!inserted) parts.push(app1);
  // Everything after the last header segment (SOS marker + scan data + EOI,
  // or EOI alone) is copied through untouched.
  const tailStart = segments.length > 0 ? segments[segments.length - 1]!.end : 2;
  parts.push(jpeg.subarray(tailStart));

  const total = parts.reduce((sum, p) => sum + p.length, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const p of parts) {
    out.set(p, offset);
    offset += p.length;
  }
  return out;
}
