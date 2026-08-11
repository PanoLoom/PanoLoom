import { describe, expect, it } from "vitest";
import {
  GPANO_NS,
  XMP_HEADER,
  injectGPano,
  listJpegSegments,
  type GPanoOptions,
} from "./gpano";

const OPTS: GPanoOptions = {
  fullPanoWidthPixels: 8192,
  fullPanoHeightPixels: 4096,
  croppedAreaLeftPixels: 0,
  croppedAreaTopPixels: 512,
  croppedAreaImageWidthPixels: 8192,
  croppedAreaImageHeightPixels: 3072,
};

/** Build a marker segment: FF <marker> <len hi> <len lo> <payload>. */
function segment(marker: number, payload: number[]): number[] {
  const len = payload.length + 2;
  return [0xff, marker, len >> 8, len & 0xff, ...payload];
}

const APP0_JFIF = segment(0xe0, [
  0x4a, 0x46, 0x49, 0x46, 0x00, // "JFIF\0"
  0x01, 0x02, // version 1.2
  0x00, // density units
  0x00, 0x01, 0x00, 0x01, // x/y density
  0x00, 0x00, // no thumbnail
]);
const DQT = segment(0xdb, [0x00, ...Array(64).fill(0x10)]);
// SOS marker + minimal scan header, fake entropy data, EOI.
const SCAN_TAIL = [...segment(0xda, [0x01, 0x01, 0x00, 0x00, 0x3f, 0x00]), 0x12, 0x34, 0x56, 0xff, 0xd9];

/** SOI + APP0(JFIF) + DQT + SOS + fake scan bytes + EOI. */
function syntheticJpeg(): Uint8Array {
  return new Uint8Array([0xff, 0xd8, ...APP0_JFIF, ...DQT, ...SCAN_TAIL]);
}

function segmentPayloadText(jpeg: Uint8Array, start: number, end: number): string {
  return new TextDecoder().decode(jpeg.subarray(start + 4, end));
}

describe("injectGPano", () => {
  it("injects an XMP APP1 after APP0 with correct structure and values", () => {
    const original = syntheticJpeg();
    const out = injectGPano(original, OPTS);

    // Segment order: APP0, then the new APP1, then DQT.
    const segs = listJpegSegments(out);
    expect(segs.map((s) => s.marker)).toEqual([0xe0, 0xe1, 0xdb]);

    const app1 = segs[1]!;
    // Marker bytes and big-endian length field (includes its own 2 bytes).
    expect(out[app1.start]).toBe(0xff);
    expect(out[app1.start + 1]).toBe(0xe1);
    const lengthField = (out[app1.start + 2]! << 8) | out[app1.start + 3]!;
    expect(lengthField).toBe(app1.end - app1.start - 2);
    expect(lengthField + 2).toBeLessThan(0xffff);

    // Payload: XMP namespace header, then the packet with all seven fields.
    const payload = segmentPayloadText(out, app1.start, app1.end);
    expect(payload.startsWith(XMP_HEADER)).toBe(true);
    const packet = payload.slice(XMP_HEADER.length);
    expect(packet).toContain('xmlns:GPano="' + GPANO_NS + '"');
    expect(packet).toContain('GPano:ProjectionType="equirectangular"');
    expect(packet).toContain('GPano:UsePanoramaViewer="True"');
    expect(packet).toContain('GPano:FullPanoWidthPixels="8192"');
    expect(packet).toContain('GPano:FullPanoHeightPixels="4096"');
    expect(packet).toContain('GPano:CroppedAreaImageWidthPixels="8192"');
    expect(packet).toContain('GPano:CroppedAreaImageHeightPixels="3072"');
    expect(packet).toContain('GPano:CroppedAreaLeftPixels="0"');
    expect(packet).toContain('GPano:CroppedAreaTopPixels="512"');
    expect(packet).toContain("<x:xmpmeta");
    expect(packet).toContain("<rdf:RDF");

    // Everything outside the spliced segment is byte-identical.
    const before = 2 + APP0_JFIF.length;
    expect(Array.from(out.subarray(0, before))).toEqual(Array.from(original.subarray(0, before)));
    expect(Array.from(out.subarray(app1.end))).toEqual(Array.from(original.subarray(before)));
    expect(out.length).toBe(original.length + (app1.end - app1.start));
  });

  it("inserts directly after SOI when no APP0/APP1 run exists", () => {
    const noApp0 = new Uint8Array([0xff, 0xd8, ...DQT, ...SCAN_TAIL]);
    const out = injectGPano(noApp0, OPTS);
    const segs = listJpegSegments(out);
    expect(segs.map((s) => s.marker)).toEqual([0xe1, 0xdb]);
    expect(segs[0]!.start).toBe(2);
  });

  it("replaces the existing GPano segment instead of duplicating it", () => {
    const once = injectGPano(syntheticJpeg(), OPTS);
    const twice = injectGPano(once, {
      ...OPTS,
      croppedAreaTopPixels: 0,
      croppedAreaImageHeightPixels: 4096,
    });

    const segs = listJpegSegments(twice);
    expect(segs.map((s) => s.marker)).toEqual([0xe0, 0xe1, 0xdb]);
    const xmpSegments = segs.filter((s) => {
      if (s.marker !== 0xe1) return false;
      return segmentPayloadText(twice, s.start, s.end).includes(GPANO_NS);
    });
    expect(xmpSegments).toHaveLength(1);

    const packet = segmentPayloadText(twice, xmpSegments[0]!.start, xmpSegments[0]!.end);
    expect(packet).toContain('GPano:CroppedAreaTopPixels="0"');
    expect(packet).toContain('GPano:CroppedAreaImageHeightPixels="4096"');
    expect(packet).not.toContain('GPano:CroppedAreaTopPixels="512"');

    // Re-injection is a pure replace: same result as injecting into the original.
    const direct = injectGPano(syntheticJpeg(), {
      ...OPTS,
      croppedAreaTopPixels: 0,
      croppedAreaImageHeightPixels: 4096,
    });
    expect(Array.from(twice)).toEqual(Array.from(direct));
  });

  it("leaves non-GPano APP1 segments (e.g. Exif) untouched and inserts after them", () => {
    const exifApp1 = segment(0xe1, [0x45, 0x78, 0x69, 0x66, 0x00, 0x00, 0x4d, 0x4d]); // "Exif\0\0MM"
    const withExif = new Uint8Array([0xff, 0xd8, ...APP0_JFIF, ...exifApp1, ...DQT, ...SCAN_TAIL]);
    const out = injectGPano(withExif, OPTS);
    const segs = listJpegSegments(out);
    expect(segs.map((s) => s.marker)).toEqual([0xe0, 0xe1, 0xe1, 0xdb]);
    // Exif APP1 stays first; the GPano XMP is the second APP1.
    const exifPayload = segmentPayloadText(out, segs[1]!.start, segs[1]!.end);
    expect(exifPayload.startsWith("Exif\0\0")).toBe(true);
    const xmpPayload = segmentPayloadText(out, segs[2]!.start, segs[2]!.end);
    expect(xmpPayload.startsWith(XMP_HEADER)).toBe(true);
  });

  it("rejects non-JPEG input", () => {
    expect(() => injectGPano(new Uint8Array([]), OPTS)).toThrow(/not a JPEG/);
    expect(() =>
      injectGPano(new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]), OPTS),
    ).toThrow(/not a JPEG/);
    expect(() => injectGPano(new Uint8Array([0xff, 0xd9]), OPTS)).toThrow(/not a JPEG/);
  });

  it("rejects truncated or malformed JPEG structure", () => {
    // SOI + APP0 whose declared length runs past the end of the buffer.
    const truncated = new Uint8Array([0xff, 0xd8, 0xff, 0xe0, 0xff, 0xff, 0x00]);
    expect(() => injectGPano(truncated, OPTS)).toThrow(/malformed|truncated/);
  });

  it("rejects invalid pano geometry", () => {
    expect(() =>
      injectGPano(syntheticJpeg(), { ...OPTS, croppedAreaTopPixels: 1.5 }),
    ).toThrow(/non-negative integer/);
    expect(() =>
      injectGPano(syntheticJpeg(), { ...OPTS, croppedAreaImageHeightPixels: 5000 }),
    ).toThrow(/outside the full pano/);
  });
});
