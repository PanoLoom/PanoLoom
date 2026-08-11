/** createImageBitmap with resize options isn't universal (WebKit rejects
 *  them) — fall back to a canvas downscale. */
export async function loadScaledBitmap(
  file: File,
  maxWidth: number,
): Promise<ImageBitmap> {
  try {
    return await createImageBitmap(file, { resizeWidth: maxWidth });
  } catch {
    const full = await createImageBitmap(file);
    if (full.width <= maxWidth) return full;
    const h = Math.round((full.height * maxWidth) / full.width);
    const canvas = new OffscreenCanvas(maxWidth, h);
    canvas.getContext("2d")!.drawImage(full, 0, 0, maxWidth, h);
    full.close();
    return createImageBitmap(canvas);
  }
}
