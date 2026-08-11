import { useEffect, useRef } from "react";
import { Viewer as PsvViewer } from "@photo-sphere-viewer/core";
import "@photo-sphere-viewer/core/index.css";

/** 360° preview: renders an equirect RGBA buffer via Photo Sphere Viewer. */
export function Viewer({
  rgba,
  width,
  height,
}: {
  rgba: ArrayBuffer;
  width: number;
  height: number;
}) {
  const el = useRef<HTMLDivElement>(null);
  const viewer = useRef<PsvViewer | null>(null);

  useEffect(() => {
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d")!;
    ctx.putImageData(
      new ImageData(new Uint8ClampedArray(rgba), width, height),
      0,
      0,
    );
    const url = canvas.toDataURL("image/png");

    viewer.current = new PsvViewer({
      container: el.current!,
      panorama: url,
      navbar: ["zoom", "move", "fullscreen"],
      defaultZoomLvl: 20,
    });
    return () => {
      viewer.current?.destroy();
      viewer.current = null;
    };
  }, [rgba, width, height]);

  return <div className="viewer" ref={el} />;
}
