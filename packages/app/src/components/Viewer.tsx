import { useEffect, useRef } from "react";
import { Viewer as PsvViewer } from "@photo-sphere-viewer/core";
import "@photo-sphere-viewer/core/index.css";

export interface SphereCorrection {
  pan: number;
  tilt: number;
  roll: number;
}

/** 360° preview: renders an equirect RGBA buffer via Photo Sphere Viewer. */
export function Viewer({
  rgba,
  width,
  height,
  correction,
  onViewer,
}: {
  rgba: ArrayBuffer;
  width: number;
  height: number;
  /** Live orientation preview (radians) — applied without re-rendering. */
  correction?: SphereCorrection;
  onViewer?: (v: PsvViewer | null) => void;
}) {
  const el = useRef<HTMLDivElement>(null);
  const viewer = useRef<PsvViewer | null>(null);

  useEffect(() => {
    viewer.current?.setOption(
      "sphereCorrection",
      correction ?? { pan: 0, tilt: 0, roll: 0 },
    );
  }, [correction]);

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
    viewer.current.addEventListener(
      "ready",
      () => {
        // three.js clamps equirect textures at the horizontal edge, which
        // renders a hairline at the 360° wrap. Switch to repeat wrapping
        // so filtering crosses the boundary. Internals are version-fragile;
        // the engine also equalizes the edge columns as a fallback.
        try {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const mesh = (viewer.current as any)?.renderer?.mesh;
          const maps = [mesh?.material?.map, mesh?.material?.[0]?.map].filter(
            Boolean,
          );
          for (const map of maps) {
            map.wrapS = 1000; // THREE.RepeatWrapping
            map.needsUpdate = true;
          }
        } catch {
          // Best-effort only.
        }
        // Test hook.
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (window as any).__psv = viewer.current;
        onViewer?.(viewer.current);
      },
      { once: true },
    );
    return () => {
      onViewer?.(null);
      viewer.current?.destroy();
      viewer.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rgba, width, height]);

  return <div className="viewer" ref={el} />;
}
