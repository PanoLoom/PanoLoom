/**
 * Seam mask editor: paint AVOID (this image's pixels never used there —
 * moving clouds, people, ghosts) or PREFER (this image wins there) on any
 * shot. Masks live at registration resolution; Apply sends the changed
 * ones to the engine and re-renders the preview.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { loadScaledBitmap } from "../lib/bitmap";

export interface MaskShot {
  id: number;
  fileName: string;
  /** Registration dims — the mask raster. */
  width: number;
  height: number;
}

export type MaskMap = Map<number, Uint8Array>;

const AVOID = 1;
const PREFER = 2;

type Tool = "pan" | "avoid" | "prefer" | "erase";

const BITMAP_W = 1400;

export function MaskEditor({
  shots,
  files,
  masks,
  onMasksChange,
  apply,
  onClose,
}: {
  shots: MaskShot[];
  files: Map<number, File>;
  masks: MaskMap;
  onMasksChange: (m: MaskMap, dirtyId: number) => void;
  apply: () => Promise<void>;
  onClose: () => void;
}) {
  const [shotIdx, setShotIdx] = useState(0);
  const shot = shots[Math.min(shotIdx, shots.length - 1)];
  const [tool, setTool] = useState<Tool>("avoid");
  const [brush, setBrush] = useState(30);
  const [busy, setBusy] = useState(false);
  const [dirty, setDirty] = useState(false);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const bitmaps = useRef<Map<number, ImageBitmap>>(new Map());
  const overlay = useRef<HTMLCanvasElement | null>(null);
  const [tick, setTick] = useState(0);
  const view = useRef({ scale: 0, ox: 0, oy: 0 });
  // Drag state is tracked here rather than via e.buttons, which some
  // engines (WebKit) leave 0 on synthesized moves.
  const stroke = useRef<
    | { mode: "pan"; x: number; y: number }
    | { mode: "paint"; x: number; y: number }
    | null
  >(null);

  useEffect(() => {
    if (!shot) return;
    let dead = false;
    if (!bitmaps.current.has(shot.id)) {
      const file = files.get(shot.id);
      if (file) {
        void loadScaledBitmap(file, BITMAP_W).then((bmp) => {
          if (dead) return;
          bitmaps.current.set(shot.id, bmp);
          setTick((t) => t + 1);
        });
      }
    }
    return () => {
      dead = true;
    };
  }, [shot, files]);

  const mask = shot
    ? (masks.get(shot.id) ??
      (() => {
        const m = new Uint8Array(shot.width * shot.height);
        return m;
      })())
    : null;

  /** Rebuild the tinted overlay bitmap from the mask raster. */
  const rebuildOverlay = useCallback(() => {
    if (!shot || !mask) return;
    const c = (overlay.current ??= document.createElement("canvas"));
    c.width = shot.width;
    c.height = shot.height;
    const ctx = c.getContext("2d")!;
    const img = ctx.createImageData(shot.width, shot.height);
    for (let i = 0; i < mask.length; i++) {
      if (mask[i] === AVOID) {
        img.data[4 * i] = 212;
        img.data[4 * i + 1] = 96;
        img.data[4 * i + 2] = 95;
        img.data[4 * i + 3] = 110;
      } else if (mask[i] === PREFER) {
        img.data[4 * i] = 108;
        img.data[4 * i + 1] = 184;
        img.data[4 * i + 2] = 117;
        img.data[4 * i + 3] = 110;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [shot, mask]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const bmp = shot ? bitmaps.current.get(shot.id) : null;
    if (!canvas || !bmp || !shot) return;
    const ctx = canvas.getContext("2d")!;
    ctx.fillStyle = "#131315";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const v = view.current;
    const s =
      v.scale || Math.min(canvas.width / bmp.width, canvas.height / bmp.height);
    ctx.save();
    ctx.translate(v.ox, v.oy);
    ctx.scale(s, s);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(bmp, 0, 0);
    if (overlay.current) {
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(overlay.current, 0, 0, bmp.width, bmp.height);
    }
    ctx.restore();
  }, [shot, tick]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    rebuildOverlay();
    draw();
  }, [rebuildOverlay, draw]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const parent = canvas.parentElement!;
    const resize = () => {
      canvas.width = parent.clientWidth;
      canvas.height = parent.clientHeight;
      draw();
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(parent);
    return () => ro.disconnect();
  }, [draw]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  /** Canvas position -> registration-space mask coordinates. */
  const toMask = (e: React.MouseEvent): { x: number; y: number } | null => {
    const canvas = canvasRef.current;
    const bmp = shot ? bitmaps.current.get(shot.id) : null;
    if (!canvas || !bmp || !shot) return null;
    const v = view.current;
    const s =
      v.scale || Math.min(canvas.width / bmp.width, canvas.height / bmp.height);
    const rect = canvas.getBoundingClientRect();
    const bx = (e.clientX - rect.left - v.ox) / s;
    const by = (e.clientY - rect.top - v.oy) / s;
    return {
      x: (bx / bmp.width) * shot.width,
      y: (by / bmp.height) * shot.height,
    };
  };

  const paint = useCallback(
    (x0: number, y0: number, x1: number, y1: number) => {
      if (!shot || !mask) return;
      const value = tool === "avoid" ? AVOID : tool === "prefer" ? PREFER : 0;
      // Brush radius is in DISPLAY bitmap px; convert to mask px.
      const r = Math.max(1, (brush * shot.width) / BITMAP_W);
      const steps = Math.max(1, Math.ceil(Math.hypot(x1 - x0, y1 - y0) / (r / 2)));
      for (let t = 0; t <= steps; t++) {
        const cx = x0 + ((x1 - x0) * t) / steps;
        const cy = y0 + ((y1 - y0) * t) / steps;
        const xa = Math.max(0, Math.floor(cx - r));
        const xb = Math.min(shot.width - 1, Math.ceil(cx + r));
        const ya = Math.max(0, Math.floor(cy - r));
        const yb = Math.min(shot.height - 1, Math.ceil(cy + r));
        for (let y = ya; y <= yb; y++) {
          for (let x = xa; x <= xb; x++) {
            if ((x - cx) ** 2 + (y - cy) ** 2 <= r * r) {
              mask[y * shot.width + x] = value;
            }
          }
        }
      }
      const next = new Map(masks);
      next.set(shot.id, mask);
      onMasksChange(next, shot.id);
      setDirty(true);
      rebuildOverlay();
      draw();
    },
    [shot, mask, tool, brush, masks, onMasksChange, rebuildOverlay, draw],
  );

  if (!shot) return null;
  const hasMask = mask?.some((v) => v !== 0) ?? false;

  return (
    <div className="cp-editor">
      <div className="cp-toolbar">
        <span className="cp-title">seam masks</span>
        <button
          className="adjust-secondary"
          disabled={shotIdx === 0}
          onClick={() => setShotIdx((i) => i - 1)}
        >
          ‹
        </button>
        <span className="cp-pairlabel">
          {shot.fileName} ({shotIdx + 1}/{shots.length})
          {hasMask ? " · masked" : ""}
        </span>
        <button
          className="adjust-secondary"
          disabled={shotIdx >= shots.length - 1}
          onClick={() => setShotIdx((i) => i + 1)}
        >
          ›
        </button>
        <span className="bar-spacer" />
        {(
          [
            ["pan", "pan"],
            ["avoid", "avoid"],
            ["prefer", "prefer"],
            ["erase", "erase"],
          ] as const
        ).map(([t, label]) => (
          <button
            key={t}
            className={`adjust-secondary mask-tool-${t}${tool === t ? " active-tool" : ""}`}
            onClick={() => setTool(t)}
          >
            {label}
          </button>
        ))}
        <label className="cp-flag">
          size
          <input
            type="range"
            min={6}
            max={120}
            value={brush}
            onChange={(e) => setBrush(Number(e.target.value))}
          />
        </label>
        <button
          className="adjust-secondary"
          disabled={!hasMask}
          onClick={() => {
            const next = new Map(masks);
            next.set(shot.id, new Uint8Array(shot.width * shot.height));
            onMasksChange(next, shot.id);
            setDirty(true);
            setTick((t) => t + 1);
          }}
        >
          Clear shot
        </button>
        <button
          className="align-btn"
          disabled={busy || !dirty}
          onClick={() => {
            setBusy(true);
            void apply().finally(() => {
              setBusy(false);
              setDirty(false);
            });
          }}
        >
          {busy ? "Rendering…" : "Apply"}
        </button>
        <button className="adjust-secondary" onClick={onClose}>
          Close
        </button>
      </div>

      <div className="mask-main">
        <canvas
          ref={canvasRef}
          className={tool === "pan" ? "grabbable" : ""}
          onWheel={(e) => {
            const canvas = canvasRef.current!;
            const bmp = bitmaps.current.get(shot.id);
            if (!bmp) return;
            const v = view.current;
            const s0 =
              v.scale ||
              Math.min(canvas.width / bmp.width, canvas.height / bmp.height);
            const s1 = Math.min(
              12,
              Math.max(0.05, s0 * (e.deltaY < 0 ? 1.2 : 1 / 1.2)),
            );
            const rect = canvas.getBoundingClientRect();
            const mx = e.clientX - rect.left;
            const my = e.clientY - rect.top;
            view.current = {
              scale: s1,
              ox: mx - ((mx - v.ox) / s0) * s1,
              oy: my - ((my - v.oy) / s0) * s1,
            };
            draw();
          }}
          onMouseDown={(e) => {
            if (tool === "pan" || e.button === 1 || e.altKey) {
              stroke.current = { mode: "pan", x: e.clientX, y: e.clientY };
              return;
            }
            const p = toMask(e);
            if (p) {
              paint(p.x, p.y, p.x, p.y);
              stroke.current = { mode: "paint", x: p.x, y: p.y };
            }
          }}
          onMouseMove={(e) => {
            const s = stroke.current;
            if (!s) return;
            if (s.mode === "pan") {
              view.current = {
                ...view.current,
                ox: view.current.ox + e.clientX - s.x,
                oy: view.current.oy + e.clientY - s.y,
              };
              stroke.current = { mode: "pan", x: e.clientX, y: e.clientY };
              draw();
              return;
            }
            const p = toMask(e);
            if (p) {
              paint(s.x, s.y, p.x, p.y);
              stroke.current = { mode: "paint", x: p.x, y: p.y };
            }
          }}
          onMouseUp={() => {
            stroke.current = null;
          }}
          onMouseLeave={() => {
            stroke.current = null;
          }}
        />
        <div className="mask-hint">
          paint <span style={{ color: "#d4605f" }}>avoid</span> over moving
          clouds/people you don&apos;t want from this shot ·{" "}
          <span style={{ color: "#6cb875" }}>prefer</span> forces this shot to
          win · Alt-drag pans
        </div>
      </div>
    </div>
  );
}
