/**
 * Control-point editor: side-by-side pan/zoom views of an image pair with
 * numbered CP markers, an error-sorted point list, and the optimizer
 * (yaw/pitch/roll always; hfov / lens distortion / shift by flag).
 * CP coordinates are registration-scale pixels (engine convention).
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type {
  EngineControlPoint,
  OptimizeFlags,
  OptimizeReport,
} from "../engine/protocol";

export interface CpShot {
  id: number;
  fileName: string;
  /** Registration dims (engine coordinate space for CPs). */
  width: number;
  height: number;
}

interface View {
  scale: number;
  ox: number;
  oy: number;
}

const BITMAP_W = 1400;

function errColor(err: number | null | undefined): string {
  if (err == null) return "#8e8e96";
  if (err < 2) return "#6cb875";
  if (err < 5) return "#e8a33d";
  return "#d4605f";
}

export function CpEditor({
  shots,
  files,
  cps,
  onCpsChange,
  optimize,
  onClose,
}: {
  shots: CpShot[];
  files: Map<number, File>;
  cps: EngineControlPoint[];
  onCpsChange: (cps: EngineControlPoint[]) => void;
  optimize: (
    cps: EngineControlPoint[],
    flags: OptimizeFlags,
  ) => Promise<OptimizeReport>;
  onClose: () => void;
}) {
  // Pairs that have control points.
  const pairs = useMemo(() => {
    const seen = new Map<string, [number, number]>();
    for (const cp of cps) {
      seen.set(`${cp.imgA}:${cp.imgB}`, [cp.imgA, cp.imgB]);
    }
    return [...seen.values()].sort((p, q) => p[0] - q[0] || p[1] - q[1]);
  }, [cps]);
  const [pairIdx, setPairIdx] = useState(0);
  const pair = pairs[Math.min(pairIdx, Math.max(0, pairs.length - 1))];

  const [selected, setSelected] = useState<number | null>(null);
  const [pending, setPending] = useState<{ x: number; y: number } | null>(null);
  const [flags, setFlags] = useState<OptimizeFlags>({
    focal: true,
    distortion: true,
    shift: false,
  });
  const [report, setReport] = useState<OptimizeReport | null>(null);
  const [busy, setBusy] = useState(false);

  // Decoded bitmaps, cached per image id.
  const bitmaps = useRef<Map<number, ImageBitmap>>(new Map());
  const [bitmapTick, setBitmapTick] = useState(0);
  useEffect(() => {
    if (!pair) return;
    let dead = false;
    for (const id of pair) {
      if (bitmaps.current.has(id)) continue;
      const file = files.get(id);
      if (!file) continue;
      void createImageBitmap(file, { resizeWidth: BITMAP_W }).then((bmp) => {
        if (dead) return;
        bitmaps.current.set(id, bmp);
        setBitmapTick((t) => t + 1);
      });
    }
    return () => {
      dead = true;
    };
  }, [pair, files]);

  const pairCps = useMemo(
    () =>
      pair
        ? cps.filter((cp) => cp.imgA === pair[0] && cp.imgB === pair[1])
        : [],
    [cps, pair],
  );

  const removeCp = useCallback(
    (id: number) => {
      onCpsChange(cps.filter((cp) => cp.id !== id));
      setSelected((s) => (s === id ? null : s));
      setReport(null);
    },
    [cps, onCpsChange],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if ((e.key === "Delete" || e.key === "Backspace") && selected != null) {
        removeCp(selected);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selected, removeCp, onClose]);

  const runOptimize = useCallback(async () => {
    setBusy(true);
    try {
      const r = await optimize(cps, flags);
      onCpsChange(
        cps.map((cp, i) => ({ ...cp, errorPx: r.cpErrorsPx[i] ?? null })),
      );
      setReport(r);
    } finally {
      setBusy(false);
    }
  }, [cps, flags, optimize, onCpsChange]);

  if (!pair) {
    return (
      <div className="cp-editor">
        <div className="cp-toolbar">
          <span className="cp-title">control points</span>
          <span className="bar-spacer" />
          <button className="adjust-secondary" onClick={onClose}>
            Close
          </button>
        </div>
        <div className="cp-empty">no control points — align first</div>
      </div>
    );
  }

  const [idA, idB] = pair;
  const shotA = shots.find((s) => s.id === idA);
  const shotB = shots.find((s) => s.id === idB);

  return (
    <div className="cp-editor">
      <div className="cp-toolbar">
        <span className="cp-title">control points</span>
        <button
          className="adjust-secondary"
          disabled={pairIdx === 0}
          onClick={() => {
            setPairIdx((i) => i - 1);
            setPending(null);
            setSelected(null);
          }}
        >
          ‹
        </button>
        <span className="cp-pairlabel">
          {shotA?.fileName} ↔ {shotB?.fileName} ({pairIdx + 1}/{pairs.length})
        </span>
        <button
          className="adjust-secondary"
          disabled={pairIdx >= pairs.length - 1}
          onClick={() => {
            setPairIdx((i) => i + 1);
            setPending(null);
            setSelected(null);
          }}
        >
          ›
        </button>
        <span className="bar-spacer" />
        {(
          [
            ["focal", "hfov"],
            ["distortion", "lens a·b·c"],
            ["shift", "shift d·e"],
          ] as const
        ).map(([key, label]) => (
          <label key={key} className="cp-flag">
            <input
              type="checkbox"
              checked={flags[key]}
              onChange={(e) =>
                setFlags((f) => ({ ...f, [key]: e.target.checked }))
              }
            />
            {label}
          </label>
        ))}
        <button
          className="align-btn"
          disabled={busy || cps.length < 4}
          onClick={() => void runOptimize()}
        >
          {busy ? "Optimizing…" : "Optimize"}
        </button>
        {report && (
          <span className="cp-rms">
            rms {report.rmsPxBefore.toFixed(2)} → {report.rmsPx.toFixed(2)} px
          </span>
        )}
        <button className="adjust-secondary" onClick={onClose}>
          Close
        </button>
      </div>

      <div className="cp-main">
        <PairCanvas
          key={`a${idA}-${bitmapTick}`}
          bitmap={bitmaps.current.get(idA) ?? null}
          shot={shotA!}
          side="A"
          cps={pairCps}
          selected={selected}
          pending={pending}
          onSelect={setSelected}
          onPlace={(x, y) => setPending({ x, y })}
        />
        <PairCanvas
          key={`b${idB}-${bitmapTick}`}
          bitmap={bitmaps.current.get(idB) ?? null}
          shot={shotB!}
          side="B"
          cps={pairCps}
          selected={selected}
          pending={pending}
          onSelect={setSelected}
          onPlace={(x, y) => {
            if (!pending) return;
            const id = cps.reduce((m, cp) => Math.max(m, cp.id), 0) + 1;
            onCpsChange([
              ...cps,
              {
                id,
                imgA: idA,
                imgB: idB,
                xA: pending.x,
                yA: pending.y,
                xB: x,
                yB: y,
                errorPx: null,
              },
            ]);
            setPending(null);
            setSelected(id);
            setReport(null);
          }}
        />

        <div className="cp-table">
          <div className="cp-hint">
            {pending
              ? "click the matching spot in the RIGHT image"
              : "click the LEFT image to add a point · wheel zooms · drag pans"}
          </div>
          {[...pairCps]
            .sort((a, b) => (b.errorPx ?? -1) - (a.errorPx ?? -1))
            .map((cp) => (
              <div
                key={cp.id}
                className={`cp-row${selected === cp.id ? " selected" : ""}`}
                onClick={() => setSelected(cp.id)}
              >
                <span className="cp-dot" style={{ background: errColor(cp.errorPx) }} />
                <span className="cp-id">#{cp.id}</span>
                <span className="cp-err">
                  {cp.errorPx != null ? `${cp.errorPx.toFixed(2)} px` : "—"}
                </span>
                <button
                  className="error-dismiss"
                  aria-label={`delete point ${cp.id}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    removeCp(cp.id);
                  }}
                >
                  ×
                </button>
              </div>
            ))}
        </div>
      </div>
    </div>
  );
}

function PairCanvas({
  bitmap,
  shot,
  side,
  cps,
  selected,
  pending,
  onSelect,
  onPlace,
}: {
  bitmap: ImageBitmap | null;
  shot: CpShot;
  side: "A" | "B";
  cps: EngineControlPoint[];
  selected: number | null;
  pending: { x: number; y: number } | null;
  onSelect: (id: number) => void;
  onPlace: (x: number, y: number) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [view, setView] = useState<View>({ scale: 0, ox: 0, oy: 0 });
  const drag = useRef<{ x: number; y: number; moved: boolean } | null>(null);

  // Registration px -> bitmap px factor.
  const bmpScale = bitmap ? bitmap.width / shot.width : 1;

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !bitmap) return;
    const ctx = canvas.getContext("2d")!;
    const { width: cw, height: chh } = canvas;
    ctx.fillStyle = "#131315";
    ctx.fillRect(0, 0, cw, chh);
    const s = view.scale || Math.min(cw / bitmap.width, chh / bitmap.height);
    ctx.save();
    ctx.translate(view.ox, view.oy);
    ctx.scale(s, s);
    ctx.drawImage(bitmap, 0, 0);
    ctx.restore();
    // Markers.
    for (const cp of cps) {
      const rx = side === "A" ? cp.xA : cp.xB;
      const ry = side === "A" ? cp.yA : cp.yB;
      const x = rx * bmpScale * s + view.ox;
      const y = ry * bmpScale * s + view.oy;
      ctx.strokeStyle = selected === cp.id ? "#e8a33d" : errColor(cp.errorPx);
      ctx.lineWidth = selected === cp.id ? 2.5 : 1.5;
      ctx.beginPath();
      ctx.arc(x, y, 7, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.fillStyle = ctx.strokeStyle;
      ctx.font = "10px 'Martian Mono', monospace";
      ctx.fillText(String(cp.id), x + 9, y - 9);
    }
    if (pending && side === "A") {
      const x = pending.x * bmpScale * s + view.ox;
      const y = pending.y * bmpScale * s + view.oy;
      ctx.strokeStyle = "#e8a33d";
      ctx.setLineDash([3, 3]);
      ctx.strokeRect(x - 8, y - 8, 16, 16);
      ctx.setLineDash([]);
    }
  }, [bitmap, cps, pending, selected, side, view, bmpScale]);

  useEffect(draw, [draw]);

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

  const effScale = (canvas: HTMLCanvasElement) =>
    view.scale ||
    (bitmap
      ? Math.min(canvas.width / bitmap.width, canvas.height / bitmap.height)
      : 1);

  return (
    <div className="cp-canvas">
      <canvas
        ref={canvasRef}
        onWheel={(e) => {
          const canvas = canvasRef.current!;
          const s0 = effScale(canvas);
          const factor = e.deltaY < 0 ? 1.2 : 1 / 1.2;
          const s1 = Math.min(10, Math.max(0.05, s0 * factor));
          const rect = canvas.getBoundingClientRect();
          const mx = e.clientX - rect.left;
          const my = e.clientY - rect.top;
          setView((v) => ({
            scale: s1,
            ox: mx - ((mx - v.ox) / s0) * s1,
            oy: my - ((my - v.oy) / s0) * s1,
          }));
        }}
        onMouseDown={(e) => {
          drag.current = { x: e.clientX, y: e.clientY, moved: false };
        }}
        onMouseMove={(e) => {
          if (!drag.current || !(e.buttons & 1)) return;
          const dx = e.clientX - drag.current.x;
          const dy = e.clientY - drag.current.y;
          if (Math.abs(dx) + Math.abs(dy) > 3) drag.current.moved = true;
          drag.current = { ...drag.current, x: e.clientX, y: e.clientY };
          setView((v) => ({ ...v, ox: v.ox + dx, oy: v.oy + dy }));
        }}
        onMouseUp={(e) => {
          const wasDrag = drag.current?.moved;
          drag.current = null;
          if (wasDrag || !bitmap) return;
          const canvas = canvasRef.current!;
          const rect = canvas.getBoundingClientRect();
          const s = effScale(canvas);
          // Canvas px -> registration px.
          const rx = (e.clientX - rect.left - view.ox) / s / bmpScale;
          const ry = (e.clientY - rect.top - view.oy) / s / bmpScale;
          if (rx < 0 || ry < 0 || rx >= shot.width || ry >= shot.height) return;
          // Near an existing marker? Select it instead.
          for (const cp of cps) {
            const px = side === "A" ? cp.xA : cp.xB;
            const py = side === "A" ? cp.yA : cp.yB;
            if (
              Math.abs(px - rx) * bmpScale * s < 9 &&
              Math.abs(py - ry) * bmpScale * s < 9
            ) {
              onSelect(cp.id);
              return;
            }
          }
          onPlace(rx, ry);
        }}
      />
      <span className="cp-side">{side === "A" ? "left" : "right"}</span>
    </div>
  );
}
