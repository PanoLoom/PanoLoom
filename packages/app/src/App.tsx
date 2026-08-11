import { useCallback, useEffect, useRef, useState } from "react";
import "@fontsource-variable/archivo";
import "@fontsource/martian-mono/400.css";
import { injectGPano } from "@panoloom/metadata";
import { EngineClient } from "./engine/client";
import { decodeFile, workScaleFor, type DecodedImage } from "./lib/decode";
import { Viewer } from "./components/Viewer";

type Shot = Omit<DecodedImage, "rgba"> & {
  dropped: boolean;
  rescued: boolean;
};

type Phase =
  | { kind: "empty" }
  | { kind: "loaded" }
  | { kind: "aligning"; startedAt: number }
  | { kind: "previewing" }
  | { kind: "preview"; rgba: ArrayBuffer; width: number; height: number };

type ExportState =
  | { kind: "idle" }
  | { kind: "running"; band: number; bands: number }
  | { kind: "encoding" };

/** Decode a file at ORIGINAL resolution to RGBA for the export path. */
async function decodeFull(
  file: File,
): Promise<{ rgba: ArrayBuffer; width: number; height: number }> {
  const bmp = await createImageBitmap(file);
  const canvas = new OffscreenCanvas(bmp.width, bmp.height);
  const ctx = canvas.getContext("2d")!;
  ctx.drawImage(bmp, 0, 0);
  const data = ctx.getImageData(0, 0, bmp.width, bmp.height);
  bmp.close();
  return {
    rgba: data.data.buffer as ArrayBuffer,
    width: data.width,
    height: data.height,
  };
}

async function saveJpeg(bytes: Uint8Array, suggestedName: string) {
  const blob = new Blob([bytes.buffer as ArrayBuffer], {
    type: "image/jpeg",
  });
  const picker = (
    window as unknown as {
      showSaveFilePicker?: (o: object) => Promise<{
        createWritable(): Promise<{
          write(b: Blob): Promise<void>;
          close(): Promise<void>;
        }>;
      }>;
    }
  ).showSaveFilePicker;
  if (picker) {
    try {
      const handle = await picker({
        suggestedName,
        types: [
          { description: "JPEG image", accept: { "image/jpeg": [".jpg"] } },
        ],
      });
      const w = await handle.createWritable();
      await w.write(blob);
      await w.close();
      return;
    } catch (e) {
      if ((e as Error).name === "AbortError") return; // user cancelled
      // fall through to anchor download
    }
  }
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = suggestedName;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

export function App() {
  const engine = useRef<EngineClient | null>(null);
  const workScale = useRef<number | null>(null);
  const files = useRef<Map<number, File>>(new Map());
  const [ready, setReady] = useState(false);
  const [threads, setThreads] = useState(0);
  const [shots, setShots] = useState<Shot[]>([]);
  const [phase, setPhase] = useState<Phase>({ kind: "empty" });
  const [exporting, setExporting] = useState<ExportState>({ kind: "idle" });
  // Canvas width cap for export; 65535 (the JPEG dimension limit) = native.
  const [exportWidth, setExportWidth] = useState(65535);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const c = new EngineClient();
    engine.current = c;
    c.init().then(
      () => {
        setThreads(c.threads);
        setReady(true);
      },
      (e: Error) => setError(e.message),
    );
    return () => c.dispose();
  }, []);

  useEffect(() => {
    if (phase.kind !== "aligning") return;
    const t = setInterval(
      () => setElapsed((Date.now() - phase.startedAt) / 1000),
      100,
    );
    return () => clearInterval(t);
  }, [phase]);

  const importFiles = useCallback(
    async (picked: FileList | File[]) => {
      setError(null);
      const list = [...picked].filter((f) => /image\/(jpeg|png)/.test(f.type));
      if (list.length === 0) return;
      for (const file of list) {
        try {
          const img = await decodeFile(file, workScale.current);
          workScale.current ??= workScaleFor(img.fullWidth, img.fullHeight);
          await engine.current!.addImage(
            img.id,
            img.rgba,
            img.width,
            img.height,
            img.posePrior,
          );
          const { rgba: _discarded, ...meta } = img;
          files.current.set(img.id, file);
          setShots((s) => [...s, { ...meta, dropped: false, rescued: false }]);
          setPhase((p) => (p.kind === "empty" ? { kind: "loaded" } : p));
        } catch (e) {
          setError(`${file.name}: ${e instanceof Error ? e.message : e}`);
        }
      }
    },
    [],
  );

  const runAlign = useCallback(async () => {
    setError(null);
    setElapsed(0);
    setPhase({ kind: "aligning", startedAt: Date.now() });
    try {
      const result = await engine.current!.align();
      setShots((s) =>
        s.map((shot) => ({
          ...shot,
          dropped: result.dropped.includes(shot.id),
          rescued: result.rescued.includes(shot.id),
        })),
      );
      setPhase({ kind: "previewing" });
      const p = await engine.current!.renderPreview(4096);
      setPhase({ kind: "preview", ...p });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setPhase(shots.length > 0 ? { kind: "loaded" } : { kind: "empty" });
    }
  }, [shots.length]);

  const runExport = useCallback(async () => {
    if (phase.kind !== "preview") return;
    setError(null);
    try {
      const placed = shots.filter((s) => !s.dropped);
      const plan = await engine.current!.beginExport(
        exportWidth,
        placed.map((s) => ({
          id: s.id,
          width: s.fullWidth,
          height: s.fullHeight,
        })),
      );
      setExporting({ kind: "running", band: 0, bands: plan.bands.length });

      const loaded = new Set<number>();
      for (let b = 0; b < plan.bands.length; b++) {
        setExporting({ kind: "running", band: b, bands: plan.bands.length });
        const needed = plan.bands[b]!.needed;
        for (const id of needed) {
          if (!loaded.has(id)) {
            const full = await decodeFull(files.current.get(id)!);
            await engine.current!.exportSetImage(
              id,
              full.rgba,
              full.width,
              full.height,
            );
            loaded.add(id);
          }
        }
        await engine.current!.exportBand(b);
        // Drop images the remaining bands don't need.
        const stillNeeded = new Set(
          plan.bands.slice(b + 1).flatMap((band) => band.needed),
        );
        for (const id of [...loaded]) {
          if (!stillNeeded.has(id)) {
            await engine.current!.exportDropImage(id);
            loaded.delete(id);
          }
        }
      }

      setExporting({ kind: "encoding" });
      const result = await engine.current!.finishExport(92);
      const withXmp = injectGPano(new Uint8Array(result.jpeg), {
        fullPanoWidthPixels: result.width,
        fullPanoHeightPixels: result.height,
        croppedAreaImageWidthPixels: result.width,
        croppedAreaImageHeightPixels: result.height,
        croppedAreaLeftPixels: 0,
        croppedAreaTopPixels: 0,
      });
      await saveJpeg(withXmp, "panoloom-360.jpg");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setExporting({ kind: "idle" });
    }
  }, [phase.kind, shots, exportWidth]);

  const busy =
    phase.kind === "aligning" ||
    phase.kind === "previewing" ||
    exporting.kind !== "idle";
  const canAlign = ready && shots.length >= 2 && !busy;
  const canExport = ready && phase.kind === "preview" && !busy;

  return (
    <div className="frame">
      <header className="bar">
        <span className="wordmark">
          Pano<em>Loom</em>
        </span>
        <span className="bar-status">
          {ready
            ? threads > 0
              ? `engine ready · ${threads} threads`
              : `engine ready`
            : `loading engine…`}
          {shots.length > 0 && ` · ${shots.length} shots`}
          {phase.kind === "aligning" && ` · ${elapsed.toFixed(1)}s`}
          {exporting.kind === "running" &&
            ` · exporting band ${exporting.band + 1}/${exporting.bands}`}
          {exporting.kind === "encoding" && ` · encoding JPEG`}
        </span>
        <span className="bar-spacer" />
        {phase.kind === "preview" && (
          <>
            <select
              className="export-size"
              disabled={!canExport}
              value={exportWidth}
              onChange={(e) => setExportWidth(Number(e.target.value))}
              title="Panorama width"
            >
              <option value={65535}>Full resolution</option>
              <option value={8192}>8192 px</option>
              <option value={4096}>4096 px</option>
            </select>
            <button
              className="align-btn ghost"
              disabled={!canExport}
              onClick={() => void runExport()}
              title="360° JPEG with Photo Sphere metadata"
            >
              Export JPEG
            </button>
          </>
        )}
        <button className="align-btn" disabled={!canAlign} onClick={runAlign}>
          Align &amp; Preview
        </button>
        <div className={`thread${busy ? " busy" : ""}`} />
      </header>

      <aside className="rail">
        {shots.length === 0 ? (
          <div className="rail-empty">
            the filmstrip
            <br />
            waits for shots
          </div>
        ) : (
          shots.map((s, i) => (
            <div
              key={s.id}
              className={`shot${s.dropped ? " dropped" : ""}${s.rescued ? " rescued" : ""}`}
              style={{ animationDelay: `${Math.min(i * 40, 400)}ms` }}
              title={
                s.dropped
                  ? "could not be matched"
                  : s.rescued
                    ? "too few features — placed from gimbal pose metadata"
                    : s.fileName
              }
            >
              <img src={s.thumbnailUrl} alt={s.fileName} />
              <div className="shot-meta">
                <div className="shot-name">{s.fileName}</div>
                <div className="shot-info">
                  {s.fullWidth}×{s.fullHeight}
                  {s.focalLength35mm ? ` · ${s.focalLength35mm}mm` : ""}
                  {s.posePrior ? " · gimbal" : ""}
                  {s.dropped ? " · unmatched" : ""}
                  {s.rescued ? " · placed by pose" : ""}
                </div>
              </div>
            </div>
          ))
        )}
      </aside>

      <main
        className="stage"
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          void importFiles(e.dataTransfer.files);
        }}
      >
        {phase.kind === "preview" ? (
          <Viewer rgba={phase.rgba} width={phase.width} height={phase.height} />
        ) : (
          <label className={`dropzone${dragOver ? " over" : ""}`}>
            <h2>
              {shots.length === 0
                ? "Drop your shots here"
                : `${shots.length} shots on the loom`}
            </h2>
            <p>
              JPEG or PNG · overlapping frames ·{" "}
              <span className="browse">browse files</span>
            </p>
            <p className="hint">
              everything runs in your browser — nothing is uploaded
            </p>
            <input
              type="file"
              accept="image/jpeg,image/png"
              multiple
              onChange={(e) => {
                if (e.target.files) void importFiles(e.target.files);
                e.target.value = "";
              }}
            />
          </label>
        )}

        {busy && (
          <div className="working">
            <div className="step">
              {phase.kind === "aligning"
                ? "weaving · features → matches → bundle adjustment"
                : "rendering preview"}
            </div>
          </div>
        )}

        {error && <div className="error-note">{error}</div>}
      </main>
    </div>
  );
}
