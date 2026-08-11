import { useCallback, useEffect, useRef, useState } from "react";
import "@fontsource-variable/archivo";
import "@fontsource/martian-mono/400.css";
import { injectGPano } from "@panoloom/metadata";
import { eulerYXZ, orientationFor } from "@panoloom/shared";
import type { Viewer as PsvViewer } from "@photo-sphere-viewer/core";
import { EngineClient } from "./engine/client";
import { decodeFile, workScaleFor, type DecodedImage } from "./lib/decode";
import { buildProject, parseProject, type ParsedProject } from "./lib/project";
import { deriveProjectName, sanitizeProjectName } from "./lib/projectName";
import { Viewer, type SphereCorrection } from "./components/Viewer";
import { CpEditor } from "./components/CpEditor";
import type { EngineControlPoint, OptimizeFlags } from "./engine/protocol";

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

async function saveBlob(
  blob: Blob,
  suggestedName: string,
  description: string,
  accept: Record<string, string[]>,
) {
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
        types: [{ description, accept }],
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

const saveJpeg = (bytes: Uint8Array, name: string) =>
  saveBlob(
    new Blob([bytes.buffer as ArrayBuffer], { type: "image/jpeg" }),
    name,
    "JPEG image",
    { "image/jpeg": [".jpg"] },
  );

export function App() {
  const engine = useRef<EngineClient | null>(null);
  const workScale = useRef<number | null>(null);
  const files = useRef<Map<number, File>>(new Map());
  const [ready, setReady] = useState(false);
  const [threads, setThreads] = useState(0);
  // Project name: derived from file names until the user renames it.
  const [projectName, setProjectName] = useState("");
  const [editingName, setEditingName] = useState(false);
  const nameEdited = useRef(false);
  const [shots, setShots] = useState<Shot[]>([]);
  const [phase, setPhase] = useState<Phase>({ kind: "empty" });
  const [exporting, setExporting] = useState<ExportState>({ kind: "idle" });
  // Canvas width cap for export; 65535 (the JPEG dimension limit) = native.
  // Full-res compositing decodes several originals per band, so devices
  // reporting little memory default to a smaller target (overridable).
  const [exportWidth, setExportWidth] = useState(() => {
    const gb = (navigator as { deviceMemory?: number }).deviceMemory;
    return gb !== undefined && gb <= 4 ? 8192 : 65535;
  });
  // A parsed .panoproj waiting for the user to re-select its photos.
  const [pendingProject, setPendingProject] = useState<ParsedProject | null>(
    null,
  );
  // Control points (registration coords); null until first generated.
  const [cps, setCps] = useState<EngineControlPoint[] | null>(null);
  const [cpEditorOpen, setCpEditorOpen] = useState(false);
  // Orientation adjustment (degrees) — previewed live, baked on Apply.
  const [adjustOpen, setAdjustOpen] = useState(false);
  const [adjust, setAdjust] = useState({ yaw: 0, pitch: 0, roll: 0 });
  const psv = useRef<PsvViewer | null>(null);
  const exportAborted = useRef(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [elapsed, setElapsed] = useState(0);

  /** Replace a crashed engine and re-import the retained files. Kept in a
   *  ref so the client's onFatal hook always calls the latest version. */
  const recoverEngine = useRef<() => void>(() => {});

  const bootEngine = useCallback((onReady?: (c: EngineClient) => void) => {
    if (typeof WebAssembly === "undefined") {
      setError("this browser can't run PanoLoom — WebAssembly is required");
      return null;
    }
    const c = new EngineClient();
    engine.current = c;
    c.onFatal = () => recoverEngine.current();
    c.init().then(
      () => {
        setThreads(c.threads);
        setReady(true);
        onReady?.(c);
      },
      (e: Error) => setError(`engine failed to start: ${e.message}`),
    );
    return c;
  }, []);

  useEffect(() => {
    const c = bootEngine();
    return () => c?.dispose();
  }, [bootEngine]);

  // Keep the derived project name in sync with the shots until the user
  // renames it (then it's theirs).
  useEffect(() => {
    if (shots.length === 0) {
      nameEdited.current = false;
      setProjectName("");
      setEditingName(false);
      return;
    }
    if (!nameEdited.current) {
      setProjectName(deriveProjectName(shots.map((s) => s.fileName)));
    }
  }, [shots]);

  useEffect(() => {
    if (phase.kind !== "aligning") return;
    const t = setInterval(
      () => setElapsed((Date.now() - phase.startedAt) / 1000),
      100,
    );
    return () => clearInterval(t);
  }, [phase]);

  /** Load a project's photos: ids come from the project, decode at the
   *  project's work scale, then restore the alignment and preview. */
  const importProjectFiles = useCallback(
    async (picked: File[], project: ParsedProject) => {
      const byName = new Map(picked.map((f) => [f.name, f]));
      const missing = project.entries.filter((e) => !byName.has(e.fileName));
      if (missing.length > 0) {
        setError(
          `missing ${missing.length} photo(s): ${missing
            .slice(0, 4)
            .map((m) => m.fileName)
            .join(", ")}${missing.length > 4 ? ", …" : ""}`,
        );
        return;
      }
      try {
        workScale.current = project.workScale;
        for (const entry of project.entries) {
          const file = byName.get(entry.fileName)!;
          const img = await decodeFile(file, project.workScale);
          if (img.fullWidth !== entry.width || img.fullHeight !== entry.height) {
            throw new Error(
              `${entry.fileName}: expected ${entry.width}×${entry.height}, got ${img.fullWidth}×${img.fullHeight}`,
            );
          }
          await engine.current!.addImage(
            entry.id,
            img.rgba,
            img.width,
            img.height,
            img.posePrior,
          );
          const { rgba: _discarded, id: _decodeId, ...meta } = img;
          files.current.set(entry.id, file);
          setShots((s) => [
            ...s,
            { ...meta, id: entry.id, dropped: false, rescued: false },
          ]);
        }
        const result = await engine.current!.importAlignment(
          project.alignmentJson,
        );
        if (project.cps.length > 0) setCps(project.cps);
        setShots((s) =>
          s.map((shot) => ({
            ...shot,
            dropped: result.dropped.includes(shot.id),
            rescued: result.rescued.includes(shot.id),
          })),
        );
        setPendingProject(null);
        setPhase({ kind: "previewing" });
        const p = await engine.current!.renderPreview(4096);
        setPhase({ kind: "preview", ...p });
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        setPhase({ kind: "empty" });
        setShots([]);
        setPendingProject(null);
      }
    },
    [],
  );

  const importFiles = useCallback(
    async (picked: FileList | File[]) => {
      setError(null);
      const all = [...picked];

      // A .panoproj routes to the project-open flow.
      const proj = all.find((f) => f.name.toLowerCase().endsWith(".panoproj"));
      if (proj) {
        try {
          setPendingProject(parseProject(await proj.text()));
          // The project file's own name IS the project name.
          setProjectName(sanitizeProjectName(proj.name.replace(/\.panoproj$/i, "")));
          nameEdited.current = true;
        } catch (e) {
          setError(e instanceof Error ? e.message : String(e));
        }
        return;
      }

      const list = all.filter((f) => /image\/(jpeg|png)/.test(f.type));
      if (list.length === 0) return;
      if (pendingProject) {
        await importProjectFiles(list, pendingProject);
        return;
      }
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
    [pendingProject, importProjectFiles],
  );

  /** Fetch the bundled sample set and import it like user files. */
  const loadSample = useCallback(async () => {
    setError(null);
    try {
      const res = await fetch("samples/ring/manifest.json");
      if (!res.ok) throw new Error("sample set unavailable");
      const manifest = (await res.json()) as { files: string[] };
      const files = await Promise.all(
        manifest.files.map(async (name) => {
          const r = await fetch(`samples/ring/${name}`);
          if (!r.ok) throw new Error(`sample ${name} unavailable`);
          return new File([await r.blob()], name, { type: "image/jpeg" });
        }),
      );
      await importFiles(files);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [importFiles]);

  // Keep the crash-recovery handler current (it needs the latest
  // importFiles closure but is invoked from a long-lived client hook).
  useEffect(() => {
    recoverEngine.current = () => {
      engine.current?.dispose();
      setReady(false);
      setThreads(0);
      setShots([]);
      setPhase({ kind: "empty" });
      setExporting({ kind: "idle" });
      setPendingProject(null);
      setError(
        "the engine ran out of memory or crashed — it has been restarted and your shots re-imported; run Align again",
      );
      const saved = [...files.current.values()];
      files.current.clear();
      bootEngine(() => {
        if (saved.length > 0) void importFiles(saved);
      });
    };
  }, [bootEngine, importFiles]);

  const saveProject = useCallback(async () => {
    try {
      const alignment = await engine.current!.exportAlignment();
      const doc = buildProject(
        shots.map((s) => ({
          id: s.id,
          fileName: s.fileName,
          fullWidth: s.fullWidth,
          fullHeight: s.fullHeight,
          focalLength35mm: s.focalLength35mm,
        })),
        alignment,
        workScale.current ?? 1,
        engine.current!.version,
        cps ?? [],
      );
      await saveBlob(
        new Blob([doc], { type: "application/json" }),
        `${projectName || "panorama"}.panoproj`,
        "PanoLoom project",
        { "application/json": [".panoproj"] },
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [shots, cps, projectName]);

  const runAlign = useCallback(async () => {
    setError(null);
    setElapsed(0);
    setCps(null);
    setCpEditorOpen(false);
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
      let msg = e instanceof Error ? e.message : String(e);
      if (msg.includes("do not overlap")) {
        msg +=
          " — check that the shots belong to one panorama and neighboring frames share ~30% of their view";
      }
      setError(msg);
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

      exportAborted.current = false;
      const loaded = new Set<number>();
      // Decode a band's sources on the main thread and hand them to the
      // worker. Runs CONCURRENTLY with the previous band's composite (the
      // client queue defers the transfers until the worker is free, but
      // the expensive decode overlaps fully).
      const ensureLoaded = async (b: number) => {
        const band = plan.bands[b];
        if (!band) return;
        for (const id of band.needed) {
          if (loaded.has(id) || exportAborted.current) continue;
          loaded.add(id);
          const full = await decodeFull(files.current.get(id)!);
          await engine.current!.exportSetImage(
            id,
            full.rgba,
            full.width,
            full.height,
          );
        }
      };

      await ensureLoaded(0);
      for (let b = 0; b < plan.bands.length; b++) {
        if (exportAborted.current) {
          await engine.current!.cancelExport();
          return;
        }
        setExporting({ kind: "running", band: b, bands: plan.bands.length });
        await Promise.all([
          engine.current!.exportBand(b),
          ensureLoaded(b + 1),
        ]);
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
      if (exportAborted.current) {
        await engine.current!.cancelExport();
        return;
      }

      setExporting({ kind: "encoding" });
      const result = await engine.current!.finishExport(92);
      // The JPEG spans the coverage crop; GPano croppedArea places it on
      // the full sphere so viewers render partial panoramas correctly.
      const withXmp = injectGPano(new Uint8Array(result.jpeg), {
        fullPanoWidthPixels: result.fullWidth,
        fullPanoHeightPixels: result.fullHeight,
        croppedAreaImageWidthPixels: result.width,
        croppedAreaImageHeightPixels: result.height,
        croppedAreaLeftPixels: result.left,
        croppedAreaTopPixels: result.top,
      });
      await saveJpeg(withXmp, `${projectName || "panorama"}.jpg`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setExporting({ kind: "idle" });
    }
  }, [phase.kind, shots, exportWidth, projectName]);

  /** Open the CP editor, generating points on first use. */
  const openCpEditor = useCallback(async () => {
    setError(null);
    try {
      if (!cps) {
        setCps(await engine.current!.autoControlPoints(12));
      }
      setCpEditorOpen(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [cps]);

  /** Run the CP optimizer, then re-render the preview with the result. */
  const optimizeCps = useCallback(
    async (points: EngineControlPoint[], flags: OptimizeFlags) => {
      const report = await engine.current!.optimizeCps(points, flags);
      setPhase({ kind: "previewing" });
      try {
        const p = await engine.current!.renderPreview(4096);
        setPhase({ kind: "preview", ...p });
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
      return report;
    },
    [],
  );

  /** Bake the adjustment into the cameras and re-render the preview. */
  const applyAdjust = useCallback(async () => {
    if (phase.kind !== "preview") return;
    setError(null);
    try {
      const r = orientationFor(adjust.yaw, adjust.pitch, adjust.roll);
      await engine.current!.orient(r.flat());
      setAdjust({ yaw: 0, pitch: 0, roll: 0 });
      setPhase({ kind: "previewing" });
      const p = await engine.current!.renderPreview(4096);
      setPhase({ kind: "preview", ...p });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [phase.kind, adjust]);

  /** Read the current view direction into the yaw/pitch fields. */
  const centerOnView = useCallback(() => {
    const pos = psv.current?.getPosition();
    if (!pos) return;
    setAdjust((a) => ({
      ...a,
      yaw: Math.round((pos.yaw * 180) / Math.PI * 10) / 10,
      pitch: Math.round((pos.pitch * 180) / Math.PI * 10) / 10,
    }));
  }, []);

  const removeShot = useCallback(async (id: number) => {
    try {
      await engine.current!.removeImage(id);
      files.current.delete(id);
      setCps(null);
      setCpEditorOpen(false);
      setShots((s) => {
        const next = s.filter((shot) => shot.id !== id);
        setPhase(next.length > 0 ? { kind: "loaded" } : { kind: "empty" });
        return next.map((shot) => ({
          ...shot,
          dropped: false,
          rescued: false,
        }));
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  const busy =
    phase.kind === "aligning" ||
    phase.kind === "previewing" ||
    exporting.kind !== "idle";
  const canAlign = ready && shots.length >= 2 && !busy;
  const canExport = ready && phase.kind === "preview" && !busy;
  const adjustDirty =
    adjust.yaw !== 0 || adjust.pitch !== 0 || adjust.roll !== 0;
  // Live orientation preview: PSV sets the sphere mesh rotation as
  // rotation.set(-tilt, -pan, roll, "YXZ"), which resolves to the mesh
  // matrix EQUALING our pano-frame rotation — so the correction is the
  // YXZ Euler decomposition of orientationFor(...). Calibrated against
  // the baked orient() per axis and combined (see M6 e2e).
  const euler = eulerYXZ(orientationFor(adjust.yaw, adjust.pitch, adjust.roll));
  const correction: SphereCorrection = {
    pan: -euler.y,
    tilt: -euler.x,
    roll: euler.z,
  };

  return (
    <div className="frame">
      <header className="bar">
        <span className="wordmark">
          Pano<em>Loom</em>
        </span>
        {shots.length > 0 &&
          (editingName ? (
            <input
              className="project-name-input"
              autoFocus
              defaultValue={projectName}
              maxLength={80}
              onFocus={(e) => e.target.select()}
              onBlur={(e) => {
                setProjectName(sanitizeProjectName(e.target.value));
                nameEdited.current = true;
                setEditingName(false);
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter") e.currentTarget.blur();
                if (e.key === "Escape") setEditingName(false);
              }}
            />
          ) : (
            <button
              className="project-name"
              title="rename project — used for saved files"
              onClick={() => setEditingName(true)}
            >
              {projectName}
            </button>
          ))}
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
        {exporting.kind !== "idle" && (
          <button
            className="align-btn ghost"
            onClick={() => {
              exportAborted.current = true;
            }}
            title="Stop after the current band"
          >
            Cancel
          </button>
        )}
        {phase.kind === "preview" && (
          <>
            <button
              className={`align-btn ghost${cpEditorOpen ? " active" : ""}`}
              disabled={busy}
              onClick={() =>
                cpEditorOpen ? setCpEditorOpen(false) : void openCpEditor()
              }
              title="Inspect and edit control points; optimize lens distortion"
            >
              Points
            </button>
            <button
              className={`align-btn ghost${adjustOpen ? " active" : ""}`}
              disabled={busy}
              onClick={() => setAdjustOpen((o) => !o)}
              title="Recenter and level the panorama"
            >
              Adjust
            </button>
            <button
              className="align-btn ghost"
              disabled={busy}
              onClick={() => void saveProject()}
              title="Save alignment as a .panoproj file — reopen later to skip re-aligning"
            >
              Save Project
            </button>
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
              {!busy && (
                <button
                  className="shot-remove"
                  aria-label={`remove ${s.fileName}`}
                  title="remove this shot"
                  onClick={() => void removeShot(s.id)}
                >
                  ×
                </button>
              )}
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
          <>
            <Viewer
              rgba={phase.rgba}
              width={phase.width}
              height={phase.height}
              correction={correction}
              onViewer={(v) => {
                psv.current = v;
              }}
            />
            {adjustOpen && (
              <div className="adjust-panel">
                <div className="adjust-title">adjust orientation</div>
                {(["yaw", "pitch", "roll"] as const).map((axis) => (
                  <label key={axis} className="adjust-row">
                    <span>{axis}</span>
                    <input
                      type="number"
                      step={0.5}
                      min={-180}
                      max={180}
                      value={adjust[axis]}
                      onChange={(e) =>
                        setAdjust((a) => ({
                          ...a,
                          [axis]: Number(e.target.value) || 0,
                        }))
                      }
                    />
                    <span className="unit">°</span>
                  </label>
                ))}
                <button className="adjust-secondary" onClick={centerOnView}>
                  center on current view
                </button>
                <div className="adjust-actions">
                  <button
                    className="adjust-secondary"
                    disabled={!adjustDirty}
                    onClick={() => setAdjust({ yaw: 0, pitch: 0, roll: 0 })}
                  >
                    Reset
                  </button>
                  <button
                    className="align-btn"
                    disabled={!adjustDirty || busy}
                    onClick={() => void applyAdjust()}
                    title="Bake this orientation into the panorama"
                  >
                    Apply
                  </button>
                </div>
              </div>
            )}
          </>
        ) : (
          <label className={`dropzone${dragOver ? " over" : ""}`}>
            <h2>
              {pendingProject
                ? `Select this project's ${pendingProject.entries.length} photos`
                : shots.length === 0
                  ? "Drop your shots here"
                  : `${shots.length} shots on the loom`}
            </h2>
            <p>
              {pendingProject ? (
                <>
                  {pendingProject.entries[0]?.fileName}
                  {pendingProject.entries.length > 1 &&
                    ` … ${pendingProject.entries[pendingProject.entries.length - 1]?.fileName}`}{" "}
                  · <span className="browse">browse files</span>
                </>
              ) : (
                <>
                  JPEG or PNG · overlapping frames · or a .panoproj project ·{" "}
                  <span className="browse">browse files</span>
                  {shots.length === 0 && (
                    <>
                      {" "}
                      · or{" "}
                      <span
                        className="browse"
                        onClick={(e) => {
                          e.preventDefault();
                          void loadSample();
                        }}
                      >
                        try a sample set
                      </span>
                    </>
                  )}
                </>
              )}
            </p>
            <p className="hint">
              everything runs in your browser — nothing is uploaded
            </p>
            <input
              type="file"
              accept="image/jpeg,image/png,.panoproj"
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

        {cpEditorOpen && cps && (
          <CpEditor
            shots={shots
              .filter((s) => !s.dropped)
              .map((s) => ({
                id: s.id,
                fileName: s.fileName,
                width: s.width,
                height: s.height,
              }))}
            files={files.current}
            cps={cps}
            onCpsChange={setCps}
            optimize={optimizeCps}
            onClose={() => setCpEditorOpen(false)}
          />
        )}

        {phase.kind !== "preview" && (
          <div className="credit">
            open source ·{" "}
            <a
              href="https://github.com/PanoLoom/PanoLoom"
              target="_blank"
              rel="noreferrer"
            >
              GitHub
            </a>{" "}
            · Apache-2.0 · engine ported from OpenCV
          </div>
        )}

        {error && (
          <div className="error-note" role="alert">
            {error}
            <button
              className="error-dismiss"
              aria-label="dismiss"
              onClick={() => setError(null)}
            >
              ×
            </button>
          </div>
        )}
      </main>
    </div>
  );
}
