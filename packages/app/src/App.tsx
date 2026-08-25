import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "@fontsource-variable/archivo";
import "@fontsource/martian-mono/400.css";
import { injectGPano } from "@panoloom/metadata";
import { eulerYXZ, orientationFor } from "@panoloom/shared";
import type { Viewer as PsvViewer } from "@photo-sphere-viewer/core";
import { EngineClient } from "./engine/client";
import { decodeFile, workScaleFor, type DecodedImage } from "./lib/decode";
import { buildProject, parseProject, type ParsedProject } from "./lib/project";
import { deriveProjectName, sanitizeProjectName } from "./lib/projectName";
import {
  clearSession,
  deleteFile as sessionDeleteFile,
  loadSession,
  saveFile as sessionSaveFile,
  saveState as sessionSaveState,
  type SessionState,
} from "./lib/session";
import { Viewer, type SphereCorrection } from "./components/Viewer";
import { CpEditor } from "./components/CpEditor";
import { MaskEditor, type MaskMap } from "./components/MaskEditor";
import type { EngineControlPoint, OptimizeFlags } from "./engine/protocol";

type Shot = Omit<DecodedImage, "rgba"> & {
  dropped: boolean;
  rescued: boolean;
};

type Phase =
  | { kind: "empty" }
  | { kind: "loaded" }
  | { kind: "aligning" }
  | { kind: "previewing" }
  | { kind: "preview"; rgba: ArrayBuffer; width: number; height: number };

type ExportState =
  | { kind: "idle" }
  /** Planning the band layout and full-res decoding the first band's
   *  sources. Silent on a large set otherwise — `busy` keys off this, so
   *  without it the button appears dead for minutes. */
  | { kind: "planning" }
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

/** Engine stage labels (pipeline.rs `stage_timed!`) in the user's terms.
 *  Unknown labels fall through verbatim, so a new engine stage still shows
 *  something rather than nothing. */
const STAGE_LABELS: Record<string, string> = {
  "orb-detect": "finding features",
  "match-pairs": "matching shots",
  estimate: "estimating camera positions",
  "bundle-adjust": "refining alignment",
  "seam-stage": "planning seams",
  "graph-cut-seams": "cutting seams",
  "compose-warp": "warping",
  "blend-feed": "preparing blend",
  blend: "blending",
};

/** Raw seconds stop being readable well before a large stitch finishes — a
 *  137-shot set runs into the thousands — so switch units as it grows. */
function formatElapsed(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const total = Math.floor(seconds);
  const pad = (n: number) => String(n).padStart(2, "0");
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  return h > 0 ? `${h}h ${pad(m)}m` : `${m}m ${pad(total % 60)}s`;
}

/** Rough wall-clock expectation for a stitch, so a long one reads as normal
 *  rather than broken.
 *
 *  Anchored on a measured in-browser run (33 shots, 10 threads,
 *  registration scale: 23s end to end) against the same set natively
 *  (~11-16s) — so the browser costs roughly 1.5-2x. The upper bands
 *  extrapolate that ratio onto the measured native 137-shot run, now
 *  11.5 min (20s aligning, the rest seam finding) since prior-seeded
 *  bundle adjustment landed. That 1.5-2x factor has NOT been verified at
 *  137 shots, so the band is deliberately wide until a real browser run
 *  at that size says otherwise.
 *
 *  Deliberately coarse, and phrased as a range: cost is driven by how much
 *  the shots OVERLAP, not by their number alone, so a precise figure would
 *  be a false promise. */
function stitchEstimate(shots: number): string | null {
  if (shots < 20) return null;
  if (shots < 45) return "under a minute";
  if (shots < 90) return "a few minutes";
  if (shots < 160) return "20–45 minutes";
  return "over an hour";
}

/** The stitch as a person would describe it. Engine stages are finer than
 *  this and some are too brief to be worth a row, so several map onto one
 *  step; anything unrecognised is ignored rather than shown raw. */
const STITCH_STEPS: { key: string; label: string; stages: string[] }[] = [
  { key: "features", label: "Finding features", stages: ["orb-detect"] },
  { key: "matching", label: "Matching shots", stages: ["match-pairs"] },
  {
    key: "aligning",
    label: "Refining alignment",
    stages: ["estimate", "bundle-adjust"],
  },
  {
    key: "seams",
    label: "Cutting seams",
    stages: ["seam-stage", "graph-cut-seams"],
  },
  {
    key: "blending",
    label: "Blending",
    stages: ["compose-warp", "blend-feed", "blend"],
  },
];

const STEP_OF_STAGE = new Map(
  STITCH_STEPS.flatMap((s) => s.stages.map((g) => [g, s.key] as const)),
);

type StepState = { startedAt: number; endedAt?: number; detail?: string };

/** Engine stages may carry sub-progress as `base:detail` (bundle adjustment
 *  reports `bundle-adjust:340/1000`). Render the friendly name plus the
 *  detail verbatim, so a long stage shows movement instead of a frozen
 *  label. */
function describeStage(stage: string): string {
  const cut = stage.indexOf(":");
  const base = cut === -1 ? stage : stage.slice(0, cut);
  const detail = cut === -1 ? "" : stage.slice(cut + 1);
  const label = STAGE_LABELS[base] ?? base;
  return detail ? `${label} ${detail}` : label;
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
  // Painted seam masks (registration dims) + ids not yet sent to engine.
  const [maskMap, setMaskMap] = useState<MaskMap>(new Map());
  const [maskEditorOpen, setMaskEditorOpen] = useState(false);
  const dirtyMasks = useRef<Set<number>>(new Set());
  // Orientation adjustment (degrees) — previewed live, baked on Apply.
  const [adjustOpen, setAdjustOpen] = useState(false);
  const [adjust, setAdjust] = useState({ yaw: 0, pitch: 0, roll: 0 });
  const psv = useRef<PsvViewer | null>(null);
  const exportAborted = useRef(false);
  // A saved session offered for restore (only when nothing is loaded yet).
  const [resumable, setResumable] = useState<
    (SessionState & { files: Map<number, File> }) | null
  >(null);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [stage, setStage] = useState<string | null>(null);
  const [lastStitch, setLastStitch] = useState<{
    seconds: number;
    shots: number;
  } | null>(null);
  const [steps, setSteps] = useState<Record<string, StepState>>({});
  /** The overlay vanishes the moment the preview renders, taking the last
   *  step's timing with it — so the breakdown stays reachable afterwards. */
  const [summaryOpen, setSummaryOpen] = useState(false);
  const activeStep = useRef<string | null>(null);
  const stitchStart = useRef<number | null>(null);
  /** Read inside the timer effect, which must not re-run when shots change. */
  const shotCount = useRef(0);

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
    c.onProgress = (s) => {
      setStage(s);
      const cut = s.indexOf(":");
      const base = cut === -1 ? s : s.slice(0, cut);
      const detail = cut === -1 ? undefined : s.slice(cut + 1);
      const key = STEP_OF_STAGE.get(base);
      if (!key) return;
      const now = Date.now();
      const changed = activeStep.current !== key;
      const previous = activeStep.current;
      activeStep.current = key;
      setSteps((prev) => {
        const current = prev[key];
        // Redundant repeats are common — several engine stages map to one
        // step. Returning `prev` unchanged skips the render entirely, which
        // is a better filter than a timer: it never drops a real update.
        if (!changed && current && current.detail === detail) return prev;
        const next = { ...prev };
        if (changed && previous && next[previous]) {
          next[previous] = {
            ...next[previous],
            endedAt: now,
            detail: undefined,
          };
        }
        next[key] = { startedAt: changed ? now : (current?.startedAt ?? now), detail };
        return next;
      });
    };
    // ?threads=N caps the pool (diagnostics / constrained machines).
    const raw = new URLSearchParams(location.search).get("threads");
    const cap = raw === null ? undefined : Number(raw);
    c.init(Number.isFinite(cap) && cap! >= 0 ? cap : undefined).then(
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

  // Offer to restore the previous session (once, while nothing is loaded).
  useEffect(() => {
    void loadSession().then((s) => {
      if (s) setResumable(s);
    });
  }, []);

  // Autosave after every milestone that ends in a rendered preview
  // (align, optimize, orient, masks, project load) plus CP/name edits.
  useEffect(() => {
    if (phase.kind !== "preview" || shots.length === 0) return;
    const t = setTimeout(() => {
      void (async () => {
        try {
          const alignmentJson = await engine.current!.exportAlignment();
          await sessionSaveState({
            savedAt: Date.now(),
            projectName,
            nameEdited: nameEdited.current,
            workScale: workScale.current,
            shots: shots.map((s) => ({
              id: s.id,
              fileName: s.fileName,
              fullWidth: s.fullWidth,
              fullHeight: s.fullHeight,
              focalLength35mm: s.focalLength35mm,
              posePrior: s.posePrior,
            })),
            alignmentJson,
            cps,
            masks: [...maskMap.entries()]
              .filter(([, m]) => m.some((v) => v !== 0))
              .map(([id, m]) => {
                const shot = shots.find((s) => s.id === id)!;
                return { id, width: shot.width, height: shot.height, data: m };
              }),
          });
        } catch {
          // Autosave is best-effort.
        }
      })();
    }, 800);
    return () => clearTimeout(t);
  }, [phase, shots, cps, maskMap, projectName]);

  /** Rebuild the whole session from IndexedDB. */
  const restoreSession = useCallback(
    async (s: SessionState & { files: Map<number, File> }) => {
      setResumable(null);
      setError(null);
      try {
        workScale.current = s.workScale;
        nameEdited.current = s.nameEdited;
        setProjectName(s.projectName);
        for (const shot of s.shots) {
          const file = s.files.get(shot.id)!;
          const img = await decodeFile(file, s.workScale);
          await engine.current!.addImage(
            shot.id,
            img.rgba,
            img.width,
            img.height,
            shot.posePrior,
          );
          const { rgba: _discarded, id: _decodeId, ...meta } = img;
          files.current.set(shot.id, file);
          setShots((prev) => [
            ...prev,
            { ...meta, id: shot.id, dropped: false, rescued: false },
          ]);
        }
        const restored: MaskMap = new Map();
        for (const m of s.masks) {
          restored.set(m.id, m.data);
          await engine.current!.setMask(
            m.id,
            m.data.slice().buffer as ArrayBuffer,
            m.width,
            m.height,
          );
        }
        if (restored.size > 0) setMaskMap(restored);
        if (s.cps) setCps(s.cps as EngineControlPoint[]);
        if (s.alignmentJson) {
          const result = await engine.current!.importAlignment(s.alignmentJson);
          setShots((prev) =>
            prev.map((shot) => ({
              ...shot,
              dropped: result.dropped.includes(shot.id),
              rescued: result.rescued.includes(shot.id),
            })),
          );
          setPhase({ kind: "previewing" });
          const p = await engine.current!.renderPreview(4096);
          setPhase({ kind: "preview", ...p });
        } else {
          setPhase({ kind: "loaded" });
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    },
    [],
  );

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

  // The clock has to span aligning AND previewing: seam finding is the
  // longest stage by far, and timing it only through alignment meant the
  // counter vanished exactly when you most wanted it. The total is kept
  // afterwards too — a stitch you looked away from should still be able to
  // tell you how long it took.
  const stitching = phase.kind === "aligning" || phase.kind === "previewing";
  useEffect(() => {
    if (stitching) {
      if (stitchStart.current === null) stitchStart.current = Date.now();
      const t = setInterval(() => {
        if (stitchStart.current !== null) {
          setElapsed((Date.now() - stitchStart.current) / 1000);
        }
      }, 100);
      return () => clearInterval(t);
    }
    if (stitchStart.current !== null) {
      const seconds = (Date.now() - stitchStart.current) / 1000;
      stitchStart.current = null;
      setElapsed(0);
      // Ignore trivial re-renders (a re-preview after an orientation nudge).
      if (seconds > 2) setLastStitch({ seconds, shots: shotCount.current });
    }
    return undefined;
  }, [stitching]);

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
          void sessionSaveFile(entry.id, file);
          setShots((s) => [
            ...s,
            { ...meta, id: entry.id, dropped: false, rescued: false },
          ]);
        }
        // Restore masks before rendering so the preview honors them.
        const restored: MaskMap = new Map();
        for (const m of project.masks) {
          restored.set(m.id, m.data);
          await engine.current!.setMask(
            m.id,
            m.data.slice().buffer as ArrayBuffer,
            m.width,
            m.height,
          );
        }
        if (restored.size > 0) setMaskMap(restored);
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
          void sessionSaveFile(img.id, file);
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
        maskMap,
        new Map(shots.map((s) => [s.id, { width: s.width, height: s.height }])),
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
  }, [shots, cps, projectName, maskMap]);

  const runAlign = useCallback(async () => {
    setError(null);
    setElapsed(0);
    setCps(null);
    setCpEditorOpen(false);
    setSteps({});
    setSummaryOpen(false);
    activeStep.current = null;
    setPhase({ kind: "aligning" });
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
    // Before the await: planning and the first band's full-resolution
    // decodes both happen before any band number exists to report, and on a
    // 137-shot set that is minutes of apparent nothing.
    setExporting({ kind: "planning" });
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

  /** Send changed masks to the engine and re-render the preview. */
  const applyMasks = useCallback(async () => {
    try {
      for (const id of [...dirtyMasks.current]) {
        const mask = maskMap.get(id);
        const shot = shots.find((s) => s.id === id);
        if (!shot) continue;
        if (!mask || mask.every((v) => v === 0)) {
          await engine.current!.clearMask(id);
        } else {
          // Copy: the buffer transfers to the worker.
          const buf = mask.slice().buffer as ArrayBuffer;
          await engine.current!.setMask(id, buf, shot.width, shot.height);
        }
        dirtyMasks.current.delete(id);
      }
      setPhase({ kind: "previewing" });
      const p = await engine.current!.renderPreview(4096);
      setPhase({ kind: "preview", ...p });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [maskMap, shots]);

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
      setMaskEditorOpen(false);
      setMaskMap((m) => {
        const next = new Map(m);
        next.delete(id);
        return next;
      });
      dirtyMasks.current.delete(id);
      void sessionDeleteFile(id);
      setShots((s) => {
        const next = s.filter((shot) => shot.id !== id);
        setPhase(next.length > 0 ? { kind: "loaded" } : { kind: "empty" });
        if (next.length === 0) void clearSession();
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

  // Drop the stage label as soon as the work ends, so a stale one never
  // lingers under the next spinner, and close out the final step so its
  // duration is not left blank.
  useEffect(() => {
    if (busy) return;
    setStage(null);
    const last = activeStep.current;
    activeStep.current = null;
    if (last) {
      setSteps((prev) =>
        prev[last]
          ? { ...prev, [last]: { ...prev[last], endedAt: Date.now(), detail: undefined } }
          : prev,
      );
    }
  }, [busy]);

  useEffect(() => {
    shotCount.current = shots.length;
  }, [shots.length]);
  const canAlign = ready && shots.length >= 2 && !busy;
  const estimate = stitchEstimate(shots.length);
  const canExport = ready && phase.kind === "preview" && !busy;
  const adjustDirty =
    adjust.yaw !== 0 || adjust.pitch !== 0 || adjust.roll !== 0;
  // Live orientation preview: PSV sets the sphere mesh rotation as
  // rotation.set(-tilt, -pan, roll, "YXZ"), which resolves to the mesh
  // matrix EQUALING our pano-frame rotation — so the correction is the
  // YXZ Euler decomposition of orientationFor(...). Calibrated against
  // the baked orient() per axis and combined (see M6 e2e).
  // Memoised: a fresh object every render would re-fire the viewer's
  // sphereCorrection effect on every unrelated state change (stage progress
  // included), which is both wasted work and how the pre-ready PSV crash
  // surfaced.
  // The elapsed clock re-renders this component every 100ms while a stitch
  // runs, so reading the wall clock here keeps the active step's timer live
  // without a second interval.
  const nowTick = Date.now();

  const correction: SphereCorrection = useMemo(() => {
    const euler = eulerYXZ(orientationFor(adjust.yaw, adjust.pitch, adjust.roll));
    return { pan: -euler.y, tilt: -euler.x, roll: euler.z };
  }, [adjust.yaw, adjust.pitch, adjust.roll]);

  return (
    <div className="frame">
      <header className="bar">
        <svg
          className="mark"
          viewBox="0 0 512 512"
          width="24"
          height="24"
          aria-hidden="true"
        >
          <g fill="none" strokeLinecap="round">
            <circle cx="256" cy="256" r="140" stroke="#8e8e96" strokeWidth="42" />
            <path
              d="M 36 296 C 122 296, 154 198, 256 198 C 358 198, 390 296, 476 296"
              stroke="#e8a33d"
              strokeWidth="48"
            />
            <path
              d="M 370.68 141.11 A 140 140 0 0 1 354.99 354.99"
              stroke="#8e8e96"
              strokeWidth="42"
            />
          </g>
        </svg>
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
          {stitching && ` · ${formatElapsed(elapsed)}`}
          {!stitching && lastStitch && (
            <button
              type="button"
              className="stitch-summary"
              onClick={() => setSummaryOpen((o) => !o)}
              title="Per-step timings for the last stitch"
              aria-expanded={summaryOpen}
            >
              {` · stitched ${lastStitch.shots} shots in ${formatElapsed(lastStitch.seconds)}`}
            </button>
          )}
          {exporting.kind === "running" &&
            ` · exporting band ${exporting.band + 1}/${exporting.bands}`}
          {exporting.kind === "planning" && ` · preparing export`}
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
              className={`align-btn ghost${maskEditorOpen ? " active" : ""}`}
              disabled={busy}
              onClick={() => {
                setCpEditorOpen(false);
                setMaskEditorOpen((o) => !o);
              }}
              title="Paint seam masks — avoid moving clouds/people, prefer a shot"
            >
              Mask
            </button>
            <button
              className={`align-btn ghost${cpEditorOpen ? " active" : ""}`}
              disabled={busy}
              onClick={() => {
                setMaskEditorOpen(false);
                if (cpEditorOpen) setCpEditorOpen(false);
                else void openCpEditor();
              }}
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
        {estimate && phase.kind !== "preview" && (
          <span className="eta-hint" title="Large sets are dominated by seam finding">
            ~{estimate}
          </span>
        )}
        <button
          className="align-btn"
          disabled={!canAlign}
          onClick={runAlign}
          title={
            estimate
              ? `${shots.length} shots — usually ${estimate}; you can keep the tab in the background`
              : undefined
          }
        >
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

        {!busy && summaryOpen && lastStitch && (
          <div className="summary-card">
            <div className="summary-head">
              last stitch · {lastStitch.shots} shots ·{" "}
              {formatElapsed(lastStitch.seconds)}
            </div>
            <ol className="steplist">
              {STITCH_STEPS.map((st) => {
                const rec = steps[st.key];
                return (
                  <li key={st.key} className={`steprow ${rec ? "done" : "todo"}`}>
                    <span className="mark" aria-hidden="true">
                      {rec ? "✓" : "–"}
                    </span>
                    <span className="name">{st.label}</span>
                    <span className="meta">
                      {rec?.endedAt !== undefined
                        ? formatElapsed((rec.endedAt - rec.startedAt) / 1000)
                        : "skipped"}
                    </span>
                  </li>
                );
              })}
            </ol>
          </div>
        )}

        {busy && (
          <div className="working">
            {exporting.kind !== "idle" ? (
              <div className="step">
                {exporting.kind === "planning"
                  ? "preparing export"
                  : exporting.kind === "running"
                    ? `exporting band ${exporting.band + 1}/${exporting.bands}`
                    : "encoding JPEG"}
              </div>
            ) : (
              <>
                <div className="step">weaving</div>
                <ol className="steplist">
                  {STITCH_STEPS.map((s) => {
                    const st = steps[s.key];
                    const state = !st ? "todo" : st.endedAt ? "done" : "now";
                    const took =
                      st?.endedAt !== undefined
                        ? formatElapsed((st.endedAt - st.startedAt) / 1000)
                        : st
                          ? formatElapsed((nowTick - st.startedAt) / 1000)
                          : "";
                    return (
                      <li key={s.key} className={`steprow ${state}`}>
                        <span className="mark" aria-hidden="true">
                          {state === "done" ? "✓" : state === "now" ? "•" : "○"}
                        </span>
                        <span className="name">{s.label}</span>
                        <span className="meta">
                          {st?.detail ? `${st.detail} · ` : ""}
                          {took}
                        </span>
                      </li>
                    );
                  })}
                </ol>
              </>
            )}
            {exporting.kind === "planning" && (
              <div className="eta">
                planning bands and decoding sources at full resolution
              </div>
            )}
            {estimate && exporting.kind === "idle" && (
              <div className="eta">
                {shots.length} shots · usually {estimate} · seam finding is
                most of it
              </div>
            )}
          </div>
        )}

        {maskEditorOpen && (
          <MaskEditor
            shots={shots
              .filter((s) => !s.dropped)
              .map((s) => ({
                id: s.id,
                fileName: s.fileName,
                width: s.width,
                height: s.height,
              }))}
            files={files.current}
            masks={maskMap}
            onMasksChange={(m, dirtyId) => {
              setMaskMap(m);
              dirtyMasks.current.add(dirtyId);
            }}
            apply={applyMasks}
            onClose={() => setMaskEditorOpen(false)}
          />
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

        {resumable && shots.length === 0 && (
          <div className="resume-note">
            <span>
              Restore last session — <strong>{resumable.projectName}</strong> (
              {resumable.shots.length} shots
              {resumable.alignmentJson ? ", aligned" : ""})
            </span>
            <button
              className="align-btn"
              onClick={() => void restoreSession(resumable)}
            >
              Restore
            </button>
            <button
              className="adjust-secondary"
              onClick={() => {
                setResumable(null);
                void clearSession();
              }}
            >
              Discard
            </button>
          </div>
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
