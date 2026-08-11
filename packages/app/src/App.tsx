import { useCallback, useEffect, useRef, useState } from "react";
import "@fontsource-variable/archivo";
import "@fontsource/martian-mono/400.css";
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

export function App() {
  const engine = useRef<EngineClient | null>(null);
  const workScale = useRef<number | null>(null);
  const [ready, setReady] = useState(false);
  const [shots, setShots] = useState<Shot[]>([]);
  const [phase, setPhase] = useState<Phase>({ kind: "empty" });
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const c = new EngineClient();
    engine.current = c;
    c.init().then(() => setReady(true), (e: Error) => setError(e.message));
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
    async (files: FileList | File[]) => {
      setError(null);
      const list = [...files].filter((f) => /image\/(jpeg|png)/.test(f.type));
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

  const busy = phase.kind === "aligning" || phase.kind === "previewing";
  const canAlign = ready && shots.length >= 2 && !busy;

  return (
    <div className="frame">
      <header className="bar">
        <span className="wordmark">
          Pano<em>Loom</em>
        </span>
        <span className="bar-status">
          {ready ? `engine ready` : `loading engine…`}
          {shots.length > 0 && ` · ${shots.length} shots`}
          {busy && ` · ${elapsed.toFixed(1)}s`}
        </span>
        <span className="bar-spacer" />
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
