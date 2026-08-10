import { useEffect, useState } from "react";
import { runEngineSmokeTest, type EngineSmokeReport } from "./engine/client";

function Check({ ok, label, detail }: { ok: boolean; label: string; detail?: string }) {
  return (
    <li style={{ display: "flex", gap: 8, alignItems: "baseline", padding: "6px 0" }}>
      <span style={{ color: ok ? "var(--ok)" : "var(--bad)", fontWeight: 600 }}>
        {ok ? "✓" : "✗"}
      </span>
      <span>{label}</span>
      {detail && <span style={{ color: "var(--text-dim)", fontSize: 13 }}>{detail}</span>}
    </li>
  );
}

export function App() {
  const [report, setReport] = useState<EngineSmokeReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    runEngineSmokeTest().then(setReport, (e: Error) => setError(e.message));
  }, []);

  return (
    <main
      style={{
        minHeight: "100dvh",
        display: "grid",
        placeItems: "center",
        padding: 24,
      }}
    >
      <section
        style={{
          background: "var(--panel)",
          border: "1px solid var(--border)",
          borderRadius: 12,
          padding: "28px 32px",
          maxWidth: 460,
          width: "100%",
        }}
      >
        <h1 style={{ margin: 0, fontSize: 22, letterSpacing: 0.3 }}>
          Pano<span style={{ color: "var(--accent)" }}>Loom</span>
        </h1>
        <p style={{ color: "var(--text-dim)", marginTop: 6, fontSize: 14 }}>
          M0 toolchain check — the stitcher itself is on its way.
        </p>
        {error && <p style={{ color: "var(--bad)" }}>Worker failed: {error}</p>}
        {report ? (
          <ul style={{ listStyle: "none", padding: 0, margin: "16px 0 0" }}>
            <Check
              ok={report.engineVersion !== "unknown"}
              label="Rust engine loaded"
              detail={`v${report.engineVersion}`}
            />
            <Check ok={report.grayscaleOk} label="Pixel round trip (JS ↔ wasm)" />
            <Check ok={report.simdSupported} label="wasm SIMD" />
            <Check
              ok={report.crossOriginIsolated}
              label="Cross-origin isolated (COOP/COEP)"
            />
            <Check
              ok={report.threadsAvailable}
              label="Threads available (SharedArrayBuffer)"
            />
            {report.error && <p style={{ color: "var(--bad)" }}>{report.error}</p>}
          </ul>
        ) : (
          !error && <p style={{ color: "var(--text-dim)" }}>Running smoke test…</p>
        )}
      </section>
    </main>
  );
}
