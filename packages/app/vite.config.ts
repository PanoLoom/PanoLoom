import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// COOP/COEP make the page cross-origin isolated so SharedArrayBuffer (and
// therefore wasm threads) are available. Production gets the same headers
// from public/_headers on Cloudflare Pages.
const isolationHeaders = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

export default defineConfig({
  plugins: [react()],
  server: { headers: isolationHeaders },
  preview: { headers: isolationHeaders },
  worker: { format: "es" },
  build: { target: "es2022" },
});
