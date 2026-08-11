import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { VitePWA } from "vite-plugin-pwa";

// COOP/COEP make the page cross-origin isolated so SharedArrayBuffer (and
// therefore wasm threads) are available. Production gets the same headers
// from public/_headers on Cloudflare Pages.
const isolationHeaders = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

export default defineConfig({
  plugins: [
    react(),
    // Installable + offline: precache the app shell and both engine
    // builds (everything runs client-side anyway). The sample set stays
    // network-only to keep the install small.
    VitePWA({
      registerType: "autoUpdate",
      includeAssets: ["icon-192.png", "icon-512.png"],
      manifest: {
        name: "PanoLoom",
        short_name: "PanoLoom",
        description:
          "Free, open-source panorama stitcher that runs entirely in your browser",
        theme_color: "#1a1a1d",
        background_color: "#131315",
        display: "standalone",
        icons: [
          { src: "icon-192.png", sizes: "192x192", type: "image/png" },
          { src: "icon-512.png", sizes: "512x512", type: "image/png" },
          {
            src: "icon-512.png",
            sizes: "512x512",
            type: "image/png",
            purpose: "maskable",
          },
        ],
      },
      workbox: {
        globPatterns: ["**/*.{js,css,html,wasm,woff,woff2}"],
        globIgnores: ["samples/**"],
        maximumFileSizeToCacheInBytes: 4 * 1024 * 1024,
      },
    }),
  ],
  server: { headers: isolationHeaders },
  preview: { headers: isolationHeaders },
  worker: { format: "es" },
  build: { target: "es2022" },
});
