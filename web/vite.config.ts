import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Relative base so the static build works at any path (e.g. GitHub Pages /<repo>/).
export default defineConfig({
  base: "./",
  plugins: [react()],
  build: { outDir: "dist", sourcemap: false, chunkSizeWarningLimit: 600 },
  test: { environment: "node", include: ["src/**/*.test.ts"] },
});
