import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Relative base: the build is served as one folder of shmorsi.github.io (shock-lab/), which copies web/dist from here.
export default defineConfig({
  base: "./",
  plugins: [react()],
  build: { outDir: "dist", sourcemap: false, chunkSizeWarningLimit: 600 },
  test: { environment: "node", include: ["src/**/*.test.ts"] },
});
