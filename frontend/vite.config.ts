import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        timeout: 600000, // 10 min — regen endpoints are slow (GPU inference)
      },
      "/files": "http://localhost:8000",
    },
  },
});
