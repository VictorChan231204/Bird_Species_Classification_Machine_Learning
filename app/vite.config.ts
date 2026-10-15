// Vite settings for the web app. base "./" makes every asset path relative, so the built site works from any
// folder, including a Hugging Face Static Space.
import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  // onnxruntime-web ships its own pre-built bundle; Vite's dependency pre-bundling would break its wasm loading.
  optimizeDeps: { exclude: ["onnxruntime-web"] },
});
