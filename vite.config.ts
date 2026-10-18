import { defineConfig } from 'vite';
import { fileURLToPath } from 'node:url';
import packageJson from './package.json';

export default defineConfig({
  define: {
    __WLLM_VERSION__: JSON.stringify(packageJson.version),
  },
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  preview: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  worker: {
    format: 'es',
  },
  optimizeDeps: {
    exclude: ['@wllama/wllama'],
  },
  build: {
    target: 'es2022',
    sourcemap: true,
    outDir: 'dist-demo',
    emptyOutDir: true,
    // Engine pages are standalone browser targets used by the demo and the
    // evaluation harness.
    rollupOptions: {
      input: {
        demo: fileURLToPath(new URL('./index.html', import.meta.url)),
        wllamaEngine: fileURLToPath(
          new URL('./engines/wllama.html', import.meta.url)
        ),
        wllmEngine: fileURLToPath(
          new URL('./engines/wllm.html', import.meta.url)
        ),
      },
    },
  },
});
