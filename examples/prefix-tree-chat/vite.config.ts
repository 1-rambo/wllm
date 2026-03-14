import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  base: './',
  plugins: [
    react(),
    {
      name: 'isolation',
      configureServer(server) {
        server.middlewares.use((_req, res, next) => {
          res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
          res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
          next();
        });
      },
    },
  ],
  resolve: {
    alias: [
      // 精确匹配包入口 "@wllama/wllama"（不影响子路径 "@wllama/wllama/src/..."）
      {
        find: /^@wllama\/wllama$/,
        replacement: path.resolve(__dirname, '../../src/index.ts'),
      },
    ],
  },
  optimizeDeps: {
    exclude: ['@wllama/wllama'],
  },
});
