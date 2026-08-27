import * as esbuild from 'esbuild';
import path from 'path';
import { fileURLToPath } from 'url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const stubs = path.join(root, 'scripts', 'browser-service-scope-stubs');
const outfile = process.argv[2];

if (!outfile) {
  throw new Error('usage: node build_browser_service_scope.mjs <output-file>');
}

await esbuild.build({
  entryPoints: [
    path.join(root, 'src', 'workers-code', 'browser-service-scope.ts'),
  ],
  bundle: true,
  minify: true,
  format: 'iife',
  target: 'es2022',
  outfile,
  plugins: [
    {
      name: 'browser-service-generated-stub',
      setup(build) {
        build.onResolve({ filter: /workers-code\/generated$/ }, () => ({
          path: path.join(stubs, 'generated.ts'),
        }));
      },
    },
  ],
  logLevel: 'warning',
});
