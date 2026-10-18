import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const prebuiltRoot = path.join(root, 'vendor', 'wllama-prebuilt');
const pin = JSON.parse(
  await readFile(path.join(root, 'vendor', 'wllama-pin.json'), 'utf8')
);

const artifacts = [
  'wllama/esm/index.js',
  'wllama/esm/wasm/wllama.wasm',
  'wllama-compat/wasm/wllama.js',
  'wllama-compat/wasm/wllama.wasm',
];

const hashes = {};
for (const relativePath of artifacts) {
  const bytes = await readFile(path.join(prebuiltRoot, relativePath));
  hashes[relativePath] = createHash('sha256').update(bytes).digest('hex');
}

const manifest = {
  version: '3.6.1',
  wllamaCommit: pin.commit,
  llamaCppCommit: pin.llamaCppCommit,
  artifacts: hashes,
};

await writeFile(
  path.join(prebuiltRoot, 'manifest.json'),
  `${JSON.stringify(manifest, null, 2)}\n`
);
console.log(`updated ${artifacts.length} prebuilt artifact hashes`);
