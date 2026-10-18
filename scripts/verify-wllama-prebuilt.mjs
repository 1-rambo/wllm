import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const prebuiltRoot = path.join(root, 'vendor', 'wllama-prebuilt');
const manifest = JSON.parse(
  await readFile(path.join(prebuiltRoot, 'manifest.json'), 'utf8')
);
const pin = JSON.parse(
  await readFile(path.join(root, 'vendor', 'wllama-pin.json'), 'utf8')
);

if (
  manifest.wllamaCommit !== pin.commit ||
  manifest.llamaCppCommit !== pin.llamaCppCommit
) {
  throw new Error('prebuilt manifest does not match vendor/wllama-pin.json');
}

async function sha256(filePath) {
  const bytes = await readFile(filePath);
  return createHash('sha256').update(bytes).digest('hex');
}

for (const [relativePath, expected] of Object.entries(manifest.artifacts)) {
  const actual = await sha256(path.join(prebuiltRoot, relativePath));
  if (actual !== expected) {
    throw new Error(`prebuilt artifact checksum mismatch: ${relativePath}`);
  }
}

const installedArtifacts = [
  ['node_modules/@wllama/wllama/esm/index.js', 'wllama/esm/index.js'],
  [
    'node_modules/@wllama/wllama/esm/wasm/wllama.wasm',
    'wllama/esm/wasm/wllama.wasm',
  ],
  [
    'node_modules/@wllama/wllama-compat/wasm/wllama.js',
    'wllama-compat/wasm/wllama.js',
  ],
  [
    'node_modules/@wllama/wllama-compat/wasm/wllama.wasm',
    'wllama-compat/wasm/wllama.wasm',
  ],
];

for (const [installedPath, manifestPath] of installedArtifacts) {
  const actual = await sha256(path.join(root, installedPath));
  if (actual !== manifest.artifacts[manifestPath]) {
    throw new Error(`installed dependency is not the pinned fork: ${installedPath}`);
  }
}


// The bundled ESM entry contains the Wllama class and all slot-state APIs.
const runtimeSource = await readFile(
  path.join(root, 'node_modules/@wllama/wllama/esm/index.js'),
  'utf8'
);
for (const marker of [
  'exportSlotState',
  'importSlotState',
  'dropSlotState',
  'parkSlotState',
  'resumeSlotState',
  'readParkedSlotState',
  'releaseParkedSlotState',
]) {
  if (!runtimeSource.includes(marker)) {
    throw new Error(`installed wllama fork is missing ${marker}`);
  }
}

console.log(
  `prebuilt wllama OK: ${manifest.wllamaCommit.slice(0, 12)}; llama.cpp OK: ${manifest.llamaCppCommit.slice(0, 12)}`
);
