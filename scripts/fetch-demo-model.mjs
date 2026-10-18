import { createHash } from 'node:crypto';
import {
  createReadStream,
  createWriteStream,
} from 'node:fs';
import {
  mkdir,
  readFile,
  rename,
  rm,
  stat,
  truncate,
} from 'node:fs/promises';
import { pipeline } from 'node:stream/promises';
import { Readable } from 'node:stream';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '..');
const manifest = JSON.parse(
  await readFile(path.join(root, 'config', 'demo-model.json'), 'utf8')
);
const checkOnly = process.argv.includes('--check');
const force = process.argv.includes('--force');
const destination = path.join(root, 'public', 'models', manifest.filename);
const partial = `${destination}.part`;
const downloadUrl =
  process.env.WLLM_MODEL_DOWNLOAD_URL ?? manifest.downloadUrl;

await mkdir(path.dirname(destination), { recursive: true });

if (!force && (await isExpectedModel(destination))) {
  console.log(`MODEL READY ${destination}`);
  process.exit(0);
}

if (checkOnly) {
  console.error(
    `model is missing or does not match config/demo-model.json: ${destination}`
  );
  console.error(
    `run "npm run model:fetch" (source: ${manifest.sourceUrl})`
  );
  process.exit(1);
}

if (force) {
  await rm(destination, { force: true });
  await rm(partial, { force: true });
}

let offset = await fileSize(partial);
console.log(
  `Fetching ${manifest.label} (${formatBytes(manifest.sizeBytes)})${
    offset > 0 ? `, resuming at ${formatBytes(offset)}` : ''
  }`
);
console.log(`Source: ${downloadUrl}`);

const headers = offset > 0 ? { Range: `bytes=${offset}-` } : {};
let response = await fetch(downloadUrl, { headers, redirect: 'follow' });
if (!response.ok) {
  throw new Error(
    `model download failed: HTTP ${response.status} ${response.statusText}`
  );
}
if (offset > 0 && response.status !== 206) {
  await truncate(partial, 0);
  offset = 0;
  response = await fetch(downloadUrl, { redirect: 'follow' });
  if (!response.ok) {
    throw new Error(
      `model download failed: HTTP ${response.status} ${response.statusText}`
    );
  }
}
if (!response.body) {
  throw new Error('model download returned an empty response body');
}

const expectedLength =
  Number(response.headers.get('content-length') ?? 0) + offset;
if (
  Number.isFinite(expectedLength) &&
  expectedLength > 0 &&
  expectedLength !== manifest.sizeBytes
) {
  console.warn(
    `warning: server length ${expectedLength} differs from manifest ${manifest.sizeBytes}`
  );
}

await pipeline(
  Readable.fromWeb(response.body),
  createWriteStream(partial, { flags: offset > 0 ? 'a' : 'w' })
);

if (!(await isExpectedModel(partial))) {
  const actualSize = await fileSize(partial);
  const actualHash = await sha256(partial);
  await rm(partial, { force: true });
  throw new Error(
    `model verification failed: size=${actualSize}, sha256=${actualHash}`
  );
}

await rm(destination, { force: true });
await rename(partial, destination);
console.log(`MODEL READY ${destination}`);

async function isExpectedModel(file) {
  try {
    const info = await stat(file);
    if (info.size !== manifest.sizeBytes) {
      return false;
    }
    return (await sha256(file)) === manifest.sha256;
  } catch {
    return false;
  }
}

async function fileSize(file) {
  try {
    return (await stat(file)).size;
  } catch {
    return 0;
  }
}

async function sha256(file) {
  const hash = createHash('sha256');
  await pipeline(createReadStream(file), hash);
  return hash.digest('hex');
}

function formatBytes(value) {
  if (value < 1024) {
    return `${value} B`;
  }
  if (value < 1024 * 1024) {
    return `${(value / 1024).toFixed(1)} KiB`;
  }
  if (value < 1024 * 1024 * 1024) {
    return `${(value / (1024 * 1024)).toFixed(1)} MiB`;
  }
  return `${(value / (1024 * 1024 * 1024)).toFixed(2)} GiB`;
}
