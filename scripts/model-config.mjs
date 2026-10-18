import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '..');
const manifest = JSON.parse(
  readFileSync(path.join(root, 'config', 'demo-model.json'), 'utf8')
);

export const demoModelPath = `/models/${manifest.filename}`;
export const demoModelRuntime = manifest.runtime;
export const smokeModelUrl = process.env.WLLM_MODEL_URL ?? demoModelPath;
