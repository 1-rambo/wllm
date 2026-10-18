#!/usr/bin/env node
import { access, readFile, stat } from "node:fs/promises";
import path from "node:path";
import { loadEvaluationConfig, modelsForExperiment } from "../lib/config.mjs";
import { findChromePath } from "../lib/browser.mjs";
import {
  assetsRoot,
  distRoot,
  implRoot,
  modelsRoot,
  platformId
} from "../lib/paths.mjs";

const loaded = await loadEvaluationConfig();
const result = {
  ok: true,
  platform: platformId(),
  platformSupported: {
    e2e: loaded.config.platforms[platformId()]?.e2e === true,
    ablation: loaded.config.platforms[platformId()]?.ablation === true
  },
  node: process.version,
  browser: null,
  files: {},
  models: {},
  warnings: [],
  errors: []
};

try {
  result.browser = await findChromePath();
} catch (error) {
  result.errors.push(error instanceof Error ? error.message : String(error));
}

for (const [label, file] of Object.entries({
  engineDist: path.join(distRoot, "engines", "wllm.html"),
  wllamaDist: path.join(distRoot, "engines", "wllama.html"),
  adapter: path.join(assetsRoot, "app-adapter.js"),
  pressure: path.join(assetsRoot, "pressure.html"),
  tavern: path.join(implRoot, "vendor", "holtwick-tavern", "index.html"),
  artifex: path.join(implRoot, "vendor", "artifex-webgame", "dist", "game.html")
})) {
  result.files[label] = await exists(file);
  if (!result.files[label]) result.errors.push(`missing ${file}`);
}

for (const model of modelsForExperiment(loaded, "e2e-1")) {
  const file = path.join(modelsRoot, model.filename);
  const status = await modelStatus(file, model);
  result.models[model.id] = status;
  if (!status.present) result.warnings.push(`model not prepared: ${model.id}`);
}

if (result.errors.length > 0) result.ok = false;
console.log(JSON.stringify(result, null, 2));
if (!result.ok) process.exit(1);

async function modelStatus(file, model) {
  try {
    const info = await stat(file);
    return {
      present: info.size === model.sizeBytes,
      expectedBytes: model.sizeBytes,
      actualBytes: info.size,
      path: file
    };
  } catch {
    return {
      present: false,
      expectedBytes: model.sizeBytes,
      actualBytes: null,
      path: file
    };
  }
}

async function exists(file) {
  try {
    await access(file);
    return true;
  } catch {
    return false;
  }
}
