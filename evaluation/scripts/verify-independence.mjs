#!/usr/bin/env node
import { readFile, readdir, stat } from "node:fs/promises";
import { spawnSync } from "node:child_process";
import path from "node:path";
import { loadEvaluationConfig, sha256 } from "../lib/config.mjs";
import { evaluationRoot, implRoot } from "../lib/paths.mjs";

const loaded = await loadEvaluationConfig();
const files = await walk(evaluationRoot);
const errors = [];
const warnings = [];

for (const file of files) {
  if (!file.endsWith(".mjs") && !file.endsWith(".js")) continue;
  const text = await readFile(file, "utf8");
  if (/\bfrom\s+["'][^"']*(?:impl\/src|\.\.\/src|\/src\/)[^"']*["']/.test(text)) {
    errors.push(`${path.relative(implRoot, file)} imports impl/src`);
  }
  const result = spawnSync(process.execPath, ["--check", file], {
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"]
  });
  if (result.status !== 0) {
    errors.push(`${path.relative(implRoot, file)} syntax error: ${result.stderr.trim()}`);
  }
}

if (!loaded.config.experiments["e2e-1"]) {
  errors.push("e2e-1 is missing from evaluation/config/e2e.json");
}
for (const id of ["e2e-1", "e2e-2", "abl-1", "abl-2", "abl-3"]) {
  if (!loaded.config.experiments[id]) errors.push(`missing experiment ${id}`);
}

if (loaded.config.modes.pilot) {
  errors.push("pilot mode must not be configured");
}
if (loaded.config.modes.formal?.repeats !== 1) {
  errors.push("formal mode must use exactly one repeat");
}
if (loaded.config.modes.smoke?.repeats !== 1) {
  errors.push("smoke mode must use exactly one repeat");
}
if (!Array.isArray(loaded.config.pressure.smokeSchedule) || loaded.config.pressure.smokeSchedule.length !== 1) {
  errors.push("smoke must use exactly one pressure schedule entry");
}
if (Object.hasOwn(loaded.preregistration, "gates")) {
  errors.push("acceptance gates must not be defined");
}
if (loaded.preregistration.protocol?.repeats !== loaded.config.modes.formal?.repeats) {
  errors.push("preregistered repeats must match formal mode");
}

const forbiddenPaths = [
  "src/experiment",
  "vendor/webarena-runtime",
  ".venv-webarena",
  "src/client/render-pressure-bridge.ts",
  "test/unit/render-pressure-bridge.test.ts",
  "scripts/analyze-bursty-valid.mjs",
  "config/quality-bench.json",
  "dist-demo/engines/webllm.html",
  "scripts/verify-wllama-pin.sh",
  "scripts/sync-wllama-prebuilt.sh",
  "vendor/e5-apps-pins.json"
];
for (const candidate of forbiddenPaths) {
  if (await exists(path.join(implRoot, candidate))) {
    errors.push(`old experiment artifact still exists: ${candidate}`);
  }
}

const activeEntryFiles = [
  "src/engines/wllm-entry.ts",
  "src/engines/wllama-entry.ts",
  "src/engines/engine-page-api.ts",
  "src/engines/wllm-page.ts"
];
for (const relativePath of activeEntryFiles) {
  const text = await readFile(path.join(implRoot, relativePath), "utf8");
  for (const marker of [
    "render-pressure-bridge",
    "observeExternalRenderPressure",
    "renderPressureBridgeUrl",
    "externalPressure"
  ]) {
    if (text.includes(marker)) {
      errors.push(`${relativePath} still references retired pressure bridge marker ${marker}`);
    }
  }
}

if (loaded.config.arms["webllm-per-tab"]) {
  errors.push("webllm-per-tab must not be an active evaluation arm");
}

const appPinsFile = path.join(implRoot, "vendor", "apps-pins.json");
try {
  const appPins = JSON.parse(await readFile(appPinsFile, "utf8"));
  if (appPins.kind !== "apps-pins" || !Array.isArray(appPins.applications) || appPins.applications.length === 0) {
    errors.push("vendor/apps-pins.json must declare pinned applications");
  }
} catch (error) {
  errors.push(`cannot read vendor/apps-pins.json: ${error instanceof Error ? error.message : String(error)}`);
}

const packageJson = JSON.parse(await readFile(path.join(implRoot, "package.json"), "utf8"));
if (packageJson.scripts?.["sync:wllama-prebuilt"]?.includes(".sh") === true) {
  errors.push("sync:wllama-prebuilt must not depend on a shell script");
}
if (packageJson.scripts?.["sync:wllama-prebuilt"]?.includes("sync-wllama-prebuilt.mjs") !== true) {
  errors.push("sync:wllama-prebuilt must point to the Node sync script");
}

const result = {
  ok: errors.length === 0,
  root: evaluationRoot,
  checkedFiles: files.length,
  configSha256: loaded.configSha256,
  preregistrationSha256: sha256(JSON.stringify(loaded.preregistration)),
  errors,
  warnings
};
console.log(JSON.stringify(result, null, 2));
if (errors.length > 0) process.exit(1);

async function walk(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const file = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      if (entry.name === "node_modules" || entry.name === "artifacts") continue;
      files.push(...await walk(file));
    } else {
      files.push(file);
    }
  }
  return files;
}

async function exists(file) {
  try {
    await stat(file);
    return true;
  } catch {
    return false;
  }
}
