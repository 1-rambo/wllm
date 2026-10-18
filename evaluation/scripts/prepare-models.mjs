#!/usr/bin/env node
import { createHash } from "node:crypto";
import {
  createReadStream,
  createWriteStream,
  existsSync
} from "node:fs";
import {
  mkdir,
  readFile,
  rename,
  rm,
  stat,
  truncate
} from "node:fs/promises";
import { pipeline } from "node:stream/promises";
import { Readable } from "node:stream";
import path from "node:path";
import { modelsRoot, resolveEvaluation } from "../lib/paths.mjs";

const args = parseArgs(process.argv.slice(2));
const manifest = JSON.parse(
  await readFile(resolveEvaluation("config", "models.json"), "utf8")
);
const selected = selectModels(manifest.models, args);
if (args.list) {
  for (const model of manifest.models) {
    console.log(`${model.id}\t${model.filename}\t${model.sizeBytes}`);
  }
  process.exit(0);
}
if (selected.length === 0) {
  throw new Error("no models selected");
}

let failed = false;
for (const model of selected) {
  const destination = path.join(modelsRoot, model.filename);
  if (args.check) {
    const status = await verify(destination, model);
    console.log(`${status ? "READY" : "MISSING"} ${model.id} ${destination}`);
    failed ||= !status;
    continue;
  }
  if (!args.force && await verify(destination, model)) {
    console.log(`READY ${model.id} ${destination}`);
    continue;
  }
  await download(model, destination, args.force);
  if (!(await verify(destination, model))) {
    failed = true;
  }
}
if (failed) process.exit(1);

async function download(model, destination, force) {
  await mkdir(path.dirname(destination), { recursive: true });
  const partial = `${destination}.part`;
  if (force) {
    await rm(destination, { force: true });
    await rm(partial, { force: true });
  }
  let offset = await fileSize(partial);
  const source = process.env.WLLM_MODEL_DOWNLOAD_URL ?? model.downloadUrl ?? model.sourceUrl;
  let response = await fetch(source, {
    headers: offset > 0 ? { Range: `bytes=${offset}-` } : {},
    redirect: "follow"
  });
  if (!response.ok) {
    throw new Error(`download failed for ${model.id}: HTTP ${response.status} ${response.statusText}`);
  }
  if (offset > 0 && response.status !== 206) {
    await truncate(partial, 0);
    offset = 0;
    response = await fetch(source, { redirect: "follow" });
    if (!response.ok) {
      throw new Error(`download failed for ${model.id}: HTTP ${response.status} ${response.statusText}`);
    }
  }
  if (!response.body) throw new Error(`download response for ${model.id} has no body`);
  const expectedLength = Number(response.headers.get("content-length") ?? 0) + offset;
  if (expectedLength > 0 && expectedLength !== model.sizeBytes) {
    console.warn(
      `warning: ${model.id} server length ${expectedLength} differs from manifest ${model.sizeBytes}`
    );
  }
  console.log(`FETCH ${model.id} (${formatBytes(model.sizeBytes)})`);
  await pipeline(
    Readable.fromWeb(response.body),
    createWriteStream(partial, { flags: offset > 0 ? "a" : "w" })
  );
  if (!(await verify(partial, model))) {
    const actualSize = await fileSize(partial);
    const actualHash = await sha256(partial);
    await rm(partial, { force: true });
    throw new Error(
      `verification failed for ${model.id}: size=${actualSize}, sha256=${actualHash}`
    );
  }
  await rm(destination, { force: true });
  await rename(partial, destination);
  console.log(`READY ${model.id} ${destination}`);
}

async function verify(file, model) {
  try {
    const info = await stat(file);
    if (info.size !== model.sizeBytes) return false;
    return (await sha256(file)) === model.sha256;
  } catch {
    return false;
  }
}

async function sha256(file) {
  const hash = createHash("sha256");
  await pipeline(createReadStream(file), hash);
  return hash.digest("hex");
}

async function fileSize(file) {
  try {
    return (await stat(file)).size;
  } catch {
    return 0;
  }
}

function selectModels(models, options) {
  if (options.model) {
    const model = models.find((candidate) => candidate.id === options.model);
    if (!model) throw new Error(`unknown model: ${options.model}`);
    return [model];
  }
  if (options.all || options.check) return models;
  return models.filter((model) => model.usage.includes("e2e"));
}

function parseArgs(argv) {
  const result = {
    model: null,
    all: false,
    check: false,
    force: false,
    list: false
  };
  for (let index = 0; index < argv.length; index += 1) {
    if (argv[index] === "--model") result.model = argv[++index];
    else if (argv[index] === "--all") result.all = true;
    else if (argv[index] === "--check") result.check = true;
    else if (argv[index] === "--force") result.force = true;
    else if (argv[index] === "--list") result.list = true;
  }
  return result;
}

function formatBytes(value) {
  if (value < 1024) return `${value} B`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KiB`;
  if (value < 1024 ** 3) return `${(value / 1024 ** 2).toFixed(1)} MiB`;
  return `${(value / 1024 ** 3).toFixed(2)} GiB`;
}
