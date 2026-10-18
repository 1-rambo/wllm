import { access, readdir, stat } from "node:fs/promises";
import path from "node:path";
import { distRoot, implRoot } from "./paths.mjs";

const sourceRoots = ["src", "engines"];
const sourceFiles = [
  "index.html",
  "package.json",
  "package-lock.json",
  "vite.config.ts",
  "vendor/wllama-prebuilt/manifest.json",
  "vendor/wllama-prebuilt/wllama/esm/wasm/wllama.wasm",
  "vendor/wllama-prebuilt/wllama-compat/wasm/wllama.wasm"
];
const distFiles = ["index.html", "engines/wllm.html", "engines/wllama.html"];
const distRoots = ["assets"];

export async function assertRuntimeBundleFresh() {
  const status = await inspectRuntimeBundle();
  if (status.fresh === false) {
    throw new Error(
      `impl/dist-demo is stale: ${status.newestSource.path} is newer than ` +
        `${status.newestDist.path}. Run "npm run build" before starting an evaluation.`
    );
  }
  return status;
}

export async function inspectRuntimeBundle() {
  const sources = [];
  for (const relative of sourceRoots) {
    sources.push(...(await filesUnder(path.join(implRoot, relative))));
  }
  for (const relative of sourceFiles) {
    const file = path.join(implRoot, relative);
    if (await exists(file)) sources.push(file);
  }

  const outputs = [];
  for (const relative of distFiles) {
    const file = path.join(distRoot, relative);
    await access(file);
    outputs.push(file);
  }
  for (const relative of distRoots) {
    outputs.push(...(await filesUnder(path.join(distRoot, relative))));
  }

  const newestSource = await newestFile(sources);
  const newestDist = await newestFile(outputs);
  return {
    fresh: newestDist.mtimeMs >= newestSource.mtimeMs,
    newestSource: relativeFile(newestSource),
    newestDist: relativeFile(newestDist),
    checkedSourceFiles: sources.length,
    checkedDistFiles: outputs.length
  };
}

async function filesUnder(root) {
  const entries = await readdir(root, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const file = path.join(root, entry.name);
    if (entry.isDirectory()) {
      files.push(...(await filesUnder(file)));
    } else if (entry.isFile()) {
      files.push(file);
    }
  }
  return files;
}

async function newestFile(files) {
  if (files.length === 0) {
    throw new Error("runtime bundle freshness check found no files");
  }
  const entries = await Promise.all(
    files.map(async (file) => ({ file, ...(await stat(file)) }))
  );
  return entries.reduce((newest, entry) =>
    entry.mtimeMs > newest.mtimeMs ? entry : newest
  );
}

function relativeFile(entry) {
  return {
    path: path.relative(implRoot, entry.file),
    mtime: new Date(entry.mtimeMs).toISOString()
  };
}

async function exists(file) {
  try {
    await access(file);
    return true;
  } catch {
    return false;
  }
}
