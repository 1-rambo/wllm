#!/usr/bin/env node
import { access, cp, mkdir, readdir, rm } from "node:fs/promises";
import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const source = path.join(root, "vendor", "wllama");
const destination = path.join(root, "vendor", "wllama-prebuilt");

run(process.execPath, [path.join(root, "scripts", "verify-wllama-pin.mjs")], {
  label: "verify pinned wllama source"
});

const buildArtifacts = [
  path.join(source, "esm", "index.js"),
  path.join(source, "esm", "wasm", "wllama.wasm"),
  path.join(source, "compat", "wasm", "wllama.js"),
  path.join(source, "compat", "wasm", "wllama.wasm")
];

for (const file of buildArtifacts) {
  await verifyFile(file);
}

await rm(path.join(destination, "wllama", "esm"), { recursive: true, force: true });
await rm(path.join(destination, "wllama", "src"), { recursive: true, force: true });
await rm(path.join(destination, "wllama-compat", "wasm"), { recursive: true, force: true });
await mkdir(path.join(destination, "wllama", "esm"), { recursive: true });
await mkdir(path.join(destination, "wllama", "src"), { recursive: true });
await mkdir(path.join(destination, "wllama-compat", "wasm"), { recursive: true });

await cp(path.join(source, "esm"), path.join(destination, "wllama", "esm"), {
  recursive: true
});
await cp(path.join(source, "compat", "wasm"), path.join(destination, "wllama-compat", "wasm"), {
  recursive: true
});

// Source maps in the ESM build point at the TypeScript tree. Copy only the
// TypeScript sources so test runners do not emit missing-source warnings.
await copyTypeScriptTree(
  path.join(source, "src"),
  path.join(destination, "wllama", "src")
);

run(process.execPath, [path.join(root, "scripts", "update-wllama-prebuilt-manifest.mjs")], {
  label: "update wllama prebuilt manifest"
});
run(process.execPath, [path.join(root, "scripts", "verify-wllama-prebuilt.mjs")], {
  label: "verify wllama prebuilt bundle"
});

console.log(`SYNCED ${destination}`);

function run(command, args, { label }) {
  const result = spawnSync(command, args, {
    cwd: root,
    env: process.env,
    stdio: "inherit"
  });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(`${label} failed with exit code ${result.status}`);
  }
}

async function copyTypeScriptTree(sourceDirectory, destinationDirectory) {
  await mkdir(destinationDirectory, { recursive: true });
  const entries = await readdir(sourceDirectory, { withFileTypes: true });
  for (const entry of entries) {
    const from = path.join(sourceDirectory, entry.name);
    const to = path.join(destinationDirectory, entry.name);
    if (entry.isDirectory()) {
      await copyTypeScriptTree(from, to);
      continue;
    }
    if (entry.isFile() && entry.name.endsWith(".ts")) {
      await cp(from, to);
    }
  }
}

async function verifyFile(file) {
  try {
    await access(file);
  } catch {
    throw new Error(
      `missing fork build artifact: ${file}. Build vendor/wllama before syncing the prebuilt package.`
    );
  }
}
