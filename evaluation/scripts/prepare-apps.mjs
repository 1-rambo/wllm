#!/usr/bin/env node
import { access, mkdir, readFile, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { spawnSync } from "node:child_process";
import path from "node:path";
import { distRoot, implRoot, npmCommand } from "../lib/paths.mjs";

const platform = process.platform;
const npm = npmCommand(platform);
const artifactDirectory = path.join(implRoot, "artifacts", "evaluation");
const manifestFile = path.join(artifactDirectory, "prepared-apps.json");
const appPinsFile = path.join(implRoot, "vendor", "apps-pins.json");
const appPins = JSON.parse(await readFile(appPinsFile, "utf8"));

await mkdir(artifactDirectory, { recursive: true });
for (const application of appPins.applications ?? []) {
  await ensurePinnedApp(application);
}
await ensureDependencies(implRoot, "impl");
await run(npm, ["run", "build"], implRoot, "build impl engine pages");
await verifyFile(path.join(distRoot, "engines", "wllm.html"));
await verifyFile(path.join(distRoot, "engines", "wllama.html"));

const artifexRoot = path.join(implRoot, "vendor", "artifex-webgame");
if (!existsSync(path.join(artifexRoot, "node_modules"))) {
  if (existsSync(path.join(artifexRoot, "package-lock.json"))) {
    await run(npm, ["ci"], artifexRoot, "install Artifex dependencies");
  } else {
    await run(npm, ["install"], artifexRoot, "install Artifex dependencies");
  }
}
await run(npm, ["run", "build"], artifexRoot, "build Artifex");
await verifyFile(path.join(artifexRoot, "dist", "game.html"));

const tavernRoot = path.join(implRoot, "vendor", "holtwick-tavern");
await verifyFile(path.join(tavernRoot, "index.html"));

const manifest = {
  schemaVersion: 1,
  generatedAt: new Date().toISOString(),
  platform,
  implRoot,
  distRoot,
  artifexDist: path.join(artifexRoot, "dist"),
  tavernRoot,
  commits: {
    wllm: gitCommit(implRoot),
    artifex: gitCommit(artifexRoot),
    tavern: gitCommit(tavernRoot)
  },
  appPins: {
    file: path.relative(implRoot, appPinsFile),
    applications: (appPins.applications ?? []).map((application) => ({
      name: application.name,
      repository: application.repository,
      commit: application.commit,
      path: application.path
    }))
  }
};
await writeFile(manifestFile, `${JSON.stringify(manifest, null, 2)}\n`);
console.log(`PREPARED ${manifestFile}`);

async function ensurePinnedApp(application) {
  const directory = path.join(implRoot, application.path);
  if (!existsSync(directory)) {
    await run(
      "git",
      ["clone", "--no-checkout", application.repository, directory],
      implRoot,
      `clone ${application.name}`
    );
    await run(
      "git",
      ["checkout", application.commit],
      directory,
      `checkout pinned ${application.name}`
    );
  }
  const actual = gitCommit(directory);
  if (actual !== application.commit) {
    throw new Error(
      `${application.name} commit mismatch: expected ${application.commit}, found ${actual ?? "unavailable"}. ` +
        "Remove the checkout and rerun npm run evaluation:prepare, or check out the pinned commit manually."
    );
  }
}

async function ensureDependencies(directory, label) {
  if (existsSync(path.join(directory, "node_modules"))) return;
  await run(npm, ["install"], directory, `install ${label} dependencies`);
}

async function run(command, args, cwd, label) {
  console.log(`> ${command} ${args.join(" ")}`);
  const result = spawnSync(command, args, {
    cwd,
    stdio: "inherit",
    env: process.env,
    shell: process.platform === "win32" && /\.cmd$/i.test(command)
  });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(`${label} failed with exit code ${result.status}`);
  }
}

async function verifyFile(file) {
  try {
    await access(file);
  } catch {
    throw new Error(`prepared artifact is missing: ${file}`);
  }
}

function gitCommit(directory) {
  const result = spawnSync("git", ["rev-parse", "HEAD"], {
    cwd: directory,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "ignore"]
  });
  return result.status === 0 ? result.stdout.trim() : null;
}
