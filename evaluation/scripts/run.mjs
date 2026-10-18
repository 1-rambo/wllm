#!/usr/bin/env node
import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const evaluationRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  ".."
);
const e2ePath = path.join(evaluationRoot, "e2e.mjs");
const argv = process.argv.slice(2);

if (argv.includes("--help") || argv.includes("-h")) {
  printHelp();
  process.exit(0);
}

const forwarded = normalizeArgs(argv);
if (forwarded.length === 0) {
  printHelp();
  process.exit(1);
}

const child = spawn(process.execPath, [e2ePath, ...forwarded], {
  cwd: path.dirname(evaluationRoot),
  stdio: "inherit",
  windowsHide: false
});

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => {
    child.kill(signal);
  });
}

child.on("error", (error) => {
  console.error(`failed to start evaluation runner: ${error.message}`);
  process.exitCode = 1;
});

child.on("exit", (code, signal) => {
  if (signal !== null) {
    console.error(`evaluation runner stopped by ${signal}`);
    process.exitCode = 1;
    return;
  }
  process.exitCode = code ?? 1;
});

function normalizeArgs(args) {
  const result = [];
  for (let index = 0; index < args.length; index += 1) {
    const value = args[index];
    if (value === "--") continue;
    result.push(value);
  }
  return result;
}

function printHelp() {
  console.log(`Usage:
  node evaluation/scripts/run.mjs --experiment <id> --mode <smoke|pilot|formal> [options]

Options:
  --experiment <id>       e2e-1, e2e-2, abl-1, abl-2, abl-3
  --mode <mode>           smoke, pilot, or formal
  --model <id>            run one model from the experiment's model set
  --all-models            run every model in the experiment's model set
  --headless              run Chrome without a visible window
  --no-judge              disable the optional judge pass
  --out <directory>       write a single selected model into this directory

Examples:
  node evaluation/scripts/run.mjs --experiment e2e-1 --mode smoke --no-judge
  node evaluation/scripts/run.mjs --experiment e2e-1 --mode pilot --model llama-3.2-1b-instruct
  node evaluation/scripts/run.mjs --experiment e2e-1 --mode formal
  node evaluation/scripts/run.mjs --experiment e2e-2 --mode formal --model qwen2.5-1.5b-instruct`);
}
