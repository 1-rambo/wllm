# wLLM Browser Serving Runtime

This directory contains the browser-side runtime and the evaluation harness for
the ArtifexWebGame + Holtwick Tavern E2E study. The active comparison is
`ours` versus `wllama-per-tab`; WebLLM is not an active experimental arm.

The formal E2E and ablation runs are intentionally user-started. The repository
provides the runner, environment checks, pinned inputs, and result viewer.

## Repository Composition

The runtime is built on wllama 3.6.1. This repository tracks the pinned
upstream metadata in `vendor/wllama-pin.json`, the local wllama and llama.cpp
patches, and the verified prebuilt wllama packages in
`vendor/wllama-prebuilt/`. Large third-party source checkouts, downloaded
models, build output, and evaluation artifacts are intentionally excluded.
The ArtifexWebGame and Holtwick Tavern checkouts are recreated at their pinned
commits by `npm run evaluation:prepare`.

## Requirements

- Node.js 20 or newer.
- Git.
- Google Chrome or Chromium with WebGPU enabled.
- At least 20 GiB of free disk space for models, build output, browser cache,
  and artifacts.
- At least 16 GiB RAM; 24 GiB or more is recommended for the 11-tab, 3B-model
  cases.

The default Chrome paths are detected automatically. Override them when needed:

```powershell
$env:CHROME_PATH = "C:\Program Files\Google\Chrome\Application\chrome.exe"
```

```bash
export CHROME_PATH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
```

## Windows Collaborator Handoff

For an external Windows machine, use the one-stop handoff instead of the manual
setup below:

```text
Run-Windows-E2E.cmd
```

It runs the environment checks, pinned model/application preparation,
verification, smoke/pilot/formal stages, reporting, and result packaging. The
full checklist and return format are in `Windows-E2E-Handoff.md`. Use
`Resume-Windows-E2E.cmd` after an infrastructure interruption and
`Package-Windows-Results.cmd` after editing `FEEDBACK.md`.

E2E-1 also runs an exploratory, non-gating click-response probe. Seeded
`page.bringToFront()` tab switches measure activation through the second
`requestAnimationFrame`; the report shows coverage and latency separately.

## First Setup

Run all commands below from `impl/`.

```bash
npm ci
npm run evaluation:models
npm run evaluation:prepare
npm run evaluation:check
npm run evaluation:verify
```

`evaluation:models` downloads the four frozen GGUF files and verifies their
size and SHA-256. `evaluation:prepare` installs/builds Artifex and checks the
pinned application commits. `evaluation:check` reports missing runtime inputs.
`evaluation:verify` enforces harness independence and the frozen config.

Useful variants:

```bash
# Download one model only.
node evaluation/scripts/prepare-models.mjs --model llama-3.2-1b-instruct

# List all model ids.
node evaluation/scripts/prepare-models.mjs --list

# Recheck every model without downloading.
node evaluation/scripts/prepare-models.mjs --check
```

## Run an Experiment

The unified runner forwards arguments to the active harness. It works on macOS
and Windows without shell-specific wrappers:

```bash
node evaluation/scripts/run.mjs --experiment e2e-1 --mode smoke --no-judge
node evaluation/scripts/run.mjs --experiment e2e-1 --mode pilot --no-judge
node evaluation/scripts/run.mjs --experiment e2e-1 --mode formal --no-judge
```

Equivalent convenience commands are available:

```bash
npm run evaluation:e2e-1:smoke
npm run evaluation:e2e-1:pilot
npm run evaluation:e2e-1
```

Use `e2e-2` in the same pattern for state recovery, or run the three macOS-only
ablations:

```bash
npm run evaluation:abl-1
npm run evaluation:abl-2
npm run evaluation:abl-3
```

A formal run without `--model` runs the complete model set for that experiment.
Run one model at a time when calibrating or retrying:

```bash
node evaluation/scripts/run.mjs \
  --experiment e2e-1 \
  --mode formal \
  --model qwen2.5-1.5b-instruct
```

Only one formal experiment may run at a time. Do not merge results from
different platforms, models, experiments, or runs.

The complete run matrix, expected record counts, artifacts, failure policy, and
Windows walkthrough are in [`evaluation/README.md`](evaluation/README.md).

## View Results

View the newest run, or select a run by directory or metadata:

```bash
npm run evaluation:report
node evaluation/scripts/report.mjs --experiment e2e-1 --model llama-3.2-1b-instruct
node evaluation/scripts/report.mjs \
  artifacts/evaluation/e2e-priority-pressure/darwin/llama-3.2-1b-instruct/2026-10-15T00-00-00.000Z
```

Machine-readable output:

```bash
node evaluation/scripts/report.mjs --json
```

Each run contains `records.jsonl`, `events.jsonl`, `summary.json`,
`gate-report.json`, `run.json`, and `latest.json`. Treat
`gate-report.json` as the source of truth for pass/fail and completeness.

## Development Checks

If you rebuild the vendored wllama fork, refresh the prebuilt package with the cross-platform command:

```bash
npm run sync:wllama-prebuilt
```

It runs `scripts/sync-wllama-prebuilt.mjs` on both macOS and Windows; no Bash wrapper is required.

```bash
npm run typecheck
npm run evaluation:verify
npm test
npm run build
npm run verify:runtime
```

`npm run model:check` checks the default demo model. The formal evaluation
matrix is checked by `npm run evaluation:check` and the evaluation model
manifest.

## Runtime Layout

- `src/`: browser runtime, scheduler, state manager, and engine-page adapters.
- `evaluation/`: active E2E harness, configs, app adapter, pressure page, and
  report tools.
- `config/` and `patches/`: pinned runtime configuration and source patches.
- `vendor/`: wllama pin metadata, the checked-in prebuilt package, and app pin
  metadata. Third-party source checkouts are local and ignored.
- `artifacts/`: local evaluation outputs, intentionally ignored by Git.

The evaluation harness never imports `src/`. It drives the public engine-page
API and records browser-visible events. The Tavern WebLLM stub is retained only
as a compatibility shim for the original application page; it is not an
experimental arm.
