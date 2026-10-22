# Windows E2E handoff

This package runs the current paper-facing `e2e-1` and `e2e-2` experiments on a native Windows machine and returns one result zip.

Use the exact tag or commit provided by the experiment owner. Do not run from a later `master` checkout and do not edit experiment configs.

## 1. Requirements

- Native Windows. Do not use WSL.
- Node.js 20 LTS or newer.
- Git.
- Google Chrome with working hardware-accelerated WebGPU.
- At least 16 GiB RAM (24 GiB or more recommended), 20 GiB free disk, and a discrete or modern integrated GPU.
- AC power, sleep disabled, screen unlocked, and no other GPU-heavy applications during the run.

Check WebGPU in Chrome by opening `chrome://gpu` and confirming that WebGPU is hardware accelerated. The runner also checks Chrome and installed system resources.

## 2. Get the exact code

Open PowerShell:

```powershell
cd $HOME
git clone https://github.com/1-rambo/wllm.git
cd wllm
git fetch --tags
git checkout windows-e2e-handoff-v2
git rev-parse HEAD
```

Send the printed commit hash to the experiment owner once, so we can record exactly which code produced the results.

If Chrome is not installed in a standard location:

```powershell
$env:CHROME_PATH = "C:\Program Files\Google\Chrome\Application\chrome.exe"
```

## 3. Run

Double-click:

```text
Run-Windows-E2E.cmd
```

The runner performs these stages automatically:

1. Check Windows, Node, Git, Chrome, RAM, disk, GPU, and AC power.
2. Install pinned dependencies.
3. Download the two small models and the judge model.
4. Clone and build the pinned ArtifexWebGame and Holtwick Tavern applications.
5. Run verification and browser environment checks.
6. Run smoke and pilot gates, then formal `e2e-1` and `e2e-2` for both models.
   Task-level `wllama-per-tab` crashes or timeouts in a complete stage are preserved as availability outcomes and do not stop later formal stages. Harness, protocol, loading, and ours-side failures still stop the pipeline.
7. Generate reports, write `RESULTS.md`, and create one result zip.

The default matrix is 4 formal runs: 2 applications scenarios x 2 models (`llama-3.2-1b-instruct`, `qwen2.5-1.5b-instruct`). Every formal run should contain 324 task/recovery records and 9 gates. First-run downloads and builds take substantial time; the full matrix can run for many hours.

Do not minimize or close Chrome. Do not put the computer to sleep. Keep the runner window open, and do not start a second runner while one is active.

## 4. Return the result

The runner opens Explorer when it finishes. Send this file to the experiment owner:

```text
impl\handoff\windows\<run-id>-results.zip
```

The zip contains the environment record, logs, raw records, summaries, gate reports, and every failed or retried attempt. Do not merge macOS and Windows measurements.

To add comments:

1. Open `impl\handoff\windows\<run-id>\FEEDBACK.md`.
2. Fill in the feedback lines.
3. Double-click `Package-Windows-Results.cmd`.
4. Send the refreshed zip.

## 5. If the run stops

When the runner reports an error, it still creates a partial result zip and preserves failed run directories. Fix only the external problem (for example, reconnect the network, close another GPU application, or reconnect AC power), then double-click:

```text
Resume-Windows-E2E.cmd
```

The resume command keeps the original model set, skips completed stages, and creates new attempt directories for failed or incomplete stages. It does not overwrite failed artifacts.

If the runner window or computer was closed abruptly, reopen it and use `Resume-Windows-E2E.cmd`; the previous in-flight stage is marked interrupted and retried in a new attempt directory.

Do not manually rerun only a favorable formal run. Do not delete failed attempts. If Chrome or Windows reports a crash, WebGPU/WASM memory failure, or missing judge output, leave the artifacts intact and send the zip.

`task-failure-outcome` is not a strict gate pass. It means the stage produced its fixed record denominator, and every execution error was localized to a `wllama-per-tab` task or recovery record. Those failed tasks remain in availability counts and are excluded from paired successful-task latency.

## 6. Optional command line

For an unattended start:

```powershell
.\Run-Windows-E2E.cmd --yes --operator "Your Name" --note "Machine description"
```

To run all four configured models instead of the default two, use `--model-set all`, but do this only when the experiment owner explicitly requests it:

```powershell
.\Run-Windows-E2E.cmd --model-set all --yes
```

To inspect the plan without starting anything:

```powershell
.\Run-Windows-E2E.cmd --dry-run
```
