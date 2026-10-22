# Windows E2E handoff

This package runs the paper-aligned `e2e-1` and `e2e-2` experiments on a native Windows machine and returns one result zip. It is self-contained source: dependencies, GGUF models, and both application checkouts are prepared on the target machine.

Do not edit the frozen configs, reduce the tab count, change the pressure ladder, or merge results from different platforms or attempts.

## 1. Requirements

- Native Windows. Do not use WSL.
- Node.js 20 LTS or newer.
- Git.
- Google Chrome with working hardware-accelerated WebGPU.
- At least 16 GiB RAM (24 GiB or more recommended), 20 GiB free disk, and a discrete or modern integrated GPU.
- AC power, sleep disabled, screen unlocked, and no other GPU-heavy applications during the run.

Check WebGPU in Chrome by opening `chrome://gpu` and confirming that WebGPU is hardware accelerated. The runner also checks Chrome and installed system resources.

## 2. Package contents

Unzip the whole package and run commands from its root. Do not copy only the runner file.

The package records its source commit automatically in `environment.json`.

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
3. Download all four paper models.
4. Clone and build the pinned ArtifexWebGame and Holtwick Tavern applications.
5. Run verification and browser environment checks.
6. Run smoke and pilot gates, then formal `e2e-1` and `e2e-2` for all four models.
   Task-level `wllama-per-tab` crashes or timeouts in a complete stage are preserved as availability outcomes and do not stop later formal stages. Harness, protocol, loading, and ours-side failures still stop the pipeline.
7. Generate reports, write `RESULTS.md`, and create one result zip.

The default matrix is 8 formal runs: 2 experiments x 4 models. Every trial opens 11 real tabs: five Artifex sessions, five Tavern sessions, and one raw WebGPU pressure page. The formal schedule is `0, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, clear-return`; each model/experiment run has `2 x 3 x 10 x 10 = 600` records. Formal E2E runs do not invoke a judge; task-quality scores and recovered-output consistency are not main-table metrics.

E2E-1 also performs an exploratory click-response probe. A seeded sequence activates still-running application tabs through `page.bringToFront()`, waits for the second `requestAnimationFrame` callback, and restores the pressure tab. Visibility state is recorded for diagnosis but is not a validity condition because automation Chrome can report background pages as visible. The probe does not call the priority API, add records to the fixed denominator, or define a gate. The report prints sample coverage and latency split by arm and pressure so the owners can decide whether the result is strong enough to include in the paper.

Do not minimize or close Chrome. Do not put the computer to sleep. Keep the runner window open, and do not start a second runner while one is active.

## 4. Return the result

The runner opens Explorer when it finishes. Send this file to the experiment owner:

```text
handoff\windows\<run-id>-results.zip
```

The zip contains the environment record, logs, raw records, summaries, gate reports, and every failed or retried attempt. Do not delete failed attempts and do not merge runs.

To add comments:

1. Open `handoff\windows\<run-id>\FEEDBACK.md`.
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

Do not manually rerun only a favorable formal run. Do not delete failed attempts. If Chrome or Windows reports a crash, WebGPU/WASM memory failure, or `Invalid magic number`, leave the artifacts intact and send the zip.

`task-failure-outcome` is not a strict gate pass. It means the stage produced its fixed record denominator, and every execution error was localized to a `wllama-per-tab` task or recovery record. Those failed tasks remain in availability counts and are excluded from paired successful-task latency.

## 6. Optional command line

For an unattended start:

```powershell
.\Run-Windows-E2E.cmd --yes --operator "Your Name" --note "Machine description"
```

The default is the full four-model paper matrix. `--model-set small` is only for debugging and must not be reported as a full Windows run:

```powershell
.\Run-Windows-E2E.cmd --model-set small --yes
```

To inspect the plan without starting anything:

```powershell
.\Run-Windows-E2E.cmd --dry-run
```

## 7. Agent handoff prompt

```text
You are running the frozen wLLM Windows end-to-end evaluation from this package.
Work only inside the extracted package root. Do not modify source, configs,
preregistration, pressure schedules, tab counts, models, or gates. First run
`.\Run-Windows-E2E.cmd --dry-run`, then verify Chrome WebGPU is hardware
accelerated and the machine has AC power, no sleep, and no other GPU-heavy apps.
Run `.\Run-Windows-E2E.cmd --yes --operator "<operator>" --model-set all`.
Keep the process alive until it exits. If interrupted, preserve artifacts and use
`.\Resume-Windows-E2E.cmd --yes`. When it exits, report the exit code, the
result zip path and SHA-256, all formal record counts and gate statuses, and any
failure. Do not summarize an incomplete or gate-failing run as a win.
```
