# Windows E2E handoff

This package runs the rapid `e2e-1` and `e2e-2` experiments on a native Windows machine and returns one result zip. Dependencies, GGUF models, and both application checkouts are prepared on the target machine.

## 1. Requirements

- Native Windows. Do not use WSL.
- Node.js 20 LTS or newer.
- Git.
- Google Chrome with working hardware-accelerated WebGPU.
- At least 16 GiB RAM (24 GiB or more recommended), 20 GiB free disk, and a discrete or modern integrated GPU.
- AC power, sleep disabled, screen unlocked, and no other GPU-heavy applications during the run.

Check WebGPU in Chrome by opening `chrome://gpu` and confirming that WebGPU is hardware accelerated. The runner also checks Chrome and installed system resources.

## 2. Package contents

Unzip the whole package and run commands from its root. Do not copy only the runner file. The package records its source commit automatically in `environment.json`.

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
6. Run one simple Llama-1B smoke for `e2e-1` and one for `e2e-2`.
7. Run eight formal combinations: 2 experiments x 4 models, with exactly one repeat each.
8. Generate reports, write `RESULTS.md`, and create one result zip.

Within every pressure cell, both experiments run the arms in the same fixed order: `ours` first and `wllama-per-tab` second. Each arm still gets a fresh browser profile and process. The fixed order avoids carrying GPU-process or native-runtime residue from the preceding arm into a later pressure cell; `run.json` records this policy as `fixed-config-order`.

There are no pilot stages and no acceptance gates. A formal combination that exits with errors or produces incomplete records is preserved, listed in `RESULTS.md`, and does not prevent later model combinations from running.

Every formal trial opens 7 real tabs: three Artifex sessions, three Tavern sessions, and one raw WebGPU pressure page. The pressure schedule is `0, 250, 500, 1000, 1500, 2000, 3000, 4000, clear-return`. Each model/experiment run has `2 x 1 x 9 x 6 = 108` records. Each smoke has `2 x 1 x 1 x 2 = 4` records. Formal E2E runs do not invoke a judge.

E2E-1 also performs an exploratory click-response probe. A seeded sequence activates still-running application tabs through `page.bringToFront()`, waits for the second `requestAnimationFrame` callback, and restores the pressure tab. The probe does not call the priority API, add records to the denominator, or define a gate.

Do not minimize or close Chrome. Do not put the computer to sleep. Keep the runner window open, and do not start a second runner while one is active.

## 4. Return the result

The runner opens Explorer when it finishes. Send this file to the experiment owner:

```text
handoff\windows\<run-id>-results.zip
```

The zip contains the environment record, logs, raw records, summaries, and every failed or retried attempt. Do not delete failed attempts and do not merge runs.

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

Resume keeps the original model set, skips commands that already completed, and creates new attempts only for stages that exited unsuccessfully. It does not overwrite failed artifacts.

If the runner window or computer was closed abruptly, reopen it and use `Resume-Windows-E2E.cmd`; the previous in-flight stage is marked interrupted and retried in a new attempt directory.

Do not manually rerun only a favorable formal run. Do not delete failed attempts. If Chrome or Windows reports a crash, WebGPU/WASM memory failure, or `Invalid magic number`, leave the artifacts intact and send the zip.

## 6. Optional command line

For an unattended start:

```powershell
.\Run-Windows-E2E.cmd --yes --operator "Your Name" --note "Machine description"
```

The default is the full four-model matrix. `--model-set small` uses Llama-1B and Qwen-1.5B for debugging and must not be reported as the full Windows result:

```powershell
.\Run-Windows-E2E.cmd --model-set small --yes
```

To inspect the planned stages without starting anything:

```powershell
.\Run-Windows-E2E.cmd --dry-run
```

The dry-run should show two smoke stages and eight formal stages for the default model set.

## 7. Agent handoff prompt

```text
You are running the rapid wLLM Windows end-to-end evaluation from this package.
Work only inside the extracted package root. Do not modify source, configs,
models, pressure schedules, or tab counts. First run
`.\Run-Windows-E2E.cmd --dry-run`, then verify Chrome WebGPU is hardware
accelerated and the machine has AC power, no sleep, and no other GPU-heavy apps.
Run `.\Run-Windows-E2E.cmd --yes --operator "<operator>" --model-set all`.
Keep the process alive until it exits. If interrupted, preserve artifacts and use
`.\Resume-Windows-E2E.cmd --yes`. When it exits, report the exit code, the
result zip path and SHA-256, all smoke and formal record counts, all execution
errors, and every incomplete or failed combination. Do not describe an
incomplete run as complete.
```
