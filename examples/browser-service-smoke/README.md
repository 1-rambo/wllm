# Browser Service Experiments

论文完整实验设计：

```text
examples/browser-service-smoke/EXPERIMENT_PLAN.zh-CN.md
```

跨设备安装、运行和回收手册：

```text
examples/browser-service-smoke/DEVICE_RUNBOOK.zh-CN.md
```

This directory exercises the same-origin multi-tab serving architecture:

```text
browser tabs -> BrowserLLMClient -> SharedWorker -> one Wllama scheduler/runtime
```

## Validation

```bash
npm run build:glue
npm run build:worker
npm run build:typedef
npm run build:tsup
npm run test:policy
npm run test:browser-service
WLLM_CPU_BUILD_DIR=/path/to/cpu-wasm-build npm run test:browser-service:model
```

For repeatable local validation without re-downloading the model:

```bash
WLLM_CPU_BUILD_DIR=/path/to/cpu-wasm-build \
WLLM_SMOKE_MODEL_FILE=/absolute/path/to/model.gguf \
npm run test:browser-service:model
```

## Experiments

Build Wasm from the current source, then run:

```bash
npm run experiment:browser-service -- \
  --scenario=activate \
  --wasm-build=/path/to/wasm-build \
  --output=artifacts/runs
```

Run a predefined multi-device matrix:

```bash
npm run experiment:browser-service:matrix -- \
  --suite=smoke \
  --group=all \
  --mode=webgpu \
  --trials=1 \
  --device-id=my-device \
  --operator=my-name \
  --device-file=/path/to/device.json \
  --model-file=/path/to/model.gguf \
  --wasm-build=/path/to/webgpu-wasm \
  --output=artifacts/runs/my-device/smoke
```

Available suites:

```text
smoke       engineering validation only
e2e         full protection-progress-responsiveness handoff and contract ablations
core        isolated intent-binding, nested-repair, and observable-boundary experiments
sensitivity parameter sweeps
scaling     client-count and queue-pressure scaling
overhead    model-load, client-attach, and control-plane cost
trace       replay a versioned browser workload trace
```

Scenarios:

- `activate`: transfer execution after a real tab focus change.
- `retain`: pause and resume a logical session across tabs.
- `admit`: resume under constrained state capacity.
- `handoff`: exercise Retain, Admit, and Activate in one end-to-end path.
- `load`: submit mixed foreground/background work across multiple tabs.
- `trace`: replay a versioned browser workload trace.

Useful switches:

```text
--scheduler=foreground-aware|dual-queue|fcfs-no-slice
--slice=<tokens>
--background-slice=<tokens>
--yield=true|false
--event-loop-yield=true|false
--foreground-interrupt=true|false
--retention=lifecycle-aware|lru|ttl|evict-on-pause|retain-until-full
--retention-ttl=<milliseconds>
--admission=monotonic|reactive|reject
--repair-budget=<steps>
--l1-cap=<tokens> --l2-cap=<tokens> --l3-cap=<tokens>
--model-file=/absolute/path/to/model.gguf
--n-ctx=<tokens> --n-batch=<tokens> --n-seq-max=<sequences>
--background-predict=<tokens> --foreground-predict=<tokens>
--session-predict=<tokens> --sessions=<count>
--pressure-sessions=<count> --pressure-predict=<tokens>
--target-prompt-repeats=<count> --pressure-prompt-repeats=<count>
--background-prompt-repeats=<count> --foreground-prompt-repeats=<count>
--client-count=<tabs> --requests=<count> --foreground-every=<count>
--trace-file=/path/to/trace.json
--ui-work-items=<compute items>
--warmup-runs=<count> --warmup-predict=<tokens>
--interaction-samples=<count> --interaction-interval-ms=<milliseconds>
--memory-cap-mb=<MiB> --seed=<integer>
--device-id=<stable-device-id> --operator=<name> --trial=<index>
--device-file=/path/to/device.json
--gpu=true|false --require-webgpu=true|false
--allow-dirty=true|false
```

Each run writes:

```text
<output>/<run-id>/events.jsonl
<output>/<run-id>/summary.json
```

The JSONL stream uses schema `wllm.browser-service.v2`.
Trace runs also write `trace-responses.json`, including generated answers,
reference criteria, and deterministic exact/must-include scores.

Aggregate many run directories:

```bash
npm run experiment:browser-service:collect -- \
  --input=artifacts/runs \
  --output=artifacts/runs/aggregate.csv
```

Prepare an auditable WebArena-derived trace:

```bash
npm run experiment:browser-service:prepare-trace -- \
  --input=/path/to/webarena/config_files/test.raw.json \
  --output=/path/to/webarena-readonly.json
```

Generate per-device paper statistics and paired comparisons:

```bash
npm run experiment:browser-service:summarize -- \
  --input=artifacts/runs/aggregate.csv \
  --output=artifacts/runs/paper-summary
```

Extract request-, admission-, state-, UI-, and timeline-level evidence:

```bash
npm run experiment:browser-service:extract-events -- \
  --input=artifacts/runs \
  --output=artifacts/runs/paper-events
```

Run the complete paper matrix on an anchor device:

```bash
npm run experiment:browser-service:paper -- \
  --profile=primary \
  --mode=webgpu \
  --device-id=my-device \
  --operator=my-name \
  --device-file=/path/to/device.json \
  --model-file=/path/to/model.gguf \
  --wasm-build=/path/to/webgpu-wasm \
  --trace-file=/path/to/webarena-trace.json \
  --output=artifacts/runs/my-device
```

The primary profile requires `--trace-file`; omitting the trace would make the
paper suite incomplete.

Run the single-request/no-reuse wllama comparison on the same model and
source-built Wasm:

```bash
npm run experiment:webarena:run -- \
  --backend=wllama \
  --model-file=/path/to/model.gguf \
  --wasm-build=/path/to/source-built-wasm \
  --eval-count=120 \
  --output=artifacts/runs/webarena-wllama.json
```
