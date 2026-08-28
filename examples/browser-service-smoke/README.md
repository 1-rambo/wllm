# Browser Service Experiments

中文完整执行手册：

```text
examples/browser-service-smoke/EXPERIMENT_PLAN.zh-CN.md
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

Scenarios:

- `activate`: transfer execution after a real tab focus change.
- `retain`: pause and resume a logical session across tabs.
- `admit`: resume under constrained state capacity.

Useful switches:

```text
--scheduler=foreground-aware|dual-queue|fcfs-no-slice
--slice=<tokens>
--background-slice=<tokens>
--yield=true|false
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

Aggregate many run directories:

```bash
npm run experiment:browser-service:collect -- \
  --input=artifacts/runs \
  --output=artifacts/runs/aggregate.csv
```
