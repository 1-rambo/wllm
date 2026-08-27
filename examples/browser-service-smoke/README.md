# Browser Service Experiments

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
```

Each run writes:

```text
<output>/<run-id>/events.jsonl
<output>/<run-id>/summary.json
```

The JSONL stream uses schema `wllm.browser-service.v1`.
