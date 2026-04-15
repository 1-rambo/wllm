# sglang-style-mmlu-hellaswag

A demo page that follows SGLang-like evaluation ideas:

- MMLU: 5-shot style prompt, select A/B/C/D by one-token probability.
- HellaSwag: 20-shot style prompt, select ending by highest continuation probability.

The page runs two modes for comparison:

- Flat: no tree/slot restore (recompute prefix).
- Tree: KV slot save/restore to reuse shared prefix.

The benchmark target can now be split and run independently:

- `MMLU` only
- `HellaSwag` only
- `MMLU + HellaSwag`

Backends:

- `wllama`: existing flat/tree comparison path.
- `web-llm (no cache)`: Exp1-aligned no-cache path for filling cross-framework rows.
	- For this backend, each sample is run as a fresh request and does not reuse prior chat context.
	- Run mode is forced to `exp1` semantics.

Experiment controls:

- Exp2 (MMLU ablation): `full`, `FCFS`, `Random`, `no-tree`
- Exp3: cache maintenance profile (time overhead ratio + snapshot/tier token usage)
- Exp4: concurrent request injection (queue-managed vs direct path, TTFT/tokens-per-second/latency)

## Run

```bash
cd examples/sglang-style-mmlu-hellaswag
npm install
npm run dev
```

## Real Data Sources

- MMLU is loaded from local CSV files extracted from `public/datasets/mmlu/data.tar`.
	- shots: `public/datasets/mmlu/data/val/<subject>_val.csv`
	- eval: `public/datasets/mmlu/data/test/<subject>_test.csv`
- HellaSwag is loaded from local JSONL path (default: `/datasets/hellaswag/hellaswag_val.jsonl`).

Make sure `hellaswag_val.jsonl` exists at `public/datasets/hellaswag/hellaswag_val.jsonl`.
