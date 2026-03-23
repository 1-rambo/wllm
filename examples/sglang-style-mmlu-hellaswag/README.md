# sglang-style-mmlu-hellaswag

A demo page that follows SGLang-like evaluation ideas:

- MMLU: 5-shot style prompt, select A/B/C/D by one-token probability.
- HellaSwag: 20-shot style prompt, select ending by highest continuation probability.

The page runs two modes for comparison:

- Flat: no tree/slot restore (recompute prefix).
- Tree: KV slot save/restore to reuse shared prefix.

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
