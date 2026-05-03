# WebArena Latency Bench

A lightweight exp1-style end-to-end latency benchmark on WebArena Verified retrieve tasks.

## Dataset

This example uses a filtered subset of WebArena Verified RETRIEVE tasks exported from the official `webarena-verified` package. The current subset keeps single-site information-seeking tasks from:

- `shopping_admin`
- `shopping`
- `gitlab`
- `reddit`

The exported task file lives at `public/datasets/webarena/retrieve_info_subset.json`.

## Real page-context extraction

To benchmark with real preloaded page content instead of site-level placeholder text, generate
`public/datasets/webarena/task_page_contexts.json` from a valid WebArena environment config:

```bash
/Users/rambo/Desktop/wllama-webgpu/.venv/bin/python \
  scripts/extract_task_page_contexts.py \
  --dataset public/datasets/webarena/retrieve_info_subset.json \
  --config /path/to/webarena-config.json \
  --output public/datasets/webarena/task_page_contexts.json
```

The app will automatically load `task_page_contexts.json` if present and merge the extracted
page text into the shared prefix. Tree reuse is then keyed by the rendered start-page context
instead of only by site name.

## Run

```bash
npm install
npm run dev
```
