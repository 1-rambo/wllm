# wllm - Web LLM Serving System

![](./README_banner.png)

wllm is a Web LLM serving system built on top of [llama.cpp](https://github.com/ggerganov/llama.cpp) + WebAssembly.

It is designed for browser-side serving scenarios where you need:

- chat-oriented request handling
- prefix reuse across conversation branches
- tiered cache with replacement policy
- queue-based serving behavior

This project keeps model execution in a worker and provides a serving-oriented API surface for web applications.

![](./assets/screenshot_0.png)

## Positioning

wllm is not just a runtime wrapper. It is a web serving system with three core pillars:

1. Prefix Tree Cache
2. Tiered Cache + Replacement Strategy
3. Queue Serving System

## Core Capabilities

- TypeScript-first API for web apps
- Worker-isolated inference (non-blocking UI)
- Browser execution with wasm (single-thread / multi-thread auto selection)
- Chat-first tree session APIs:
  - `chatSessionInit`
  - `chatSessionChat`
  - `chatFromNode`
  - `chatGetState`
  - `chatReset`
- Tiered cache state observability (L1/L2/L3 tokens/slots, promotions/demotions, disk read/write counters)
- Queue-based chat request scheduling with reuse-aware scoring
- Model split loading and parallel download support

## API Surface Policy

- User-facing APIs are chat-oriented.
- Internal low-level KV slot actions (`kv_seq_save/restore/rm`) are not exposed as public interfaces.

This keeps integration semantics stable for serving workloads and avoids bypassing chat transaction flow.

## Examples

- Main app example: [examples/main](./examples/main)
- Prefix-tree chat demo: [examples/prefix-tree-chat](./examples/prefix-tree-chat)
- Agentic benchmark (A/B serving comparison): [examples/agentic-benchmark-ab](./examples/agentic-benchmark-ab)
- SGLang-style benchmark (MMLU + HellaSwag): [examples/sglang-style-mmlu-hellaswag](./examples/sglang-style-mmlu-hellaswag)
- UI isolation benchmark (No-Tech vs Full-Tech): [examples/UI-bench](./examples/UI-bench)

## Install

```bash
npm i @wllama/wllama
```

## Quick Start

```ts
import { Wllama } from '@wllama/wllama';

const CONFIG_PATHS = {
  'single-thread/wllama.wasm': '/esm/single-thread/wllama.wasm',
  'multi-thread/wllama.wasm': '/esm/multi-thread/wllama.wasm',
};

const wllama = new Wllama(CONFIG_PATHS, { preferWebGPU: true });

await wllama.loadModelFromUrl('/models/model.gguf', {
  n_ctx: 8192,
  n_batch: 512,
  n_seq_max: 1,
  kv_unified: true,
});

await wllama.chatSessionInit(1024 * 1024 * 1024, {
  enabled: true,
});

const result = await wllama.chatFromNode(0, 'Hello', {
  nPredict: 128,
});

console.log(result.assistantText);
```

For complete integration reference, see [examples/main/src/utils/wllama.context.tsx](./examples/main/src/utils/wllama.context.tsx).

## Model Preparation

- Recommended GGUF chunk size: up to 512MB
- Recommended quantization for web serving: Q4/Q5/Q6
- Very large files may hit ArrayBuffer size limits in browsers; prefer split GGUF files

Split example:

```bash
./llama-gguf-split --split-max-size 512M ./my_model.gguf ./my_model
```

## Build From Source

```bash
git clone --recurse-submodules https://github.com/ngxson/wllama.git
cd wllm
npm i
npm run build:wasm
npm run build
```

## Notes

- Multi-thread wasm requires proper COEP/COOP headers.
- Some benchmark bundles may show chunk-size warnings in Vite build; those are not functional errors.

## Project Status

- Active serving-system evolution on top of llama.cpp
- Tree cache, tiered replacement, and queue serving behaviors are first-class priorities
