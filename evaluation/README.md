# E2E 评测运行手册

本手册覆盖 ArtifexWebGame + Holtwick Tavern 的快速 E2E 评测。当前协议面向快速迭代：每个 E2E 只做一个简单 smoke，正式实验只跑一次，不设置 pilot、acceptance gate 或自动判定 pass/fail。

## 1. 当前矩阵

主比较固定为 `ours` 对 `wllama-per-tab`。WebLLM 不参与。

| 平台 | 实验 | 模型 | 正式规模 |
|---|---|---|---|
| macOS / Windows | `e2e-1` | 4 个主模型 | 432 task records |
| macOS / Windows | `e2e-2` | 4 个主模型 | 432 recovery/background records |

四个主模型：

- `llama-3.2-1b-instruct`
- `llama-3.2-3b-instruct`
- `qwen2.5-1.5b-instruct`
- `qwen2.5-3b-instruct`

正式 trial 使用 6 个应用会话（3 个 Artifex、3 个 Tavern）和 1 个可见 raw WebGPU pressure tab。压力档位为 `0, 250, 500, 1000, 1500, 2000, 3000, 4000, clear-return`。

记录数口径：

```text
单个模型/实验：2 arms × 1 repeat × 9 pressures × 6 app tabs = 108
单个实验四模型：4 × 108 = 432
```

smoke 每个实验只跑一次，固定使用第一个主模型（Llama-1B），使用 1 Artifex + 1 Tavern + 1 pressure tab：

```text
2 arms × 1 repeat × 1 pressure × 2 app tabs = 4
```

## 2. 前置准备

从 `impl/` 目录执行：

```bash
npm ci
npm run evaluation:models
npm run evaluation:prepare
npm run evaluation:check
npm run evaluation:verify
```

要求：

- Node.js 20 或更高版本。
- Git、Google Chrome 和可用的硬件加速 WebGPU。
- 至少 16 GiB 内存；7-tab、3B 模型和 `wllama-per-tab` 同时运行时建议 24 GiB 以上。
- 至少 20 GiB 可用磁盘。
- 保持电源连接、关闭自动休眠；实验时不要让 Chrome 最小化。

Chrome 不在默认位置时设置 `CHROME_PATH`。完整模型准备与 manifest 校验仍由 `evaluation:models`、`evaluation:check` 负责。

## 3. 快速运行

先各跑一个 smoke：

```bash
npm run evaluation:e2e-1:smoke
npm run evaluation:e2e-2:smoke
```

再运行正式实验：

```bash
npm run evaluation:e2e-1
npm run evaluation:e2e-2
```

正式命令默认运行四个主模型。只重跑一个模型时：

```bash
node evaluation/scripts/run.mjs --experiment e2e-1 --mode formal --model qwen2.5-1.5b-instruct --no-judge
node evaluation/scripts/run.mjs --experiment e2e-2 --mode formal --model llama-3.2-3b-instruct --no-judge
```

`--out` 只能与单个 `--model` 一起使用。不要覆盖、拼接或挑选已有 run。

配置中的 `modes.formal.repeats` 固定为 `1`；不要为了当前结果临时增加重复次数。

## 4. 输出

结果分别写入：

```text
impl/artifacts/evaluation/e2e-priority-pressure/<os>/<model-id>/<timestamp>/
impl/artifacts/evaluation/e2e-recovery-pressure/<os>/<model-id>/<timestamp>/
```

每个 run 目录包含：

```text
records.jsonl
events.jsonl
summary.json
run.json
latest.json
```

发生 harness 级失败时还会有：

```text
failure.json
```

当前 E2E runner 不生成 `gate-report.json`。`summary.json` 的 completeness、`errorCount`、实际记录数和原始 `records.jsonl` 是报告依据。smoke 的 4 条记录或正式 run 的 108 条记录如果不完整，必须直接报告为 incomplete；出现执行错误时保留原始错误记录，不删除失败任务，也不用 gate 把结果改写成 pass。

E2E-1 另外记录探索性的 click-response：按固定 seed 激活仍在运行的应用 tab，测到第二个 `requestAnimationFrame` callback，再恢复 pressure tab。该操作不调用 priority API，也不改变任务分母。它没有 gate；如果覆盖不足或结果不稳定，可以不进入论文主表。

## 5. 报告

查看最新结果：

```bash
npm run evaluation:report
```

指定 run：

```bash
node evaluation/scripts/report.mjs artifacts/evaluation/e2e-priority-pressure/darwin/llama-3.2-1b-instruct/2026-10-23T00-00-00.000Z
```

筛选：

```bash
node evaluation/scripts/report.mjs --experiment e2e-1 --platform win32 --model llama-3.2-1b-instruct
```

机器可读输出：

```bash
node evaluation/scripts/report.mjs --json
```

报告把 task quality、serving/priority、recovery、click-response、frame performance 和 paired contrasts 分开输出。

## 6. 失败处理

- Smoke 失败时先检查环境、模型、应用构建和页面适配器，再重新运行 smoke。
- 正式 run 失败时保留整个 run 目录和日志；修复明确的基础设施问题后使用新的 timestamp 重跑该组合。
- Windows 一键 runner 在某个正式模型组合失败时会记录错误并继续下一个组合，不会因为 `wllama-per-tab` 的错误或缺少 gate 而停止整批实验。
- 不能删除失败记录、缩小 7-tab 规模、降低压力档位或把 partial run 拼成完整结果。
- 只有 Vite reload、端口冲突等明确瞬态错误可以在保留旧目录后重启；崩溃、WASM memory error、`Invalid magic number`、device lost 都应作为真实结果报告。

## 7. 静态验证

提交或交付前执行：

```bash
npm run typecheck
npm run evaluation:verify
npm test
npm run build
```

`evaluation:verify` 检查 harness 不导入 `impl/src`，并确认当前配置没有 pilot、正式重复数为 1、smoke schedule 只有一项且 preregistration 不含 acceptance gates。

## 8. Windows 一键运行

外部 Windows 机器使用仓库根目录的一站式入口：

```text
Run-Windows-E2E.cmd
```

它会完成环境检查、依赖与模型准备、两个应用构建、静态验证、2 个 smoke、8 个正式 run（2 个实验 × 4 个模型）以及结果打包。具体回传方式见 `Windows-E2E-Handoff.md`。

## 9. 约束

- Evaluation harness 不得 import `impl/src`。
- 不修改 `sections/*.tex`、`references.bib`、`deprecated/` 或 archived WebArena artifacts。
- 正式实验期间不改变模型、prompt、tab 数、pressure schedule 或统计口径。
- 旧实验结果只作为历史 artifact 保留，不并入当前快速 E2E 结果。
