# E2E 评测运行手册

本手册覆盖 ArtifexWebGame + Holtwick Tavern 的真实应用 E2E 评测，以及三个本机 ablation。实验设计以
`docs/superpowers/plans/2026-10-15-e2e-artifex-tavern-pressure-plan.md` 为准；本目录中的配置在正式运行前已经冻结，不要在看到结果后修改。

## 1. 范围与分工

| 平台 | 主实验 | 模型 | Ablation |
|---|---|---|---|
| macOS | `e2e-1`、`e2e-2` 全量 | 四个主模型 | `abl-1`、`abl-2`、`abl-3` |
| Windows | `e2e-1`、`e2e-2` 全量 | 四个主模型 | 不运行 |

四个主模型均为 `Q4_K_M`：

- `llama-3.2-1b-instruct`
- `llama-3.2-3b-instruct`
- `qwen2.5-1.5b-instruct`
- `qwen2.5-3b-instruct`

同一个 run 的 Artifex 和 Tavern 必须使用同一个模型。主实验按 OS、模型和实验分别保存；跨平台只比较效应方向，不直接合并绝对延迟。

模型族与尺寸的冻结映射如下：

| 模型族 | 小模型 | 大模型 | 主实验平台 | Ablation 平台 |
|---|---|---|---|---|
| Llama | `llama-3.2-1b-instruct` | `llama-3.2-3b-instruct` | macOS + Windows | 仅小模型，macOS |
| Qwen | `qwen2.5-1.5b-instruct` | `qwen2.5-3b-instruct` | macOS + Windows | 仅小模型，macOS |

- `e2e-1`、`e2e-2`：两个平台都运行四个主模型，每个平台分别完成 4 个模型 × 324 条记录。
- `abl-1`、`abl-2`、`abl-3`：只在 macOS 上运行两个小模型，每个 ablation 为 2 个模型 × 324 条记录。
- Windows 不运行 ablation；跨平台不使用同一模型在同一 run 中混跑。
- WebLLM 不进入当前主实验或 ablation；主对照是 `ours` 与 `wllama-per-tab`。

每个模型的正式记录数是固定的：

| 实验 | 每个模型 | 完整模型集 |
|---|---:|---:|
| `e2e-1` | 324 task records | 1296 |
| `e2e-2` | 324 recovery/background records | 1296 |
| `abl-1` / `abl-2` / `abl-3` | 324 records | 648（两个小模型） |

计算口径：`2 arms × repeats × pressure levels × app tabs`。judge 分数附加在已有 task 记录上，不另算 task record。

## 2. 前置条件

- Node.js 20 或更高版本。
- Google Chrome 或 Chromium，且能够运行 WebGPU。
- Git 和可用的 C/C++ 构建环境（仅在本地 vendor 依赖尚未准备时需要）。
- 至少 16 GiB 内存；10-tab、3B 模型和 wllama-per-tab 同时运行时建议 24 GiB 以上。
- 至少 20 GiB 可用磁盘；四个 GGUF、应用构建产物、浏览器缓存和 artifacts 都会占用空间。
- macOS 与 Windows 使用同主版本 Chrome。固定电源策略、关闭自动休眠，并避免运行其他 GPU 或浏览器负载。
- 实验过程使用真实可见 tab。Chrome 不能被最小化；pressure tab 在某些档位会短暂退到后台，但 trial 结束时必须重新处于 `visible`。

Windows 上如 Chrome 不在标准位置，设置 `CHROME_PATH`：

```powershell
$env:CHROME_PATH = "C:\Program Files\Google\Chrome\Application\chrome.exe"
```

macOS 上如使用非默认 Chrome：

```bash
export CHROME_PATH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
```

## 3. 首次准备

以下命令都从 `impl/` 目录执行。

### macOS / Linux shell

```bash
cd impl
npm ci
npm run evaluation:models
npm run evaluation:prepare
npm run evaluation:check
npm run evaluation:verify
```

### Windows PowerShell

```powershell
cd impl
npm ci
npm run evaluation:models
npm run evaluation:prepare
npm run evaluation:check
npm run evaluation:verify
```

以上四步也可合并为 `npm run evaluation:setup`。

如果本机重建了 `vendor/wllama`，用 `npm run sync:wllama-prebuilt` 刷新预编译包；该命令是 Node 脚本，macOS 和 Windows 都能直接运行。

首次模型下载约 6 GiB，脚本会校验字节数和 SHA-256。若只需要一个模型：

```bash
node evaluation/scripts/prepare-models.mjs --model llama-3.2-1b-instruct
```

列出模型 ID：

```bash
node evaluation/scripts/prepare-models.mjs --list
```

`evaluation:check` 会检查 Chrome、构建产物、Artifex、Tavern 和模型文件。模型缺失只是 warning；正式运行前必须把对应模型补齐。当前矩阵中尤其要确认
`qwen2.5-3b-instruct` 已经下载。

`evaluation:prepare` 会读取 `vendor/apps-pins.json`：目录不存在时按固定 repository/commit 获取 Artifex 和 Tavern；目录已存在时会校验当前 commit。commit 不匹配会直接失败，不会悄悄使用另一个版本。

## 4. 运行前门禁

先运行 smoke。smoke 使用 2 个 app tab（1 Artifex + 1 Tavern）和 1 个 pressure tab，只验证适配器、页面加载、priority API 和基本执行链：

```bash
npm run evaluation:e2e-1:smoke
npm run evaluation:e2e-1:pilot
```

`e2e-2` 也必须先 smoke 和 pilot：

```bash
npm run evaluation:e2e-2:smoke
npm run evaluation:e2e-2:pilot
```

Ablation 在正式运行前可用统一 runner 做 smoke；`--no-judge` 已在当前 package 命令中固定：

```bash
node evaluation/scripts/run.mjs --experiment abl-1 --mode smoke --no-judge
node evaluation/scripts/run.mjs --experiment abl-2 --mode smoke --no-judge
node evaluation/scripts/run.mjs --experiment abl-3 --mode smoke --no-judge
```


`pilot` 用于校准 tab 规模、压力档位、timeout 和切换时序。pilot 不应被当作正式结果；正式运行前确认 `evaluation:check` 和 `evaluation:verify` 仍然通过。

正式运行前还应运行：

```bash
npm run typecheck
npm run evaluation:verify
npm test
npm run build
npm run model:check
npm run verify:runtime
npm run smoke:browser
```

## 5. 正式运行

同一时间只运行一个正式实验。不要并行跑 `e2e-1`、`e2e-2` 或 ablation。

### E2E-1：优先级调度 + 渲染压力

```bash
npm run evaluation:e2e-1
```

### E2E-2：状态恢复 + 渲染压力

```bash
npm run evaluation:e2e-2
```

两个正式实验都会按当前平台选择主模型集合。正式运行结果分别写入：

```text
impl/artifacts/evaluation/e2e-priority-pressure/<os>/<model-id>/<timestamp>/
impl/artifacts/evaluation/e2e-recovery-pressure/<os>/<model-id>/<timestamp>/
```

### Ablation（仅 macOS）

```bash
npm run evaluation:abl-1
npm run evaluation:abl-2
npm run evaluation:abl-3
```

Ablation 默认只用两个小模型：

```text
llama-3.2-1b-instruct
qwen2.5-1.5b-instruct
```

Windows 上运行 ablation 会被配置拒绝，这是预期行为。

### 只重跑一个模型


推荐直接调用跨平台 runner，避免 Windows 上 npm 参数分隔符差异：

```bash
node evaluation/scripts/run.mjs --experiment e2e-1 --mode formal --model qwen2.5-1.5b-instruct
node evaluation/scripts/run.mjs --experiment e2e-2 --mode formal --model llama-3.2-3b-instruct
node evaluation/scripts/run.mjs --experiment abl-1 --mode formal --model llama-3.2-1b-instruct
```

`--out` 只能和 `--model` 一起使用；多模型 formal run 会按 `<platform>/<model>/<timestamp>` 自动分开目录。

单个模型的正式 run 仍然独立写入它自己的 timestamp 目录；不要覆盖或合并以前的结果。

## 6. 每个 run 的预期数据

E2E-1 每个 trial 有 9 个 app session；E2E-2 每个 trial 有 1 个 recovery target 和 8 个 background session。正式 schedule、重复数和 tab 数由冻结配置控制。

每个 run 目录至少包含：

```text
records.jsonl
events.jsonl
summary.json
gate-report.json
run.json
latest.json
```

其中：

- `records.jsonl`：任务、恢复、judge 记录。
- `events.jsonl`：priority、pressure、生命周期和 trial 级事件。
- `summary.json`：按 arm/pressure/app/role 聚合，以及配对 bootstrap 对比。
- `gate-report.json`：冻结 gate 的 id、status、效应量和分母/分子；只看这一份文件判断 pass/fail。
- `run.json`：平台、模型、config hash、preregistration 和 artifact 路径。

## 7. 查看与验证结果

查看最新 run：

```bash
npm run evaluation:report
```

指定 run 目录或 `latest.json`：

```bash
node evaluation/scripts/report.mjs impl/artifacts/evaluation/e2e-priority-pressure/darwin/llama-3.2-1b-instruct/2026-10-15T00-00-00.000Z
node evaluation/scripts/report.mjs impl/artifacts/evaluation/e2e-priority-pressure/latest.json
```

如果最新指针包含多个模型，按字段筛选：

```bash
node evaluation/scripts/report.mjs --experiment e2e-1 --platform win32 --model llama-3.2-1b-instruct
```

机器可读输出：

```bash
node evaluation/scripts/report.mjs --json
```

报表会把任务质量、serving/priority、frame 性能和配对差值分开打印，避免把 judge 分数与响应时间混成一列。

## 8. 判定与失败处理

只有在下面条件同时成立时，才把 run 当成正式实验结果：

1. `summary.json` 的 completeness 达到预期，且 `errorCount === 0`。
2. 所有 task/recovery 记录都有 terminal 状态。
3. pressure tab 在测量窗口内为 `visible`。
4. E2E-2 的每条 recovery 记录都有 hidden、frozen、reload、active 证据。
5. `gate-report.json` 没有 `fail` 或 `incomplete`。
6. 没有 OOM、wasm memory、`Invalid magic number`、memory access 或 device lost。
7. judge 有效比例达到配置中的 `minimumValidFraction`。

以下情况必须保留原 run，记录为 incomplete 或 invalid，不能删除、补记录、挑窗口或降低 tab 数：

- Chrome 崩溃、WASM memory error、`Invalid magic number`、device lost。
- 10-tab 或 3B 配置在 Windows 上资源不足。
- pressure tab 不可见或生命周期证据缺失。
- judge 输出缺失/无效比例过高。
- 运行在过程中被中止，或 `summary.json` 不完整。

只有 Vite reload、端口冲突等明确的基础设施瞬态错误，才允许在保留失败目录后使用新的 timestamp 重启。不能把两个 run 拼接为一份结果。

## 9. 结果归档

每个 OS、模型和实验组独立归档。至少保留：

- `run.json` 中的 platform、模型 hash、config hash、Artifex/Tavern commit。
- `records.jsonl` 和 `events.jsonl` 原始数据。
- `summary.json`、`gate-report.json` 和失败 run 的 `failure.json`。
- Chrome、GPU driver、电源模式和异常情况的简短记录。

跨平台比较只写“效应方向是否一致”，不写跨 OS 的绝对 TTFT、wall time 或 FPS 差值。模型族比较只在同一 OS、同一实验组内做。

## 10. 代码约束

- evaluation harness 不得 import `impl/src`。
- 不修改 `sections/*.tex`、`references.bib`、`deprecated/` 或 archived WebArena artifacts。
- 不在正式实验运行中调整策略参数、模型、prompt、压力 schedule 或统计口径。
- 旧 E1-E5 结果只作为历史 artifact 保留，不并入本次 E2E 结果。

## 11. Windows 新机器完整步骤

以下步骤在原生 Windows 的 PowerShell 中执行，不使用 WSL：

1. 安装 Node.js 20 LTS、Git 和 Google Chrome，并确认 `node --version`、`npm --version`、`git --version` 可用。
2. 克隆或复制完整仓库，保持 `impl/vendor/`、`impl/vendor/wllama-prebuilt/`、`impl/package-lock.json` 和 `impl/evaluation/config/` 的相对位置不变。
3. 打开 PowerShell，进入 `impl` 目录。
4. 如果 Chrome 不在标准路径，设置 `CHROME_PATH`。
5. 执行：

```powershell
npm ci
npm run evaluation:setup
npm run evaluation:check
npm run evaluation:verify
```

`evaluation:setup` 会下载模型、按 pin 准备 Artifex/Tavern、构建 engine pages 并检查环境。首次运行下载量约 6 GiB；如果只想先验证一个小模型：

```powershell
node evaluation/scripts/prepare-models.mjs --model llama-3.2-1b-instruct
node evaluation/scripts/run.mjs --experiment e2e-1 --mode smoke --no-judge
node evaluation/scripts/run.mjs --experiment e2e-1 --mode pilot --model llama-3.2-1b-instruct --no-judge
node evaluation/scripts/run.mjs --experiment e2e-1 --mode formal --model llama-3.2-1b-instruct
```

6. 查看结果：

```powershell
node evaluation/scripts/report.mjs --experiment e2e-1 --model llama-3.2-1b-instruct
```

7. 结果位于：

```text
impl\artifacts\evaluation\e2e-priority-pressure\win32\<model-id>\<timestamp>\
impl\artifacts\evaluation\e2e-recovery-pressure\win32\<model-id>\<timestamp>\
```

Windows 只运行 `e2e-1` 和 `e2e-2`。如果 10-tab、3B 或 `wllama-per-tab` 触发资源阻塞，保留原始 run 并在 `failure.json`/报告中标记 `platform resource block`；不要减少 tab 数或更换模型后继续同一正式 run。
