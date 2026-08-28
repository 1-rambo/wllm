# Retain–Admit–Activate 跨设备运行手册

本文档是配套运行手册，用于把浏览器多标签页实验分发给多位同学，在 Windows 和 macOS
设备上并行执行。目标不是继续做功能调试，而是用统一代码、模型、Wasm
产物和参数，收集能够支撑系统论文主张的真实设备结果。

论文级实验设计、baseline、消融、图表和 claim-evidence 关系见：

```text
examples/browser-service-smoke/EXPERIMENT_PLAN.zh-CN.md
```

> 当前实验协议版本：`wllm.browser-service.experiment.v2`（事件 schema 为 `wllm.browser-service.v2`）
>
> 主实验必须使用干净的 Git 工作区。实验脚本默认拒绝从 dirty worktree
> 记录结果，以避免不同同学实际运行了不同代码。

## 0. Review 时需要先确认的决定

在把任务正式分给同学之前，请负责人确认以下项目。参数一旦冻结，不要让不同
设备自行修改。

| 决定 | 当前建议 | Review 重点 |
| --- | --- | --- |
| 正式模型 | `Llama-3.2-1B-Instruct-Q4_0.gguf` | 模型大小是否适合最低配置设备；是否与论文旧实验保持一致 |
| 正式执行模式 | Playwright Chromium + WebGPU | 是否只做 Chrome-first；Safari/Edge 不混入主表 |
| 设备数量 | 最少 6，建议 8–12 | Windows/macOS 是否各覆盖至少 3 类设备 |
| Core 重复次数 | 每 case 10 次 | 时间预算是否允许；若不允许，至少 5 次并统一 |
| Sensitivity 重复次数 | 每点 5 次 | 只在代表性设备上运行 |
| Activate workload | background 256 tokens；foreground 32 tokens | 是否足以形成稳定的背景长任务 |
| Retain workload | passive pause 2 秒；12 个 pressure sessions；TTL baseline 1 秒 | 是否能在正式 1B 模型上制造可区分的 L1/L2/L3/rebuild |
| Admit workload | `retention=lru`；12 sessions；caps 128/128/128；monotonic budget 2 | 正式 1B/WebGPU pilot 是否能同时触发 repair、budget-bound reject 和 reactive completion |
| 冷却与顺序 | case 间 15 秒；每轮轮换 case 顺序 | 是否适合实验室时间预算 |
| Trace replay | 使用 `suite=trace` | 是否已有可审计的 WebArena task config 和 page-context sidecar |

当前 Admit 参数来自 Linux CPU/Tiny Stories 的压力校准：较宽松的
`256/1024/4096 + budget 16` 无法区分三个策略，而
`128/128/128 + budget 2` 能同时观察 bounded repair、reactive 继续完成和
reject baseline 的取舍。正式 1B/WebGPU 分发前仍须统一做一次 pilot；如果
该模型下不能产生目标压力，应该由负责人统一调整矩阵并升级协议版本，而不是让
同学各自改参数。

## 1. 与论文结构的对应关系

本方案直接对应当前论文
`\sysname: Resumable and Responsive LLM Serving in the Browser` 的修订故事。
论文把一次浏览器 foreground handoff 分成因果连续的三步：

```text
Retain paused state -> Admit resumed state -> Activate foreground request
```

它们不是三个互不相关的优化，而是用户返回一个本地助手标签页时依次可能支付的
三类成本。

| 论文部分 | 论文问题或主张 | 本方案实验 | 主要证据 |
| --- | --- | --- | --- |
| Abstract / Introduction 的 foreground-handoff 总体主张 | 浏览器生命周期需要贯穿状态保留、恢复准入和前台调度 | 三组 core experiment 的联合结果 | Resume TTFT、admission latency/churn、activation latency、background cost |
| Introduction Challenge 1: selective resume retention | 暂停会话多于 live slots 时，不能简单 retain-all 或 evict-all | C1 Retain | lifecycle-aware 对比 LRU、TTL、evict-on-pause、retain-until-full |
| Methodology: Resumable Session Store | 生命周期、近期前台状态和恢复代价是否能指导状态保留 | C1 Retain + pause sensitivity | 恢复来源分布、恢复时间、释放量、成功率 |
| Introduction Challenge 2: disruptive resume admission | 恢复一个状态可能引发 demotion、checkpoint、prune 和 rejection | C2 Admit | monotonic 对比 reactive 和 reject |
| Methodology / Analysis: Monotonic State Admission | repair 是否有进展、有界，并保持容量约束 | C2 Admit + repair-budget sensitivity | repair steps、deficit、victims、moved tokens、reject、success rate |
| Introduction Challenge 3: delayed foreground activation | priority bit 不能中断已经发出的背景执行单元 | C3 Activate | foreground-aware 对比 dual-queue、FCFS/no-slice、no-yield |
| Methodology / Analysis: Foreground-Aware Cooperative Slicing | bounded slice 是否缩短前台 handoff，同时控制后台代价 | C3 Activate + slice sensitivity | activation、foreground TTFT、next-paint、slice tail、background wall time |
| Evaluation: controlled session graphs | 先做端到端，再隔离三个机制的因果效果 | `suite=e2e` + `suite=core` | foreground-handoff 主比较和三组受控 workload |
| Evaluation: interactive co-runs | 同时测 serving 和浏览器交互 | Activate 组 | first-token 与 input-to-next-paint |
| Evaluation: cross-platform robustness | 结论是否只在单一设备成立 | Windows/macOS 多设备矩阵 | per-device median/IQR 与跨设备趋势 |

当前 runner 已覆盖论文 Evaluation 的完整受控执行层：

1. controlled session graph；
2. concurrent browser interaction；
3. cross-device robustness；
4. WebArena-derived trace replay；
5. sensitivity、scaling 和 overhead。

论文 Introduction 还提出了 auditable WebArena-derived traces，用于证明这些
failure 在真实 agent 工作负载中是否频繁发生。`suite=trace` 已实现该 replay；
但正式运行仍必须准备可审计的公开 task config，最好同时提供真实
page-context sidecar。controlled workload 回答因果机制，trace replay 回答
workload relevance，二者共同构成论文证据。

本文中的 `RQ1/RQ2/RQ3` 是为了组织实验而使用的子问题编号。论文当前正文
仍以一个总 Research Question 加三个 challenges 来组织；后续写 Evaluation
时可以保留一个总 RQ，再把三组实验写成 RQ1–RQ3，也可以继续称为
Challenge 1–3，但全文必须保持一致。

### 1.1 当前论文文本需要随后同步的地方

本方案不直接修改 Overleaf 仓库，但实验完成后论文至少需要同步：

1. Methodology 目前仍有“dedicated Worker”和“lifecycle integration remains”
   一类过期描述；实际实现已经是完整 SharedWorker、lifecycle telemetry 和
   foreground dispatch。
2. Evaluation 当前仍标为 legacy results draft，且设备写成单台 Apple M4。
   新结果应按本文第 15 节结构替换，不应与旧 MMLU/HellaSwag 数字混写。
3. Abstract 中三个 `[TBD]` 分别应由 Retain、Admit、Activate 的正式结果填写。
4. Introduction 中“controlled workloads + auditable trace replay + interactive
   co-runs”的承诺，应分别由 core、`suite=trace` 和 Activate UI 指标闭环。

## 2. Research Questions 和判定逻辑

### C1 / RQ1：Retain

对应论文：

- Introduction 的 `selective resume retention`。
- Background 的 `Retain everything` 和 `Evict background sessions`
  两类失败。
- Methodology 的 `Resumable Session Store`。

研究问题：

> 在 paused-session working set 超过 live-state capacity 时，
> lifecycle-aware retention 能否减少用户返回时的冷恢复或重建成本，同时
> 不把所有暂停状态都永久占在 live slots 中？

实验目的：

1. 证明保留策略会真实改变恢复来源，而不只是改了一个策略名称。
2. 量化从 L1、L2、L3 或 rebuild 恢复对 resume latency 的影响。
3. 检查 lifecycle-aware 是否在不同 pause duration 和设备上优于简单基线。
4. 量化保留收益所对应的资源代价。

必要比较：

- `lifecycle-aware`：完整 Retain。
- `lru`：不使用浏览器生命周期语义。
- `ttl`：只使用固定时间窗口。
- `evict-on-pause`：暂停时立即释放。
- `retain-until-full`：只等容量压力发生。

主要指标：

- `sessionResumeToDispatchMs`：标签页恢复到第一次执行。
- `sessionResumeToFirstTokenMs`：标签页恢复到首 token。
- `resumedSessionSource`：`l1`、`l2`、`l3` 或 `rebuild`。
- `resumeRestoreMs`、`resumeRebuildMs`。
- `stateReleaseCount`、`releasedTokens`。
- `requestSuccessRate`。

支持论文主张的结果形态应是：

- lifecycle-aware 更经常从更热的层恢复，或降低 resume-to-first-token；
- 相比 retain-until-full，它不应通过无界 live-state 占用获得收益；
- 相比 evict-on-pause，它应显著减少冷恢复；
- 相比 LRU/TTL，它应在不同 pause duration 下表现更稳健。

如果 LRU 或 TTL 在所有设备和敏感性点上都与 lifecycle-aware 等价，则论文
应收窄 Retain 的 novelty claim，而不是隐瞒该结果。

### C2 / RQ2：Admit

对应论文：

- Introduction 的 `disruptive resume admission`。
- Background 的 `Evict reactively until the request fits`。
- Methodology 和 Analysis 的 `Monotonic State Admission`。

研究问题：

> 在 token 和 live-slot 都受限时，monotonic admission 能否用有限、
> 可观测、持续缩小 deficit 的修复路径完成恢复，并避免无界 churn？

实验目的：

1. 证明准入压力确实触发 demotion、checkpoint、removal 或 fallback。
2. 验证 repair steps 有界，并受 `repair-budget` 控制。
3. 比较成功准入、修复代价和直接拒绝之间的权衡。
4. 为论文的 capacity invariant 和 termination analysis 提供运行时证据。

必要比较：

- `monotonic`：完整 Admit。
- `reactive`：不受 monotonic repair budget 约束的基线。
- `reject`：容量不足直接拒绝。

主要指标：

- `requestSuccessRate`、`requestFailureCount`。
- `admissionCommitCount`、`admissionFallbackCount`、
  `admissionRejectCount`。
- `admissionRepairStepsP95`、`admissionLatencyP95Ms`。
- `admissionInitialTokenDeficitP95`、
  `admissionInitialSlotDeficitP95`。
- `admissionDemotedTokens`、`admissionCheckpointedTokens`、
  `admissionRemovedStates`、`admissionVictimCount`。

支持论文主张的结果形态应是：

- monotonic 的 repair steps 不超过配置的 budget；
- 没有容量越界或无界 repair；
- 相比 reject，monotonic 可以通过有限 repair 完成一部分原本会被拒绝的恢复；
- 相比 reactive，monotonic 用可预测的 tail repair cost 换取必要时的受控拒绝，
  而 reactive 可能完成更多请求，但 repair chain 可以超过同一 budget。

这里的目标不是证明 monotonic 在所有指标上都优于 reactive。核心主张是
**bounded and auditable repair**：系统明确限制一次恢复愿意付出的修复工作，
并把超预算结果表现为可观测的 reject，而不是继续产生无界 churn。

`reject` 组出现请求失败是预期结果。紧 budget 的 monotonic 也可能在用完
repair budget 后产生受控拒绝；这正是 bounded repair 的语义。只要每个失败
都由 `admission_reject` 对应，且日志、events 和 summary 一致，就不是实验
事故。

### C3 / RQ3：Activate

对应论文：

- Introduction 的 `delayed foreground activation`。
- Background 的 `Set a foreground priority bit` 失败。
- Methodology 和 Analysis 的 `Foreground-Aware Cooperative Slicing`。

研究问题：

> 当前景标签页在背景 inference unit 执行期间到达时，
> foreground-aware cooperative slicing 能否缩短 handoff 和 TTFT，
> 同时把后台吞吐损失控制在可接受范围？

实验目的：

1. 区分“队列优先级正确”和“当前执行单元能及时到达 boundary”。
2. 测量 slice size 对 foreground latency 和 background cost 的前沿。
3. 验证主动让出浏览器事件循环对真实 lifecycle 信号传递的作用。
4. 同时测 serving latency 和浏览器可见交互延迟。

必要比较：

- `foreground-aware`：完整 Activate。
- `dual-queue`：有旧队列机制，但没有浏览器前台优先。
- `fcfs-no-slice`：没有 cooperative boundary。
- `foreground-aware + yield=false`：去掉浏览器 task-queue yield。

主要指标：

- `foregroundActivationMs`：前台 lifecycle 信号到第一次 slice dispatch。
- `foregroundFirstTokenMs`：前台 lifecycle 信号到前台请求首 token。
- `foregroundTotalWallMs`：前台请求端到端时间。
- `inputToNextPaintP95Ms`：输入事件到下一次绘制。
- `sliceDurationP95Ms`、`sliceDurationMaxMs`。
- `backgroundTotalWallMs`。

支持论文主张的结果形态应是：

- foreground-aware 明显低于 dual-queue 和 FCFS 的 activation/TTFT；
- 更小 background slice 降低 foreground delay；
- 收益没有伴随不可接受的 background wall-time 膨胀；
- no-yield 的结果说明事件循环 yield 是否必要。

## 3. Claim–Evidence Matrix

| Claim / RQ | Reviewer 会问什么 | 主实验 | 基线/消融 | 核心指标 | 论文落点 |
| --- | --- | --- | --- | --- | --- |
| C1 / RQ1 Retain | 生命周期信息是否真的改善恢复，而不是普通 recency 就够了？ | `core/retain` | LRU、TTL、evict-on-pause、retain-until-full | resume-to-first-token、restore/rebuild、residency source、released tokens | Evaluation: Retain；对应 Introduction Challenge 1 |
| C2 / RQ2 Admit | 压力下是否能完成准入，并且 repair 有界、可解释？ | `core/admit` | reactive、reject | success rate、admission P95、repair steps、fallback/reject、churn | Evaluation: Admit；对应 Challenge 2 和 capacity/termination analysis |
| C3 / RQ3 Activate | foreground priority 是否真的缩短 handoff，还是仍被当前长任务阻塞？ | `core/activate` | dual-queue、FCFS/no-slice、no-yield | activation、foreground TTFT、next-paint、slice tail、background wall | Evaluation: Activate；对应 Challenge 3 和 residual-bound analysis |
| Cross-device robustness | 结果是否只在作者的一台 Mac 上成立？ | Windows/macOS 同协议复现 | 不同 CPU/GPU/内存档位 | per-device median/IQR、趋势一致性、失败率 | Experimental Setup + Cross-device subsection |
| Workload importance | 三类 failure 在真实 agent trace 中是否常见？ | `suite=trace` 的 WebArena-derived replay | trace statistics | pause/resume frequency、working-set pressure、handoff overlap | Evaluation 中独立 workload-characterization subsection |

这张表是审稿视角下的证据闭环。Core experiment 回答机制因果性；设备矩阵
回答可移植性；trace replay 回答现实发生频率。三者不能互相替代。

## 4. 实验包的层次

| 层次 | 用途 | 是否进入论文结果 | 默认重复 |
| --- | --- | --- | ---: |
| Smoke | 验证环境、WebGPU、SharedWorker、遥测和错误路径 | 否 | 1 |
| Core | 回答 RQ1/RQ2/RQ3 的主比较 | 是 | 10 |
| Sensitivity | slice、pause duration、repair budget 敏感性 | 是，通常图或附录 | 5 |
| Trace replay | 证明真实 web-agent workload 中 failure 的频率和重要性 | `suite=trace` | WebArena-derived trace |

Smoke 只验证关键路径，不能当作正式 evidence，也不能与 core 数据混合。

### 4.1 全局固定参数

除各 case 明确覆盖的参数外，矩阵默认使用：

```text
seed=42
n_batch=128
memory_cap_mb=1024（Smoke 为 128）
warmup_runs=1
warmup_predict=4
interaction_samples=20（Smoke 为 1）
interaction_interval_ms=20
core trials=10
sensitivity trials=5
e2e/core/sensitivity/scaling/overhead/trace cooldown=15000 ms
case order=rotate
```

每个 run 都重新启动浏览器与 SharedWorker、重新初始化模型和会话。矩阵层在
不同 run 之间保持固定冷却时间，并按 trial 轮换 case 顺序。

压力会话在目标会话暂停后创建，用于形成容量竞争。压力会话在恢复目标会话
前通过 `experiment_cleanup` 释放；这部分清理量单独记录在
`cleanupReleaseCount` 和 `cleanupReleasedTokens`，不计入 Retain 策略本身的
`stateReleaseCount`。

完整预定义矩阵也保存在：

```text
artifacts/manifests/browser-service-experiment-matrix.csv
```

## 5. 负责人需要统一准备的材料

不要让每位同学自行选择模型或重新编译不同版本的 Wasm。负责人应统一提供：

1. 固定代码提交。
2. 固定 GGUF 模型。
3. 固定 WebGPU Wasm build 目录。
4. 可选的 CPU Wasm build 目录，用于 smoke 和故障定位。
5. 本文档以及设备元数据模板。

建议正式模型固定为仓库现有实验所采用的：

```text
Llama-3.2-1B-Instruct-Q4_0.gguf
```

Tiny Stories 15M 模型只用于 smoke，不作为论文正式性能结果。实验 runner
会自动记录模型大小和 SHA-256；汇总前必须检查所有正式结果的
`model.sha256` 完全相同。

建议分发目录：

```text
experiment-kit/
├── model/
│   └── Llama-3.2-1B-Instruct-Q4_0.gguf
├── wasm-webgpu/
│   ├── wllama.js
│   └── wllama.wasm
├── wasm-cpu/
│   ├── wllama.js
│   └── wllama.wasm
└── device.json
```

正式 WebGPU 结果必须使用同一份 `wasm-webgpu`。脚本会记录
`buildArtifacts.wllamaWasmSha256`，汇总时应检查一致性。

## 6. 设备覆盖建议

最小建议是 6 台机器，推荐 8–12 台：

| 平台 | 最低数量 | 建议覆盖 |
| --- | ---: | --- |
| Windows 11 | 3 | Intel 核显、AMD 核显或 APU、NVIDIA/AMD 独显 |
| macOS Apple Silicon | 3 | 至少两代芯片；8/16/24 GB 或更高内存档位 |
| Intel Mac | 可选 1 | 如果实验室仍有设备，可作为兼容性补充 |

同一物理设备只使用一个稳定的 `deviceId`。推荐命名：

```text
win-i7-irisxe-16gb-01
win-r7-780m-32gb-01
win-i7-rtx4060-32gb-01
mac-m1-8gb-01
mac-m2pro-16gb-01
mac-m3max-36gb-01
```

不要把同一台机器重命名成多台设备，也不要把不同机器复用同一
`deviceId`。

## 7. 设备运行环境

### 7.1 通用软件

- Git。
- Node.js 24；至少保证所有同学使用同一个 Node 主版本。
- npm，使用仓库锁文件执行 `npm ci`。
- Playwright Chromium。
- 真实 WebGPU 实验需要图形桌面会话，不能通过无显示设备的普通
  SSH 会话运行。
- 如果同学只使用负责人分发的 Wasm 产物，不需要安装 CMake、
  Emscripten 或 Docker。

安装依赖：

```bash
git submodule update --init --recursive
npm ci
npx playwright install chromium
```

检查版本：

```bash
node --version
npm --version
npx playwright --version
git rev-parse HEAD
git status --short
```

主实验开始前，`git status --short` 必须没有输出。

负责人应向所有同学给出固定的 `<EXPERIMENT_COMMIT>`。不要在实验当天
直接执行不带提交号确认的 `git pull`。推荐流程：

```bash
git fetch --all
git checkout dev
git reset --hard <EXPERIMENT_COMMIT>
git submodule update --init --recursive
git status --short
```

如果团队不希望同学执行 `reset --hard`，可以由负责人分发同一份代码压缩包。
关键不是采用哪种分发方式，而是所有人的 Git revision、llama.cpp revision、
模型和 Wasm 哈希都一致。

### 7.2 Windows

建议使用 PowerShell，不要在 WSL 中运行正式 WebGPU 数据。WSL 的 GPU、
窗口系统和原生 Windows Chrome 路径不同，不能与 Windows 原生结果混合。

准备工作：

1. 更新 Windows 11。
2. 更新 Intel、AMD 或 NVIDIA 显卡驱动。
3. 接通电源。
4. 关闭节电模式。
5. Windows 电源模式设为“最佳性能”或固定的团队约定模式。
6. 暂停大型下载、游戏、视频会议和系统更新。
7. 不要远程桌面到机器后再采正式数据；远程桌面可能改变 GPU adapter。

PowerShell 示例：

```powershell
git fetch --all
git checkout dev
git reset --hard <EXPERIMENT_COMMIT>
git submodule update --init --recursive
npm ci
npx playwright install chromium

Copy-Item .\examples\browser-service-smoke\device.example.json .\device.json
notepad .\device.json
```

### 7.3 macOS

准备工作：

1. 接通电源。
2. 关闭低电量模式。
3. 固定外接显示器配置；不要一半实验接外屏、一半不接。
4. 关闭大型编译、视频会议、云盘全量同步等后台任务。
5. 实验期间不要合盖。

Terminal 示例：

```bash
git fetch --all
git checkout dev
git reset --hard <EXPERIMENT_COMMIT>
git submodule update --init --recursive
npm ci
npx playwright install chromium

cp examples/browser-service-smoke/device.example.json device.json
open -e device.json
```

### 7.4 device.json

从 `device.example.json` 复制并填写。至少填：

- 设备编号和操作者。
- 操作系统版本。
- CPU。
- 内存容量。
- GPU 型号和驱动版本。
- 台式机或笔记本。
- 是否接通电源、是否节电。
- 外接显示器数量。

runner 会自动记录浏览器版本、User-Agent、逻辑 CPU 数量、系统内存、
WebGPU adapter 信息。手工文件用于补充浏览器无法可靠获取的信息。

`device.json` 中的 `deviceId` 必须与命令行 `--device-id` 完全一致，否则
runner 会拒绝运行。

## 8. 运行前的统一条件

每批正式实验都必须满足：

1. 代码提交相同，并且使用负责人冻结的 `<EXPERIMENT_COMMIT>`。
2. `git status --short` 为空。
3. 模型 SHA-256 相同。
4. Wasm SHA-256 相同。
5. 使用 Playwright Chromium，不混入 Edge、Safari 或系统 Chrome。
6. WebGPU 主实验使用 `--mode=webgpu`。该模式强制
   `actualExecutionMode=webgpu`，如果回退 CPU 会直接失败。
7. WebGPU 使用有窗口模式。运行期间会弹出 Chromium，不要关闭、
   最小化或手工切换实验标签页。
8. 同一设备整个实验期间保持相同电源和显示器条件。
9. 先重启机器，静置约 3 分钟，再开始首批。
10. 主矩阵默认每轮之间冷却 15 秒，并轮换 case 顺序，减少热节流和顺序偏差。
11. 每个 case 在正式运行前自动执行 1 个短 warmup 请求；warmup 后重置会话并
    清空 telemetry，因此不计入结果。
12. 不同策略在每个 trial 中轮换执行顺序，避免某个策略永远吃到冷机或热机。
13. Core Activate 每个 trial 记录 20 次 next-paint 样本；单次 smoke 的
    `inputToNextPaintP95Ms` 只用于验证采集链路。

## 9. Smoke：每台设备先跑

Smoke 只验证环境和机制链路，不进入论文主表。

### 9.1 macOS / Linux

```bash
npm run experiment:browser-service:matrix -- \
  --suite=smoke \
  --group=all \
  --mode=webgpu \
  --trials=1 \
  --device-id=mac-m2pro-16gb-01 \
  --operator=alice \
  --device-file="$PWD/device.json" \
  --model-file="/absolute/path/to/Llama-3.2-1B-Instruct-Q4_0.gguf" \
  --wasm-build="/absolute/path/to/wasm-webgpu" \
  --output="$PWD/artifacts/runs/mac-m2pro-16gb-01/smoke"
```

### 9.2 Windows PowerShell

```powershell
npm run experiment:browser-service:matrix -- `
  --suite=smoke `
  --group=all `
  --mode=webgpu `
  --trials=1 `
  --device-id=win-i7-rtx4060-32gb-01 `
  --operator=alice `
  --device-file="$PWD\device.json" `
  --model-file="D:\wllm-kit\model\Llama-3.2-1B-Instruct-Q4_0.gguf" `
  --wasm-build="D:\wllm-kit\wasm-webgpu" `
  --output="$PWD\artifacts\runs\win-i7-rtx4060-32gb-01\smoke"
```

Smoke 通过条件：

- Matrix 命令退出码为 0。
- 每个 run 目录都有 `summary.json` 和 `events.jsonl`。
- `actualExecutionMode` 为 `webgpu`。
- `gitDirty=false`。
- `deviceMetadata.deviceId` 与命令行设备编号一致。
- `operationErrors` 为空；`admit-reject-tight` 除外，它应出现受控
  `admissionRejectCount > 0`，且错误数与请求失败数一致。
- Activate 两个方法都有非空 activation 和 TTFT。
- lifecycle-aware Retain 恢复成功。
- evict-on-pause 至少出现一次策略释放。
- monotonic Admit 有成功请求；紧容量 smoke 中通常应看到 repair 或
  fallback。

同一个输出目录和 run ID 默认不允许覆盖。需要重跑时使用新的输出目录。
只有明确保留旧目录后，才可以传 `--overwrite=true`。

Smoke 完成后，负责人先检查 `matrix.json` 和每个 `summary.json`，确认
模型/Wasm 哈希、实际执行模式和错误路径正确，再允许该设备进入 core。

## 10. 正式核心实验

正式结果默认每个 case 跑 10 次。若设备特别慢，可先跑 3 次做 pilot，
确认参数后再补足 10 次。论文数据不要把 pilot 和 final 混在一起。

单台设备全部 core 包含：

```text
Activate: 6 cases × 10 trials = 60 runs
Retain:   6 cases × 10 trials = 60 runs
Admit:    4 cases × 10 trials = 40 runs
Total:                             160 runs
```

如果按 group 分工，每位同学只运行自己负责的 40–60 个 core runs。实际耗时
由模型和 GPU 决定，应先执行 `--trials=1` pilot，据此估算，不要在文档里用
Tiny Stories smoke 的耗时推测 1B 模型正式耗时。

为了便于分工，一个同学可以只负责一个 group：

### 10.1 Activate 组

```bash
npm run experiment:browser-service:matrix -- \
  --suite=core \
  --group=activate \
  --mode=webgpu \
  --trials=10 \
  --device-id=<DEVICE_ID> \
  --operator=<NAME> \
  --device-file=<DEVICE_JSON> \
  --model-file=<MODEL_GGUF> \
  --wasm-build=<WEBGPU_WASM_DIR> \
  --output=<OUTPUT_DIR>
```

固定 workload：

```text
n_ctx=8192
n_seq_max=8
n_batch=128
background_predict=256
foreground_predict=32
seed=42
warmup_runs=1
```

比较：

```text
activate-full
activate-dual-queue
activate-size-aware
activate-fcfs-no-slice
activate-no-yield
activate-no-grace
```

### 10.2 Retain 组

```bash
npm run experiment:browser-service:matrix -- \
  --suite=core \
  --group=retain \
  --mode=webgpu \
  --trials=10 \
  --device-id=<DEVICE_ID> \
  --operator=<NAME> \
  --device-file=<DEVICE_JSON> \
  --model-file=<MODEL_GGUF> \
  --wasm-build=<WEBGPU_WASM_DIR> \
  --output=<OUTPUT_DIR>
```

固定 workload：

```text
n_ctx=8192
n_seq_max=8
pause_state=passive
pause_ms=2000
pressure_sessions=12
pressure_predict=16
session_predict=16
target_prompt_repeats=8
pressure_prompt_repeats=8
L1/L2/L3 token caps=512/2048/8192
lifecycle-aware retention_ttl=30000 ms
TTL baseline retention_ttl=1000 ms
```

比较：

```text
retain-lifecycle-aware
retain-hybrid-no-lifecycle
retain-lru
retain-ttl
retain-evict-on-pause
retain-until-full
```

### 10.3 Admit 组

```bash
npm run experiment:browser-service:matrix -- \
  --suite=core \
  --group=admit \
  --mode=webgpu \
  --trials=10 \
  --device-id=<DEVICE_ID> \
  --operator=<NAME> \
  --device-file=<DEVICE_JSON> \
  --model-file=<MODEL_GGUF> \
  --wasm-build=<WEBGPU_WASM_DIR> \
  --output=<OUTPUT_DIR>
```

固定 workload：

```text
retention=lru（隔离 Admit，不施加 lifecycle protection）
n_ctx=8192
n_seq_max=8
sessions=12
session_predict=16
foreground_predict=16
L1/L2/L3 token caps=128/128/128
monotonic repair_budget=2
```

比较：

```text
admit-monotonic
admit-reactive
admit-reactive-lru
admit-reject
```

`admit-reject` 出现受控失败是基线语义，不要删除该 run。`admit-monotonic`
在 budget 用尽后出现受控失败也属于有效结果，不应为了提高 success rate
删除。

`admit-reactive` 使用相同容量、workload 和记录用 budget 值，但其语义是不受
monotonic budget 约束地继续 repair；该字段不作为 reactive 的终止界限。
`admit-reject` 也不执行 repair。因此，正式结果应把三者解释为：

- monotonic：有限修复，必要时受控拒绝；
- reactive：更积极完成，但可能付出更长 repair chain；
- reject：不修复，容量不足立即失败。

## 11. 参数敏感性实验

敏感性实验默认每个点 5 次：

```bash
npm run experiment:browser-service:matrix -- \
  --suite=sensitivity \
  --group=all \
  --mode=webgpu \
  --trials=5 \
  --device-id=<DEVICE_ID> \
  --operator=<NAME> \
  --device-file=<DEVICE_JSON> \
  --model-file=<MODEL_GGUF> \
  --wasm-build=<WEBGPU_WASM_DIR> \
  --output=<OUTPUT_DIR>
```

当前矩阵：

- Activate slice：`4, 8, 16, 32, 64`。
- Retain pause：`0, 1, 5, 30, 60` 秒；TTL 固定 30 秒。
- Admit repair budget：`1, 2, 4, 8, 16, 32`；固定 caps 为
  `128/128/128`，用于画出 success rate 与 repair bound 的取舍曲线。
- Admit equal per-tier capacity：`128, 256, 512 tokens`。
- Retain target prompt repeats：`4, 8, 16`，用于改变目标 prefix 长度。

敏感性实验不要求每台设备都跑。建议选择：

- 一台中档 Windows 笔记本。
- 一台 Windows 独显设备。
- 一台 Apple Silicon 基础款。
- 一台 Apple Silicon Pro/Max。

## 12. CPU 模式的用途

CPU 模式用于：

- CI 或服务器 smoke。
- 排除 WebGPU 驱动问题。
- 验证事件和文件格式。

CPU 模式不与 WebGPU 主实验混在同一个图或表中：

```bash
npm run experiment:browser-service:matrix -- \
  --suite=smoke \
  --group=all \
  --mode=cpu \
  --trials=1 \
  --device-id=<DEVICE_ID>-cpu-check \
  --operator=<NAME> \
  --device-file=<DEVICE_JSON> \
  --model-file=<SMOKE_MODEL> \
  --wasm-build=<CPU_WASM_DIR> \
  --output=<OUTPUT_DIR>
```

## 13. 输出文件

每个矩阵目录包含：

```text
<output>/
├── matrix.json
├── <run-id-1>/
│   ├── events.jsonl
│   └── summary.json
├── <run-id-2>/
│   ├── events.jsonl
│   └── summary.json
└── ...
```

`summary.json` 自动记录：

- Git 和 llama.cpp revision。
- dirty-worktree 标记。
- 模型和 Wasm SHA-256。
- 操作系统、CPU、内存。
- 浏览器版本和 User-Agent。
- WebGPU adapter。
- 实际启用的 CPU/WebGPU 模式。
- 完整命令参数。
- 核心指标和错误。

`events.jsonl` 是原始事件证据，不允许只上传 summary 而丢弃 events。

## 14. 汇总

将多台设备的目录放到同一个总目录后执行：

```bash
npm run experiment:browser-service:collect -- \
  --input=/path/to/all-runs \
  --output=/path/to/all-runs/aggregate.csv \
  --expected-git=<EXPERIMENT_COMMIT> \
  --expected-model-sha256=<MODEL_SHA256> \
  --expected-wasm-sha256=<WASM_SHA256>
```

collector 会展开每个 `summary.json`，并生成统一 CSV。`valid=false`
通常表示：

- dirty worktree。
- 请求模式和实际模式不一致。
- 出现非预期错误。
- Git、模型或 Wasm 与负责人给出的 expected 值不一致。

具体原因会写入 `invalidReasons`。汇总前，负责人必须提供 expected Git、
model SHA-256 和 Wasm SHA-256；不满足者保留原始文件，但不进入论文统计。

汇总时先按以下键分组：

```text
deviceId
actualExecutionMode
case_id / runId 中的 case 部分
```

报告每个设备、每个 case 的：

- median。
- P25 和 P75。
- 最小值和最大值作为诊断信息。
- 成功率。

不要只把所有设备混在一起求一个平均值。设备是系统环境的一部分，应先展示
per-device 结果，再报告跨设备总体趋势。

### 14.1 统计和作图规则

- 单设备单 case：报告 10 次 trial 的 median、P25、P75。
- 跨设备：先计算每台设备的 median，再报告设备 median 的分布。
- 方法比较：在同一设备内按 trial 配对，报告绝对差和相对变化。
- 延迟指标方向为越低越好；success rate 越高越好。
- `resumedSessionSource` 报告各来源比例，而不是把字符串强行转成一个平均值。
- `reject` 基线同时报告 success rate 和 rejection rate。
- 合法极值不删除；环境事故按第 16 节规则排除。
- Smoke、pilot、CPU check、WebGPU final 必须分目录，不能混合。

## 15. 论文结果表和 Evaluation 组织建议

建议把论文最终 Evaluation 组织成以下结构：

1. **Experimental Setup**：设备矩阵、模型、统一 Wasm、浏览器版本、重复次数。
2. **RQ1 / Retain**：恢复来源、resume TTFT 和状态占用代价。
3. **RQ2 / Admit**：成功率、repair chain、churn 和 rejection。
4. **RQ3 / Activate**：activation、foreground TTFT、UI tail 与后台代价。
5. **Sensitivity**：pause duration、repair budget、slice size。
6. **Cross-device robustness**：Windows/macOS 分设备结果和总体趋势。
7. **Auditable trace replay**：报告 WebArena-derived workload 的发生频率和端到端结果。
8. **Overhead and limitations**：控制面开销、浏览器/驱动限制和不支持设备。


### 15.1 Activate 主表

| Device | Method | Activation ms ↓ | Foreground TTFT ms ↓ | Next-paint P95 ms ↓ | Foreground wall ms ↓ | Background wall ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TBD | Foreground-aware | TBD | TBD | TBD | TBD | TBD |
| TBD | Dual queue | TBD | TBD | TBD | TBD | TBD |
| TBD | FCFS/no-slice | TBD | TBD | TBD | TBD | TBD |
| TBD | No-yield | TBD | TBD | TBD | TBD | TBD |

### 15.2 Retain 主表

| Device | Policy | Resume-to-first-token ms ↓ | Restore ms ↓ | Rebuild ms ↓ | L1/L2/L3/Rebuild distribution | Released tokens | Success rate ↑ |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| TBD | Lifecycle-aware | TBD | TBD | TBD | TBD | TBD | TBD |
| TBD | LRU | TBD | TBD | TBD | TBD | TBD | TBD |
| TBD | TTL | TBD | TBD | TBD | TBD | TBD | TBD |
| TBD | Evict-on-pause | TBD | TBD | TBD | TBD | TBD | TBD |
| TBD | Retain-until-full | TBD | TBD | TBD | TBD | TBD | TBD |

### 15.3 Admit 主表

| Device | Policy | Success rate ↑ | Admission latency P95 ms ↓ | Repair steps P95 ↓ | Fallbacks | Rejects | Demoted tokens | Removed states |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TBD | Monotonic | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| TBD | Reactive | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| TBD | Reject | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## 16. 有效性和排除规则

预先固定规则，避免看到结果后再挑数据。

有效 run：

- `gitDirty=false`。
- Git、llama.cpp、模型和 Wasm 哈希符合统一版本。
- `requestedExecutionMode == actualExecutionMode`。
- Activate/Retain：`operationErrors=[]` 且 `requestFailureCount=0`。
- Admit：允许受控 admission failure，但
  `operationErrors.length == requestFailureCount`，且必须伴随
  `admissionRejectCount>0`；其他运行时异常仍无效。
- 输出文件完整。
- `protocolVersion` 与冻结协议一致。

允许重跑但必须保留原始失败目录的情况：

- 同学手工关闭浏览器。
- 机器休眠、断电或系统自动重启。
- 端口、磁盘空间、模型路径等环境错误。
- WebGPU adapter 不可用。
- 出现不能由 `admission_reject` 一一解释的运行时异常。

不能作为排除理由：

- 结果“太慢”。
- 与预期趋势不一致。
- 某次结果是合法运行中的极值。

在最终统计脚本中必须保留 `valid` 和 `invalidReasons` 两列，不能在 Excel
中手工删行后丢失排除原因。

## 17. 运行纪律

- 每次只运行一个矩阵进程。
- 不并行开多个实验 runner。
- 浏览器弹出后不要手动操作实验页面。
- 不要在实验过程中插拔显示器或切换显卡模式。
- 保留冷却时间。
- 每个设备完成一个 group 后再开始下一个 group。
- 记录异常，但不要自行修改参数。

## 18. 常见问题

### 18.1 `WebGPU was required but ... did not activate`

说明实际回退到了 CPU。处理：

1. 确认不是 SSH/远程桌面环境。
2. 更新图形驱动和操作系统。
3. 确认使用 `--mode=webgpu`，浏览器应为有窗口模式。
4. 重新执行 smoke。
5. 仍失败则把该设备标记为“不支持正式 WebGPU 实验”，保留日志。

### 18.2 Playwright 找不到 Chromium

```bash
npx playwright install chromium
```

也可显式设置 `CHROMIUM_EXECUTABLE_PATH`，但正式结果应尽量统一使用
Playwright 下载的 Chromium。

### 18.3 `dirty worktree`

```bash
git status --short
```

提交、还原或 stash 本地修改。`--allow-dirty=true` 只允许用于开发 smoke，
不得用于正式数据。

### 18.4 内存不足或浏览器被系统终止

- 关闭其他大型应用。
- 确认模型是统一的 1B Q4 模型，而不是更大的模型。
- 不要擅自降低 `n_ctx` 或容量参数。
- 把日志交给负责人决定是否调整全体协议。

### 18.5 Admit case 显示失败

`reject` baseline 的失败是预期行为；紧 budget 的 monotonic 也可能发生受控
失败。检查：

```text
admissionRejectCount > 0
requestFailureCount > 0
operationErrors.length == requestFailureCount
```

若是其他异常，或三者不能对应，则不是有效数据。

## 19. 同学提交给负责人的内容

每台设备最终提交：

```text
<device-id>/
├── device.json
├── smoke/
├── core-activate/       # 若负责 Activate
├── core-retain/         # 若负责 Retain
├── core-admit/          # 若负责 Admit
└── sensitivity/         # 若被分配
```

同时提交一段文字：

```text
deviceId:
operator:
Git revision:
model SHA-256:
Wasm SHA-256:
OS:
GPU:
driver:
power mode:
external display:
smoke passed: yes/no
assigned group:
abnormal events:
```

不要只提交截图。真正需要的是完整的 `matrix.json`、`summary.json` 和
`events.jsonl`。

## 20. 当前 Linux CPU smoke 参考

2026-08-28 在 Linux CPU source-built Wasm、Tiny Stories 15M 模型上，
一次 smoke 得到：

| Case | 关键结果 |
| --- | --- |
| Activate / foreground-aware | activation 约 46 ms；foreground TTFT 约 227 ms |
| Activate / dual-queue | activation 约 597 ms；foreground TTFT 约 771 ms |
| Retain / lifecycle-aware | 恢复来源 L1；resume-to-first-token 约 233 ms；无策略释放 |
| Retain / evict-on-pause | 释放 71 tokens；恢复来源 L2；resume-to-first-token 约 432 ms |
| Admit / monotonic tight | 3 次 fallback；repair-step P95 为 2；成功率 100% |
| Admit / reject tight | 2 次 admission reject；请求成功率 60%，属于预期拒绝基线 |

这些数值只说明机制和采集链路工作正常，不是正式论文结果，也不能替代
Windows/macOS WebGPU 数据。

这里的 `inputToNextPaintP95Ms` 每个 smoke run 只有一次交互样本，因此只能验证
采集链路。正式论文中的 UI tail 必须依赖 core 的多 trial，或后续交互压力
实验中的多次输入样本。

### 20.1 Admit 参数校准记录

同一 Linux CPU/Tiny Stories 环境下，最初的 core 参数
`caps=256/1024/4096, budget=16` 几乎不产生容量压力，无法区分 monotonic、
reactive 和 reject。将三层 token cap 统一收紧为 `128/128/128`，并把
monotonic budget 设为 2 后，一次开发校准得到：

| Policy | Success rate | Repair-step P95 | Fallbacks | Rejects | 解释 |
| --- | ---: | ---: | ---: | ---: | --- |
| Monotonic | 92.3% | 2 | 7 | 1 | repair 严格停在 budget，超预算时受控拒绝 |
| Reactive | 100% | 3 | 7 | 0 | 完成更多请求，但 repair chain 超过同一 budget |
| Reject | 75.0% | 0 | 0 | 2 | 不执行 repair，容量不足直接拒绝 |

这组校准说明新矩阵能够测出论文真正要讨论的 tradeoff，因此方案将
`128/128/128 + budget 2` 作为正式 pilot 的起点。它仍然不是论文结果：
运行时工作区为 dirty，模型也只是 15M。正式 1B/WebGPU pilot 必须再次确认
三个策略都处于有区分度、但不会让绝大多数请求失败的压力区间。

## 21. 分工建议

推荐按设备和 group 双维度分工：

| 同学 | 设备 | 任务 |
| --- | --- | --- |
| A | Windows Intel 核显 | smoke + core activate |
| B | Windows AMD APU | smoke + core retain |
| C | Windows NVIDIA 独显 | smoke + core admit |
| D | MacBook Air Apple Silicon | smoke + core activate |
| E | MacBook Pro Apple Silicon Pro | smoke + core retain |
| F | MacBook Pro Apple Silicon Max | smoke + core admit |
| G/H | 代表性设备 | sensitivity |

如果人手允许，让每台设备都跑全部 core 矩阵最好；如果时间有限，至少确保
每个机制在 Windows 和 macOS 上各有两台设备。
