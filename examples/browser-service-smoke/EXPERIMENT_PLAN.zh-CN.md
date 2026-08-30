# wLLM 论文完整实验方案

本文档定义能够直接支撑论文 Evaluation 章节的完整实验包。它不是 smoke
说明书。Smoke 只负责排除环境和实现错误，不进入论文结果。Windows/macOS
安装、命令和数据回收细节见同目录下的：

```text
DEVICE_RUNBOOK.zh-CN.md
```

可直接转发给实验执行同学的任务说明见：

```text
EXPERIMENT_ASSIGNMENT.zh-CN.md
```

本文的北极星是回答论文总问题：

> 浏览器内的共享 LLM 服务，能否在有限本地容量下保留有价值的暂停会话，
> 以有限且可审计的代价恢复该会话，并在用户返回标签页时及时把执行权交给
> 前台，同时避免不可接受的后台吞吐、内存、磁盘和 UI 代价？

完整证据链为：

```text
构造的异构 Web workload 确实覆盖目标压力条件
        ↓
端到端 foreground handoff 得到改善
        ↓
Retain / Admit / Activate 分别解释改善来自哪里
        ↓
消融证明三个跨层 invariant 都有独立作用
        ↓
敏感性、扩展性和成本说明适用边界
        ↓
跨 Windows/macOS 设备证明结论不是单机偶然
```

---

## 1. 先冻结 Methodology：论文到底实现了什么

实验必须验证真实实现，而不是验证一个比代码更理想化的描述。当前实现应按
下面四层写进 Methodology。

### 1.1 Browser-resident shared service

同源标签页通过 `BrowserLLMClient` 连接一个版本化 `SharedWorker`。该
SharedWorker 只加载一份模型，并维护：

- 所有客户端的 lifecycle 状态；
- 一棵跨请求共享的 conversation tree；
- 一个统一请求队列；
- 一组有限的 live sequence slots；
- L1/L2/L3 状态及其恢复路径；
- 版本化事件流。

这一层解决“每个标签页各自持有模型、彼此不知道谁在前台”的架构问题。它是
三个机制能够联合工作的前提，但不是第四个复杂算法。

### 1.2 Retain：lifecycle protection + tiered replacement

代码中的 Retain 不是一个已经训练好的
`p(resume) × avoided-cost` 预测器。真实实现是两阶段策略：

1. TypeScript 控制面根据 active/passive/hidden、最近 foreground 时间以及
   TTL，生成需要保护的 session-head 集合；
2. native tier manager 在未被保护的候选中按照 replacement policy 选择
   victim。完整系统使用 age、state size 和 access frequency 的 hybrid
   score。

状态具有四种恢复结果：

- L1：live KV state，可直接继续；
- L2：内存 token checkpoint，需要 replay；
- L3：磁盘 checkpoint，需要读回再 replay；
- rebuild：没有可用 checkpoint，从逻辑对话重建。

因此论文中的价值函数应作为**设计动机**，不能声称当前系统显式学习了
resume probability。正文应准确写成“lifecycle-defined protection set +
cost-aware replacement among unprotected states”。

### 1.3 Admit：budgeted monotonic repair

恢复或 checkpoint 一个状态前，系统检查目标 tier 的 token 容量和 live slot
容量。容量不足时逐步 demote、checkpoint 或 remove victim。每次 victim
selection 都计为一个 repair step；`victimIds` 另行记录本次 admission 涉及的
不同节点。每个成功步骤必须减少当前层占用或释放 live slot。

完整策略设置 repair budget。达到 budget 仍无法满足容量时，操作被拒绝。
该过程：

- 保持每层 `usage <= capacity`；
- 保证 repair decision 数有界；
- 记录 initial deficit、repair steps、victims、demoted/checkpointed tokens、
  removed states 和 latency；
- 允许一次最终失败的 admission 已经产生部分有效 demotion。

因此 Methodology 不应写成“先规划完整事务再原子提交”。当前实现是
**online monotonic repair with bounded work and observable partial displacement**。

### 1.4 Activate：event-observable cooperative boundaries

请求的 prefill 和 decode 被拆成 cooperative units。每个 unit 结束后重新调度：

- foreground 请求优先；
- background 请求在 foreground 等待时使用更小的 slice；
- lifecycle 由 hidden 转为 active 时设置一个很短的 handoff grace，
  让刚到达的 foreground request 有机会在后台任务被重新 dispatch 前入队；
- 每个 yield 后显式让出浏览器 task queue；
- overdue queue 保留等待年龄，避免后台任务永久饥饿；
- service-cost estimate 用于同一优先级内的排序。

关键不是“设置优先级”，而是缩短已经运行的 background unit 的 residual
blocking time，并确保 lifecycle 消息能够进入 SharedWorker event loop。

### 1.5 Methodology 与代码的审计结论

| 论文机制 | 当前实现 | 可做论文主张 | 不应声称 |
| --- | --- | --- | --- |
| Shared browser service | SharedWorker 单实例、统一模型和队列 | 多标签页共享一个模型服务；后续 client attach 无需重复加载模型 | 跨浏览器或跨 origin 共享 |
| Retain | lifecycle protection、TTL、hybrid/LRU/LFU/size/random replacement | lifecycle 信息改变被保护状态，并影响恢复层级和恢复延迟 | 已学习精确 resume probability |
| Admit | token/slot capacity check、monotonic repair、budget、telemetry | repair 有界、容量不越界、成功率与 displacement 可观测 | admission 原子回滚；任何压力下都不拒绝 |
| Activate | foreground priority、prefill/decode slicing、background slice、grace、event-loop yield | 前台 activation 受 slice boundary 控制，可降低 handoff tail | GPU kernel 抢占或浏览器内核修改 |
| Cost model/fairness | 在线时间估计、overdue promotion | 可报告预测误差、queue wait 和 overdue 行为 | 严格实时 deadline guarantee |

---

## 2. Evaluation 的研究问题

### RQ0：端到端效果

完整 Protection–Progress–Responsiveness contract chain 能否改善一次真实的
tab-return handoff，而不
只是分别优化三个内部指标？

主指标：

- foreground resume-to-first-token；
- foreground request P95 latency；
- input-to-next-paint P95；
- workload makespan；
- aggregate tokens/s；
- background P95 latency；
- request success rate。

### RQ1：Lifecycle-to-state binding

在 session head、physical tier 和 native slot 持续变化时，最新 lifecycle intent
能否稳定绑定到正确的目标状态，并在后续 admission 中保持 target-binding miss
和 protected-victim violation 为 0？在此 correctness 条件成立后，它是否使返回
会话更常留在热层并降低 resume TTFT？

### RQ2：Admit

容量不足时，monotonic admission 能否把 repair work 限制在明确 budget 内，
保持容量约束，并在成功率和 state displacement 之间形成可解释的取舍？

### RQ3：Event-observable execution boundary

foreground transition 到达时，bounded native unit 与 macrotask yield 能否分别
降低 native residual blocking 和 client-to-SharedWorker signal delay，并使
foreground request 更快得到 dispatch，同时控制 checkpoint、后台延迟和吞吐
代价？

### RQ4：异构 Web workload relevance

在可审计的 WebArena-derived 浏览器任务 replay 中，是否覆盖 pause/resume、
状态压力和 foreground/background overlap，并且完整系统的收益能否跨不同站点
上下文保持？该实验不估计自然用户行为中的发生频率。

### RQ5：开销与扩展性

机制本身的 scheduler、cache hint、admission 和 telemetry 开销是否足够低；
随着标签页数、队列深度、上下文长度和内存压力增长，系统是否稳定？

---

## 3. Baseline 设计

### 3.1 论文主表中的同栈 baseline

主结论优先使用同一个 GGUF、同一个 Wasm build、同一个浏览器和同一套
SharedWorker substrate，只替换待比较策略。这样可以避免不同模型格式和
runtime implementation 混淆机制效果。

| 名称 | Retain | Admit | Activate | 回答的问题 |
| --- | --- | --- | --- | --- |
| **wLLM-Full** | lifecycle-aware + hybrid | monotonic bounded repair | foreground-aware slicing | 完整系统 |
| **Shared-Default** | LRU | reactive repair | dual-queue fixed slicing | 一个合理但不使用本文三项设计的共享服务 |
| **Cold-FCFS** | evict-on-pause | reactive | FCFS/no-slice | 用户返回时冷恢复且没有前台 handoff |
| **w/o Retain** | LRU/hybrid without lifecycle protection | monotonic | foreground-aware | Retain 的端到端贡献 |
| **w/o Bounded Admit** | lifecycle-aware | reactive | foreground-aware | Admit 的端到端贡献 |
| **w/o Foreground Activate** | lifecycle-aware | monotonic | dual-queue | Activate 的端到端贡献 |

`Shared-Default` 是主要系统 baseline。三个 `w/o` 版本是严格 component
ablation。`Cold-FCFS` 给出最简单实现的下界。

### 3.2 外部系统 baseline

外部 baseline 只回答“与现有浏览器推理引擎相比，基础推理效率是否合理”，
不用于证明 Retain、Admit 或 Activate，因为现有引擎没有相同的多标签页状态
语义。

| Baseline | 用法 | 公平性约束 | 报告位置 |
| --- | --- | --- | --- |
| wllama no-reuse path | 同一 fork runtime，每个请求前 reset，不跨请求复用 prefix | 同 GGUF、量化、浏览器、prompt、输出长度 | Appendix 或基础性能表 |
| WebLLM | 主流 WebGPU browser LLM engine | 同模型家族和规模、相同 prompt/output；明确模型格式差异 | Appendix 或基础性能表 |

不得把 WebLLM 的单请求吞吐差异写成本文 foreground-handoff 机制的收益。
如果无法获得足够接近的模型和量化，保留该结果为补充，不作为 headline。

外部 baseline 的自动化入口为：

```bash
npm run experiment:webarena:prepare -- \
  --input=/path/to/webarena/config_files/test.raw.json

npm run experiment:webarena:run -- \
  --backend=wllama \
  --model-file=<GGUF_FILE> \
  --wasm-build=<SOURCE_BUILT_WASM_DIR> \
  --eval-count=120 \
  --output=<OUT>/wllama.json

npm run experiment:webarena:run -- \
  --backend=web-llm \
  --webllm-model=Llama-3.2-1B-Instruct-q4f32_1-MLC \
  --eval-count=120 \
  --output=<OUT>/webllm.json

npm run experiment:webarena:score -- \
  --input=<OUT>/wllama.json \
  --output=<OUT>/wllama-accuracy.json
```

本地 scorer 对 exact/must-include 提供确定性检查，对 fuzzy-match 仅提供
normalized-contains proxy；最终 accuracy 必须使用 WebArena 官方 evaluator
复核。这样可以检查系统优化没有通过输出为空或质量退化换取速度。

### 3.3 各机制的强 baseline

**Retain**

- lifecycle-aware + hybrid；
- hybrid without lifecycle protection；
- strict LRU；
- TTL；
- evict-on-pause；
- retain-until-full。

TTL 不能只选择一个明显不利的常数。先在 anchor 设备的独立 calibration run
中执行 `1/5/30/60 s` sweep，按预先确定的 resume TTFT、state footprint 和
rejection 规则选择一个值，再通过 `--tuned-retention-ttl-ms` 冻结到所有正式
core 和 replication run。主表报告该 tuned fixed-TTL，附录报告完整 sweep。

**Admit**

- monotonic + hybrid；
- reactive + hybrid；
- reactive + global LRU；
- immediate reject。

**Activate**

- foreground-aware；
- foreground-aware + fixed foreground/background slice；
- dual-queue/static priority；
- single-size-aware；
- FCFS/no-slice；
- foreground-aware without event-loop macrotask yield，但保留 checkpoint/requeue；
- foreground-aware without cooperative slicing；
- foreground-aware without handoff grace。

---

## 4. 工作负载

### 4.1 Controlled handoff graph

这是论文的主因果 workload，不是 smoke。一个 run 包含四个标签页：

1. 前台目标标签页建立一个较长的可恢复 session；
2. 目标标签页进入 hidden；
3. 其他标签页产生多个 competing sessions，制造 tier/slot 压力；
4. 至少一个后台标签页开始长 generation；
5. 用户切回目标标签页并立即提交 follow-up；
6. 所有请求完成后记录最终状态和成本。

该 workload 在同一个 run 中连续触发 Retain、Admit 和 Activate，所以它是
论文端到端主表的依据。

默认参数：

```text
clients=4
pause=5 s
pressure_sessions=8
target_prompt_repeats=16
pressure_prompt_repeats=12
background_predict=128
foreground_predict=32
n_ctx=8192
n_seq_max=8
L1/L2/L3=512/2048/8192 tokens
```

正式模型上先用 1 trial 校准，使 Full、Shared-Default 和 Cold-FCFS 都能完成
大多数请求，同时至少一部分 run 出现非 L1 restore 或 admission repair。

### 4.2 Controlled mechanism workloads

三个 microbenchmark 分别只改变一个阶段：

- Retain：固定恢复目标和压力过程，只替换 retention policy；
- Admit：固定 `retention=lru` 以关闭 lifecycle protection，固定状态集合、容量和
  恢复目标，只替换 admission policy；
- Activate：固定后台长任务和前台到达时刻，只替换 scheduler/yield/grace。

### 4.3 WebArena-derived replay

真实 workload 由公开 WebArena task configuration 生成，必须保存：

- 上游仓库 revision；
- 原始 `test.raw.json` 的 hash；
- 选中的 task IDs；
- 过滤规则；
- 可选 page-context sidecar；
- 最终 trace JSON。

默认选择单站点、`string_match`、只读 retrieve tasks，排除会修改站点状态的
post/delete/edit/purchase 等任务。按 site 映射到四个逻辑标签页，任务到达时
切换相应 tab 的 lifecycle，并在该 tab 的 session 上继续请求。

生成命令：

```bash
npm run experiment:browser-service:prepare-trace -- \
  --input=/path/to/webarena/config_files/test.raw.json \
  --contexts=/path/to/task_page_contexts.json \
  --output=/path/to/traces/webarena-readonly.json \
  --task-count=80 \
  --client-count=4 \
  --switch-interval-ms=750 \
  --output-tokens=32
```

若没有 page-context sidecar，可以先做 workload-characterization，但不得把只有
task intent 的 replay 表述成完整 WebArena agent execution。主论文最好使用
真实页面提取的 context；完整环境准备方法见已有
`examples/webarena-latency-bench/scripts/`。

### 4.4 UI co-run

在真实 WebGPU 页面中同时运行持续动画/输入任务和 LLM workload。采集：

- average FPS；
- P95/P99 frame time；
- long-frame ratio；
- input-to-next-paint P50/P95/P99；
- foreground TTFT；
- inference makespan。

已有 `ui-stability-bench` 可作为基础，但正式版本必须加入
`Shared-Default` 与 `wLLM-Full`，不能只比较 “No-Tech/Full-Tech”。

---

## 5. 必做实验总表

### E0：Motivation characterization

**目的：** 使用公开 browser-agent task 和页面上下文构造可审计、确定性的
多标签页 replay，并验证该 replay 确实产生了论文要研究的 pause、resume、
容量压力和前后台重叠。它不用于证明这些行为在自然用户中的发生频率。

输入：WebArena-derived trace 和真实 page context。

输出：

- 同时存活的 logical sessions 分布；
- pause duration CDF；
- resume fraction；
- reuse distance；
- resume 时 working-set/capacity 比；
- foreground arrival 与 background execution overlap 比例；
- 若不做 slicing，foreground 遇到的 residual unit time 分布。

论文产物：

- Figure 2：pause duration 和 reuse distance CDF；
- Table 1：trace 数量、站点、session 数、resume 比例、overlap 比例；
- 一个代表性 timeline。

E0 支撑“该问题会在代表性 browser-agent workload 中出现”，但构造的 tab
schedule 不能证明自然用户行为中的发生频率。若论文要主张 prevalence，还需另行
采集匿名化真实浏览器 lifecycle trace。

### E1：端到端 foreground handoff

**目的：** 回答 RQ0，作为论文最重要的结果。

运行：

```bash
npm run experiment:browser-service:matrix -- \
  --suite=e2e --group=e2e --mode=webgpu --trials=10 \
  --device-id=<ID> --operator=<NAME> --device-file=<DEVICE_JSON> \
  --model-file=<MODEL> --wasm-build=<WASM> --output=<OUT>
```

方法：

- wLLM-Full；
- Shared-Default；
- Cold-FCFS；
- 三个单组件 ablation。

主指标：

- foreground resume-to-first-token；
- foreground completion latency；
- input-to-next-paint P95；
- request success rate；
- workload makespan 和 aggregate tokens/s；
- background P95 latency；
- peak tier occupancy 和 snapshot bytes。

论文产物：

- Table 2：三种完整系统的主要延迟、成功率和吞吐；
- Figure 3：foreground latency CDF；
- Figure 4：foreground latency 与 background slowdown 的 Pareto 图；
- Figure 5：三个 component ablation。

### E2：真实 trace replay

**目的：** 回答 RQ4，证明收益不限于人工构造的单个 case。

运行：

```bash
npm run experiment:browser-service:matrix -- \
  --suite=trace --group=e2e --mode=webgpu --trials=5 \
  --trace-file=<TRACE_JSON> \
  --device-id=<ID> --operator=<NAME> --device-file=<DEVICE_JSON> \
  --model-file=<MODEL> --wasm-build=<WASM> --output=<OUT>
```

比较：

- trace-full；
- trace-shared-default；
- trace-cold-fcfs。

报告：

- request TTFT P50/P95/P99；
- request latency P50/P95/P99；
- completion rate；
- throughput；
- L1/L2/L3/rebuild 比例；
- admission repair/reject；
- peak state footprint；
- per-site breakdown。
- exact/must-include 可确定评分任务的输出准确率；fuzzy-match 使用官方 evaluator。

论文产物：Table 3 和按 site 分组的 CDF/box plot。

### E3：Retain mechanism

**目的：** 回答 RQ1。

运行 `suite=core, group=retain`，每 case 10 次。

变量：

- policy；
- pause duration；
- working-set pressure；
- target prefix length。

指标：

- resume-to-first-token；
- L1/L2/L3/rebuild distribution；
- protected target 是否在保护窗口内持续出现在 latest applied hot set；
- protected-target binding miss count（完整 lifecycle-aware 配置必须为 0）；
- protected-victim violation count（必须为 0）；
- cache-hint failure count（正式 run 必须为 0）；
- replayed/rebuilt work；
- released states/tokens；
- peak snapshot bytes；
- disk reads/writes；
- success rate。

主表至少包含生命周期策略、去掉 lifecycle 的 hybrid、LRU、TTL、
evict-on-pause 和 retain-until-full。若 lifecycle-aware 与
hybrid-no-lifecycle 没有稳定差异，应收窄 Retain novelty，而不是只挑有利设备。

### E4：Admit mechanism

**目的：** 回答 RQ2。

运行 `suite=core, group=admit`，每 case 10 次。

指标：

- completion/rejection rate；
- admission latency P50/P95；
- repair steps P50/P95/max；
- initial token/slot deficit；
- demoted/checkpointed tokens；
- victim count；
- removed states；
- capacity violation count。
- protected-victim violation count（启用 protection 的配置必须为 0）。
- cache-hint failure count（正式 run 必须为 0）。

核心判据：

1. monotonic 的每次 repair steps 不得超过 budget；
2. 所有 state snapshot 中 `usage <= cap`；
3. 相比 reject，monotonic 完成更多可修复请求；
4. 相比 reactive，monotonic 限制 tail repair cost，但可能牺牲部分成功率。

### E5：Activate mechanism

**目的：** 回答 RQ3。

运行 `suite=core, group=activate`，每 case 10 次。

指标：

- foreground activation P50/P95/P99；
- lifecycle signal delivery P50/P95；
- foreground TTFT；
- residual background slice P95/max；
- foreground completion；
- background completion slowdown；
- UI next-paint P95；
- makespan 和 throughput；
- overdue promotion count。

必须同时比较 fixed foreground-aware slice 和 fixed sliced dual queue，不能只与
FCFS/no-slice 比。前者隔离 adaptive background slice，后者隔离 foreground
priority；否则无法说明收益来自哪个部分。

此外，`no-event-loop-yield` 必须只关闭 macrotask handoff，保留 cooperative
checkpoint/requeue；`no-cooperative-slicing` 才关闭整个切片路径。两者不能合并，
否则无法区分 native residual blocking 和 SharedWorker 消息不可见这两个边界。

### E6：Sensitivity

主文保留三个最直接的 tradeoff：

1. Retain：pause duration `0/1/5/30/60 s`；
2. Admit：repair budget `1/2/4/8/16/32`；
3. Activate：slice `4/8/16/32/64 tokens`。

Activate 的 slice sweep 除 foreground/background latency 与 throughput 外，还必须
报告 checkpoint count、admission count、repair steps 和 moved tokens。原因是
每次 cooperative yield 都会把 partial state 带回 admission path；如果只看调度
延迟，就没有验证 Responsiveness 与 Progress 的交互代价。

此外，主文需要报告一个直接验证 contract interaction 的二维 sweep：

```text
lifecycle-protection budget K = 1/2/4
admission repair budget B = 1/4/8
```

该实验固定完整 handoff workload，联合报告 warm-restore 比例、resume TTFT、
admission rejection、repair steps 和 displaced tokens。如果增加 `K` 只改善
restore、却不会提高 admission pressure，或者改变 `B` 不会改变 completion/churn，
则“protection 与 progress contract 相互制约”的统一故事不成立。

附录：

- Admit sessions `4/8/12/16`；
- background generation `64/128/256/512 tokens`；
- equal per-tier capacity `128/256/512 tokens`；
- target prompt repeats `4/8/16`，作为 prefix-length sensitivity。

每点 5 次，在四台代表性设备上运行。

### E7：Scaling and saturation

**目的：** 回答 RQ5 中的扩展性。

运行 `suite=scaling, group=system`：

- clients `2/4/8`；
- requests 为 `4 × clients`；
- wLLM-Full 与 FCFS baseline；
- 每点 5 次。

报告：

- foreground/background request latency P95；
- aggregate tokens/s；
- completion rate；
- queue-wait P95；
- makespan；
- peak state footprint；
- overdue promotions。

如果 8 tabs 已经超过最低配置设备能力，不隐藏失败，应报告饱和点。

### E8：成本和控制面开销

**目的：** 回答“收益是否来自大量额外资源”。

需要分解：

- 首次 model load 时间；
- 2/4/8 个总标签页下，额外 `N-1` 个标签页的总 attach 时间；
- scheduler decision 总时间和 P95；
- cache-hint 总时间和 P95；
- admission planning/repair latency；
- restore/rebuild/prefill/decode/post 时间；
- control-plane time / workload makespan；
- peak L1/L2/L3 tokens；
- snapshot bytes；
- disk reads/writes；
- foreground 获益对应的 background slowdown。

运行：

```bash
npm run experiment:browser-service:matrix -- \
  --suite=overhead --group=system --mode=webgpu --trials=10 ...
```

注意：`modelLoadMs` 与 `additionalClientAttachMs` 证明 SharedWorker 复用模型加载，
但当前浏览器自动化不能可靠给出跨平台 GPU allocation bytes。论文中应报告
runtime state footprint，而不是把 JS heap 或操作系统 RSS 冒充 GPU memory。

### E9：Case study 与 failure analysis

从正式 trace 中按预先规则选择四类 case：

1. **Hot return**：短暂停后从 L1 恢复；
2. **Cold but recoverable**：从 L2/L3 replay；
3. **Repair-bound rejection**：monotonic 达到 budget；
4. **Foreground interruption**：长背景任务中途发生 tab return。

每个 case 画一条事件 timeline：

```text
lifecycle → resume → admission steps → slice boundary → first token → complete
```

选择规则必须由事件字段决定，而不是看完结果后挑最好看的样例。另列真实失败：

- incoming state 本身大于 tier cap；
- WebGPU adapter 不可用；
- 内存压力导致浏览器终止；
- 长 prefill unit 仍造成较高 activation tail；
- background starvation 或大量 overdue promotion。

---

## 6. 实验矩阵与运行规模

### 6.1 主论文设备

最低 6 台，建议 8 台：

- Windows 11：Intel iGPU、AMD APU、NVIDIA/AMD discrete GPU；
- macOS：至少两代 Apple Silicon，覆盖 8/16/24 GB 或以上；
- Linux 只用于开发和 CPU sanity，不进入 Chrome-first 跨设备主结论。

### 6.2 重复次数

| 实验 | 默认重复 |
| --- | ---: |
| Smoke | 1，不进论文 |
| E1 controlled end-to-end | 10 |
| E2 trace replay | 5 |
| E3/E4/E5 mechanism | 10 |
| E6 sensitivity | 5 |
| E7 scaling | 5 |
| E8 overhead | 10 |

每个设备内部按 trial 配对比较，case 顺序轮换，case 间冷却 15 秒。报告 median、
IQR 和 P95；跨设备先求 device-level median，再汇总，不能把所有 run 直接混成
一个平均数。

### 6.3 正式模型

主实验固定一个约 1B 的 instruction-tuned Q4 模型。可以使用当前建议的
Llama-3.2-1B-Instruct-Q4_0，但正式分发前必须确认：

- 许可允许实验和 artifact 分发；
- 所有设备可稳定加载；
- tokenizer、chat template 和 GGUF 完全相同；
- WebLLM 外部 baseline 使用尽可能接近的模型家族与规模。

Tiny Stories 15M 只允许 smoke 和 runner 调试。

### 6.4 分工方式

两台 anchor 设备各运行一次完整论文矩阵：

```bash
npm run experiment:browser-service:paper -- \
  --profile=primary \
  --mode=webgpu \
  --device-id=<ID> \
  --operator=<NAME> \
  --device-file=<DEVICE_JSON> \
  --model-file=<MODEL> \
  --wasm-build=<WASM> \
  --trace-file=<TRACE_JSON> \
  --output=<OUT>
```

其余设备运行 replication profile，覆盖端到端主 baseline 和三个核心机制的
代表性比较：

```bash
npm run experiment:browser-service:paper -- \
  --profile=replication \
  --mode=webgpu \
  --device-id=<ID> \
  --operator=<NAME> \
  --device-file=<DEVICE_JSON> \
  --model-file=<MODEL> \
  --wasm-build=<WASM> \
  --output=<OUT>
```

建议：

- Anchor A：一台中高端 Windows；
- Anchor B：一台 Apple Silicon Mac；
- Replication：至少再增加两台 Windows 和两台 Mac；
- 外部 WebLLM baseline、完整 sensitivity 和 trace replay 只要求 anchor
  设备执行；
- 其余设备只验证主要结论的方向和跨设备稳定性。

运行规模和职责可概括为：

| 角色 | 设备要求 | Profile/内容 | 默认正式 runs | 论文用途 |
| --- | --- | --- | ---: | --- |
| Anchor A | Windows WebGPU | `primary` | 530 | 一套完整主表、敏感性、扩展性、开销、trace |
| Anchor B | Apple Silicon WebGPU | `primary` | 530 | 跨平台完整复核 |
| Replication | 至少 2 台 Windows + 2 台 Mac | `replication` | 每台 55 | 核心结论的跨设备稳定性 |
| Optional external baseline | 两台 anchor | WebLLM + wllama no-reuse | 依 task 数而定 | 证明基础推理效率没有失真 |

这里的 530 个 primary runs 来自 60 个 E2E、180 个 Core、215 个
Sensitivity、30 个 Scaling、30 个 Overhead 和 15 个 Trace runs。可以按 suite
分批执行，但不能在批次间改变冻结 commit、模型、Wasm、TTL 或 workload 参数。

---

## 7. 论文表格和图

### Table 1：Workload characterization

| Workload | Tasks | Tabs | Sessions | Resume fraction | Median/P95 pause | P95 live working set | FG/BG overlap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Controlled | TBD | 4 | TBD | TBD | TBD | TBD | TBD |
| WebArena-derived | TBD | 4 | TBD | TBD | TBD | TBD | TBD |

### Table 2：End-to-end handoff

| System | Resume TTFT P50/P95 ↓ | FG latency P95 ↓ | Next-paint P95 ↓ | Success ↑ | Throughput ↑ | BG slowdown ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cold-FCFS | TBD | TBD | TBD | TBD | TBD | TBD |
| Shared-Default | TBD | TBD | TBD | TBD | TBD | TBD |
| wLLM-Full | TBD | TBD | TBD | TBD | TBD | TBD |

### Table 3：WebArena-derived replay

| System | TTFT P50/P95 ↓ | Latency P50/P95 ↓ | Completion ↑ | Throughput ↑ | L1/L2/L3/Rebuild | Peak state |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Cold-FCFS | TBD | TBD | TBD | TBD | TBD | TBD |
| Shared-Default | TBD | TBD | TBD | TBD | TBD | TBD |
| wLLM-Full | TBD | TBD | TBD | TBD | TBD | TBD |

### Table 4：Retain

| Policy | Resume TTFT P50/P95 ↓ | L1 hit ↑ | Rebuild ↓ | Released tokens | Peak state bytes | Disk reads/writes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Lifecycle + Hybrid | TBD | TBD | TBD | TBD | TBD | TBD |
| Hybrid, no lifecycle | TBD | TBD | TBD | TBD | TBD | TBD |
| LRU | TBD | TBD | TBD | TBD | TBD | TBD |
| TTL | TBD | TBD | TBD | TBD | TBD | TBD |
| Evict on pause | TBD | TBD | TBD | TBD | TBD | TBD |
| Retain until full | TBD | TBD | TBD | TBD | TBD | TBD |

### Table 5：Admit

| Policy | Success ↑ | Admission P95 ↓ | Repair P95/max ↓ | Victims | Tokens moved | Rejects | Violations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Monotonic + Hybrid | TBD | TBD | TBD | TBD | TBD | TBD | 0 required |
| Reactive + Hybrid | TBD | TBD | TBD | TBD | TBD | TBD | 0 required |
| Reactive + LRU | TBD | TBD | TBD | TBD | TBD | TBD | 0 required |
| Immediate reject | TBD | TBD | 0 | 0 | 0 | TBD | 0 required |

### Table 6：Activate

| Scheduler | Signal P95 ↓ | Post-signal activation P95 ↓ | FG TTFT P95 ↓ | Next-paint P95 ↓ | BG slowdown ↓ | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Foreground-aware | TBD | TBD | TBD | TBD | TBD | TBD |
| Fixed foreground slice | TBD | TBD | TBD | TBD | TBD | TBD |
| Dual queue | TBD | TBD | TBD | TBD | TBD | TBD |
| Size-aware | TBD | TBD | TBD | TBD | TBD | TBD |
| FCFS/no-slice | TBD | TBD | TBD | TBD | TBD | TBD |
| No event-loop yield | TBD | TBD | TBD | TBD | TBD | TBD |
| No cooperative slicing | TBD | TBD | TBD | TBD | TBD | TBD |
| No grace | TBD | TBD | TBD | TBD | TBD | TBD |

### Table 7：Overhead

| Device | Model load | Extra-tab attach | Scheduler P95 | Cache-hint P95 | Control-plane % | Peak state bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### Figures

1. Motivation CDF：pause duration、reuse distance、foreground overlap。
2. End-to-end foreground latency CDF。
3. Foreground latency vs background slowdown Pareto。
4. 三组件消融柱状图。
5. Retain pause-duration 曲线。
6. Admit budget-success/churn 曲线。
7. Activate slice-latency/throughput 曲线。
8. Client scaling 曲线。
9. 代表性 event timeline。

---

## 8. 统计规则

- 延迟报告 P50、P95，必要时报告 P99；
- 每个配置至少 5 个有效 trial，主表默认 10；
- 同设备、同 trial 做配对比较；
- 报告绝对差和百分比差；
- 设备是 random effect，不把不同设备的 raw runs 当独立同分布样本；
- 关键比较提供 bootstrap 95% confidence interval；
- 不因结果不符合预期而删除 run；
- controlled admission rejection 是结果，不是 crash；
- timeout、浏览器退出、adapter 丢失等基础设施错误保留原始目录并标无效；
- accuracy 不是本文核心指标，但 baseline 必须使用相同 prompt、sampling 和
  output cap，避免用质量下降换取性能。

---

## 9. 数据收集与直接生成论文统计

汇总：

```bash
npm run experiment:browser-service:collect -- \
  --input=<ALL_RUNS> \
  --output=<ALL_RUNS>/aggregate.csv \
  --expected-git=<COMMIT> \
  --expected-model-sha256=<MODEL_SHA256> \
  --expected-wasm-sha256=<WASM_SHA256> \
  --expected-trace-sha256=<TRACE_SHA256>
```

生成按设备、suite、case 聚合的论文数据和配对比较：

```bash
npm run experiment:browser-service:summarize -- \
  --input=<ALL_RUNS>/aggregate.csv \
  --output=<ALL_RUNS>/paper-summary
```

输出：

```text
paper-summary.csv
paired-comparisons.csv
paper-results.json
paper-tables.md
```

提取 request/admission/state/UI/timeline 级别的画图数据：

```bash
npm run experiment:browser-service:extract-events -- \
  --input=<ALL_RUNS> \
  --output=<ALL_RUNS>/paper-events
```

输出：

```text
request-level.csv
admission-level.csv
state-snapshots.csv
ui-measurements.csv
timelines.csv
```

这些文件直接对应第 7 节的表格、CDF、tradeoff 图和 case timeline。原始
`events.jsonl` 仍须保留用于审计。

---

## 10. 执行顺序与停止条件

### P0：正式实验前必须完成

1. 冻结 Methodology 表述，删除 dedicated Worker、尚未接 lifecycle 等过期描述；
2. 冻结代码 commit、模型和 WebGPU Wasm；
3. 在一台 Windows 和一台 macOS 上分别完成 smoke；
4. 在正式 1B/WebGPU 上各跑一轮 E1/E3/E4/E5 pilot；
5. 校准 capacity，使 Retain 出现至少两种恢复层级，Admit 出现非零 repair，
   Activate 的后台任务在 foreground arrival 时仍在运行；
6. 生成并审计 WebArena-derived trace。

### P1：主文数据

1. E0 workload characterization；
2. E1 end-to-end；
3. E2 trace replay；
4. E3 Retain；
5. E4 Admit；
6. E5 Activate；
7. E8 overhead。

### P2：完整性和附录

1. E6 sensitivity；
2. E7 scaling；
3. E9 case study/failure analysis；
4. 外部 WebLLM 和 wllama no-reuse 基础性能比较。

能够开始写完整 Evaluation 的最低停止条件：

- Table 1–7 的所有主指标有真实数据；
- 每个 headline claim 至少有主实验和一个机制实验支持；
- 至少 Windows/macOS 各三台设备；
- WebArena-derived trace 的来源、筛选和 event sequence 可审计；
- 所有主实验 run 来自同一干净 commit、模型和 Wasm；
- capacity violation 为 0；
- 不再用 Tiny Stories smoke 数字填写论文。

---

## 11. 当前实现状态

已经具备：

- SharedWorker 多标签页共享模型；
- Protection、Progress、Responsiveness 三个 contract；
- `smoke/core/e2e/sensitivity/scaling/overhead/trace` suites；
- 版本化事件、wall-clock 时间和 lifecycle 时间；
- request、slice、admission、state、UI 事件；
- model load、client attach、scheduler/cache-hint/control-plane 开销；
- state footprint 和 capacity violation 检查；
- WebArena task config 到 replay trace 的转换器；
- 跨设备 metadata、结果 collector 和论文统计 summarizer。

仍需要外部准备：

1. 正式 1B Q4 模型；
2. 统一 WebGPU Wasm build；
3. 6–12 台 Windows/macOS 设备；
4. WebArena `test.raw.json`；
5. 最好提供真实 page-context sidecar；
6. 外部 WebLLM baseline 所需的模型缓存。

当前 Linux CPU/Tiny Stories 结果只证明基础设施可执行，不属于论文 evidence。

---

## 12. 论文 Evaluation 可直接采用的结构

```text
6.1 Experimental Setup
    Hardware, browser, model, workloads, baselines, metrics, statistics

6.2 Are browser foreground handoffs a material workload?
    WebArena-derived characterization (E0)

6.3 End-to-end foreground handoff
    Controlled graph + trace replay (E1, E2)

6.4 Why does Retain work?
    Policy comparison + pause/pressure sensitivity (E3, E6)

6.5 Is admission bounded under pressure?
    Success/repair/churn/capacity invariant (E4, E6)

6.6 How quickly can execution move to the foreground?
    Scheduler comparison + slice tradeoff + UI co-run (E5, E6)

6.7 Overhead, scaling, and resource cost
    Control-plane, state footprint, disk I/O, clients and saturation (E7, E8)

6.8 Case studies and limitations
    Four event timelines and observed failure cases (E9)
```

按这套方案跑完后，结果可以直接落入论文的 Evaluation 章节，而不是再从 smoke
数据临时拼故事。
