# wLLM 论文实验任务通知

下面的“可直接转发版本”用于向实验执行同学解释本次实验到底要证明什么。具体安装、
命令、故障处理和数据回收请配合 `DEVICE_RUNBOOK.zh-CN.md` 使用；完整的
claim、baseline、消融、指标和论文图表设计见 `EXPERIMENT_PLAN.zh-CN.md`。

---

## 可直接转发版本

这次实验不是做 smoke，也不是简单比较几个配置谁更快，而是要为论文验证一个完整的
系统主张：当多个浏览器标签页共享一个本地 LLM runtime 时，用户切回某个标签页会
同时涉及三个问题，即会话状态是否仍可低成本恢复、容量不足时恢复过程是否有界，以及
后台推理能否及时把执行权交给前台。我们的系统分别用 Protection、Progress 和
Responsiveness 三个 contract 处理这三个问题，因此实验必须同时回答“端到端是否
有效”“每个机制为什么有效”以及“这些收益付出了什么代价”。

你需要在分配给你的 Windows 或 macOS 设备上，使用负责人统一提供的代码 commit、
1B Q4 GGUF 模型、WebGPU Wasm 和 WebArena-derived trace。请不要自行修改模型、
容量、slice、浏览器参数或实验次数。先填写 `device.json` 并运行一次 smoke，确认
实际执行模式为 WebGPU，SharedWorker、状态管理和 telemetry 正常，而且没有非预期
错误；smoke 只检查环境，不进入论文结果。通过后再运行负责人分配的正式 profile
或实验组：

1. **端到端 handoff**：比较 wLLM-Full、Shared-Default、Cold-FCFS 和三个组件
   消融。目的是证明完整系统能够降低用户返回标签页后的首 token 和前台尾延迟，
   同时检查请求成功率、后台延迟、吞吐和 UI next-paint，避免以牺牲后台任务或
   正确性换取前台速度。
2. **Retain**：比较 lifecycle-aware、无 lifecycle 的 hybrid、LRU、TTL、
   evict-on-pause 和 retain-until-full。这里不只是看 cache hit，而是验证
   lifecycle intent 是否始终绑定到正确的 session head、被保护状态是否从未成为
   admission victim，以及这种保护是否确实带来更热的恢复路径和更低的 resume
   TTFT。
3. **Admit**：比较 bounded monotonic repair、reactive repair、reactive+LRU
   和 immediate reject。目的是证明多层状态迁移期间容量始终不越界，monotonic
   repair 的 victim-selection 次数不超过 budget，并展示“完成更多请求”和“限制
   repair tail/churn”之间的真实取舍。受控 admission reject 是有效结果，不要
   删除。
4. **Activate**：比较完整 foreground-aware 方法、固定 foreground slice、
   dual queue、size-aware、FCFS/no-slice、no-event-loop-yield、
   no-cooperative-slicing 和 no-grace。目的是分别验证 bounded native unit、
   SharedWorker macrotask yield 和 request-arrival grace 三个环节，重点记录
   signal delivery、request arrival、post-arrival dispatch、foreground TTFT、
   UI next-paint、后台 slowdown 和吞吐。
5. **Anchor 设备的补充任务**：运行 WebArena-derived trace、参数敏感性、扩展性
   和 overhead。它们分别用于证明结论不只成立于单个人工 case，找出 protection
   budget、repair budget 和 slice size 的有效区间，并量化 client 数增长、状态
   footprint、checkpoint I/O、调度和 cache-hint 的实际成本。

正式实验必须使用干净工作区并保留脚本生成的完整目录。不要只发截图或手工整理后的
Excel；每个 run 的 `summary.json`、`events.jsonl` 和矩阵目录中的
`matrix.json` 都必须提交。遇到浏览器退出、机器休眠、WebGPU 不可用或路径错误，
可以保留失败目录后重跑；但结果慢、出现极值、monotonic/reject 发生受控拒绝，都
不能作为删数据的理由。完成后请同时提交设备信息、Git revision、模型与 Wasm
SHA-256、所跑实验组、异常记录和整个输出目录。最终统计会在同一设备内按 trial
配对，报告 median、IQR、P95 和 bootstrap 95% CI，并先按设备汇总，再讨论跨设备
趋势。

---

## 负责人分配任务时需要补全

```text
实验代码分支：dev
冻结 commit：<EXPERIMENT_COMMIT>
你的 deviceId：<DEVICE_ID>
你的实验角色：<anchor-primary / replication / 指定 group>
正式模型：<MODEL_GGUF>
模型 SHA-256：<MODEL_SHA256>
WebGPU Wasm：<WEBGPU_WASM_DIR>
Wasm SHA-256：<WASM_SHA256>
device.json 模板：examples/browser-service-smoke/device.example.json
Trace 文件（anchor 才需要）：<TRACE_JSON>
结果提交位置：<UPLOAD_PATH>
截止时间：<DEADLINE>
```

## 推荐分工

### Anchor 设备

至少一台 Windows 和一台 Apple Silicon Mac。两台都运行完整 `primary`
profile，用于主表、完整消融、敏感性、扩展性、开销和 trace replay：

```bash
npm run experiment:browser-service:paper -- \
  --profile=primary \
  --mode=webgpu \
  --device-id=<DEVICE_ID> \
  --operator=<NAME> \
  --device-file=<DEVICE_JSON> \
  --model-file=<MODEL_GGUF> \
  --wasm-build=<WEBGPU_WASM_DIR> \
  --trace-file=<TRACE_JSON> \
  --tuned-retention-ttl-ms=<FROZEN_TTL_MS> \
  --output=<OUTPUT_DIR>
```

当前 primary profile 共包含：

```text
E2E:           6 cases × 10 trials =  60 runs
Core:         18 cases × 10 trials = 180 runs
Sensitivity: 43 cases ×  5 trials = 215 runs
Scaling:       6 cases ×  5 trials =  30 runs
Overhead:      3 cases × 10 trials =  30 runs
Trace:         3 cases ×  5 trials =  15 runs
Total:                                530 runs
```

正式开跑前先用 `--trials=1` 估算该设备耗时。若需要分批，可通过
`--suites=e2e,core` 等参数拆分，但所有批次必须使用同一冻结配置。

### Replication 设备

至少再安排两台 Windows 和两台 Mac，运行 `replication` profile，验证主结论在
不同硬件上方向一致：

```bash
npm run experiment:browser-service:paper -- \
  --profile=replication \
  --mode=webgpu \
  --device-id=<DEVICE_ID> \
  --operator=<NAME> \
  --device-file=<DEVICE_JSON> \
  --model-file=<MODEL_GGUF> \
  --wasm-build=<WEBGPU_WASM_DIR> \
  --tuned-retention-ttl-ms=<FROZEN_TTL_MS> \
  --output=<OUTPUT_DIR>
```

Replication profile 包含 11 个 case，每个默认 5 次，共 55 个正式 run：

- 三个端到端系统：Full、Shared-Default、Cold-FCFS；
- Retain：Lifecycle+Hybrid 与 LRU；
- Admit：Monotonic 与 Reactive；
- Activate：Full、固定 foreground slice、dual queue、no-event-loop-yield。

### 仅负责单个机制的同学

如果按机制分工，使用下面之一：

```bash
# Retain：6 cases × 10 trials
npm run experiment:browser-service:matrix -- \
  --suite=core --group=retain --mode=webgpu --trials=10 ...

# Admit：4 cases × 10 trials
npm run experiment:browser-service:matrix -- \
  --suite=core --group=admit --mode=webgpu --trials=10 ...

# Activate：8 cases × 10 trials
npm run experiment:browser-service:matrix -- \
  --suite=core --group=activate --mode=webgpu --trials=10 ...
```

## 正式运行前的统一校准

校准只在两台 anchor 设备上进行，由负责人统一决定参数，不能由各执行者分别调参。

1. 用正式模型分别跑一轮 E2E、Retain、Admit、Activate。
2. 确认 Retain 至少出现两类 restore source，而不是所有策略都停留在 L1。
3. 确认 Admit 同时出现非零 repair、monotonic 的 budget-bound 行为和 reactive
   的更长 repair chain，但不能让绝大多数请求都失败。
4. 确认 Activate 的 foreground 到达时后台请求仍在运行。
5. 先完成 TTL 的 `1/5/30/60 s` calibration，按预先固定的 resume-TTFT、
   state-footprint 和 rejection 规则选定一个 TTL，然后所有设备通过
   `--tuned-retention-ttl-ms` 使用同一个值。
6. 参数一旦冻结，pilot 结果与正式结果分目录保存，pilot 不进入论文统计。

## 结果验收

一组结果只有同时满足以下条件才可进入论文：

- `requestedExecutionMode` 与 `actualExecutionMode` 都是 `webgpu`；
- Git、模型和 Wasm 与冻结版本一致，且 `gitDirty=false`；
- `cacheHintFailureCount=0`、`protectedVictimViolationCount=0`、
  `capacityViolationCount=0`、`admissionBudgetViolationCount=0`；
- 需要保护目标的 case 中，`protectedTargetHintCoverage=1`，且发生 admission
  observation 时 `protectedTargetAdmissionCoverage=1`；
- 除受控 admission rejection 外，没有 request failure 或 operation error；
- 每个 run 的原始 event stream 完整存在。

这些检查只判定数据是否可信，不预设性能必须提高。若某个基线与完整方法相当，或某个
设备出现相反趋势，也应原样提交，由负责人据此收窄论文 claim 或分析适用边界。
