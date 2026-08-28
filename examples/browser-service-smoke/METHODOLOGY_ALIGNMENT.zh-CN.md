# Methodology 与实现对齐说明

本文档用于在修改论文 Methodology 前核对“论文写的机制”和“代码真正实现的
机制”。Paper 仓库不在本分支中修改。

## 1. 系统边界

当前系统是同源浏览器标签页共享的本地 LLM service：

```text
Page lifecycle / user request
          ↓
BrowserLLMClient
          ↓ MessagePort
Versioned SharedWorker
          ↓
Wllama engine-chat control plane
          ↓
llama.cpp tree state + finite sequence slots + tier manager
```

SharedWorker 负责唯一模型实例、客户端登记、lifecycle 转发、请求流式返回和
统一 telemetry。它不是多个标签页各自加载模型后再做协调。

## 2. Retain 的真实机制

### 输入

- `foreground`
- `lifecycleState ∈ {active, passive, hidden, frozen}`
- `changedAt`
- `lastForegroundAt`
- 每个 client/logical-session 的 head node
- native tier 中的 age、size、access frequency

### 决策

1. lifecycle policy 生成 protected session heads；
2. pending/active 请求的 parent 也进入 hot-node hint；
3. protection set 按 lifecycle urgency 和最近 foreground 时间排序，并受
   `evictionHintTopK` 限制，避免“所有 session 都 protected”使 eviction
   失去选择空间；
4. native replacement 在非 protected 候选中选择 victim；
5. victim 进入更冷层或被移除；
6. resume 时按 L1、L2、L3、rebuild 顺序恢复。

### 当前完整配置

```text
retention=lifecycle-aware
replacement=hybrid
```

Hybrid eviction score 使用 age、token footprint 和 access count。Lifecycle
部分当前是 protection，不是概率预测。

### 论文应修改

当前 Methodology 的

```text
V(v,t) = p_resume(v,t) R_avoid(v) - lambda LiveCost(v,t)
```

可以保留为设计目标，但不能表述成已实现的 learned predictor。准确描述应为：

> wLLM first derives a lifecycle-aware protection set and then applies a
> cost-aware replacement score to unprotected states.

## 3. Admit 的真实机制

### 输入

- incoming token footprint；
- target tier；
- 当前各层 token usage/capacity；
- live sequence-slot usage；
- protected node/ancestry；
- repair budget。

### 决策

容量不足时逐步执行 demotion、checkpoint 或 removal。每步减少 token/slot
deficit，或把一个已选择 victim 移到已经预留容量的更低层。同一 admission
记录：

```text
initialTokenDeficit
initialSlotDeficit
repairSteps
victimIds
demotedTokens
checkpointedTokens
removedStates
latencyMs
success
```

### 策略

- `monotonic`：达到 repair budget 后停止并拒绝；
- `reactive`：继续修复直到成功或没有可行 victim；
- `reject`：发现容量压力直接拒绝。

### 论文应修改

当前实现不是完整 plan 后原子 commit，也没有 rollback。因此不要写成
transactional admission。准确主张是：

> online monotonic repair bounds the number of displacement decisions and
> exposes partial displacement when an admission is rejected.

## 4. Activate 的真实机制

### 输入

- client foreground/lifecycle state；
- normal/overdue queues；
- request age；
- online service-cost estimate；
- current prefill/decode stage；
- foreground/background token budgets。

### 决策

- 每个 cooperative boundary 重新选请求；
- foreground-aware 模式先选 foreground；
- background 在有 foreground pending 时缩短 token budget；
- prefill 同时受 token budget 和 wall-time bound 控制；
- lifecycle 变为 foreground 后进入 handoff grace；
- yield 后通过 macrotask 让 SharedWorker 有机会处理 lifecycle 消息；
- overdue promotion 保留老请求的等待权利。

### 不应声称

- 不提供 GPU kernel preemption；
- 不保证硬实时 deadline；
- activation bound 包含当前 Wasm/GPU 调用能返回控制权的 residual time。

## 5. Tier 语义

当前代码中的 L3 写入 Emscripten runtime 的文件系统路径。除非正式 Wasm build
明确把该路径挂载到持久化后端，否则论文应称为：

```text
cold filesystem-backed checkpoint tier
```

不要直接声称它是浏览器持久化磁盘或 OPFS。若论文必须主张跨 reload 的持久化，
需要另行把 L3 mount 到 OPFS 并增加 reload/recovery 实验。

## 6. Claim 与实验

| Claim | 必要实验 | 失败后的论文处理 |
| --- | --- | --- |
| lifecycle 信息改善 resume | Retain policy comparison + pause/pressure sensitivity | 若只与 LRU 持平，降级为工程整合点 |
| monotonic repair 有界 | budget sweep + per-admission max + zero capacity violations | 若超 budget，属于 correctness bug |
| foreground handoff 更快 | fixed-slice、FCFS、no-yield、no-grace 对比 | 若只胜 FCFS 不胜 fixed slice，收窄 scheduler novelty |
| 三机制联合有效 | controlled handoff + trace replay + three ablations | 若某一 ablation 无影响，不能声称三者缺一不可 |
| 开销可接受 | control-plane、state footprint、disk/file I/O、background slowdown | 报告真实 tradeoff，不只报 foreground gain |
| 真实场景相关 | WebArena-derived content replay；若主张自然发生频率则另需真实用户 trace | synthetic lifecycle 不能证明自然 prevalence |

## 7. 论文当前必须修正的陈述

在正式投稿前，Paper 至少需要修正：

1. `dedicated Web Worker` 改为 `versioned SharedWorker service`；
2. 删除 “lifecycle integration remains”；
3. 把 Retain 的公式从“已实现预测器”改为设计目标，并描述 protection set；
4. 把 Admit 从“完整规划/原子安装”改为 online bounded monotonic repair；
5. 把 L3 的 “disk” 改为与最终 build 的实际文件系统一致；
6. 用新 Evaluation 完整替换旧 MMLU/HellaSwag legacy results；
7. 不把构造的 WebArena lifecycle schedule 写成真实用户行为频率；
8. Abstract 的数字只能来自正式 1B/WebGPU、干净提交、多次重复结果。
