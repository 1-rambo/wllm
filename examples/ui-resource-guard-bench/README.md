# ui-resource-guard-bench

用于论文展示“浏览器资源有限条件下，LLM 推理服务对 UI 稳定性影响”的独立实验页面。
当前版本已接入真实 `wllama` 推理请求。
`guarded` 组仅使用 `wllama` 原生能力：engine chat queue、queue_overloaded 背压、chat tree/tiered cache。

## 实验设计

- 组 A（baseline）: 无三技术
  - 请求直连并发到真实模型推理
  - 不使用 chatFromNode/chatSession 的原生队列路径
  - 不使用原生树会话缓存
- 组 B（guarded）: 有三技术
  - 原生队列调度（`chatFromNode` 内部 enqueue/process）
  - 原生背压（`engineChatQueueMaxPending` 触发 `queue_overloaded`）
  - 原生树缓存复用（`chatSessionInit/chatFromNode`）

在同一页面运行动画渲染，并统计:

- UI 渲染: 平均 FPS、P95 frame time
- 点击任务响应: 平均延迟、P95 延迟（基于 event loop 探针 + 手动按钮测试）
- 请求侧: 提交/完成/丢弃/失败、平均和 P95 请求延迟
- 请求终态: 平均和 P95 终态延迟（完成/丢弃/超时统一口径，避免幸存者偏差）
- 推理侧: 平均和 P95 服务耗时、TTFT

关键参数（建议保持论文预设）：

- `engineChatQueueMaxPending`: 原生队列最大 pending
- `engineChatServiceUpperBoundMs`: 原生调度预算上界
- `treeMemoryCapMB`: 树会话内存上限

修改以上原生参数后，需要“加载/重载模型”生效。

## 一键运行

```bash
cd examples/ui-resource-guard-bench
npm install
npm run dev
```

打开页面后点击“一键跑实验”。

建议流程：

1. 点击“加载模型”
2. 点击“论文参数预设”
3. 等状态变为 `Model ready`
4. 点击“一键跑实验”
5. 导出 CSV 用于论文图表

## 建议论文叙事点

- 在高压档位下，baseline 组通常会出现 FPS 下降、frame time 与点击延迟上升。
- guarded 组由于“队列调度 + 缓存复用 + 背压限制”，通常可保持更稳定的 UI 与交互响应。
- 论文解读建议：将“提交量”与“完成量/丢弃量”同时展示，强调系统稳定性来自可控退化，而不是无限排队。
- 页面支持导出 CSV，可直接用于论文画图。
