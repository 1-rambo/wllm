# agentic-benchmark-ab

一个用于验证 AgenticBrowserBenchmark 全量任务性能收益的 A/B 示例项目。

## 目标

- 只关注性能：TTFT、token/s
- A 组：无缓存
- B 组：树状缓存
- 公共前缀：系统提示词 + 网页内容（start_url/open_url + web context template）
- 覆盖全量任务：自动加载 `AgenticBrowserBenchmark/dataset_v2.0/*.json`

## 当前状态

当前版本已经接入真实 `@wllama/wllama` 推理，不包含任何模拟逻辑：

- 真实模型加载（本地 GGUF）
- 真实流式生成
- 实测 TTFT / token/s / n_reused
- A/B（No Cache vs Tree Cache）全量任务跑批

## 目录

- `src/benchmark/dataset-loader.ts`：加载与标准化 benchmark 数据
- `src/benchmark/prefix-builder.ts`：构建共享前缀
- `src/benchmark/wllama-agent.ts`：真实模型 A/B 执行器
- `src/benchmark/ab-runner.ts`：执行器导出入口
- `src/App.tsx`：可视化结果页面

## 运行

```bash
cd examples/agentic-benchmark-ab
npm install
npm run dev
```

默认模型目录：

- `/Users/rambo/Desktop/wllama-webgpu/examples/prefix-tree-chat/public`

可在页面中修改 `Model Base Directory`。
