# UI Stability Bench

同页运行两类负载：

- 持续渲染一个中等复杂度的 WebGPU UI workload
- 同时发起一批本地模型推理请求

页面会顺序对比两种模式：

- `No-Tech`: 无树缓存、无 engine-chat 队列、FCFS 直跑
- `Full-Tech`: 共享前缀 + tree cache + engine-chat 队列/切片

页面还支持 `Pressure Scan`：

- 自动扫描多档压力
- 生成 `FPS vs Pressure` 曲线
- 生成 `UI P95 Frame vs Pressure` 曲线
- 生成 `Makespan vs Pressure` 曲线
- 导出完整 `scan report.json`

目标是观察：

- UI 帧稳定性是否变差
- 整批请求全部完成需要多久
- 使用技术点后，UI 与整体完成时间是否一起改善

当前默认 workload 特意偏向：

- 超长共享前缀
- 较短回答
- 更重的同页 WebGPU UI 负载

这样更容易体现这套引擎当前最擅长的部分：共享前缀复用、prefill 切分、以及减少长 prompt 对 UI 的冲击。

## 运行

```bash
cd examples/ui-stability-bench
npm install
npm run dev
```

然后点击页面里的：

- `Run A/B Bench`
- 或 `Run Pressure Scan`

也可以在仓库根目录直接启动：

```bash
npm run example:ui-stability-bench
```

默认 `Model URL` 会指向仓库里现成的 `examples/sglang-style-mmlu-hellaswag/model/...gguf`。
如果你本地模型放在别处，直接改页面里的 `Model URL` 输入框即可。

## 产出

页面支持下载：

- `Download Logs`
- `Download Bench Report`
- `Download Scan Report`

其中导出的 json 会额外包含：

- 每个请求的完成/失败结果
- 浏览器环境信息
- 每个压力点对应的真实参数
- runtime debug snapshot，方便继续迭代调度与切片参数
