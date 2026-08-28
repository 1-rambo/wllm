const quantile = (values, q) => {
  if (!values.length) return 0;
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[
    Math.min(sorted.length - 1, Math.floor(q * (sorted.length - 1)))
  ];
};

export const createUiWorkload = async (canvas, recordEvent) => {
  let running = false;
  let frameHandle = 0;
  let lastFrameAt = 0;
  let samples = [];
  let workItems = 0;
  let gpuState = null;
  let canvas2d = null;

  const initializeBackend = async (enableWebGpu) => {
    if (!enableWebGpu) {
      canvas2d = canvas.getContext('2d', { alpha: false });
      return;
    }
    const adapter = await navigator.gpu.requestAdapter().catch(() => null);
    const device = adapter
      ? await adapter.requestDevice().catch(() => null)
      : null;
    if (device) {
      const module = device.createShaderModule({
        code: `
          @group(0) @binding(0) var<storage, read_write> values: array<f32>;
          @compute @workgroup_size(64)
          fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            if (id.x >= arrayLength(&values)) { return; }
            var value = values[id.x] + f32(id.x) * 0.0001;
            for (var i = 0u; i < 32u; i = i + 1u) {
              value = sin(value) + cos(value * 1.0001);
            }
            values[id.x] = value;
          }
        `,
      });
      gpuState = {
        device,
        pipeline: device.createComputePipeline({
          layout: 'auto',
          compute: { module, entryPoint: 'main' },
        }),
        buffer: null,
        bindGroup: null,
        size: 0,
      };
      return;
    }
    canvas2d = canvas.getContext('2d', { alpha: false });
  };
  const prepareGpuBuffer = (size) => {
    if (!gpuState || gpuState.size === size) return;
    gpuState.buffer?.destroy();
    gpuState.buffer = gpuState.device.createBuffer({
      size: Math.max(4, size * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    gpuState.bindGroup = gpuState.device.createBindGroup({
      layout: gpuState.pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: gpuState.buffer } }],
    });
    gpuState.size = size;
  };

  const frame = async (now) => {
    if (!running) return;
    if (lastFrameAt > 0) samples.push(now - lastFrameAt);
    lastFrameAt = now;
    if (gpuState && workItems > 0) {
      prepareGpuBuffer(workItems);
      const encoder = gpuState.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(gpuState.pipeline);
      pass.setBindGroup(0, gpuState.bindGroup);
      pass.dispatchWorkgroups(Math.ceil(workItems / 64));
      pass.end();
      gpuState.device.queue.submit([encoder.finish()]);
      await gpuState.device.queue.onSubmittedWorkDone();
    } else if (canvas2d && workItems > 0) {
      canvas2d.fillStyle = '#08111f';
      canvas2d.fillRect(0, 0, canvas.width, canvas.height);
      for (let i = 0; i < workItems; i++) {
        const x = (i * 17 + now * 0.07) % canvas.width;
        const y = (i * 31 + now * 0.04) % canvas.height;
        canvas2d.fillStyle = `hsl(${(i + now * 0.02) % 360} 70% 55%)`;
        canvas2d.fillRect(x, y, 2, 2);
      }
    }
    if (running) frameHandle = requestAnimationFrame(frame);
  };

  return {
    async start(nextWorkItems = 0, enableWebGpu = true) {
      await initializeBackend(enableWebGpu);
      workItems = Math.max(0, Math.floor(nextWorkItems));
      samples = [];
      lastFrameAt = 0;
      running = true;
      frameHandle = requestAnimationFrame(frame);
    },
    async stop() {
      running = false;
      if (frameHandle) cancelAnimationFrame(frameHandle);
      frameHandle = 0;
      const totalMs = samples.reduce((sum, value) => sum + value, 0);
      const result = {
        backend: gpuState ? 'webgpu-compute' : 'canvas2d',
        sampleCount: samples.length,
        averageFps: totalMs > 0 ? (samples.length * 1000) / totalMs : 0,
        frameTimeP50Ms: quantile(samples, 0.5),
        frameTimeP95Ms: quantile(samples, 0.95),
        frameTimeP99Ms: quantile(samples, 0.99),
        maxFrameTimeMs: samples.length ? Math.max(...samples) : 0,
        longFrameRatio:
          samples.length > 0
            ? samples.filter((value) => value > 50).length / samples.length
            : 0,
        workItems,
      };
      await recordEvent('ui_measurement', result);
      return result;
    },
  };
};
