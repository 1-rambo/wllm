export interface RendererWindowStats {
  sampleCount: number;
  avgFps: number;
  avgFrameMs: number;
  p95FrameMs: number;
  maxFrameMs: number;
  longFrameRatioPct: number;
}

export interface RendererLiveStats extends RendererWindowStats {
  running: boolean;
  instanceCount: number;
}

export interface RendererInitOptions {
  instanceCount: number;
}

type MeasurementWindow = {
  label: string;
  frameTimes: number[];
};

const TRIANGLE_STRIDE_FLOATS = 8;
const LIVE_HISTORY_LIMIT = 180;

const shaderCode = `
struct Uniforms {
  time: f32,
  aspect: f32,
  pad0: f32,
  pad1: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VSIn {
  @location(0) offset: vec2<f32>,
  @location(1) scale: f32,
  @location(2) phase: f32,
  @location(3) color: vec4<f32>,
}

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
}

@vertex
fn vsMain(input: VSIn, @builtin(vertex_index) vertexIndex: u32) -> VSOut {
  var base = array<vec2<f32>, 3>(
    vec2<f32>(0.0, 0.030),
    vec2<f32>(-0.026, -0.022),
    vec2<f32>(0.026, -0.022),
  );

  let wobble = vec2<f32>(
    sin(uniforms.time * 0.8 + input.phase) * 0.035,
    cos(uniforms.time * 1.15 + input.phase) * 0.035
  );

  let spin = uniforms.time * 0.35 + input.phase;
  let c = cos(spin);
  let s = sin(spin);
  let rotated = vec2<f32>(
    base[vertexIndex].x * c - base[vertexIndex].y * s,
    base[vertexIndex].x * s + base[vertexIndex].y * c,
  );

  let pos = input.offset + wobble + rotated * input.scale;
  var out: VSOut;
  out.position = vec4<f32>(pos.x / max(0.6, uniforms.aspect), pos.y, 0.0, 1.0);
  out.color = input.color;
  return out;
}

@fragment
fn fsMain(input: VSOut) -> @location(0) vec4<f32> {
  return input.color;
}
`;

function percentile(values: number[], p: number): number {
  if (!values.length) {
    return 0;
  }
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.floor((sorted.length - 1) * p)));
  return sorted[index];
}

function summarizeFrames(frameTimes: number[]): RendererWindowStats {
  if (!frameTimes.length) {
    return {
      sampleCount: 0,
      avgFps: 0,
      avgFrameMs: 0,
      p95FrameMs: 0,
      maxFrameMs: 0,
      longFrameRatioPct: 0,
    };
  }
  const total = frameTimes.reduce((sum, ms) => sum + ms, 0);
  const avgFrameMs = total / frameTimes.length;
  const longFrames = frameTimes.filter((ms) => ms > 33.3).length;
  return {
    sampleCount: frameTimes.length,
    avgFps: avgFrameMs > 0 ? 1000 / avgFrameMs : 0,
    avgFrameMs,
    p95FrameMs: percentile(frameTimes, 0.95),
    maxFrameMs: Math.max(...frameTimes),
    longFrameRatioPct: (longFrames / frameTimes.length) * 100,
  };
}

function createInstanceData(instanceCount: number): Float32Array {
  const data = new Float32Array(instanceCount * TRIANGLE_STRIDE_FLOATS);
  for (let i = 0; i < instanceCount; i += 1) {
    const base = i * TRIANGLE_STRIDE_FLOATS;
    const radius = 0.18 + 0.74 * Math.sqrt(((i * 9301 + 49297) % 233280) / 233280);
    const angle = (i * 2.399963229728653) % (Math.PI * 2);
    data[base + 0] = Math.cos(angle) * radius;
    data[base + 1] = Math.sin(angle) * radius;
    data[base + 2] = 0.55 + ((i % 11) / 11) * 0.85;
    data[base + 3] = (i * 0.37) % (Math.PI * 2);
    data[base + 4] = 0.30 + ((i * 17) % 100) / 160;
    data[base + 5] = 0.45 + ((i * 29) % 100) / 190;
    data[base + 6] = 0.65 + ((i * 43) % 100) / 210;
    data[base + 7] = 0.18;
  }
  return data;
}

function uploadFloat32(device: GPUDevice, buffer: GPUBuffer, data: Float32Array): void {
  const byteLength = data.byteLength;
  const upload = new ArrayBuffer(byteLength);
  new Uint8Array(upload).set(new Uint8Array(data.buffer, data.byteOffset, byteLength));
  device.queue.writeBuffer(buffer, 0, upload, 0, byteLength);
}

export class WebGpuUiStressRenderer {
  private canvas: HTMLCanvasElement;
  private device: GPUDevice | null = null;
  private context: GPUCanvasContext | null = null;
  private pipeline: GPURenderPipeline | null = null;
  private uniformBuffer: GPUBuffer | null = null;
  private instanceBuffer: GPUBuffer | null = null;
  private bindGroup: GPUBindGroup | null = null;
  private animationFrame: number | null = null;
  private frameHistory: number[] = [];
  private currentWindow: MeasurementWindow | null = null;
  private startedAt = 0;
  private lastFrameAt = 0;
  private instanceCount = 0;
  private running = false;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
  }

  async init(options: RendererInitOptions): Promise<void> {
    if (!navigator.gpu) {
      throw new Error('WebGPU is not available in this browser.');
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error('No WebGPU adapter available.');
    }
    this.device = await adapter.requestDevice();
    this.context = this.canvas.getContext('webgpu');
    if (!this.context) {
      throw new Error('Canvas webgpu context is unavailable.');
    }

    const format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({
      device: this.device,
      format,
      alphaMode: 'premultiplied',
    });

    const module = this.device.createShaderModule({ code: shaderCode });
    this.pipeline = this.device.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module,
        entryPoint: 'vsMain',
        buffers: [
          {
            arrayStride: TRIANGLE_STRIDE_FLOATS * 4,
            stepMode: 'instance',
            attributes: [
              { shaderLocation: 0, offset: 0, format: 'float32x2' },
              { shaderLocation: 1, offset: 8, format: 'float32' },
              { shaderLocation: 2, offset: 12, format: 'float32' },
              { shaderLocation: 3, offset: 16, format: 'float32x4' },
            ],
          },
        ],
      },
      fragment: {
        module,
        entryPoint: 'fsMain',
        targets: [
          {
            format,
            blend: {
              color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
              alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            },
          },
        ],
      },
      primitive: {
        topology: 'triangle-list',
      },
    });

    this.uniformBuffer = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: { buffer: this.uniformBuffer },
        },
      ],
    });

    this.setInstanceCount(options.instanceCount);
  }

  setInstanceCount(instanceCount: number): void {
    if (!this.device) {
      return;
    }
    this.instanceCount = Math.max(1, Math.floor(instanceCount));
    const data = createInstanceData(this.instanceCount);
    this.instanceBuffer?.destroy();
    this.instanceBuffer = this.device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    uploadFloat32(this.device, this.instanceBuffer, data);
  }

  start(): void {
    if (!this.device || !this.context || !this.pipeline || !this.instanceBuffer || !this.uniformBuffer || !this.bindGroup) {
      throw new Error('Renderer is not initialized.');
    }
    if (this.running) {
      return;
    }
    this.running = true;
    this.startedAt = performance.now();
    this.lastFrameAt = this.startedAt;
    const render = () => {
      if (!this.running || !this.device || !this.context || !this.pipeline || !this.instanceBuffer || !this.uniformBuffer || !this.bindGroup) {
        return;
      }
      const now = performance.now();
      const dt = Math.max(0.0001, now - this.lastFrameAt);
      this.lastFrameAt = now;
      this.frameHistory.push(dt);
      if (this.frameHistory.length > LIVE_HISTORY_LIMIT) {
        this.frameHistory.splice(0, this.frameHistory.length - LIVE_HISTORY_LIMIT);
      }
      if (this.currentWindow) {
        this.currentWindow.frameTimes.push(dt);
      }

      const time = (now - this.startedAt) / 1000;
      const aspect = Math.max(1, this.canvas.width) / Math.max(1, this.canvas.height);
      uploadFloat32(this.device, this.uniformBuffer, new Float32Array([time, aspect, 0, 0]));

      const encoder = this.device.createCommandEncoder();
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: this.context.getCurrentTexture().createView(),
            clearValue: { r: 0.035, g: 0.046, b: 0.072, a: 1.0 },
            loadOp: 'clear',
            storeOp: 'store',
          },
        ],
      });
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, this.bindGroup);
      pass.setVertexBuffer(0, this.instanceBuffer);
      pass.draw(3, this.instanceCount, 0, 0);
      pass.end();
      this.device.queue.submit([encoder.finish()]);

      this.animationFrame = requestAnimationFrame(render);
    };
    this.animationFrame = requestAnimationFrame(render);
  }

  stop(): void {
    this.running = false;
    if (this.animationFrame !== null) {
      cancelAnimationFrame(this.animationFrame);
      this.animationFrame = null;
    }
  }

  destroy(): void {
    this.stop();
    this.instanceBuffer?.destroy();
    this.uniformBuffer?.destroy();
    this.instanceBuffer = null;
    this.uniformBuffer = null;
    this.bindGroup = null;
    this.pipeline = null;
    this.context = null;
    this.device = null;
  }

  beginMeasurement(label: string): void {
    this.currentWindow = { label, frameTimes: [] };
  }

  endMeasurement(): { label: string; stats: RendererWindowStats } {
    const label = this.currentWindow?.label ?? 'unknown';
    const stats = summarizeFrames(this.currentWindow?.frameTimes ?? []);
    this.currentWindow = null;
    return { label, stats };
  }

  getLiveStats(): RendererLiveStats {
    return {
      running: this.running,
      instanceCount: this.instanceCount,
      ...summarizeFrames(this.frameHistory),
    };
  }
}
