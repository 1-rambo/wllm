import { describe, expect, it } from 'vitest';
import { webgpuRuntimeEnv } from '../../src/native/wllama-native-engine';

describe('native WebGPU runtime environment', () => {
  it('uses the baseline queue-wait ceiling and bounded submission policy', () => {
    expect(
      webgpuRuntimeEnv({
        wasmUrl: 'wllama.wasm',
        modelUrl: 'model.gguf',
        queueWaitTimeoutMs: 300_000,
        webgpuCommandSubmitBatchSize: 32,
        webgpuMaxInflightBatches: 4,
      })
    ).toEqual({
      GGML_WEBGPU_WAIT_TIMEOUT_MS: '300000',
      GGML_WEBGPU_COMMAND_SUBMIT_BATCH_SIZE: '32',
      GGML_WEBGPU_MAX_INFLIGHT_BATCHES: '4',
    });
  });
});
