import { describe, expect, it } from 'vitest';
import {
  createWllamaEnginePage,
  type WllamaPageRuntime,
} from '../../src/engines/wllama-page';

function fakeRuntime(): WllamaPageRuntime & {
  events: string[];
  loadParams: Record<string, unknown> | null;
  policies: Array<{
    commandSubmitBatchSize?: number;
    maxInflightBatches?: number;
  }>;
} {
  const events: string[] = [];
  const policies: Array<{
    commandSubmitBatchSize?: number;
    maxInflightBatches?: number;
  }> = [];
  let loadParams: Record<string, unknown> | null = null;
  let loaded = false;
  return {
    events,
    get loadParams() {
      return loadParams;
    },
    policies,
    isModelLoaded: () => loaded,
    async loadModelFromUrl(_source: string, params?: Record<string, unknown>) {
      loadParams = params === undefined ? null : { ...params };
      events.push('load');
      loaded = true;
    },
    setWebGpuPolicy(policy) {
      events.push('policy');
      policies.push(policy);
    },
    async createCompletion() {
      return (async function* () {})();
    },
  };
}

describe('wllama WebGPU submission policy', () => {
  it('applies the bounded policy after the model is loaded', async () => {
    const runtime = fakeRuntime();
    const page = createWllamaEnginePage({
      runtimeFactory: () => runtime,
      engineVersion: 'test',
      modelUrl: 'model.gguf',
      nCtx: 2048,
      nBatch: 128,
      nParallel: 1,
      temperature: 0,
      seed: 1,
      webgpuSubmissionPolicy: {
        commandSubmitBatchSize: 8,
        maxInflightBatches: 2,
      },
      scope: {},
    });

    await page.prewarm();

    expect(runtime.events).toEqual(['load', 'policy']);
    expect(runtime.loadParams).toMatchObject({
      n_parallel: 1,
      cont_batching: false,
      cache_idle_slots: false,
    });
    expect(runtime.policies).toEqual([
      {
        commandSubmitBatchSize: 8,
        maxInflightBatches: 2,
      },
    ]);
  });
});
