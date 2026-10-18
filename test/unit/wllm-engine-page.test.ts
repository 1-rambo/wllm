import { describe, expect, it } from 'vitest';
import type {
  CoordinatorDiagnostics,
  CoordinatorEvent,
} from '../../src/transport/protocol';
import type {
  LifecycleRecord,
  NativeRuntimeEvent,
  RequestRecord,
} from '../../src/contracts';
import type { EngineProgressEvent } from '../../src/engines/engine-page-api';
import type {
  GpuBurstPolicy,
  RenderPressureSignal,
} from '../../src/contracts';
import {
  createWllmEnginePage,
  estimateWllmPromptTokens,
  type WllmPageClient,
} from '../../src/engines/wllm-page';

class FakeClient implements WllmPageClient {
  readonly attached = new Map<string, LifecycleRecord>();
  readonly submitted: RequestRecord[] = [];
  readonly cancelled: string[] = [];
  readonly records = new Map<string, RequestRecord>();
  readonly pressures: RenderPressureSignal[] = [];
  lifecycleAccepted = true;
  diagnosticsValue: CoordinatorDiagnostics = diagnostics();

  async reportRenderPressure(
    signal: RenderPressureSignal
  ): Promise<GpuBurstPolicy> {
    this.pressures.push(signal);
    return {
      level: 'normal',
      commandSubmitBatchSize: 64,
      maxInflightBatches: 2,
      submissionPauseMs: 0,
    };
  }

  async attach(clientId: string, lifecycle: LifecycleRecord): Promise<void> {
    this.attached.set(clientId, lifecycle);
  }

  async detach(clientId: string): Promise<void> {
    this.attached.delete(clientId);
  }

  async setLifecycle(lifecycle: LifecycleRecord): Promise<boolean> {
    this.attached.set(lifecycle.clientId, lifecycle);
    return this.lifecycleAccepted;
  }

  async submit(
    request: Omit<RequestRecord, 'lifecycle'>,
    lifecycle?: LifecycleRecord
  ): Promise<{ accepted: boolean; reason?: string }> {
    const record = {
      ...request,
      lifecycle: lifecycle ?? {
        clientId: request.session.clientId,
        epoch: 1,
        state: 'active' as const,
        foreground: true,
        changedAt: Date.now(),
      },
    } as RequestRecord;
    this.submitted.push(record);
    this.records.set(record.requestId, record);
    return { accepted: true };
  }

  async cancel(requestId: string): Promise<boolean> {
    this.cancelled.push(requestId);
    return true;
  }

  async status(requestId: string): Promise<RequestRecord | null> {
    return this.records.get(requestId) ?? null;
  }

  async diagnostics(): Promise<CoordinatorDiagnostics> {
    return this.diagnosticsValue;
  }
}

function diagnostics(): CoordinatorDiagnostics {
  return {
    completedTier: 'L1',
    runtime: {
      nativeBound: true,
      inFlight: false,
      activeRequestId: null,
      pendingRequests: 0,
      pendingForeground: false,
      renderAware: true,
      gpuBurstPolicy: {
        level: 'normal',
        commandSubmitBatchSize: 64,
        maxInflightBatches: 2,
        submissionPauseMs: 0,
      },
      executionMode: 'deadline',
      lease: null,
      lastNativeUnit: null,
      renderGate: { enabled: true, open: true },
      execution: {
        maxPrefillTokens: 128,
        maxDecodeTokens: 64,
        foregroundDecodeTokens: 32,
        backgroundDecodeTokens: 8,
        backgroundPrefillTokens: null,
        targetUnitWallTimeMs: 30,
        adaptiveUnits: true,
        nativeTimeBudgetEnabled: true,
      },
      completedRequests: [],
      cancelledRequests: [],
      lastError: null,
    },
    recovery: { scanned: 0, recovered: 0, skipped: 0, errors: [] },
    state: {
      l1Tokens: 0,
      l2Tokens: 0,
      l3Bytes: 0,
      residentNodes: 0,
      parkedNodes: 0,
      parkedBytes: 0,
    },
    residencies: [],
    stateHistory: [],
  };
}

function runtimeReady(): NativeRuntimeEvent {
  return {
    type: 'runtime',
    phase: 'ready',
    modelUrl: 'model.gguf',
  };
}

async function waitFor<T>(read: () => T | null | undefined): Promise<T> {
  for (let index = 0; index < 100; index += 1) {
    const value = read();
    if (value !== null && value !== undefined) {
      return value;
    }
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  throw new Error('condition not reached');
}

describe('shared-coordinator engine page', () => {
  it('prewarms one shared bridge and attaches logical sessions independently', async () => {
    const client = new FakeClient();
    let ensured = 0;
    const api = createWllmEnginePage({
      client: () => client,
      bridge: () => ({
        ensure: async () => {
          ensured += 1;
        },
        release: () => undefined,
      }),
      engineVersion: 'test',
      modelUrl: 'model.gguf',
      executionMode: 'deadline',
      scope: {},
    });

    const prewarm = api.prewarm();
    api.handleRuntimeEvent(runtimeReady());
    await prewarm;
    expect(ensured).toBe(1);
    expect(api.status().ready).toBe(true);

    await api.createSession({ sessionId: 's1', site: 'shopping' });
    await api.createSession({ sessionId: 's2', site: 'reddit' });
    expect(api.status().sessions).toEqual(['s1', 's2']);
    expect(client.attached.size).toBe(2);
    expect(new Set(client.attached.keys()).size).toBe(2);
  });

  it('carries the whole-run frame tail and announced sessions in metrics', async () => {
    const client = new FakeClient();
    const frame = {
      source: 'presented-frame',
      frameCount: 240,
      p50Ms: 16.6,
      p95Ms: 24.1,
      p99Ms: 41.2,
      maxMs: 88.5,
      over33Ratio: 0.05,
      over50Ratio: 0.01,
      longestStallMs: 88.5,
      refreshHz: 60,
    };
    const announced: string[] = [];
    const api = createWllmEnginePage({
      client: () => client,
      bridge: () => ({ ensure: async () => undefined, release: () => undefined }),
      engineVersion: 'test',
      modelUrl: 'model.gguf',
      executionMode: 'deadline',
      frameMetrics: () => frame,
      onSessionCreated: (sessionId) => announced.push(sessionId),
      scope: {},
    });
    api.handleRuntimeEvent(runtimeReady());
    await api.prewarm();
    await api.createSession({ sessionId: 's1', site: 'shopping' });

    expect(announced).toEqual(['s1']);
    // The experiment record needs the tail from the whole block, so the page
    // hands over the monitor snapshot rather than a rolling controller view.
    expect((await api.metrics()).frame).toEqual(frame);
  });

  it('forwards an injected render sample to the high-priority session', async () => {
    const client = new FakeClient();
    const api = createWllmEnginePage({
      client: () => client,
      bridge: () => ({ ensure: async () => undefined, release: () => undefined }),
      engineVersion: 'test',
      modelUrl: 'model.gguf',
      executionMode: 'deadline',
      scope: {},
    });
    api.handleRuntimeEvent(runtimeReady());
    await api.prewarm();
    await api.createSession({ sessionId: 'background', site: 'shopping' });
    await api.createSession({ sessionId: 'foreground', site: 'reddit' });
    await api.setSessionPriority('background', 'normal');

    await api.reportRenderPressure({
      source: 'presented-frame',
      frameCount: 42,
      p50Ms: 22,
      p95Ms: 51,
      p99Ms: 80,
      maxMs: 120,
      overBudgetRatio: 0.8,
      over33Ratio: 0.7,
      over50Ratio: 0.4,
      longestStallMs: 120,
      refreshHz: 60,
      pageOwned: true,
    });

    expect(client.pressures).toHaveLength(1);
    expect(client.pressures[0]).toMatchObject({
      clientId: 'wllm-page:foreground',
      epoch: 1,
      p95Ms: 51,
      pageOwned: true,
    });
  });

  it('advances priority epochs and reports coordinator acceptance', async () => {
    const client = new FakeClient();
    const api = createWllmEnginePage({
      client: () => client,
      bridge: () => ({ ensure: async () => undefined, release: () => undefined }),
      engineVersion: 'test',
      modelUrl: 'model.gguf',
      executionMode: 'deadline',
      scope: {},
    });
    api.handleRuntimeEvent(runtimeReady());
    await api.prewarm();
    await api.createSession({ sessionId: 's1', site: 'shopping' });

    expect(await api.setSessionPriority('s1', 'normal')).toMatchObject({
      accepted: true,
      epoch: 2,
    });
    expect(client.attached.get('wllm-page:s1')).toMatchObject({
      epoch: 2,
      foreground: false,
    });
    expect(await api.setSessionPriority('s1', 'high')).toMatchObject({
      accepted: true,
      epoch: 3,
    });

    client.lifecycleAccepted = false;
    expect(await api.setSessionPriority('s1', 'normal')).toMatchObject({
      accepted: false,
      epoch: 4,
    });
  });

  it('exposes session priority as an explicit high/normal contract', async () => {
    const client = new FakeClient();
    const api = createWllmEnginePage({
      client: () => client,
      bridge: () => ({ ensure: async () => undefined, release: () => undefined }),
      engineVersion: 'test',
      modelUrl: 'model.gguf',
      executionMode: 'deadline',
      scope: {},
    });
    api.handleRuntimeEvent(runtimeReady());
    await api.prewarm();
    await api.createSession({ sessionId: 's1', site: 'shopping' });

    expect(client.attached.get('wllm-page:s1')).toMatchObject({
      epoch: 1,
      foreground: true,
    });
    expect(await api.setSessionPriority('s1', 'normal')).toMatchObject({
      accepted: true,
      epoch: 2,
      level: 'normal',
    });
    expect(client.attached.get('wllm-page:s1')).toMatchObject({
      epoch: 2,
      foreground: false,
    });
    expect(await api.setSessionPriority('s1', 'normal')).toMatchObject({
      accepted: true,
      epoch: 2,
      level: 'normal',
    });
    expect(await api.setSessionPriority('s1', 'high')).toMatchObject({
      accepted: true,
      epoch: 3,
    });

    expect(api.priorityLog()).toHaveLength(3);
    expect(api.priorityLog().map((entry) => entry.level)).toEqual([
      'normal',
      'normal',
      'high',
    ]);
    expect(api.priorityLog().every((entry) => entry.sessionId === 's1')).toBe(true);
  });

  it('reports the lease that shaped the block so an artifact can name its reason', async () => {
    const client = new FakeClient();
    const lease = {
      source: 'deadline' as const,
      reason: 'frame-deadline' as const,
      leaseId: 17,
      budgetMs: 16.7,
      maxDecodeTokens: 4,
      maxPrefillTokens: 4,
      queue: {
        commandSubmitBatchSize: 32,
        maxInflightBatches: 1,
        basis: 'local-barrier' as const,
      },
      frameBudgetMs: 16.7,
      frameDebtMs: 8.2,
      refreshHz: 60,
      signalAgeMs: 12,
      occupancyOvershootMs: null,
      queueWaitMs: null,
      lastUnit: null,
      recoveries: 0,
      lastRecoveryAt: null,
      lastRecoveryReason: null,
    };
    client.diagnosticsValue.runtime.lease = lease;
    const api = createWllmEnginePage({
      client: () => client,
      bridge: () => ({ ensure: async () => undefined, release: () => undefined }),
      engineVersion: 'test',
      modelUrl: 'model.gguf',
      executionMode: 'deadline',
      scope: {},
    });
    api.handleRuntimeEvent(runtimeReady());
    await api.prewarm();

    expect((await api.metrics()).lease).toEqual(lease);
  });

  it('uses the first streamed token chunk for TTFT', async () => {
    let clock = 100;
    const client = new FakeClient();
    const progress: EngineProgressEvent[] = [];
    const api = createWllmEnginePage({
      client: () => client,
      bridge: () => ({ ensure: async () => undefined, release: () => undefined }),
      engineVersion: 'test',
      modelUrl: 'model.gguf',
      executionMode: 'deadline',
      scope: {},
      now: () => clock,
    });
    api.subscribeProgress?.((event) => progress.push(event));
    api.handleRuntimeEvent(runtimeReady());
    await api.prewarm();
    await api.createSession({ sessionId: 's1', site: 'shopping' });

    const pending = api.generate('s1', {
      requestId: 'request-stream',
      promptText: 'find the order',
      maxNewTokens: 2,
    });
    const request = await waitFor(() => client.submitted[0]);

    clock = 135;
    api.handleEvent({
      type: 'token',
      clientId: request.session.clientId,
      sessionId: request.session.sessionId,
      requestId: request.requestId,
      text: 'hel',
      decodeGenerated: 1,
      elapsedMs: 20,
    });
    clock = 145;
    api.handleEvent({
      type: 'token',
      clientId: request.session.clientId,
      sessionId: request.session.sessionId,
      requestId: request.requestId,
      text: 'lo',
      decodeGenerated: 2,
      elapsedMs: 30,
    });
    clock = 200;
    api.handleEvent(unitEvent(request.requestId, 2, 2, ['hello']));
    api.handleEvent({
      type: 'complete',
      clientId: request.session.clientId,
      sessionId: request.session.sessionId,
      requestId: request.requestId,
      output: 'hello',
    });

    const generation = await pending;
    expect(generation.ttftMs).toBe(35);
    expect(generation.decodeMs).toBe(10);
    expect(generation.firstChunkTokens).toBe(1);
    expect(generation.text).toBe('hello');
    expect(progress.map((event) => event.type)).toEqual([
      'token',
      'token',
      'complete',
    ]);
    expect(progress.at(-1)).toMatchObject({
      sessionId: 's1',
      requestId: 'request-stream',
      decodeGenerated: 2,
    });
    expect(progress.filter((event) => event.type === 'token').map((event) => event.text)).toEqual([
      'hel',
      'lo',
    ]);
  });

  it('turns coordinator units into one generation without hiding cancellation', async () => {
    const client = new FakeClient();
    const api = createWllmEnginePage({
      client: () => client,
      bridge: () => ({ ensure: async () => undefined, release: () => undefined }),
      engineVersion: 'test',
      modelUrl: 'model.gguf',
      executionMode: 'deadline',
      scope: {},
    });
    api.handleRuntimeEvent(runtimeReady());
    await api.prewarm();
    await api.createSession({ sessionId: 's1', site: 'shopping' });

    const pending = api.generate('s1', {
      requestId: 'request-1',
      promptText: 'find the order',
      maxNewTokens: 8,
    });
    const request = await waitFor(() => client.submitted[0]);
    expect(estimateWllmPromptTokens('find the order')).toHaveLength(3);

    api.handleEvent(unitEvent(request.requestId, 1, 4, ['hello ']));
    api.handleEvent(unitEvent(request.requestId, 8, 8, ['world']));
    api.handleEvent({
      type: 'complete',
      clientId: request.session.clientId,
      sessionId: request.session.sessionId,
      requestId: request.requestId,
      output: 'hello world',
    });
    const generation = await pending;
    expect(generation.text).toBe('hello world');
    expect(generation.decodeTokens).toBe(8);
    expect(generation.units).toBe(2);
    expect(generation.stopReason).toBe('length');
    expect(api.status().activeRequests).toEqual([]);

    const second = api.generate('s1', {
      requestId: 'request-2',
      promptText: 'another',
      maxNewTokens: 4,
    });
    await waitFor(() => client.submitted[1]);
    await api.cancel('s1');
    const secondRequest = client.submitted[1]!;
    api.handleEvent({
      type: 'fenced',
      clientId: secondRequest.session.clientId,
      sessionId: secondRequest.session.sessionId,
      requestId: secondRequest.requestId,
    });
    const cancelled = await second;
    expect(cancelled.stopReason).toBe('cancelled');
    expect(client.cancelled).toContain(secondRequest.requestId);
  });

  it('rejects a second request on the same logical session', async () => {
    const client = new FakeClient();
    const api = createWllmEnginePage({
      client: () => client,
      bridge: () => ({ ensure: async () => undefined, release: () => undefined }),
      engineVersion: 'test',
      modelUrl: 'model.gguf',
      executionMode: 'deadline',
      scope: {},
    });
    api.handleRuntimeEvent(runtimeReady());
    await api.prewarm();
    await api.createSession({ sessionId: 's1', site: 'shopping' });
    const first = api.generate('s1', {
      requestId: 'first',
      promptText: 'one',
      maxNewTokens: 2,
    });
    await waitFor(() => client.submitted[0]);
    await expect(
      api.generate('s1', {
        requestId: 'second',
        promptText: 'two',
        maxNewTokens: 2,
      })
    ).rejects.toThrow(/already has a running request/);
    const request = client.submitted[0]!;
    api.handleEvent({
      type: 'complete',
      clientId: request.session.clientId,
      sessionId: request.session.sessionId,
      requestId: request.requestId,
      output: '',
    });
    await first;
  });
});

function unitEvent(
  requestId: string,
  decoded: number,
  prefill: number,
  outputChunks: string[]
): CoordinatorEvent {
  return {
    type: 'unit',
    clientId: 'client',
    sessionId: 'session',
    requestId,
    progress: {
      requestId,
      stage: 'decode',
      prefillProcessed: prefill,
      prefillTotal: prefill,
      decodeGenerated: decoded,
      decodeTotal: 8,
      done: false,
      cancelled: false,
      isError: false,
      outputChunks,
      elapsedMs: 10,
      slotId: 1,
      tokenAccounting: 'native-token',
    },
    budget: {
      maxPrefillTokens: 128,
      maxDecodeTokens: 8,
      wallTimeMs: 100,
    },
  };
}
