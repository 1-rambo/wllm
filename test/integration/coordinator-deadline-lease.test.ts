import { describe, expect, it } from 'vitest';
import { SharedWorkerCoordinator } from '../../src/coordinator/shared-worker-coordinator';
import { FakeNativeEngine } from '../../src/native/fake-native-engine';
import { MemoryBlobStore } from '../../src/state/checkpoint-store';
import { lifecycle, request } from '../helpers/factories';
import type {
  NativeUnitBudget,
  RenderPressureSignal,
  RequestRecord,
  StateManagerConfig,
} from '../../src/contracts';
import type { FrameDeadlineLeaseOptions } from '../../src/execution/frame-deadline-lease';

function frame(
  overrides: Partial<RenderPressureSignal> = {}
): RenderPressureSignal {
  return {
    clientId: 'c1',
    epoch: 1,
    sampledAt: 0,
    source: 'presented-frame',
    frameCount: 12,
    p50Ms: 16,
    p95Ms: 18,
    p99Ms: 20,
    maxMs: 24,
    overBudgetRatio: 0.1,
    over33Ratio: 0,
    over50Ratio: 0,
    longestStallMs: 0,
    refreshHz: 60,
    ...overrides,
  };
}

const pressured = (): RenderPressureSignal =>
  frame({ p95Ms: 44, p99Ms: 100, over33Ratio: 0.5, longestStallMs: 120 });

function deadlineCoordinator(
  native: FakeNativeEngine,
  overrides: {
    renderPause?: (pauseMs: number) => Promise<void>;
    lease?: Partial<FrameDeadlineLeaseOptions>;
    leaseBatchSize?: number;
    batchHandoff?: boolean;
    handoffBatchSize?: number;
    handoffPrefillTokens?: number;
    preemptOnActivation?: boolean;
    state?: Partial<StateManagerConfig>;
  } = {}
) {
  let now = 1_000;
  const units: NativeUnitBudget[] = [];
  const coordinator = new SharedWorkerCoordinator(
    native,
    new MemoryBlobStore(),
    {
      executionMode: 'deadline',
      scheduler: { graceMs: 0 },
      clock: { now: () => now },
      macrotaskYield: async () => undefined,
      ...(overrides.renderPause ? { renderPause: overrides.renderPause } : {}),
      ...(overrides.lease ? { lease: overrides.lease } : {}),
      ...(overrides.leaseBatchSize === undefined
        ? {}
        : { leaseBatchSize: overrides.leaseBatchSize }),
      ...(overrides.batchHandoff === undefined
        ? {}
        : { batchHandoff: overrides.batchHandoff }),
      ...(overrides.handoffBatchSize === undefined
        ? {}
        : { handoffBatchSize: overrides.handoffBatchSize }),
      ...(overrides.handoffPrefillTokens === undefined
        ? {}
        : { handoffPrefillTokens: overrides.handoffPrefillTokens }),
      ...(overrides.preemptOnActivation === undefined
        ? {}
        : { preemptOnActivation: overrides.preemptOnActivation }),
      ...(overrides.state ? { state: overrides.state } : {}),
      execution: {
        maxUnitWallTimeMs: 400,
        unitWallTimeGraceMs: 400,
        nativeTimeBudgetEnabled: true,
      },
    }
  );
  coordinator.subscribe((event) => {
    if (event.type === 'unit') {
      units.push(event.budget);
    }
  });
  return {
    coordinator,
    units,
    advance: (ms: number) => {
      now += ms;
    },
  };
}

// A request that has already completed prompt processing. Prompt work is
// admitted one request at a time; decode is the stage that shares a graph.
function decodeRequest(
  requestId: string,
  options: Partial<Omit<RequestRecord, 'requestId'>> = {}
): RequestRecord {
  return request(requestId, {
    stage: 'decode',
    prefillProcessed: 4,
    maxNewTokens: 200,
    ...options,
  });
}

describe('SharedWorkerCoordinator deadline lease', () => {
  it('shapes units with a lease and reports it in diagnostics', async () => {
    const native = new FakeNativeEngine();
    const { coordinator, units, advance } = deadlineCoordinator(native);
    await coordinator.attach('c1', lifecycle('c1'));

    // two clear samples put the lease in its work-conserving shape
    await coordinator.updateRenderPressure(frame());
    advance(500);
    await coordinator.updateRenderPressure(frame());
    advance(100);

    await coordinator.submit(
      request('c1:clear', { promptTokens: [1, 2, 3, 4], maxNewTokens: 8 })
    );
    await coordinator.pump(1);

    expect(units).toHaveLength(1);
    expect(units[0]?.leaseId).toBeGreaterThan(0);
    expect(units[0]?.leaseSource).toBe('deadline');
    // Leased units stop at the native graph/token boundary; the frame
    // budget is carried by the lease, not by a truncating wall-clock timer.
    expect(units[0]?.nativeTimeBudgetMs).toBeUndefined();
    expect(units[0]?.wallTimeMs).toBeGreaterThan(400);

    const diagnostics = await coordinator.diagnostics();
    expect(diagnostics.runtime.executionMode).toBe('deadline');
    expect(diagnostics.runtime.gpuBurstPolicy.source).toBe('deadline');
    expect(diagnostics.runtime.lease?.reason).toBe('work-conserving');
    expect(diagnostics.runtime.lease?.budgetMs).toBe(400);
    expect(native.gpuPolicies.at(-1)?.source).toBe('deadline');
  });

  it('admits same-stage sessions into one deadline graph boundary', async () => {
    const native = new FakeNativeEngine();
    const { coordinator, units, advance } = deadlineCoordinator(native, {
      leaseBatchSize: 2,
    });
    await coordinator.attach(
      'c1',
      lifecycle('c1', { foreground: false, state: 'hidden' })
    );
    await coordinator.attach(
      'c2',
      lifecycle('c2', { foreground: false, state: 'hidden' })
    );
    await coordinator.updateRenderPressure(frame());
    advance(500);
    await coordinator.updateRenderPressure(frame());
    advance(100);

    await coordinator.submit(
      decodeRequest('c1:batch', { promptTokens: [1, 2, 3, 4], maxNewTokens: 1 })
    );
    await coordinator.submit(
      decodeRequest('c2:batch', { promptTokens: [5, 6, 7, 8], maxNewTokens: 1 })
    );
    await coordinator.pump(1);

    // Both requests are shaped by one lease. The scheduler removes both
    // heads together, so the second request cannot be stranded behind a
    // graph boundary the first request happens to finish first.
    expect(units).toHaveLength(2);
    expect(new Set(units.map((unit) => unit.leaseId)).size).toBe(1);
    expect(units.every((unit) => unit.maxDecodeTokens >= 1)).toBe(true);
  });

  it('keeps ragged prompt processing single-request', async () => {
    const native = new FakeNativeEngine();
    const { coordinator, units } = deadlineCoordinator(native, {
      leaseBatchSize: 3,
    });
    for (const clientId of ['c1', 'c2', 'c3']) {
      await coordinator.attach(
        clientId,
        lifecycle(clientId, { foreground: false, state: 'hidden' })
      );
      await coordinator.submit(
        request(`${clientId}:prefill`, { maxNewTokens: 64 })
      );
    }

    await coordinator.pump(1);
    expect(units).toHaveLength(1);
    await coordinator.pump(1);
    expect(units).toHaveLength(2);
    await coordinator.pump(1);
    expect(units).toHaveLength(3);
  });

  it('lets a pending handoff share one graph with a background peer', async () => {
    const native = new FakeNativeEngine();
    const { coordinator, units } = deadlineCoordinator(native, {
      leaseBatchSize: 4,
      batchHandoff: true,
      handoffBatchSize: 2,
      handoffPrefillTokens: 64,
    });
    await coordinator.attach(
      'c1',
      lifecycle('c1', { foreground: false, state: 'hidden' })
    );
    await coordinator.attach(
      'c2',
      lifecycle('c2', { foreground: false, state: 'hidden' })
    );
    await coordinator.submit(
      request('c1:handoff', { maxNewTokens: 2, waitBudgetMs: 1_000 })
    );
    await coordinator.submit(
      request('c2:peer', { maxNewTokens: 2, waitBudgetMs: 1_000 })
    );
    await coordinator.updateLifecycle(
      lifecycle('c1', { epoch: 2, foreground: true, state: 'active' })
    );

    await coordinator.pump(1);

    expect(units).toHaveLength(2);
    expect(new Set(units.map((unit) => unit.leaseId)).size).toBe(1);
  });

  it('keeps an answered demoted promotion on the priority lease', async () => {
    const native = new FakeNativeEngine();
    const { coordinator, units, advance } = deadlineCoordinator(native, {
      leaseBatchSize: 1,
      preemptOnActivation: true,
    });
    await coordinator.attach(
      'c1',
      lifecycle('c1', { foreground: false, state: 'hidden' })
    );
    await coordinator.attach('c2', lifecycle('c2'));
    await coordinator.submit(
      request('c1:priority', { maxNewTokens: 4, waitBudgetMs: 1_000 })
    );
    await coordinator.updateLifecycle(
      lifecycle('c1', { epoch: 2, foreground: true, state: 'active' })
    );
    await coordinator.updateRenderPressure(
      { ...pressured(), epoch: 2 }
    );
    advance(500);
    await coordinator.updateRenderPressure(
      { ...pressured(), epoch: 2 }
    );

    await coordinator.pump(1);
    expect(units[0]?.leaseSource).toBe('deadline');
    expect(units[0]?.maxDecodeTokens).toBeGreaterThanOrEqual(1);

    await coordinator.updateLifecycle(
      lifecycle('c1', { epoch: 3, foreground: false, state: 'hidden' })
    );
    await coordinator.updateRenderPressure(
      { ...pressured(), clientId: 'c2', epoch: 1 }
    );
    await coordinator.pump(1);

    expect(units[1]?.leaseSource).toBe('deadline');
    expect(units[1]?.maxDecodeTokens).toBeGreaterThan(1);
  });

  it('requeues a batch member that cannot reserve a native slot', async () => {
    const native = new FakeNativeEngine({ slotCount: 1 });
    const { coordinator } = deadlineCoordinator(native, {
      leaseBatchSize: 2,
      state: { maxResidentNodes: 1, l1TokenCap: 4, l2TokenCap: 64 },
    });
    const failures: string[] = [];
    coordinator.subscribe((event) => {
      if (event.type === 'failed') {
        failures.push(event.error);
      }
    });
    for (const clientId of ['c1', 'c2']) {
      await coordinator.attach(
        clientId,
        lifecycle(clientId, { foreground: false, state: 'hidden' })
      );
      await coordinator.submit(
        request(`${clientId}:capacity`, {
          promptTokens: [1, 2, 3, 4],
          maxNewTokens: 1,
        })
      );
    }

    const steps = await coordinator.pump(20);

    expect(steps).toBeGreaterThan(0);
    expect(failures).toEqual([]);
    expect(await coordinator.status('c1:capacity')).toBeNull();
    expect(await coordinator.status('c2:capacity')).toBeNull();
  });

  it('cuts a running batch when a member tab is activated', async () => {
    const native = new FakeNativeEngine();
    const interrupts: string[] = [];
    const baseInterrupt = native.interrupt.bind(native);
    native.interrupt = async (nativeRequestId: string) => {
      interrupts.push(nativeRequestId);
      await baseInterrupt(nativeRequestId);
    };
    // Hold one member of the batch so the group really is in flight when
    // the click arrives, the way a real multi-tab decode graph is.
    let releaseAdvance: () => void = () => undefined;
    const inFlight = new Promise<void>((resolve) => {
      releaseAdvance = resolve;
    });
    let holdFirstAdvance = true;
    const baseAdvance = native.advance.bind(native);
    native.advance = async (nativeRequestId, budget) => {
      if (holdFirstAdvance) {
        holdFirstAdvance = false;
        await inFlight;
      }
      return baseAdvance(nativeRequestId, budget);
    };

    const { coordinator } = deadlineCoordinator(native, {
      leaseBatchSize: 3,
      preemptOnActivation: true,
    });
    for (const clientId of ['c1', 'c2', 'c3']) {
      await coordinator.attach(
        clientId,
        lifecycle(clientId, { foreground: false, state: 'hidden' })
      );
      await coordinator.submit(
        decodeRequest(`${clientId}:batch`)
      );
    }

    const pumping = coordinator.pump(1);
    for (let attempt = 0; attempt < 20 && native.advanceCalls === 0; ) {
      attempt += 1;
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
    expect(native.advanceCalls).toBeGreaterThan(0);
    expect(interrupts).toEqual([]);

    // The human clicks the tab whose own request is inside the running
    // batch. The ordinary peers are cut so the promoted request does not
    // wait for the slowest peer, while the promoted request itself keeps its
    // in-flight stream.
    await coordinator.updateLifecycle(
      lifecycle('c2', { epoch: 2, foreground: true, state: 'active' })
    );

    expect(new Set(interrupts)).toEqual(new Set(['c1:batch', 'c3:batch']));
    expect(interrupts).not.toContain('c2:batch');

    releaseAdvance();
    await pumping;
  });

  it('retries activation preemption after a running batch binds native ids', async () => {
    const native = new FakeNativeEngine();
    const interrupts: string[] = [];
    const baseInterrupt = native.interrupt.bind(native);
    native.interrupt = async (nativeRequestId: string) => {
      interrupts.push(nativeRequestId);
      await baseInterrupt(nativeRequestId);
    };

    let releaseSubmissions: () => void = () => undefined;
    const submissionsReleased = new Promise<void>((resolve) => {
      releaseSubmissions = resolve;
    });
    let signalFirstSubmission: () => void = () => undefined;
    const firstSubmissionEntered = new Promise<void>((resolve) => {
      signalFirstSubmission = resolve;
    });
    let holdSubmissions = true;
    const baseSubmit = native.submit.bind(native);
    native.submit = async (submitted) => {
      if (holdSubmissions) {
        signalFirstSubmission();
        await submissionsReleased;
        holdSubmissions = false;
      }
      return baseSubmit(submitted);
    };

    const { coordinator } = deadlineCoordinator(native, {
      leaseBatchSize: 3,
      preemptOnActivation: true,
    });
    for (const clientId of ['c1', 'c2', 'c3']) {
      await coordinator.attach(
        clientId,
        lifecycle(clientId, { foreground: false, state: 'hidden' })
      );
      await coordinator.submit(
        decodeRequest(`${clientId}:batch`)
      );
    }

    const pumping = coordinator.pump(1);
    await firstSubmissionEntered;

    await coordinator.attach(
      'c4',
      lifecycle('c4', { foreground: false, state: 'hidden' })
    );
    await coordinator.updateLifecycle(
      lifecycle('c4', { epoch: 2, foreground: true, state: 'active' })
    );
    await coordinator.submit(
      request('c4:late-promotion', {
        maxNewTokens: 200,
        lifecycle: lifecycle('c4', {
          epoch: 2,
          foreground: true,
          state: 'active',
        }),
      })
    );
    expect(interrupts).toEqual([]);

    releaseSubmissions();
    await pumping;

    expect(new Set(interrupts)).toEqual(
      new Set(['c1:batch', 'c2:batch', 'c3:batch'])
    );
    expect(interrupts).toHaveLength(3);
  });

  it('retains activation preemption until a later batch binds native ids', async () => {
    const native = new FakeNativeEngine();
    const interrupts: string[] = [];
    const baseInterrupt = native.interrupt.bind(native);
    native.interrupt = async (nativeRequestId: string) => {
      interrupts.push(nativeRequestId);
      await baseInterrupt(nativeRequestId);
    };

    const { coordinator } = deadlineCoordinator(native, {
      leaseBatchSize: 3,
      preemptOnActivation: true,
    });
    for (const clientId of ['c1', 'c2', 'c3']) {
      await coordinator.attach(
        clientId,
        lifecycle(clientId, { foreground: false, state: 'hidden' })
      );
      await coordinator.submit(
        decodeRequest(`${clientId}:batch`)
      );
    }
    await coordinator.attach(
      'c4',
      lifecycle('c4', { foreground: false, state: 'hidden' })
    );

    // The promotion arrives before c4 has submitted a request, so no unit
    // can be cut yet. The debt must survive until the ordinary batch starts.
    await coordinator.updateLifecycle(
      lifecycle('c4', { epoch: 2, foreground: true, state: 'active' })
    );
    expect(interrupts).toEqual([]);

    await coordinator.pump(1);
    expect(new Set(interrupts)).toEqual(
      new Set(['c1:batch', 'c2:batch', 'c3:batch'])
    );
    expect(interrupts).toHaveLength(3);
  });

  it('consumes one activation wave instead of re-cutting later batches', async () => {
    const native = new FakeNativeEngine();
    const interrupts: string[] = [];
    const baseInterrupt = native.interrupt.bind(native);
    native.interrupt = async (nativeRequestId: string) => {
      interrupts.push(nativeRequestId);
      await baseInterrupt(nativeRequestId);
    };

    const { coordinator } = deadlineCoordinator(native, {
      leaseBatchSize: 3,
      preemptOnActivation: true,
    });
    for (const clientId of ['c1', 'c2', 'c3']) {
      await coordinator.attach(
        clientId,
        lifecycle(clientId, { foreground: false, state: 'hidden' })
      );
      await coordinator.submit(
        decodeRequest(`${clientId}:batch`)
      );
    }
    await coordinator.attach(
      'c4',
      lifecycle('c4', { foreground: false, state: 'hidden' })
    );
    await coordinator.updateLifecycle(
      lifecycle('c4', { epoch: 2, foreground: true, state: 'active' })
    );

    // The first ordinary batch is cut, then goes idle before c4 has submitted
    // a request.
    await coordinator.pump(1);
    expect(new Set(interrupts)).toEqual(
      new Set(['c1:batch', 'c2:batch', 'c3:batch'])
    );
    expect(interrupts).toHaveLength(3);

    // A later ordinary batch must not be interrupted again just because the
    // promotion has not produced a token yet. Otherwise concurrent
    // promotions keep creating zero-token host-abort units and starve the
    // very request they are trying to accelerate.
    await coordinator.pump(1);
    expect(new Set(interrupts)).toEqual(
      new Set(['c1:batch', 'c2:batch', 'c3:batch'])
    );
    expect(interrupts).toHaveLength(3);

    await coordinator.submit(
      request('c4:late-promotion', {
        maxNewTokens: 200,
        lifecycle: lifecycle('c4', {
          epoch: 2,
          foreground: true,
          state: 'active',
        }),
      })
    );
    await coordinator.pump(1);
    expect(interrupts).toHaveLength(3);
  });

  it('cuts a running batch when a promoted session has not submitted yet', async () => {
    const native = new FakeNativeEngine();
    const interrupts: string[] = [];
    const baseInterrupt = native.interrupt.bind(native);
    native.interrupt = async (nativeRequestId: string) => {
      interrupts.push(nativeRequestId);
      await baseInterrupt(nativeRequestId);
    };
    let releaseAdvance: () => void = () => undefined;
    const inFlight = new Promise<void>((resolve) => {
      releaseAdvance = resolve;
    });
    let holdFirstAdvance = true;
    const baseAdvance = native.advance.bind(native);
    native.advance = async (nativeRequestId, budget) => {
      if (holdFirstAdvance) {
        holdFirstAdvance = false;
        await inFlight;
      }
      return baseAdvance(nativeRequestId, budget);
    };

    const { coordinator } = deadlineCoordinator(native, {
      leaseBatchSize: 3,
      preemptOnActivation: true,
    });
    for (const clientId of ['c1', 'c2', 'c3']) {
      await coordinator.attach(
        clientId,
        lifecycle(clientId, { foreground: false, state: 'hidden' })
      );
      await coordinator.submit(
        decodeRequest(`${clientId}:batch`)
      );
    }
    await coordinator.attach(
      'c4',
      lifecycle('c4', { foreground: false, state: 'hidden' })
    );
    await coordinator.updateLifecycle(
      lifecycle('c4', { epoch: 2, foreground: true, state: 'active' })
    );

    const pumping = coordinator.pump(1);
    for (let attempt = 0; attempt < 20 && native.advanceCalls === 0; ) {
      attempt += 1;
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
    expect(native.advanceCalls).toBeGreaterThan(0);
    expect(new Set(interrupts)).toEqual(
      new Set(['c1:batch', 'c2:batch', 'c3:batch'])
    );
    expect(interrupts).toHaveLength(3);

    // The promotion edge happened before submit. The ordinary batch that
    // started in the meantime is cut at its first native boundary so the
    // promoted session can run next.
    await coordinator.submit(
      request('c4:late-promotion', {
        maxNewTokens: 200,
        lifecycle: lifecycle('c4', {
          epoch: 2,
          foreground: true,
          state: 'active'
        })
      })
    );

    expect(new Set(interrupts)).toEqual(
      new Set(['c1:batch', 'c2:batch', 'c3:batch'])
    );
    expect(interrupts).toHaveLength(3);

    releaseAdvance();
    await pumping;
  });

  it('applies a mid-unit pressure signal at the next graph boundary', async () => {
    const native = new FakeNativeEngine();
    const { coordinator, units, advance } = deadlineCoordinator(native);
    await coordinator.attach('c1', lifecycle('c1'));
    await coordinator.updateRenderPressure(frame());
    advance(500);
    await coordinator.updateRenderPressure(frame());
    advance(100);
    await coordinator.submit(
      // long enough that one work-conserving unit cannot finish it
      request('c1:r1', { promptTokens: [1, 2, 3, 4], maxNewTokens: 200 })
    );
    await coordinator.pump(1);

    // pressure arrives while the first unit is already recorded
    await coordinator.updateRenderPressure(pressured());
    advance(500);
    await coordinator.updateRenderPressure(pressured());
    advance(100);
    await coordinator.pump(1);

    expect(units).toHaveLength(2);
    expect(units[0]?.nativeTimeBudgetMs).toBeUndefined();
    expect(units[1]?.leaseId).not.toBe(units[0]?.leaseId);
    expect(units[1]?.maxDecodeTokens ?? 0).toBeGreaterThanOrEqual(1);
    expect(units[1]?.wallTimeMs ?? 0).toBeGreaterThan(400);
  });

  it('parks an unfinished sequence between units instead of starving later turns', async () => {
    const native = new FakeNativeEngine({ slotCount: 2 });
    const { coordinator, advance } = deadlineCoordinator(native, {
      state: { l1TokenCap: 8 },
    });
    await coordinator.attach('c1', lifecycle('c1'));
    await coordinator.attach(
      'c2',
      lifecycle('c2', { foreground: false, state: 'hidden' })
    );
    await coordinator.submit(
      request('c1:first', { promptTokens: [1, 2, 3, 4], maxNewTokens: 200 })
    );
    await coordinator.submit(
      request('c2:second', { promptTokens: [5, 6, 7, 8], maxNewTokens: 200 })
    );

    await coordinator.pump(1);
    // Make the second session overdue so the scheduler's shortest-work
    // preference cannot run the first request to completion in the test.
    advance(500);
    await coordinator.pump(2);

    // A sequence waiting between graph boundaries can be parked safely. If
    // its native id were treated as permanently pinned, two long requests
    // could hold every slot while later sessions starve.
    expect(native.parkedStateCount).toBeGreaterThan(0);
    expect((await coordinator.status('c1:first'))?.nativeRequestId).toBeNull();
    expect((await coordinator.status('c2:second'))?.nativeRequestId).not.toBeNull();

    await coordinator.pump(100);
    expect(
      (await coordinator.status('c2:second'))?.decodeGenerated ?? 0
    ).toBeGreaterThan(0);
  });

  it('isolates a priority graph by parking idle ordinary sequences', async () => {
    const native = new FakeNativeEngine({ slotCount: 4 });
    const { coordinator } = deadlineCoordinator(native, {
      leaseBatchSize: 1,
      state: { l1TokenCap: 1_024, maxResidentNodes: 4 },
    });
    await coordinator.attach(
      'c1',
      lifecycle('c1', { foreground: false, state: 'hidden' })
    );
    await coordinator.submit(
      request('c1:background', {
        promptTokens: [1, 2, 3, 4],
        maxNewTokens: 200,
      })
    );
    await coordinator.pump(1);

    expect(native.parkedStateCount).toBe(0);
    expect(await coordinator.status('c1:background')).not.toBeNull();
    expect(coordinator.stateManager.getResidency('c1:main')?.level).toBe('L1');
    const decodeBeforeIsolation =
      (await coordinator.status('c1:background'))?.decodeGenerated ?? 0;

    await coordinator.attach('c2', lifecycle('c2'));
    await coordinator.submit(
      request('c2:priority', {
        promptTokens: [5, 6, 7, 8],
        maxNewTokens: 2,
      })
    );
    // The barriered stale-signal path parks the ordinary graph on the first
    // turn and admits the parked native slot on the next graph boundary.
    await coordinator.pump(1);

    // The ordinary sequence keeps its generated tokens and continuation, but
    // gives up its live native slot before the priority graph starts.
    expect(native.parkedStateCount).toBeGreaterThan(0);
    expect(coordinator.stateManager.getResidency('c1:main')?.level).toBe('L2');
    expect((await coordinator.status('c1:background'))?.nativeRequestId).toBeNull();
    // The barriered priority graph runs to its two-token ceiling in that
    // same admission, so completion is stronger than observing one token.
    expect((await coordinator.diagnostics()).runtime.completedRequests).toContain(
      'c2:priority'
    );

    await coordinator.pump(100);
    const resumed = await coordinator.status('c1:background');
    expect(resumed?.nativeRequestId).not.toBeNull();
    expect(resumed?.decodeGenerated ?? 0).toBeGreaterThan(decodeBeforeIsolation);
    expect((await coordinator.diagnostics()).runtime.lastError).toBeNull();
  });

  it('falls back to the bounded safety shape while the signal is stale', async () => {
    const native = new FakeNativeEngine();
    const { coordinator, units } = deadlineCoordinator(native);
    await coordinator.attach('c1', lifecycle('c1'));
    await coordinator.submit(
      request('c1:stale', { promptTokens: [1, 2, 3, 4], maxNewTokens: 4 })
    );

    const steps = await coordinator.pump(10);

    expect(steps).toBeGreaterThan(0);
    expect(units[0]?.leaseSource).toBe('fallback');
    expect(units[0]?.nativeTimeBudgetMs).toBeUndefined();
    const diagnostics = await coordinator.diagnostics();
    expect(diagnostics.runtime.lease?.reason).toBe('stale-signal');
    // the fallback still finishes the request: no gate-style stall
    expect(await coordinator.status('c1:stale')).toBeNull();
    expect(native.advanceCalls).toBeGreaterThan(0);
  });

  it('sidelines background sessions without stalling them', async () => {
    const native = new FakeNativeEngine();
    const { coordinator, units, advance } = deadlineCoordinator(native);
    await coordinator.attach('c1', lifecycle('c1'));
    await coordinator.updateRenderPressure(frame());
    advance(500);
    await coordinator.updateRenderPressure(frame());
    advance(100);
    await coordinator.updateLifecycle(
      lifecycle('c1', {
        epoch: 2,
        foreground: false,
        state: 'hidden',
      })
    );
    await coordinator.submit(
      request('c1:bg', {
        promptTokens: [1, 2, 3, 4],
        maxNewTokens: 4,
        lifecycle: lifecycle('c1', {
          epoch: 2,
          foreground: false,
          state: 'hidden',
        }),
      })
    );

    await coordinator.pump(4);

    expect(units[0]?.leaseSource).toBe('background');
    expect(units[0]?.nativeTimeBudgetMs).toBeUndefined();
  });

  it('keeps making progress under sustained pressure', async () => {
    const native = new FakeNativeEngine();
    const { coordinator, units, advance } = deadlineCoordinator(native);
    await coordinator.attach('c1', lifecycle('c1'));
    for (let sample = 0; sample < 4; sample += 1) {
      await coordinator.updateRenderPressure(pressured());
      advance(500);
    }
    await coordinator.submit(
      request('c1:hot', { promptTokens: [1, 2, 3, 4], maxNewTokens: 10 })
    );

    const steps = await coordinator.pump(40);

    expect(steps).toBeGreaterThan(0);
    expect(units.length).toBeGreaterThan(0);
    expect(units.length).toBeLessThan(40);
    for (const budget of units) {
      expect(budget.nativeTimeBudgetMs).toBeUndefined();
      expect(budget.maxDecodeTokens).toBeGreaterThanOrEqual(1);
      expect(budget.wallTimeMs).toBeGreaterThan(0);
    }
    // the request finished even though every unit was leased a short budget
    expect(await coordinator.status('c1:hot')).toBeNull();
  });

  it('cancels a leased request and stops pumping it', async () => {
    const native = new FakeNativeEngine();
    const { coordinator, units, advance } = deadlineCoordinator(native);
    await coordinator.attach('c1', lifecycle('c1'));
    await coordinator.updateRenderPressure(pressured());
    advance(500);
    await coordinator.updateRenderPressure(pressured());
    advance(100);
    await coordinator.submit(
      request('c1:cancel', { promptTokens: [1, 2, 3, 4], maxNewTokens: 12 })
    );
    await coordinator.pump(1);
    expect(units).toHaveLength(1);

    expect(await coordinator.cancel('c1:cancel')).toBe(true);
    expect(await coordinator.pump(2)).toBe(0);
  });

  it('can disable rest explicitly for the no-pause ablation', async () => {
    const native = new FakeNativeEngine();
    const pauses: number[] = [];
    const { coordinator, advance } = deadlineCoordinator(native, {
      lease: { maxPauseMs: 0 },
      renderPause: async (pauseMs) => {
        pauses.push(pauseMs);
      },
    });
    await coordinator.attach('c1', lifecycle('c1'));
    await coordinator.updateRenderPressure(frame());
    advance(500);
    await coordinator.updateRenderPressure(frame());
    advance(100);
    await coordinator.submit(
      request('c1:clear-pause', { promptTokens: [1, 2, 3, 4], maxNewTokens: 8 })
    );
    await coordinator.pump(1);
    // a page whose frames are inside the deadline is never paused
    expect(pauses).toEqual([]);

    await coordinator.updateRenderPressure(pressured());
    advance(500);
    await coordinator.updateRenderPressure(pressured());
    advance(100);
    await coordinator.submit(
      request('c1:leased-pause', { promptTokens: [1, 2, 3, 4], maxNewTokens: 8 })
    );
    await coordinator.pump(1);

    // The ablation disables the bounded rest while retaining the frame-
    // cadence window as the only throttle.
    expect(pauses).toEqual([]);
    const policy = native.gpuPolicies.at(-1);
    expect(policy?.source).toBe('deadline');
    expect(policy?.submissionPauseMs).toBe(0);
  });

  it('rests the renderer briefly under the default deadline policy', async () => {
    const native = new FakeNativeEngine();
    const pauses: number[] = [];
    const { coordinator, advance } = deadlineCoordinator(native, {
      renderPause: async (pauseMs) => {
        pauses.push(pauseMs);
      },
    });
    await coordinator.attach('c1', lifecycle('c1'));
    await coordinator.updateRenderPressure(pressured());
    advance(500);
    await coordinator.updateRenderPressure(pressured());
    advance(100);
    await coordinator.submit(
      request('c1:leased-pause', { promptTokens: [1, 2, 3, 4], maxNewTokens: 8 })
    );
    await coordinator.pump(1);

    // 100 ms P99 against a five-frame deadline of 83.33 ms: the default
    // policy spends a bounded part of the excess away from the GPU.
    expect(pauses.length).toBeGreaterThan(0);
    expect(pauses.every((pause) => pause > 0)).toBe(true);
    expect(pauses.every((pause) => pause <= 16)).toBe(true);
    expect(native.gpuPolicies.at(-1)?.submissionPauseMs).toBe(pauses.at(-1));
  });
});
