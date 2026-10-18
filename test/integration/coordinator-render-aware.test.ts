import { describe, expect, it, vi } from 'vitest';
import { SharedWorkerCoordinator } from '../../src/coordinator/shared-worker-coordinator';
import { FakeNativeEngine } from '../../src/native/fake-native-engine';
import { MemoryBlobStore } from '../../src/state/checkpoint-store';
import { lifecycle, request } from '../helpers/factories';
import type { RenderPressureSignal } from '../../src/contracts';

function pressure(
  clientId: string,
  epoch = 1,
  overrides: Partial<RenderPressureSignal> = {}
): RenderPressureSignal {
  return {
    clientId,
    epoch,
    sampledAt: 0,
    source: 'presented-frame',
    frameCount: 8,
    p50Ms: 16,
    p95Ms: 50,
    maxMs: 50,
    overBudgetRatio: 0.4,
    p99Ms: 60,
    over33Ratio: 0.4,
    over50Ratio: 0.2,
    longestStallMs: 120,
    ...overrides,
  };
}

function coordinatorWithClock(native: FakeNativeEngine) {
  let now = 1_000;
  const coordinator = new SharedWorkerCoordinator(native, new MemoryBlobStore(), {
    scheduler: { graceMs: 0 },
    clock: { now: () => now },
    macrotaskYield: async () => undefined,
  });
  return {
    coordinator,
    advance: (ms: number) => {
      now += ms;
    },
  };
}

describe('SharedWorkerCoordinator render-aware burst control', () => {
  it('pushes downgraded policies to the native engine', async () => {
    const native = new FakeNativeEngine();
    const { coordinator } = coordinatorWithClock(native);
    await coordinator.attach('c1', lifecycle('c1'));

    await coordinator.updateRenderPressure(pressure('c1'));
    await coordinator.updateRenderPressure(pressure('c1'));

    expect(
      (await coordinator.diagnostics()).runtime.gpuBurstPolicy.level
    ).toBe('guarded');
    expect(native.gpuPolicies.map((policy) => policy.level)).toEqual([
      'normal',
      'guarded',
    ]);
    expect((await coordinator.diagnostics()).runtime.gpuBurstPolicy.level).toBe(
      'guarded'
    );
  });

  it('narrows the unit shape under pressure and restores it once clear', async () => {
    const native = new FakeNativeEngine();
    const { coordinator } = coordinatorWithClock(native);
    await coordinator.attach('c1', lifecycle('c1'));
    const base = coordinator.execution.config.foregroundDecodeTokens;
    const basePrefill = coordinator.execution.config.maxPrefillTokens;

    await coordinator.updateRenderPressure(pressure('c1'));
    await coordinator.updateRenderPressure(pressure('c1'));
    const guarded = coordinator.execution.config.foregroundDecodeTokens;
    expect(guarded).toBeLessThan(base);
    expect(coordinator.execution.config.maxPrefillTokens).toBeLessThan(
      basePrefill
    );

    await coordinator.updateRenderPressure(pressure('c1'));
    await coordinator.updateRenderPressure(pressure('c1'));
    expect(coordinator.execution.config.foregroundDecodeTokens).toBeLessThan(
      guarded
    );

    // Recovery is gated by the median window, so two levels back to normal
    // needs more clear samples than the six-per-level counter alone.
    for (let sample = 0; sample < 20; sample += 1) {
      await coordinator.updateRenderPressure(
        pressure('c1', 1, { p95Ms: 18, overBudgetRatio: 0 })
      );
    }
    expect(coordinator.execution.config.foregroundDecodeTokens).toBe(base);
    expect(coordinator.execution.config.maxPrefillTokens).toBe(basePrefill);
  });

  it('keeps unsliced units intact under render pressure', async () => {
    const native = new FakeNativeEngine();
    const coordinator = new SharedWorkerCoordinator(
      native,
      new MemoryBlobStore(),
      {
        scheduler: { graceMs: 0 },
        clock: { now: () => 1_000 },
        macrotaskYield: async () => undefined,
        executionMode: 'unsliced',
        execution: {
          maxPrefillTokens: 4_096,
          baseDecodeTokens: 4_096,
          foregroundDecodeTokens: 4_096,
          backgroundDecodeTokens: 4_096,
          maxUnitWallTimeMs: 30_000,
          unitWallTimeGraceMs: 30_000,
          adaptiveUnits: false,
          targetUnitWallTimeMs: 30_000,
          nativeTimeBudgetEnabled: false,
          maxDecodeTokens: 4_096,
        },
      }
    );
    await coordinator.attach('c1', lifecycle('c1'));

    for (let sample = 0; sample < 4; sample += 1) {
      await coordinator.updateRenderPressure(pressure('c1'));
    }

    const diagnostics = await coordinator.diagnostics();
    expect(diagnostics.runtime.executionMode).toBe('unsliced');
    expect(diagnostics.runtime.gpuBurstPolicy.level).toBe('constrained');
    expect(diagnostics.runtime.lease).toBeNull();
    expect(diagnostics.runtime.execution).toEqual({
      maxPrefillTokens: 4_096,
      maxDecodeTokens: 4_096,
      foregroundDecodeTokens: 4_096,
      backgroundDecodeTokens: 4_096,
      backgroundPrefillTokens: null,
      targetUnitWallTimeMs: 30_000,
      adaptiveUnits: false,
      nativeTimeBudgetEnabled: false,
    });
  });

  it('yields before native work when render pressure selects a pause', async () => {
    const native = new FakeNativeEngine();
    let now = 1_000;
    const renderPause = vi.fn(async () => undefined);
    const coordinator = new SharedWorkerCoordinator(
      native,
      new MemoryBlobStore(),
      {
        scheduler: { graceMs: 0 },
        clock: { now: () => now },
        macrotaskYield: async () => undefined,
        renderPause,
        burst: { pauseOnPressure: true },
      }
    );
    await coordinator.attach('c1', lifecycle('c1'));
    await coordinator.updateRenderPressure(pressure('c1'));
    await coordinator.updateRenderPressure(pressure('c1'));
    await coordinator.submit(request('c1:r1', { maxNewTokens: 1 }));

    await coordinator.pump(1);

    expect(renderPause).toHaveBeenCalledWith(32);
  });

  it('holds new units once the armed gate reaches the worst level', async () => {
    const native = new FakeNativeEngine();
    const coordinator = new SharedWorkerCoordinator(
      native,
      new MemoryBlobStore(),
      {
        scheduler: { graceMs: 0 },
        clock: { now: () => 1_000 },
        macrotaskYield: async () => undefined,
        renderGate: true,
      }
    );
    await coordinator.attach('c1', lifecycle('c1'));
    for (let sample = 0; sample < 4; sample += 1) {
      await coordinator.updateRenderPressure(pressure('c1'));
    }
    expect(
      (await coordinator.diagnostics()).runtime.gpuBurstPolicy.level
    ).toBe('constrained');

    await coordinator.submit(request('c1:gated', { maxNewTokens: 4 }));
    expect(await coordinator.pump(1)).toBe(0);

    for (let sample = 0; sample < 20; sample += 1) {
      await coordinator.updateRenderPressure(
        pressure('c1', 1, { p95Ms: 18, overBudgetRatio: 0 })
      );
    }
    expect(
      (await coordinator.diagnostics()).runtime.gpuBurstPolicy.level
    ).toBe('normal');
    expect(await coordinator.pump(1)).toBe(1);
  });

  it('force-opens a held gate so inference cannot starve', async () => {
    const native = new FakeNativeEngine();
    let now = 1_000;
    const coordinator = new SharedWorkerCoordinator(
      native,
      new MemoryBlobStore(),
      {
        scheduler: { graceMs: 0 },
        clock: { now: () => now },
        macrotaskYield: async () => undefined,
        renderGate: true,
        renderGateMaxHoldMs: 1_500,
        renderGateMinOpenMs: 300,
      }
    );
    await coordinator.attach('c1', lifecycle('c1'));
    for (let sample = 0; sample < 4; sample += 1) {
      await coordinator.updateRenderPressure(pressure('c1'));
    }
    expect(
      (await coordinator.diagnostics()).runtime.gpuBurstPolicy.level
    ).toBe('constrained');

    await coordinator.submit(request('c1:gate-hold', { maxNewTokens: 4 }));
    expect(await coordinator.pump(1)).toBe(0);

    now += 1_000;
    await coordinator.updateRenderPressure(pressure('c1'));
    expect(await coordinator.pump(1)).toBe(0);

    // Past the hold cap the gate opens even though pressure has not cleared.
    now += 600;
    await coordinator.updateRenderPressure(pressure('c1'));
    expect(await coordinator.pump(1)).toBe(1);

    // Once the minimum open window expires the gate closes again.
    now += 400;
    await coordinator.updateRenderPressure(pressure('c1'));
    await coordinator.submit(request('c1:gate-hold-2', { maxNewTokens: 4 }));
    expect(await coordinator.pump(1)).toBe(0);
  });

  it('reopens the gate inside one clear window without the hold cap', async () => {
    const native = new FakeNativeEngine();
    let now = 1_000;
    const coordinator = new SharedWorkerCoordinator(
      native,
      new MemoryBlobStore(),
      {
        scheduler: { graceMs: 0 },
        clock: { now: () => now },
        macrotaskYield: async () => undefined,
        renderGate: true,
        // far beyond the clear window: reopening here cannot come from the
        // liveness fallback that force-opens a stuck gate
        renderGateMaxHoldMs: 60_000,
        renderGateMinOpenMs: 300,
      }
    );
    await coordinator.attach('c1', lifecycle('c1'));
    for (let sample = 0; sample < 4; sample += 1) {
      await coordinator.updateRenderPressure(pressure('c1'));
    }
    await coordinator.submit(request('c1:gate-recover', { maxNewTokens: 4 }));
    expect(await coordinator.pump(1)).toBe(0);
    expect((await coordinator.diagnostics()).runtime.renderGate).toEqual({
      enabled: true,
      open: false,
    });

    // The contention schedule holds pressure for 4 s and then clears for
    // 6 s; a gate that outlives its own clear window is what turned the
    // earlier recovery measurement into a 10.4 s tail.
    const clearStartsAt = now;
    const clearWindowEnds = clearStartsAt + 6_000;
    let reopenedAt: number | null = null;
    let normalAt: number | null = null;
    while (now < clearWindowEnds && normalAt === null) {
      now += 500;
      await coordinator.updateRenderPressure(
        pressure('c1', 1, { p95Ms: 19, overBudgetRatio: 0 })
      );
      await coordinator.pump(1);
      const diagnostics = await coordinator.diagnostics();
      // leaving the worst level is what reopens the gate, so the gate opens
      // no later than the step back to the fast shape
      if (reopenedAt === null && diagnostics.runtime.renderGate.open) {
        reopenedAt = now;
      }
      if (diagnostics.runtime.gpuBurstPolicy.level === 'normal') {
        normalAt = now;
      }
    }
    expect(reopenedAt).not.toBeNull();
    expect(normalAt).not.toBeNull();
    expect(reopenedAt! - clearStartsAt).toBeLessThan(6_000);
    expect(normalAt! - clearStartsAt).toBeLessThan(6_000);
    expect(reopenedAt!).toBeLessThanOrEqual(normalAt!);
  });

  it('rejects pressure from a non-foreground client', async () => {
    const native = new FakeNativeEngine();
    const { coordinator } = coordinatorWithClock(native);
    await coordinator.attach('c1', lifecycle('c1'));
    await coordinator.attach(
      'c2',
      lifecycle('c2', { foreground: false, state: 'hidden' })
    );

    await coordinator.updateRenderPressure(pressure('c2'));
    await coordinator.updateRenderPressure(pressure('c2'));

    expect(
      (await coordinator.diagnostics()).runtime.gpuBurstPolicy.level
    ).toBe('normal');
    expect(native.gpuPolicies).toEqual([]);
  });

  it('rejects pressure from a stale lifecycle epoch', async () => {
    const native = new FakeNativeEngine();
    const { coordinator } = coordinatorWithClock(native);
    await coordinator.attach('c1', lifecycle('c1', { epoch: 2 }));

    await coordinator.updateRenderPressure(pressure('c1', 1));
    await coordinator.updateRenderPressure(pressure('c1', 1));

    expect(
      (await coordinator.diagnostics()).runtime.gpuBurstPolicy.level
    ).toBe('normal');
    expect(native.gpuPolicies).toEqual([]);
  });

  it('returns to normal and ignores feedback when render awareness is disabled', async () => {
    const native = new FakeNativeEngine();
    const { coordinator } = coordinatorWithClock(native);
    await coordinator.attach('c1', lifecycle('c1'));
    await coordinator.updateRenderPressure(pressure('c1'));
    await coordinator.updateRenderPressure(pressure('c1'));
    expect(
      (await coordinator.diagnostics()).runtime.gpuBurstPolicy.level
    ).toBe('guarded');

    await coordinator.configureRenderAware(false);
    await coordinator.updateRenderPressure(pressure('c1'));
    await coordinator.updateRenderPressure(pressure('c1'));

    expect(
      (await coordinator.diagnostics()).runtime.gpuBurstPolicy.level
    ).toBe('normal');
    expect(native.gpuPolicies.at(-1)?.level).toBe('normal');
    expect((await coordinator.diagnostics()).runtime.renderAware).toBe(false);
  });

  it('returns to normal when the reporting client detaches', async () => {
    const native = new FakeNativeEngine();
    const { coordinator } = coordinatorWithClock(native);
    await coordinator.attach('c1', lifecycle('c1'));
    await coordinator.updateRenderPressure(pressure('c1'));
    await coordinator.updateRenderPressure(pressure('c1'));
    expect(
      (await coordinator.diagnostics()).runtime.gpuBurstPolicy.level
    ).toBe('guarded');

    await coordinator.detach('c1');

    expect(
      (await coordinator.diagnostics()).runtime.gpuBurstPolicy.level
    ).toBe('normal');
    expect(native.gpuPolicies.at(-1)?.level).toBe('normal');
  });
});
