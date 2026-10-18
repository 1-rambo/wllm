import { describe, expect, it } from 'vitest';
import { SharedWorkerCoordinator } from '../../src/coordinator/shared-worker-coordinator';
import { FakeNativeEngine } from '../../src/native/fake-native-engine';
import { MemoryBlobStore } from '../../src/state/checkpoint-store';
import type { CoordinatorEvent } from '../../src/transport/protocol';
import { lifecycle, request } from '../helpers/factories';

const execution = {
  maxPrefillTokens: 16,
  baseDecodeTokens: 2,
  foregroundDecodeTokens: 2,
  backgroundDecodeTokens: 2,
  prefillMsPerToken: 1,
  maxUnitWallTimeMs: 100,
} as const;

describe('rebuild baseline', () => {
  it('releases the live slot and replays the completed turn on the next request', async () => {
    const native = new FakeNativeEngine({
      prefillMsPerToken: 1,
      decodeMsPerToken: 1,
    });
    const coordinator = new SharedWorkerCoordinator(
      native,
      new MemoryBlobStore(),
      {
        modelHash: 'model-a',
        completedTier: 'rebuild',
        scheduler: { graceMs: 0 },
        state: { l1TokenCap: 64, l2TokenCap: 64 },
        execution,
        macrotaskYield: async () => undefined,
      }
    );
    const events: CoordinatorEvent[] = [];
    coordinator.subscribe((event) => events.push(event));

    await coordinator.attach('c1', lifecycle('c1'));
    await coordinator.submit(
      request('c1:r1', {
        promptText: 'first turn',
        promptTokens: [1, 2, 3, 4],
        maxNewTokens: 2,
      })
    );
    await coordinator.pump(10);

    expect(coordinator.stateManager.getResidency('c1:main')?.level).toBe(
      'rebuild'
    );
    expect((await native.slotStats()).every((slot) => !slot.inUse)).toBe(true);
    expect(coordinator.stateManager.getNode('c1:main')?.continuationText).toContain(
      'first turn'
    );

    await coordinator.submit(
      request('c1:r2', {
        parentNodeId: 'c1:main',
        promptText: 'second turn',
        promptTokens: [5, 6],
        maxNewTokens: 2,
      })
    );
    await coordinator.pump(10);

    const secondUnits = events.filter(
      (event): event is Extract<CoordinatorEvent, { type: 'unit' }> =>
        event.type === 'unit' && event.requestId === 'c1:r2'
    );
    expect(secondUnits.length).toBeGreaterThan(0);
    expect(secondUnits.every((event) => event.progress.prefillCached === 0)).toBe(
      true
    );
    expect(
      secondUnits.some((event) => event.progress.prefillTotal >= 6)
    ).toBe(true);
    expect(coordinator.stateManager.getNode('c1:main')?.continuationText).toContain(
      'second turn'
    );
  });

  it('keeps every turn self contained when the continuation cap is zero', async () => {
    const native = new FakeNativeEngine({
      prefillMsPerToken: 1,
      decodeMsPerToken: 1,
    });
    const coordinator = new SharedWorkerCoordinator(
      native,
      new MemoryBlobStore(),
      {
        modelHash: 'model-a',
        completedTier: 'rebuild',
        continuationTokenCap: 0,
        scheduler: { graceMs: 0 },
        state: { l1TokenCap: 64, l2TokenCap: 64 },
        execution,
        macrotaskYield: async () => undefined,
      }
    );
    const events: CoordinatorEvent[] = [];
    coordinator.subscribe((event) => events.push(event));

    await coordinator.attach('c1', lifecycle('c1'));
    await coordinator.submit(
      request('c1:r1', {
        promptText: 'first turn',
        promptTokens: [1, 2, 3, 4],
        maxNewTokens: 2,
      })
    );
    await coordinator.pump(10);
    expect(coordinator.stateManager.getNode('c1:main')?.continuationText).toBe(
      ''
    );

    await coordinator.submit(
      request('c1:r2', {
        parentNodeId: 'c1:main',
        promptText: 'second turn',
        promptTokens: [5, 6],
        maxNewTokens: 2,
      })
    );
    await coordinator.pump(10);

    const secondUnits = events.filter(
      (event): event is Extract<CoordinatorEvent, { type: 'unit' }> =>
        event.type === 'unit' && event.requestId === 'c1:r2'
    );
    expect(secondUnits.length).toBeGreaterThan(0);
    // Nothing from the first turn is replayed, so the second request only
    // prefills its own two tokens instead of the accumulated history.
    expect(
      Math.max(...secondUnits.map((event) => event.progress.prefillTotal))
    ).toBe(2);
    expect(coordinator.stateManager.getNode('c1:main')?.continuationText).toBe(
      ''
    );
  });

  it('trims a capped continuation to its newest tokens', async () => {
    const native = new FakeNativeEngine({
      prefillMsPerToken: 1,
      decodeMsPerToken: 1,
    });
    const coordinator = new SharedWorkerCoordinator(
      native,
      new MemoryBlobStore(),
      {
        modelHash: 'model-a',
        completedTier: 'rebuild',
        continuationTokenCap: 2,
        scheduler: { graceMs: 0 },
        state: { l1TokenCap: 64, l2TokenCap: 64 },
        execution,
        macrotaskYield: async () => undefined,
      }
    );

    await coordinator.attach('c1', lifecycle('c1'));
    await coordinator.submit(
      request('c1:r1', {
        promptText: 'alpha beta gamma delta',
        promptTokens: [1, 2, 3, 4],
        maxNewTokens: 2,
      })
    );
    await coordinator.pump(10);

    const trimmed =
      coordinator.stateManager.getNode('c1:main')?.continuationText ?? '';
    // The newest text survives, the oldest turn is dropped, and the cap keeps
    // the replayed continuation far below the two turns that produced it.
    expect(trimmed).not.toContain('alpha');
    expect(trimmed).toContain('tok02');
    expect(trimmed.length).toBeLessThan('alpha beta gamma delta'.length);
  });
});
