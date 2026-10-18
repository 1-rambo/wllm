import { describe, expect, it } from 'vitest';
import { SharedWorkerCoordinator } from '../../src/coordinator/shared-worker-coordinator';
import { FakeNativeEngine } from '../../src/native/fake-native-engine';
import { MemoryBlobStore } from '../../src/state/checkpoint-store';
import type { CoordinatorEvent } from '../../src/transport/protocol';
import { lifecycle, request } from '../helpers/factories';

describe('coordinator bounded serving', () => {
  it('parks an idle session in L2 and promotes it back to L1', async () => {
    const coordinator = new SharedWorkerCoordinator(
      new FakeNativeEngine({ prefillMsPerToken: 1, decodeMsPerToken: 1 }),
      new MemoryBlobStore(),
      {
        scheduler: {
          graceMs: 0,
          waitBaseMs: 0,
          waitDepthFactor: 0,
          serviceBoundMs: 0,
          waitBudgetCapMs: 0,
        },
        state: { l1TokenCap: 4, l2TokenCap: 64, repairBudget: 8 },
        execution: {
          maxPrefillTokens: 4,
          baseDecodeTokens: 2,
          foregroundDecodeTokens: 2,
          backgroundDecodeTokens: 2,
          prefillMsPerToken: 1,
          maxUnitWallTimeMs: 100,
        },
        macrotaskYield: async () => undefined,
      }
    );

    const levels: Array<{ nodeId: string; level: string; slotId: number | null }> =
      [];
    coordinator.subscribe((event: CoordinatorEvent) => {
      if (event.type === 'residency') {
        levels.push({
          nodeId: event.nodeId,
          level: event.level,
          slotId: event.slotId,
        });
      }
    });

    await coordinator.attach('c1', lifecycle('c1'));
    await coordinator.attach(
      'c2',
      lifecycle('c2', { foreground: false, state: 'hidden' })
    );
    await coordinator.submit(
      request('c1:r1', { maxNewTokens: 6, lifecycle: lifecycle('c1') })
    );
    await coordinator.submit(
      request('c2:r2', {
        maxNewTokens: 6,
        lifecycle: lifecycle('c2', { foreground: false, state: 'hidden' }),
      })
    );
    await coordinator.pump(1);
    expect(coordinator.stateManager.getResidency('c1:main')?.level).toBe('L1');

    // the zero wait budget makes c2 due immediately after c1's first unit,
    // so admission deterministically overflows the small L1 pool
    await coordinator.pump(8);

    // The first request finishes and its live slot is parked; a later turn
    // for the same session must promote that state back into L1.
    expect(coordinator.stateManager.getResidency('c1:main')?.level).toBe('L2');
    await coordinator.submit(
      request('c1:r3', { maxNewTokens: 2, lifecycle: lifecycle('c1') })
    );
    await coordinator.pump(2);

    const c1Levels = levels
      .filter((entry) => entry.nodeId === 'c1:main')
      .map((entry) => entry.level);
    const parkedIndex = c1Levels.indexOf('L2');
    const restoredIndex = c1Levels.lastIndexOf('L1');

    expect(parkedIndex).toBeGreaterThanOrEqual(0);
    expect(restoredIndex).toBeGreaterThan(parkedIndex);
    expect(
      levels.some(
        (entry) =>
          entry.nodeId === 'c1:main' &&
          entry.level === 'L1' &&
          entry.slotId !== null
      )
    ).toBe(true);
    expect(
      levels.some((entry) => entry.nodeId === 'c2:main' && entry.level === 'L1')
    ).toBe(true);
  });

  it('keeps completed sessions resident when parking is disabled', async () => {
    const coordinator = new SharedWorkerCoordinator(
      new FakeNativeEngine({ prefillMsPerToken: 1, decodeMsPerToken: 1 }),
      new MemoryBlobStore(),
      {
        parkCompleted: false,
        scheduler: { graceMs: 0 },
        state: { l1TokenCap: 64, l2TokenCap: 64 },
        execution: {
          maxPrefillTokens: 4,
          baseDecodeTokens: 2,
          foregroundDecodeTokens: 2,
          backgroundDecodeTokens: 2,
          prefillMsPerToken: 1,
          maxUnitWallTimeMs: 100,
        },
        macrotaskYield: async () => undefined,
      }
    );
    const levels: string[] = [];
    coordinator.subscribe((event: CoordinatorEvent) => {
      if (event.type === 'residency') {
        levels.push(event.level);
      }
    });
    await coordinator.attach('c1', lifecycle('c1'));
    await coordinator.submit(
      request('c1:r1', { maxNewTokens: 2, lifecycle: lifecycle('c1') })
    );
    await coordinator.pump(3);

    expect(coordinator.stateManager.getResidency('c1:main')?.level).toBe('L1');
    expect(levels).not.toContain('L2');
  });

  it('reclaims L2 capacity when completed sessions exceed the cap', async () => {
    const coordinator = new SharedWorkerCoordinator(
      new FakeNativeEngine({ prefillMsPerToken: 1, decodeMsPerToken: 1 }),
      new MemoryBlobStore(),
      {
        scheduler: { graceMs: 0 },
        state: { l1TokenCap: 4, l2TokenCap: 8, repairBudget: 8 },
        execution: {
          maxPrefillTokens: 4,
          baseDecodeTokens: 2,
          foregroundDecodeTokens: 2,
          backgroundDecodeTokens: 2,
          prefillMsPerToken: 1,
          maxUnitWallTimeMs: 100,
        },
        macrotaskYield: async () => undefined,
      }
    );

    for (const clientId of ['c1', 'c2']) {
      await coordinator.attach(clientId, lifecycle(clientId));
      await coordinator.submit(
        request(`${clientId}:r1`, {
          maxNewTokens: 2,
          lifecycle: lifecycle(clientId),
        })
      );
      await coordinator.pump(3);
    }

    expect(coordinator.stateManager.totals.l2Tokens).toBeLessThanOrEqual(8);
    expect(
      ['c1:main', 'c2:main'].map(
        (nodeId) => coordinator.stateManager.getResidency(nodeId)?.level
      )
    ).toContain('L3');
  });
});
