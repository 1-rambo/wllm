import { describe, expect, it } from 'vitest';
import { SharedWorkerCoordinator } from '../src/coordinator/shared-worker-coordinator';
import { FakeNativeEngine } from '../src/native/fake-native-engine';
import { MemoryBlobStore } from '../src/state/checkpoint-store';
import type { CoordinatorEvent } from '../src/transport/protocol';
import { lifecycle, request } from './helpers/factories';

describe('preliminary serving smoke test', () => {
  it('completes a mixed foreground/background workload without violating bounds', async () => {
    const native = new FakeNativeEngine({
      slotCount: 8,
      prefillMsPerToken: 1,
      decodeMsPerToken: 1,
    });
    const coordinator = new SharedWorkerCoordinator(native, new MemoryBlobStore(), {
      scheduler: { graceMs: 0 },
      execution: {
        maxPrefillTokens: 4,
        baseDecodeTokens: 4,
        foregroundDecodeTokens: 4,
        backgroundDecodeTokens: 2,
        prefillMsPerToken: 1,
        maxUnitWallTimeMs: 16,
      },
      macrotaskYield: async () => undefined,
    });

    const submissions = [
      ['bg1', false],
      ['bg2', false],
      ['bg3', false],
      ['fg', true],
    ] as const;
    for (const [clientId, foreground] of submissions) {
      await coordinator.attach(
        clientId,
        lifecycle(clientId, {
          foreground,
          state: foreground ? 'active' : 'hidden',
        })
      );
      await coordinator.submit(
        request(`${clientId}:r`, {
          promptTokens: [1, 2, 3, 4, 5, 6],
          maxNewTokens: 6,
        })
      );
    }

    const events: CoordinatorEvent[] = [];
    coordinator.subscribe((event) => events.push(event));
    await coordinator.pump(100);

    const completionOrder = events
      .filter(
        (event): event is Extract<CoordinatorEvent, { type: 'complete' }> =>
          event.type === 'complete'
      )
      .map((event) => event.requestId);
    expect(completionOrder[0]).toBe('fg:r');
    expect(new Set(completionOrder)).toEqual(
      new Set(['fg:r', 'bg1:r', 'bg2:r', 'bg3:r'])
    );
    expect(coordinator.scheduler.pendingCount).toBe(0);
  });
});
