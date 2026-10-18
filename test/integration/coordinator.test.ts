import { describe, expect, it, vi } from 'vitest';
import { SharedWorkerCoordinator } from '../../src/coordinator/shared-worker-coordinator';
import { FakeNativeEngine } from '../../src/native/fake-native-engine';
import { MemoryBlobStore } from '../../src/state/checkpoint-store';
import { lifecycle, request } from '../helpers/factories';
import type { CoordinatorEvent } from '../../src/transport/protocol';

describe('SharedWorkerCoordinator', () => {
  it('completes queued work and exposes a stable completion event', async () => {
    const coordinator = new SharedWorkerCoordinator(
      new FakeNativeEngine({ prefillMsPerToken: 1, decodeMsPerToken: 1 }),
      new MemoryBlobStore(),
      {
        scheduler: { graceMs: 0 },
        execution: {
          maxPrefillTokens: 4,
          baseDecodeTokens: 4,
          foregroundDecodeTokens: 4,
          backgroundDecodeTokens: 4,
          prefillMsPerToken: 1,
          maxUnitWallTimeMs: 100,
        },
        macrotaskYield: async () => undefined,
      }
    );
    const events: unknown[] = [];
    const unsubscribe = coordinator.subscribe((event) => events.push(event));
    await coordinator.attach('c1', lifecycle('c1'));
    await coordinator.submit(request('c1:r1', { maxNewTokens: 3 }));

    await coordinator.pump(2);
    expect(events).toContainEqual(
      expect.objectContaining({
        type: 'complete',
        requestId: 'c1:r1',
        output: expect.stringContaining('tok01'),
      })
    );
    expect(coordinator.scheduler.pendingCount).toBe(0);
    unsubscribe();
  });

  it('cancels queued work when a client detaches', async () => {
    const native = new FakeNativeEngine();
    const coordinator = new SharedWorkerCoordinator(native, new MemoryBlobStore(), {
      scheduler: { graceMs: 0 },
      macrotaskYield: async () => undefined,
    });
    await coordinator.attach('c1', lifecycle('c1'));
    await coordinator.submit(request('c1:r1'));
    await coordinator.detach('c1');

    expect(coordinator.scheduler.pendingCount).toBe(0);
    await expect(coordinator.submit(request('c1:r2'))).resolves.toEqual({
      accepted: false,
      reason: 'fenced',
    });
  });

  it('never starts a request fenced by a detach/attach generation change', async () => {
    const coordinator = new SharedWorkerCoordinator(
      new FakeNativeEngine(),
      new MemoryBlobStore(),
      { scheduler: { graceMs: 0 }, macrotaskYield: async () => undefined }
    );
    await coordinator.attach('c1', lifecycle('c1', { epoch: 1 }));
    await coordinator.submit(request('c1:r1'));
    await coordinator.detach('c1');
    await coordinator.attach('c1', lifecycle('c1', { epoch: 2 }));

    const runs = await coordinator.pump(1);
    expect(runs).toBe(0);
  });

  it('emits a failed event when a native unit throws', async () => {
    const native = new FakeNativeEngine({
      prefillMsPerToken: 1,
      decodeMsPerToken: 1,
    });
    const coordinator = new SharedWorkerCoordinator(
      native,
      new MemoryBlobStore(),
      {
        scheduler: { graceMs: 0 },
        execution: {
          maxPrefillTokens: 4,
          baseDecodeTokens: 4,
          foregroundDecodeTokens: 4,
          backgroundDecodeTokens: 4,
          prefillMsPerToken: 1,
          maxUnitWallTimeMs: 100,
        },
        macrotaskYield: async () => undefined,
      }
    );
    const events: CoordinatorEvent[] = [];
    coordinator.subscribe((event) => events.push(event));
    await coordinator.attach('c1', lifecycle('c1'));
    await coordinator.submit(request('c1:r1', { maxNewTokens: 3 }));
    vi.spyOn(native, 'advance').mockRejectedValueOnce(
      new Error('native advance failed for c1:r1')
    );

    await expect(coordinator.pump(1)).rejects.toThrow(/native advance failed/);

    expect(events).toContainEqual(
      expect.objectContaining({
        type: 'failed',
        requestId: 'c1:r1',
        error: expect.stringContaining('native advance failed'),
      })
    );
    const diagnostics = await coordinator.diagnostics();
    expect(diagnostics.runtime.lastError).toContain('native advance failed');
    expect(diagnostics.runtime.cancelledRequests).toContain('c1:r1');
  });

  it('yields between bounded units so lifecycle updates can arrive', async () => {
    const yieldMacrotask = vi.fn(async () => undefined);
    const coordinator = new SharedWorkerCoordinator(
      new FakeNativeEngine({ prefillMsPerToken: 1, decodeMsPerToken: 1 }),
      new MemoryBlobStore(),
      {
        scheduler: { graceMs: 0 },
        execution: {
          maxPrefillTokens: 2,
          baseDecodeTokens: 1,
          foregroundDecodeTokens: 1,
          backgroundDecodeTokens: 1,
          prefillMsPerToken: 1,
          maxUnitWallTimeMs: 10,
        },
        macrotaskYield: yieldMacrotask,
      }
    );
    await coordinator.attach('c1', lifecycle('c1'));
    await coordinator.submit(request('c1:r1', { maxNewTokens: 2 }));
    await coordinator.pump(4);

    expect(yieldMacrotask).toHaveBeenCalledTimes(2);
  });
});
