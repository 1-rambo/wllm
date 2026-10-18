import { describe, expect, it } from 'vitest';
import { SharedWorkerCoordinator } from '../../src/coordinator/shared-worker-coordinator';
import { FakeNativeEngine } from '../../src/native/fake-native-engine';
import { MemoryBlobStore } from '../../src/state/checkpoint-store';
import type { CoordinatorEvent } from '../../src/transport/protocol';
import { lifecycle, request } from '../helpers/factories';

const execution = {
  maxPrefillTokens: 16,
  baseDecodeTokens: 4,
  foregroundDecodeTokens: 4,
  backgroundDecodeTokens: 4,
  prefillMsPerToken: 1,
  maxUnitWallTimeMs: 100,
} as const;

function coordinator(native: FakeNativeEngine, store: MemoryBlobStore) {
  return new SharedWorkerCoordinator(native, store, {
    modelHash: 'model-a',
    completedTier: 'L3',
    scheduler: { graceMs: 0 },
    state: { l1TokenCap: 128, l2TokenCap: 128, l3ByteCap: 1_024_000 },
    execution,
    macrotaskYield: async () => undefined,
  });
}

describe('context-tree forks', () => {
  it('gives siblings independent KV while preserving the parent checkpoint', async () => {
    const native = new FakeNativeEngine({
      slotCount: 8,
      prefillMsPerToken: 1,
      decodeMsPerToken: 1,
    });
    const shared = coordinator(native, new MemoryBlobStore());
    await shared.attach('c1', lifecycle('c1'));

    await shared.submit(
      request('c1:root', {
        nodeId: 'root',
        promptText: 'root',
        promptTokens: [1, 2, 3, 4],
        maxNewTokens: 2,
      })
    );
    await shared.pump(10);
    expect(shared.stateManager.getNode('root')?.tokenIds).toEqual([1, 2, 3, 4]);

    await shared.submit(
      request('c1:child-a', {
        nodeId: 'child-a',
        forkFromNodeId: 'root',
        promptText: 'child a',
        promptTokens: [5, 6],
        maxNewTokens: 2,
      })
    );
    await shared.pump(10);

    await shared.submit(
      request('c1:child-b', {
        nodeId: 'child-b',
        forkFromNodeId: 'root',
        promptText: 'child b',
        promptTokens: [7, 8],
        maxNewTokens: 2,
      })
    );
    await shared.pump(10);

    expect(shared.stateManager.getNode('child-a')?.parentNodeId).toBe('root');
    expect(shared.stateManager.getNode('child-b')?.parentNodeId).toBe('root');
    expect(shared.stateManager.getNode('child-a')?.tokenIds).toEqual([
      1, 2, 3, 4, 5, 6,
    ]);
    expect(shared.stateManager.getNode('child-b')?.tokenIds).toEqual([
      1, 2, 3, 4, 7, 8,
    ]);
    expect(shared.stateManager.getNode('root')?.tokenIds).toEqual([1, 2, 3, 4]);
    expect(shared.stateManager.getResidency('root')).toMatchObject({
      level: 'L1',
    });
  });

  it('restores a child from L3 after restart and keeps its parent lineage', async () => {
    const store = new MemoryBlobStore();
    const first = coordinator(
      new FakeNativeEngine({
        slotCount: 8,
        prefillMsPerToken: 1,
        decodeMsPerToken: 1,
      }),
      store
    );
    await first.attach('c1', lifecycle('c1'));
    await first.submit(
      request('c1:root', {
        nodeId: 'root',
        promptText: 'root',
        promptTokens: [1, 2, 3, 4],
        maxNewTokens: 2,
      })
    );
    await first.pump(10);
    await first.submit(
      request('c1:child', {
        nodeId: 'child',
        forkFromNodeId: 'root',
        promptText: 'child',
        promptTokens: [5, 6],
        maxNewTokens: 2,
      })
    );
    await first.pump(10);

    const replacement = coordinator(
      new FakeNativeEngine({
        slotCount: 8,
        prefillMsPerToken: 1,
        decodeMsPerToken: 1,
      }),
      store
    );
    await expect(replacement.whenReady()).resolves.toMatchObject({
      recovered: 2,
      skipped: 0,
    });
    expect(replacement.stateManager.getNode('child')?.parentNodeId).toBe('root');
    expect(replacement.stateManager.getNode('child')?.tokenIds).toEqual([
      1, 2, 3, 4, 5, 6,
    ]);

    const events: CoordinatorEvent[] = [];
    replacement.subscribe((event) => events.push(event));
    await replacement.attach('c1', lifecycle('c1'));
    await replacement.submit(
      request('c1:resume', {
        nodeId: 'child',
        promptText: 'resume',
        promptTokens: [9, 10],
        maxNewTokens: 2,
      })
    );
    await replacement.pump(10);

    expect(replacement.stateManager.getNode('child')?.tokenIds).toEqual([
      1, 2, 3, 4, 5, 6, 9, 10,
    ]);
    expect(
      events.some(
        (event) =>
          event.type === 'unit' &&
          event.requestId === 'c1:resume' &&
          (event.progress.prefillCached ?? 0) >= 6
      )
    ).toBe(true);
  });
});
