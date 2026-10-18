import { describe, expect, it } from 'vitest';
import { SharedWorkerCoordinator } from '../../src/coordinator/shared-worker-coordinator';
import { FakeNativeEngine } from '../../src/native/fake-native-engine';
import { MemoryBlobStore } from '../../src/state/checkpoint-store';
import type { CoordinatorEvent } from '../../src/transport/protocol';
import {
  deserializeCheckpoint,
  serializeCheckpoint,
} from '../../src/utils/serialization';
import { lifecycle, request } from '../helpers/factories';

const execution = {
  maxPrefillTokens: 4,
  baseDecodeTokens: 2,
  foregroundDecodeTokens: 2,
  backgroundDecodeTokens: 2,
  prefillMsPerToken: 1,
  maxUnitWallTimeMs: 100,
} as const;

describe('coordinator restart recovery', () => {
  it('rehydrates an addressed L3 node and resumes its restored prefix', async () => {
    const store = new MemoryBlobStore();
    const first = new SharedWorkerCoordinator(
      new FakeNativeEngine({ prefillMsPerToken: 1, decodeMsPerToken: 1 }),
      store,
      {
        modelHash: 'model-a',
        completedTier: 'L3',
        scheduler: { graceMs: 0 },
        state: { l1TokenCap: 64, l2TokenCap: 64, l3ByteCap: 1_024 },
        execution,
        macrotaskYield: async () => undefined,
      }
    );
    await first.whenReady();
    await first.attach('c1', lifecycle('c1'));
    await first.submit(
      request('c1:r1', {
        promptText: 'first turn',
        promptTokens: [1, 2, 3, 4],
        maxNewTokens: 2,
      })
    );
    await first.pump(10);

    expect(first.stateManager.getResidency('c1:main')?.level).toBe('L3');
    expect(await store.listKeys()).toHaveLength(1);

    // The old manager, coordinator, and native engine are all gone. Only the
    // OPFS-backed L3 envelope survives into the replacement runtime.
    const second = new SharedWorkerCoordinator(
      new FakeNativeEngine({ prefillMsPerToken: 1, decodeMsPerToken: 1 }),
      store,
      {
        modelHash: 'model-a',
        completedTier: 'L3',
        scheduler: { graceMs: 0 },
        state: { l1TokenCap: 64, l2TokenCap: 64, l3ByteCap: 1_024 },
        execution,
        macrotaskYield: async () => undefined,
      }
    );
    const recovery = await second.whenReady();
    expect(recovery).toMatchObject({ recovered: 1, skipped: 0 });
    expect(second.stateManager.getNode('c1:main')?.turnText).toBe('first turn');
    expect(second.stateManager.getResidency('c1:main')?.level).toBe('L3');

    const events: CoordinatorEvent[] = [];
    second.subscribe((event) => events.push(event));
    await second.attach('c1', lifecycle('c1'));
    await second.submit(
      request('c1:r2', {
        parentNodeId: 'c1:main',
        promptText: 'second turn',
        promptTokens: [5, 6],
        maxNewTokens: 2,
      })
    );
    await second.pump(10);

    const levels = events
      .filter(
        (event): event is Extract<CoordinatorEvent, { type: 'residency' }> =>
          event.type === 'residency' && event.nodeId === 'c1:main'
      )
      .map((event) => event.level);
    expect(levels).toContain('L1');
    expect(levels.at(-1)).toBe('L3');
    expect(levels).not.toContain('rebuild');
    expect(
      events.some(
        (event) =>
          event.type === 'unit' &&
          event.requestId === 'c1:r2' &&
          (event.progress.prefillCached ?? 0) >= 4
      )
    ).toBe(true);
    expect(second.stateManager.getNode('c1:main')?.tokenIds).toEqual([
      1, 2, 3, 4, 5, 6,
    ]);
  });

  it('keeps a checkpoint from another model rather than claiming recovery', async () => {
    const store = new MemoryBlobStore();
    const seed = new SharedWorkerCoordinator(
      new FakeNativeEngine(),
      store,
      {
        modelHash: 'model-a',
        completedTier: 'L3',
        state: { l1TokenCap: 64, l2TokenCap: 64 },
        execution,
        macrotaskYield: async () => undefined,
      }
    );
    await seed.whenReady();
    await seed.attach('c1', lifecycle('c1'));
    await seed.submit(
      request('c1:r1', {
        promptTokens: [1, 2, 3, 4],
        maxNewTokens: 1,
      })
    );
    await seed.pump(6);

    const replacement = new SharedWorkerCoordinator(
      new FakeNativeEngine(),
      store,
      {
        modelHash: 'model-b',
        completedTier: 'L3',
        state: { l1TokenCap: 64, l2TokenCap: 64 },
        execution,
        macrotaskYield: async () => undefined,
      }
    );
    const recovery = await replacement.whenReady();

    expect(recovery).toMatchObject({ recovered: 0, skipped: 1 });
    expect(recovery.errors.join('\n')).toContain('model hash mismatch');
    expect(await store.listKeys()).toHaveLength(1);
  });

  it('rejects a checksum-corrupted envelope without deleting it', async () => {
    const store = new MemoryBlobStore();
    const seed = new SharedWorkerCoordinator(
      new FakeNativeEngine(),
      store,
      {
        modelHash: 'model-a',
        completedTier: 'L3',
        state: { l1TokenCap: 64, l2TokenCap: 64 },
        execution,
        macrotaskYield: async () => undefined,
      }
    );
    await seed.whenReady();
    await seed.attach('c1', lifecycle('c1'));
    await seed.submit(
      request('c1:r1', {
        promptTokens: [1, 2, 3, 4],
        maxNewTokens: 1,
      })
    );
    await seed.pump(6);

    const [key] = await store.listKeys();
    expect(key).toBeTruthy();
    const envelope = deserializeCheckpoint((await store.get(key!))!);
    envelope.bytes[0] = (envelope.bytes[0] ?? 0) ^ 0xff;
    await store.put(key!, serializeCheckpoint(envelope));

    const replacement = new SharedWorkerCoordinator(
      new FakeNativeEngine(),
      store,
      {
        modelHash: 'model-a',
        completedTier: 'L3',
        state: { l1TokenCap: 64, l2TokenCap: 64 },
        execution,
        macrotaskYield: async () => undefined,
      }
    );
    const recovery = await replacement.whenReady();
    expect(recovery).toMatchObject({ recovered: 0, skipped: 1 });
    expect(recovery.errors.join('\n')).toContain('checksum mismatch');
    expect(await store.listKeys()).toEqual([key]);
  });
});
