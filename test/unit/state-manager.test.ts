import { describe, expect, it, vi } from 'vitest';
import { FakeNativeEngine } from '../../src/native/fake-native-engine';
import { MemoryBlobStore } from '../../src/state/checkpoint-store';
import { MultiTierStateManager } from '../../src/state/multi-tier-state-manager';
import { deserializeCheckpoint } from '../../src/utils/serialization';
import { node, session } from '../helpers/factories';

describe('MultiTierStateManager', () => {
  it('restores a warm L2 checkpoint back into a live L1 slot', async () => {
    const native = new FakeNativeEngine({ slotCount: 2 });
    await native.submit({
      requestId: 'native-a',
      promptTokens: [1, 2, 3, 4],
      maxNewTokens: 100,
    });
    await native.advance('native-a', {
      maxPrefillTokens: 4,
      maxDecodeTokens: 0,
      wallTimeMs: 100,
    });

    const manager = new MultiTierStateManager(native, new MemoryBlobStore());
    manager.registerLiveNode(node('a'), 0);

    await manager.demote('a', 'L2');
    expect(manager.getResidency('a')).toMatchObject({ level: 'L2', slotId: null });

    const restored = await manager.restore('a');
    expect(restored).toMatchObject({ success: true, source: 'L2', replayedTokens: 0 });
    expect(manager.getResidency('a')?.level).toBe('L1');
    expect(manager.getResidency('a')?.slotId).not.toBeNull();
    const snapshot = manager.listResidency();
    expect(snapshot).toEqual([
      expect.objectContaining({ nodeId: 'a', level: 'L1' }),
    ]);
    const first = snapshot[0];
    if (!first) throw new Error('missing residency snapshot');
    first.level = 'L2';
    expect(manager.getResidency('a')?.level).toBe('L1');
  });

  it('parks tier two state behind a native handle, not a host copy', async () => {
    const native = new FakeNativeEngine({ slotCount: 2 });
    await native.submit({
      requestId: 'native-c',
      promptTokens: [1, 2, 3, 4],
      maxNewTokens: 100,
    });
    await native.advance('native-c', {
      maxPrefillTokens: 4,
      maxDecodeTokens: 0,
      wallTimeMs: 100,
    });

    const manager = new MultiTierStateManager(native, new MemoryBlobStore());
    manager.registerLiveNode(node('c'), 0);
    await manager.demote('c', 'L2');

    // the live slot is released and the state waits inside the native heap
    expect(manager.getResidency('c')).toMatchObject({ level: 'L2', slotId: null });
    expect(native.parkedStateCount).toBe(1);
    expect(manager.hostBufferBytes('c')).toBe(0);
    expect(manager.bufferBytes('c')).toBeGreaterThan(0);

    // resuming consumes the parked copy: the state lives in the slot afterwards
    const restored = await manager.restore('c');
    expect(restored).toMatchObject({ success: true, source: 'L2' });
    expect(manager.getResidency('c')?.level).toBe('L1');
    expect(native.parkedStateCount).toBe(0);
  });

  it('serializes tier two out to tier three and releases the native copy', async () => {
    const native = new FakeNativeEngine({ slotCount: 2 });
    await native.submit({
      requestId: 'native-d',
      promptTokens: [1, 2, 3, 4],
      maxNewTokens: 100,
    });
    await native.advance('native-d', {
      maxPrefillTokens: 4,
      maxDecodeTokens: 0,
      wallTimeMs: 100,
    });

    const store = new MemoryBlobStore();
    const manager = new MultiTierStateManager(native, store);
    manager.registerLiveNode(node('d'), 0);

    await manager.demote('d', 'L2');
    expect(native.parkedStateCount).toBe(1);

    await manager.demote('d', 'L3');
    // tier three is the only copy left, and it is on the store, not in the heap
    expect(manager.getResidency('d')).toMatchObject({ level: 'L3' });
    expect(native.parkedStateCount).toBe(0);
    expect(manager.hostBufferBytes('d')).toBe(0);
    expect(await store.listKeys()).toHaveLength(1);

    // and it comes back into the native heap when admitted to tier two
    await manager.admitNode('d', 'L2');
    expect(native.parkedStateCount).toBe(1);
    expect(manager.hostBufferBytes('d')).toBe(0);
  });

  it('invalidates native-local tiers while preserving durable L3 checkpoints', async () => {
    const native = new FakeNativeEngine({ slotCount: 3 });
    for (const id of ['live', 'parked', 'durable']) {
      await native.submit({
        requestId: id,
        promptTokens: [1, 2, 3, 4],
        maxNewTokens: 100,
      });
      await native.advance(id, {
        maxPrefillTokens: 4,
        maxDecodeTokens: 0,
        wallTimeMs: 100,
      });
    }

    const manager = new MultiTierStateManager(
      native,
      new MemoryBlobStore()
    );
    manager.registerLiveNode(node('live'), 0);
    manager.registerLiveNode(node('parked'), 1);
    manager.registerLiveNode(node('durable'), 2);
    await manager.demote('parked', 'L2');
    await manager.demote('durable', 'L3');

    manager.invalidateNativeState();

    expect(manager.getResidency('live')).toMatchObject({
      level: 'rebuild',
      slotId: null,
    });
    expect(manager.getResidency('parked')).toMatchObject({
      level: 'rebuild',
      slotId: null,
    });
    expect(manager.getResidency('durable')).toMatchObject({ level: 'L3' });
    expect(manager.getResidency('durable')?.checkpointKey).not.toBeNull();
  });

  it('moves a node through L3 and restores it from the blob store', async () => {
    const native = new FakeNativeEngine({ slotCount: 2 });
    await native.submit({
      requestId: 'native-b',
      promptTokens: [1, 2, 3, 4],
      maxNewTokens: 100,
    });
    await native.advance('native-b', {
      maxPrefillTokens: 4,
      maxDecodeTokens: 0,
      wallTimeMs: 100,
    });

    const store = new MemoryBlobStore();
    const manager = new MultiTierStateManager(native, store);
    manager.registerLiveNode(node('b'), 0);

    await manager.demote('b', 'L3');
    expect(manager.getResidency('b')).toMatchObject({ level: 'L3', slotId: null });
    expect(await store.listKeys()).toHaveLength(1);

    const restored = await manager.restore('b');
    expect(restored).toMatchObject({ success: true, source: 'L3', replayedTokens: 0 });
    expect(manager.getResidency('b')?.level).toBe('L1');
  });

  it('repairs capacity monotonically and respects the admission budget', async () => {
    const native = new FakeNativeEngine({ slotCount: 4 });
    for (const id of ['a', 'b', 'c']) {
      await native.submit({
        requestId: id,
        promptTokens: [1, 2, 3, 4],
        maxNewTokens: 100,
      });
      await native.advance(id, {
        maxPrefillTokens: 4,
        maxDecodeTokens: 0,
        wallTimeMs: 100,
      });
    }

    const manager = new MultiTierStateManager(native, new MemoryBlobStore(), {
      l1TokenCap: 8,
      l2TokenCap: 32,
      l3ByteCap: 1_024,
      repairBudget: 1,
    });
    manager.registerLiveNode(node('a', { lastAccessAt: 10 }), 0);
    manager.registerLiveNode(node('b', { lastAccessAt: 20 }), 1);
    manager.registerNode(node('c'), 'L2');
    manager.installL2Checkpoint('c', await native.slotExport(2));

    const rejected = await manager.admitNode('c', 'L1', 0);
    expect(rejected).toMatchObject({
      success: false,
      reason: 'budget_exhausted',
      steps: 0,
    });

    const admitted = await manager.admitNode('c', 'L1', 1);
    expect(admitted).toMatchObject({ success: true, steps: 1, victims: ['a'] });
    expect(manager.totals.l1Tokens).toBeLessThanOrEqual(8);
  });

  it('skips a zero-token resident when token capacity requires real reduction', async () => {
    const native = new FakeNativeEngine({ slotCount: 4 });
    await native.submit({
      requestId: 'native-zero',
      promptTokens: [],
      maxNewTokens: 0,
    });
    await native.submit({
      requestId: 'native-live',
      promptTokens: [1, 2, 3, 4],
      maxNewTokens: 0,
    });
    await native.advance('native-live', {
      maxPrefillTokens: 4,
      maxDecodeTokens: 0,
      wallTimeMs: 100,
    });
    await native.submit({
      requestId: 'native-next',
      promptTokens: [1, 2, 3, 4],
      maxNewTokens: 100,
    });
    await native.advance('native-next', {
      maxPrefillTokens: 4,
      maxDecodeTokens: 0,
      wallTimeMs: 100,
    });

    const manager = new MultiTierStateManager(
      native,
      new MemoryBlobStore(),
      { l1TokenCap: 4, l2TokenCap: 16 }
    );
    manager.registerLiveNode(
      node('zero', { tokenIds: [], residentTokens: 0, lastAccessAt: 0 }),
      0
    );
    manager.registerLiveNode(
      node('live', { tokenIds: [1, 2, 3, 4], residentTokens: 4, lastAccessAt: 100 }),
      1
    );
    manager.registerNode(
      node('next', { tokenIds: [1, 2, 3, 4], residentTokens: 4 }),
      'L2'
    );
    manager.installL2Checkpoint('next', await native.slotExport(2));

    await expect(manager.admitNode('next', 'L1')).resolves.toMatchObject({
      success: true,
      victims: ['live'],
    });
    expect(manager.getResidency('zero')?.level).toBe('L1');
    expect(manager.getResidency('live')?.level).toBe('L2');
    expect(manager.getResidency('next')?.level).toBe('L1');
  });

  it('accepts a zero-token victim when only the resident-node cap is exceeded', async () => {
    const native = new FakeNativeEngine({ slotCount: 4 });
    await native.submit({
      requestId: 'native-zero',
      promptTokens: [],
      maxNewTokens: 0,
    });
    await native.submit({
      requestId: 'native-live',
      promptTokens: [1, 2, 3, 4],
      maxNewTokens: 0,
    });
    await native.advance('native-live', {
      maxPrefillTokens: 4,
      maxDecodeTokens: 0,
      wallTimeMs: 100,
    });
    await native.submit({
      requestId: 'native-next',
      promptTokens: [1, 2, 3, 4],
      maxNewTokens: 100,
    });
    await native.advance('native-next', {
      maxPrefillTokens: 4,
      maxDecodeTokens: 0,
      wallTimeMs: 100,
    });

    const manager = new MultiTierStateManager(
      native,
      new MemoryBlobStore(),
      { l1TokenCap: 8, maxResidentNodes: 2, l2TokenCap: 16 }
    );
    manager.registerLiveNode(
      node('zero', { tokenIds: [], residentTokens: 0, lastAccessAt: 0 }),
      0
    );
    manager.registerLiveNode(
      node('live', { tokenIds: [1, 2, 3, 4], residentTokens: 4, lastAccessAt: 100 }),
      1
    );
    manager.registerNode(
      node('next', { tokenIds: [1, 2, 3, 4], residentTokens: 4 }),
      'L2'
    );
    manager.installL2Checkpoint('next', await native.slotExport(2));

    await expect(manager.admitNode('next', 'L1')).resolves.toMatchObject({
      success: true,
      victims: ['zero'],
    });
    expect(manager.getResidency('zero')?.level).toBe('L2');
    expect(manager.getResidency('live')?.level).toBe('L1');
    expect(manager.getResidency('next')?.level).toBe('L1');
  });

  it('enforces L2 capacity when an L1 admission spills a victim to L2', async () => {
    const native = new FakeNativeEngine({ slotCount: 3 });
    for (const id of ['a', 'd']) {
      await native.submit({
        requestId: id,
        promptTokens: [1, 2, 3, 4],
        maxNewTokens: 100,
      });
      await native.advance(id, {
        maxPrefillTokens: 4,
        maxDecodeTokens: 0,
        wallTimeMs: 100,
      });
    }

    const manager = new MultiTierStateManager(
      native,
      new MemoryBlobStore(),
      { l1TokenCap: 2, l2TokenCap: 6 }
    );
    manager.registerLiveNode(node('a', { residentTokens: 2 }), 0);
    manager.registerNode(node('d', { residentTokens: 4 }), 'L2');
    manager.installL2Checkpoint('d', await native.slotExport(1));
    manager.registerNode(node('c', { residentTokens: 2 }), 'rebuild');

    await expect(manager.admitNode('c', 'L1')).resolves.toMatchObject({
      success: true,
    });
    expect(manager.totals.l2Tokens).toBeLessThanOrEqual(6);
    expect(manager.getResidency('c')).toMatchObject({ level: 'rebuild' });
    expect(manager.getResidency('d')).toMatchObject({ level: 'L3' });
    expect(native.parkedStateCount).toBe(1);
  });

  it('re-checks L2 capacity after parking reveals the full checkpoint', async () => {
    const native = new FakeNativeEngine({ slotCount: 2 });
    for (const id of ['a', 'b']) {
      await native.submit({
        requestId: id,
        promptTokens: [1, 2, 3, 4],
        maxNewTokens: 100,
      });
      await native.advance(id, {
        maxPrefillTokens: 4,
        maxDecodeTokens: 0,
        wallTimeMs: 100,
      });
    }

    const manager = new MultiTierStateManager(
      native,
      new MemoryBlobStore(),
      { l2TokenCap: 6 }
    );
    manager.registerLiveNode(node('a', { residentTokens: 2 }), 0);
    manager.registerLiveNode(node('b', { residentTokens: 2 }), 1);

    await manager.demote('a', 'L2');
    expect(manager.totals.l2Tokens).toBe(4);

    // Parking b reveals four tokens even though its accounting said two,
    // so admission must repair the physical total again after the park.
    await expect(manager.admitNode('b', 'L2')).resolves.toMatchObject({
      success: true,
    });
    expect(manager.totals.l2Tokens).toBeLessThanOrEqual(6);
    expect(manager.getResidency('a')).toMatchObject({ level: 'L3' });
    expect(native.parkedStateCount).toBe(1);
  });

  it('restores a node that a later parking spilled out to tier three', async () => {
    const native = new FakeNativeEngine({ slotCount: 2 });
    for (const id of ['a', 'b']) {
      await native.submit({
        requestId: id,
        promptTokens: [1, 2, 3, 4],
        maxNewTokens: 100,
      });
      await native.advance(id, {
        maxPrefillTokens: 4,
        maxDecodeTokens: 0,
        wallTimeMs: 100,
      });
    }

    const store = new MemoryBlobStore();
    const manager = new MultiTierStateManager(native, store, {
      l2TokenCap: 6,
    });
    manager.registerLiveNode(node('a', { residentTokens: 2 }), 0);
    manager.registerLiveNode(node('b', { residentTokens: 2 }), 1);

    await manager.demote('a', 'L2');
    await manager.admitNode('b', 'L2');
    expect(manager.getResidency('a')?.level).toBe('L3');

    // The spilled blob is durable state, not a live native handle: the
    // restore must reinstall it from the store.
    const restored = await manager.restore('a');
    expect(restored).toMatchObject({ success: true, source: 'L3' });
    expect(manager.getResidency('a')?.level).toBe('L1');
  });

  it('never parks a node that still has native work in flight', async () => {
    const native = new FakeNativeEngine({ slotCount: 2 });
    const manager = new MultiTierStateManager(native, new MemoryBlobStore(), {
      l1TokenCap: 4,
    });
    manager.registerLiveNode(node('active'), 0);
    manager.registerNode(node('next'), 'rebuild');

    await expect(
      manager.admitNode('next', 'L1', 8, new Set(['active']))
    ).resolves.toMatchObject({
      success: false,
      reason: 'protected_capacity',
      residentLevel: 'rebuild',
    });
    expect(manager.getResidency('active')).toMatchObject({ level: 'L1' });
    expect(native.parkedStateCount).toBe(0);
  });

  it('falls back to rebuild when no serialized checkpoint exists', async () => {
    const manager = new MultiTierStateManager(
      new FakeNativeEngine(),
      new MemoryBlobStore()
    );
    manager.registerNode(node('cold'), 'rebuild');
    await expect(manager.restore('cold')).resolves.toMatchObject({
      success: true,
      source: 'rebuild',
      replayedTokens: 4,
    });
    expect(manager.getResidency('cold')?.level).toBe('rebuild');
  });

  it('rejects L3 admission when every victim candidate is protected', async () => {
    const manager = new MultiTierStateManager(
      new FakeNativeEngine(),
      new MemoryBlobStore(),
      { l3ByteCap: 8 }
    );
    manager.registerNode(node('a'), 'L3');
    manager.registerNode(node('b'), 'L3');
    manager.registerNode(node('c'), 'L2');
    manager.setProtectedSessions([session('c1')]);

    await expect(manager.admitNode('c', 'L3')).resolves.toMatchObject({
      success: false,
      reason: 'protected_capacity',
      residentLevel: 'L2',
    });
  });

  it('moves a checkpoint-less L2 node to rebuild rather than fabricating state', async () => {
    const store = new MemoryBlobStore();
    const manager = new MultiTierStateManager(new FakeNativeEngine(), store);
    manager.registerNode(node('warm'), 'L2');

    await manager.demote('warm', 'L3');
    expect(manager.getResidency('warm')?.level).toBe('rebuild');
    expect(await store.listKeys()).toEqual([]);
  });

  it('forks a parent checkpoint into independent child slots without consuming it', async () => {
    const native = new FakeNativeEngine({ slotCount: 4 });
    const snapshotSpy = vi.spyOn(native, 'slotSnapshot');
    await native.submit({
      requestId: 'parent-native',
      promptTokens: [1, 2, 3, 4],
      maxNewTokens: 4,
    });
    await native.advance('parent-native', {
      maxPrefillTokens: 4,
      maxDecodeTokens: 0,
      wallTimeMs: 100,
    });

    const store = new MemoryBlobStore();
    const manager = new MultiTierStateManager(native, store);
    manager.registerLiveNode(
      node('parent', { tokenIds: [1, 2, 3, 4], continuationText: 'parent' }),
      0
    );
    await manager.demote('parent', 'L3');

    manager.registerNode(
      node('child-a', {
        parentNodeId: 'parent',
        tokenIds: [1, 2, 3, 4],
        continuationText: 'parent',
      }),
      'L2'
    );
    manager.registerNode(
      node('child-b', {
        parentNodeId: 'parent',
        tokenIds: [1, 2, 3, 4],
        continuationText: 'parent',
      }),
      'L2'
    );

    await expect(manager.forkNode('parent', 'child-a')).resolves.toMatchObject({
      success: true,
      targetLevel: 'L1',
    });
    await expect(manager.forkNode('parent', 'child-b')).resolves.toMatchObject({
      success: true,
      targetLevel: 'L1',
    });

    const parent = manager.getResidency('parent');
    const childA = manager.getResidency('child-a');
    const childB = manager.getResidency('child-b');
    expect(parent).toMatchObject({ level: 'L1' });
    expect(childA).toMatchObject({ level: 'L1' });
    expect(childB).toMatchObject({ level: 'L1' });
    expect(new Set([parent?.slotId, childA?.slotId, childB?.slotId]).size).toBe(3);
    const parentSnapshot = await native.slotSnapshot(parent!.slotId!);
    expect(parentSnapshot.tokens).toEqual([1, 2, 3, 4]);
    expect(snapshotSpy).toHaveBeenCalledTimes(2);
  });

  it('persists parent lineage across checkpoint recovery', async () => {
    const native = new FakeNativeEngine({ slotCount: 4 });
    await native.submit({
      requestId: 'parent-native',
      promptTokens: [1, 2, 3, 4],
      maxNewTokens: 4,
    });
    await native.advance('parent-native', {
      maxPrefillTokens: 4,
      maxDecodeTokens: 0,
      wallTimeMs: 100,
    });

    const store = new MemoryBlobStore();
    const manager = new MultiTierStateManager(native, store);
    manager.registerLiveNode(node('parent', { tokenIds: [1, 2, 3, 4] }), 0);
    await manager.demote('parent', 'L3');
    manager.registerNode(
      node('child', {
        parentNodeId: 'parent',
        tokenIds: [1, 2, 3, 4],
      }),
      'L2'
    );
    await manager.forkNode('parent', 'child');
    await manager.demote('child', 'L3');

    const envelopes = await Promise.all(
      (await store.listKeys()).map(async (key) =>
        deserializeCheckpoint((await store.get(key))!)
      )
    );
    expect(
      envelopes.find((envelope) => envelope.nodeId === 'child')?.parentNodeId
    ).toBe('parent');

    const recovered = new MultiTierStateManager(
      new FakeNativeEngine({ slotCount: 4 }),
      store
    );
    await recovered.recoverFromStore();
    expect(recovered.getNode('child')?.parentNodeId).toBe('parent');
  });
});
