import type {
  BlobStore,
  Clock,
  CheckpointEnvelope,
  NativeSlotCheckpoint,
  ResidencyLevel,
  ResidencyRecord,
  RestoreResult,
  SessionKey,
  SessionNode,
  StateAdmissionResult,
  StateManagerConfig,
} from '../contracts';
import {
  CheckpointError,
  systemClock,
  sessionKeyString,
  WllmError,
} from '../contracts';
import type { NativeEngine } from '../native/native-engine';
import {
  checkpointChecksum,
  deserializeCheckpoint,
  makeCheckpointEnvelope,
  serializeCheckpoint,
} from '../utils/serialization';
import type { TelemetrySink } from '../telemetry';

const DEFAULT_CONFIG: StateManagerConfig = {
  l1TokenCap: 4_096,
  l2TokenCap: 8_192,
  l3ByteCap: 64 * 1024 * 1024,
  repairBudget: 8,
  replacementPolicy: 'hybrid',
  lifecycleProtection: true,
  evictOnPause: false,
  admissionPolicy: 'monotonic-bounded',
};

const EMPTY_NODE_SET: ReadonlySet<string> = new Set();

export interface CheckpointRecoverySummary {
  scanned: number;
  recovered: number;
  skipped: number;
  errors: string[];
}

interface ForkSnapshotCache {
  parentNodeId: string;
  slotId: number;
  tokenIds: number[];
  continuationText: string;
  checkpoint: NativeSlotCheckpoint;
}

export class MultiTierStateManager {
  readonly config: StateManagerConfig;
  private readonly nodes = new Map<string, SessionNode>();
  private readonly residency = new Map<string, ResidencyRecord>();
  private readonly l2Checkpoints = new Map<string, NativeSlotCheckpoint>();
  private protectedSessions = new Set<string>();
  private modelHash = 'prototype-model';
  // A completed parent is immutable while children fork from it. Keep one
  // native snapshot so a fan-out imports the same KV bytes instead of
  // copying the full parent state out of the native heap once per child.
  private forkSnapshot: ForkSnapshotCache | null = null;

  constructor(
    private readonly native: NativeEngine,
    private readonly store: BlobStore,
    config: Partial<StateManagerConfig> = {},
    private readonly clock: Clock = systemClock,
    private readonly telemetry?: TelemetrySink
  ) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  registerNode(node: SessionNode, level: ResidencyLevel = 'L2'): ResidencyRecord {
    this.nodes.set(node.nodeId, cloneNode(node));
    const existing = this.residency.get(node.nodeId);
    if (existing) {
      existing.tokenCount = residentTokenCount(node);
      existing.lastAccessAt = node.lastAccessAt;
      existing.accessCount = node.accessCount;
      return { ...existing };
    }
    const record: ResidencyRecord = {
      nodeId: node.nodeId,
      level,
      slotId: null,
      tokenCount: residentTokenCount(node),
      checkpointKey: null,
      lastAccessAt: node.lastAccessAt,
      accessCount: node.accessCount,
    };
    this.residency.set(node.nodeId, record);
    this.telemetry?.emit(
      'state_residency',
      { nodeId: node.nodeId, level, tokens: record.tokenCount },
      { clientId: node.session.clientId, sessionId: node.session.sessionId }
    );
    return { ...record };
  }

  getNode(nodeId: string): SessionNode | undefined {
    const node = this.nodes.get(nodeId);
    return node ? cloneNode(node) : undefined;
  }

  getResidency(nodeId: string): ResidencyRecord | undefined {
    const record = this.residency.get(nodeId);
    return record ? { ...record } : undefined;
  }

  listResidency(): ResidencyRecord[] {
    return [...this.residency.values()].map((record) => ({ ...record }));
  }

  setProtectedSessions(keys: SessionKey[]): void {
    this.protectedSessions =
      this.config.lifecycleProtection === false
        ? new Set()
        : new Set(keys.map(sessionKeyString));
  }

  protectedSessionCount(): number {
    return this.protectedSessions.size;
  }

  setModelHash(modelHash: string): void {
    this.modelHash = modelHash;
  }

  configure(config: Partial<StateManagerConfig>): void {
    Object.assign(this.config, config);
  }

  // A native worker owns every L1 slot and every parked L2 handle. Once its
  // process is gone, those identifiers belong to a dead address space and
  // must never be handed to the replacement worker. L3 checkpoints are the
  // only state that survives this boundary because they live in the store.
  invalidateNativeState(): void {
    this.l2Checkpoints.clear();
    this.forkSnapshot = null;
    for (const record of this.residency.values()) {
      if (record.level !== 'L1' && record.level !== 'L2') {
        continue;
      }
      record.level = 'rebuild';
      record.slotId = null;
      record.checkpointKey = null;
      const node = this.nodes.get(record.nodeId);
      this.telemetry?.emit(
        'state_residency',
        {
          nodeId: record.nodeId,
          level: record.level,
          tokens: record.tokenCount,
          reason: 'native-worker-lost',
        },
        node
          ? { clientId: node.session.clientId, sessionId: node.session.sessionId }
          : undefined
      );
    }
  }
  // Rebuild the logical node registry from persisted L3 envelopes. This is
  // the restart path that makes an application-retained node identifier
  // addressable after the previous SharedWorker context is gone.
  async recoverFromStore(): Promise<CheckpointRecoverySummary> {
    const summary: CheckpointRecoverySummary = {
      scanned: 0,
      recovered: 0,
      skipped: 0,
      errors: [],
    };
    let keys: string[];
    try {
      keys = await this.store.listKeys();
    } catch (error) {
      summary.errors.push(
        `listKeys: ${error instanceof Error ? error.message : String(error)}`
      );
      return summary;
    }
    const latest = new Map<
      string,
      { key: string; envelope: CheckpointEnvelope }
    >();
    const validKeys = new Set<string>();

    for (const key of keys) {
      summary.scanned += 1;
      try {
        const bytes = await this.store.get(key);
        if (!bytes) {
          summary.skipped += 1;
          continue;
        }
        const envelope = deserializeCheckpoint(bytes);
        this.validateRecoverableEnvelope(envelope);
        validKeys.add(key);
        const previous = latest.get(envelope.nodeId);
        if (!previous || envelope.createdAt > previous.envelope.createdAt) {
          latest.set(envelope.nodeId, { key, envelope });
        }
      } catch (error) {
        summary.skipped += 1;
        summary.errors.push(
          `${key}: ${error instanceof Error ? error.message : String(error)}`
        );
      }
    }

    for (const [nodeId, selected] of latest) {
      if (this.nodes.has(nodeId)) {
        continue;
      }
      const envelope = selected.envelope;
      const now = envelope.createdAt;
      const continuation = continuationMetadata(envelope.bytes);
      this.nodes.set(nodeId, {
        nodeId,
        session: { ...envelope.session },
        parentNodeId: envelope.parentNodeId ?? null,
        turnText: continuation.turnText,
        continuationText: continuation.continuationText,
        tokenIds: [...envelope.tokens],
        residentTokens: Math.max(
          envelope.tokens.length,
          continuation.residentTokens ?? 0
        ),
        createdAt: now,
        lastAccessAt: now,
        accessCount: 1,
      });
      this.residency.set(nodeId, {
        nodeId,
        level: 'L3',
        slotId: null,
        tokenCount: Math.max(
          envelope.tokens.length,
          continuation.residentTokens ?? 0
        ),
        checkpointKey: selected.key,
        lastAccessAt: now,
        accessCount: 1,
      });
      summary.recovered += 1;
    }

    // Keep one addressable checkpoint per node and remove stale generations.
    for (const key of keys) {
      const selected = [...latest.values()].some((entry) => entry.key === key);
      if (validKeys.has(key) && !selected) {
        await this.store.delete(key).catch(() => undefined);
      }
    }

    return summary;
  }

  registerLiveNode(node: SessionNode, slotId: number): ResidencyRecord {
    const record = this.registerNode(node, 'L1');
    const resident = this.requireResidency(node.nodeId);
    resident.level = 'L1';
    resident.slotId = slotId;
    resident.tokenCount = residentTokenCount(node);
    return { ...record, level: 'L1', slotId }; 
  }

  installL2Checkpoint(
    nodeId: string,
    checkpoint: NativeSlotCheckpoint
  ): void {
    const node = this.requireNode(nodeId);
    this.l2Checkpoints.set(nodeId, {
      ...checkpoint,
      bytes: new Uint8Array(checkpoint.bytes),
      tokens: [...checkpoint.tokens],
    });
    this.synchronizeNodeWithCheckpoint(nodeId, checkpoint);
    this.setResidency(nodeId, 'L2', null, residentTokenCount(node));
  }

  async admitNode(
    nodeId: string,
    targetLevel: ResidencyLevel = 'L1',
    repairBudget = this.config.repairBudget,
    // Nodes with native work in flight cannot be parked safely. Capacity
    // admission may leave L1 temporarily over budget; the caller releases
    // these nodes after their request reaches a completion boundary.
    pinnedNodeIds: ReadonlySet<string> = EMPTY_NODE_SET
  ): Promise<StateAdmissionResult> {
    const node = this.requireNode(nodeId);
    const current = this.requireResidency(nodeId);
    if (current.level === targetLevel) {
      return this.result(
        true,
        nodeId,
        targetLevel,
        0,
        [],
        false,
        current.level
      );
    }

    const victims: string[] = [];
    let steps = 0;
    const policy = this.config.admissionPolicy ?? 'monotonic-bounded';
    const effectiveBudget = this.effectiveRepairBudget(policy, repairBudget);
    let remainingBudget = effectiveBudget;
    const required = residentTokenCount(node);

    if (targetLevel === 'L1') {
      const maxResidentNodes =
        this.config.maxResidentNodes ?? Number.POSITIVE_INFINITY;
      while (
        this.l1Tokens() + required > this.config.l1TokenCap ||
        this.l1NodeCount() >= maxResidentNodes
      ) {
        if (remainingBudget <= 0) {
          return this.result(
            false,
            nodeId,
            targetLevel,
            steps,
            victims,
            true,
            current.level,
            'budget_exhausted'
          );
        }
        const tokenOverage =
          this.l1Tokens() + required > this.config.l1TokenCap;
        const victim = this.selectVictim(
          'L1',
          nodeId,
          true,
          pinnedNodeIds,
          // A zero-token resident can satisfy the slot-count bound, but it
          // cannot make progress when token capacity is the blocking bound.
          tokenOverage ? 1 : 0
        );
        if (!victim) {
          return this.result(
            false,
            nodeId,
            targetLevel,
            steps,
            victims,
            steps > 0,
            current.level,
            'protected_capacity'
          );
        }
        const beforeTokens = this.l1Tokens();
        const beforeNodes = this.l1NodeCount();
        await this.demote(victim.nodeId, 'L2');
        steps += 1;
        remainingBudget -= 1;
        victims.push(victim.nodeId);
        if (tokenOverage && this.l1Tokens() >= beforeTokens) {
          throw new CheckpointError(
            `state repair did not strictly reduce L1 tokens for ${victim.nodeId}`
          );
        }
        if (!tokenOverage && this.l1NodeCount() >= beforeNodes) {
          throw new CheckpointError(
            `state repair did not reduce L1 resident nodes for ${victim.nodeId}`
          );
        }
      }
    } else if (targetLevel === 'L2') {
      while (this.l2Tokens() + required > this.config.l2TokenCap) {
        if (remainingBudget <= 0) {
          return this.result(
            false,
            nodeId,
            targetLevel,
            steps,
            victims,
            true,
            current.level,
            'budget_exhausted'
          );
        }
        // L2 capacity is a physical wasm-heap bound. Lifecycle protection
        // can keep a session live in L1 longer, but it cannot make a parked
        // native state disappear; idle protected residents yield to L3 here.
        const victim = this.selectVictim('L2', nodeId, true, EMPTY_NODE_SET, 1);
        if (!victim) {
          return this.result(
            false,
            nodeId,
            targetLevel,
            steps,
            victims,
            steps > 0,
            current.level,
            'protected_capacity'
          );
        }
        const before = this.l2Tokens();
        await this.demote(victim.nodeId, 'L3');
        steps += 1;
        remainingBudget -= 1;
        victims.push(victim.nodeId);
        if (this.l2Tokens() >= before) {
          throw new CheckpointError(
            `state repair did not strictly reduce L2 tokens for ${victim.nodeId}`
          );
        }
      }
    } else if (targetLevel === 'L3') {
      const estimatedBytes = required * 4;
      while (this.l3Bytes() + estimatedBytes > this.config.l3ByteCap) {
        if (remainingBudget <= 0) {
          return this.result(
            false,
            nodeId,
            targetLevel,
            steps,
            victims,
            true,
            current.level,
            'budget_exhausted'
          );
        }
        const victim = this.selectVictim('L3', nodeId, false, EMPTY_NODE_SET, 1);
        if (!victim) {
          return this.result(
            false,
            nodeId,
            targetLevel,
            steps,
            victims,
            steps > 0,
            current.level,
            'protected_capacity'
          );
        }
        const before = this.l3Bytes();
        await this.deleteCheckpoint(victim.nodeId);
        await this.demote(victim.nodeId, 'rebuild');
        steps += 1;
        remainingBudget -= 1;
        victims.push(victim.nodeId);
        if (this.l3Bytes() >= before) {
          throw new CheckpointError(
            `state repair did not strictly reduce L3 bytes for ${victim.nodeId}`
          );
        }
      }
    }

    const source = current.level;
    if (source === 'L2' && targetLevel === 'L1') {
      const checkpoint = this.l2Checkpoints.get(nodeId);
      if (!checkpoint) {
        return this.rebuildResult(nodeId, victims, steps);
      }
      // A parked state can go stale (a released native handle, an eviction
      // that raced the resume). Rebuilding the prefix is a declared residency
      // level, so degrade into it instead of failing the request.
      try {
        const slot = await this.native.slotImport(checkpoint);
        this.l2Checkpoints.delete(nodeId);
        this.setResidency(nodeId, 'L1', slot.slotId, residentTokenCount(node));
      } catch (error) {
        await this.releaseL2Node(nodeId).catch(() => undefined);
        this.reportRestoreFailure(nodeId, 'L2', error);
        return this.rebuildResult(nodeId, victims, steps);
      }
    } else if (source === 'L3' && targetLevel === 'L1') {
      const envelope = await this.loadEnvelope(nodeId);
      try {
        const slot = await this.native.slotImport(envelope);
        this.setResidency(nodeId, 'L1', slot.slotId, residentTokenCount(node));
      } catch (error) {
        this.reportRestoreFailure(nodeId, 'L3', error);
        return this.rebuildResult(nodeId, victims, steps);
      }
    } else if (source === 'L3' && targetLevel === 'L2') {
      const envelope = await this.loadEnvelope(nodeId);
      // tier three to tier two: reload the serialized state into the native heap
      // and keep only the handle, so the host drops its copy of the bytes
      const parked = await this.native.slotParkCheckpoint({
        bytes: envelope.bytes,
        ...(envelope.kv ? { kv: new Uint8Array(envelope.kv) } : {}),
        tokens: envelope.tokens,
        stage: envelope.stage,
        decodeGenerated: envelope.decodeGenerated,
        formatVersion: envelope.formatVersion,
        checksum: envelope.checksum,
      });
      this.l2Checkpoints.set(nodeId, parked);
      this.setResidency(nodeId, 'L2', null, residentTokenCount(node));
      await this.enforceL2Capacity(nodeId);
    } else if (source === 'L1' && targetLevel === 'L2') {
      await this.demote(nodeId, 'L2');
    } else if (source === 'L1' && targetLevel === 'L3') {
      await this.demote(nodeId, 'L3');
    } else if (source === 'L2' && targetLevel === 'L3') {
      await this.demote(nodeId, 'L3');
    } else if (source === 'rebuild') {
      return this.rebuildResult(nodeId, victims, steps);
    } else {
      return this.result(
        false,
        nodeId,
        targetLevel,
        steps,
        victims,
        false,
        current.level,
        'checkpoint_mismatch'
      );
    }


    return this.result(
      true,
      nodeId,
      targetLevel,
      steps,
      victims,
      false,
      targetLevel
    );
  }

  async checkpointNode(nodeId: string): Promise<StateAdmissionResult> {
    const node = this.requireNode(nodeId);
    const current = this.requireResidency(nodeId);
    if (current.level !== 'L1' || current.slotId === null) {
      return this.result(
        false,
        nodeId,
        'L3',
        0,
        [],
        false,
        current.level,
        'checkpoint_mismatch'
      );
    }
    const checkpoint = await this.native.slotExport(current.slotId);
    this.synchronizeNodeWithCheckpoint(nodeId, checkpoint);
    const envelope = makeCheckpointEnvelope(checkpoint, {
      checkpointKey: `${nodeId}:${this.clock.now()}`,
      nodeId,
      session: node.session,
      parentNodeId: node.parentNodeId,
      modelHash: this.modelHashFor(node),
      createdAt: this.clock.now(),
    });
    await this.store.put(envelope.checkpointKey, serializeCheckpoint(envelope));
    await this.native.slotDrop(current.slotId);
    this.setResidency(
      nodeId,
      'L3',
      null,
      residentTokenCount(node),
      envelope.checkpointKey
    );
    this.telemetry?.emit(
      'state_restore',
      { nodeId, source: 'L1', target: 'L3', bytes: envelope.bytes.byteLength },
      { clientId: node.session.clientId, sessionId: node.session.sessionId }
    );
    return this.result(true, nodeId, 'L3', 0, [], false, 'L3');
  }

  /**
   * Import an immutable snapshot of a completed parent into a child slot.
   * The parent slot is copied, not consumed, so independent children can
   * share the same logical checkpoint while keeping separate native KV.
   */
  async forkNode(
    parentNodeId: string,
    childNodeId: string,
    repairBudget = this.config.repairBudget
  ): Promise<StateAdmissionResult> {
    const parent = this.requireNode(parentNodeId);
    const child = this.requireNode(childNodeId);
    if (child.parentNodeId !== parentNodeId) {
      return this.result(
        false,
        childNodeId,
        'L1',
        0,
        [],
        false,
        this.requireResidency(childNodeId).level,
        'checkpoint_mismatch'
      );
    }

    const parentAdmission = await this.admitNode(
      parentNodeId,
      'L1',
      repairBudget
    );
    if (!parentAdmission.success) {
      return {
        ...parentAdmission,
        nodeId: childNodeId,
      };
    }
    const parentResidency = this.requireResidency(parentNodeId);
    if (parentResidency.slotId === null) {
      return this.result(
        false,
        childNodeId,
        'L1',
        0,
        [...parentAdmission.victims],
        false,
        this.requireResidency(childNodeId).level,
        'checkpoint_mismatch'
      );
    }

    // Snapshot before touching the child so a capacity failure cannot
    // consume or invalidate the parent checkpoint.
    const snapshot = await this.snapshotForkParent(
      parent,
      parentResidency.slotId
    );
    const childResidency = this.requireResidency(childNodeId);
    if (childResidency.level === 'L1' && childResidency.slotId !== null) {
      await this.native.slotDrop(childResidency.slotId).catch(() => undefined);
    }
    await this.releaseL2Node(childNodeId);
    if (childResidency.checkpointKey) {
      await this.deleteCheckpoint(childNodeId);
    }

    const imported = await this.native.slotImport(snapshot);
    const continuation = continuationMetadata(snapshot.bytes);
    child.tokenIds = [...snapshot.tokens];
    child.continuationText = parent.continuationText ?? parent.turnText;
    child.residentTokens = Math.max(
      snapshot.tokens.length,
      continuation.residentTokens ?? snapshot.tokens.length
    );
    child.lastAccessAt = this.clock.now();
    this.setResidency(
      childNodeId,
      'L1',
      imported.slotId,
      residentTokenCount(child)
    );
    this.telemetry?.emit(
      'state_restore',
      {
        nodeId: childNodeId,
        parentNodeId,
        source: 'parent-checkpoint',
        target: 'L1',
        tokens: snapshot.tokens.length,
      },
      {
        clientId: child.session.clientId,
        sessionId: child.session.sessionId,
      }
    );
    return this.result(
      true,
      childNodeId,
      'L1',
      parentAdmission.steps,
      [...parentAdmission.victims],
      false,
      'L1'
    );
  }

  private async snapshotForkParent(
    parent: SessionNode,
    slotId: number
  ): Promise<NativeSlotCheckpoint> {
    const continuationText = parent.continuationText ?? parent.turnText;
    const cached = this.forkSnapshot;
    if (
      cached !== null &&
      cached.parentNodeId === parent.nodeId &&
      cached.slotId === slotId &&
      cached.continuationText === continuationText &&
      cached.tokenIds.length === parent.tokenIds.length &&
      cached.tokenIds.every((token, index) => token === parent.tokenIds[index])
    ) {
      return cached.checkpoint;
    }
    const checkpoint = await this.native.slotSnapshot(slotId);
    this.forkSnapshot = {
      parentNodeId: parent.nodeId,
      slotId,
      tokenIds: [...parent.tokenIds],
      continuationText,
      checkpoint,
    };
    return checkpoint;
  }

  async demote(
    nodeId: string,
    targetLevel: Exclude<ResidencyLevel, 'L1'>
  ): Promise<void> {
    const node = this.requireNode(nodeId);
    const current = this.requireResidency(nodeId);
    if (current.level === targetLevel) {
      return;
    }
    if (current.level === 'L1' && current.slotId !== null) {
      // A priority cut can land after host submit but before llama.cpp has
      // installed the prompt. That slot has no checkpoint to park or export,
      // so preserve correctness by rebuilding it from the logical history.
      const parkable =
        targetLevel === 'rebuild' ||
        (await this.nativeSlotParkable(current.slotId));
      if (!parkable) {
        await this.dropLiveNode(nodeId);
      } else {
        try {
          if (targetLevel === 'L2') {
            // tier two: the state stays in the native heap behind a handle and the
            // live slot is released, so the coordinator never copies KV bytes
            const checkpoint = await this.native.slotPark(current.slotId);
            this.l2Checkpoints.set(nodeId, checkpoint);
            this.synchronizeNodeWithCheckpoint(nodeId, checkpoint);
            this.setResidency(nodeId, 'L2', null, residentTokenCount(node));
            await this.enforceL2Capacity(nodeId);
          } else if (targetLevel === 'L3') {
            const checkpoint = await this.native.slotExport(current.slotId);
            this.synchronizeNodeWithCheckpoint(nodeId, checkpoint);
            const envelope = makeCheckpointEnvelope(checkpoint, {
              checkpointKey: `${nodeId}:${this.clock.now()}`,
              nodeId,
              session: node.session,
              parentNodeId: node.parentNodeId,
              modelHash: this.modelHashFor(node),
              createdAt: this.clock.now(),
            });
            await this.store.put(
              envelope.checkpointKey,
              serializeCheckpoint(envelope)
            );
            await this.native.slotDrop(current.slotId);
            this.setResidency(
              nodeId,
              'L3',
              null,
              residentTokenCount(node),
              envelope.checkpointKey
            );
          } else {
            await this.native.slotDrop(current.slotId);
            this.setResidency(nodeId, targetLevel, null, residentTokenCount(node));
          }
        } catch (error) {
          if (!isNativeSlotStateEmpty(error)) {
            throw error;
          }
          await this.dropLiveNode(nodeId);
        }
      }
    } else if (current.level === 'L2' && targetLevel === 'L3') {
      const checkpoint = this.l2Checkpoints.get(nodeId);
      if (!checkpoint) {
        this.setResidency(nodeId, 'rebuild', null, residentTokenCount(node));
      } else {
        // tier three: serialize out of the native heap, persist, then release
        // the native copy so the state exists only as the stored blob
        const serialized = await this.native.slotReadParked(checkpoint);
        this.synchronizeNodeWithCheckpoint(nodeId, serialized);
        const envelope = makeCheckpointEnvelope(serialized, {
          checkpointKey: `${nodeId}:${this.clock.now()}`,
          nodeId,
          session: node.session,
          parentNodeId: node.parentNodeId,
          modelHash: this.modelHashFor(node),
          createdAt: this.clock.now(),
        });
        await this.store.put(
          envelope.checkpointKey,
          serializeCheckpoint(envelope)
        );
        await this.native.slotReleaseParked(checkpoint);
        this.l2Checkpoints.delete(nodeId);
        this.setResidency(
          nodeId,
          'L3',
          null,
          residentTokenCount(node),
          envelope.checkpointKey
        );
      }
    } else if (targetLevel === 'rebuild') {
      await this.releaseL2Node(nodeId);
      this.setResidency(nodeId, 'rebuild', null, residentTokenCount(node));
    }
    const updated = this.requireResidency(nodeId);
    this.telemetry?.emit(
      'state_residency',
      { nodeId, level: updated.level, tokens: updated.tokenCount },
      { clientId: node.session.clientId, sessionId: node.session.sessionId }
    );
  }

  // the backend released the live slot without giving us a checkpoint
  private async nativeSlotParkable(slotId: number): Promise<boolean> {
    const stats = await this.native.slotStats().catch(() => []);
    const slot = stats.find((entry) => entry.slotId === slotId);
    return slot?.inUse === true && slot.parkable !== false;
  }

  async dropLiveNode(nodeId: string): Promise<void> {
    const record = this.residency.get(nodeId);
    if (!record) {
      return;
    }
    if (record.level === 'L1' && record.slotId !== null) {
      await this.native.slotDrop(record.slotId).catch(() => undefined);
    }
    await this.releaseL2Node(nodeId);
    this.setResidency(nodeId, 'rebuild', null, record.tokenCount);
  }

  // free a native parked state when its node leaves tier two
  private async releaseL2Node(nodeId: string): Promise<void> {
    const checkpoint = this.l2Checkpoints.get(nodeId);
    this.l2Checkpoints.delete(nodeId);
    if (checkpoint?.parkedHandle !== undefined) {
      await this.native.slotReleaseParked(checkpoint);
    }
  }

  async restore(
    nodeId: string,
    repairBudget = this.config.repairBudget
  ): Promise<RestoreResult> {
    const node = this.requireNode(nodeId);
    const current = this.requireResidency(nodeId);
    const source = current.level;
    if (source === 'rebuild') {
      return this.rebuildResult(nodeId, [], 0);
    }
    if (source === 'L1') {
      const updated = this.requireResidency(nodeId);
      updated.lastAccessAt = this.clock.now();
      updated.accessCount += 1;
      return {
        ...this.result(true, nodeId, 'L1', 0, [], false, 'L1'),
        source,
        replayedTokens: 0,
      };
    }
    if (source === 'L2' && !this.l2Checkpoints.has(nodeId)) {
      return this.rebuildResult(nodeId, [], 0);
    }
    const admission = await this.admitNode(nodeId, 'L1', repairBudget);
    if (!admission.success) {
      return {
        ...admission,
        source,
        replayedTokens: admission.partial ? residentTokenCount(node) : 0,
      };
    }
    const updated = this.requireResidency(nodeId);
    updated.lastAccessAt = this.clock.now();
    updated.accessCount += 1;
    this.telemetry?.emit(
      'state_restore',
      { nodeId, source, target: 'L1', tokens: residentTokenCount(node) },
      { clientId: node.session.clientId, sessionId: node.session.sessionId }
    );
    return {
      ...admission,
      source,
      replayedTokens: 0,
    };
  }

  async deleteCheckpoint(nodeId: string): Promise<void> {
    const record = this.residency.get(nodeId);
    if (record?.checkpointKey) {
      await this.store.delete(record.checkpointKey);
      record.checkpointKey = null;
    }
  }

  // bytes held in the in-memory (L2) buffer for a node, when it is parked there
  bufferBytes(nodeId: string): number {
    const checkpoint = this.l2Checkpoints.get(nodeId);
    if (!checkpoint) {
      return 0;
    }
    return (
      checkpoint.kv?.byteLength ??
      checkpoint.parkedByteLength ??
      checkpoint.bytes.byteLength
    );
  }

  // host-side copy of a tier-two node's state. Tier two parks the state in the
  // native heap, so this is zero unless a backend without a park bank falls
  // back to exporting the bytes.
  hostBufferBytes(nodeId: string): number {
    return this.l2Checkpoints.get(nodeId)?.kv?.byteLength ?? 0;
  }

  get totals(): { l1Tokens: number; l2Tokens: number; l3Bytes: number } {
    return {
      l1Tokens: this.l1Tokens(),
      l2Tokens: this.l2Tokens(),
      l3Bytes: this.l3Bytes(),
    };
  }

  get residentNodes(): number {
    return this.l1NodeCount();
  }

  get trackedNodes(): number {
    return this.residency.size;
  }

  get parkedNodes(): number {
    return [...this.residency.values()].filter((record) => record.level === 'L2')
      .length;
  }

  get parkedBytes(): number {
    return [...this.residency.keys()].reduce(
      (sum, nodeId) => sum + this.bufferBytes(nodeId),
      0
    );
  }

  private async loadEnvelope(nodeId: string): Promise<CheckpointEnvelope> {
    const record = this.requireResidency(nodeId);
    if (!record.checkpointKey) {
      throw new CheckpointError(`missing checkpoint key for ${nodeId}`);
    }
    const bytes = await this.store.get(record.checkpointKey);
    if (!bytes) {
      throw new CheckpointError(`checkpoint not found for ${nodeId}`);
    }
    const envelope = deserializeCheckpoint(bytes);
    if (envelope.nodeId !== nodeId) {
      throw new CheckpointError('checkpoint node mismatch');
    }
    this.validateEnvelope(envelope);
    return envelope;
  }

  private validateEnvelope(envelope: CheckpointEnvelope): void {
    if (envelope.formatVersion !== 1 && envelope.formatVersion !== 2) {
      throw new CheckpointError(
        `unsupported checkpoint version: ${envelope.formatVersion}`
      );
    }
    const expected = checkpointChecksum(envelope);
    if (expected !== envelope.checksum) {
      throw new CheckpointError('checkpoint checksum mismatch');
    }
    if (envelope.modelHash !== this.modelHashForNode(envelope.nodeId)) {
      throw new CheckpointError('checkpoint model hash mismatch');
    }
    const node = this.requireNode(envelope.nodeId);
    if (
      envelope.tokens.length > node.tokenIds.length ||
      envelope.tokens.some((token, index) => token !== node.tokenIds[index])
    ) {
      throw new CheckpointError('checkpoint token prefix mismatch');
    }
  }

  private validateRecoverableEnvelope(envelope: CheckpointEnvelope): void {
    if (envelope.formatVersion !== 1 && envelope.formatVersion !== 2) {
      throw new CheckpointError('unsupported checkpoint version');
    }
    if (envelope.nodeId.length === 0 || envelope.tokens.length === 0) {
      throw new CheckpointError('checkpoint has no addressable node state');
    }
    if (!envelope.session.clientId || !envelope.session.sessionId) {
      throw new CheckpointError('checkpoint has no session identity');
    }
    if (checkpointChecksum(envelope) !== envelope.checksum) {
      throw new CheckpointError('checkpoint checksum mismatch');
    }
    if (envelope.modelHash !== this.modelHash) {
      throw new CheckpointError('checkpoint model hash mismatch');
    }
  }

  private synchronizeNodeWithCheckpoint(
    nodeId: string,
    checkpoint: NativeSlotCheckpoint
  ): void {
    const node = this.requireNode(nodeId);
    const continuation = continuationMetadata(checkpoint.bytes);
    const reportedResidentTokens = continuation.residentTokens;
    // The backend tokenizer is authoritative for checkpoint tokens. Keeping
    // this prefix synchronized makes a later turn append instead of replacing
    // the recovered prefix.
    const previousResidentTokens = residentTokenCount(node);
    node.tokenIds = [...checkpoint.tokens];
    if (continuation.continuationText) {
      node.continuationText = continuation.continuationText;
    }
    node.residentTokens = Math.max(
      checkpoint.tokens.length,
      reportedResidentTokens ?? previousResidentTokens
    );
  }

  private modelHashFor(node: SessionNode): string {
    return this.modelHashForNode(node.nodeId);
  }

  private modelHashForNode(nodeId: string): string {
    void nodeId;
    return this.modelHash;
  }


  private async enforceL2Capacity(excludeNodeId: string): Promise<void> {
    while (this.l2Tokens() > this.config.l2TokenCap) {
      const victim = this.selectVictim('L2', excludeNodeId, true, EMPTY_NODE_SET, 1);
      if (!victim) {
        throw new CheckpointError(
          `L2 capacity ${this.config.l2TokenCap} cannot admit a new parked node`
        );
      }
      const before = this.l2Tokens();
      await this.demote(victim.nodeId, 'L3');
      if (this.l2Tokens() >= before) {
        throw new CheckpointError(
          `state repair did not strictly reduce L2 tokens for ${victim.nodeId}`
        );
      }
    }
  }

  private selectVictim(
    level: ResidencyLevel,
    excludeNodeId: string,
    allowProtectedFallback = false,
    pinnedNodeIds: ReadonlySet<string> = EMPTY_NODE_SET,
    minimumTokenCount = 0
  ): ResidencyRecord | undefined {
    const eligible = (record: ResidencyRecord): boolean =>
      record.level === level &&
      record.nodeId !== excludeNodeId &&
      !pinnedNodeIds.has(record.nodeId) &&
      record.tokenCount >= minimumTokenCount;
    const candidates = [...this.residency.values()]
      .filter((record) => eligible(record) && !this.isProtected(record.nodeId))
      .sort((a, b) => this.victimScore(a) - this.victimScore(b));
    if (candidates[0]) {
      return candidates[0];
    }
    if (!allowProtectedFallback) {
      return undefined;
    }
    // Lifecycle protection may yield to capacity. Native work in flight may
    // not: parking an incomplete request is the one eviction the backend
    // cannot complete safely.
    return [...this.residency.values()]
      .filter(eligible)
      .sort((a, b) => this.victimScore(a) - this.victimScore(b))[0];
  }

  private victimScore(record: ResidencyRecord): number {
    if (this.config.replacementPolicy === 'lru') {
      return record.lastAccessAt - record.accessCount * 10;
    }
    if (this.config.replacementPolicy === 'ttl') {
      return record.lastAccessAt;
    }
    return (
      record.lastAccessAt -
      record.accessCount * 100 -
      record.tokenCount * 0.5
    );
  }

  private effectiveRepairBudget(
    policy: NonNullable<StateManagerConfig['admissionPolicy']>,
    requested: number
  ): number {
    const normalized = Math.max(0, Math.floor(requested));
    if (policy === 'immediate-reject') {
      return 0;
    }
    if (policy === 'reactive') {
      return Math.min(1, normalized);
    }
    return normalized;
  }

  private isProtected(nodeId: string): boolean {
    const node = this.nodes.get(nodeId);
    return node ? this.protectedSessions.has(sessionKeyString(node.session)) : false;
  }

  private setResidency(
    nodeId: string,
    level: ResidencyLevel,
    slotId: number | null,
    tokenCount: number,
    checkpointKey: string | null = null
  ): void {
    const record = this.requireResidency(nodeId);
    record.level = level;
    record.slotId = slotId;
    record.tokenCount = tokenCount;
    record.checkpointKey = checkpointKey;
  }

  private result(
    success: boolean,
    nodeId: string,
    targetLevel: ResidencyLevel,
    steps: number,
    victims: string[],
    partial: boolean,
    residentLevel: ResidencyLevel,
    reason?: StateAdmissionResult['reason']
  ): StateAdmissionResult {
    return {
      success,
      nodeId,
      targetLevel,
      steps,
      victims: [...victims],
      partial,
      residentLevel,
      ...(reason ? { reason } : {}),
    };
  }

  // A restore that could not reinstall its saved state is still progress: the
  // node degrades to a rebuild, which the harness measures as its own resume
  // source. Surface the cause so a silent latency or quality shift is traceable.
  private reportRestoreFailure(
    nodeId: string,
    source: ResidencyLevel,
    error: unknown
  ): void {
    const detail = error instanceof Error ? error.message : String(error);
    const session = this.nodes.get(nodeId)?.session;
    this.telemetry?.emit(
      'state_restore_failed',
      { nodeId, source, detail },
      session === undefined
        ? {}
        : { clientId: session.clientId, sessionId: session.sessionId }
    );
  }

  private rebuildResult(
    nodeId: string,
    victims: string[],
    steps: number
  ): RestoreResult {
    const node = this.requireNode(nodeId);
    this.setResidency(nodeId, 'rebuild', null, residentTokenCount(node));
    return {
      success: true,
      nodeId,
      targetLevel: 'rebuild',
      steps,
      victims: [...victims],
      partial: false,
      residentLevel: 'rebuild',
      source: 'rebuild',
      replayedTokens: residentTokenCount(node),
    };
  }

  private requireNode(nodeId: string): SessionNode {
    const node = this.nodes.get(nodeId);
    if (!node) {
      throw new CheckpointError(`unknown session node: ${nodeId}`);
    }
    return node;
  }

  private requireResidency(nodeId: string): ResidencyRecord {
    const record = this.residency.get(nodeId);
    if (!record) {
      throw new CheckpointError(`unknown residency record: ${nodeId}`);
    }
    return record;
  }

  private l1Tokens(): number {
    return [...this.residency.values()]
      .filter((record) => record.level === 'L1')
      .reduce((sum, record) => sum + record.tokenCount, 0);
  }

  private l1NodeCount(): number {
    return [...this.residency.values()].filter(
      (record) => record.level === 'L1'
    ).length;
  }

  private l2Tokens(): number {

    return [...this.residency.values()]
      .filter((record) => record.level === 'L2')
      .reduce((sum, record) => sum + record.tokenCount, 0);
  }

  private l3Bytes(): number {
    return [...this.residency.values()]
      .filter((record) => record.level === 'L3')
      .reduce((sum, record) => sum + record.tokenCount * 4, 0);
  }
}

function isNativeSlotStateEmpty(error: unknown): boolean {
  return (
    (error instanceof WllmError && error.code === 'slot_state_empty') ||
    (error instanceof Error && /\(status 3\)/.test(error.message))
  );
}

function residentTokenCount(node: SessionNode): number {
  return Math.max(0, node.residentTokens ?? node.tokenIds.length);
}

function continuationMetadata(bytes: Uint8Array): {
  turnText: string;
  continuationText: string;
  residentTokens: number | null;
} {
  try {
    const parsed = JSON.parse(new TextDecoder().decode(bytes)) as {
      promptText?: unknown;
      generatedText?: unknown;
      prefillProcessed?: unknown;
    };
    return {
      turnText: typeof parsed.promptText === 'string' ? parsed.promptText : '',
      continuationText:
        typeof parsed.promptText === 'string'
          ? parsed.promptText +
            (typeof parsed.generatedText === 'string'
              ? parsed.generatedText
              : '')
          : '',
      residentTokens:
        typeof parsed.prefillProcessed === 'number' &&
        Number.isFinite(parsed.prefillProcessed) &&
        parsed.prefillProcessed >= 0
          ? Math.floor(parsed.prefillProcessed)
          : null,
    };
  } catch {
    return { turnText: '', continuationText: '', residentTokens: null };
  }
}

function cloneNode(node: SessionNode): SessionNode {
  return {
    ...node,
    session: { ...node.session },
    tokenIds: [...node.tokenIds],
  };
}
