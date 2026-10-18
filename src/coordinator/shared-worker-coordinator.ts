import type {
  ClientId,
  ExecutionConfig,
  ExecutionOutcome,
  GpuBurstPolicy,
  LifecycleRecord,
  ExecutionLease,
  GpuBurstLevel,
  ExecutionMode,
  LeaseDiagnostics,
  LeaseUnitDiagnostics,
  NodeId,
  NativeUnitProgress,
  NativeTokenChunk,
  RenderPressureSignal,
  RequestId,
  RequestRecord,
  SchedulerConfig,
  SessionKey,
  SessionNode,
  StateManagerConfig,
  StateAdmissionResult,
} from '../contracts';
import { RequestFencedError } from '../contracts';
import type {
  CoordinatorDiagnostics,
  CoordinatorEvent,
  StateTransitionEvent,
} from '../transport/protocol';
import { PriorityAwareScheduler } from '../scheduler/priority-aware-scheduler';
import { MultiTierStateManager } from '../state/multi-tier-state-manager';
import type { CheckpointRecoverySummary } from '../state/multi-tier-state-manager';
import { BoundedExecutionController } from '../execution/bounded-execution-controller';
import {
  GPU_BURST_POLICIES,
  RenderAwareBurstController,
  unitSizingForLevel,
  type RenderAwareBurstControllerOptions,
} from '../execution/render-aware-burst-controller';
import {
  FrameDeadlineLease,
  type FrameDeadlineLeaseOptions,
} from '../execution/frame-deadline-lease';
import { InMemoryTelemetry, type TelemetrySink } from '../telemetry';
import type { NativeEngine } from '../native/native-engine';
import { trimPrefixToTokenBudget } from '../native/context-window';
import type { BlobStore, Clock } from '../contracts';
import { systemClock } from '../contracts';

export type CompletedTier = 'L1' | 'L2' | 'L3' | 'rebuild';

export interface CoordinatorOptions {
  scheduler?: Partial<SchedulerConfig>;
  // Diagnostics-only provenance of the native runtime. It does not affect
  // scheduling, but lets an experiment prove that inference is not coupled
  // to the lifecycle of one application tab.
  nativeWorkerHost?: 'shared-worker' | 'page';
  coordinatorInstanceId?: string;
  state?: Partial<StateManagerConfig>;
  execution?: Partial<ExecutionConfig>;
  // Maximum text a session may carry into its next turn. When set, the
  // continuation a rebuilt slot replays is trimmed to this many tokens and
  // a resident slot is not resumed for a fresh turn, so `0` makes every turn
  // a self contained request. The evaluation harness can send one stateless
  // turn at a time, so replaying accumulated turns there only costs a
  // full re-prefill of history the model never asked for. Undefined keeps the
  // demo's accumulate-and-replay behaviour.
  continuationTokenCap?: number;
  // demo/experiment switch: keep completed sessions in L1 when false
  parkCompleted?: boolean;
  // explicit target for completed turns; overrides parkCompleted when set
  completedTier?: CompletedTier;
  // identity of the loaded model, used to reject stale L3 checkpoints
  modelHash?: string;
  telemetry?: TelemetrySink;
  clock?: Clock;
  macrotaskYield?: () => Promise<void>;
  renderPause?: (pauseMs: number) => Promise<void>;
  // experiment control: when false, render-pressure feedback is ignored
  renderAware?: boolean;
  // experiment control: hold new units while pressure sits at its worst level.
  // Work already admitted finishes; nothing new is submitted until it clears.
  renderGate?: boolean;
  // A gate that can stay shut indefinitely would starve inference when the
  // pressure signal is biased. After this hold it reopens for a minimum
  // window, so gated serving keeps a bounded duty cycle instead of stalling.
  renderGateMaxHoldMs?: number;
  renderGateMinOpenMs?: number;
  // experiment control: level thresholds and the optional pacing fallback
  burst?: Partial<RenderAwareBurstControllerOptions>;
  // Experiment switch: which controller shapes one cooperative unit.
  // Defaults to `bounded`, so existing deployments keep the fixed shape.
  executionMode?: ExecutionMode;
  // Tuning for the deadline lease; ignored by the fixed modes.
  lease?: Partial<FrameDeadlineLeaseOptions>;
  // Number of same-class requests admitted into one deadline graph
  // admission. 1 preserves the strictly serial scheduler; values above one
  // let the native server continuous-batch independent sequences.
  leaseBatchSize?: number;
  // An unanswered promotion may share one graph admission with ordinary
  // work. The promotion still returns its one-token response, but the graph
  // does not become a lone low-throughput unit that delays every other
  // session.
  batchHandoff?: boolean;
  // Total requests in a graph whose head owes an unanswered promotion,
  // including the hand-off itself. Smaller graphs answer the promotion
  // sooner; larger graphs amortise the weight read over more work.
  handoffBatchSize?: number;
  // Advance a promoted session's next token in a separate native unit as
  // soon as the priority update arrives, instead of waiting for the ordinary
  // work-conserving graph to end. The native runtime may continuous-batch
  // the express unit together with the graph already in flight.
  expressHandoff?: boolean;
  // Cut the current work-conserving unit at its next safe token boundary
  // when a session is promoted. The interrupted requests keep their KV and
  // are requeued; the promoted session is selected at the new boundary.
  preemptOnActivation?: boolean;
  // Minimum prompt tokens a hand-off admission may evaluate, even when the
  // frame lease has fallen back to its smallest pressure shape.
  handoffPrefillTokens?: number;
  // High-priority graph isolation: park idle non-priority sequences at the
  // native graph boundary so their resident slots cannot join or queue ahead
  // of the priority unit. Parked KV and continuation state are retained for
  // the next normal unit.
  priorityNativeIsolation?: boolean;
  // E4 retention switch: a hidden tab parks its completed live state at the
  // lifecycle boundary instead of waiting for unrelated capacity pressure.
  evictOnPause?: boolean;
}

interface RequestFence {
  clientId: ClientId;
  generation: number;
}

interface ResidencyEventDetail {
  action?: StateTransitionEvent['action'];
  previousLevel?: StateTransitionEvent['previousLevel'];
  admission?: StateAdmissionResult;
}

// A request can only be advanced once its node owns a live slot; otherwise it
// waits for capacity, or is refused when the configured policy says so.
type ResidentAdmission = 'ready' | 'deferred' | 'rejected';

export class SharedWorkerCoordinator {
  readonly scheduler: PriorityAwareScheduler;
  readonly stateManager: MultiTierStateManager;
  readonly execution: BoundedExecutionController;
  readonly telemetry: TelemetrySink;
  readonly clock: Clock;
  private readonly fences = new Map<ClientId, number>();
  private readonly requestFences = new Map<RequestId, RequestFence>();
  private readonly attachedClients = new Set<ClientId>();
  private readonly eventListeners = new Set<
    (event: CoordinatorEvent) => void
  >();
  private inFlight = false;
  private engineSequence = 0;
  private lastError: string | null = null;
  private readonly requestNodes = new Map<RequestId, NodeId>();
  private readonly lastRequestByNode = new Map<NodeId, RequestRecord>();
  private readonly stateHistory: StateTransitionEvent[] = [];
  private stateSequence = 0;
  private completedTier: CompletedTier;
  private readonly continuationTokenCap: number | undefined;
  private readonly burstController: RenderAwareBurstController;
  private renderAwareEnabled: boolean;
  private readonly renderGateEnabled: boolean;
  private renderGateClosed = false;
  private renderGateClosedSince = 0;
  private renderGateOpenUntil = 0;
  private readonly renderGateMaxHoldMs: number;
  private readonly renderGateMinOpenMs: number;
  private readonly baseExecutionConfig: ExecutionConfig;
  private executionMode: ExecutionMode;
  private leaseController: FrameDeadlineLease | null;
  private readonly leaseOptions: Partial<FrameDeadlineLeaseOptions>;
  private readonly leaseBatchSize: number;
  private readonly batchHandoff: boolean;
  private readonly handoffBatchSize: number;
  private readonly expressHandoff: boolean;
  private readonly preemptOnActivation: boolean;
  private readonly handoffPrefillTokens: number;
  private readonly evictOnPause: boolean;
  private readonly priorityNativeIsolation: boolean;
  private expressDraining = false;
  private expressQueued = false;
  private readonly expressInFlight = new Set<RequestId>();
  private readonly activeUnitRequestIds = new Set<RequestId>();
  private readonly batchPinnedRequestIds = new Set<RequestId>();
  private readonly pendingActivationPreempts = new Set<ClientId>();
  private readonly interruptedActivationRequestIds = new Set<RequestId>();
  private activeLease: ExecutionLease | null = null;
  private lastLeaseUnit: LeaseUnitDiagnostics | null = null;
  private lastNativeUnit: CoordinatorDiagnostics['runtime']['lastNativeUnit'] =
    null;
  // Bounded history of completed units. The deadline screen reads it at the
  // end of an episode to separate native compute from coordinator gaps.
  private readonly recentLeaseUnits: LeaseUnitDiagnostics[] = [];
  private lastLeaseUnitEndAt = 0;
  private desiredBurstPolicy: GpuBurstPolicy = GPU_BURST_POLICIES[0]!;
  private appliedBurstPolicy: GpuBurstPolicy | null = null;
  private lastPressureClientId: ClientId | null = null;
  private readonly renderPause: (pauseMs: number) => Promise<void>;
  private readonly recovery: Promise<CheckpointRecoverySummary>;
  private readonly nativeWorkerHost: 'shared-worker' | 'page' | undefined;
  private readonly coordinatorInstanceId: string | undefined;

  constructor(

    private readonly native: NativeEngine,
    store: BlobStore,
    options: CoordinatorOptions = {}
  ) {
    this.clock = options.clock ?? systemClock;
    this.renderAwareEnabled = options.renderAware ?? true;
    this.nativeWorkerHost = options.nativeWorkerHost;
    this.coordinatorInstanceId = options.coordinatorInstanceId;
    this.renderGateEnabled = options.renderGate ?? false;
    this.renderGateMaxHoldMs = options.renderGateMaxHoldMs ?? 1_500;
    this.renderGateMinOpenMs = options.renderGateMinOpenMs ?? 300;
    this.burstController = new RenderAwareBurstController(options.burst);
    this.executionMode = options.executionMode ?? 'bounded';
    this.leaseOptions = { ...options.lease };
    this.leaseBatchSize = clampInteger(options.leaseBatchSize ?? 2, 1, 8);
    this.batchHandoff = options.batchHandoff ?? false;
    this.handoffBatchSize = clampInteger(
      options.handoffBatchSize ?? 2,
      1,
      this.leaseBatchSize
    );
    this.expressHandoff = options.expressHandoff ?? false;
    this.preemptOnActivation = options.preemptOnActivation ?? false;
    this.handoffPrefillTokens = clampInteger(
      options.handoffPrefillTokens ?? 64,
      1,
      4_096
    );
    this.evictOnPause = options.evictOnPause ?? false;
    this.priorityNativeIsolation =
      options.priorityNativeIsolation ?? true;
    this.leaseController = null;
    this.renderPause = options.renderPause ?? defaultRenderPause;
    this.completedTier =
      options.completedTier ?? (options.parkCompleted === false ? 'L1' : 'L2');
    this.continuationTokenCap = normalizeContinuationTokenCap(
      options.continuationTokenCap
    );
    this.scheduler = new PriorityAwareScheduler(options.scheduler);
    this.telemetry = options.telemetry ?? new InMemoryTelemetry(this.clock);
    this.stateManager = new MultiTierStateManager(
      native,
      store,
      options.state,
      this.clock,
      this.telemetry
    );
    if (options.modelHash) {
      this.stateManager.setModelHash(options.modelHash);
    }
    this.recovery = this.stateManager
      .recoverFromStore()
      .then((summary) => {
        this.telemetry.emit('state_restore', {
          source: 'L3-store',
          target: 'registry',
          scanned: summary.scanned,
          recovered: summary.recovered,
          skipped: summary.skipped,
          errors: summary.errors,
        });
        return summary;
      })
      .catch((error: unknown) => {
        const summary: CheckpointRecoverySummary = {
          scanned: 0,
          recovered: 0,
          skipped: 1,
          errors: [error instanceof Error ? error.message : String(error)],
        };
        this.telemetry.emit('state_restore', {
          source: 'L3-store',
          target: 'registry',
          error: summary.errors[0],
        });
        return summary;
      });
    this.execution = new BoundedExecutionController(
      native,
      options.execution,
      options.macrotaskYield,
      {
        onNativeSubmit: () => {
          this.flushPendingActivationPreempts();
        },
        onBoundary: async (request, progress) => {
          this.telemetry.emit(
            'unit_end',
            {
              stage: progress.stage,
              prefillProcessed: progress.prefillProcessed,
              decodeGenerated: progress.decodeGenerated,
              elapsedMs: progress.elapsedMs,
            },
            this.contextFor(request)
          );
        },
      }
    );
    // Snapshot the effective config rather than the raw overrides, so a level
    // change back to `normal` restores values the deployment never set.
    this.baseExecutionConfig = { ...this.execution.config };
    if (this.executionMode === 'deadline') {
      this.leaseController = this.createLeaseController();
    }
  }

  configureState(
    config: Partial<StateManagerConfig>,
    parkCompleted?: boolean,
    completedTier?: CompletedTier
  ): void {
    if (completedTier !== undefined) {
      this.completedTier = completedTier;
    } else if (parkCompleted !== undefined) {
      this.completedTier = parkCompleted ? 'L2' : 'L1';
    }
    this.stateManager.configure(config);
  }

  // Called when the page-owned native worker disappears. Slot ids and L2
  // parked handles are address-space local; retaining them would make the
  // replacement worker read another process's dead state.
  handleNativeWorkerLoss(reason: string): void {
    this.stateManager.invalidateNativeState();
    const requests = new Map<RequestId, RequestRecord>();
    for (const request of this.scheduler.pendingRequests()) {
      requests.set(request.requestId, request);
    }
    for (const request of this.lastRequestByNode.values()) {
      requests.set(request.requestId, request);
    }
    for (const request of requests.values()) {
      request.nativeRequestId = null;
      request.slotId = null;
      request.resumeSlotId = null;
    }
    this.activeLease = null;
    this.lastLeaseUnit = null;
    this.lastNativeUnit = null;
    this.lastError = reason;
    this.leaseController?.reset();
    this.burstController.reset();
    this.telemetry.emit('native_worker_loss', { reason });
  }
  whenReady(): Promise<CheckpointRecoverySummary> {
    return this.recovery;
  }

  async diagnostics(): Promise<CoordinatorDiagnostics> {
    const recovery = await this.recovery;
    return {
      completedTier: this.completedTier,
      runtime: {
        nativeBound: this.native.bound ?? true,
        inFlight: this.inFlight,
        ...(this.nativeWorkerHost === undefined
          ? {}
          : { nativeWorkerHost: this.nativeWorkerHost }),
        ...(this.coordinatorInstanceId === undefined
          ? {}
          : { coordinatorInstanceId: this.coordinatorInstanceId }),
        activeRequestId: this.scheduler.activeId,
        pendingRequests: this.scheduler.pendingCount,
        pendingForeground: this.scheduler.hasPendingHighPriority(),
        renderAware: this.renderAwareEnabled,
        gpuBurstPolicy: { ...this.desiredBurstPolicy },
        executionMode: this.executionMode,
        lease: this.leaseDiagnostics(),
        lastNativeUnit: this.lastNativeUnit,
        renderGate: {
          enabled: this.renderGateEnabled && this.renderAwareEnabled,
          open: !this.renderGateClosed,
        },
        execution: {
          maxPrefillTokens: this.execution.config.maxPrefillTokens,
          maxDecodeTokens: this.execution.config.maxDecodeTokens,
          foregroundDecodeTokens:
            this.execution.config.foregroundDecodeTokens,
          backgroundDecodeTokens:
            this.execution.config.backgroundDecodeTokens,
          // Undefined means the prefill clamp still follows the decode
          // ceiling, which is the legacy hidden-tab shape.
          backgroundPrefillTokens:
            this.execution.config.backgroundPrefillTokens ?? null,
          targetUnitWallTimeMs: this.execution.config.targetUnitWallTimeMs,
          adaptiveUnits: this.execution.config.adaptiveUnits !== false,
          nativeTimeBudgetEnabled:
            this.execution.config.nativeTimeBudgetEnabled !== false,
        },
        completedRequests: this.scheduler.completedRequests().slice(-32),
        cancelledRequests: this.scheduler.cancelledRequests().slice(-32),
        lastError: this.lastError,
      },
      recovery: {
        scanned: recovery.scanned,
        recovered: recovery.recovered,
        skipped: recovery.skipped,
        errors: [...recovery.errors],
      },
      state: {
        ...this.stateManager.totals,
        residentNodes: this.stateManager.residentNodes,
        trackedNodes: this.stateManager.trackedNodes,
        parkedNodes: this.stateManager.parkedNodes,
        parkedBytes: this.stateManager.parkedBytes,
      },
      residencies: this.stateManager.listResidency(),
      stateHistory: this.stateHistory.map((event) => ({ ...event })),
    };
  }

  // execution slicing is a demo control: disabling it makes each unit carry the
  // whole request, which is the condition the rendering comparison needs
  // The lease's ceiling is the deployment watchdog, not a fixed quantum.
  // An explicit lease.maxBudgetMs still wins for controlled experiments.
  private createLeaseController(): FrameDeadlineLease {
    const explicitMaxBudgetMs = this.leaseOptions.maxBudgetMs;
    return new FrameDeadlineLease({
      ...this.leaseOptions,
      maxBudgetMs:
        explicitMaxBudgetMs ?? this.baseExecutionConfig.maxUnitWallTimeMs,
    });
  }

  configureExecution(config: Partial<ExecutionConfig>): void {
    Object.assign(this.baseExecutionConfig, config);
    if (this.executionMode === 'deadline') {
      // deadline units are shaped per boundary, not by a burst level.
      // Rebuild the lease with the updated deployment ceiling; this also
      // discards samples from the previous execution configuration.
      this.execution.configure({ ...this.baseExecutionConfig });
      this.leaseController = this.createLeaseController();
      return;
    }
    if (this.executionMode === 'unsliced') {
      // The unsliced control has no level-shaped quantum. Render pressure
      // must not turn it into a second bounded arm.
      this.execution.configure({ ...this.baseExecutionConfig });
      return;
    }
    this.applyBurstShape();
  }

  // Experiment harnesses reuse one shared coordinator across arms, so the
  // execution mode has to be re-selectable per tab rather than fixed at
  // construction. Switching modes resets the render controllers and returns
  // the backend to the deployment's own policy; the next boundary re-derives
  // the shape from the new mode.
  async configureExecutionMode(mode: ExecutionMode): Promise<void> {
    if (mode === this.executionMode) {
      return;
    }
    this.executionMode = mode;
    this.leaseController = mode === 'deadline' ? this.createLeaseController() : null;
    this.activeLease = null;
    this.lastLeaseUnit = null;
    this.burstController.reset();
    this.execution.configure({ ...this.baseExecutionConfig });
    await this.setDesiredBurstPolicy(GPU_BURST_POLICIES[0]!, 'stale', null);
  }

  // Render pressure narrows the unit shape; clearing it restores whatever the
  // deployment configured. Both directions go through the base config so a
  // level change can never leave a shrunk unit size behind.
  private applyBurstShape(): void {
    if (this.executionMode !== 'bounded') {
      // Deadline uses the lease and unsliced must keep one unit per request;
      // neither mode may inherit the bounded arm's level-based token caps.
      return;
    }
    this.execution.configure({
      ...this.baseExecutionConfig,
      ...unitSizingForLevel(
        this.desiredBurstPolicy.level,
        this.baseExecutionConfig
      ),
    });
  }

  async configureRenderAware(enabled: boolean): Promise<void> {
    this.renderAwareEnabled = enabled;
    if (enabled) {
      return;
    }
    this.leaseController?.reset();
    this.activeLease = null;
    this.lastLeaseUnit = null;
    this.burstController.reset();
    this.lastPressureClientId = null;
    await this.setDesiredBurstPolicy(GPU_BURST_POLICIES[0]!, 'stale', null);
  }

  async attach(clientId: ClientId, lifecycle: LifecycleRecord): Promise<void> {
    await this.recovery;

    this.fences.set(clientId, (this.fences.get(clientId) ?? 0) + 1);
    this.attachedClients.add(clientId);
    this.scheduler.updateLifecycle(lifecycle, this.clock.now());
    this.telemetry.emit(
      'lifecycle',
      { action: 'attach', epoch: lifecycle.epoch, state: lifecycle.state },
      { clientId }
    );
  }

  async detach(clientId: ClientId): Promise<void> {
    this.bumpFence(clientId);
    this.attachedClients.delete(clientId);
    if (this.lastPressureClientId === clientId) {
      this.burstController.reset();
      this.leaseController?.reset();
      await this.setDesiredBurstPolicy(
        GPU_BURST_POLICIES[0]!,
        'stale',
        clientId
      );
    }
    const requests = this.scheduler
      .pendingRequests()
      .filter((request) => request.session.clientId === clientId);
    for (const request of requests) {
      await this.cancel(request.requestId);
    }
    this.telemetry.emit('lifecycle', { action: 'detach' }, { clientId });
    this.scheduler.removeClient(clientId);
  }

  async updateLifecycle(record: LifecycleRecord): Promise<boolean> {
    const accepted = this.scheduler.updateLifecycle(record, this.clock.now());
    if (accepted) {
      this.telemetry.emit(
        'lifecycle',
        {
          state: record.state,
          foreground: record.foreground,
          epoch: record.epoch,
        },
        { clientId: record.clientId }
      );
      if (!record.foreground && this.evictOnPause) {
        await this.evictPausedClient(record.clientId);
      }
      // A real activation creates (or re-ranks) an answer debt. Start an
      // express unit now rather than waiting for the work-conserving graph
      // to return to the scheduler.
      if (record.foreground) {
        this.preemptForActivation(record.clientId);
      }
      this.scheduleExpress();
      if (!record.foreground && this.lastPressureClientId === record.clientId) {
        this.burstController.reset();
        this.leaseController?.reset();
        await this.setDesiredBurstPolicy(
          GPU_BURST_POLICIES[0]!,
          'stale',
          record.clientId
        );
      }
    }
    return accepted;
  }
  private async evictPausedClient(clientId: ClientId): Promise<void> {
    const nodeIds = new Set<NodeId>();
    for (const [requestId, nodeId] of this.requestNodes) {
      if (this.activeUnitRequestIds.has(requestId)) {
        continue;
      }
      const request = this.scheduler.get(requestId);
      if (request?.nativeRequestId) {
        continue;
      }
      const node = this.stateManager.getNode(nodeId);
      if (node?.session.clientId === clientId) {
        nodeIds.add(nodeId);
      }
    }
    for (const nodeId of nodeIds) {
      const before = this.stateManager.getResidency(nodeId);
      if (before?.level !== 'L1') {
        continue;
      }
      const admission = await this.stateManager.admitNode(
        nodeId,
        'L2',
        this.stateManager.config.repairBudget
      );
      const request = this.lastRequestByNode.get(nodeId);
      if (request) {
        this.emitResidency(request, nodeId, {
          action: 'evict-on-pause',
          previousLevel: before.level,
          admission,
        });
      }
    }
  }

  private preemptForActivation(clientId: ClientId): void {
    if (!this.preemptOnActivation) {
      return;
    }
    // The promotion can arrive just before the next batch enters the active
    // set. Keep the request pending even when there is nothing to cut yet;
    // the native-submit hook will retry as soon as that batch binds an id.
    // Native ids are assigned asynchronously by submit(). Remember the
    // promotion and retry at onNativeSubmit instead of silently skipping a
    // batch whose ids are not populated yet.
    this.pendingActivationPreempts.add(clientId);
    this.flushPendingActivationPreempts();
  }

  private flushPendingActivationPreempts(): void {
    if (this.pendingActivationPreempts.size === 0) {
      return;
    }
    for (const clientId of [...this.pendingActivationPreempts]) {
      const active = this.activeUnitRequests();
      const now = this.clock.now();
      // Nothing is running: the scheduler owns the next choice. A later
      // submit/setPriority call will re-arm this client if it still owes a
      // token; persisting the flag here only cuts later, unrelated work.
      if (active.length === 0) {
        if (!this.scheduler.hasPromotionDebt(clientId, now)) {
          this.pendingActivationPreempts.delete(clientId);
        }
        continue;
      }
      const ownActive = active.filter(
        (request) => request.session.clientId === clientId
      );
      const peers = active.filter(
        (request) => request.session.clientId !== clientId
      );
      const peersAwaitingNativeId = peers.some(
        (request) => request.nativeRequestId === null
      );
      let interrupted = 0;
      for (const request of peers) {
        if (
          !request.nativeRequestId ||
          this.interruptedActivationRequestIds.has(request.requestId) ||
          // Older, still-unanswered promotions keep their turn. A new
          // promotion cuts ordinary peers rather than aborting another
          // session that is also waiting for its first token.
          this.scheduler.promotionPending(request.requestId, now)
        ) {
          continue;
        }
        this.interruptedActivationRequestIds.add(request.requestId);
        interrupted += 1;
        void this.native
          .interrupt(request.nativeRequestId)
          .catch(() => undefined);
      }
      if (interrupted > 0) {
        this.telemetry.emit(
          'lifecycle',
          { action: 'preempt-activation', interrupted },
          { clientId }
        );
      }
      // Native ids may still be binding. Keep the edge armed until the
      // submit hook can cut a real request.
      if (peersAwaitingNativeId) {
        continue;
      }
      // The promoted session is already active, or every cuttable peer was
      // handled. One activation owns one boundary wave: once this batch is
      // cut, later batches are owned by the scheduler instead of being
      // interrupted forever while the promoted request waits its turn.
      const peersSettled = peers.every(
        (request) =>
          this.interruptedActivationRequestIds.has(request.requestId) ||
          this.scheduler.promotionPending(request.requestId, now)
      );
      if (ownActive.length > 0 || peers.length === 0 || peersSettled) {
        this.pendingActivationPreempts.delete(clientId);
        continue;
      }
      if (!this.scheduler.hasPromotionDebt(clientId, now)) {
        this.pendingActivationPreempts.delete(clientId);
      }
    }
  }

  private activeUnitRequests(): RequestRecord[] {
    return [...this.activeUnitRequestIds]
      .map((requestId) => this.scheduler.get(requestId))
      .filter((request): request is RequestRecord => request !== undefined);
  }

  private clearActivationPreemptStateIfIdle(): void {
    if (this.activeUnitRequestIds.size > 0) {
      return;
    }
    this.interruptedActivationRequestIds.clear();
    const now = this.clock.now();
    for (const clientId of [...this.pendingActivationPreempts]) {
      if (!this.scheduler.hasPromotionDebt(clientId, now)) {
        this.pendingActivationPreempts.delete(clientId);
      }
    }
  }

  private scheduleExpress(): void {
    if (!this.expressHandoff) {
      return;
    }
    this.expressQueued = true;
    if (this.expressDraining) {
      return;
    }
    void this.drainExpress();
  }

  private async drainExpress(): Promise<void> {
    if (this.expressDraining) {
      return;
    }
    this.expressDraining = true;
    try {
      while (this.expressQueued) {
        this.expressQueued = false;
        const candidates = this.scheduler.pendingAcknowledgements(
          this.clock.now()
        );
        for (const candidate of candidates) {
          if (this.expressInFlight.has(candidate.requestId)) {
            continue;
          }
          if (this.activeUnitRequestIds.has(candidate.requestId)) {
            continue;
          }
          const request = this.scheduler.claim(candidate.requestId);
          if (!request) {
            continue;
          }
          this.expressInFlight.add(request.requestId);
          void this.runExpressUnit(request).finally(() => {
            this.expressInFlight.delete(request.requestId);
            this.scheduleExpress();
          });
        }
      }
    } finally {
      this.expressDraining = false;
    }
  }

  private async runExpressUnit(request: RequestRecord): Promise<void> {
    if (!this.isRequestFenceCurrent(request)) {
      await this.cancel(request.requestId);
      return;
    }
    const node = this.ensureNode(request);
    this.activeUnitRequestIds.add(request.requestId);
    try {
      const admission = await this.ensureResident(request, node);
      if (admission !== 'ready') {
        await this.holdOrReject(request, admission);
        return;
      }
      const unitStartAt = this.clock.now();
      const decodedBefore = request.decodeGenerated;
      const outcome = await this.execution.runUnit(
        request,
        true,
        this.expressLease(unitStartAt),
        true
      );
      await this.settleRequestUnit(request, node, outcome, {
        decodedBefore,
        batchSize: 1,
        unitStartAt,
      });
    } catch (error) {
      await this.failBatchRequest(request, error);
    } finally {
      this.activeUnitRequestIds.delete(request.requestId);
      this.clearActivationPreemptStateIfIdle();
    }
  }

  private expressLease(createdAt: number): ExecutionLease {
    return {
      leaseId: -1,
      source: 'deadline',
      reason: 'work-conserving',
      budgetMs: 5_000,
      maxDecodeTokens: 1,
      maxPrefillTokens: this.handoffPrefillTokens,
      submissionPauseMs: 0,
      frameBudgetMs: 1000 / 60,
      frameDebtMs: 0,
      queueWaitMs: null,
      createdAt,
      commandSubmitBatchSize: 64,
      maxInflightBatches: 0,
      basis: 'unlimited',
    };
  }

  async updateRenderPressure(
    signal: RenderPressureSignal
  ): Promise<GpuBurstPolicy> {
    const lifecycle = this.scheduler.lifecycleFor(signal.clientId);
    if (
      !lifecycle ||
      !lifecycle.foreground ||
      lifecycle.epoch !== signal.epoch
    ) {
      return { ...this.desiredBurstPolicy };
    }
    if (!this.renderAwareEnabled) {
      return { ...this.desiredBurstPolicy };
    }
    if (this.leaseController) {
      // Deadline mode reads the same signal, but turns it into a lease
      // rather than a level. The signal may arrive while a unit is running;
      // the lease is applied at the next graph boundary.
      const now = this.clock.now();
      this.leaseController.observe(signal, now);
      this.lastPressureClientId = signal.clientId;
      const lease = this.leaseController.lease({ now, foreground: true });
      await this.applyLease(lease, signal.clientId);
      return { ...this.desiredBurstPolicy };
    }
    const previous = this.desiredBurstPolicy;
    const next = this.burstController.observe(signal, this.clock.now());
    this.lastPressureClientId = signal.clientId;
    const reason =
      burstLevelIndex(next) > burstLevelIndex(previous)
        ? 'pressure'
        : 'recovery';
    await this.setDesiredBurstPolicy(next, reason, signal.clientId);
    return { ...this.desiredBurstPolicy };
  }

  async submit(
    request: RequestRecord
  ): Promise<{ accepted: boolean; reason?: string }> {
    await this.recovery;
    if (!this.isFenceCurrent(request.session.clientId)) {
      return { accepted: false, reason: 'fenced' };
    }
    const result = this.scheduler.enqueue(request, this.clock.now());
    if (result.accepted) {
      this.requestFences.set(request.requestId, {
        clientId: request.session.clientId,
        generation: this.fences.get(request.session.clientId) ?? 0,
      });
      this.telemetry.emit(
        'request_enqueue',
        {
          stage: request.stage,
          promptTokens: request.promptTokens.length,
          maxNewTokens: request.maxNewTokens,
        },
        this.contextFor(request)
      );
      // A promotion can arrive before the application submits its first
      // request. In that case the lifecycle update found no active request
      // to cut, and a later background batch could otherwise make the
      // promoted request wait for an unrelated graph boundary.
      if (this.scheduler.promotionPending(request.requestId, this.clock.now())) {
        this.preemptForActivation(request.session.clientId);
        this.scheduleExpress();
      }
    }
    return result;
  }

  async cancel(requestId: RequestId): Promise<boolean> {
    const request = this.scheduler.get(requestId);
    if (!request) {
      return false;
    }
    if (request.nativeRequestId) {
      await this.native.cancel(request.nativeRequestId).catch(() => undefined);
      this.telemetry.emit(
        'native_cancel',
        {},
        this.contextFor(request)
      );
    }
    this.scheduler.cancel(requestId);
    this.requestFences.delete(requestId);
    this.telemetry.emit('request_cancel', {}, this.contextFor(request));
    return true;
  }

  async status(requestId: RequestId): Promise<RequestRecord | null> {
    const request = this.scheduler.get(requestId);
    return request ? { ...request } : null;
  }

  async pump(maxSteps = 1): Promise<number> {
    await this.recovery;
    let completedSteps = 0;
    for (let step = 0; step < maxSteps; step += 1) {
      const ran = await this.pumpOnce();
      if (!ran) {
        break;
      }
      completedSteps += 1;
    }
    return completedSteps;
  }

  get running(): boolean {
    return this.inFlight;
  }

  subscribe(listener: (event: CoordinatorEvent) => void): () => void {
    this.eventListeners.add(listener);
    return () => {
      this.eventListeners.delete(listener);
    };
  }

  publishRuntime(event: Extract<CoordinatorEvent, { type: 'runtime' }>): void {
    this.emitEvent(event);
  }

  publishTokenChunk(chunk: NativeTokenChunk): void {
    const request = this.scheduler.get(chunk.requestId);
    if (!request) {
      return;
    }
    this.emitEvent({
      type: 'token',
      ...this.contextFor(request),
      text: chunk.text,
      decodeGenerated: chunk.decodeGenerated,
      elapsedMs: chunk.elapsedMs,
    });
  }

  protectedSessions(): SessionKey[] {
    return this.scheduler.protectedSessionKeys(this.clock.now());
  }

  // bounded serving: one session node per (client, session) pair, or per
  // explicit parent node when the caller sends a context tree.
  private nodeIdFor(request: RequestRecord): NodeId {
    return (
      request.nodeId ??
      request.parentNodeId ??
      `${request.session.clientId}:${request.session.sessionId}`
    );
  }

  private ensureNode(request: RequestRecord): SessionNode {
    const nodeId = this.nodeIdFor(request);
    this.requestNodes.set(request.requestId, nodeId);
    this.lastRequestByNode.set(nodeId, request);
    const existing = this.stateManager.getNode(nodeId);
    const startsTurn = isFreshTurn(request);
    const forkParentId = request.forkFromNodeId ?? null;
    const parent = forkParentId
      ? this.stateManager.getNode(forkParentId)
      : undefined;
    if (forkParentId !== null && parent === undefined) {
      throw new Error(`unknown fork parent node: ${forkParentId}`);
    }
    // A stateless deployment's turn carries its whole context, so the node's
    // text is replaced instead of accumulated. The KV prefix still comes from
    // the resident slot, which is what makes the turn cheap without changing
    // what the model sees.
    const selfContainedTurn = this.continuationTokenCap === 0;
    const priorResidentTokens = existing
      ? existing.residentTokens ?? existing.tokenIds.length
      : 0;
    const projectedResidentTokens = existing
      ? priorResidentTokens +
        (startsTurn && !selfContainedTurn ? request.promptTokens.length : 0)
      : request.promptTokens.length;
    const node: SessionNode = existing
      ? {
          ...existing,
          session: { ...request.session },
          turnText:
            startsTurn && !selfContainedTurn
              ? appendTurnText(existing.turnText, request.promptText)
              : startsTurn
                ? request.promptText ?? ''
                : existing.turnText,
          tokenIds:
            startsTurn && !selfContainedTurn
              ? [...existing.tokenIds, ...request.promptTokens]
              : startsTurn
                ? [...request.promptTokens]
                : [...existing.tokenIds],
          residentTokens: this.boundedResidentTokens(projectedResidentTokens),
          lastAccessAt: this.clock.now(),
          accessCount: existing.accessCount + 1,
        }
      : {
          nodeId,
          session: request.session,
          parentNodeId:
            forkParentId ??
            (request.parentNodeId !== nodeId ? request.parentNodeId : null),
          turnText: parent
            ? appendTurnText(parent.turnText, request.promptText)
            : request.promptText ?? '',
          continuationText: parent
            ? parent.continuationText ?? parent.turnText
            : '',
          tokenIds: parent
            ? [...parent.tokenIds, ...request.promptTokens]
            : [...request.promptTokens],
          residentTokens: this.boundedResidentTokens(projectedResidentTokens),
          createdAt: request.createdAt,
          lastAccessAt: this.clock.now(),
          accessCount: 1,
        };
    const level = this.stateManager.getResidency(nodeId)?.level ?? 'L2';
    this.stateManager.registerNode(node, level);
    if (!existing && forkParentId !== null) {
      request.forkPending = true;
      if (parent) {
        request.replayText = parent.continuationText ?? parent.turnText;
        request.replayTokens = [...parent.tokenIds];
        request.keepReplayPrefix = true;
      }
    }
    if (forkParentId !== null || node.parentNodeId !== null) {
      // A forked child resumes the parent's KV and evaluates only the
      // request suffix supplied with this turn.
      request.replaceContext = true;
    }
    if (existing) {
      if (!selfContainedTurn) {
        if (existing.continuationText) {
          request.replayText = existing.continuationText;
        }
        request.replayTokens = [...existing.tokenIds];
        if (existing.parentNodeId !== null) {
          request.keepReplayPrefix = true;
        }
      }
    }
    return node;
  }

  private async ensureResident(
    request: RequestRecord,
    node: SessionNode
  ): Promise<ResidentAdmission> {
    const previousLevel =
      this.stateManager.getResidency(node.nodeId)?.level ?? 'rebuild';
    if (request.forkPending === true) {
      const parentNodeId = request.forkFromNodeId;
      if (!parentNodeId) {
        throw new Error(`request ${request.requestId} has no fork parent`);
      }
      const fork = await this.stateManager.forkNode(
        parentNodeId,
        node.nodeId,
        this.stateManager.config.repairBudget
      );
      for (const victimNodeId of fork.victims) {
        this.releaseVictimRequests(victimNodeId, request.requestId);
      }
      if (!fork.success) {
        throw new Error(
          `fork ${parentNodeId} -> ${node.nodeId} failed: ${
            fork.reason ?? fork.residentLevel
          }`
        );
      }
      delete request.forkPending;
      const forkResidency = this.stateManager.getResidency(node.nodeId);
      if (
        forkResidency?.level !== 'L1' ||
        forkResidency.slotId === null
      ) {
        throw new Error(`forked node ${node.nodeId} has no live slot`);
      }
      request.resumeSlotId = forkResidency.slotId;
      request.replaceContext = true;
    }
    const admission = await this.stateManager.admitNode(
      node.nodeId,
      'L1',
      this.stateManager.config.repairBudget,
      this.pinnedResidentNodes(node.nodeId)
    );
    for (const victimNodeId of admission.victims) {
      this.releaseVictimRequests(victimNodeId, request.requestId);
    }
    const residency = this.stateManager.getResidency(node.nodeId);
    if (
      residency?.level !== 'rebuild' &&
      request.keepReplayPrefix !== true
    ) {
      delete request.replayText;
      delete request.replayTokens;
    }
    // A stateless turn replaces the slot's context instead of extending it, so
    // nothing is replayed into the prompt. The admission still stands and the
    // residency event is reported as usual.
    const statelessFreshTurn =
      this.continuationTokenCap === 0 && isFreshTurn(request);
    if (statelessFreshTurn) {
      // the request carries its whole context, so the resumed slot must not
      // replay the previous turn into it; the shared prefix is still matched
      // against the resident KV by the backend
      request.replaceContext = true;
    }
    if (
      admission.success &&
      residency?.level === 'L1' &&
      residency.slotId !== null
    ) {
      // the node is live again: either it never left L1, or its parked KV
      // state was restored into a slot and must be resumed
      request.resumeSlotId = residency.slotId;
    }
    const liveAdmission =
      admission.success &&
      residency?.level === 'L1' &&
      residency.slotId !== null;
    this.emitResidency(request, node.nodeId, {
      // A degraded admission leaves the node without a live slot, so the
      // event has to read as a rebuild rather than a successful restore.
      action:
        !admission.success || admission.residentLevel === 'rebuild'
          ? 'drop'
          : previousLevel === 'L1'
            ? 'admit'
            : 'restore',
      previousLevel,
      admission,
    });
    if (liveAdmission) {
      return 'ready';
    }
    // Pinned work must never be evicted just to respect a token cap, so an
    // over-cap request may still run while the native pool has room. Without
    // a free slot it has nowhere to go and must wait instead.
    // The native serving pool is bounded, so a node without a live slot must
    // not be submitted: the engine would either take another session's slot
    // or fail outright. Capacity pressure is handled here instead.
    const policy =
      this.stateManager.config.admissionPolicy ?? 'monotonic-bounded';
    if (await this.hasIdleNativeSlot(request.requestId)) {
      const onlyOverCap = admission.reason !== 'budget_exhausted';
      if (onlyOverCap || policy !== 'immediate-reject') {
        return 'ready';
      }
    }
    return policy === 'immediate-reject' &&
      admission.reason === 'budget_exhausted'
      ? 'rejected'
      : 'deferred';
  }

  // A slot that no serving request owns yet. Units already running in the
  // coordinator but not yet submitted reserve their own slot, so they are
  // subtracted to keep two lanes from claiming the last free one.
  private async hasIdleNativeSlot(requestId: RequestId): Promise<boolean> {
    const stats = await this.native.slotStats().catch(() => []);
    const free = stats.filter((entry) => !entry.inUse).length;
    if (free <= 0) {
      return false;
    }
    let reserved = 0;
    for (const activeId of this.activeUnitRequestIds) {
      if (activeId === requestId) {
        continue;
      }
      const pending = this.scheduler.get(activeId);
      if (pending && !pending.nativeRequestId) {
        reserved += 1;
      }
    }
    return free > reserved;
  }

  // Capacity refusal is a scheduling decision, not an execution error: the
  // request goes back to the queue and is retried once a slot frees, unless
  // the configured admission policy refuses it outright.
  private async holdOrReject(
    request: RequestRecord,
    admission: Exclude<ResidentAdmission, 'ready'>
  ): Promise<void> {
    if (admission === 'rejected') {
      const policy =
        this.stateManager.config.admissionPolicy ?? 'monotonic-bounded';
      throw new Error(`request rejected: admission-${policy}`);
    }
    this.scheduler.requeue(request.requestId, this.clock.now());
    this.telemetry.emit(
      'request_requeue',
      {
        stage: request.stage,
        serviceMs: request.serviceMs,
        reason: 'capacity',
      },
      this.contextFor(request)
    );
    // A capacity wait must not become a hot admission loop while a peer unit
    // still holds the only live slots.
    await yieldToEventLoop();
  }

  // bounded serving: once a turn is done, keep the session warm in L2 (or
  // fall back to a rebuild when the backend already dropped the slot)
  // A request with a native id but no completion boundary still owns a live
  // sequence. The state manager must not park it just to make room for a
  // younger request; that eviction path is unsafe while a graph is partial.
  private pinnedResidentNodes(excludeNodeId: NodeId): ReadonlySet<NodeId> {
    const pinned = new Set<NodeId>();
    for (const [requestId, nodeId] of this.requestNodes) {
      if (nodeId === excludeNodeId) {
        continue;
      }
      // Only a unit that is actually executing holds its slot. A sequence
      // that is waiting between units still owns a native id, but its KV
      // can be parked safely at the completed graph boundary. Treating
      // every native id as pinned let the first two long sessions occupy
      // both L1 slots forever while later tabs starved.
      if (
        this.activeUnitRequestIds.has(requestId) ||
        this.batchPinnedRequestIds.has(requestId)
      ) {
        pinned.add(nodeId);
      }
    }
    return pinned;
  }

  private async parkCompletedNode(
    request: RequestRecord,
    node: SessionNode
  ): Promise<void> {
    if (this.completedTier === 'L1') {
      return;
    }
    if (this.completedTier === 'rebuild') {
      await this.stateManager.dropLiveNode(node.nodeId);
      this.emitResidency(request, node.nodeId, {
        action: 'drop',
        previousLevel: 'L1',
      });
      return;
    }
    const residency = this.stateManager.getResidency(node.nodeId);
    if (residency?.level !== 'L1' || residency.slotId === null) {
      return;
    }
    const stats = await this.native.slotStats().catch(() => []);
    const live = stats.some(
      (entry) => entry.slotId === residency.slotId && entry.inUse
    );
    if (!live) {
      await this.stateManager.dropLiveNode(node.nodeId);
      this.emitResidency(request, node.nodeId, {
        action: 'drop',
        previousLevel: 'L1',
      });
      return;
    }
    let admission: StateAdmissionResult | null = null;
    try {
      // The completed turn is an admission into L2, not an unconditional
      // demotion. Capacity repair has to run here too, otherwise a long-lived
      // SharedWorker accumulates native parked copies past the L2 cap.
      admission = await this.stateManager.admitNode(
        node.nodeId,
        this.completedTier,
        this.stateManager.config.repairBudget
      );
      if (!admission.success) {
        await this.stateManager.dropLiveNode(node.nodeId);
      }
    } catch {
      await this.stateManager.dropLiveNode(node.nodeId);
    }
    this.emitResidency(request, node.nodeId, {
      action: admission?.success ? 'park' : 'drop',
      previousLevel: 'L1',
      ...(admission === null ? {} : { admission }),
    });
  }

  private releaseVictimRequests(
    victimNodeId: NodeId,
    currentRequestId: RequestId
  ): void {
    for (const [requestId, nodeId] of this.requestNodes) {
      if (nodeId !== victimNodeId || requestId === currentRequestId) {
        continue;
      }
      const victim = this.scheduler.get(requestId);
      if (!victim) {
        continue;
      }
      // the victim's KV state left L1 and its slot was released, so the next
      // unit has to re-submit into the restored slot
      victim.nativeRequestId = null;
      victim.slotId = null;
      victim.resumeSlotId = null;
      this.emitResidency(victim, victimNodeId, { action: 'drop' });
    }
  }

  private recordResidency(request: RequestRecord, node: SessionNode): void {
    const updated: SessionNode = {
      ...node,
      residentTokens: this.boundedResidentTokens(
        request.residentTokens ??
          request.promptTokens.length + request.decodeGenerated
      ),
      lastAccessAt: this.clock.now(),
    };
    const level = this.stateManager.getResidency(node.nodeId)?.level ?? 'L2';
    this.stateManager.registerNode(updated, level);
    if (request.slotId !== null) {
      const residency = this.stateManager.getResidency(node.nodeId);
      if (residency?.level !== 'L1' || residency.slotId !== request.slotId) {
        this.stateManager.registerLiveNode(updated, request.slotId);
      }
    }
    this.emitResidency(request, node.nodeId);
  }

  private withCompletedContinuation(
    node: SessionNode,
    request: RequestRecord
  ): SessionNode {
    const prompt = request.promptText ?? '';
    const generated = request.outputChunks.join('');
    const withPrompt = appendTurnText(node.continuationText ?? '', prompt);
    const accumulated =
      generated.length > 0 ? withPrompt + generated : withPrompt;
    return {
      ...node,
      continuationText: this.capContinuationText(accumulated),
    };
  }

  /**
   * Bound what a session carries into its next turn. Trimming keeps the
   * newest text, so a capped session rebuilds from the most recent turn
   * instead of re-prefilling its whole history.
   */
  private boundedResidentTokens(value: number): number {
    const cap = this.continuationTokenCap;
    if (cap === undefined || cap === 0) return value;
    return Math.min(value, cap);
  }

  private capContinuationText(text: string): string {
    const cap = this.continuationTokenCap;
    if (cap === undefined || text.length === 0) {
      return text;
    }
    return trimPrefixToTokenBudget(text, cap).text;
  }

  private emitResidency(
    request: RequestRecord,
    nodeId: NodeId,
    detail: ResidencyEventDetail = {}
  ): void {
    const residency = this.stateManager.getResidency(nodeId);
    if (!residency) {
      return;
    }
    const totals = this.stateManager.totals;
    const admission = detail.admission;
    const state: StateTransitionEvent = {
      sequence: ++this.stateSequence,
      timestampMs: this.clock.now(),
      clientId: request.session.clientId,
      sessionId: request.session.sessionId,
      nodeId,
      action: detail.action ?? 'account',
      level: residency.level,
      tokenCount: residency.tokenCount,
      bufferBytes: this.stateManager.bufferBytes(nodeId),
      hostBytes: this.stateManager.hostBufferBytes(nodeId),
      totals,
      residentNodes: this.stateManager.residentNodes,
      trackedNodes: this.stateManager.trackedNodes,
      parkedNodes: this.stateManager.parkedNodes,
      parkedBytes: this.stateManager.parkedBytes,
      ...(detail.previousLevel === undefined
        ? {}
        : { previousLevel: detail.previousLevel }),
      ...(admission === undefined
        ? {}
        : {
            admissionPolicy: this.stateManager.config.admissionPolicy ??
              'monotonic-bounded',
            admissionSuccess: admission.success,
            ...(admission.reason === undefined
              ? {}
              : { admissionReason: admission.reason }),
            steps: admission.steps,
            victims: [...admission.victims],
          }),
    };
    this.stateHistory.push(state);
    if (this.stateHistory.length > 512) {
      this.stateHistory.splice(0, this.stateHistory.length - 512);
    }
    this.emitEvent({
      type: 'residency',
      ...this.contextFor(request),
      nodeId,
      level: residency.level,
      slotId: residency.slotId,
      tokenCount: residency.tokenCount,
      bufferBytes: state.bufferBytes,
      hostBytes: state.hostBytes,
      totals,
      state,
    });
  }

  private async pumpOnce(): Promise<boolean> {
    if (this.inFlight) {
      return false;
    }
    await this.syncGpuBurstPolicy();
    if (this.renderGateEnabled && this.renderAwareEnabled) {
      const now = this.clock.now();
      const wantsClosed = this.desiredBurstPolicy.level === 'constrained';
      if (!wantsClosed) {
        this.renderGateClosedSince = 0;
        this.renderGateOpenUntil = 0;
      }
      let closed = wantsClosed && now >= this.renderGateOpenUntil;
      if (closed) {
        if (this.renderGateClosedSince === 0) {
          this.renderGateClosedSince = now;
        } else if (
          now - this.renderGateClosedSince >= this.renderGateMaxHoldMs
        ) {
          this.renderGateOpenUntil = now + this.renderGateMinOpenMs;
          this.renderGateClosedSince = 0;
          closed = false;
        }
      }
      if (closed !== this.renderGateClosed) {
        this.renderGateClosed = closed;
        this.telemetry.emit('render_gate', {
          open: !closed,
          level: this.desiredBurstPolicy.level,
          forced: wantsClosed && !closed,
        });
      }
      if (closed) {
        return false;
      }
    }
    this.stateManager.setProtectedSessions(this.protectedSessions());
    const selection = this.scheduler.selectNext(this.clock.now());
    if (!selection || !selection.runnable) {
      return false;
    }
    const request = selection.request;
    if (!this.isRequestFenceCurrent(request)) {
      this.emitEvent({
        type: 'fenced',
        ...this.contextFor(request),
      });
      await this.cancel(request.requestId);
      return false;
    }
    // Cooperative batching is opt-in while it is being screened. The head
    // request has already been removed from the scheduler queue; peers of
    // the same lifecycle class and stage are admitted with it so one graph
    // boundary can cover several independent sequences. The native server
    // continuous-batches them, which shares the weight read across requests
    // instead of buying throughput by taking more GPU time from the page.
    if (this.executionMode === 'deadline' && this.leaseBatchSize > 1) {
      // A promotion that has not been answered yet is high-priority-shaped.
      // It leads its own small admission instead of waiting for an ordinary
      // background graph; the cap below keeps that graph short.
      const headHandoff = this.scheduler.promotionPending(
        request.requestId,
        this.clock.now()
      );
      const headPriority =
        headHandoff ||
        this.scheduler.priorityServing(request.requestId, this.clock.now());
      // A pending hand-off may share its graph with a bounded number of
      // peers. Letting it ride in a full background batch gives the batch
      // better weight-read amortisation, but it also stretches the promoted
      // session's answer to the slowest peer. The reverse must not happen
      // either: an ordinary background graph never absorbs a pending
      // hand-off, so the promotion is always the head of its own admission.
      //
      // Ragged prompt processing does not get the same benefit: the native
      // batch limit is shared across sequences, so grouping prompts of
      // different lengths leaves slots idle and multiplies graph steps.
      // Keep prefill admissions single-request and batch only decode, where
      // every sequence contributes one token to a shared weight read.
      const handoffPeers =
        headPriority
          ? this.batchHandoff
            ? this.handoffBatchSize - 1
            : 0
          : request.stage === 'prefill'
            ? 0
            : this.leaseBatchSize - 1;
      const peers = this.scheduler.selectBatch(
        this.clock.now(),
        handoffPeers,
        (candidate) =>
          (headPriority && this.batchHandoff
            ? true
            : candidate.lifecycle.foreground ===
              request.lifecycle.foreground) &&
          (headPriority
            ? this.batchHandoff
            : !this.scheduler.promotionPending(
                candidate.requestId,
                this.clock.now()
              )) &&
          candidate.stage === request.stage &&
          (candidate.session.clientId !== request.session.clientId ||
            candidate.session.sessionId !== request.session.sessionId)
      );
      if (peers.length > 0) {
        this.inFlight = true;
        try {
          return await this.runDeadlineBatch([request, ...peers]);
        } finally {
          this.inFlight = false;
        }
      }
    }

    this.inFlight = true;
    try {
      this.activeUnitRequestIds.add(request.requestId);
      this.flushPendingActivationPreempts();
      this.telemetry.emit(
        'request_select',
        {
          foreground: request.lifecycle.foreground,
          scheduledAt: this.clock.now(),
        },
        this.contextFor(request)
      );
      this.engineSequence += 1;
      const node = this.ensureNode(request);
      const unitStartAt = this.clock.now();
      const gapMs =
        this.lastLeaseUnitEndAt > 0
          ? Math.max(0, unitStartAt - this.lastLeaseUnitEndAt)
          : 0;
      const priorityAtStart =
        this.scheduler.promotionPending(request.requestId, unitStartAt) ||
        this.scheduler.priorityServing(request.requestId, unitStartAt);
      if (priorityAtStart) {
        await this.isolateIdleNativeSequences(
          new Set([node.nodeId])
        );
      }
      const lease = await this.leaseForRequest(request);
      const tunedLease =
        lease && this.scheduler.promotionPending(request.requestId, unitStartAt)
          ? {
              ...lease,
              maxPrefillTokens: Math.max(
                lease.maxPrefillTokens,
                this.handoffPrefillTokens
              ),
            }
          : lease;
      const decodedBefore = request.decodeGenerated;
      // Snapshot the priority class that shaped this lease: a promotion can
      // rewrite the live record while the unit is still running.
      const foregroundAtStart = request.lifecycle.foreground;
      const pauseStartAt = this.clock.now();
      await this.pauseBeforeNativeUnit(request);
      const pauseMs = this.clock.now() - pauseStartAt;
      const admissionStartAt = this.clock.now();
      const admission = await this.ensureResident(request, node);
      const admissionMs = this.clock.now() - admissionStartAt;
      if (admission !== 'ready') {
        await this.holdOrReject(request, admission);
        return true;
      }
      const outcome = await this.execution.runUnit(
        request,
        this.scheduler.hasPendingHighPriority(),
        tunedLease ?? undefined,
        this.scheduler.promotionPending(request.requestId, unitStartAt),
        !request.lifecycle.foreground &&
          this.scheduler.priorityServing(request.requestId, unitStartAt)
      );
      const unitEndAt = this.clock.now();
      this.lastLeaseUnitEndAt = unitEndAt;
      this.recordNativeUnit(outcome.progress);
      if (lease) {
        this.observeLeaseUnit(request, tunedLease ?? lease, outcome, decodedBefore, {
          startedAtMs: unitStartAt,
          endedAtMs: unitEndAt,
          gapMs,
          admissionMs,
          pauseMs,
          foregroundAtStart,
        });
      }
      this.noteDecodeProgress(request, decodedBefore, unitEndAt);
      const completedNode = outcome.completed
        ? this.withCompletedContinuation(node, request)
        : node;
      this.recordResidency(request, completedNode);
      if (outcome.completed) {
        await this.parkCompletedNode(request, completedNode);
      }
      this.emitEvent({
        type: 'unit',
        ...this.contextFor(request),
        progress: outcome.progress,
        budget: outcome.budget,
      });
      if (outcome.completed) {
        this.scheduler.complete(request.requestId);
        this.requestFences.delete(request.requestId);
        this.emitEvent({
          type: 'complete',
          ...this.contextFor(request),
          output: request.outputChunks.join(''),
        });
        this.telemetry.emit(
          'request_complete',
          {
            output: request.outputChunks.join(''),
            generated: request.decodeGenerated,
          },
          this.contextFor(request)
        );
      } else {
        this.scheduler.requeue(request.requestId, this.clock.now());
        this.telemetry.emit(
          'request_requeue',
          { stage: request.stage, serviceMs: request.serviceMs },
          this.contextFor(request)
        );
      }
      return true;
    } catch (error) {
      if (request.nativeRequestId) {
        await this.native.cancel(request.nativeRequestId).catch(() => undefined);
      }
      this.scheduler.cancel(request.requestId);
      this.requestFences.delete(request.requestId);
      // A unit-level failure used to be swallowed by the pump loop, leaving
      // the page stuck on "queued" forever. Record it and surface it.
      const message = error instanceof Error ? error.message : String(error);
      this.lastError = message;
      this.emitEvent({
        type: 'failed',
        ...this.contextFor(request),
        error: message,
      });
      throw error;
    } finally {
      this.activeUnitRequestIds.delete(request.requestId);
      this.inFlight = false;
      this.scheduleExpress();
      this.clearActivationPreemptStateIfIdle();
    }
  }

  /**
   * Advance a small group of same-class requests through one deadline
   * admission. The native server receives the advances concurrently, so
   * llama.cpp can continuous-batch the independent sequences. Frame pacing
   * stays outside the batch: the whole group is admitted under one lease and
   * the pump does not return to the scheduler until every member reaches its
   * token boundary.
   */
  private async runDeadlineBatch(
    requests: RequestRecord[]
  ): Promise<boolean> {
    const primary = requests[0];
    if (!primary) {
      return false;
    }
    const nodes = new Map<RequestId, SessionNode>();
    const accepted: RequestRecord[] = [];
    // Priority class that admitted each member, captured before the first
    // await so a promotion during the batch cannot rewrite the trace.
    const foregroundAtStart = new Map<RequestId, boolean>();
    for (const request of requests) {
      foregroundAtStart.set(request.requestId, request.lifecycle.foreground);
      if (!this.isRequestFenceCurrent(request)) {
        this.emitEvent({
          type: 'fenced',
          ...this.contextFor(request),
        });
        await this.cancel(request.requestId);
        continue;
      }
      accepted.push(request);
      nodes.set(request.requestId, this.ensureNode(request));
    }
    if (accepted.length === 0) {
      return false;
    }

    for (const request of accepted) {
      this.activeUnitRequestIds.add(request.requestId);
    }

    this.flushPendingActivationPreempts();

    const selectedBatchSize = accepted.length;
    const unitStartAt = this.clock.now();
    const gapMs =
      this.lastLeaseUnitEndAt > 0
        ? Math.max(0, unitStartAt - this.lastLeaseUnitEndAt)
        : 0;
    let lease: ExecutionLease | null = null;
    let pauseMs = 0;
    let admissionMs = 0;
    this.batchPinnedRequestIds.clear();
    for (const request of accepted) {
      this.batchPinnedRequestIds.add(request.requestId);
      this.telemetry.emit(
        'request_select',
        {
          foreground: request.lifecycle.foreground,
          scheduledAt: unitStartAt,
          batchSize: selectedBatchSize,
        },
        this.contextFor(request)
      );
    }

    try {
      lease = await this.leaseForRequest(primary);
      if (
        lease &&
        this.scheduler.promotionPending(primary.requestId, unitStartAt)
      ) {
        lease = {
          ...lease,
          maxPrefillTokens: Math.max(
            lease.maxPrefillTokens,
            this.handoffPrefillTokens
          ),
        };
      }
      const priorityAtStart =
        this.scheduler.promotionPending(primary.requestId, unitStartAt) ||
        this.scheduler.priorityServing(primary.requestId, unitStartAt);
      if (priorityAtStart) {
        await this.isolateIdleNativeSequences(
          new Set([...nodes.values()].map((node) => node.nodeId))
        );
      }
      const pauseStartAt = this.clock.now();
      await this.pauseBeforeNativeUnit(primary);
      pauseMs = this.clock.now() - pauseStartAt;
      const admissionStartAt = this.clock.now();
      const executable: RequestRecord[] = [];
      for (const request of accepted) {
        const node = nodes.get(request.requestId);
        if (!node) {
          throw new Error(`missing session node for ${request.requestId}`);
        }
        const admission = await this.ensureResident(request, node);
        if (admission === 'ready') {
          executable.push(request);
          continue;
        }
        if (admission === 'rejected') {
          const policy =
            this.stateManager.config.admissionPolicy ?? 'monotonic-bounded';
          await this.failBatchRequest(
            request,
            new Error(`request rejected: admission-${policy}`)
          );
          continue;
        }
        await this.holdOrReject(request, admission);
        this.activeUnitRequestIds.delete(request.requestId);
        this.batchPinnedRequestIds.delete(request.requestId);
      }
      admissionMs = this.clock.now() - admissionStartAt;

      if (executable.length === 0) {
        return true;
      }
      const executableBatchSize = executable.length;

      const settled = await Promise.allSettled(
        executable.map(async (request) => {
          const node = nodes.get(request.requestId);
          if (!node) {
            throw new Error(`missing session node for ${request.requestId}`);
          }
          const decodedBefore = request.decodeGenerated;
          const outcome = await this.execution.runUnit(
            request,
            this.scheduler.hasPendingHighPriority(),
            lease ?? undefined,
            this.scheduler.promotionPending(request.requestId, unitStartAt),
            !request.lifecycle.foreground &&
              this.scheduler.priorityServing(request.requestId, unitStartAt)
          );
          await this.settleRequestUnit(request, node, outcome, {
            decodedBefore,
            batchSize: executableBatchSize,
            ...(lease ? { lease } : {}),
            unitStartAt,
            gapMs,
            admissionMs,
            pauseMs,
            foregroundAtStart:
              foregroundAtStart.get(request.requestId) ?? false,
          });
          return outcome;
        })
      );

      for (let index = 0; index < settled.length; index += 1) {
        const result = settled[index];
        const request = executable[index];
        if (!request || !result || result.status === 'fulfilled') {
          continue;
        }
        await this.failBatchRequest(request, result.reason);
      }
      return true;
    } finally {
      for (const request of accepted) {
        this.activeUnitRequestIds.delete(request.requestId);
      }
      this.batchPinnedRequestIds.clear();
      this.scheduleExpress();
      this.clearActivationPreemptStateIfIdle();
    }
  }

  private async settleRequestUnit(
    request: RequestRecord,
    node: SessionNode,
    outcome: ExecutionOutcome,
    timing: {
      decodedBefore: number;
      batchSize: number;
      lease?: ExecutionLease;
      unitStartAt: number;
      gapMs?: number;
      admissionMs?: number;
      pauseMs?: number;
      foregroundAtStart?: boolean;
    }
  ): Promise<void> {
    const unitEndAt = this.clock.now();
    this.lastLeaseUnitEndAt = unitEndAt;
    this.recordNativeUnit(outcome.progress, timing.batchSize);
    if (timing.lease) {
      this.observeLeaseUnit(
        request,
        timing.lease,
        outcome,
        timing.decodedBefore,
        {
          startedAtMs: timing.unitStartAt,
          endedAtMs: unitEndAt,
          gapMs: timing.gapMs ?? 0,
          admissionMs: timing.admissionMs ?? 0,
          pauseMs: timing.pauseMs ?? 0,
          batchSize: timing.batchSize,
          foregroundAtStart: timing.foregroundAtStart ?? false,
        }
      );
    }
    this.noteDecodeProgress(request, timing.decodedBefore, unitEndAt);
    const completedNode = outcome.completed
      ? this.withCompletedContinuation(node, request)
      : node;
    this.recordResidency(request, completedNode);
    if (outcome.completed) {
      await this.parkCompletedNode(request, completedNode);
    }
    this.emitEvent({
      type: 'unit',
      ...this.contextFor(request),
      progress: outcome.progress,
      budget: outcome.budget,
    });
    if (outcome.completed) {
      this.scheduler.complete(request.requestId);
      this.requestFences.delete(request.requestId);
      this.emitEvent({
        type: 'complete',
        ...this.contextFor(request),
        output: request.outputChunks.join(''),
      });
      this.telemetry.emit(
        'request_complete',
        {
          output: request.outputChunks.join(''),
          generated: request.decodeGenerated,
          batchSize: timing.batchSize,
        },
        this.contextFor(request)
      );
    } else {
      this.scheduler.requeue(request.requestId, this.clock.now());
      this.telemetry.emit(
        'request_requeue',
        {
          stage: request.stage,
          serviceMs: request.serviceMs,
          batchSize: timing.batchSize,
        },
        this.contextFor(request)
      );
    }
  }

  private async failBatchRequest(
    request: RequestRecord,
    error: unknown
  ): Promise<void> {
    if (request.nativeRequestId) {
      await this.native.cancel(request.nativeRequestId).catch(() => undefined);
    }
    this.scheduler.cancel(request.requestId);
    this.requestFences.delete(request.requestId);
    const message = error instanceof Error ? error.message : String(error);
    this.lastError = message;
    this.emitEvent({
      type: 'failed',
      ...this.contextFor(request),
      error: message,
    });
  }

  // A priority unit should not inherit resident native work from ordinary
  // sessions. Parking releases their serving slots while retaining KV in the
  // native heap. The next ordinary admission imports the same state and
  // resumes from its existing token boundary.
  private async isolateIdleNativeSequences(
    keepResident: ReadonlySet<NodeId>
  ): Promise<void> {
    if (!this.priorityNativeIsolation) {
      return;
    }
    const candidates = this.stateManager
      .listResidency()
      .filter(
        (record) =>
          record.level === 'L1' &&
          record.slotId !== null &&
          !keepResident.has(record.nodeId)
      );
    for (const candidate of candidates) {
      // A graph still owned by another execution lane must never be parked.
      // Only the selected priority request is active here; the check keeps
      // the helper safe if batching is enabled later.
      let busy = false;
      for (const [requestId, nodeId] of this.requestNodes) {
        if (
          nodeId === candidate.nodeId &&
          (this.activeUnitRequestIds.has(requestId) ||
            this.batchPinnedRequestIds.has(requestId))
        ) {
          busy = true;
          break;
        }
      }
      if (busy) {
        continue;
      }
      await this.stateManager.demote(candidate.nodeId, 'L2');
      // slotPark releases the native request identity. Clear every request
      // bound to this node so the next admission submits against the imported
      // slot instead of advancing a now-deleted native id.
      for (const [requestId, nodeId] of this.requestNodes) {
        if (nodeId !== candidate.nodeId) {
          continue;
        }
        const parked = this.scheduler.get(requestId);
        if (!parked) {
          continue;
        }
        parked.nativeRequestId = null;
        parked.slotId = null;
        parked.resumeSlotId = null;
      }
      const victim = this.lastRequestByNode.get(candidate.nodeId);
      if (victim) {
        this.emitResidency(victim, candidate.nodeId, {
          action: 'park',
          previousLevel: 'L1',
        });
      }
    }
  }

  private async pauseBeforeNativeUnit(request: RequestRecord): Promise<void> {
    if (!this.renderAwareEnabled) {
      return;
    }
    const rawPauseMs = this.desiredBurstPolicy.submissionPauseMs;
    const pauseMs = Number.isFinite(rawPauseMs)
      ? Math.max(0, Math.floor(rawPauseMs))
      : 0;
    if (pauseMs === 0) {
      return;
    }
    this.telemetry.emit(
      'render_yield',
      { pauseMs, level: this.desiredBurstPolicy.level },
      this.contextFor(request)
    );
    await this.renderPause(pauseMs);
  }

  private async syncGpuBurstPolicy(): Promise<void> {
    if (this.executionMode === 'deadline') {
      // The lease is the only source of the policy in this mode; the level
      // table's stale-release path would otherwise overwrite it.
      await this.pushGpuBurstPolicy();
      return;
    }
    const next = this.burstController.release(this.clock.now());
    if (!sameGpuBurstPolicy(next, this.desiredBurstPolicy)) {
      await this.setDesiredBurstPolicy(
        next,
        'stale',
        this.lastPressureClientId
      );
      return;
    }
    await this.pushGpuBurstPolicy();
  }

  // Deadline mode: the lease for this request is computed at the graph
  // boundary, pushed to the backend, and then handed to the execution
  // controller so the unit and the submitted GPU policy agree.
  private async leaseForRequest(
    request: RequestRecord
  ): Promise<ExecutionLease | null> {
    if (!this.leaseController || !this.renderAwareEnabled) {
      return null;
    }
    const now = this.clock.now();
    const foreground = request.lifecycle.foreground;
    const handoff =
      !foreground && this.scheduler.promotionPending(request.requestId, now);
    const priority =
      !foreground && this.scheduler.priorityServing(request.requestId, now);
    const hiddenContext =
      request.lifecycle.state === 'hidden' ||
      request.lifecycle.state === 'frozen';
    const lease = this.leaseController.lease({
      now,
      foreground,
      background: hiddenContext && !handoff && !priority,
      handoff,
      priority,
    });
    await this.applyLease(lease, request.session.clientId);
    return lease;
  }

  private recordNativeUnit(
    progress: NativeUnitProgress,
    batchSize = 1
  ): void {
    const counters = progress.counters;
    this.lastNativeUnit = {
      batchSize,
      elapsedMs: progress.elapsedMs,
      promptTokens: counters?.promptN ?? null,
      cacheTokens: counters?.cacheN ?? null,
      promptMs: counters?.promptMs ?? null,
      predictedTokens: counters?.predictedN ?? null,
      predictedMs: counters?.predictedMs ?? null,
      unitStop: progress.unitStop ?? null,
    };
  }

  private observeLeaseUnit(
    request: RequestRecord,
    lease: ExecutionLease,
    outcome: ExecutionOutcome,
    decodedBefore: number,
    timing: LeaseUnitTiming
  ): void {
    const progress = outcome.progress;
    const counters = progress.counters;
    const decodeTokens = Math.max(0, progress.decodeGenerated - decodedBefore);
    // How far this unit's GPU residency ran past the window the lease
    // admitted. The browser path cannot read a completion-wait counter, so
    // occupancy is the queue signal it reports.
    const occupancyOvershootMs = Math.max(
      0,
      progress.elapsedMs - lease.budgetMs
    );
    // llama.cpp times the prompt and prediction phases of every unit. Set
    // against the host wall time they separate GPU work from the queue wait
    // and host round trips a shared engine pays between graphs. Both are
    // null when the native build reported no timings for this unit.
    const nativeMs =
      counters?.promptMs === undefined && counters?.predictedMs === undefined
        ? null
        : (counters?.promptMs ?? 0) + (counters?.predictedMs ?? 0);
    this.lastLeaseUnit = {
      batchSize: timing.batchSize ?? 1,
      startedAtMs: timing.startedAtMs,
      endedAtMs: timing.endedAtMs,
      gapMs: Math.max(0, timing.gapMs),
      admissionMs: Math.max(0, timing.admissionMs),
      pauseMs: Math.max(0, timing.pauseMs),
      leaseId: lease.leaseId,
      reason: lease.reason,
      budgetMs: lease.budgetMs,
      elapsedMs: progress.elapsedMs,
      decodeTokens,
      occupancyOvershootMs,
      queueWaitMs: null,
      commandSubmitBatchSize: lease.commandSubmitBatchSize,
      maxInflightBatches: lease.maxInflightBatches,
      promptTokens: counters?.promptN ?? null,
      cacheTokens: counters?.cacheN ?? null,
      promptMs: counters?.promptMs ?? null,
      predictedTokens: counters?.predictedN ?? null,
      predictedMs: counters?.predictedMs ?? null,
      hostOverheadMs:
        nativeMs === null ? null : Math.max(0, progress.elapsedMs - nativeMs),
      unitStop: progress.unitStop ?? null,
      requestId: request.requestId,
      sessionId: request.session.sessionId,
      foreground: request.lifecycle.foreground,
      foregroundAtStart: timing.foregroundAtStart ?? request.lifecycle.foreground,
      maxDecodeTokens: lease.maxDecodeTokens,
      maxPrefillTokens: lease.maxPrefillTokens,
      stage: progress.stage,
    };
    this.recentLeaseUnits.push(this.lastLeaseUnit);
    if (this.recentLeaseUnits.length > LEASE_UNIT_HISTORY) {
      this.recentLeaseUnits.splice(
        0,
        this.recentLeaseUnits.length - LEASE_UNIT_HISTORY
      );
    }
    this.leaseController?.observeUnit({
      reason: lease.reason,
      decodeTokens: counters?.predictedN ?? decodeTokens,
      ...(counters?.predictedMs === undefined
        ? {}
        : { predictedMs: counters.predictedMs }),
      ...(counters?.promptN === undefined
        ? {}
        : { promptTokens: counters.promptN }),
      ...(counters?.promptMs === undefined
        ? {}
        : { promptMs: counters.promptMs }),
      elapsedMs: progress.elapsedMs,
      budgetMs: lease.budgetMs,
      occupancyOvershootMs,
    });
    this.telemetry.emit(
      'lease_unit',
      {
        leaseId: lease.leaseId,
        reason: lease.reason,
        budgetMs: lease.budgetMs,
        elapsedMs: progress.elapsedMs,
        decodeTokens,
        occupancyOvershootMs,
        queueWaitMs: null,
        maxInflightBatches: lease.maxInflightBatches,
        commandSubmitBatchSize: lease.commandSubmitBatchSize,
      },
      this.contextFor(request)
    );
  }

  /**
   * Tell the scheduler that this unit produced a decode token. The
   * activation debt that shaped the queue order is settled from the
   * observable event (a token the human could see), not from the unit
   * merely being admitted.
   */
  private noteDecodeProgress(
    request: RequestRecord,
    decodedBefore: number,
    now: number
  ): void {
    if (request.decodeGenerated <= decodedBefore) {
      return;
    }
    this.scheduler.noteDecodeProgress(request.requestId, now);
  }

  private async applyLease(
    lease: ExecutionLease,
    clientId: ClientId | null
  ): Promise<void> {
    this.activeLease = lease;
    const policy = leasePolicy(lease);
    if (sameGpuBurstPolicy(policy, this.desiredBurstPolicy)) {
      // Keep diagnostics on the newest lease id without re-pushing an
      // identical physical policy to the native worker every unit.
      this.desiredBurstPolicy = { ...policy };
      return;
    }
    const reason: 'pressure' | 'recovery' | 'stale' =
      lease.source === 'fallback' || lease.source === 'background'
        ? 'stale'
        : lease.reason === 'work-conserving'
          ? 'recovery'
          : 'pressure';
    await this.setDesiredBurstPolicy(policy, reason, clientId);
  }

  private leaseDiagnostics(): LeaseDiagnostics | null {
    const diagnostics = this.leaseController?.diagnostics({
      now: this.clock.now(),
    });
    return diagnostics
      ? {
          ...diagnostics,
          lastUnit: this.lastLeaseUnit,
          recentUnits: [...this.recentLeaseUnits],
        }
      : null;
  }

  private async setDesiredBurstPolicy(
    policy: GpuBurstPolicy,
    reason: 'pressure' | 'recovery' | 'stale' | 'initial',
    clientId: ClientId | null
  ): Promise<void> {
    const changed = !sameGpuBurstPolicy(policy, this.desiredBurstPolicy);
    this.desiredBurstPolicy = { ...policy };
    this.applyBurstShape();
    if (changed && clientId) {
      this.emitEvent({
        type: 'burst',
        clientId,
        policy: { ...this.desiredBurstPolicy },
        reason,
      });
    }
    await this.pushGpuBurstPolicy();
  }

  private async pushGpuBurstPolicy(): Promise<void> {
    if (
      this.appliedBurstPolicy &&
      sameGpuBurstPolicy(this.appliedBurstPolicy, this.desiredBurstPolicy)
    ) {
      return;
    }
    if (this.native.bound === false) {
      return;
    }
    const target = { ...this.desiredBurstPolicy };
    try {
      await this.native.setGpuBurstPolicy(target);
      if (sameGpuBurstPolicy(target, this.desiredBurstPolicy)) {
        this.appliedBurstPolicy = target;
      }
      this.telemetry.emit('gpu_burst_policy', { ...target });
    } catch (error) {
      this.telemetry.emit('gpu_burst_policy_error', {
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }

  private isFenceCurrent(clientId: ClientId): boolean {
    return this.attachedClients.has(clientId);
  }

  private isRequestFenceCurrent(request: RequestRecord): boolean {
    const fence = this.requestFences.get(request.requestId);
    if (!fence) {
      return true;
    }
    return (
      fence.generation === (this.fences.get(fence.clientId) ?? fence.generation)
    );
  }

  private bumpFence(clientId: ClientId): void {
    this.fences.set(clientId, (this.fences.get(clientId) ?? 0) + 1);
    for (const [requestId, request] of this.scheduler.pendingRequests().map(
      (item) => [item.requestId, item] as const
    )) {
      if (request.session.clientId === clientId) {
        const fence = this.requestFences.get(requestId);
        if (fence) {
          fence.generation = -1;
        }
      }
    }
  }

  private contextFor(request: RequestRecord): {
    requestId: string;
    clientId: string;
    sessionId: string;
  } {
    return {
      requestId: request.requestId,
      clientId: request.session.clientId,
      sessionId: request.session.sessionId,
    };
  }

  private emitEvent(event: CoordinatorEvent): void {
    for (const listener of this.eventListeners) {
      listener(event);
    }
  }

  private assertFence(request: RequestRecord): void {
    if (!this.isRequestFenceCurrent(request)) {
      throw new RequestFencedError(request.requestId);
    }
  }
}

// Cooperative yield for capacity waits. A zero-delay timer keeps the worker
// responsive without spinning the admission path at full speed.
function yieldToEventLoop(): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, 0);
  });
}

function sameGpuBurstPolicy(a: GpuBurstPolicy, b: GpuBurstPolicy): boolean {
  return (
    a.level === b.level &&
    a.commandSubmitBatchSize === b.commandSubmitBatchSize &&
    a.maxInflightBatches === b.maxInflightBatches &&
    a.submissionPauseMs === b.submissionPauseMs &&
    (a.source ?? 'fixed') === (b.source ?? 'fixed')
  );
}

// The lease already decided the submission shape; the level is only the
// probe-visible label for how much of a frame the lease admitted.
// Coordinator-side timing for one completed unit. Kept separate from the
// lease so the lease stays a pure controller.
interface LeaseUnitTiming {
  startedAtMs: number;
  endedAtMs: number;
  gapMs: number;
  admissionMs: number;
  pauseMs: number;
  batchSize?: number;
  // Lifecycle class at admission, before any later click could rewrite it.
  foregroundAtStart?: boolean;
}

// Enough history to hold one episode's units at the pressures this repo
// screens at, without growing the diagnostics payload without bound.
const LEASE_UNIT_HISTORY = 512;

function leasePolicy(lease: ExecutionLease): GpuBurstPolicy {
  const share =
    lease.frameBudgetMs > 0 ? lease.budgetMs / lease.frameBudgetMs : 1;
  const level: GpuBurstLevel =
    lease.reason === 'work-conserving' && !lease.maxInflightBatches
      ? 'normal'
      : share >= 0.75
        ? 'guarded'
        : 'constrained';
  return {
    level,
    commandSubmitBatchSize: lease.commandSubmitBatchSize,
    maxInflightBatches: lease.maxInflightBatches,
    // The lease's own rest interval: an empty-queue window is what the
    // renderer needs when the queue lease alone is not enough.
    submissionPauseMs: lease.submissionPauseMs,
    source: 'deadline',
    leaseId: lease.leaseId,
  };
}

function defaultRenderPause(pauseMs: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, pauseMs));
}

function burstLevelIndex(policy: GpuBurstPolicy): number {
  return GPU_BURST_POLICIES.findIndex((entry) => entry.level === policy.level);
}

function isFreshTurn(request: RequestRecord): boolean {
  return (
    request.prefillProcessed === 0 &&
    request.decodeGenerated === 0 &&
    request.outputChunks.length === 0
  );
}

function normalizeContinuationTokenCap(
  value: number | undefined
): number | undefined {
  if (value === undefined || !Number.isFinite(value)) {
    return undefined;
  }
  return Math.max(0, Math.floor(value));
}

function appendTurnText(prefix: string, turnText: string | undefined): string {
  if (!turnText) {
    return prefix;
  }
  if (!prefix) {
    return turnText;
  }
  const separator = turnText.startsWith(' ') ? '' : ' ';
  return `${prefix}${separator}${turnText}`;
}
function clampInteger(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) {
    return min;
  }
  return Math.min(max, Math.max(min, Math.floor(value)));
}
