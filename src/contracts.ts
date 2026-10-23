export type ClientId = string;
export type SessionId = string;
export type RequestId = string;
export type NodeId = string;
export type CheckpointKey = string;

export type LifecycleState = 'active' | 'passive' | 'hidden' | 'frozen';
export type ResidencyLevel = 'L1' | 'L2' | 'L3' | 'rebuild';
export type RequestStage = 'prefill' | 'decode';
export type NativeStage = RequestStage | 'done';

export interface LifecycleRecord {
  clientId: ClientId;
  epoch: number;
  state: LifecycleState;
  // High-priority marker for this session. The key keeps its historical name
  // because it is part of the lifecycle wire record that recorded runs and
  // archived artifacts are written against; the runtime treats it purely as
  // session priority and never derives it from tab visibility. Several
  // sessions may carry it at the same time.
  foreground: boolean;
  changedAt: number;
}

export interface SessionKey {
  clientId: ClientId;
  sessionId: SessionId;
}

export interface SessionNode {
  nodeId: NodeId;
  session: SessionKey;
  parentNodeId: NodeId | null;
  turnText: string;
  // exact prompt plus generated text for the latest completed turn; used
  // when a node has no native state and must be rebuilt from text
  continuationText?: string;
  tokenIds: number[];
  // tokens actually resident in the native KV cache (falls back to tokenIds.length)
  residentTokens?: number;
  createdAt: number;
  lastAccessAt: number;
  accessCount: number;
}

export interface RequestRecord {
  requestId: RequestId;
  session: SessionKey;
  parentNodeId: NodeId | null;
  // Explicit target node for this request. `parentNodeId` is retained as a
  // legacy alias for callers that predate context trees.
  nodeId?: NodeId | null;
  // The completed node whose immutable checkpoint seeds this request's
  // target node. When omitted, the target node is resumed directly.
  forkFromNodeId?: NodeId | null;
  // Internal coordinator marker: the target node has been registered but
  // its parent checkpoint has not been imported into a child slot yet.
  forkPending?: boolean;
  dependencyRequestIds: RequestId[];
  promptText?: string;
  promptTokens: number[];
  seed?: number;
  temperature?: number;
  maxNewTokens: number;
  lifecycle: LifecycleRecord;
  createdAt: number;
  enqueuedAt: number;
  waitBudgetMs: number;
  serviceEstimateMs: number;
  serviceMs: number;
  stage: RequestStage;
  prefillProcessed: number;
  decodeGenerated: number;
  // where decodeGenerated comes from: llama.cpp token counters, or the
  // streamed-text-chunk fallback used when a backend cannot report counters
  tokenAccounting: TokenAccounting;
  counters?: NativeTokenCounters;
  nativeRequestId: string | null;
  slotId: number | null;
  // slot that already holds this session's KV state and must be resumed
  resumeSlotId?: number | null;
  // tokens that llama.cpp served from the restored KV cache in the last unit
  prefillCached?: number;
  // total tokens currently resident in the native slot
  residentTokens?: number;
  // bounded units llama.cpp closed itself rather than by host abort
  boundedStops?: number;
  // why the most recent bounded unit ended
  unitStop?: NativeUnitStop;
  // exponential moving averages measured from native counters; the execution
  // controller sizes the next bounded unit so it lands inside the wall-clock
  // target instead of relying on the host abort
  estimatedDecodeTokensPerMs?: number;
  estimatedPrefillMsPerToken?: number;
  outputChunks: string[];
  // set only when this request is replayed from a rebuild path; the
  // execution controller prepends it before submitting to the native engine
  replayText?: string;
  replayTokens?: number[];
  // Keep the replay prefix even when the slot is already resident. A
  // context-tree child imports the parent KV, but llama.cpp still needs the
  // full logical prompt to match that resident prefix instead of treating
  // the child suffix as a fresh sequence.
  keepReplayPrefix?: boolean;
  // Attention bookkeeping. `foregroundAtMs` is the last time the owning
  // client became foreground and `lastDecodeAtMs` is the last time a unit
  // produced a decode token. While no token has arrived since the
  // Priority bookkeeping. `foregroundAtMs` is the last time the owning
  // session was promoted to high priority and `lastDecodeAtMs` is the last
  // time a unit produced a decode token. While no token has arrived since
  // the promotion the request still owes its caller progress, so the
  // scheduler keeps that debt ahead of ordinary work instead of letting a
  // burst of later promotions bury a session that has not answered yet.
  foregroundAtMs?: number;
  lastDecodeAtMs?: number;
  // the request carries its whole context, so a resumed slot must not have
  // the previous turn prepended. The slot still stays bound to the session
  // so the backend can match and reuse the shared KV prefix.
  replaceContext?: boolean;
}

// Experiment/ablation switch for how one cooperative unit is shaped.
export type ExecutionMode =
  | 'bounded'
  | 'unsliced'
  | 'deadline';

// - bounded: fixed deployment shape, native cooperative time budget on
// - unsliced: one unit carries the whole request
// - deadline: FrameDeadlineLease derives the budget from presented frames
// - fallback: bounded safety shape after a stale signal or queue timeout
// - background: page is not foreground; only the background slice is admitted
export type ExecutionLeaseSource =
  | 'deadline'
  | 'bounded'
  | 'fallback'
  | 'background';

export interface NativeUnitBudget {
  maxPrefillTokens: number;
  maxDecodeTokens: number;
  // Native task wall-clock target checked at token boundaries. This is the
  // cooperative scheduler quantum; it can overshoot by one native batch.
  nativeTimeBudgetMs?: number;
  // Lease this unit was admitted under, when a deadline lease shaped it.
  leaseId?: number;
  leaseSource?: ExecutionLeaseSource;
  // Host watchdog armed above the native target. It only fires when the
  // native runtime fails to return at a token boundary.
  wallTimeMs: number;
}

export type GpuBurstLevel = 'normal' | 'guarded' | 'constrained';

// Why the lease picked the shape it did. `work-conserving` means no render
// pressure is visible, so the lease spends the deployment's full budget;
// `frame-deadline` means the budget came from the presented-frame deadline;
// `queue-backpressure` means measured GPU queue wait broke the deadline proof
// and the lease fell back to a local barrier.
export type LeaseReason =
  | 'work-conserving'
  | 'frame-deadline'
  | 'queue-backpressure'
  | 'background'
  | 'stale-signal'
  | 'recovery';

export interface QueueLease {
  // Commands submitted before the backend re-reads the policy.
  commandSubmitBatchSize: number;
  // Command batches allowed in flight before the backend waits. 0 = no limit.
  maxInflightBatches: number;
  // How the capacity was derived; `local-barrier` is the no-queue-feedback
  // path that keeps at most one batch outstanding.
  basis: 'unlimited' | 'budget-share' | 'queue-wait' | 'local-barrier';
}

// One lease is the amount of GPU work the coordinator is willing to admit
// before the next graph boundary. It is recomputed from presented frames and
// measured token cost, so it is not a fixed quantum.
export interface ExecutionLease extends QueueLease {
  leaseId: number;
  source: ExecutionLeaseSource;
  reason: LeaseReason;
  budgetMs: number;
  maxDecodeTokens: number;
  maxPrefillTokens: number;
  // Time the coordinator spends away from the GPU before the next unit. The
  // queue lease bounds how deep our submissions get; this bounds how often
  // we submit at all, which is what gives a contended renderer an empty
  // queue window. It is derived from the presented tail, not configured.
  submissionPauseMs: number;
  // Frame budget of the display the signal came from (1000 / refreshHz).
  frameBudgetMs: number;
  // How far presented frames currently sit past that budget.
  frameDebtMs: number;
  // Measured GPU queue wait, when the backend could report it.
  queueWaitMs: number | null;
  createdAt: number;
}

export interface LeaseUnitDiagnostics {
  leaseId: number;
  reason: LeaseReason;
  budgetMs: number;
  elapsedMs: number;
  decodeTokens: number;
  // How far the unit's GPU residency ran past the window the lease admitted.
  // This is the occupancy signal the lease reacts to: the unit was still on
  // the queue after the deadline it was admitted under.
  occupancyOvershootMs: number;
  // Completion wait reported by a backend that can measure the queue
  // directly. Null on hosts without such a counter, which is the case for
  // the browser path; occupancy stays measurable there, a wait does not.
  queueWaitMs: number | null;
  commandSubmitBatchSize: number;
  maxInflightBatches: number;
  // Native accounting for the same unit. `elapsedMs` is host wall time and
  // covers queue wait and host round trips as well as GPU residence, so a
  // unit whose native terms are much smaller than its elapsed time waited
  // instead of computing. Null when the backend reported no counters.
  promptTokens: number | null;
  // Prompt tokens llama.cpp served from the resident KV prefix instead of
  // re-evaluating them. Null when the backend reports no counters.
  cacheTokens: number | null;
  promptMs: number | null;
  predictedTokens: number | null;
  predictedMs: number | null;
  // elapsed minus the native terms: the per-unit host and queue cost.
  hostOverheadMs: number | null;
  // Identity of the logical request this unit advanced, so a trace can be
  // aligned with the click or lifecycle event that caused it. Optional
  // because lease history predates this field in older artifacts.
  requestId?: string;
  sessionId?: string;
  foreground?: boolean;
  // Lifecycle class observed when the unit was admitted. The live record can
  // be rewritten by a later click while the unit is still running, so reading
  // only the current value misattributes foreground units in the trace.
  foregroundAtStart?: boolean;
  // Token caps the lease admitted for this unit.
  maxDecodeTokens?: number;
  maxPrefillTokens?: number;
  stage?: string;
  // How the unit ended, so a run shows whether llama.cpp closed the unit at
  // a graph boundary or the host watchdog aborted it.
  unitStop: NativeUnitStop | null;
  // Coordinator-side timing around the unit. `gapMs` is idle time between
  // the previous unit's end and this unit's start (pump wake-up and lease
  // computation). `pauseMs` is the render yield and `admissionMs` is the
  // residency admission / KV restore that runs before the native advance.
  startedAtMs?: number;
  endedAtMs?: number;
  gapMs?: number;
  admissionMs?: number;
  pauseMs?: number;
  // Number of logical requests advanced together at this graph boundary.
  // A value above one means the native server may run continuous batching
  // across independent sequences instead of paying one weight read per
  // request. The frame lease still bounds the batch as one GPU admission.
  batchSize?: number;
}

export interface LeaseDiagnostics {
  source: ExecutionLeaseSource;
  reason: LeaseReason;
  leaseId: number;
  budgetMs: number;
  maxDecodeTokens: number;
  maxPrefillTokens: number;
  queue: QueueLease;
  frameBudgetMs: number;
  frameDebtMs: number;
  refreshHz: number | null;
  signalAgeMs: number | null;
  // Measured GPU residency overshoot of the most recent unit, and the
  // rolling estimate the queue barrier is decided from.
  occupancyOvershootMs: number | null;
  queueWaitMs: number | null;
  submissionPauseMs?: number;
  measuredTokenCostMs?: number | null;
  // Clear-conditioned native prefill cost used to bound a prefill graph.
  // Pressure samples do not update this estimate because their timings also
  // include the renderer sharing the GPU queue.
  measuredPrefillMsPerToken?: number | null;
  // Most recently completed unit. It makes it possible to verify that a
  // lease change actually reached the native graph boundary during a run.
  lastUnit: LeaseUnitDiagnostics | null;
  // Bounded ring of recent units, oldest first. Sampled diagnostics at the
  // end of an episode therefore carry the unit timeline of that episode,
  // which is what separates native compute from scheduling gaps.
  recentUnits?: LeaseUnitDiagnostics[];
  // number of times the lease dropped to a safety shape, and when
  recoveries: number;
  lastRecoveryAt: number | null;
  lastRecoveryReason: LeaseReason | null;
}

// Dynamic WebGPU submission policy. maxInflightBatches = 0 means unlimited;
// commandSubmitBatchSize is capped by the value used to size the parameter arena.
export interface GpuBurstPolicy {
  level: GpuBurstLevel;
  commandSubmitBatchSize: number;
  maxInflightBatches: number;
  // Coordinator-side yield inserted before the next native unit. This is not
  // sent to the backend; it gives rendering a window between submissions.
  submissionPauseMs: number;
  // Which controller produced this policy; defaults to `fixed`.
  source?: GpuBurstPolicySource;
  // Lease that produced the shape, when source is `deadline`.
  leaseId?: number;
}

// `fixed` policies come from the level table; `deadline` policies were derived
// by FrameDeadlineLease. leaseId lets a harness line a policy change up with
// the unit that ran under it.
export type GpuBurstPolicySource = 'fixed' | 'deadline';

export type RenderPressureSource =
  | 'presented-frame'
  | 'animation-frame'
  | 'synthetic';

export interface RenderPressureSignal extends FrameHealthTail {
  clientId: ClientId;
  epoch: number;
  sampledAt: number;
  source: RenderPressureSource;
  frameCount: number;
  p50Ms: number;
  p95Ms: number;
  maxMs: number;
  overBudgetRatio: number;
  // True when the sample came from a canvas the page itself presents (the
  // application render target). False marks the monitor's
  // synthetic helper canvas, which keeps stall detection alive but is not
  // evidence that a renderer is competing for the queue. Undefined keeps
  // older clients on the previous behaviour.
  pageOwned?: boolean;
}

// Extended presented-frame health. The tail fields are what a deadline lease
// needs: a P95 can hide one long stall, and the display budget is the unit
// the lease reasons in. `refreshHz` is optional because a headless or
// occluded page cannot always measure its display; the lease falls back to
// 60 Hz only when the signal omits it.
export interface FrameHealthTail {
  p99Ms: number;
  over33Ratio: number;
  over50Ratio: number;
  // longest run of consecutive over-budget frames, in milliseconds
  longestStallMs: number;
  refreshHz?: number;
}

// Counters reported by llama.cpp for one bounded unit. `promptN` counts prompt
// tokens the unit evaluated, `cacheN` counts prompt tokens it reused from the
// slot key-value cache, and `predictedN` counts tokens the unit decoded.
export interface NativeTokenCounters {
  promptN: number;
  cacheN: number;
  predictedN: number;
  promptMs?: number;
  predictedMs?: number;
}

export type TokenAccounting = 'native-token' | 'stream-chunk';

// Why the most recent bounded unit ended. The `native-*` variants mean
// llama.cpp closed the task itself at a token boundary with the slot KV
// retained, so no work was aborted mid-token.
// - native-decode-budget: the per-unit n_decode_max token budget ran out
// - native-prefill-budget: the per-unit n_prefill_max prompt chunk ran out
// - native-time-budget: the best-effort per-unit time budget ran out
// - native-length: a backend reported a length stop without the native flag
// - host-abort: the host wall-clock guard aborted an in-flight request
// - complete: the request finished, so there is no boundary to resume
export type NativeUnitStop =
  | 'native-decode-budget'
  | 'native-prefill-budget'
  | 'native-time-budget'
  | 'native-length'
  | 'host-abort'
  | 'complete';

export interface NativeUnitProgress {
  requestId: string;
  stage: NativeStage;
  prefillProcessed: number;
  prefillTotal: number;
  decodeGenerated: number;
  decodeTotal: number;
  done: boolean;
  cancelled: boolean;
  isError: boolean;
  outputChunks: string[];
  elapsedMs: number;
  slotId: number | null;
  // token accounting source for decodeGenerated, with the exact counters when
  // the backend reported them for this unit
  tokenAccounting: TokenAccounting;
  counters?: NativeTokenCounters;
  // prompt tokens served from the slot KV cache instead of being recomputed
  prefillCached?: number;
  // total tokens resident in the native slot after this unit
  residentTokens?: number;
  // bounded units llama.cpp closed at a token boundary (not by host abort)
  boundedStops?: number;
  // why this bounded unit ended; the native-* values mean llama.cpp closed
  // it at a token boundary with the slot KV retained
  unitStop?: NativeUnitStop;
}

export interface NativeTokenChunk {
  requestId: string;
  text: string;
  // Exact native decode position after this chunk, when the backend reports
  // token counters. Falls back to the streamed chunk count otherwise.
  decodeGenerated: number;
  // Time since this native unit started. The page still timestamps arrival
  // locally for end-to-end TTFT; this value is diagnostic only.
  elapsedMs: number;
}

export interface NativeSubmitRequest {
  requestId: string;
  promptText?: string;
  promptTokens: number[];
  maxNewTokens: number;
  seed?: number;
  temperature?: number;
  resumeSlotId?: number;
  // true when this request starts a new turn from a restored session slot
  newTurn?: boolean;
  // true when the request carries its whole context: the resumed slot keeps
  // its KV, but the previous turn is not prepended to the prompt
  replaceContext?: boolean;
}

export interface NativeSlotCheckpoint {
  // engine-level continuation metadata (JSON)
  bytes: Uint8Array;
  // real sequence KV bytes when the native backend supports bounded serving
  kv?: Uint8Array;
  // set when the state lives in the native heap behind a park handle instead
  // of in host-side bytes; tier two keeps this and releases the live slot
  parkedHandle?: number;
  // byte footprint of the native copy, for residency accounting
  parkedByteLength?: number;
  tokens: number[];
  stage: RequestStage;
  decodeGenerated: number;
  formatVersion: number;
  checksum: string;
}

export interface CheckpointEnvelope extends NativeSlotCheckpoint {
  checkpointKey: CheckpointKey;
  nodeId: NodeId;
  session: SessionKey;
  // Immutable parent lineage for a logical fork. Optional so checkpoints
  // written before context trees remain recoverable.
  parentNodeId?: NodeId | null;
  modelHash: string;
  createdAt: number;
}

export interface NativeSlotStats {
  slotId: number;
  inUse: boolean;
  tokens: number;
  state: NativeStage | 'idle';
  // A bound slot can still be empty before its first native stream starts.
  // Only parkable slots may be exported or moved out of L1.
  parkable?: boolean;
}

export type NativeRuntimePhase = 'download' | 'load' | 'ready' | 'error';

export interface NativeRuntimeEvent {
  type: 'runtime';
  phase: NativeRuntimePhase;
  modelUrl: string;
  loaded?: number;
  total?: number;
  message?: string;
}

export interface SchedulerConfig {
  maxPending: number;
  waitBaseMs: number;
  waitDepthFactor: number;
  serviceBoundMs: number;
  waitBudgetCapMs: number;
  graceMs: number;
  retentionWindowMs: number;
  maxProtectedHeads: number;
  estimateWarmupRequests: number;
  // A burst of promotions creates several unanswered debts. An older debt is
  // raised after this much waiting so a later promotion cannot starve an
  // earlier session indefinitely.
  priorityFairnessMs: number;
  // How long an unresolved promotion keeps its priority after the session has
  // been returned to `normal`. Within the window the request is served ahead
  // of ordinary work until it produces its next decode token, so a promotion
  // that lands in the middle of a prefill chunk is not silently deferred
  // behind every later promotion.
  priorityHandoffMs: number;
  // How long an already answered promotion keeps the priority serving class
  // after its session has been returned to `normal`. The answered burst
  // stays in the priority lane long enough to finish the unit that is
  // already in flight; past this window the request is ordinary work again,
  // so a chain of promotions cannot leave every past target holding the lane
  // and its long pressure-shaped units for the rest of the request.
  priorityDecayMs: number;
  // How many consecutive units the high-priority lane may take once it has
  // already answered, before the scheduler serves normal work for one unit.
  // High priority keeps its rank; this only bounds how long a normal session
  // can be starved while one session stays promoted.
  highPriorityBurstUnits: number;
  // How many consecutive units the current resident sequence may take
  // before overdue cold work can preempt it. This bounds checkpoint churn
  // without allowing a resident sequence to starve the other sessions.
  residentBurstUnits: number;
}

export interface SelectionResult {
  request: RequestRecord;
  runnable: boolean;
  blockedBy: RequestId[];
}

export interface EnqueueResult {
  accepted: boolean;
  reason?: 'stale_lifecycle' | 'pending_limit' | 'duplicate';
}

export interface StateManagerConfig {
  l1TokenCap: number;
  // How many sessions may hold a serving slot at once. This is the native
  // slot count, which the token cap alone does not capture: four short
  // sessions fit the token budget but still need four live slots.
  maxResidentNodes?: number;
  l2TokenCap: number;
  l3ByteCap: number;
  repairBudget: number;
  replacementPolicy: 'hybrid' | 'lru' | 'ttl';
  // E4 policy switches. Defaults preserve the production hybrid path.
  lifecycleProtection?: boolean;
  evictOnPause?: boolean;
  admissionPolicy?:
    | 'monotonic-bounded'
    | 'reactive'
    | 'immediate-reject';
}

export interface ResidencyRecord {
  nodeId: NodeId;
  level: ResidencyLevel;
  slotId: number | null;
  tokenCount: number;
  checkpointKey: CheckpointKey | null;
  lastAccessAt: number;
  accessCount: number;
}

export interface StateAdmissionResult {
  success: boolean;
  nodeId: NodeId;
  targetLevel: ResidencyLevel;
  steps: number;
  victims: NodeId[];
  partial: boolean;
  residentLevel: ResidencyLevel;
  reason?:
    | 'protected_capacity'
    | 'budget_exhausted'
    | 'checkpoint_mismatch'
    | 'store_error';
}

export interface RestoreResult extends StateAdmissionResult {
  source: ResidencyLevel;
  replayedTokens: number;
}

export interface ExecutionConfig {
  maxPrefillTokens: number;
  baseDecodeTokens: number;
  foregroundDecodeTokens: number;
  backgroundDecodeTokens: number;
  // Ceiling for a hidden tab's prefill chunk. Prefill and decode have very
  // different costs, so one hidden-tab ceiling cannot serve both: a decode
  // sized quantum leaves a hidden prompt crawling for tens of seconds,
  // which delays every tab's first token and the click protocol built on
  // top of it. Defaults to backgroundDecodeTokens so existing deployments
  // keep their previous shape.
  backgroundPrefillTokens?: number;
  prefillMsPerToken: number;
  maxUnitWallTimeMs: number;
  // the host wall-clock guard is a watchdog, not the unit boundary: it has to
  // tolerate one slow native batch, otherwise it aborts units the native
  // token budget was about to close cleanly
  unitWallTimeGraceMs: number;
  // adaptive bounded units: observed native throughput sets the next unit's
  // token budget so units stay inside targetUnitWallTimeMs
  adaptiveUnits: boolean;
  targetUnitWallTimeMs: number;
  // set false for the unsliced control: the request still gets the host
  // watchdog, but llama.cpp does not close it on the cooperative time quantum
  nativeTimeBudgetEnabled: boolean;
  minDecodeTokens: number;
  maxDecodeTokens: number;
}

export interface ExecutionOutcome {
  request: RequestRecord;
  progress: NativeUnitProgress;
  budget: NativeUnitBudget;
  requeued: boolean;
  completed: boolean;
}

export interface BlobStore {
  put(key: string, value: Uint8Array): Promise<void>;
  get(key: string): Promise<Uint8Array | null>;
  delete(key: string): Promise<void>;
  listKeys(): Promise<string[]>;
}

export interface Clock {
  now(): number;
}

export const systemClock: Clock = {
  now: () => Date.now(),
};

export class WllmError extends Error {
  readonly code: string;

  constructor(message: string, code: string) {
    super(message);
    this.name = 'WllmError';
    this.code = code;
  }
}

export class StaleLifecycleError extends WllmError {
  constructor(clientId: ClientId, epoch: number) {
    super(`stale lifecycle epoch for ${clientId}: ${epoch}`, 'stale_lifecycle');
    this.name = 'StaleLifecycleError';
  }
}

export class CapacityError extends WllmError {
  constructor(message: string) {
    super(message, 'capacity');
    this.name = 'CapacityError';
  }
}

export class CheckpointError extends WllmError {
  constructor(message: string) {
    super(message, 'checkpoint');
    this.name = 'CheckpointError';
  }
}

export class RequestFencedError extends WllmError {
  constructor(requestId: RequestId) {
    super(`request is fenced: ${requestId}`, 'request_fenced');
    this.name = 'RequestFencedError';
  }
}

export function sessionKeyEquals(a: SessionKey, b: SessionKey): boolean {
  return a.clientId === b.clientId && a.sessionId === b.sessionId;
}

export function sessionKeyString(key: SessionKey): string {
  return `${key.clientId}\u0000${key.sessionId}`;
}
