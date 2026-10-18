/**
 * Shared-coordinator engine page used by the `ours-*` experiment arms.
 *
 * The page owns one SharedWorker coordinator and exposes the same narrow
 * `window.__wllmEnginePage` surface as the per-tab baselines. Multiple logical
 * sessions live behind that one coordinator, which is the deployment
 * shape the paper's scheduling claim is about. The external baselines keep
 * their own one-engine-per-tab pages and cannot reach this object.
 */

import type {
  CoordinatorDiagnostics,
  CoordinatorEvent,
} from '../transport/protocol';
import type {
  LifecycleRecord,
  NativeRuntimeEvent,
  RequestId,
  RequestRecord,
} from '../contracts';
import type { GpuBurstPolicy, RenderPressureSignal } from '../contracts';
import type { FramePressureSample } from '../client/frame-pressure-monitor';
import type {
  BrowserEngineGeneration,
  BrowserEngineGenerateRequest,
  BrowserEngineMetrics,
  BrowserEngineSessionOptions,
  BrowserEngineStopReason,
} from '../engine-page-types';
import type { BrowserEngineFrameMetrics } from '../engine-page-types';
import {
  installEnginePage,
  presentCoordinatorGlobals,
  round2,
  type EnginePageApi,
  type EnginePageStatus,
  type EngineProgressEvent,
} from './engine-page-api';
import type {
  PriorityAuditEntry,
  PriorityUpdate,
  SessionPriority,
} from './engine-page-api';

export interface WllmPageClient {
  attach(clientId: string, lifecycle: LifecycleRecord): Promise<void>;
  detach(clientId: string): Promise<void>;
  setLifecycle(lifecycle: LifecycleRecord): Promise<boolean>;
  reportRenderPressure(signal: RenderPressureSignal): Promise<GpuBurstPolicy>;
  submit(
    request: Omit<RequestRecord, 'lifecycle'>,
    lifecycle?: LifecycleRecord
  ): Promise<{ accepted: boolean; reason?: string }>;
  cancel(requestId: RequestId): Promise<boolean>;
  status(requestId: RequestId): Promise<RequestRecord | null>;
  diagnostics(): Promise<CoordinatorDiagnostics>;
}

export interface WllmPageBridge {
  ensure(): Promise<void>;
  release(): void;
}

export interface WllmEnginePageOptions {
  client: () => WllmPageClient;
  bridge: () => WllmPageBridge;
  engineVersion: string;
  modelUrl: string;
  executionMode: string;
  priorityMode?: 'enabled' | 'disabled';
  initialPriority?: SessionPriority;
  requestTimeoutMs?: number;
  readyTimeoutMs?: number;
  frameMetrics?: () => BrowserEngineFrameMetrics | null;
  onSessionCreated?: (sessionId: string, clientId: string) => void;
  now?: () => number;
  scope?: Record<string, unknown>;
}

export interface WllmEnginePageApi extends EnginePageApi {
  reportRenderPressure(sample: FramePressureSample): Promise<void>;
  setSessionPriority(
    sessionId: string,
    level: SessionPriority
  ): Promise<PriorityUpdate>;
  priorityLog(): readonly PriorityAuditEntry[];
  handleEvent(event: CoordinatorEvent): void;
  handleRuntimeEvent(event: NativeRuntimeEvent): void;
}

interface SessionState {
  sessionId: string;
  clientId: string;
  epoch: number;
  activeRequestId: string | null;
  attached: boolean;
  priority: SessionPriority;
}

interface ActiveGeneration {
  sessionId: string;
  requestId: string;
  startedAt: number;
  maxNewTokens: number;
  firstTokenAt: number | null;
  lastTokenAt: number | null;
  firstChunkTokens: number | null;
  completedAt: number | null;
  units: number;
  decodeTokens: number;
  decodeMs: number;
  prefillTokens: number;
  prefillMs: number | null;
  cacheTokens: number | null;
  prefillProcessedTokens: number;
  text: string;
  cancelled: boolean;
  lastProgress: {
    stage: string;
    prefillProcessed: number;
    prefillTotal: number;
    decodeGenerated: number;
    elapsedMs: number;
  } | null;
  resolve: (value: BrowserEngineGeneration) => void;
  reject: (error: Error) => void;
  timer: ReturnType<typeof setTimeout> | null;
}

interface RunTotals {
  requests: number;
  decodeTokens: number;
  decodeMs: number;
  units: number;
}

export function createWllmEnginePage(
  options: WllmEnginePageOptions
): WllmEnginePageApi {
  const now = options.now ?? (() => performance.now());
  const scope =
    options.scope ?? (globalThis as unknown as Record<string, unknown>);
  const requestTimeoutMs = options.requestTimeoutMs ?? 10 * 60_000;
  const readyTimeoutMs = options.readyTimeoutMs ?? 5 * 60_000;
  const sessions = new Map<string, SessionState>();
  const active = new Map<string, ActiveGeneration>();
  const priorityAudit: PriorityAuditEntry[] = [];
  const progressListeners = new Set<
    (event: EngineProgressEvent) => void
  >();
  const totals: RunTotals = {
    requests: 0,
    decodeTokens: 0,
    decodeMs: 0,
    units: 0,
  };
  let ready = false;
  let readyError: string | null = null;
  let readyWaiters: (() => void)[] = [];
  let readyRejecters: ((error: Error) => void)[] = [];
  let readyTimer: ReturnType<typeof setTimeout> | null = null;
  let prewarmPromise: Promise<void> | null = null;

  function status(): EnginePageStatus {
    return {
      engine: 'wllm',
      engineVersion: options.engineVersion,
      ready,
      loading: prewarmPromise !== null && !ready && readyError === null,
      error: readyError,
      loadProgress: ready ? 1 : null,
      modelUrl: options.modelUrl,
      sessions: [...sessions.keys()],
      activeRequests: [...active.keys()],
      coordinatorGlobals: presentCoordinatorGlobals(scope),
      priorityMode: options.priorityMode ?? 'enabled',
      initialPriority: options.initialPriority ?? 'high',
    };
  }

  function markReady(): void {
    if (ready) {
      return;
    }
    ready = true;
    readyError = null;
    if (readyTimer !== null) {
      clearTimeout(readyTimer);
      readyTimer = null;
    }
    const waiters = readyWaiters;
    readyWaiters = [];
    readyRejecters = [];
    for (const waiter of waiters) {
      waiter();
    }
  }

  function markReadyError(error: unknown): void {
    const message = describe(error);
    readyError = message;
    if (readyTimer !== null) {
      clearTimeout(readyTimer);
      readyTimer = null;
    }
    const rejecters = readyRejecters;
    readyWaiters = [];
    readyRejecters = [];
    for (const rejecter of rejecters) {
      rejecter(new Error(message));
    }
  }

  async function prewarm(): Promise<void> {
    if (ready) {
      return;
    }
    if (readyError !== null) {
      throw new Error(readyError);
    }
    if (prewarmPromise === null) {
      prewarmPromise = prewarmOnce().finally(() => {
        prewarmPromise = null;
      });
    }
    await prewarmPromise;
  }

  async function prewarmOnce(): Promise<void> {
    if (readyTimer === null) {
      readyTimer = setTimeout(() => {
        // A browser with an already-warm model can miss the runtime event if
        // the coordinator bound before this page subscribed. `nativeBound`
        // is then the only cross-process signal left, so treat it as ready
        // after a grace period instead of hanging the whole run.
        void options
          .client()
          .diagnostics()
          .then((diagnostics) => {
            if (diagnostics.runtime.nativeBound) {
              markReady();
            } else {
              markReadyError(new Error('inference worker did not bind'));
            }
          })
          .catch(markReadyError);
      }, 1_000);
    }
    await options.bridge().ensure();
    await new Promise<void>((resolve, reject) => {
      if (ready) {
        resolve();
        return;
      }
      readyWaiters.push(resolve);
      readyRejecters.push(reject);
      if (readyTimer === null) {
        readyTimer = setTimeout(() => {
          reject(new Error(`model did not become ready within ${readyTimeoutMs} ms`));
        }, readyTimeoutMs);
      }
    });
  }

  async function createSession(
    sessionOptions: BrowserEngineSessionOptions
  ): Promise<void> {
    const existing = sessions.get(sessionOptions.sessionId);
    if (existing) {
      return;
    }
    const clientId = `wllm-page:${sessionOptions.sessionId}`;
    const state: SessionState = {
      sessionId: sessionOptions.sessionId,
      clientId,
      epoch: 1,
      activeRequestId: null,
      attached: false,
      priority: options.initialPriority ?? 'high',
    };
    sessions.set(state.sessionId, state);
    const lifecycle = lifecycleFor(
      state,
      effectivePriority(state, options.priorityMode)
    );
    await options.client().attach(state.clientId, lifecycle);
    state.attached = true;
    options.onSessionCreated?.(state.sessionId, state.clientId);
  }

  function requireSession(sessionId: string): SessionState {
    const session = sessions.get(sessionId);
    if (!session) {
      throw new Error(`unknown session ${sessionId}`);
    }
    return session;
  }

  async function generate(
    sessionId: string,
    request: BrowserEngineGenerateRequest
  ): Promise<BrowserEngineGeneration> {
    const session = requireSession(sessionId);
    if (session.activeRequestId !== null) {
      throw new Error(
        `session ${sessionId} already has a running request on this coordinator`
      );
    }
    await prewarm();
    const client = options.client();
    const requestId = uniqueRequestId(request.requestId, active);
    const startedAt = now();
    const lifecycle = lifecycleFor(
      session,
      effectivePriority(session, options.priorityMode)
    );
    const promptTokens = estimateTokens(request.promptText);
    const payload: Omit<RequestRecord, 'lifecycle'> = {
      requestId,
      session: { clientId: session.clientId, sessionId: session.sessionId },
      parentNodeId: request.nodeId ?? session.sessionId,
      nodeId: request.nodeId ?? session.sessionId,
      ...(request.forkFromNodeId === undefined
        ? {}
        : { forkFromNodeId: request.forkFromNodeId }),
      dependencyRequestIds: [],
      promptText: request.promptText,
      promptTokens,
      maxNewTokens: Math.max(1, request.maxNewTokens),
      ...(request.seed === undefined ? {} : { seed: request.seed }),
      ...(request.temperature === undefined
        ? {}
        : { temperature: request.temperature }),
      createdAt: Date.now(),
      enqueuedAt: Date.now(),
      waitBudgetMs: 0,
      serviceEstimateMs: 0,
      serviceMs: 0,
      stage: 'prefill',
      prefillProcessed: 0,
      decodeGenerated: 0,
      tokenAccounting: 'native-token',
      nativeRequestId: null,
      slotId: null,
      outputChunks: [],
    };
    const generation = new Promise<BrowserEngineGeneration>((resolve, reject) => {
      const record: ActiveGeneration = {
        sessionId: session.sessionId,
        requestId,
        startedAt,
        maxNewTokens: Math.max(1, request.maxNewTokens),
        firstTokenAt: null,
        lastTokenAt: null,
        firstChunkTokens: null,
        completedAt: null,
        units: 0,
        decodeTokens: 0,
        decodeMs: 0,
        prefillTokens: 0,
        prefillMs: null,
        cacheTokens: null,
        prefillProcessedTokens: 0,
        text: '',
        cancelled: false,
        lastProgress: null,
        resolve,
        reject,
        timer: null,
      };
      record.timer = setTimeout(() => {
        void client.cancel(requestId).catch(() => undefined);
        finishGeneration(record, 'cancelled', `request timed out after ${requestTimeoutMs} ms`);
      }, requestTimeoutMs);
      active.set(requestId, record);
      session.activeRequestId = requestId;
    });

    const submitted = await client.submit(payload, lifecycle);
    if (!submitted.accepted) {
      const record = active.get(requestId);
      if (record) {
        finishGeneration(
          record,
          'failed',
          `request rejected: ${submitted.reason ?? 'unknown'}`
        );
      }
      throw new Error(`request rejected: ${submitted.reason ?? 'unknown'}`);
    }
    return generation;
  }

  async function cancel(sessionId: string): Promise<void> {
    const session = requireSession(sessionId);
    const requestId = session.activeRequestId;
    if (requestId === null) {
      return;
    }
    const record = active.get(requestId);
    if (record) {
      record.cancelled = true;
    }
    await options.client().cancel(requestId);
  }

  async function setSessionPriority(
    sessionId: string,
    level: SessionPriority
  ): Promise<PriorityUpdate> {
    const session = requireSession(sessionId);
    const atMs = Date.now();
    const applied = options.priorityMode !== 'disabled';
    if (session.priority === level) {
      const update = {
        accepted: true,
        applied,
        epoch: session.epoch,
        level,
        atMs,
      } satisfies PriorityUpdate;
      recordPriority(sessionId, update);
      return update;
    }
    session.epoch += 1;
    session.priority = level;
    const accepted = await options.client().setLifecycle(
      lifecycleFor(
        session,
        effectivePriority(session, options.priorityMode)
      )
    );
    const update = {
      accepted,
      applied,
      epoch: session.epoch,
      level,
      atMs,
    } satisfies PriorityUpdate;
    recordPriority(sessionId, update);
    return update;
  }

  function recordPriority(
    sessionId: string,
    update: PriorityUpdate
  ): void {
    priorityAudit.push({ sessionId, ...update });
    if (priorityAudit.length > 1024) {
      priorityAudit.splice(0, priorityAudit.length - 1024);
    }
  }

  function priorityLog(): readonly PriorityAuditEntry[] {
    return priorityAudit.map((entry) => ({ ...entry }));
  }

  // The evaluation harness samples the foreground render page out of process
  // and injects that signal here. Keeping the injection on the engine page
  // means the coordinator takes the same path as a browser-owned frame
  // monitor, while evaluation stays independent of `src/`.
  async function reportRenderPressure(
    sample: FramePressureSample
  ): Promise<void> {
    const session =
      [...sessions.values()].find((candidate) => candidate.priority === 'high') ??
      sessions.values().next().value;
    if (!session) {
      return;
    }
    const sampledAt = Date.now();
    const signal: RenderPressureSignal = {
      clientId: session.clientId,
      epoch: session.epoch,
      sampledAt,
      source: sample.source,
      frameCount: sample.frameCount,
      p50Ms: sample.p50Ms,
      p95Ms: sample.p95Ms,
      p99Ms: sample.p99Ms,
      maxMs: sample.maxMs,
      overBudgetRatio: sample.overBudgetRatio,
      over33Ratio: sample.over33Ratio,
      over50Ratio: sample.over50Ratio,
      longestStallMs: sample.longestStallMs,
      ...(sample.refreshHz === undefined ? {} : { refreshHz: sample.refreshHz }),
      ...(sample.pageOwned === undefined ? {} : { pageOwned: sample.pageOwned }),
    };
    scope.__wllmRenderPressure = signal;
    // The evaluation relay polls this hook from another page. Waiting for
    // the coordinator RPC here made the poll cadence collapse whenever a
    // native unit was in flight, so the lease often saw a stale or missing
    // sample. The local copy is the immediately observable contract; the
    // coordinator consumes the same signal asynchronously at its next
    // graph boundary.
    void options.client().reportRenderPressure(signal).catch(() => undefined);
  }

  async function disposeSession(sessionId: string): Promise<void> {
    const session = requireSession(sessionId);
    if (session.activeRequestId !== null) {
      await cancel(sessionId);
    }
    if (session.attached) {
      await options.client().detach(session.clientId).catch(() => undefined);
    }
    sessions.delete(sessionId);
  }

  async function metrics(): Promise<BrowserEngineMetrics> {
    const diagnostics = await options.client().diagnostics();
    void diagnostics;
    return {
      engine: 'wllm',
      frame: options.frameMetrics?.() ?? null,
      decodeTokensPerSecond:
        totals.decodeMs > 0 ? (totals.decodeTokens / totals.decodeMs) * 1000 : null,
      prefillTokensPerSecond: null,
      ttftMs: null,
      unitsPerRequest:
        totals.requests > 0 ? round2(totals.units / totals.requests) : null,
      usedJsHeapBytes: null,
      deviceMemoryBytes: null,
      // The lease is read at the end of the block, so an artifact can show
      // which reason shaped the units instead of inferring it from a policy
      // level. Baselines never reach this page.
      lease: diagnostics.runtime.lease,
    };
  }

  function handleRuntimeEvent(event: NativeRuntimeEvent): void {
    if (event.phase === 'ready') {
      markReady();
    } else if (event.phase === 'error') {
      markReadyError(new Error(event.message ?? 'native runtime error'));
    }
  }

  function handleEvent(event: CoordinatorEvent): void {
    if (event.type === 'runtime') {
      handleRuntimeEvent(event);
      return;
    }
    if (event.type === 'residency') {
      emitProgress({
        sessionId: event.sessionId,
        requestId: event.requestId,
        type: 'state',
        state: event.state,
        decodeGenerated: 0,
        timestampMs: event.state.timestampMs,
      });
      return;
    }
    const requestId = 'requestId' in event ? event.requestId : null;
    if (requestId === null) {
      return;
    }
    const record = active.get(requestId);
    if (!record) {
      return;
    }
    if (event.type === 'token') {
      if (record.firstTokenAt === null) {
        record.firstTokenAt = now();
        record.firstChunkTokens = event.decodeGenerated;
      }
      record.lastTokenAt = now();
      record.text += event.text;
      record.decodeTokens = Math.max(
        record.decodeTokens,
        event.decodeGenerated
      );
      emitProgress({
        sessionId: record.sessionId,
        requestId: record.requestId,
        type: 'token',
        text: event.text,
        decodeGenerated: record.decodeTokens,
        timestampMs: Date.now(),
      });
      return;
    }
    if (event.type === 'unit') {
      const before = record.decodeTokens;
      const progress = event.progress;
      record.units += 1;
      record.lastProgress = {
        stage: progress.stage,
        prefillProcessed: progress.prefillProcessed,
        prefillTotal: progress.prefillTotal,
        decodeGenerated: progress.decodeGenerated,
        elapsedMs: progress.elapsedMs,
      };
      // `prefillTotal` is the context position the native runtime reports, so
      // it keeps growing while the request decodes (prompt + generated). The
      // first non-zero value is the prompt length, which is what this metric
      // claims to be; taking the max over units overstated it by the whole
      // decode (891 reported for a 636-token prompt with 256 tokens generated).
      if (record.prefillTokens === 0 && progress.prefillTotal > 0) {
        record.prefillTokens = progress.prefillTotal;
      }
      const counters = progress.counters;
      if (counters?.cacheN !== undefined) {
        record.cacheTokens = Math.max(record.cacheTokens ?? 0, counters.cacheN);
      }
      if (counters?.promptN !== undefined) {
        record.prefillProcessedTokens += Math.max(0, counters.promptN);
      }
      if (progress.outputChunks.length > 0 && record.firstTokenAt === null) {
        record.firstTokenAt = now();
        // A sliced engine hands over a whole unit at once, so remember how
        // many tokens that first handover already contained.
        record.firstChunkTokens = progress.decodeGenerated;
      }
      if (progress.decodeGenerated > before) {
        record.decodeMs += progress.elapsedMs;
      }
      record.decodeTokens = Math.max(record.decodeTokens, progress.decodeGenerated);
      return;
    }
    if (event.type === 'complete') {
      record.text = event.output;
      finishGeneration(record, 'complete');
      return;
    }
    if (event.type === 'failed') {
      finishGeneration(record, 'failed', event.error);
      return;
    }
    if (event.type === 'fenced') {
      finishGeneration(record, 'cancelled', 'request was fenced');
    }
  }

  function finishGeneration(
    record: ActiveGeneration,
    outcome: 'complete' | 'cancelled' | 'failed',
    error?: string
  ): void {
    if (record.completedAt !== null) {
      return;
    }
    record.completedAt = now();
    emitProgress({
      sessionId: record.sessionId,
      requestId: record.requestId,
      type: outcome === 'complete' ? 'complete' : 'failed',
      decodeGenerated: record.decodeTokens,
      timestampMs: Date.now(),
      ...(outcome === 'complete'
        ? {}
        : { error: error ?? `generation ${outcome}` }),
    });
    if (record.timer !== null) {
      clearTimeout(record.timer);
      record.timer = null;
    }
    active.delete(record.requestId);
    const session = [...sessions.values()].find(
      (candidate) => candidate.activeRequestId === record.requestId
    );
    if (session) {
      session.activeRequestId = null;
    }
    if (outcome === 'failed') {
      record.reject(new Error(error ?? 'generation failed'));
      return;
    }
    totals.requests += 1;
    totals.decodeTokens += record.decodeTokens;
    totals.units += record.units;
    const streamedDecodeMs =
      record.firstTokenAt !== null && record.lastTokenAt !== null
        ? Math.max(0, record.lastTokenAt - record.firstTokenAt)
        : null;
    const decodeMs = Math.max(
      0,
      streamedDecodeMs !== null
        ? streamedDecodeMs
        : record.decodeMs > 0
          ? record.decodeMs
          : (record.completedAt ?? record.startedAt) - record.startedAt
    );
    totals.decodeMs += decodeMs;
    const stopReason: BrowserEngineStopReason =
      outcome === 'cancelled'
        ? 'cancelled'
        : record.decodeTokens >= record.maxNewTokens
          ? 'length'
          : 'stop';
    record.resolve({
      requestId: record.requestId,
      text: record.text,
      ttftMs:
        record.firstTokenAt === null ? null : record.firstTokenAt - record.startedAt,
      decodeMs,
      decodeTokens: record.decodeTokens,
      firstChunkTokens: record.firstChunkTokens ?? 0,
      prefillTokens: record.prefillTokens || null,
      cacheTokens: record.cacheTokens,
      prefillProcessedTokens:
        record.prefillProcessedTokens > 0
          ? record.prefillProcessedTokens
          : null,
      units: record.units,
      stopReason,
    });
  }

  function emitProgress(event: EngineProgressEvent): void {
    for (const listener of progressListeners) {
      try {
        listener(event);
      } catch {
        // Progress consumers are observability only and cannot fail serving.
      }
    }
  }

  return {
    status,
    prewarm,
    createSession,
    generate,
    cancel,
    metrics,
    disposeSession,
    setSessionPriority,
    priorityLog,
    reportRenderPressure,
    handleEvent,
    handleRuntimeEvent,
    subscribeProgress(listener: (event: EngineProgressEvent) => void) {
      progressListeners.add(listener);
      return () => {
        progressListeners.delete(listener);
      };
    },
  };
}

export function installWllmEnginePage(
  scope: Record<string, unknown>,
  options: WllmEnginePageOptions
): WllmEnginePageApi {
  const api = createWllmEnginePage({ ...options, scope });
  installEnginePage(scope, api as unknown as EnginePageApi);
  return api;
}

function effectivePriority(
  state: SessionState,
  mode: WllmEnginePageOptions['priorityMode']
): SessionPriority {
  return mode === 'disabled' ? 'normal' : state.priority;
}

function lifecycleFor(
  state: SessionState,
  priority: SessionPriority
): LifecycleRecord {
  const highPriority = priority === 'high';
  return {
    clientId: state.clientId,
    epoch: state.epoch,
    // The engine page carries explicit session priority, not browser tab
    // visibility. A normal-priority session is still runnable work and must
    // not inherit the hidden-tab background slice.
    state: 'active',
    foreground: highPriority,
    changedAt: Date.now(),
  };
}

function uniqueRequestId(
  requested: string,
  active: ReadonlyMap<string, ActiveGeneration>
): string {
  if (!active.has(requested)) {
    return requested;
  }
  let suffix = 2;
  while (active.has(`${requested}#${suffix}`)) {
    suffix += 1;
  }
  return `${requested}#${suffix}`;
}

function estimateTokens(text: string): number[] {
  const chunks = text.match(/\w+|[^\s\w]/g) ?? [];
  return chunks.length > 0 ? new Array(chunks.length).fill(0) : [0];
}

function describe(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export { estimateTokens as estimateWllmPromptTokens };
