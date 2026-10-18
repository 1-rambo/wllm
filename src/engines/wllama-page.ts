/**
 * wllama, deployed the way the wllama project deploys it: one engine per tab.
 *
 * This page is the `wllama-no-reuse` baseline. It owns exactly one `Wllama`
 * instance, it never queues work for another tab, and it has no scheduler:
 * a request either runs on this engine or it waits for it, which is the
 * natural single-tab deployment the paper compares against.
 */

import type {
  BrowserEngineGeneration,
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

export interface WllamaPageChunk {
  choices?: { text?: string; finish_reason?: string | null }[];
  prompt_progress?: {
    total?: number;
    cache?: number;
    processed?: number;
  } | null;
  timings?: {
    predicted_n?: number;
    prompt_n?: number;
    prompt_ms?: number;
    predicted_ms?: number;
  } | null;
}

export interface WllamaPageRuntime {
  isModelLoaded(): boolean;
  loadModelFromUrl(
    source: string,
    params?: Record<string, unknown>
  ): Promise<void>;
  loadModelDirect?(
    source: string,
    params?: Record<string, unknown>
  ): Promise<void>;
  setWebGpuPolicy?(policy: {
    commandSubmitBatchSize?: number;
    maxInflightBatches?: number;
  }): void;
  createCompletion(
    options: Record<string, unknown>
  ): Promise<AsyncIterable<WllamaPageChunk>>;
  exit?(): Promise<void>;
}

export interface WllamaEnginePageOptions {
  modelLoadMode?: 'cache' | 'direct';
  runtimeFactory: () => WllamaPageRuntime;
  engineVersion: string;
  modelUrl: string;
  nCtx: number;
  nBatch: number;
  nParallel: number;
  temperature: number;
  seed: number;
  nThreads?: number;
  nGpuLayers?: number;
  useCache?: boolean;
  // A fixed upper bound on the WebGPU submission queue. The upstream
  // default is effectively unlimited, which lets one tab monopolise the
  // device under a multi-tab load. This does not change request-level
  // token or prefill budgets.
  webgpuSubmissionPolicy?: {
    commandSubmitBatchSize: number;
    maxInflightBatches: number;
  };
  // wllama keeps a prompt/KV cache per engine instance by default, so a
  // benchmark that reuses one engine across independent questions prefills
  // only the suffix after the shared few-shot prefix. That makes per-request
  // TTFT incomparable with engines that re-prefill, so the evaluation arms
  // switch it off and record the choice.
  cachePrompt?: boolean;
  frameMetrics?: () => BrowserEngineFrameMetrics | null;
  errorDetail?: () => string | null;
  now?: () => number;
  scope?: Record<string, unknown>;
}

interface SessionState {
  active: AbortController | null;
}

interface LastRun {
  ttftMs: number | null;
  decodeMs: number;
  decodeTokens: number;
  prefillTokens: number | null;
  cacheTokens: number | null;
  prefillProcessedTokens: number | null;
  prefillMs: number | null;
}

export function createWllamaEnginePage(
  options: WllamaEnginePageOptions
): EnginePageApi {
  const now = options.now ?? (() => performance.now());
  const scope =
    options.scope ?? (globalThis as unknown as Record<string, unknown>);
  const sessions = new Map<string, SessionState>();
  const progressListeners = new Set<
    (event: EngineProgressEvent) => void
  >();
  let runtime: WllamaPageRuntime | null = null;
  let loadPromise: Promise<void> | null = null;
  let ready = false;
  let loading = false;
  let error: string | null = null;
  let loadProgress: number | null = null;
  let lastRun: LastRun | null = null;

  function status(): EnginePageStatus {
    return {
      engine: 'wllama',
      engineVersion: options.engineVersion,
      ready,
      loading,
      error,
      loadProgress,
      modelUrl: options.modelUrl,
      sessions: [...sessions.keys()],
      activeRequests: [...sessions.entries()]
        .filter(([, session]) => session.active !== null)
        .map(([sessionId]) => sessionId),
      coordinatorGlobals: presentCoordinatorGlobals(scope),
    };
  }

  async function prewarm(): Promise<void> {
    if (ready || runtime?.isModelLoaded() === true) {
      ready = true;
      return;
    }
    if (loadPromise === null) {
      loadPromise = load();
    }
    await loadPromise;
  }

  async function load(): Promise<void> {
    loading = true;
    error = null;
    try {
      runtime = options.runtimeFactory();
      const loadParams = {
        n_ctx: options.nCtx,
        n_batch: options.nBatch,
        n_parallel: Math.max(1, options.nParallel),
        n_threads: options.nThreads,
        n_gpu_layers: options.nGpuLayers ?? defaultGpuLayers(),
        kv_unified: false,
        // This page has one logical session. Keep native batching explicit so
        // both comparison arms use the same execution configuration.
        cont_batching: false,
        cache_idle_slots: false,
        warmup: true,
        seed: options.seed,
        useCache: options.useCache ?? true,
        progressCallback: (progress: { loaded?: number; total?: number }) => {
          const loaded = progress.loaded ?? 0;
          const total = progress.total ?? 0;
          loadProgress = total > 0 ? round2(loaded / total) : null;
        },
      };
      if (options.modelLoadMode === 'direct' && runtime.loadModelDirect) {
        await runtime.loadModelDirect(options.modelUrl, loadParams);
      } else {
        await runtime.loadModelFromUrl(options.modelUrl, loadParams);
      }
      if (options.webgpuSubmissionPolicy) {
        if (!runtime.setWebGpuPolicy) {
          throw new Error(
            'wllama runtime does not expose setWebGpuPolicy'
          );
        }
        runtime.setWebGpuPolicy(options.webgpuSubmissionPolicy);
      }
      ready = true;
    } catch (loadError) {
      const detailed = withRuntimeDetail(loadError);
      error = describe(detailed);
      // A failed load must be retryable: the next prewarm builds a new engine.
      loadPromise = null;
      runtime = null;
      throw detailed;
    } finally {
      loading = false;
    }
  }

  async function createSession(
    options: BrowserEngineSessionOptions
  ): Promise<void> {
    if (!sessions.has(options.sessionId)) {
      sessions.set(options.sessionId, { active: null });
    }
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
    request: {
      requestId: string;
      promptText: string;
      maxNewTokens: number;
      seed?: number;
      temperature?: number;
    }
  ): Promise<BrowserEngineGeneration> {
    const session = requireSession(sessionId);
    if (session.active !== null) {
      throw new Error(
        `session ${sessionId} already has a running request on this engine`
      );
    }
    const controller = new AbortController();
    session.active = controller;
    // The controller is registered before the model finishes loading so that a
    // cancel issued while the engine is still warming up is not dropped.
    const startedAt = now();
    let firstTokenAt: number | null = null;
    let lastTokenAt: number | null = null;
    let text = '';
    let decodeTokens = 0;
    let streamedTokens = 0;
    let prefillTokens: number | null = null;
    let prefillProcessedTokens: number | null = null;
    let cacheTokens: number | null = null;
    let prefillMs: number | null = null;
    let stopReason: BrowserEngineStopReason = 'length';
    let cancelled = false;

    try {
      await prewarm();
      if (!runtime) {
        throw new Error('wllama engine failed to load');
      }
      if (controller.signal.aborted) {
        cancelled = true;
        stopReason = 'cancelled';
      }
      const stream = controller.signal.aborted
        ? null
        : await runtime.createCompletion({
            prompt: request.promptText,
            stream: true,
            max_tokens: Math.max(1, request.maxNewTokens),
            temperature: request.temperature ?? options.temperature,
            seed: request.seed ?? options.seed,
            cache_prompt: options.cachePrompt ?? true,
            return_progress: true,
            timings_per_token: true,
            abortSignal: controller.signal,
          });

      for await (const chunk of stream ?? []) {
        const choice = chunk.choices?.[0];
        const piece = choice?.text ?? '';
        if (piece.length > 0) {
          text += piece;
          const at = now();
          firstTokenAt ??= at;
          lastTokenAt = at;
          streamedTokens += 1;
          decodeTokens = Math.max(decodeTokens, streamedTokens);
          emitProgress({
            sessionId,
            requestId: request.requestId,
            type: 'token',
            text: piece,
            decodeGenerated: decodeTokens,
            timestampMs: Date.now(),
          });
        }
        const promptProgress = chunk.prompt_progress;
        if (promptProgress) {
          prefillTokens = Math.max(
            prefillTokens ?? 0,
            promptProgress.total ?? 0
          );
          if (typeof promptProgress.processed === 'number') {
            prefillProcessedTokens = Math.max(
              prefillProcessedTokens ?? 0,
              promptProgress.processed
            );
          }
          if (typeof promptProgress.cache === 'number') {
            cacheTokens = Math.max(cacheTokens ?? 0, promptProgress.cache);
          }
        }
        const predicted = chunk.timings?.predicted_n;
        if (typeof predicted === 'number') {
          decodeTokens = Math.max(decodeTokens, predicted);
        }
        const promptMs = chunk.timings?.prompt_ms;
        if (typeof promptMs === 'number') {
          prefillMs = promptMs;
        }
        if (choice?.finish_reason === 'stop') {
          stopReason = 'stop';
        }
      }
    } catch (streamError) {
      if (controller.signal.aborted) {
        cancelled = true;
        stopReason = 'cancelled';
      } else {
        emitProgress({
          sessionId,
          requestId: request.requestId,
          type: 'failed',
          decodeGenerated: decodeTokens,
          timestampMs: Date.now(),
          error: describe(streamError),
        });
        throw withRuntimeDetail(streamError);
      }
    } finally {
      if (controller.signal.aborted && !cancelled) {
        stopReason = 'cancelled';
      }
      session.active = null;
    }

    const finishedAt = now();
    const ttftMs = firstTokenAt === null ? null : firstTokenAt - startedAt;
    const decodeMs =
      firstTokenAt === null || lastTokenAt === null
        ? Math.max(0, finishedAt - startedAt)
        : Math.max(0, lastTokenAt - firstTokenAt);
    lastRun = {
      ttftMs,
      decodeMs,
      decodeTokens,
      prefillTokens,
      cacheTokens,
      prefillProcessedTokens,
      prefillMs,
    };
    emitProgress({
      sessionId,
      requestId: request.requestId,
      type: stopReason === 'cancelled' ? 'failed' : 'complete',
      decodeGenerated: decodeTokens,
      timestampMs: Date.now(),
      ...(stopReason === 'cancelled'
        ? { error: 'generation cancelled' }
        : {}),
    });
    return {
      requestId: request.requestId,
      text,
      ttftMs: ttftMs === null ? null : round2(ttftMs),
      decodeMs: round2(decodeMs),
      decodeTokens,
      prefillTokens,
      cacheTokens,
      prefillProcessedTokens,
      // A per-tab wllama engine has no slicing layer, so the run record has to
      // say "not applicable" instead of reporting a unit count it never used.
      units: null,
      stopReason,
    };
  }

  async function cancel(sessionId: string): Promise<void> {
    const session = sessions.get(sessionId);
    session?.active?.abort();
  }

  async function metrics(): Promise<BrowserEngineMetrics> {
    // Reported with the same rounding the recorded generation uses, so the
    // per-run metrics and the run record cannot disagree.
    const ttftMs = lastRun?.ttftMs ?? null;
    const decodeTokensPerSecond =
      lastRun && lastRun.decodeMs > 0 && lastRun.decodeTokens > 0
        ? round2((lastRun.decodeTokens / lastRun.decodeMs) * 1000)
        : null;
    const prefillTokensPerSecond =
      lastRun &&
      lastRun.prefillMs !== null &&
      lastRun.prefillMs > 0 &&
      lastRun.prefillTokens !== null &&
      lastRun.prefillTokens > 0
        ? round2((lastRun.prefillTokens / lastRun.prefillMs) * 1000)
        : null;
    return {
      engine: 'wllama',
      frame: options.frameMetrics?.() ?? null,
      decodeTokensPerSecond,
      prefillTokensPerSecond,
      ttftMs: ttftMs === null ? null : round2(ttftMs),
      unitsPerRequest: null,
      usedJsHeapBytes: readUsedHeapBytes(scope),
      deviceMemoryBytes: readDeviceMemoryBytes(),
    };
  }

  async function disposeSession(sessionId: string): Promise<void> {
    const session = sessions.get(sessionId);
    session?.active?.abort();
    sessions.delete(sessionId);
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

  function withRuntimeDetail(error: unknown): unknown {
    const detail = options.errorDetail?.() ?? null;
    const base = describe(error);
    return detail === null || detail === ''
      ? error
      : new Error(`${base}\nRuntime log:\n${detail}`);
  }

  return {
    status,
    prewarm,
    createSession,
    generate,
    cancel,
    metrics,
    disposeSession,
    subscribeProgress(listener: (event: EngineProgressEvent) => void) {
      progressListeners.add(listener);
      return () => {
        progressListeners.delete(listener);
      };
    },
  };
}

export function installWllamaEnginePage(
  scope: Record<string, unknown>,
  options: WllamaEnginePageOptions
): EnginePageApi {
  const api = createWllamaEnginePage({ ...options, scope });
  installEnginePage(scope, api);
  return api;
}

function defaultGpuLayers(): number {
  return typeof navigator !== 'undefined' && 'gpu' in navigator ? 99_999 : 0;
}

function readUsedHeapBytes(scope: Record<string, unknown>): number | null {
  const performanceScope = scope.performance as
    | { memory?: { usedJSHeapSize?: number } }
    | undefined;
  const used = performanceScope?.memory?.usedJSHeapSize;
  return typeof used === 'number' ? used : null;
}

function readDeviceMemoryBytes(): number | null {
  const navigatorScope = globalThis.navigator as
    | { deviceMemory?: number }
    | undefined;
  const gigabytes = navigatorScope?.deviceMemory;
  return typeof gigabytes === 'number' ? gigabytes * 1024 ** 3 : null;
}

function describe(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
