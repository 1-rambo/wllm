import compatWasmUrl from '@wllama/wllama-compat/wasm/wllama.wasm?url';
import compatWorkerCode from '@wllama/wllama-compat/wasm/wllama.js?raw';
import { LoggerWithoutDebug, Wllama } from '@wllama/wllama/esm/index.js';
import type {
  NativeRuntimeEvent,
  NativeSlotCheckpoint,
  NativeSlotStats,
  NativeSubmitRequest,
  NativeTokenChunk,
  NativeTokenCounters,
  NativeUnitBudget,
  NativeUnitProgress,
  RequestStage,
  TokenAccounting,
  GpuBurstPolicy,
} from '../contracts';
import { WllmError } from '../contracts';
import type { NativeEngine } from './native-engine';
import { checksumBytes } from '../utils/bytes';
import {
  contextWindowShape,
  estimateTextTokens,
  trimPrefixToTokenBudget,
  type ContextWindowShape,
} from './context-window';
import { modelIdentity } from '../utils/model-identity';

interface WllamaCompletionChoice {
  text: string;
  finish_reason: 'stop' | 'length' | 'content_filter' | null;
}

export interface WllamaCompletionChunk {
  choices: WllamaCompletionChoice[];
  prompt_progress?: {
    total: number;
    cache: number;
    processed: number;
    time_ms: number;
  } | null;
  timings?: {
    predicted_n: number;
    // llama.cpp reports these counters on every chunk when the request sets
    // timings_per_token; prompt_n counts prompt tokens evaluated in this
    // request and cache_n counts prompt tokens reused from the slot cache.
    prompt_n?: number;
    cache_n?: number;
    prompt_ms?: number;
    predicted_ms?: number;
    // true when the native server closed this unit at a token boundary
    // because the per-unit time budget ran out (slot KV is retained)
    bounded_stop?: boolean;
    // exact native budget that closed the unit; prevents the host from
    // guessing between prefill, decode, and time stops
    bounded_stop_reason?:
      | 'prefill-budget'
      | 'decode-budget'
      | 'unit-time-budget'
      | 'predict-time-budget';
  } | null;
}

export interface WllamaRuntime {
  isModelLoaded(): boolean;
  loadModelFromUrl(
    source: string,
    params?: Record<string, unknown>
  ): Promise<void>;
  loadModelDirect?(
    source: string,
    params?: Record<string, unknown>
  ): Promise<void>;
  createCompletion(
    options: Record<string, unknown>
  ): Promise<AsyncIterable<WllamaCompletionChunk>>;
  // bounded serving: slot-state API, present in the pinned fork build
  exportSlotState?(
    slotId: number,
    options?: { drop?: boolean }
  ): Promise<{ kv: Uint8Array; tokens: number[]; tokenCount: number }>;
  importSlotState?(
    slotId: number,
    state: { kv: Uint8Array; tokens: number[] }
  ): Promise<number>;
  dropSlotState?(slotId: number): Promise<number>;
  // tier two: sequence states parked inside the wasm heap behind a handle
  parkSlotState?(
    slotId: number
  ): Promise<{ handle: number; tokenCount: number; byteLength: number }>;
  resumeSlotState?(handle: number, slotId: number): Promise<number>;
  readParkedSlotState?(handle: number): Promise<{
    handle: number;
    kv: Uint8Array;
    tokens: number[];
    tokenCount: number;
    byteLength: number;
  }>;
  releaseParkedSlotState?(
    handle: number
  ): Promise<{ tokenCount: number; byteLength: number }>;
  // runtime WebGPU submission policy; values are read at graph boundaries
  setWebGpuPolicy?(
    policy: {
      commandSubmitBatchSize?: number;
      maxInflightBatches?: number;
    }
  ): void;
  exit(): Promise<void>;
}

export interface WllamaNativeEngineConfig {
  wasmUrl: string;
  modelUrl: string;
  // Direct loading avoids the browser's storage quota by handing the fetched
  // GGUF Blob to wllama. Cache loading remains the default for the demo.
  modelLoadMode?: 'cache' | 'direct';
  nCtx?: number;
  nBatch?: number;
  nThreads?: number;
  nGpuLayers?: number;
  // `true` keeps llama.cpp's own auto decision and `false` forces the
  // unfused attention path. The fused kernels are what the MLC baseline
  // runs by default, so a decoder comparison has to state which of the
  // two paths a measured arm used.
  flashAttn?: boolean;
  // WebGPU submission controls exposed by the pinned ggml backend.
  webgpuCommandSubmitBatchSize?: number;
  webgpuMaxInflightBatches?: number;
  // A queued graph may legitimately outlive a render-pressure window. Keep
  // the same safety ceiling as the per-tab wllama entry page.
  queueWaitTimeoutMs?: number;
  nParallel?: number;
  temperature?: number;
  seed?: number;
  onRuntimeEvent?: (event: NativeRuntimeEvent) => void;
  // Streamed text chunks are forwarded while the native unit is still
  // running. The unit itself remains the scheduling boundary.
  onTokenChunk?: (chunk: NativeTokenChunk) => void;
  runtime?: WllamaRuntime;
}

interface WllamaRequestState {
  requestId: string;
  slotId: number;
  promptText: string;
  promptTokens: number[];
  generatedText: string;
  outputChunks: string[];
  prefillProcessed: number;
  prefillTotal: number;
  decodeGenerated: number;
  maxNewTokens: number;
  seed: number;
  temperature: number;
  prefillCached: number;
  stage: RequestStage;
  tokenAccounting: TokenAccounting;
  counters?: NativeTokenCounters;
  // how many bounded units were closed by the native time budget rather
  // than by the host abort safety net
  boundedStops: number;
  done: boolean;
  cancelled: boolean;
  abortController: AbortController | null;
  // Set when a lifecycle cut arrives before this request owns the stream,
  // which is what happens to every queued member of a cooperative group.
  interruptPending: boolean;
}

interface SerializedContinuation {
  formatVersion: 1;
  promptText: string;
  generatedText: string;
  promptTokens: number[];
  prefillProcessed: number;
  prefillTotal: number;
  decodeGenerated: number;
  maxNewTokens: number;
  seed?: number;
  temperature?: number;
  stage: RequestStage;
  modelHash: string;
  // diagnostics: which serving slot produced the checkpoint, so a failed
  // restore can be compared against the slot it targets
  originSlotId?: number;
}

// checksum covering both the continuation metadata and the exported KV bytes
function checkpointDigest(bytes: Uint8Array, kv?: Uint8Array): string {
  if (!kv || kv.byteLength === 0) {
    return checksumBytes(bytes);
  }
  const merged = new Uint8Array(bytes.byteLength + kv.byteLength);
  merged.set(bytes, 0);
  merged.set(kv, bytes.byteLength);
  return checksumBytes(merged);
}

const DEFAULT_CTX = 2_048;
const DEFAULT_BATCH = 128;
const DEFAULT_PARALLEL = 4;

export class WllamaNativeEngine implements NativeEngine {
  private readonly runtime: WllamaRuntime;
  private readonly states = new Map<string, WllamaRequestState>();
  private readonly slots = new Map<number, string | null>();
  private readonly modelHash: string;
  private pendingGpuPolicy: GpuBurstPolicy | null = null;
  private appliedGpuPolicy: GpuBurstPolicy | null = null;
  private nextSlot = 0;
  // context bookkeeping: how often a retained conversation had to drop its
  // oldest turns to stay inside the slot, and by how much
  private contextTrims = 0;
  private lastContextTrim: {
    trimmedTokens: number;
    perSlotTokens: number;
    atMs: number;
  } | null = null;
  private loadPromise: Promise<void> | null = null;
  private loaded = false;

  constructor(private readonly config: WllamaNativeEngineConfig) {
    this.runtime =
      config.runtime ?? createWllamaRuntime(config.wasmUrl, config);
    this.modelHash = modelIdentity(config.modelUrl);
    const slotCount = Math.max(1, config.nParallel ?? DEFAULT_PARALLEL);
    for (let slot = 0; slot < slotCount; slot += 1) {
      this.slots.set(slot, null);
    }
  }

  async prewarm(): Promise<void> {
    if (this.loaded) {
      this.emitRuntime({ phase: 'ready', message: 'Model ready' });
      return;
    }
    await this.ensureLoaded();
  }

  // The native error used to name a context size nobody configured; these
  // diagnostics state the real per-slot budget and how often a retained
  // conversation had to drop its oldest turns to fit it.
  contextDiagnostics(): ContextWindowShape & {
    trims: number;
    lastTrim: {
      trimmedTokens: number;
      perSlotTokens: number;
      atMs: number;
    } | null;
  } {
    return {
      ...this.contextWindow(0),
      trims: this.contextTrims,
      lastTrim: this.lastContextTrim === null ? null : { ...this.lastContextTrim },
    };
  }

  private contextWindow(maxNewTokens: number): ContextWindowShape {
    return contextWindowShape({
      nCtx: this.config.nCtx ?? DEFAULT_CTX,
      nParallel: this.config.nParallel ?? DEFAULT_PARALLEL,
      maxNewTokens,
    });
  }

  private recordContextTrim(
    trimmedTokens: number,
    context: ContextWindowShape
  ): void {
    this.contextTrims += 1;
    this.lastContextTrim = {
      trimmedTokens,
      perSlotTokens: context.perSlotTokens,
      atMs: Date.now(),
    };
  }

  async setGpuBurstPolicy(policy: GpuBurstPolicy): Promise<void> {
    this.pendingGpuPolicy = {
      level: policy.level,
      commandSubmitBatchSize: Math.max(
        1,
        Math.floor(policy.commandSubmitBatchSize)
      ),
      maxInflightBatches: Math.max(
        0,
        Math.floor(policy.maxInflightBatches)
      ),
      submissionPauseMs: Number.isFinite(policy.submissionPauseMs)
        ? Math.max(0, Math.floor(policy.submissionPauseMs))
        : 0,
    };
    if (this.loaded) {
      this.applyGpuBurstPolicy();
    }
  }

  async submit(request: NativeSubmitRequest): Promise<NativeUnitProgress> {
    await this.ensureLoaded();
    const previous = this.states.get(request.requestId);
    if (previous !== undefined) {
      // A persistent session may reuse its request id for the next turn. That
      // is a new turn, not a duplicate submission: the finished state is only
      // kept so the resident slot can be found, and the resume path below
      // takes over that slot's KV.
      const reusesOwnSlot = request.resumeSlotId === previous.slotId;
      if (!previous.done && !previous.cancelled && !reusesOwnSlot) {
        throw new WllmError(
          `native request already exists: ${request.requestId}`,
          'duplicate_native_request'
        );
      }
      if (!reusesOwnSlot) {
        // no slot was handed over, so the old one must not stay pinned to a
        // state that is about to be replaced
        if (this.slots.get(previous.slotId) === request.requestId) {
          this.slots.set(previous.slotId, null);
        }
      }
      this.states.delete(request.requestId);
    }

    const resumedSlotId = request.resumeSlotId;
    const resumedRequestId =
      resumedSlotId === undefined ? null : this.slots.get(resumedSlotId) ?? null;
    const resumed =
      resumedRequestId === null
        ? undefined
        : this.states.get(resumedRequestId);

    const promptText =
      request.promptText ?? tokensToText(request.promptTokens);
    const startsNewTurn = Boolean(request.newTurn && resumed);
    const context = this.contextWindow(request.maxNewTokens);
    const requestTokens = estimateTextTokens(promptText);
    // `promptTokens` is an estimate supplied by the page layer, so a text
    // prompt cannot be judged safely by that estimate alone. Artifex's
    // prompt, for example, is estimated at 2079 tokens but native llama.cpp
    // tokenizes the same text to 1091. Defer text prompts to the native
    // tokenizer; keep the explicit guard for token-only submissions.
    const hasExplicitPromptText =
      typeof request.promptText === 'string' &&
      request.promptText.trim().length > 0;
    const requestMustFit = !hasExplicitPromptText;
    if (requestMustFit && requestTokens > context.promptBudgetTokens) {
      throw new WllmError(
        `request needs ${requestTokens} tokens but this engine's per-slot ` +
          `prompt budget is ${context.promptBudgetTokens} ` +
          `(nCtx=${context.nCtx}, nParallel=${context.nParallel}, ` +
          `maxNewTokens=${request.maxNewTokens})`,
        'context_overflow'
      );
    }
    // a new user turn keeps the restored text prefix byte-for-byte so the
    // backend can match it against the imported KV cache -- as long as the
    // prefix still fits beside the new turn. A session that outgrows its
    // slot keeps the newest turns instead of dying inside llama.cpp.
    // A self-contained turn replaces the slot's context: the request already
    // carries the whole prompt, so nothing is replayed into it. The slot stays
    // bound to the session, which is what lets the backend match the shared
    // prefix against the resident KV instead of evaluating it again.
    const replacesContext = startsNewTurn && request.replaceContext === true;
    const rawPrefix =
      startsNewTurn && resumed && !replacesContext
        ? resumed.promptText + resumed.generatedText
        : '';
    const prefixBudget = Math.max(0, context.promptBudgetTokens - requestTokens);
    const prefixTrim =
      rawPrefix.length === 0 || estimateTextTokens(rawPrefix) <= prefixBudget
        ? { text: rawPrefix, droppedTokens: 0, droppedChars: 0 }
        : trimPrefixToTokenBudget(rawPrefix, prefixBudget);
    const resumedPrefix = prefixTrim.text;
    if (prefixTrim.droppedTokens > 0) {
      this.recordContextTrim(prefixTrim.droppedTokens, context);
    }
    // Chat-template turn deltas start with a special token and must attach
    // directly to the resident prefix. Adding whitespace changes tokenization
    // and makes a restored session diverge from full-history replay.
    const turnSeparator =
      resumedPrefix.length > 0 &&
      !promptText.startsWith(' ') &&
      !promptText.startsWith('<|')
        ? ' '
        : '';
    // Only once the request is known to fit: a refused request must not
    // reserve a slot, and a resumed request must keep its parked state
    // addressable until the replacement is admitted.
    const slotId = resumedSlotId ?? this.allocateSlot(request.requestId);
    if (slotId === null) {
      throw new WllmError('no free native slot', 'capacity');
    }
    if (resumedRequestId) {
      this.states.delete(resumedRequestId);
    }
    const state: WllamaRequestState = {
      requestId: request.requestId,
      slotId,
      promptText: startsNewTurn
        ? resumedPrefix + turnSeparator + promptText
        : promptText,
      promptTokens: [...request.promptTokens],
      generatedText: startsNewTurn ? '' : resumed?.generatedText ?? '',
      outputChunks: [],
      prefillProcessed: replacesContext ? 0 : resumed?.prefillProcessed ?? 0,
      prefillTotal: Math.max(
        1,
        startsNewTurn
          ? (replacesContext ? 0 : resumed?.prefillProcessed ?? 0) +
            request.promptTokens.length
          : request.promptTokens.length,
        replacesContext ? 0 : resumed?.prefillTotal ?? 0
      ),
      prefillCached: 0,
      decodeGenerated: startsNewTurn ? 0 : resumed?.decodeGenerated ?? 0,
      tokenAccounting: resumed?.tokenAccounting ?? 'stream-chunk',
      boundedStops: 0,
      ...(resumed?.counters ? { counters: { ...resumed.counters } } : {}),
      maxNewTokens: request.maxNewTokens,
      seed: request.seed ?? resumed?.seed ?? this.config.seed ?? 0,
      // The harness pins one temperature for every arm; carrying it on the
      // request keeps the sampler identical across deployments.
      temperature:
        request.temperature ??
        resumed?.temperature ??
        this.config.temperature ??
        0.8,
      stage:
        startsNewTurn || request.promptTokens.length > 0 ? 'prefill' : 'decode',
      done: false,
      cancelled: false,
      abortController: null,
      interruptPending: false,
    };
    this.states.set(request.requestId, state);
    this.slots.set(slotId, request.requestId);
    return this.progress(state, [], 0);
  }

  async advance(
    requestId: string,
    budget: NativeUnitBudget
  ): Promise<NativeUnitProgress> {
    await this.ensureLoaded();
    const state = this.requireState(requestId);
    if (state.done || state.cancelled) {
      return this.progress(state, [], 0);
    }
    // A click can arrive after the coordinator issued this request's slice
    // but before the worker reached it: a cooperative group starts several
    // advances at once and only one of them owns the stream. Consume the
    // remembered cut here so a queued slice returns at the click boundary
    // instead of decoding a whole chunk after the human already moved.
    if (state.interruptPending) {
      state.interruptPending = false;
      return this.progressWithStop(state, [], 0, 'host-abort');
    }

    const remaining = Math.max(0, state.maxNewTokens - state.decodeGenerated);
    if (remaining === 0) {
      state.done = true;
      state.stage = 'decode';
      return this.progress(state, [], 0);
    }

    const unitBudget = Math.max(
      1,
      Math.min(remaining, budget.maxDecodeTokens || 1)
    );
    const wallTimeMs = Math.max(1, budget.wallTimeMs);
    // The native token budget (n_decode_max) is the deterministic boundary.
    // The native wall-clock budget (t_unit_max_ms) is checked by llama.cpp
    // at token boundaries against the full task wall time, so WebGPU timer
    // accounting cannot hide a slow decode step. The host wall-clock abort
    // remains a larger last-resort guard for a runtime that stalls.
    const nativeUnitMs =
      budget.nativeTimeBudgetMs === undefined
        ? undefined
        : Math.max(1, budget.nativeTimeBudgetMs);
    const controller = new AbortController();
    state.abortController = controller;
    let boundary = false;
    let completed = false;
    let prefillSeen = false;
    let unitGenerated = 0;
    // llama.cpp resets its slot counters per task, so a unit starts from zero.
    // The streamed text is only a fallback: one chunk may carry several tokens.
    const baseDecoded = state.decodeGenerated;
    let unitTokenCount = 0;
    let nativeCountersSeen = false;
    let nativeBoundedStop = false;
    let nativeBoundedReason:
      | 'prefill-budget'
      | 'decode-budget'
      | 'unit-time-budget'
      | 'predict-time-budget'
      | undefined;
    let unitStop: NativeUnitProgress['unitStop'] = 'complete';
    let unitPromptN = 0;
    let unitCacheN = 0;
    let unitPromptMs: number | undefined;
    let unitPredictedMs: number | undefined;
    const chunks: string[] = [];
    const startedAt = performance.now();
    let timeout: ReturnType<typeof setTimeout> | undefined;
    // an abort issued from inside the stream loop must wait until the loop has
    // exited, otherwise the async iterator is left half-consumed
    let abortPending = false;
    const abortForWallTime = () => {
      boundary = true;
      unitStop = 'host-abort';
      controller.abort();
    };
    timeout = setTimeout(abortForWallTime, wallTimeMs);

    try {
      const stream = await this.runtime.createCompletion({
        prompt: state.promptText + state.generatedText,
        stream: true,
        // pin every unit of this session to its own serving slot
        id_slot: state.slotId,
        // native prompt-chunk boundary: this call must not evaluate more
        // than the controller budget, even if the full prompt is larger
        n_prefill_max: budget.maxPrefillTokens,
        // native token boundary: this unit decodes at most unitBudget tokens,
        // then llama.cpp yields the slot with its KV cache intact. This is
        // the deterministic boundary on every backend.
        n_decode_max: unitBudget,
        ...(nativeUnitMs === undefined
          ? {}
          : {
              // native task wall-clock boundary: this covers prompt and
              // decode and stops at a token boundary with the slot KV retained
              t_unit_max_ms: nativeUnitMs,
              // secondary brake for backends whose prediction timer
              // advances between tokens; on WebGPU it undercounts
              t_max_predict_ms: nativeUnitMs,
            }),
        // the request-level cap stays the remaining budget, so llama.cpp's
        // own length stop keeps meaning "the whole request is done"
        max_tokens: remaining,
        temperature: state.temperature,
        seed: state.seed,
        cache_prompt: true,
        return_progress: true,
        // ask llama.cpp for its own token counters on every streamed chunk,
        // so decode accounting is token-level rather than chunk-level
        timings_per_token: true,
        abortSignal: controller.signal,
      });

      for await (const chunk of stream) {
        const promptProgress = chunk.prompt_progress;
        if (promptProgress) {
          prefillSeen = true;
          state.prefillProcessed = Math.max(
            state.prefillProcessed,
            promptProgress.processed
          );
          state.prefillTotal = Math.max(
            state.prefillTotal,
            promptProgress.total
          );
          // tokens llama.cpp reused from the slot KV cache (no re-prefill)
          state.prefillCached = Math.max(
            state.prefillCached,
            promptProgress.cache
          );
        }

        const choice = chunk.choices[0];
        const text = choice?.text ?? '';

        const timings = chunk.timings;
        if (timings?.bounded_stop === true) {
          // the native server already ended this task at a token boundary and
          // kept the slot KV; which budget fired is classified after the
          // stream drains, from the counters this unit reported
          boundary = true;
          nativeBoundedStop = true;
          nativeBoundedReason = timings.bounded_stop_reason;
        }
        if (timings && Number.isFinite(timings.predicted_n)) {
          nativeCountersSeen = true;
          unitTokenCount = Math.max(unitTokenCount, timings.predicted_n);
          if (Number.isFinite(timings.prompt_n)) {
            unitPromptN = Math.max(unitPromptN, timings.prompt_n as number);
          }
          if (Number.isFinite(timings.cache_n)) {
            unitCacheN = Math.max(unitCacheN, timings.cache_n as number);
            state.prefillCached = Math.max(state.prefillCached, unitCacheN);
            state.prefillProcessed = Math.max(
              state.prefillProcessed,
              unitCacheN + unitPromptN
            );
          }
          if (Number.isFinite(timings.prompt_ms)) {
            unitPromptMs = Math.max(unitPromptMs ?? 0, timings.prompt_ms as number);
          }
          if (Number.isFinite(timings.predicted_ms)) {
            unitPredictedMs = Math.max(
              unitPredictedMs ?? 0,
              timings.predicted_ms as number
            );
          }
        }

        if (text.length > 0) {
          state.generatedText += text;
          chunks.push(text);
          state.decodeGenerated += 1;
          unitGenerated += 1;
        }

        // native counters win; the streamed chunk count is only a lower bound
        state.decodeGenerated = Math.max(
          baseDecoded + unitTokenCount,
          baseDecoded + unitGenerated
        );

        if (text.length > 0) {
          try {
            this.config.onTokenChunk?.({
              requestId,
              text,
              decodeGenerated: state.decodeGenerated,
              elapsedMs: Math.max(0, performance.now() - startedAt),
            });
          } catch {
            // Chunk delivery is best effort: a disconnected port must not
            // abort the native unit that is still producing the final text.
          }
        }

        if (choice?.finish_reason === 'stop') {
          completed = true;
          break;
        }

        if (choice?.finish_reason === 'length') {
          if (state.decodeGenerated >= state.maxNewTokens) {
            completed = true;
            unitStop = 'complete';
          } else {
            boundary = true;
            unitStop = 'native-length';
          }
          break;
        }

        // Let the native time budget close this unit at a token boundary.
        // The elapsed check below only fires if the worker keeps emitting
        // chunks past the budget, and stays as the last-resort abort.
        const elapsedMs = performance.now() - startedAt;
        if (elapsedMs >= wallTimeMs) {
          boundary = true;
          unitStop = 'host-abort';
          abortPending = true;
          break;
        }
      }
    } catch (error) {
      if (!boundary && !isAbortError(error)) {
        throw new WllmError(
          `wllama advance failed for ${requestId}: ${errorMessage(error)}`,
          'native_error'
        );
      }
    } finally {
      if (timeout !== undefined) {
        clearTimeout(timeout);
      }
      if (abortPending) {
        // safe now: the stream loop has already exited
        controller.abort();
      }
      state.abortController = null;
    }

    // A native bounded stop keeps its slot KV, so the unit can be resumed
    // without prefill. New native builds report the exact budget in
    // bounded_stop_reason; the counter-based branch below only supports older
    // prebuilt artifacts that do not carry the explicit reason yet.
    if (nativeBoundedStop) {
      state.boundedStops += 1;
      if (!completed) {
        const unitDecoded = state.decodeGenerated - baseDecoded;
        if (nativeBoundedReason !== undefined) {
          const mappedReason = mapNativeBoundedReason(nativeBoundedReason);
          if (mappedReason !== undefined) {
            unitStop = mappedReason;
          } else {
            unitStop =
              unitDecoded <= 0
                ? 'native-prefill-budget'
                : unitDecoded >= unitBudget
                  ? 'native-decode-budget'
                  : 'native-time-budget';
          }
        } else {
          // Older native builds do not report the reason explicitly. Keep a
          // conservative fallback for compatibility with those artifacts.
          unitStop =
            unitDecoded <= 0
              ? 'native-prefill-budget'
              : unitDecoded >= unitBudget
                ? 'native-decode-budget'
                : 'native-time-budget';
        }
      }
    }

    if (nativeCountersSeen) {
      state.tokenAccounting = 'native-token';
      state.counters = {
        promptN: unitPromptN,
        cacheN: unitCacheN,
        predictedN: state.decodeGenerated - baseDecoded,
        ...(unitPromptMs === undefined ? {} : { promptMs: unitPromptMs }),
        ...(unitPredictedMs === undefined
          ? {}
          : { predictedMs: unitPredictedMs }),
      };
    } else if (unitGenerated > 0) {
      // backend without per-token counters: report the honest fallback
      state.tokenAccounting = 'stream-chunk';
      delete state.counters;
    }

    if (completed || state.decodeGenerated >= state.maxNewTokens) {
      state.done = true;
      state.stage = 'decode';
    } else if (prefillSeen && state.prefillProcessed < state.prefillTotal) {
      state.stage = 'prefill';
    } else {
      state.stage = 'decode';
    }

    return this.progressWithStop(
      state,
      chunks,
      Math.max(0, performance.now() - startedAt),
      unitStop
    );
  }

  async status(requestId: string): Promise<NativeUnitProgress> {
    return this.progress(this.requireState(requestId), [], 0);
  }

  async cancel(requestId: string): Promise<void> {
    const state = this.requireState(requestId);
    state.abortController?.abort();
    state.cancelled = true;
    state.done = true;
  }

  async interrupt(requestId: string): Promise<void> {
    const state = this.states.get(requestId);
    // Abort only the in-flight stream. The request, slot, and KV state stay
    // alive so the next admission can resume from the token boundary.
    state?.abortController?.abort();
    if (state) {
      state.interruptPending = true;
    }
  }

  async slotExport(slotId: number): Promise<NativeSlotCheckpoint> {
    const state = this.requireSlotState(slotId);
    const bytes = this.continuationBytes(state);

    // Bounded serving: pull the real sequence KV out of the serving context and
    // release the slot. `tokens` then reflects what llama.cpp actually holds,
    // which is what lets a later resume skip prefill instead of replaying it.
    if (typeof this.runtime.exportSlotState === 'function') {
      const exported = await this.runtime.exportSlotState(slotId, { drop: true });
      this.states.delete(state.requestId);
      this.slots.set(slotId, null);
      return {
        bytes,
        kv: exported.kv,
        tokens: [...exported.tokens],
        stage: state.stage,
        decodeGenerated: state.decodeGenerated,
        formatVersion: 2,
        checksum: checkpointDigest(bytes, exported.kv),
      };
    }

    return {
      bytes,
      tokens: [...state.promptTokens],
      stage: state.stage,
      decodeGenerated: state.decodeGenerated,
      formatVersion: 1,
      checksum: checksumBytes(bytes),
    };
  }

  async slotSnapshot(slotId: number): Promise<NativeSlotCheckpoint> {
    const state = this.requireSlotState(slotId);
    const bytes = this.continuationBytes(state);
    if (typeof this.runtime.exportSlotState === 'function') {
      const exported = await this.runtime.exportSlotState(slotId, {
        drop: false,
      });
      return {
        bytes,
        kv: exported.kv,
        tokens: [...exported.tokens],
        stage: state.stage,
        decodeGenerated: state.decodeGenerated,
        formatVersion: 2,
        checksum: checkpointDigest(bytes, exported.kv),
      };
    }
    return {
      bytes,
      tokens: [...state.promptTokens],
      stage: state.stage,
      decodeGenerated: state.decodeGenerated,
      formatVersion: 1,
      checksum: checksumBytes(bytes),
    };
  }

  // Tier two parking: the sequence state stays inside the native heap behind a
  // handle and the live serving slot is released. Host-side we keep only the
  // continuation metadata and the handle, never the KV bytes.
  async slotPark(slotId: number): Promise<NativeSlotCheckpoint> {
    const state = this.requireSlotState(slotId);
    if (typeof this.runtime.parkSlotState !== 'function') {
      // backend without a park bank: fall back to exporting the bytes
      return this.slotExport(slotId);
    }
    const bytes = this.continuationBytes(state);
    const parked = await this.runtime.parkSlotState(slotId);
    this.states.delete(state.requestId);
    this.slots.set(slotId, null);
    return {
      bytes,
      tokens: [...state.promptTokens],
      stage: state.stage,
      decodeGenerated: state.decodeGenerated,
      formatVersion: 3,
      checksum: `${checksumBytes(bytes)}:park:${parked.handle}`,
      parkedHandle: parked.handle,
      parkedByteLength: parked.byteLength,
    };
  }

  // Tier three to tier two: load serialized bytes back into the native heap so
  // the host can drop its copy of them.
  async slotParkCheckpoint(
    checkpoint: NativeSlotCheckpoint
  ): Promise<NativeSlotCheckpoint> {
    if (checkpoint.parkedHandle !== undefined) {
      return checkpoint;
    }
    if (typeof this.runtime.parkSlotState !== 'function') {
      return checkpoint;
    }
    const slot = await this.slotImport(checkpoint);
    try {
      return await this.slotPark(slot.slotId);
    } catch (error) {
      await this.slotDrop(slot.slotId).catch(() => undefined);
      throw error;
    }
  }

  // Tier three serialization: copy the parked bytes out of the native heap. The
  // native copy survives, so a failed write can still fall back to it.
  async slotReadParked(
    checkpoint: NativeSlotCheckpoint
  ): Promise<NativeSlotCheckpoint> {
    const handle = checkpoint.parkedHandle;
    if (handle === undefined) {
      return checkpoint;
    }
    if (typeof this.runtime.readParkedSlotState !== 'function') {
      return checkpoint;
    }
    const parked = await this.runtime.readParkedSlotState(handle);
    return {
      ...checkpoint,
      kv: parked.kv,
      tokens: [...parked.tokens],
      formatVersion: 2,
      checksum: checkpointDigest(checkpoint.bytes, parked.kv),
    };
  }

  // Release the native copy once its serialized form is durable, or when the
  // node is evicted. A missing handle is already released, so it is not an error.
  async slotReleaseParked(checkpoint: NativeSlotCheckpoint): Promise<void> {
    const handle = checkpoint.parkedHandle;
    if (handle === undefined) {
      return;
    }
    if (typeof this.runtime.releaseParkedSlotState !== 'function') {
      return;
    }
    await this.runtime.releaseParkedSlotState(handle).catch(() => undefined);
  }

  async slotImport(
    checkpoint: NativeSlotCheckpoint,
    targetSlotId?: number
  ): Promise<NativeSlotStats> {
    await this.ensureLoaded();
    const continuation = decodeContinuation(checkpoint.bytes);
    if (continuation.modelHash !== this.modelHash) {
      throw new WllmError('checkpoint model mismatch', 'checkpoint_mismatch');
    }

    // A sequence state blob is sequence-affine in the native fork: a state
    // exported from one serving sequence cannot be installed into another,
    // so a restore prefers the slot the state came from when it is free.
    const originSlotId = continuation.originSlotId;
    const slotId =
      targetSlotId ??
      (originSlotId !== undefined && this.slots.get(originSlotId) === null
        ? this.claimSlot(originSlotId, 'import')
        : this.allocateSlot('import'));
    if (slotId === null) {
      throw new WllmError('no free native slot', 'capacity');
    }
    // Bounded serving: restore the real KV cache first, then register the
    // continuation so the next unit reuses the restored prefix.
    let residentTokens = continuation.promptTokens.length;
    const parkedHandle = checkpoint.parkedHandle;
    // The native park handle is an in-heap pointer. It is the right restore
    // path only for a live tier-two checkpoint; once the state has been
    // serialized (tier three) the bytes are authoritative and any handle is
    // either released or belongs to another instance.
    const serializedState = checkpoint.kv?.byteLength ?? 0;
    if (
      parkedHandle !== undefined &&
      serializedState === 0 &&
      typeof this.runtime.resumeSlotState === 'function'
    ) {
      // tier two resume: the bytes never reached the host
      residentTokens = await this.runtime.resumeSlotState(
        parkedHandle,
        slotId
      );
      if (residentTokens <= 0) {
        throw new WllmError(
          'native backend refused the parked state',
          'checkpoint_mismatch'
        );
      }
      // the live slot owns the state now, so free the parked copy
      await this.slotReleaseParked(checkpoint);
    }
    if (
      checkpoint.kv &&
      checkpoint.kv.byteLength > 0 &&
      typeof this.runtime.importSlotState === 'function'
    ) {
      try {
        residentTokens = await this.runtime.importSlotState(slotId, {
          kv: checkpoint.kv,
          tokens: checkpoint.tokens,
        });
      } catch (error) {
        // native reports a coarse status code, so attach a structural
        // description of the blob to make a failed restore diagnosable
        const originSlot = continuation.originSlotId ?? 'unknown';
        const reason = error instanceof Error ? error.message : String(error);
        throw new WllmError(
          `native import rejected checkpoints/${checkpoint.checksum} (${describeSlotBlob(
            checkpoint.kv,
            checkpoint.tokens
          )}) import_slot=${slotId} origin_slot=${originSlot}: ${reason}`,
          'checkpoint_mismatch'
        );
      }
      if (residentTokens <= 0) {
        throw new WllmError(
          'native backend refused the slot state',
          'checkpoint_mismatch'
        );
      }
    }

    const previousRequestId = this.slots.get(slotId);
    if (previousRequestId) {
      this.states.delete(previousRequestId);
    }
    const requestId = `import:${slotId}:${checkpoint.checksum}`;
    this.states.set(requestId, {
      requestId,
      slotId,
      promptText: continuation.promptText,
      promptTokens: [...continuation.promptTokens],
      generatedText: continuation.generatedText,
      outputChunks: [],
      prefillProcessed: residentTokens,
      prefillTotal: Math.max(residentTokens, continuation.prefillTotal),
      prefillCached: 0,
      decodeGenerated: continuation.decodeGenerated,
      boundedStops: 0,
      // an imported state already carries its prompt in the native cache
      tokenAccounting: 'native-token',
      counters: {
        promptN: 0,
        cacheN: residentTokens,
        predictedN: continuation.decodeGenerated,
      },
      maxNewTokens: continuation.maxNewTokens,
      seed: continuation.seed ?? this.config.seed ?? 0,
      temperature:
        continuation.temperature ?? this.config.temperature ?? 0.8,
      stage: continuation.stage,
      done: false,
      cancelled: false,
      abortController: null,
      interruptPending: false,
    });
    this.slots.set(slotId, requestId);
    return {
      slotId,
      inUse: true,
      tokens: residentTokens,
      state: continuation.stage,
    };
  }

  async slotDrop(slotId: number): Promise<void> {
    const requestId = this.slots.get(slotId);
    if (requestId) {
      const state = this.states.get(requestId);
      state?.abortController?.abort();
      this.states.delete(requestId);
    }
    this.slots.set(slotId, null);
    // release the KV cache held by the serving context so L1 accounting is real
    if (typeof this.runtime.dropSlotState === 'function') {
      await this.runtime.dropSlotState(slotId).catch(() => undefined);
    }
  }

  async slotStats(): Promise<NativeSlotStats[]> {
    return [...this.slots.entries()].map(([slotId, requestId]) => {
      if (!requestId) {
        return { slotId, inUse: false, tokens: 0, state: 'idle' };
      }
      const state = this.requireState(requestId);
      return {
        slotId,
        inUse: true,
        tokens: state.promptTokens.length + state.decodeGenerated,
        state: state.done ? 'done' : state.stage,
      };
    });
  }

  async dispose(): Promise<void> {
    await this.runtime.exit();
    this.states.clear();
  }

  private async ensureLoaded(): Promise<void> {
    if (this.loaded) {
      return;
    }
    if (!this.loadPromise) {
      this.loadPromise = this.loadRuntime();
    }
    return this.loadPromise;
  }

  private async loadRuntime(): Promise<void> {
    try {
      this.emitRuntime({
        phase: 'load',
        message: 'Loading model metadata and WebAssembly runtime',
      });
      const loadParams = {
        n_ctx: this.config.nCtx ?? DEFAULT_CTX,
        n_batch: this.config.nBatch ?? DEFAULT_BATCH,
        n_threads: this.config.nThreads,
        n_gpu_layers: this.config.nGpuLayers ?? detectGpuLayers(),
        ...(this.config.flashAttn === undefined
          ? {}
          : { flash_attn: this.config.flashAttn }),
        n_parallel: Math.max(1, this.config.nParallel ?? DEFAULT_PARALLEL),
        // Bounded units deliberately stop between prompt chunks and resume
        // from the same serving slot. A unified KV cache clears that slot
        // whenever it becomes idle, so keep independent slot state and
        // disable the server-managed idle prompt cache.
        kv_unified: false,
        // The coordinator owns cross-session scheduling. Native continuous
        // batching can leave an aborted background completion in the next
        // graph, making a promoted session decode with the whole resident
        // set even after the coordinator has serialized its own work.
        cont_batching: false,
        cache_idle_slots: false,
        warmup: true,
        seed: this.config.seed ?? 0,
        progressCallback: (progress: { loaded: number; total: number }) => {
          this.emitRuntime({
            phase: 'download',
            loaded: progress.loaded,
            total: progress.total,
          });
        },
      };
      if (
        this.config.modelLoadMode === 'direct' &&
        typeof this.runtime.loadModelDirect === 'function'
      ) {
        await this.runtime.loadModelDirect(this.config.modelUrl, loadParams);
      } else {
        await this.runtime.loadModelFromUrl(this.config.modelUrl, {
          ...loadParams,
          useCache: true,
        });
      }
      this.loaded = true;
      this.applyGpuBurstPolicy();
      this.emitRuntime({ phase: 'ready', message: 'Model ready' });
    } catch (error) {
      this.emitRuntime({ phase: 'error', message: errorMessage(error) });
      throw new WllmError(
        `failed to load wllama model: ${errorMessage(error)}`,
        'native_load_error'
      );
    }
  }

  private applyGpuBurstPolicy(): void {
    if (!this.pendingGpuPolicy || !this.runtime.setWebGpuPolicy) {
      return;
    }
    if (
      this.appliedGpuPolicy?.level === this.pendingGpuPolicy.level &&
      this.appliedGpuPolicy.commandSubmitBatchSize ===
        this.pendingGpuPolicy.commandSubmitBatchSize &&
      this.appliedGpuPolicy.maxInflightBatches ===
        this.pendingGpuPolicy.maxInflightBatches
    ) {
      return;
    }
    this.runtime.setWebGpuPolicy({
      commandSubmitBatchSize: this.pendingGpuPolicy.commandSubmitBatchSize,
      maxInflightBatches: this.pendingGpuPolicy.maxInflightBatches,
    });
    this.appliedGpuPolicy = { ...this.pendingGpuPolicy };
  }

  private emitRuntime(
    event: Omit<NativeRuntimeEvent, 'type' | 'modelUrl'>
  ): void {
    const runtimeEvent: NativeRuntimeEvent = {
      type: 'runtime',
      modelUrl: this.config.modelUrl,
      ...event,
    };
    this.config.onRuntimeEvent?.(runtimeEvent);
  }

  private allocateSlot(owner: string): number | null {
    for (let offset = 0; offset < this.slots.size; offset += 1) {
      const slotId = (this.nextSlot + offset) % this.slots.size;
      if (this.slots.get(slotId) === null) {
        this.slots.set(slotId, owner);
        this.nextSlot = (slotId + 1) % this.slots.size;
        return slotId;
      }
    }
    return null;
  }

  // claim one specific idle slot; restores use this so the destination
  // sequence matches the one that produced the state
  private claimSlot(slotId: number, owner: string): number | null {
    if (this.slots.get(slotId) !== null) {
      return null;
    }
    this.slots.set(slotId, owner);
    return slotId;
  }

  private requireState(requestId: string): WllamaRequestState {
    const state = this.states.get(requestId);
    if (!state) {
      throw new WllmError(
        `native request not found: ${requestId}`,
        'unknown_request'
      );
    }
    return state;
  }

  private requireSlotState(slotId: number): WllamaRequestState {
    const requestId = this.slots.get(slotId);
    if (!requestId) {
      throw new WllmError(`slot is empty: ${slotId}`, 'empty_slot');
    }
    return this.requireState(requestId);
  }

  // continuation metadata travels with every checkpoint format; the KV bytes
  // themselves are either parked natively or serialized separately
  private continuationBytes(state: WllamaRequestState): Uint8Array {
    const continuation: SerializedContinuation = {
      formatVersion: 1,
      promptText: state.promptText,
      generatedText: state.generatedText,
      promptTokens: [...state.promptTokens],
      prefillProcessed: state.prefillProcessed,
      prefillTotal: state.prefillTotal,
      decodeGenerated: state.decodeGenerated,
      maxNewTokens: state.maxNewTokens,
      seed: state.seed,
      temperature: state.temperature,
      stage: state.stage,
      modelHash: this.modelHash,
      originSlotId: state.slotId,
    };
    return new TextEncoder().encode(JSON.stringify(continuation));
  }

  private progress(
    state: WllamaRequestState,
    outputChunks: string[],
    elapsedMs: number
  ): NativeUnitProgress {
    return this.progressWithStop(state, outputChunks, elapsedMs, undefined);
  }

  private progressWithStop(
    state: WllamaRequestState,
    outputChunks: string[],
    elapsedMs: number,
    unitStop: NativeUnitProgress['unitStop']
  ): NativeUnitProgress {
    return {
      requestId: state.requestId,
      stage: state.done ? 'done' : state.stage,
      prefillProcessed: state.prefillProcessed,
      prefillTotal: state.prefillTotal,
      decodeGenerated: state.decodeGenerated,
      decodeTotal: state.maxNewTokens,
      done: state.done,
      cancelled: state.cancelled,
      isError: false,
      outputChunks: [...outputChunks],
      elapsedMs,
      slotId: state.done && state.cancelled ? null : state.slotId,
      tokenAccounting: state.tokenAccounting,
      ...(state.counters ? { counters: { ...state.counters } } : {}),
      prefillCached: state.prefillCached,
      residentTokens: state.prefillProcessed,
      boundedStops: state.boundedStops,
      ...(unitStop === undefined ? {} : { unitStop }),
    };
  }
}

function createWllamaRuntime(
  wasmUrl: string,
  config: WllamaNativeEngineConfig
): WllamaRuntime {
  const runtimeEnv = webgpuRuntimeEnv(config);
  const runtime = new Wllama(
    { default: wasmUrl },
    {
      logger: LoggerWithoutDebug,
      allowOffline: true,
      parallelDownloads: 3,
      ...(Object.keys(runtimeEnv).length > 0 ? { runtimeEnv } : {}),
    }
  );
  // Browsers without JSPI/MEMORY64 (Safari, older Chromium) need the Asyncify
  // build. Serving it from our own origin keeps the demo free of CDN calls.
  runtime.setCompat({
    wasm: compatWasmUrl,
    worker: { code: compatWorkerCode },
  });
  const runtimeWithDirectLoad = runtime as unknown as WllamaRuntime;
  runtimeWithDirectLoad.loadModelDirect = async (source, params = {}) => {
    const requestInit: RequestInit = {};
    const signal = params.signal;
    if (signal instanceof AbortSignal) requestInit.signal = signal;
    const response = await fetch(source, requestInit);
    if (!response.ok || !response.body) {
      throw new Error(`Failed to fetch ${source}: HTTP ${response.status}`);
    }
    const blob = await response.blob();
    const progressCallback = params.progressCallback as
      | ((progress: { loaded: number; total: number }) => void)
      | undefined;
    progressCallback?.({ loaded: blob.size, total: blob.size });
    type LoadModelParams = NonNullable<Parameters<Wllama['loadModel']>[1]>;
    await runtime.loadModel([blob], params as LoadModelParams);
  };
  return runtime as unknown as WllamaRuntime;
}

export function webgpuRuntimeEnv(
  config: WllamaNativeEngineConfig
): Record<string, string> {
  const env: Record<string, string> = {};
  // Exported for the runtime-parity unit test; the engine calls this helper
  // before constructing wllama.
  if (config.queueWaitTimeoutMs !== undefined) {
    env.GGML_WEBGPU_WAIT_TIMEOUT_MS = String(
      Math.max(30_000, Math.floor(config.queueWaitTimeoutMs))
    );
  }
  if (config.webgpuCommandSubmitBatchSize !== undefined) {
    env.GGML_WEBGPU_COMMAND_SUBMIT_BATCH_SIZE = String(
      Math.max(1, Math.floor(config.webgpuCommandSubmitBatchSize))
    );
  }
  if (config.webgpuMaxInflightBatches !== undefined) {
    env.GGML_WEBGPU_MAX_INFLIGHT_BATCHES = String(
      Math.max(1, Math.floor(config.webgpuMaxInflightBatches))
    );
  }
  return env;
}

// the fork wraps a sequence state blob in a small header; describing it here
// turns an opaque native status code into something a run can act on
function describeSlotBlob(kv: Uint8Array, tokens: number[]): string {
  const parts = [`kv=${kv.byteLength}B`, `tokens=${tokens.length}`];
  if (kv.byteLength < 24) {
    parts.push('header=truncated');
    return parts.join(' ');
  }
  const view = new DataView(kv.buffer, kv.byteOffset, kv.byteLength);
  const magic = view.getUint32(0, true);
  const version = view.getUint32(4, true);
  const sizeMain = Number(view.getBigUint64(8, true));
  const sizeDraft = Number(view.getBigUint64(16, true));
  const expected = 24 + sizeMain + sizeDraft;
  const badTokens = tokens.filter((token) => !Number.isInteger(token)).length;
  parts.push(
    `magic=${magic.toString(16)}`,
    `version=${version}`,
    `size_main=${sizeMain}`,
    `size_draft=${sizeDraft}`,
    `expected_kv=${expected}`,
    `header_ok=${
      magic === 0x54534c53 && version === 1 && expected === kv.byteLength
    }`,
    `first_token=${tokens[0] ?? 'none'}`,
    `non_integer_tokens=${badTokens}`
  );
  // the packed stream is [marker, version, count, ids..., mediaCount, ...];
  // decoding it here shows whether a token left the model's id range
  if (tokens.length >= 4 && tokens[0] === -1 && tokens[1] === 1) {
    const count = tokens[2] ?? 0;
    const ids = tokens.slice(3, 3 + count);
    const mediaCount = tokens[3 + count] ?? -1;
    const outOfRange = ids.filter(
      (id) => !Number.isInteger(id) || id < 0 || id >= 32_000
    ).length;
    parts.push(
      `prompt_tokens=${count}`,
      `token_min=${Math.min(...ids)}`,
      `token_max=${Math.max(...ids)}`,
      `out_of_range=${outOfRange}`,
      `media_slots=${mediaCount}`,
      `packed_words_used=${4 + count + mediaCount}`
    );
  }
  return parts.join(' ');
}

function decodeContinuation(bytes: Uint8Array): SerializedContinuation {

  let parsed: unknown;
  try {
    parsed = JSON.parse(new TextDecoder().decode(bytes));
  } catch {
    throw new WllmError('invalid continuation checkpoint', 'checkpoint_mismatch');
  }
  if (!isSerializedContinuation(parsed)) {
    throw new WllmError(
      'unsupported continuation checkpoint',
      'checkpoint_mismatch'
    );
  }
  return parsed;
}

function isSerializedContinuation(value: unknown): value is SerializedContinuation {
  if (typeof value !== 'object' || value === null) {
    return false;
  }
  const candidate = value as Partial<SerializedContinuation>;
  return (
    candidate.formatVersion === 1 &&
    typeof candidate.promptText === 'string' &&
    typeof candidate.generatedText === 'string' &&
    Array.isArray(candidate.promptTokens) &&
    typeof candidate.prefillProcessed === 'number' &&
    typeof candidate.prefillTotal === 'number' &&
    typeof candidate.decodeGenerated === 'number' &&
    typeof candidate.maxNewTokens === 'number' &&
    (candidate.stage === 'prefill' || candidate.stage === 'decode') &&
    typeof candidate.modelHash === 'string'
  );
}

function tokensToText(tokens: number[]): string {
  return tokens.join(' ');
}

function detectGpuLayers(): number {
  return typeof navigator !== 'undefined' && 'gpu' in navigator ? 99_999 : 0;
}

function mapNativeBoundedReason(
  reason: NonNullable<WllamaCompletionChunk['timings']>['bounded_stop_reason']
): NativeUnitProgress['unitStop'] {
  switch (reason) {
    case 'prefill-budget':
      return 'native-prefill-budget';
    case 'decode-budget':
      return 'native-decode-budget';
    case 'unit-time-budget':
    case 'predict-time-budget':
      return 'native-time-budget';
  }
}

function isAbortError(error: unknown): boolean {
  return (
    (typeof error === 'object' &&
      error !== null &&
      'name' in error &&
      (error as { name?: unknown }).name === 'AbortError') ||
    (typeof error === 'object' &&
      error !== null &&
      'name' in error &&
      (error as { name?: unknown }).name === 'WllamaAbortError')
  );
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
