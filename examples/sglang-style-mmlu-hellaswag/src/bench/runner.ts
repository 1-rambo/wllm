import { Wllama } from '@wllama/wllama';
import wllamaSingle from '@wllama/wllama/src/single-thread/wllama.wasm?url';
import wllamaMulti from '@wllama/wllama/src/multi-thread/wllama.wasm?url';
import type {
  BenchConfig,
  BenchDiagnostics,
  BenchLogEvent,
  BenchProgressEvent,
  BenchReport,
  BenchSummary,
  CacheProfile,
  HellaSwagItem,
  MmluExperimentMode,
  MMLUItem,
  QueueVsDirectSummary,
  QAResult,
} from './types';

const WLLAMA_CONFIG_PATHS = {
  'single-thread/wllama.wasm': wllamaSingle,
  'multi-thread/wllama.wasm': wllamaMulti,
};

const CHOICE_LABELS = ['A', 'B', 'C', 'D'] as const;
const DECODE_CHUNK_SIZE = 128;
const DECODE_RETRY_CHUNK_SIZE = 32;
const DECODE_STEP_TIMEOUT_MS = 10000;
const EXIT_TIMEOUT_MS = 2000;
const BENCH_REQUEST_TIMEOUT_MS = 30000;
const EXP4_REQUEST_TIMEOUT_MS = 30000;
const INT32_MAX = 0x7fffffff;
const DEFAULT_TRUE_TREE_TIER_L1_CAP = 8192;
const DEFAULT_TRUE_TREE_TIER_L2_CAP = 32768;
const DEFAULT_TRUE_TREE_TIER_L3_CAP = 131072;
const TREE_SEQ_HARD_LIMIT = 256;
const TREE_SEQ_REBUILD_WATERMARK = 240;

function assertNotAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw new DOMException('Aborted', 'AbortError');
  }
}

function avg(values: number[]): number {
  if (!values.length) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function makeSeededRng(seed: number): () => number {
  let s = (seed >>> 0) || 1;
  return () => {
    s = (1664525 * s + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

function shuffleWithSeed<T>(items: T[], seed: number): T[] {
  const out = items.slice();
  const rng = makeSeededRng(seed);
  for (let i = out.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rng() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

function safeText(text: string): string {
  return text.replace(/\s+/g, ' ').trim();
}

function stringifyLoggerArgs(args: unknown[]): string {
  return args
    .map((v) => {
      if (typeof v === 'string') return v;
      try {
        return JSON.stringify(v);
      } catch {
        return String(v);
      }
    })
    .join(' ')
    .trim();
}

function createBenchWllamaLogger(onLog?: (e: BenchLogEvent) => void) {
  const NATIVE_KEEP_PATTERNS = [
    '@@ERROR@@',
    '@@WARN@@',
    'GGML_ASSERT(',
    'Device lost',
    'WaitAny returned',
  ];
  const relayTrace = (...args: unknown[]) => {
    if (!onLog) return;
    const text = stringifyLoggerArgs(args);
    if (NATIVE_KEEP_PATTERNS.some((p) => text.includes(p))) {
      onLog({ text });
      return;
    }
    // Intentionally drop EngineChatTrace to keep benchmark logs concise.
  };

  return {
    debug: (...args: unknown[]) => {
      console.debug(...args);
      relayTrace(...args);
    },
    log: (...args: unknown[]) => {
      console.log(...args);
      relayTrace(...args);
    },
    warn: (...args: unknown[]) => {
      console.warn(...args);
      relayTrace(...args);
    },
    error: (...args: unknown[]) => {
      console.error(...args);
      relayTrace(...args);
    },
  };
}

function estimateMmluEvalItemCount(
  data: MMLUItem[],
  shots: number,
  evalCount: number,
  experimentMode: MmluExperimentMode
): number {
  const grouped = new Map<string, number>();
  for (const item of data) {
    const key = item.subject || 'unknown';
    grouped.set(key, (grouped.get(key) ?? 0) + 1);
  }

  let base = 0;
  for (const count of grouped.values()) {
    const avail = Math.max(0, count - shots);
    base += Math.min(evalCount, avail);
  }
  return experimentMode === 'exp2-random-twice' ? base * 2 : base;
}

function estimateHellaEvalItemCount(data: HellaSwagItem[], shots: number, evalCount: number): number {
  return Math.min(evalCount, Math.max(0, data.length - shots));
}

function isDecodeTimeoutError(err: unknown): boolean {
  const msg = err instanceof Error ? err.message : String(err);
  return msg.includes('decode chunk timeout');
}

function withTimeout<T>(
  promise: Promise<T>,
  timeoutMs: number,
  timeoutMessage: string,
  onTimeout?: () => void
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timeoutId = setTimeout(() => {
      onTimeout?.();
      reject(new Error(timeoutMessage));
    }, timeoutMs);

    promise
      .then((value) => {
        clearTimeout(timeoutId);
        resolve(value);
      })
      .catch((err) => {
        clearTimeout(timeoutId);
        reject(err);
      });
  });
}

function isExp4TimeoutError(err: unknown): boolean {
  const msg = err instanceof Error ? err.message : String(err);
  return msg.includes('[Exp4Timeout]');
}

function isBenchTimeoutError(err: unknown): boolean {
  const msg = err instanceof Error ? err.message : String(err);
  return msg.includes('[BenchTimeout]');
}

function withBenchTimeout<T>(promise: Promise<T>, phase: string): Promise<T> {
  return withTimeout(
    promise,
    BENCH_REQUEST_TIMEOUT_MS,
    `[BenchTimeout] ${phase} timed out after ${BENCH_REQUEST_TIMEOUT_MS}ms`
  );
}

function mergeAbortSignals(a?: AbortSignal, b?: AbortSignal): AbortSignal | undefined {
  if (!a) return b;
  if (!b) return a;
  const ctor = AbortSignal as unknown as { any?: (signals: AbortSignal[]) => AbortSignal };
  if (typeof ctor.any === 'function') {
    return ctor.any([a, b]);
  }
  const c = new AbortController();
  const abort = () => c.abort();
  if (a.aborted || b.aborted) {
    c.abort();
    return c.signal;
  }
  a.addEventListener('abort', abort, { once: true });
  b.addEventListener('abort', abort, { once: true });
  return c.signal;
}

function withBenchTimeoutAbortable<T>(
  run: (timeoutSignal: AbortSignal) => Promise<T>,
  phase: string
): Promise<T> {
  const timeoutController = new AbortController();
  return withTimeout(
    run(timeoutController.signal),
    BENCH_REQUEST_TIMEOUT_MS,
    `[BenchTimeout] ${phase} timed out after ${BENCH_REQUEST_TIMEOUT_MS}ms`,
    () => timeoutController.abort()
  );
}

function chatFromNodeWithBenchTimeout(
  runtime: Wllama,
  parentNodeId: number,
  prompt: string,
  options: Record<string, unknown>,
  phase: string
): Promise<{ nodeId: number; assistantText: string }> {
  return withBenchTimeoutAbortable((timeoutSignal) => {
    const mergedSignal = mergeAbortSignals(
      options.abortSignal as AbortSignal | undefined,
      timeoutSignal
    );
    return runtime.chatFromNode(parentNodeId, prompt, {
      ...options,
      abortSignal: mergedSignal,
    });
  }, phase);
}

function isRuntimeDisposedError(err: unknown): boolean {
  const msg = (err instanceof Error ? err.message : String(err)).toLowerCase();
  return msg.includes('terminated')
    || msg.includes('disposed')
    || msg.includes('proxy')
    || msg.includes('worker');
}

function classifyBenchFailure(err: unknown): 'timeout' | 'abort' | 'disposed' | 'other' {
  const msg = err instanceof Error ? err.message : String(err);
  if (isBenchTimeoutError(err)) {
    return 'timeout';
  }
  if (/abort signal from llama\.cpp/i.test(msg)) {
    return 'abort';
  }
  if (isRuntimeDisposedError(err)) {
    return 'disposed';
  }
  return 'other';
}

function extractBenchTimeoutPhase(message: string): string {
  const m = message.match(/\[BenchTimeout\]\s+(.+?)\s+timed out after/i);
  return m?.[1] ?? 'unknown-phase';
}

function getTreeMemoryCapBytes(
  memoryCapMB: number,
  onLog?: (e: BenchLogEvent) => void,
  scope?: string
): number {
  const requestedBytes = Math.max(0, Math.floor(memoryCapMB * 1024 * 1024));
  const clampedBytes = Math.min(requestedBytes, INT32_MAX - 1);
  if (requestedBytes !== clampedBytes) {
    onLog?.({
      text: `[TreeCap${scope ? `/${scope}` : ''}] requestedBytes=${requestedBytes} exceeds safe int32 cap=${INT32_MAX - 1}, clamped=${clampedBytes}`,
    });
  }
  return clampedBytes;
}

function buildTieredCacheOptions(config: BenchConfig, onLog?: (e: BenchLogEvent) => void): {
  enabled: boolean;
  l1TokenCap?: number;
  l2TokenCap?: number;
  l3TokenCap?: number;
  pruneL1L2TokenThreshold?: number;
  pruneL2L3TokenThreshold?: number;
  replacementPolicy?: BenchConfig['trueTreeReplacementPolicy'];
  fallbackApplied: boolean;
} {
  if (!config.trueTreeTieredCacheEnabled) {
    return { enabled: false, fallbackApplied: false };
  }

  const l1TokenCap = Math.max(0, Number(config.trueTreeTierL1TokenCap) || 0);
  const l2TokenCap = Math.max(0, Number(config.trueTreeTierL2TokenCap) || 0);
  const l3TokenCap = Math.max(0, Number(config.trueTreeTierL3TokenCap) || 0);
  const pruneL1L2TokenThreshold = Math.max(0, Number(config.trueTreePruneL1L2TokenThreshold) || 0);
  const pruneL2L3TokenThreshold = Math.max(0, Number(config.trueTreePruneL2L3TokenThreshold) || 0);
  const replacementPolicy = config.trueTreeReplacementPolicy;

  const allCapsZero = l1TokenCap === 0 && l2TokenCap === 0 && l3TokenCap === 0;
  if (allCapsZero) {
    onLog?.({
      text: `[WARN] True-tree tiered cache enabled but all tier caps are 0; applying defaults L1/L2/L3=${DEFAULT_TRUE_TREE_TIER_L1_CAP}/${DEFAULT_TRUE_TREE_TIER_L2_CAP}/${DEFAULT_TRUE_TREE_TIER_L3_CAP}`,
    });

    return {
      enabled: true,
      l1TokenCap: DEFAULT_TRUE_TREE_TIER_L1_CAP,
      l2TokenCap: DEFAULT_TRUE_TREE_TIER_L2_CAP,
      l3TokenCap: DEFAULT_TRUE_TREE_TIER_L3_CAP,
      pruneL1L2TokenThreshold,
      pruneL2L3TokenThreshold,
      replacementPolicy,
      fallbackApplied: true,
    };
  }

  return {
    enabled: true,
    l1TokenCap,
    l2TokenCap,
    l3TokenCap,
    pruneL1L2TokenThreshold,
    pruneL2L3TokenThreshold,
    replacementPolicy,
    fallbackApplied: false,
  };
}

type TreeProbeState = {
  rootId: number;
  activeNodeId: number;
  nodeCount: number;
  contextMemoryBytes: number;
  memoryCapBytes: number;
  totalSnapshotTokenBytes: number;
  tierStats: {
    l1Tokens: number;
    l2Tokens: number;
    l3Tokens: number;
    l1Slots: number;
    l2Slots: number;
    l3Slots: number;
    promotions: number;
    demotions: number;
    diskReads: number;
    diskWrites: number;
    l3OverflowEvents: number;
    restoreAttempts: number;
    restoreHitsL1: number;
    restoreHitsL2: number;
    restoreHitsL3: number;
    restoreMisses: number;
    restoreRebuilds: number;
    parentRecoverAttempts: number;
    parentRecoverSuccesses: number;
    parentRecoverFailures: number;
  };
};

async function probeTreeState(
  runtime: Wllama,
  label: string,
  onLog?: (e: BenchLogEvent) => void
): Promise<TreeProbeState | null> {
  try {
    const state = await withBenchTimeout(
      (runtime as unknown as {
        chatGetState: () => Promise<{
          nodes: Map<number, unknown>;
          rootId: number;
          activeNodeId: number;
          contextMemoryBytes: number;
          memoryCapBytes: number;
          totalSnapshotTokenBytes: number;
          tierStats: {
            l1Tokens: number;
            l2Tokens: number;
            l3Tokens: number;
            l1Slots: number;
            l2Slots: number;
            l3Slots: number;
            promotions: number;
            demotions: number;
            diskReads: number;
            diskWrites: number;
            l3OverflowEvents: number;
            restoreAttempts: number;
            restoreHitsL1: number;
            restoreHitsL2: number;
            restoreHitsL3: number;
            restoreMisses: number;
            restoreRebuilds: number;
            parentRecoverAttempts: number;
            parentRecoverSuccesses: number;
            parentRecoverFailures: number;
          };
        }>;
      }).chatGetState(),
      `Exp3Probe/${label}/chatGetState`
    );

    const probed: TreeProbeState = {
      rootId: state.rootId,
      activeNodeId: state.activeNodeId,
      nodeCount: state.nodes?.size ?? 0,
      contextMemoryBytes: state.contextMemoryBytes ?? 0,
      memoryCapBytes: state.memoryCapBytes ?? 0,
      totalSnapshotTokenBytes: state.totalSnapshotTokenBytes ?? 0,
      tierStats: {
        l1Tokens: state.tierStats?.l1Tokens ?? 0,
        l2Tokens: state.tierStats?.l2Tokens ?? 0,
        l3Tokens: state.tierStats?.l3Tokens ?? 0,
        l1Slots: state.tierStats?.l1Slots ?? 0,
        l2Slots: state.tierStats?.l2Slots ?? 0,
        l3Slots: state.tierStats?.l3Slots ?? 0,
        promotions: state.tierStats?.promotions ?? 0,
        demotions: state.tierStats?.demotions ?? 0,
        diskReads: state.tierStats?.diskReads ?? 0,
        diskWrites: state.tierStats?.diskWrites ?? 0,
        l3OverflowEvents: state.tierStats?.l3OverflowEvents ?? 0,
        restoreAttempts: state.tierStats?.restoreAttempts ?? 0,
        restoreHitsL1: state.tierStats?.restoreHitsL1 ?? 0,
        restoreHitsL2: state.tierStats?.restoreHitsL2 ?? 0,
        restoreHitsL3: state.tierStats?.restoreHitsL3 ?? 0,
        restoreMisses: state.tierStats?.restoreMisses ?? 0,
        restoreRebuilds: state.tierStats?.restoreRebuilds ?? 0,
        parentRecoverAttempts: state.tierStats?.parentRecoverAttempts ?? 0,
        parentRecoverSuccesses: state.tierStats?.parentRecoverSuccesses ?? 0,
        parentRecoverFailures: state.tierStats?.parentRecoverFailures ?? 0,
      },
    };

    const restoreHitTotal =
      probed.tierStats.restoreHitsL1 +
      probed.tierStats.restoreHitsL2 +
      probed.tierStats.restoreHitsL3;
    const restoreHitRatePct =
      probed.tierStats.restoreAttempts > 0
        ? (restoreHitTotal / probed.tierStats.restoreAttempts) * 100
        : 0;
    const restoreL1RatePct =
      probed.tierStats.restoreAttempts > 0
        ? (probed.tierStats.restoreHitsL1 / probed.tierStats.restoreAttempts) * 100
        : 0;
    const restoreL2RatePct =
      probed.tierStats.restoreAttempts > 0
        ? (probed.tierStats.restoreHitsL2 / probed.tierStats.restoreAttempts) * 100
        : 0;
    const restoreL3RatePct =
      probed.tierStats.restoreAttempts > 0
        ? (probed.tierStats.restoreHitsL3 / probed.tierStats.restoreAttempts) * 100
        : 0;

    onLog?.({
      text: `[Exp3Probe/${label}] root=${probed.rootId} active=${probed.activeNodeId} nodes=${probed.nodeCount} ctxBytes=${probed.contextMemoryBytes} capBytes=${probed.memoryCapBytes} snapshotBytes=${probed.totalSnapshotTokenBytes} tierTokens(L1/L2/L3)=${probed.tierStats.l1Tokens}/${probed.tierStats.l2Tokens}/${probed.tierStats.l3Tokens} tierSlots(L1/L2/L3)=${probed.tierStats.l1Slots}/${probed.tierStats.l2Slots}/${probed.tierStats.l3Slots} promo/demote=${probed.tierStats.promotions}/${probed.tierStats.demotions} diskR/W=${probed.tierStats.diskReads}/${probed.tierStats.diskWrites} l3Overflow=${probed.tierStats.l3OverflowEvents} restoreHit=${restoreHitTotal}/${probed.tierStats.restoreAttempts}(${restoreHitRatePct.toFixed(1)}%) byTier(L1/L2/L3)=${probed.tierStats.restoreHitsL1}/${probed.tierStats.restoreHitsL2}/${probed.tierStats.restoreHitsL3}(${restoreL1RatePct.toFixed(1)}%/${restoreL2RatePct.toFixed(1)}%/${restoreL3RatePct.toFixed(1)}%) misses=${probed.tierStats.restoreMisses} rebuilds=${probed.tierStats.restoreRebuilds}`,
    });
    return probed;
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    onLog?.({ text: `[Exp3Probe/${label}] unavailable: ${msg}` });
    return null;
  }
}

async function logTreeContentPreview(
  runtime: Wllama,
  label: string,
  onLog?: (e: BenchLogEvent) => void
): Promise<void> {
  try {
    const state = await withBenchTimeout(
      (runtime as unknown as {
        chatGetState: () => Promise<{
          nodes: Map<number, {
            id: number;
            status: string;
            cachedTokenCount: number;
            snapshotTokenBytes: number;
            turn?: {
              user?: string;
              assistant?: string;
            };
          }>;
          tierStats: {
            l1Tokens: number;
            l2Tokens: number;
            l3Tokens: number;
            diskWrites: number;
            diskReads: number;
          };
        }>;
      }).chatGetState(),
      `Exp3Preview/${label}/chatGetState`
    );

    const nodes = Array.from(state.nodes.values())
      .filter((n) => (n.snapshotTokenBytes ?? 0) > 0)
      .sort((a, b) => (b.snapshotTokenBytes ?? 0) - (a.snapshotTokenBytes ?? 0))
      .slice(0, 3);

    onLog?.({
      text: `[Exp3Preview/${label}] tierTokens(L1/L2/L3)=${state.tierStats?.l1Tokens ?? 0}/${state.tierStats?.l2Tokens ?? 0}/${state.tierStats?.l3Tokens ?? 0} diskR/W=${state.tierStats?.diskReads ?? 0}/${state.tierStats?.diskWrites ?? 0} sampledNodes=${nodes.length}`,
    });

    void nodes;
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    onLog?.({ text: `[Exp3Preview/${label}] unavailable: ${msg}` });
  }
}

function hardTerminateWllama(
  wllama: Wllama,
  onLog?: (e: BenchLogEvent) => void,
  label: string = 'wllama'
): boolean {
  try {
    const runtime = wllama as unknown as {
      proxy?: {
        worker?: Worker;
      };
    };
    const worker = runtime.proxy?.worker;
    if (!worker) {
      onLog?.({ text: `[${label}] hard terminate: worker not found` });
      return false;
    }

    worker.terminate();
    if (runtime.proxy) {
      runtime.proxy.worker = undefined;
    }
    (wllama as unknown as { proxy?: unknown }).proxy = null;
    onLog?.({ text: `[${label}] hard terminate done` });
    return true;
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    onLog?.({ text: `[${label}] hard terminate error: ${msg}` });
    return false;
  }
}

async function safeExitWllama(
  wllama: Wllama,
  onLog?: (e: BenchLogEvent) => void,
  label: string = 'wllama'
): Promise<boolean> {
  try {
    let timedOut = false;
    await Promise.race([
      wllama.exit(),
      new Promise<void>((resolve) => {
        setTimeout(() => {
          timedOut = true;
          resolve();
        }, EXIT_TIMEOUT_MS);
      }),
    ]);
    if (timedOut) {
      onLog?.({ text: `[${label}] exit timeout after ${EXIT_TIMEOUT_MS}ms` });
      onLog?.({ text: `[${label}] trying hard terminate...` });
      return hardTerminateWllama(wllama, onLog, label);
    }
    onLog?.({ text: `[${label}] exit done` });
    return true;
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    onLog?.({ text: `[${label}] exit error: ${msg}` });
    onLog?.({ text: `[${label}] trying hard terminate after exit error...` });
    return hardTerminateWllama(wllama, onLog, label);
  }
}

async function createLoadedWllama(
  config: BenchConfig,
  onLog?: (e: BenchLogEvent) => void
): Promise<Wllama> {
  const wllama = new Wllama(WLLAMA_CONFIG_PATHS, {
    preferWebGPU: true,
    noPerf: false,
    suppressNativeLog: false,
    engineChatTraceEnabled: true,
    logger: createBenchWllamaLogger(onLog),
  });
  onLog?.({ text: '[Wllama] loading model...' });
  await wllama.loadModelFromUrl(config.modelUrl, {
    useCache: true,
    n_ctx: config.nCtx,
    n_batch: config.nBatch,
    // Keep full KV buffer semantics; avoids seq_cp assertion in tree save/restore.
    n_seq_max: 1,
    kv_unified: true,
  });
  onLog?.({ text: '[Wllama] model loaded.' });
  return wllama;
}

async function replaceLoadedWllama(
  oldWllama: Wllama | null,
  config: BenchConfig,
  onLog?: (e: BenchLogEvent) => void,
  reason: string = 'replace'
): Promise<Wllama> {
  if (oldWllama) {
    const skipProbeBeforeDispose = /timeout|disposed/i.test(reason);
    if (!skipProbeBeforeDispose) {
      await probeTreeState(oldWllama, `before-dispose/${reason}`, onLog);
    } else {
      onLog?.({ text: `[Wllama] skip pre-dispose probe for fast recovery (${reason})` });
    }
    onLog?.({ text: `[Wllama] disposing old runtime (${reason})` });
    const exited = await safeExitWllama(oldWllama, onLog, `Wllama/${reason}`);
    if (!exited) {
      throw new Error(`[Wllama] old runtime did not exit cleanly (${reason}), refusing to create new model`);
    }
  }

  const fresh = await createLoadedWllama(config, onLog);
  return fresh;
}

async function logRuntimeDebugSnapshot(
  runtime: Wllama,
  label: string,
  onLog?: (e: BenchLogEvent) => void
): Promise<void> {
  try {
    const debugInfo = await withTimeout(
      (runtime as unknown as { _getDebugInfo: () => Promise<unknown> })._getDebugInfo(),
      1500,
      `[RuntimeDebug/${label}] _getDebugInfo timed out after 1500ms`
    );

    const compact = JSON.stringify(debugInfo);
    const preview = compact.length > 1200 ? `${compact.slice(0, 1200)}...` : compact;
    onLog?.({ text: `[RuntimeDebug/${label}] ${preview}` });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    onLog?.({ text: `[RuntimeDebug/${label}] unavailable: ${msg}` });
  }
}

function mmluQuestionBlock(item: MMLUItem, withAnswer = false): string {
  const lines = [
    'Question:',
    safeText(item.question),
    'Options:',
    `A) ${safeText(item.choices[0])}`,
    `B) ${safeText(item.choices[1])}`,
    `C) ${safeText(item.choices[2])}`,
    `D) ${safeText(item.choices[3])}`,
  ];
  if (withAnswer) {
    lines.push(`Correct Answer: ${CHOICE_LABELS[item.answerIndex]}`);
  } else {
    lines.push('Your Answer:');
  }
  return lines.join('\n');
}

function buildMmluSharedPrefix(shots: MMLUItem[]): string {
  const parts = [
    'You are taking a multiple-choice exam with options A/B/C/D.',
    `Below are ${shots.length} solved examples. Use them as references for style and reasoning depth.`,
    'Each example includes a question, options, and the correct answer.',
    'These examples are finished. Do NOT continue numbering examples in the next response.',
    '',
  ];

  shots.forEach((s, idx) => {
    parts.push(`### Example ${idx + 1} (Solved)`);
    parts.push(mmluQuestionBlock(s, true));
    parts.push('---');
  });

  parts.push('End of solved examples.');
  parts.push('The next task is a NEW question, not an example.');
  parts.push('Respond with only one letter (A, B, C, or D) and no explanation.');

  return parts.join('\n');
}

function buildHellaSharedPrefix(shots: HellaSwagItem[]): string {
  const parts = [
    'Select the most likely continuation among options A/B/C/D.',
    'Choose based on commonsense plausibility.',
    '',
  ];

  for (const s of shots) {
    parts.push(`Context: ${safeText(s.ctx)}`);
    parts.push(`A. ${safeText(s.endings[0])}`);
    parts.push(`B. ${safeText(s.endings[1])}`);
    parts.push(`C. ${safeText(s.endings[2])}`);
    parts.push(`D. ${safeText(s.endings[3])}`);
    parts.push(`Answer: ${CHOICE_LABELS[s.label]}`);
    parts.push('');
  }

  return parts.join('\n');
}

function buildMmluQuestionOnlyPrompt(item: MMLUItem): string {
  return [
    '### New Question (NOT an example)',
    mmluQuestionBlock(item, false),
    'Choose the best option among A, B, C, D.',
    'Respond with only one letter (A, B, C, or D) and no explanation.',
  ].join('\n\n');
}

function buildHellaQuestionOnlyPrompt(item: HellaSwagItem): string {
  return [
    '### New Question (NOT an example)',
    `Context: ${safeText(item.ctx)}`,
    `A. ${safeText(item.endings[0])}`,
    `B. ${safeText(item.endings[1])}`,
    `C. ${safeText(item.endings[2])}`,
    `D. ${safeText(item.endings[3])}`,
    'Answer with exactly one letter: A, B, C, or D.',
  ].join('\n');
}

function parseChoiceIndex(text: string): number {
  const upper = text.toUpperCase();
  const lines = upper
    .split(/\r?\n/)
    .map((x) => x.trim())
    .filter((x) => x.length > 0);

  // Prefer explicit final-answer tags, scanning from the end.
  for (let i = lines.length - 1; i >= 0; i -= 1) {
    const m = lines[i].match(/^FINAL_ANSWER\s*[:：]\s*([ABCD])\b/);
    if (m) {
      return CHOICE_LABELS.indexOf(m[1] as typeof CHOICE_LABELS[number]);
    }
  }

  // Fallback: if the last non-empty line is a single option letter.
  const last = lines[lines.length - 1] ?? '';
  const single = last.match(/^([ABCD])$/);
  if (single) {
    return CHOICE_LABELS.indexOf(single[1] as typeof CHOICE_LABELS[number]);
  }

  return -1;
}

async function runChoicePrompt(
  runtime: Wllama,
  mode: 'flat' | 'tree',
  prompt: string,
  signal?: AbortSignal,
  parentNodeId: number = 0,
  startedAtMs?: number
): Promise<{ predIndex: number; answerText: string; latencyMs: number; ttftMs: number; tokensPerSecond: number }> {
  const t0 = startedAtMs ?? performance.now();
  let firstTokenAt = 0;
  const options = {
    nPredict: 1,
    abortSignal: signal,
    sampling: {
      temp: 0,
      top_k: 1,
      grammar: 'root ::= "A" | "B" | "C" | "D"',
    },
    onChunk: () => {
      if (!firstTokenAt) {
        firstTokenAt = performance.now();
      }
    },
  };

  const answerText = (await chatFromNodeWithBenchTimeout(
    runtime,
    parentNodeId,
    prompt,
    options,
    `runChoicePrompt/${mode}/chatFromNode`
  )).assistantText;

  const latencyMs = Math.max(0, performance.now() - t0);
  const ttftMs = Math.max(0, (firstTokenAt || performance.now()) - t0);
  const outTokenCount = Math.max(1, (await runtime.tokenize(answerText || '', true)).length);
  const tokensPerSecond = (outTokenCount * 1000) / Math.max(1e-6, latencyMs);
  return {
    predIndex: parseChoiceIndex(answerText),
    answerText,
    latencyMs,
    ttftMs,
    tokensPerSecond,
  };
}

async function getNextTokenDistribution(wllama: Wllama): Promise<Map<number, number>> {
  const logits = await wllama.getLogits(-1);
  const map = new Map<number, number>();
  for (const row of logits) {
    map.set(row.token, row.p);
  }
  return map;
}

async function scoreContinuation(
  wllama: Wllama,
  continuationTokens: number[],
  signal?: AbortSignal,
  onFirstToken?: () => void
): Promise<number> {
  let score = 0;
  let seenFirst = false;
  for (const tok of continuationTokens) {
    assertNotAborted(signal);
    const dist = await getNextTokenDistribution(wllama);
    if (!seenFirst) {
      seenFirst = true;
      onFirstToken?.();
    }
    const p = Math.max(dist.get(tok) ?? 1e-12, 1e-12);
    score += Math.log(p);
    await wllama.decode([tok], {});
  }
  return score;
}

async function decodeChunked(
  wllama: Wllama,
  tokens: number[],
  opts: {
    label: string;
    onLog?: (e: BenchLogEvent) => void;
    signal?: AbortSignal;
    chunkSize?: number;
    stepTimeoutMs?: number;
  }
): Promise<void> {
  const {
    label,
    onLog,
    signal,
    chunkSize = DECODE_CHUNK_SIZE,
    stepTimeoutMs = DECODE_STEP_TIMEOUT_MS,
  } = opts;
  const total = tokens.length;
  if (total === 0) {
    return;
  }

  const chunks = Math.ceil(total / chunkSize);

  for (let offset = 0; offset < total; offset += chunkSize) {
    assertNotAborted(signal);
    const end = Math.min(offset + chunkSize, total);
    const idx = Math.floor(offset / chunkSize) + 1;
    const chunkTokens = tokens.slice(offset, end);
    let timeoutId: ReturnType<typeof setTimeout> | null = null;
    try {
      await Promise.race([
        wllama.decode(chunkTokens, {}),
        new Promise<never>((_, reject) => {
          timeoutId = setTimeout(() => {
            reject(
              new Error(
                `[${label}] decode chunk timeout after ${stepTimeoutMs}ms at chunk ${idx}/${chunks} range=${offset}-${end - 1}`
              )
            );
          }, stepTimeoutMs);
        }),
      ]);
    } finally {
      if (timeoutId) clearTimeout(timeoutId);
    }
  }
}

async function runMmlu(
  wllama: Wllama,
  data: MMLUItem[],
  shots: number,
  evalCount: number,
  mode: 'flat' | 'tree',
  config: BenchConfig,
  experimentMode: MmluExperimentMode,
  randomSeed: number,
  slotBase: number,
  onLog?: (e: BenchLogEvent) => void,
  signal?: AbortSignal,
  recoverWllama?: (oldRuntime: Wllama, reason: string) => Promise<Wllama>,
  onQuestionDone?: () => void,
  onQuestionFailure?: (kind: 'timeout' | 'abort' | 'disposed' | 'other', details: string) => void,
  onCacheMaintenanceMs?: (ms: number) => void
): Promise<{
  results: QAResult[];
  latencyMs: number[];
  ttftMs: number[];
  tokensPerSecond: number[];
  wllama: Wllama;
  exp2NodeCacheStats?: {
    attempts: number;
    sharedHits: number;
    sharedMisses: number;
    sharedHitRatePct: number;
    questionHits: number;
    questionMisses: number;
    questionHitRatePct: number;
  };
}> {
  let runtime = wllama;
  const treeMemoryCapBytes = getTreeMemoryCapBytes(config.trueTreeMemoryCapMB, onLog, `MMLU/${mode}`);
  const tieredCacheOpts = buildTieredCacheOptions(config, onLog);
  onLog?.({
    text: `[Exp3Cfg/MMLU/${mode}] chatSessionInit capBytes=${treeMemoryCapBytes} tierEnabled=${tieredCacheOpts.enabled} tierCaps(L1/L2/L3)=${tieredCacheOpts.l1TokenCap ?? 0}/${tieredCacheOpts.l2TokenCap ?? 0}/${tieredCacheOpts.l3TokenCap ?? 0} thresholds(L1L2/L2L3)=${tieredCacheOpts.pruneL1L2TokenThreshold ?? 0}/${tieredCacheOpts.pruneL2L3TokenThreshold ?? 0} replacementPolicy=${tieredCacheOpts.replacementPolicy ?? 'hybrid'} fallbackApplied=${tieredCacheOpts.fallbackApplied}`,
  });
  const rows: QAResult[] = [];
  const latencyMs: number[] = [];
  const ttftMs: number[] = [];
  const tokensPerSecond: number[] = [];
  const trackCacheOp = async <T>(fn: () => Promise<T>): Promise<T> => {
    const t0 = performance.now();
    const out = await fn();
    onCacheMaintenanceMs?.(Math.max(0, performance.now() - t0));
    return out;
  };
  let timeoutDiagCount = 0;

  const grouped = new Map<string, MMLUItem[]>();
  for (const item of data) {
    const key = item.subject || 'unknown';
    const bucket = grouped.get(key);
    if (bucket) {
      bucket.push(item);
    } else {
      grouped.set(key, [item]);
    }
  }

  const subjectEntries = [...grouped.entries()];
  if (experimentMode === 'exp2-random-twice' && mode === 'tree') {
    const plans = subjectEntries.map(([subject, items]) => {
      const shotItems = items.slice(0, shots);
      const baseEvalItems = items.slice(shots, shots + evalCount);
      return {
        subject,
        shotItems,
        baseEvalItems,
        sharedPrefix: buildMmluSharedPrefix(shotItems),
      };
    }).filter((p) => p.baseEvalItems.length > 0);

    if (!plans.length) {
      onLog?.({ text: `[MMLU/${mode}] no eval items for Exp2 random-twice, skipping.` });
      return { results: rows, latencyMs, ttftMs, tokensPerSecond, wllama: runtime };
    }

    if (plans.length <= 1) {
      onLog?.({ text: `[MMLU/${mode}] Exp2 random-twice currently has ${plans.length} subject; randomization is mostly intra-subject.` });
    }

    const runQuestionFromNode = async (
      parentNodeId: number,
      questionPrompt: string,
      startedAt: number
    ): Promise<{
      nodeId: number;
      predIndex: number;
      answerText: string;
      nativeOutput: string;
      nativeFinalOutput: string;
      parseSource: 'prompt_final_answer';
      latencyMs: number;
      ttftMs: number;
      tokensPerSecond: number;
    }> => {
      let firstTokenAt = 0;
      const judged = await chatFromNodeWithBenchTimeout(runtime, parentNodeId, questionPrompt, {
        nPredict: 40,
        abortSignal: signal,
        sampling: {
          temp: 0.2,
          top_p: 0.9,
          top_k: 40,
        },
        onChunk: () => {
          if (!firstTokenAt) {
            firstTokenAt = performance.now();
          }
        },
      }, `MMLU/${mode}/question/chatFromNode`);
      const nativeOutput = judged.assistantText || '';
      const nativeFinalOutput = nativeOutput;
      const bestIdx = parseChoiceIndex(nativeOutput);
      const parseSource: 'prompt_final_answer' = 'prompt_final_answer';
      const answerText = nativeFinalOutput;

      const latency = Math.max(0, performance.now() - startedAt);
      const ttft = Math.max(0, (firstTokenAt || performance.now()) - startedAt);
      const outTokenCount = Math.max(1, (await runtime.tokenize(answerText, true)).length);
      const tps = (outTokenCount * 1000) / Math.max(1e-6, latency);
      return {
        nodeId: judged.nodeId,
        predIndex: bestIdx,
        answerText,
        nativeOutput,
        nativeFinalOutput,
        parseSource,
        latencyMs: latency,
        ttftMs: ttft,
        tokensPerSecond: tps,
      };
    };

    const sharedPrefixBySubject = new Map<string, string>(
      plans.map((p) => [p.subject, p.sharedPrefix])
    );

    const rebuildExp2Session = async () => {
      await trackCacheOp(() => withBenchTimeout(runtime.chatSessionInit(
        treeMemoryCapBytes,
        tieredCacheOpts
      ), `MMLU/${mode}/exp2/chatSessionInit`));
      await probeTreeState(runtime, `MMLU/${mode}/exp2/afterChatSessionInit`, onLog);
      const state = await trackCacheOp(() => withBenchTimeout(
        runtime.chatGetState(),
        `MMLU/${mode}/exp2/chatGetState`
      ));
      const rootNodeId = state.rootId;
      onLog?.({ text: `[MMLU/${mode}] Exp2 shared session ready root=${rootNodeId} (lazy subject prefix warmup)` });
      return rootNodeId;
    };

    let rootNodeId = await rebuildExp2Session();
    let sharedNodeBySubject = new Map<string, number>();

    const ensureSubjectSharedNode = async (subject: string): Promise<number> => {
      const cached = sharedNodeBySubject.get(subject);
      if (cached !== undefined) {
        return cached;
      }

      const sharedPrefix = sharedPrefixBySubject.get(subject) ?? '';
      const setup = await chatFromNodeWithBenchTimeout(runtime, rootNodeId, sharedPrefix, {
        nPredict: 0,
        abortSignal: signal,
      }, `MMLU/${mode}/exp2/subject=${subject}/sharedPrefixChatFromNode`);
      sharedNodeBySubject.set(subject, setup.nodeId);
      onLog?.({ text: `[MMLU/${mode}] subject=${subject} lazy shared node prepared node=${setup.nodeId}` });
      return setup.nodeId;
    };

    const maybeRotateExp2Session = async () => {
      const state = await trackCacheOp(() => withBenchTimeout(
        runtime.chatGetState(),
        `MMLU/${mode}/exp2/seqBudget/chatGetState`
      ));
      const nodeCount = state.nodes?.size ?? 0;
      if (nodeCount < TREE_SEQ_REBUILD_WATERMARK) {
        return;
      }
      onLog?.({
        text: `[MMLU/${mode}] nodeCount=${nodeCount} approaching seq limit=${TREE_SEQ_HARD_LIMIT}; proactively rotating shared session`,
      });
      rootNodeId = await rebuildExp2Session();
      sharedNodeBySubject = new Map<string, number>();
    };

    const evalQueue = plans.flatMap((p) => {
      const doubled: Array<{ subject: string; q: MMLUItem }> = [];
      for (const q of p.baseEvalItems) {
        doubled.push({ subject: p.subject, q });
        doubled.push({ subject: p.subject, q });
      }
      return doubled;
    });
    const evalItems = shuffleWithSeed(evalQueue, randomSeed);
    const questionNodeById = new Map<string, number>();
    let nodeCacheAttempts = 0;
    let sharedNodeCacheHits = 0;
    let questionNodeCacheHits = 0;

    const probeNodeCached = (state: Awaited<ReturnType<Wllama['chatGetState']>>, nodeId?: number): boolean => {
      if (nodeId === undefined) {
        return false;
      }
      const node = state.nodes.get(nodeId);
      return (node?.cachedTokenCount ?? 0) > 0;
    };

    for (let i = 0; i < evalItems.length; i += 1) {
      assertNotAborted(signal);
      await maybeRotateExp2Session();
      const { subject, q } = evalItems[i];

      const cacheProbeState = await trackCacheOp(() => withBenchTimeout(
        runtime.chatGetState(),
        `MMLU/${mode}/exp2/cacheProbe/chatGetState`
      ));
      nodeCacheAttempts += 1;
      const sharedNodeIdForProbe = sharedNodeBySubject.get(subject);
      const questionNodeIdForProbe = questionNodeById.get(q.id);
      const sharedHit = probeNodeCached(cacheProbeState, sharedNodeIdForProbe);
      const questionHit = probeNodeCached(cacheProbeState, questionNodeIdForProbe);
      if (sharedHit) sharedNodeCacheHits += 1;
      if (questionHit) questionNodeCacheHits += 1;

      const questionPrompt = buildMmluQuestionOnlyPrompt(q);
      let solved = false;
      for (let attempt = 0; attempt < 2 && !solved; attempt += 1) {
        try {
          const t0 = performance.now();
          const sharedNodeId = await ensureSubjectSharedNode(subject);
          const out = await runQuestionFromNode(sharedNodeId, questionPrompt, t0);

          latencyMs.push(out.latencyMs);
          ttftMs.push(out.ttftMs);
          tokensPerSecond.push(out.tokensPerSecond);

          const bestIdx = out.predIndex;
          onLog?.({
            text: `[MMLU/${mode}] ${q.id} subject=${subject} pred=${bestIdx >= 0 ? CHOICE_LABELS[bestIdx] : '?'} gt=${CHOICE_LABELS[q.answerIndex]} parse=${out.parseSource} raw="${safeText(out.answerText)}"`,
          });

          rows.push({
            id: q.id,
            gtIndex: q.answerIndex,
            predIndexFlat: -1,
            predIndexTree: bestIdx,
            correctFlat: false,
            correctTree: bestIdx === q.answerIndex,
            explainFlat: '',
            explainTree: `pred=${bestIdx >= 0 ? CHOICE_LABELS[bestIdx] : '?'}`,
          });
          questionNodeById.set(q.id, out.nodeId);
          onQuestionDone?.();
          solved = true;
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          const kind = classifyBenchFailure(err);
          onQuestionFailure?.(kind, `MMLU/${mode}/${subject}/${q.id}: ${msg}`);
          if (kind === 'timeout' && timeoutDiagCount < 3) {
            timeoutDiagCount += 1;
            const phase = extractBenchTimeoutPhase(msg);
            onLog?.({ text: `[TimeoutDiag/MMLU/${mode}] q=${q.id} subject=${subject} attempt=${attempt + 1} phase=${phase} promptChars=${questionPrompt.length} sharedNode=${sharedNodeBySubject.get(subject) ?? 0}` });
            await probeTreeState(runtime, `MMLU/${mode}/timeout/${q.id}/attempt=${attempt + 1}`, onLog);
            await logRuntimeDebugSnapshot(runtime, `MMLU/${mode}/timeout/${q.id}/attempt=${attempt + 1}`, onLog);
          }
          onLog?.({ text: `[MMLU/${mode}] ${q.id} subject=${subject} attempt=${attempt + 1} failed: ${msg}` });
          if (!recoverWllama) {
            throw err;
          }

          const retryable = isBenchTimeoutError(err) || isRuntimeDisposedError(err);
          if (retryable && attempt === 0) {
            onLog?.({ text: `[MMLU/${mode}] ${q.id} subject=${subject} recovering runtime after timeout/disposed...` });
            runtime = await recoverWllama(runtime, `MMLU/${mode} ${q.id} timeout/disposed`);
            rootNodeId = await rebuildExp2Session();
            sharedNodeBySubject = new Map<string, number>();
            questionNodeById.clear();
            onLog?.({ text: `[MMLU/${mode}] ${q.id} subject=${subject} retrying on recreated runtime` });
            continue;
          }

          if (!retryable) {
            runtime = await recoverWllama(runtime, `MMLU/${mode} ${q.id} failed`);
            rootNodeId = await rebuildExp2Session();
            sharedNodeBySubject = new Map<string, number>();
            questionNodeById.clear();
          }

          rows.push({
            id: q.id,
            gtIndex: q.answerIndex,
            predIndexFlat: -1,
            predIndexTree: -1,
            correctFlat: false,
            correctTree: false,
            explainFlat: '',
            explainTree: 'skipped: runtime error',
          });
          onQuestionDone?.();
          solved = true;
        }
      }
    }

    const sharedNodeCacheMisses = Math.max(0, nodeCacheAttempts - sharedNodeCacheHits);
    const questionNodeCacheMisses = Math.max(0, nodeCacheAttempts - questionNodeCacheHits);
    const sharedNodeHitRatePct = nodeCacheAttempts > 0
      ? (sharedNodeCacheHits / nodeCacheAttempts) * 100
      : 0;
    const questionNodeHitRatePct = nodeCacheAttempts > 0
      ? (questionNodeCacheHits / nodeCacheAttempts) * 100
      : 0;
    onLog?.({
      text: `[Diag] exp2NodeCache sharedHit/miss=${sharedNodeCacheHits}/${sharedNodeCacheMisses}(${sharedNodeHitRatePct.toFixed(1)}%) questionHit/miss=${questionNodeCacheHits}/${questionNodeCacheMisses}(${questionNodeHitRatePct.toFixed(1)}%) attempts=${nodeCacheAttempts}`,
    });

    return {
      results: rows,
      latencyMs,
      ttftMs,
      tokensPerSecond,
      wllama: runtime,
      exp2NodeCacheStats: {
        attempts: nodeCacheAttempts,
        sharedHits: sharedNodeCacheHits,
        sharedMisses: sharedNodeCacheMisses,
        sharedHitRatePct: sharedNodeHitRatePct,
        questionHits: questionNodeCacheHits,
        questionMisses: questionNodeCacheMisses,
        questionHitRatePct: questionNodeHitRatePct,
      },
    };
  }

  for (let s = 0; s < subjectEntries.length; s += 1) {
    const [subject, items] = subjectEntries[s];
    const shotItems = items.slice(0, shots);
    const baseEvalItems = items.slice(shots, shots + evalCount);
    let evalItems: MMLUItem[] = [];
    if (experimentMode === 'exp2-random-twice') {
      const doubled: MMLUItem[] = [];
      for (const q of baseEvalItems) {
        doubled.push(q, q);
      }
      evalItems = shuffleWithSeed(doubled, randomSeed + s);
    } else {
      evalItems = baseEvalItems;
    }
    if (!evalItems.length) {
      onLog?.({ text: `[MMLU/${mode}] subject=${subject} has no eval items, skipping.` });
      continue;
    }
    const sharedPrefix = buildMmluSharedPrefix(shotItems);
    onLog?.({ text: `[MMLU/${mode}] subject=${subject} shared prefix chars=${sharedPrefix.length}` });
    let sharedNodeId = 0;

    const runQuestionFromNode = async (
      parentNodeId: number,
      questionPrompt: string,
      startedAt: number
    ): Promise<{
      predIndex: number;
      answerText: string;
      nativeOutput: string;
      nativeFinalOutput: string;
      parseSource: 'prompt_final_answer';
      latencyMs: number;
      ttftMs: number;
      tokensPerSecond: number;
    }> => {
      let firstTokenAt = 0;
      const judged = await withBenchTimeout(runtime.chatFromNode(parentNodeId, questionPrompt, {
        nPredict: 40,
        abortSignal: signal,
        sampling: {
          temp: 0.2,
          top_p: 0.9,
          top_k: 40,
        },
        onChunk: () => {
          if (!firstTokenAt) {
            firstTokenAt = performance.now();
          }
        },
      }), `MMLU/${mode}/question/chatFromNode`);
      const nativeOutput = judged.assistantText || '';
      const nativeFinalOutput = nativeOutput;
      const bestIdx = parseChoiceIndex(nativeOutput);
      const parseSource: 'prompt_final_answer' = 'prompt_final_answer';
      const answerText = nativeFinalOutput;

      const latency = Math.max(0, performance.now() - startedAt);
      const ttft = Math.max(0, (firstTokenAt || performance.now()) - startedAt);
      const outTokenCount = Math.max(1, (await runtime.tokenize(answerText, true)).length);
      const tps = (outTokenCount * 1000) / Math.max(1e-6, latency);
      return {
        predIndex: bestIdx,
        answerText,
        nativeOutput,
        nativeFinalOutput,
        parseSource,
        latencyMs: latency,
        ttftMs: ttft,
        tokensPerSecond: tps,
      };
    };

    if (mode === 'tree') {
      await trackCacheOp(() => withBenchTimeout(runtime.chatSessionInit(
        treeMemoryCapBytes,
        tieredCacheOpts
      ), `MMLU/${mode}/subject=${subject}/chatSessionInit`));
      await probeTreeState(runtime, `MMLU/${mode}/subject=${subject}/afterChatSessionInit`, onLog);
      const state = await trackCacheOp(() => withBenchTimeout(
        runtime.chatGetState(),
        `MMLU/${mode}/subject=${subject}/chatGetState`
      ));
      const rootNodeId = state.rootId;
      onLog?.({ text: `[MMLU/${mode}] subject=${subject} chat session ready root=${rootNodeId}` });

      const setupPrompt = sharedPrefix;
      const setup = await withBenchTimeout(runtime.chatFromNode(rootNodeId, setupPrompt, {
        nPredict: 0,
        abortSignal: signal,
      }), `MMLU/${mode}/subject=${subject}/sharedPrefixChatFromNode`);
      sharedNodeId = setup.nodeId;
      onLog?.({ text: `[MMLU/${mode}] subject=${subject} shared node prepared node=${sharedNodeId}` });
    }

    const maybeRotateSubjectSession = async () => {
      if (mode !== 'tree') {
        return;
      }
      const state = await trackCacheOp(() => withBenchTimeout(
        runtime.chatGetState(),
        `MMLU/${mode}/subject=${subject}/seqBudget/chatGetState`
      ));
      const nodeCount = state.nodes?.size ?? 0;
      if (nodeCount < TREE_SEQ_REBUILD_WATERMARK) {
        return;
      }

      onLog?.({
        text: `[MMLU/${mode}] subject=${subject} nodeCount=${nodeCount} approaching seq limit=${TREE_SEQ_HARD_LIMIT}; proactively rotating session`,
      });
      await trackCacheOp(() => withBenchTimeout(runtime.chatSessionInit(
        treeMemoryCapBytes,
        tieredCacheOpts
      ), `MMLU/${mode}/subject=${subject}/seqBudget/chatSessionInit`));
      const refreshed = await trackCacheOp(() => withBenchTimeout(
        runtime.chatGetState(),
        `MMLU/${mode}/subject=${subject}/seqBudget/refreshedChatGetState`
      ));
      const setup = await withBenchTimeout(runtime.chatFromNode(refreshed.rootId, sharedPrefix, {
        nPredict: 0,
        abortSignal: signal,
      }), `MMLU/${mode}/subject=${subject}/seqBudget/sharedPrefixChatFromNode`);
      sharedNodeId = setup.nodeId;
      onLog?.({ text: `[MMLU/${mode}] subject=${subject} shared node rebuilt for seq budget node=${sharedNodeId}` });
    };

    for (let i = 0; i < evalItems.length; i += 1) {
      assertNotAborted(signal);
      await maybeRotateSubjectSession();
      const q = evalItems[i];
      const questionPrompt = buildMmluQuestionOnlyPrompt(q);
      let solved = false;
      for (let attempt = 0; attempt < 2 && !solved; attempt += 1) {
        try {
          const t0 = performance.now();
          let out: {
            predIndex: number;
            answerText: string;
            nativeOutput: string;
            nativeFinalOutput: string;
            parseSource: 'prompt_final_answer' | 'repair_final_answer' | 'logits_fallback';
            latencyMs: number;
            ttftMs: number;
            tokensPerSecond: number;
          };

          if (mode === 'flat') {
            // Flat baseline: from zero, run two rounds per item.
            await trackCacheOp(() => withBenchTimeout(runtime.chatSessionInit(
              treeMemoryCapBytes,
              tieredCacheOpts
            ), `MMLU/${mode}/${q.id}/chatSessionInit`));
            await probeTreeState(runtime, `MMLU/${mode}/${q.id}/afterChatSessionInit`, onLog);
            const state = await trackCacheOp(() => withBenchTimeout(
              runtime.chatGetState(),
              `MMLU/${mode}/${q.id}/chatGetState`
            ));
            const setup = await withBenchTimeout(runtime.chatFromNode(state.rootId, sharedPrefix, {
              nPredict: 0,
              abortSignal: signal,
            }), `MMLU/${mode}/${q.id}/sharedPrefixChatFromNode`);
            out = await runQuestionFromNode(setup.nodeId, questionPrompt, t0);
          } else {
            // Tree mode: shared prefix is computed once per subject; each item branches from that node.
            out = await runQuestionFromNode(sharedNodeId, questionPrompt, t0);
          }

          latencyMs.push(out.latencyMs);
          ttftMs.push(out.ttftMs);
          tokensPerSecond.push(out.tokensPerSecond);

          const bestIdx = out.predIndex;
          onLog?.({
            text: `[MMLU/${mode}] ${q.id} pred=${bestIdx >= 0 ? CHOICE_LABELS[bestIdx] : '?'} gt=${CHOICE_LABELS[q.answerIndex]} parse=${out.parseSource} raw="${safeText(out.answerText)}"`,
          });

          rows.push({
            id: q.id,
            gtIndex: q.answerIndex,
            predIndexFlat: mode === 'flat' ? bestIdx : -1,
            predIndexTree: mode === 'tree' ? bestIdx : -1,
            correctFlat: mode === 'flat' ? bestIdx === q.answerIndex : false,
            correctTree: mode === 'tree' ? bestIdx === q.answerIndex : false,
            explainFlat: mode === 'flat' ? `pred=${bestIdx >= 0 ? CHOICE_LABELS[bestIdx] : '?'}` : '',
            explainTree: mode === 'tree' ? `pred=${bestIdx >= 0 ? CHOICE_LABELS[bestIdx] : '?'}` : '',
          });
          onQuestionDone?.();
          solved = true;
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          const kind = classifyBenchFailure(err);
          onQuestionFailure?.(kind, `MMLU/${mode}/${subject}/${q.id}: ${msg}`);
          if (kind === 'timeout' && timeoutDiagCount < 3) {
            timeoutDiagCount += 1;
            const phase = extractBenchTimeoutPhase(msg);
            onLog?.({ text: `[TimeoutDiag/MMLU/${mode}] q=${q.id} subject=${subject} attempt=${attempt + 1} phase=${phase} promptChars=${questionPrompt.length} sharedNode=${sharedNodeId}` });
            await probeTreeState(runtime, `MMLU/${mode}/timeout/${q.id}/attempt=${attempt + 1}`, onLog);
            await logRuntimeDebugSnapshot(runtime, `MMLU/${mode}/timeout/${q.id}/attempt=${attempt + 1}`, onLog);
          }
          onLog?.({ text: `[MMLU/${mode}] ${q.id} attempt=${attempt + 1} failed: ${msg}` });
          if (!recoverWllama) {
            throw err;
          }

          const retryable = isBenchTimeoutError(err) || isRuntimeDisposedError(err);
          if (retryable && attempt === 0) {
            onLog?.({ text: `[MMLU/${mode}] ${q.id} recovering runtime after timeout/disposed...` });
            runtime = await recoverWllama(runtime, `MMLU/${mode} ${q.id} timeout/disposed`);
            if (mode === 'tree') {
              await trackCacheOp(() => withBenchTimeout(runtime.chatSessionInit(
                treeMemoryCapBytes,
                tieredCacheOpts
              ), `MMLU/${mode}/${q.id}/recover/chatSessionInit`));
              await probeTreeState(runtime, `MMLU/${mode}/${q.id}/recover/afterChatSessionInit`, onLog);
              const state = await trackCacheOp(() => withBenchTimeout(
                runtime.chatGetState(),
                `MMLU/${mode}/${q.id}/recover/chatGetState`
              ));
              const rootNodeId = state.rootId;
              const setup = await withBenchTimeout(runtime.chatFromNode(rootNodeId, sharedPrefix, {
                nPredict: 0,
                abortSignal: signal,
              }), `MMLU/${mode}/${q.id}/recover/sharedPrefixChatFromNode`);
              sharedNodeId = setup.nodeId;
              onLog?.({ text: `[MMLU/${mode}] ${q.id} shared node rebuilt after recovery node=${sharedNodeId}` });
            }
            onLog?.({ text: `[MMLU/${mode}] ${q.id} retrying on recreated runtime` });
            continue;
          }

          if (!retryable) {
            runtime = await recoverWllama(runtime, `MMLU/${mode} ${q.id} failed`);
            if (mode === 'tree') {
              await trackCacheOp(() => withBenchTimeout(runtime.chatSessionInit(
                treeMemoryCapBytes,
                tieredCacheOpts
              ), `MMLU/${mode}/${q.id}/recover-nonretry/chatSessionInit`));
              await probeTreeState(runtime, `MMLU/${mode}/${q.id}/recover-nonretry/afterChatSessionInit`, onLog);
              const state = await trackCacheOp(() => withBenchTimeout(
                runtime.chatGetState(),
                `MMLU/${mode}/${q.id}/recover-nonretry/chatGetState`
              ));
              const rootNodeId = state.rootId;
              const setup = await withBenchTimeout(runtime.chatFromNode(rootNodeId, sharedPrefix, {
                nPredict: 0,
                abortSignal: signal,
              }), `MMLU/${mode}/${q.id}/recover-nonretry/sharedPrefixChatFromNode`);
              sharedNodeId = setup.nodeId;
              onLog?.({ text: `[MMLU/${mode}] ${q.id} shared node rebuilt after nonretry recovery node=${sharedNodeId}` });
            }
          }

          rows.push({
            id: q.id,
            gtIndex: q.answerIndex,
            predIndexFlat: -1,
            predIndexTree: -1,
            correctFlat: false,
            correctTree: false,
            explainFlat: mode === 'flat' ? 'skipped: runtime error' : '',
            explainTree: mode === 'tree' ? 'skipped: runtime error' : '',
          });
          onQuestionDone?.();
          solved = true;
        }
      }
    }
  }

  return { results: rows, latencyMs, ttftMs, tokensPerSecond, wllama: runtime };
}

async function runHella(
  wllama: Wllama,
  data: HellaSwagItem[],
  shots: number,
  evalCount: number,
  mode: 'flat' | 'tree',
  config: BenchConfig,
  slotBase: number,
  onLog?: (e: BenchLogEvent) => void,
  signal?: AbortSignal,
  recoverWllama?: (oldRuntime: Wllama, reason: string) => Promise<Wllama>,
  onQuestionDone?: () => void,
  onQuestionFailure?: (kind: 'timeout' | 'abort' | 'disposed' | 'other', details: string) => void,
  onCacheMaintenanceMs?: (ms: number) => void
): Promise<{ results: QAResult[]; latencyMs: number[]; ttftMs: number[]; tokensPerSecond: number[]; wllama: Wllama }> {
  let runtime = wllama;
  const treeMemoryCapBytes = getTreeMemoryCapBytes(config.trueTreeMemoryCapMB, onLog, `Hella/${mode}`);
  const tieredCacheOpts = buildTieredCacheOptions(config, onLog);
  onLog?.({
    text: `[Exp3Cfg/Hella/${mode}] chatSessionInit capBytes=${treeMemoryCapBytes} tierEnabled=${tieredCacheOpts.enabled} tierCaps(L1/L2/L3)=${tieredCacheOpts.l1TokenCap ?? 0}/${tieredCacheOpts.l2TokenCap ?? 0}/${tieredCacheOpts.l3TokenCap ?? 0} thresholds(L1L2/L2L3)=${tieredCacheOpts.pruneL1L2TokenThreshold ?? 0}/${tieredCacheOpts.pruneL2L3TokenThreshold ?? 0} replacementPolicy=${tieredCacheOpts.replacementPolicy ?? 'hybrid'} fallbackApplied=${tieredCacheOpts.fallbackApplied}`,
  });
  void slotBase;
  const trackCacheOp = async <T>(fn: () => Promise<T>): Promise<T> => {
    const t0 = performance.now();
    const out = await fn();
    onCacheMaintenanceMs?.(Math.max(0, performance.now() - t0));
    return out;
  };
  const shotItems = data.slice(0, shots);
  const evalItems = data.slice(shots, shots + evalCount);
  const sharedPrefix = buildHellaSharedPrefix(shotItems);
  let rootNodeId = 0;
  let sharedNodeId = 0;
  if (mode === 'tree') {
    await trackCacheOp(() => withBenchTimeout(runtime.chatSessionInit(
      treeMemoryCapBytes,
      tieredCacheOpts
    ), `Hella/${mode}/chatSessionInit`));
    await probeTreeState(runtime, `Hella/${mode}/afterChatSessionInit`, onLog);
    const state = await trackCacheOp(() => withBenchTimeout(
      runtime.chatGetState(),
      `Hella/${mode}/chatGetState`
    ));
    rootNodeId = state.rootId;
    onLog?.({ text: `[Hella/${mode}] chat session ready root=${rootNodeId}` });
    const setup = await withBenchTimeout(runtime.chatFromNode(rootNodeId, sharedPrefix, {
      nPredict: 0,
      abortSignal: signal,
    }), `Hella/${mode}/sharedPrefixChatFromNode`);
    sharedNodeId = setup.nodeId;
    onLog?.({ text: `[Hella/${mode}] shared node prepared node=${sharedNodeId}` });
  }

  const rows: QAResult[] = [];
  const latencyMs: number[] = [];
  const ttftMs: number[] = [];
  const tokensPerSecond: number[] = [];
  let timeoutDiagCount = 0;

  for (let i = 0; i < evalItems.length; i += 1) {
    assertNotAborted(signal);
    const q = evalItems[i];
    const questionPrompt = buildHellaQuestionOnlyPrompt(q);
    const t0 = performance.now();
    let solved = false;
    for (let attempt = 0; attempt < 2 && !solved; attempt += 1) {
      try {
        let out: { predIndex: number; answerText: string; latencyMs: number; ttftMs: number; tokensPerSecond: number };
        if (mode === 'flat') {
          await trackCacheOp(() => withBenchTimeout(runtime.chatSessionInit(
            treeMemoryCapBytes,
            tieredCacheOpts
          ), `Hella/${mode}/${q.id}/chatSessionInit`));
          const state = await trackCacheOp(() => withBenchTimeout(
            runtime.chatGetState(),
            `Hella/${mode}/${q.id}/chatGetState`
          ));
          const setup = await withBenchTimeout(runtime.chatFromNode(state.rootId, sharedPrefix, {
            nPredict: 0,
            abortSignal: signal,
          }), `Hella/${mode}/${q.id}/sharedPrefixChatFromNode`);
          out = await runChoicePrompt(runtime, mode, questionPrompt, signal, setup.nodeId, t0);
        } else {
          out = await runChoicePrompt(runtime, mode, questionPrompt, signal, sharedNodeId, t0);
        }
        latencyMs.push(out.latencyMs);
        ttftMs.push(out.ttftMs);
        tokensPerSecond.push(out.tokensPerSecond);

        const bestIdx = out.predIndex;
        onLog?.({
          text: `[Hella/${mode}] ${q.id} pred=${bestIdx >= 0 ? CHOICE_LABELS[bestIdx] : '?'} gt=${CHOICE_LABELS[q.label]} raw="${safeText(out.answerText)}"`,
        });

        rows.push({
          id: q.id,
          gtIndex: q.label,
          predIndexFlat: mode === 'flat' ? bestIdx : -1,
          predIndexTree: mode === 'tree' ? bestIdx : -1,
          correctFlat: mode === 'flat' ? bestIdx === q.label : false,
          correctTree: mode === 'tree' ? bestIdx === q.label : false,
          explainFlat: mode === 'flat' ? `pred=${bestIdx >= 0 ? CHOICE_LABELS[bestIdx] : '?'}` : '',
          explainTree: mode === 'tree' ? `pred=${bestIdx >= 0 ? CHOICE_LABELS[bestIdx] : '?'}` : '',
        });
        onQuestionDone?.();
        solved = true;
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        const kind = classifyBenchFailure(err);
        onQuestionFailure?.(kind, `Hella/${mode}/${q.id}: ${msg}`);
        if (kind === 'timeout' && timeoutDiagCount < 3) {
          timeoutDiagCount += 1;
          const phase = extractBenchTimeoutPhase(msg);
          onLog?.({ text: `[TimeoutDiag/Hella/${mode}] q=${q.id} attempt=${attempt + 1} phase=${phase} promptChars=${questionPrompt.length} sharedNode=${sharedNodeId}` });
          await probeTreeState(runtime, `Hella/${mode}/timeout/${q.id}/attempt=${attempt + 1}`, onLog);
        }
        onLog?.({ text: `[Hella/${mode}] ${q.id} attempt=${attempt + 1} failed: ${msg}` });
        if (!recoverWllama) {
          throw err;
        }

        const retryable = isBenchTimeoutError(err) || isRuntimeDisposedError(err);
        if (retryable && attempt === 0) {
          onLog?.({ text: `[Hella/${mode}] ${q.id} recovering runtime after timeout/disposed...` });
          runtime = await recoverWllama(runtime, `Hella/${mode} ${q.id} timeout/disposed`);
          if (mode === 'tree') {
            await trackCacheOp(() => withBenchTimeout(runtime.chatSessionInit(
              treeMemoryCapBytes,
              tieredCacheOpts
            ), `Hella/${mode}/${q.id}/recover/chatSessionInit`));
            await probeTreeState(runtime, `Hella/${mode}/${q.id}/recover/afterChatSessionInit`, onLog);
            const state = await trackCacheOp(() => withBenchTimeout(
              runtime.chatGetState(),
              `Hella/${mode}/${q.id}/recover/chatGetState`
            ));
            rootNodeId = state.rootId;
            const setup = await withBenchTimeout(runtime.chatFromNode(rootNodeId, sharedPrefix, {
              nPredict: 0,
              abortSignal: signal,
            }), `Hella/${mode}/${q.id}/recover/sharedPrefixChatFromNode`);
            sharedNodeId = setup.nodeId;
          }
          onLog?.({ text: `[Hella/${mode}] ${q.id} retrying on recreated runtime` });
          continue;
        }

        if (!retryable) {
          runtime = await recoverWllama(runtime, `Hella/${mode} ${q.id} failed`);
          if (mode === 'tree') {
            await trackCacheOp(() => withBenchTimeout(runtime.chatSessionInit(
              treeMemoryCapBytes,
              tieredCacheOpts
            ), `Hella/${mode}/${q.id}/recover-nonretry/chatSessionInit`));
            await probeTreeState(runtime, `Hella/${mode}/${q.id}/recover-nonretry/afterChatSessionInit`, onLog);
            const state = await trackCacheOp(() => withBenchTimeout(
              runtime.chatGetState(),
              `Hella/${mode}/${q.id}/recover-nonretry/chatGetState`
            ));
            rootNodeId = state.rootId;
            const setup = await withBenchTimeout(runtime.chatFromNode(rootNodeId, sharedPrefix, {
              nPredict: 0,
              abortSignal: signal,
            }), `Hella/${mode}/${q.id}/recover-nonretry/sharedPrefixChatFromNode`);
            sharedNodeId = setup.nodeId;
          }
        }

        rows.push({
          id: q.id,
          gtIndex: q.label,
          predIndexFlat: -1,
          predIndexTree: -1,
          correctFlat: false,
          correctTree: false,
          explainFlat: mode === 'flat' ? 'skipped: runtime error' : '',
          explainTree: mode === 'tree' ? 'skipped: runtime error' : '',
        });
        onQuestionDone?.();
        solved = true;
      }
    }
  }

  return { results: rows, latencyMs, ttftMs, tokensPerSecond, wllama: runtime };
}

function mergeRows(flatRows: QAResult[], treeRows: QAResult[]): QAResult[] {
  const treeMap = new Map(treeRows.map((r) => [r.id, r]));
  return flatRows.map((f) => {
    const t = treeMap.get(f.id);
    if (!t) return f;
    return {
      id: f.id,
      gtIndex: f.gtIndex,
      predIndexFlat: f.predIndexFlat,
      predIndexTree: t.predIndexTree,
      correctFlat: f.correctFlat,
      correctTree: t.correctTree,
      explainFlat: f.explainFlat,
      explainTree: t.explainTree,
    };
  });
}

function summarize(
  benchmark: 'MMLU' | 'HellaSwag',
  shots: number,
  evalCount: number,
  rows: QAResult[],
  latencyFlat: number[],
  latencyTree: number[],
  ttftFlat: number[],
  ttftTree: number[],
  tpsFlat: number[],
  tpsTree: number[]
): BenchSummary {
  const accFlat = rows.filter((r) => r.correctFlat).length / Math.max(rows.length, 1);
  const accTree = rows.filter((r) => r.correctTree).length / Math.max(rows.length, 1);
  const avgFlat = avg(latencyFlat);
  const avgTree = avg(latencyTree);
  const avgTtftFlat = avg(ttftFlat);
  const avgTtftTree = avg(ttftTree);
  const avgTpsFlat = avg(tpsFlat);
  const avgTpsTree = avg(tpsTree);
  const speedupPct = avgFlat > 0 ? ((avgFlat - avgTree) / avgFlat) * 100 : 0;
  const ttftSpeedupPct = avgTtftFlat > 0 ? ((avgTtftFlat - avgTtftTree) / avgTtftFlat) * 100 : 0;
  const tpsGainPct = avgTpsFlat > 0 ? ((avgTpsTree - avgTpsFlat) / avgTpsFlat) * 100 : 0;
  return {
    benchmark,
    shots,
    evalCount,
    accFlat,
    accTree,
    avgTtftMsFlat: avgTtftFlat,
    avgTtftMsTree: avgTtftTree,
    ttftSpeedupPct,
    avgTokensPerSecondFlat: avgTpsFlat,
    avgTokensPerSecondTree: avgTpsTree,
    tpsGainPct,
    avgLatencyMsFlat: avgFlat,
    avgLatencyMsTree: avgTree,
    speedupPct,
    results: rows,
  };
}

function emptySummary(benchmark: 'MMLU' | 'HellaSwag', shots: number, evalCount: number): BenchSummary {
  return {
    benchmark,
    shots,
    evalCount,
    accFlat: 0,
    accTree: 0,
    avgTtftMsFlat: 0,
    avgTtftMsTree: 0,
    ttftSpeedupPct: 0,
    avgTokensPerSecondFlat: 0,
    avgTokensPerSecondTree: 0,
    tpsGainPct: 0,
    avgLatencyMsFlat: 0,
    avgLatencyMsTree: 0,
    speedupPct: 0,
    results: [],
  };
}

function buildCacheProfile(
  maintenanceMs: number,
  runStartedAt: number,
  state?: {
    totalSnapshotTokenBytes?: number;
    tierStats?: {
      l1Tokens?: number;
      l2Tokens?: number;
      l3Tokens?: number;
    };
  } | null
): CacheProfile {
  const runTotalMs = Math.max(0, performance.now() - runStartedAt);
  const tier = state?.tierStats;
  return {
    maintenanceMs,
    runTotalMs,
    maintenancePct: runTotalMs > 0 ? (maintenanceMs / runTotalMs) * 100 : 0,
    snapshotTokenBytes: state?.totalSnapshotTokenBytes ?? 0,
    tierL1Tokens: tier?.l1Tokens ?? 0,
    tierL2Tokens: tier?.l2Tokens ?? 0,
    tierL3Tokens: tier?.l3Tokens ?? 0,
  };
}

async function tryGetTreeState(runtime: Wllama): Promise<{
  totalSnapshotTokenBytes: number;
  tierStats: {
    l1Tokens: number;
    l2Tokens: number;
    l3Tokens: number;
    promotions: number;
    demotions: number;
    diskReads: number;
    diskWrites: number;
    l3OverflowEvents: number;
    restoreAttempts: number;
    restoreHitsL1: number;
    restoreHitsL2: number;
    restoreHitsL3: number;
    restoreMisses: number;
    restoreRebuilds: number;
    parentRecoverAttempts: number;
    parentRecoverSuccesses: number;
    parentRecoverFailures: number;
  };
} | null> {
  try {
    const state = await (runtime as unknown as {
      chatGetState: () => Promise<{
        totalSnapshotTokenBytes: number;
        tierStats: {
          l1Tokens: number;
          l2Tokens: number;
          l3Tokens: number;
          promotions: number;
          demotions: number;
          diskReads: number;
          diskWrites: number;
          l3OverflowEvents: number;
          restoreAttempts: number;
          restoreHitsL1: number;
          restoreHitsL2: number;
          restoreHitsL3: number;
          restoreMisses: number;
          restoreRebuilds: number;
          parentRecoverAttempts: number;
          parentRecoverSuccesses: number;
          parentRecoverFailures: number;
        };
      }>;
    }).chatGetState();
    return state;
  } catch {
    return null;
  }
}

type RequestPerf = {
  ttftMs: number;
  latencyMs: number;
  tokenCount: number;
  tokensPerSecond: number;
};

function summarizePerf(rows: RequestPerf[], failed: number): {
  avgTtftMs: number;
  avgLatencyMs: number;
  avgTokensPerSecond: number;
  totalTokens: number;
  failed: number;
} {
  return {
    avgTtftMs: avg(rows.map((x) => x.ttftMs)),
    avgLatencyMs: avg(rows.map((x) => x.latencyMs)),
    avgTokensPerSecond: avg(rows.map((x) => x.tokensPerSecond)),
    totalTokens: rows.reduce((sum, x) => sum + Math.max(0, x.tokenCount), 0),
    failed,
  };
}

async function runQueueVsDirectMmlu(
  wllama: Wllama,
  mmluData: MMLUItem[],
  config: BenchConfig,
  signal?: AbortSignal,
  onLog?: (e: BenchLogEvent) => void,
  recoverWllama?: (oldRuntime: Wllama, reason: string) => Promise<Wllama>
): Promise<{ summary: QueueVsDirectSummary; wllama: Wllama }> {
  let runtime = wllama;
  let recoveryChain: Promise<void> = Promise.resolve();

  const shouldTraceRequest = (idx: number, total: number): boolean => {
    if (idx < 3) return true;
    if (idx === total - 1) return true;
    return idx % 16 === 0;
  };

  const recoverRuntime = async (badRuntime: Wllama, reason: string): Promise<void> => {
    if (!recoverWllama) {
      throw new Error(`[Exp4] ${reason}: recover callback not provided`);
    }

    const runRecovery = async () => {
      if (runtime !== badRuntime) {
        onLog?.({ text: `[Exp4] ${reason}: runtime already replaced by another recovery` });
        return;
      }
      onLog?.({ text: `[Exp4] ${reason}: force replace runtime` });
      runtime = await recoverWllama(badRuntime, reason);
      onLog?.({ text: `[Exp4] ${reason}: runtime recreated` });
    };

    const next = recoveryChain.then(runRecovery, runRecovery);
    recoveryChain = next.catch(() => {});
    await next;
  };

  const timeoutError = (phase: string, requestId: number) =>
    `[Exp4Timeout] ${phase} request=${requestId} timed out after ${EXP4_REQUEST_TIMEOUT_MS}ms`;

  const evalQueue = mmluData.slice(config.mmluShots);
  const sampleCount = Math.max(1, Number(config.exp4SampleCount) || 1);
  const evalItems = shuffleWithSeed(evalQueue, config.randomSeed + 404).slice(0, sampleCount);
  onLog?.({
    text: `[Exp4] sampling availableEval=${evalQueue.length} requested=${sampleCount} selected=${evalItems.length} seed=${config.randomSeed + 404}`,
  });
  if (evalItems.length > 0) {
    const samplePreview = evalItems.slice(0, 5).map((x) => x.id).join(', ');
    onLog?.({ text: `[Exp4] samplePreview(first<=5)=${samplePreview}` });
  }
  const prompts = evalItems.map((q) => [
    'Answer with exactly one letter: A, B, C, or D.',
    mmluQuestionBlock(q, false),
  ].join('\n\n'));

  const runBatch = async (useQueue: boolean): Promise<{ rows: RequestPerf[]; failed: number; wallClockMs: number }> => {
    const rows: RequestPerf[] = [];
    let failed = 0;
    const mode = useQueue ? 'queue' : 'direct';
    const batchStartedAt = performance.now();
    onLog?.({ text: `[Exp4/${mode}] batchStart requestCount=${prompts.length}` });

    const runOne = async (prompt: string, i: number, arrivedAtMs: number): Promise<void> => {
      for (let attempt = 0; attempt < 2; attempt += 1) {
        const activeRuntime = runtime;
        const serviceStartedAt = performance.now();
        const t0 = arrivedAtMs;
        let firstTokenAt = 0;
        const traced = shouldTraceRequest(i, prompts.length);
        const queueWaitMs = Math.max(0, serviceStartedAt - arrivedAtMs);
        if (traced) {
          onLog?.({ text: `[Exp4/${mode}] request=${i} attempt=${attempt + 1} start queueWaitMs=${queueWaitMs.toFixed(2)}` });
        }
        try {
          const markFirstToken = () => {
            if (!firstTokenAt) {
              firstTokenAt = performance.now();
              if (traced) {
                onLog?.({ text: `[Exp4/${mode}] request=${i} attempt=${attempt + 1} firstTokenCaptured` });
              }
            }
          };

          const commonOptions = {
            nPredict: Math.max(1, config.exp4OutputTokens),
            abortSignal: signal,
          };

          const treeChatOptions = {
            ...commonOptions,
            // chatFromNode surfaces streaming via onChunk.
            onChunk: () => {
              markFirstToken();
            },
          };

          const text = (
            await withTimeout(
              (activeRuntime as unknown as {
                chatFromNode: (parentId: number, userText: string, options: unknown) => Promise<{ assistantText: string }>;
              }).chatFromNode(0, prompt, treeChatOptions),
              EXP4_REQUEST_TIMEOUT_MS,
              timeoutError(useQueue ? 'queue' : 'direct-tree', i)
            )
          ).assistantText;

          const latencyMs = Math.max(0, performance.now() - t0);
          const ttftMs = Math.max(0, (firstTokenAt || performance.now()) - t0);
          const tokCount = Math.max(1, (await activeRuntime.tokenize(text, true)).length);
          const tokensPerSecond = (tokCount * 1000) / Math.max(1e-6, latencyMs);
          if (!firstTokenAt && traced) {
            onLog?.({ text: `[Exp4/${mode}] request=${i} attempt=${attempt + 1} firstTokenMissingFallback=true` });
          }
          rows.push({ ttftMs, latencyMs, tokenCount: tokCount, tokensPerSecond });
          if (traced) {
            onLog?.({
              text: `[Exp4/${mode}] request=${i} attempt=${attempt + 1} done ttftMs=${ttftMs.toFixed(2)} latencyMs=${latencyMs.toFixed(2)} toks=${tokCount} tps=${tokensPerSecond.toFixed(3)}`,
            });
          }
          return;
        } catch (err) {
          const retryable = isExp4TimeoutError(err) || isRuntimeDisposedError(err);
          const canRetry = retryable && attempt === 0;
          const msg = err instanceof Error ? err.message : String(err);
          onLog?.({ text: `[Exp4/${useQueue ? 'queue' : 'direct'}] request=${i} attempt=${attempt + 1} failed: ${msg}` });

          if (canRetry) {
            const reason = isExp4TimeoutError(err)
              ? `Exp4/${useQueue ? 'queue' : 'direct'} request=${i} timeout`
              : `Exp4/${useQueue ? 'queue' : 'direct'} request=${i} runtime-disposed`;
            await recoverRuntime(activeRuntime, reason);
            onLog?.({ text: `[Exp4/${useQueue ? 'queue' : 'direct'}] request=${i} retrying on fresh runtime` });
            continue;
          }

          failed += 1;
          return;
        }
      }
    };

    if (useQueue) {
      onLog?.({
        text: `[Exp4/queue] scheduler=runtime-rfg arrival=simultaneous-at-batchStart submit=all-at-once requestCount=${prompts.length}`,
      });
      await Promise.all(prompts.map((prompt, i) => runOne(prompt, i, batchStartedAt)));
    } else {
      onLog?.({ text: '[Exp4/direct] scheduler=fcfs arrival=simultaneous-at-batchStart dispatch=sequential path=chatFromNode' });
      for (let i = 0; i < prompts.length; i += 1) {
        await runOne(prompts[i], i, batchStartedAt);
      }
    }

    const wallClockMs = Math.max(0, performance.now() - batchStartedAt);
    onLog?.({ text: `[Exp4/${mode}] batchDone success=${rows.length} failed=${failed} wallClockMs=${wallClockMs.toFixed(2)}` });

    return { rows, failed, wallClockMs };
  };

  const queueBatch = await runBatch(true);
  try {
    await withTimeout(
      (runtime as unknown as { chatReset: () => Promise<unknown> }).chatReset(),
      EXP4_REQUEST_TIMEOUT_MS,
      `[Exp4Timeout] chatReset timed out after ${EXP4_REQUEST_TIMEOUT_MS}ms`
    );
    onLog?.({ text: '[Exp4] chatReset done before direct batch' });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    onLog?.({ text: `[Exp4] chatReset failed: ${msg}` });
    if (isExp4TimeoutError(err) || isRuntimeDisposedError(err)) {
      await recoverRuntime(runtime, 'Exp4 chatReset failed');
    } else {
      throw err;
    }
  }
  const directBatch = await runBatch(false);

  const queueSummary = summarizePerf(queueBatch.rows, queueBatch.failed);
  const directSummary = summarizePerf(directBatch.rows, directBatch.failed);
  const batchTokensPerSecondQueue =
    (queueSummary.totalTokens * 1000) / Math.max(1e-6, queueBatch.wallClockMs);
  const batchTokensPerSecondDirect =
    (directSummary.totalTokens * 1000) / Math.max(1e-6, directBatch.wallClockMs);
  return {
    wllama: runtime,
    summary: {
    requestCount: prompts.length,
    failedCountQueue: queueSummary.failed,
    failedCountDirect: directSummary.failed,
    avgTtftMsQueue: queueSummary.avgTtftMs,
    avgTtftMsDirect: directSummary.avgTtftMs,
    avgLatencyMsQueue: queueSummary.avgLatencyMs,
    avgLatencyMsDirect: directSummary.avgLatencyMs,
    avgTokensPerSecondQueue: queueSummary.avgTokensPerSecond,
    avgTokensPerSecondDirect: directSummary.avgTokensPerSecond,
    batchWallClockMsQueue: queueBatch.wallClockMs,
    batchWallClockMsDirect: directBatch.wallClockMs,
    batchTokensPerSecondQueue,
    batchTokensPerSecondDirect,
    },
  };
}

function summarizeSinglePath(
  benchmark: 'MMLU' | 'HellaSwag',
  shots: number,
  evalCount: number,
  rows: QAResult[],
  latency: number[],
  ttft: number[],
  tps: number[]
): BenchSummary {
  const acc = rows.filter((r) => r.correctTree).length / Math.max(1, rows.length);
  const avgLatency = avg(latency);
  const avgTtft = avg(ttft);
  const avgTps = avg(tps);
  return {
    benchmark,
    shots,
    evalCount,
    accFlat: acc,
    accTree: acc,
    avgTtftMsFlat: avgTtft,
    avgTtftMsTree: avgTtft,
    ttftSpeedupPct: 0,
    avgTokensPerSecondFlat: avgTps,
    avgTokensPerSecondTree: avgTps,
    tpsGainPct: 0,
    avgLatencyMsFlat: avgLatency,
    avgLatencyMsTree: avgLatency,
    speedupPct: 0,
    results: rows,
  };
}

async function runWebllmNoCacheChoice(
  engine: any,
  prompt: string,
  options: {
    maxTokens: number;
    temperature: number;
    topP: number;
    hardConstraintABCD?: boolean;
  }
): Promise<{ predIndex: number; answerText: string; latencyMs: number; ttftMs: number; tokensPerSecond: number }> {
  const t0 = performance.now();
  let firstTokenAt = 0;
  let answerText = '';
  let completionTokens = 0;

  // Ensure strict no-cache semantics per sample.
  await engine.resetChat();

  const request: Record<string, unknown> = {
    messages: [{ role: 'user', content: prompt }],
    temperature: options.temperature,
    top_p: options.topP,
    max_tokens: options.maxTokens,
    stream: true,
    stream_options: { include_usage: true },
  };
  if (options.hardConstraintABCD) {
    request.response_format = {
      type: 'grammar',
      grammar: 'root ::= "A" | "B" | "C" | "D"',
    };
  }

  const stream = await engine.chat.completions.create(request);

  for await (const chunk of stream) {
    const c = chunk as {
      choices?: Array<{
        delta?: { content?: string | null };
      }>;
      usage?: { completion_tokens?: number };
    };
    const delta = c.choices?.[0]?.delta?.content ?? '';
    if (delta) {
      if (!firstTokenAt) {
        firstTokenAt = performance.now();
      }
      answerText += delta;
    }
    if (typeof c.usage?.completion_tokens === 'number') {
      completionTokens = c.usage.completion_tokens;
    }
  }

  const latencyMs = Math.max(0, performance.now() - t0);
  const ttftMs = Math.max(0, (firstTokenAt || performance.now()) - t0);
  if (completionTokens <= 0) {
    completionTokens = Math.max(1, safeText(answerText).length > 0 ? 1 : 0);
  }
  const tokensPerSecond = (completionTokens * 1000) / Math.max(1e-6, latencyMs);

  const predIndex = parseChoiceIndex(answerText);
  return { predIndex, answerText, latencyMs, ttftMs, tokensPerSecond };
}

async function runWebllmNoCacheBench(
  config: BenchConfig,
  mmluData: MMLUItem[],
  hellaData: HellaSwagItem[],
  onLog?: (e: BenchLogEvent) => void,
  onProgress?: (e: BenchProgressEvent) => void,
  signal?: AbortSignal
): Promise<BenchReport> {
  const webllm = await import('@mlc-ai/web-llm');
  const runMmluTarget = config.target === 'mmlu' || config.target === 'both';
  const runHellaTarget = config.target === 'hella' || config.target === 'both';
  let totalQuestions = 0;
  if (runMmluTarget && config.mmluEvalCount > 0) {
    totalQuestions += estimateMmluEvalItemCount(
      mmluData,
      config.mmluShots,
      config.mmluEvalCount,
      'exp1-sequential-once'
    );
  }
  if (runHellaTarget && config.hellaEvalCount > 0) {
    totalQuestions += estimateHellaEvalItemCount(hellaData, config.hellaShots, config.hellaEvalCount);
  }

  let doneQuestions = 0;
  onProgress?.({ current: 0, total: totalQuestions, label: 'Loading web-llm' });
  onLog?.({ text: `[web-llm] loading model=${config.webllmModelId}` });
  const engine = await webllm.CreateMLCEngine(config.webllmModelId, {
    initProgressCallback: (p: { progress?: number; text?: string }) => {
      const progressPct = typeof p?.progress === 'number' ? ` ${(p.progress * 100).toFixed(1)}%` : '';
      onLog?.({ text: `[web-llm/init]${progressPct} ${p?.text ?? ''}`.trim() });
    },
  });
  onLog?.({ text: '[web-llm] model loaded.' });

  const reportQuestionDone = (label: string) => {
    doneQuestions += 1;
    onProgress?.({ current: doneQuestions, total: totalQuestions, label });
  };

  try {
    let mmluSummary: BenchSummary | undefined;
    if (runMmluTarget && config.mmluEvalCount > 0) {
      const grouped = new Map<string, MMLUItem[]>();
      for (const item of mmluData) {
        const key = item.subject || 'unknown';
        const list = grouped.get(key) ?? [];
        list.push(item);
        grouped.set(key, list);
      }
      const rows: QAResult[] = [];
      const latencyMs: number[] = [];
      const ttftMs: number[] = [];
      const tokensPerSecond: number[] = [];

      for (const [subject, items] of grouped.entries()) {
        const shotItems = items.slice(0, config.mmluShots);
        const evalItems = items.slice(config.mmluShots, config.mmluShots + config.mmluEvalCount);
        if (!evalItems.length) continue;

        const sharedPrefix = buildMmluSharedPrefix(shotItems);
        for (const q of evalItems) {
          assertNotAborted(signal);
          const prompt = [sharedPrefix, buildMmluQuestionOnlyPrompt(q)].join('\n\n');
          const out = await runWebllmNoCacheChoice(engine, prompt, {
            maxTokens: 40,
            temperature: 0.2,
            topP: 0.9,
            hardConstraintABCD: false,
          });
          const bestIdx = out.predIndex;
          rows.push({
            id: q.id,
            gtIndex: q.answerIndex,
            predIndexFlat: bestIdx,
            predIndexTree: bestIdx,
            correctFlat: bestIdx === q.answerIndex,
            correctTree: bestIdx === q.answerIndex,
            explainFlat: `pred=${bestIdx >= 0 ? CHOICE_LABELS[bestIdx] : '?'}`,
            explainTree: `pred=${bestIdx >= 0 ? CHOICE_LABELS[bestIdx] : '?'}`,
          });
          latencyMs.push(out.latencyMs);
          ttftMs.push(out.ttftMs);
          tokensPerSecond.push(out.tokensPerSecond);
          onLog?.({
            text: `[MMLU/web-llm] ${q.id} subject=${subject} pred=${bestIdx >= 0 ? CHOICE_LABELS[bestIdx] : '?'} gt=${CHOICE_LABELS[q.answerIndex]} raw="${safeText(out.answerText)}"`,
          });
          reportQuestionDone('MMLU web-llm');
        }
      }
      mmluSummary = summarizeSinglePath('MMLU', config.mmluShots, rows.length, rows, latencyMs, ttftMs, tokensPerSecond);
    }

    let hellaSummary: BenchSummary | undefined;
    if (runHellaTarget && config.hellaEvalCount > 0) {
      const shotItems = hellaData.slice(0, config.hellaShots);
      const evalItems = hellaData.slice(config.hellaShots, config.hellaShots + config.hellaEvalCount);
      const sharedPrefix = buildHellaSharedPrefix(shotItems);
      const rows: QAResult[] = [];
      const latencyMs: number[] = [];
      const ttftMs: number[] = [];
      const tokensPerSecond: number[] = [];

      for (const q of evalItems) {
        assertNotAborted(signal);
        const prompt = [sharedPrefix, buildHellaQuestionOnlyPrompt(q)].join('\n\n');
        const out = await runWebllmNoCacheChoice(engine, prompt, {
          maxTokens: 1,
          temperature: 0,
          topP: 1,
          hardConstraintABCD: true,
        });
        const bestIdx = out.predIndex;
        rows.push({
          id: q.id,
          gtIndex: q.label,
          predIndexFlat: bestIdx,
          predIndexTree: bestIdx,
          correctFlat: bestIdx === q.label,
          correctTree: bestIdx === q.label,
          explainFlat: `pred=${bestIdx >= 0 ? CHOICE_LABELS[bestIdx] : '?'}`,
          explainTree: `pred=${bestIdx >= 0 ? CHOICE_LABELS[bestIdx] : '?'}`,
        });
        latencyMs.push(out.latencyMs);
        ttftMs.push(out.ttftMs);
        tokensPerSecond.push(out.tokensPerSecond);
        onLog?.({
          text: `[Hella/web-llm] ${q.id} pred=${bestIdx >= 0 ? CHOICE_LABELS[bestIdx] : '?'} gt=${CHOICE_LABELS[q.label]} raw="${safeText(out.answerText)}"`,
        });
        reportQuestionDone('Hella web-llm');
      }
      hellaSummary = summarizeSinglePath('HellaSwag', config.hellaShots, rows.length, rows, latencyMs, ttftMs, tokensPerSecond);
    }

    onProgress?.({ current: doneQuestions, total: totalQuestions, label: 'Done' });

    return {
      modelUrl: config.modelUrl,
      config,
      mmlu: mmluSummary,
      hella: hellaSummary,
      cacheProfile: {
        maintenanceMs: 0,
        runTotalMs: 0,
        maintenancePct: 0,
        snapshotTokenBytes: 0,
        tierL1Tokens: 0,
        tierL2Tokens: 0,
        tierL3Tokens: 0,
      },
      diagnostics: {
        runtimeRestartCount: 0,
        timeoutFailureCount: 0,
        abortFailureCount: 0,
        disposedFailureCount: 0,
        otherFailureCount: 0,
      },
    };
  } finally {
    try {
      await engine.unload();
      onLog?.({ text: '[web-llm] unloaded.' });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      onLog?.({ text: `[web-llm] unload failed: ${msg}` });
    }
  }
}

export async function runSglangStyleBench(
  config: BenchConfig,
  mmluData: MMLUItem[],
  hellaData: HellaSwagItem[],
  onLog?: (e: BenchLogEvent) => void,
  onProgress?: (e: BenchProgressEvent) => void,
  signal?: AbortSignal
): Promise<BenchReport> {
  if (config.backend === 'web-llm') {
    return runWebllmNoCacheBench(config, mmluData, hellaData, onLog, onProgress, signal);
  }

  let wllama: Wllama | null = await createLoadedWllama(config, onLog);
  const runStartedAt = performance.now();
  let cacheMaintenanceMs = 0;
  const onCacheMaintenanceMs = (ms: number) => {
    cacheMaintenanceMs += Math.max(0, ms);
  };

  let totalQuestions = 0;
  let doneQuestions = 0;
  const runExp4Only = config.experimentRunMode === 'exp4';
  const activeMmluExperimentMode = config.experimentRunMode === 'exp2'
    ? 'exp2-random-twice'
    : 'exp1-sequential-once';
  const runMmluTarget = config.target === 'mmlu' || config.target === 'both';
  const runHellaTarget = config.target === 'hella' || config.target === 'both';
  if (!runExp4Only && runMmluTarget && config.mmluEvalCount > 0) {
    const mmluEvalTotal = estimateMmluEvalItemCount(
      mmluData,
      config.mmluShots,
      config.mmluEvalCount,
      activeMmluExperimentMode
    );
    totalQuestions += activeMmluExperimentMode === 'exp2-random-twice' ? mmluEvalTotal : mmluEvalTotal * 2;
  }
  if (!runExp4Only && runHellaTarget && config.hellaEvalCount > 0) {
    const hellaEvalTotal = estimateHellaEvalItemCount(hellaData, config.hellaShots, config.hellaEvalCount);
    totalQuestions += hellaEvalTotal * 2;
  }
  onProgress?.({ current: 0, total: totalQuestions, label: 'Starting' });

  const reportQuestionDone = (label: string) => {
    doneQuestions += 1;
    onProgress?.({ current: doneQuestions, total: totalQuestions, label });
  };

  const diagnostics: BenchDiagnostics = {
    runtimeRestartCount: 0,
    timeoutFailureCount: 0,
    abortFailureCount: 0,
    disposedFailureCount: 0,
    otherFailureCount: 0,
    timeoutPhaseCounts: {},
  };

  const reportQuestionFailure = (
    kind: 'timeout' | 'abort' | 'disposed' | 'other',
    details: string
  ) => {
    if (kind === 'timeout') {
      diagnostics.timeoutFailureCount += 1;
      const phase = extractBenchTimeoutPhase(details);
      diagnostics.timeoutPhaseCounts![phase] = (diagnostics.timeoutPhaseCounts![phase] ?? 0) + 1;
      // timeout details are already emitted in capped per-question timeout diagnostics.
    }
    else if (kind === 'abort') diagnostics.abortFailureCount += 1;
    else if (kind === 'disposed') diagnostics.disposedFailureCount += 1;
    else diagnostics.otherFailureCount += 1;
  };

  try {
    assertNotAborted(signal);
    const runMmluTree = true;
    onLog?.({ text: `[Bench] target=${config.target} runMode=${config.experimentRunMode} mmluExperimentMode=${activeMmluExperimentMode}` });
    let exp2NodeCacheStats:
      | {
        attempts: number;
        sharedHits: number;
        sharedMisses: number;
        sharedHitRatePct: number;
        questionHits: number;
        questionMisses: number;
        questionHitRatePct: number;
      }
      | undefined;

    let mmluSummary: BenchSummary | undefined;
    if (!runExp4Only && runMmluTarget && config.mmluEvalCount > 0) {
      if (activeMmluExperimentMode === 'exp2-random-twice') {
        onLog?.({ text: 'Running MMLU (Exp2 tree-only policy ablation)...' });
        const mmluTree = await runMmlu(
          wllama,
          mmluData,
          config.mmluShots,
          config.mmluEvalCount,
          'tree',
          config,
          activeMmluExperimentMode,
          config.randomSeed,
          1,
          onLog,
          signal,
          async (oldRuntime, reason) => {
            diagnostics.runtimeRestartCount += 1;
            wllama = await replaceLoadedWllama(oldRuntime, config, onLog, reason);
            return wllama;
          },
          () => reportQuestionDone('MMLU tree'),
          reportQuestionFailure,
          onCacheMaintenanceMs
        );
        wllama = mmluTree.wllama;
        exp2NodeCacheStats = mmluTree.exp2NodeCacheStats;
        mmluSummary = summarize(
          'MMLU',
          config.mmluShots,
          mmluTree.results.length,
          mmluTree.results,
          [],
          mmluTree.latencyMs,
          [],
          mmluTree.ttftMs,
          [],
          mmluTree.tokensPerSecond
        );
      } else {
        onLog?.({ text: 'Running MMLU (flat)...' });
        const mmluFlat = await runMmlu(
          wllama,
          mmluData,
          config.mmluShots,
          config.mmluEvalCount,
          'flat',
          config,
          activeMmluExperimentMode,
          config.randomSeed,
          1,
          onLog,
          signal,
          async (oldRuntime, reason) => {
            diagnostics.runtimeRestartCount += 1;
            wllama = await replaceLoadedWllama(oldRuntime, config, onLog, reason);
            return wllama;
          },
          () => reportQuestionDone('MMLU flat'),
          reportQuestionFailure,
          onCacheMaintenanceMs
        );
        wllama = mmluFlat.wllama;

        if (runMmluTree) {
          onLog?.({ text: `Running MMLU (tree, mode=${activeMmluExperimentMode})...` });
          const mmluTree = await runMmlu(
            wllama,
            mmluData,
            config.mmluShots,
            config.mmluEvalCount,
            'tree',
            config,
            activeMmluExperimentMode,
            config.randomSeed,
            1,
            onLog,
            signal,
            async (oldRuntime, reason) => {
              diagnostics.runtimeRestartCount += 1;
              wllama = await replaceLoadedWllama(oldRuntime, config, onLog, reason);
              return wllama;
            },
            () => reportQuestionDone('MMLU tree'),
            reportQuestionFailure,
            onCacheMaintenanceMs
          );
          wllama = mmluTree.wllama;

          const mmluRows = mergeRows(mmluFlat.results, mmluTree.results);
          mmluSummary = summarize(
            'MMLU',
            config.mmluShots,
            mmluRows.length,
            mmluRows,
            mmluFlat.latencyMs,
            mmluTree.latencyMs,
            mmluFlat.ttftMs,
            mmluTree.ttftMs,
            mmluFlat.tokensPerSecond,
            mmluTree.tokensPerSecond
          );
        }
      }
    } else if (!runExp4Only && runMmluTarget) {
      onLog?.({ text: 'Skipping MMLU (mmluEvalCount=0).' });
      mmluSummary = emptySummary('MMLU', config.mmluShots, config.mmluEvalCount);
    }

    let hellaSummary: BenchSummary | undefined;
    if (!runExp4Only && runHellaTarget && config.hellaEvalCount > 0) {
      onLog?.({ text: 'Running HellaSwag (flat)...' });
      const hellaFlat = await runHella(
        wllama,
        hellaData,
        config.hellaShots,
        config.hellaEvalCount,
        'flat',
        config,
        1,
        onLog,
        signal,
        async (oldRuntime, reason) => {
          diagnostics.runtimeRestartCount += 1;
          wllama = await replaceLoadedWllama(oldRuntime, config, onLog, reason);
          return wllama;
        },
        () => reportQuestionDone('Hella flat'),
        reportQuestionFailure,
        onCacheMaintenanceMs
      );
      wllama = hellaFlat.wllama;

      onLog?.({ text: 'Running HellaSwag (tree)...' });
      const hellaTree = await runHella(
        wllama,
        hellaData,
        config.hellaShots,
        config.hellaEvalCount,
        'tree',
        config,
        1,
        onLog,
        signal,
        async (oldRuntime, reason) => {
          diagnostics.runtimeRestartCount += 1;
          wllama = await replaceLoadedWllama(oldRuntime, config, onLog, reason);
          return wllama;
        },
        () => reportQuestionDone('Hella tree'),
        reportQuestionFailure,
        onCacheMaintenanceMs
      );
      wllama = hellaTree.wllama;

      const hellaRows = mergeRows(hellaFlat.results, hellaTree.results);
      hellaSummary = summarize(
        'HellaSwag',
        config.hellaShots,
        config.hellaEvalCount,
        hellaRows,
        hellaFlat.latencyMs,
        hellaTree.latencyMs,
        hellaFlat.ttftMs,
        hellaTree.ttftMs,
        hellaFlat.tokensPerSecond,
        hellaTree.tokensPerSecond
      );
    } else if (!runExp4Only && runHellaTarget) {
      onLog?.({ text: 'Skipping HellaSwag (hellaEvalCount=0).' });
      hellaSummary = emptySummary('HellaSwag', config.hellaShots, config.hellaEvalCount);
    }

    let queueVsDirect: QueueVsDirectSummary | undefined;
    const exp4AvailableEvalItems = Math.max(0, mmluData.length - config.mmluShots);
    if (runExp4Only && runMmluTarget && exp4AvailableEvalItems > 0) {
      onLog?.({ text: `[Exp4] queue vs direct, concurrency=${config.exp4Concurrency}, sampleCount=${config.exp4SampleCount}` });
      const exp4 = await runQueueVsDirectMmlu(
        wllama,
        mmluData,
        config,
        signal,
        onLog,
        async (oldRuntime, reason) => {
          diagnostics.runtimeRestartCount += 1;
          onLog?.({ text: `[Diag] runtime restart #${diagnostics.runtimeRestartCount}, reason=${reason}` });
          wllama = await replaceLoadedWllama(oldRuntime, config, onLog, reason);
          return wllama;
        }
      );
      queueVsDirect = exp4.summary;
      wllama = exp4.wllama;
    } else if (runExp4Only && runMmluTarget) {
      onLog?.({ text: '[Exp4] skipped because there are no evaluable MMLU rows after shots.' });
    } else if (runExp4Only && !runMmluTarget) {
      onLog?.({ text: '[Exp4] skipped because target has no MMLU.' });
    } else if (!runExp4Only) {
      onLog?.({ text: '[Exp4] skipped because runMode is not exp4.' });
    }

    onProgress?.({ current: Math.min(doneQuestions, totalQuestions), total: totalQuestions, label: 'Done' });

    await probeTreeState(wllama, 'bench-final-before-summary', onLog);
    await logTreeContentPreview(wllama, 'bench-final-before-summary', onLog);
    const treeState = await tryGetTreeState(wllama);
    if (!treeState) {
      onLog?.({ text: '[Exp3] tree state unavailable; snapshot/tier stats remain zero.' });
    }
    onLog?.({ text: `[Diag] restarts=${diagnostics.runtimeRestartCount} failures(timeout/abort/disposed/other)=${diagnostics.timeoutFailureCount}/${diagnostics.abortFailureCount}/${diagnostics.disposedFailureCount}/${diagnostics.otherFailureCount}` });
    if (treeState?.tierStats) {
      const attempts = treeState.tierStats.restoreAttempts ?? 0;
      const hitsL1 = treeState.tierStats.restoreHitsL1 ?? 0;
      const hitsL2 = treeState.tierStats.restoreHitsL2 ?? 0;
      const hitsL3 = treeState.tierStats.restoreHitsL3 ?? 0;
      const promotions = treeState.tierStats.promotions ?? 0;
      const demotions = treeState.tierStats.demotions ?? 0;
      const diskReads = treeState.tierStats.diskReads ?? 0;
      const diskWrites = treeState.tierStats.diskWrites ?? 0;
      const l3OverflowEvents = treeState.tierStats.l3OverflowEvents ?? 0;
      const misses = treeState.tierStats.restoreMisses ?? 0;
      const rebuilds = treeState.tierStats.restoreRebuilds ?? 0;
      const parentRecoverAttempts = treeState.tierStats.parentRecoverAttempts ?? 0;
      const parentRecoverSuccesses = treeState.tierStats.parentRecoverSuccesses ?? 0;
      const parentRecoverFailures = treeState.tierStats.parentRecoverFailures ?? 0;
      const tierStatsExt = treeState.tierStats as unknown as Record<string, number>;
      const slotAllocHits = tierStatsExt.slotAllocHits ?? 0;
      const slotAllocMisses = tierStatsExt.slotAllocMisses ?? 0;
      const slotEvictL1 = tierStatsExt.slotEvictL1 ?? 0;
      const slotEvictL2 = tierStatsExt.slotEvictL2 ?? 0;
      const slotEvictL3 = tierStatsExt.slotEvictL3 ?? 0;
      const fallbackReplays = tierStatsExt.fallbackReplays ?? 0;
      const hits = hitsL1 + hitsL2 + hitsL3;
      const hitRatePct = attempts > 0 ? (hits / attempts) * 100 : 0;
      const slotAllocTotal = slotAllocHits + slotAllocMisses;
      diagnostics.exp2CacheStats = {
        restoreAttempts: attempts,
        restoreHits: hits,
        restoreHitRatePct: hitRatePct,
        restoreHitsL1: hitsL1,
        restoreHitsL2: hitsL2,
        restoreHitsL3: hitsL3,
        restoreMisses: misses,
        restoreRebuilds: rebuilds,
        promotions,
        demotions,
        diskReads,
        diskWrites,
        l3OverflowEvents,
        parentRecoverAttempts,
        parentRecoverSuccesses,
        parentRecoverFailures,
        slotAllocHits,
        slotAllocMisses,
        slotEvictL1,
        slotEvictL2,
        slotEvictL3,
        fallbackReplays,
        nodeCacheAttempts: exp2NodeCacheStats?.attempts,
        sharedNodeCacheHits: exp2NodeCacheStats?.sharedHits,
        sharedNodeCacheMisses: exp2NodeCacheStats?.sharedMisses,
        sharedNodeCacheHitRatePct: exp2NodeCacheStats?.sharedHitRatePct,
        questionNodeCacheHits: exp2NodeCacheStats?.questionHits,
        questionNodeCacheMisses: exp2NodeCacheStats?.questionMisses,
        questionNodeCacheHitRatePct: exp2NodeCacheStats?.questionHitRatePct,
      };
      if (exp2NodeCacheStats) {
        onLog?.({
          text: `[Diag] exp2NodeCache sharedHit/miss=${exp2NodeCacheStats.sharedHits}/${exp2NodeCacheStats.sharedMisses}(${exp2NodeCacheStats.sharedHitRatePct.toFixed(1)}%) questionHit/miss=${exp2NodeCacheStats.questionHits}/${exp2NodeCacheStats.questionMisses}(${exp2NodeCacheStats.questionHitRatePct.toFixed(1)}%) attempts=${exp2NodeCacheStats.attempts}`,
        });
      }
      onLog?.({
        text: `[Diag] exp2Restore attempts=${attempts} hit=${hits}(${hitRatePct.toFixed(1)}%) byTier(L1/L2/L3)=${hitsL1}/${hitsL2}/${hitsL3} miss=${misses} rebuilds=${rebuilds}`,
      });
      onLog?.({
        text: `[Diag/Internal] slotAlloc noEvict/needEvict=${slotAllocHits}/${slotAllocMisses} evict(L1/L2/L3)=${slotEvictL1}/${slotEvictL2}/${slotEvictL3} fallbackReplay=${fallbackReplays}`,
      });
    }
    if (diagnostics.timeoutFailureCount > 0) {
      const topTimeouts = Object.entries(diagnostics.timeoutPhaseCounts ?? {})
        .sort((a, b) => b[1] - a[1])
        .slice(0, 3)
        .map(([phase, count]) => `${phase}:${count}`)
        .join(', ');
      onLog?.({ text: `[Diag] timeoutTopPhases=${topTimeouts || 'n/a'}` });
    }

    return {
      modelUrl: config.modelUrl,
      config,
      mmlu: mmluSummary,
      hella: hellaSummary,
      cacheProfile: buildCacheProfile(cacheMaintenanceMs, runStartedAt, treeState),
      queueVsDirect,
      diagnostics,
    };
  } finally {
    if (wllama) {
      await probeTreeState(wllama, 'bench-finally-before-exit', onLog);
      await logTreeContentPreview(wllama, 'bench-finally-before-exit', onLog);
      await safeExitWllama(wllama, onLog, 'runSglangStyleBench/finally');
      wllama = null;
    }
  }
}
