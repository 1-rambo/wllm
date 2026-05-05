import { Wllama } from '@wllama/wllama';
import wllamaSingle from '@wllama/wllama/src/single-thread/wllama.wasm?url';
import wllamaMulti from '@wllama/wllama/src/multi-thread/wllama.wasm?url';
import type {
  BenchConfig,
  BenchDiagnostics,
  BenchFailureRecord,
  BenchLogEvent,
  BenchProgressEvent,
  BenchReport,
  BenchSampleMetric,
  BenchSeriesStats,
  BenchSummary,
  CacheOccupancySample,
  CacheProfile,
  CdfPoint,
  WebArenaTask,
} from './types';

const WLLAMA_CONFIG_PATHS = {
  'single-thread/wllama.wasm': wllamaSingle,
  'multi-thread/wllama.wasm': wllamaMulti,
};

const SITE_PROFILES: Record<string, string> = {
  shopping: [
    'This site is an e-commerce product storefront.',
    'Retrieve tasks usually require reading product pages, reviews, reviewer names, or product attributes.',
    'The answer should be concise and directly grounded in the task description.',
  ].join(' '),
  shopping_admin: [
    'This site is an e-commerce admin dashboard.',
    'Retrieve tasks usually ask for counts, best-selling products, brands, review statistics, or store analytics.',
    'The answer should be short and data-oriented.',
  ].join(' '),
  gitlab: [
    'This site is a GitLab project hosting environment.',
    'Retrieve tasks usually ask about commits, users, issues, merge requests, or project activity over time.',
    'The answer should be concise and factual.',
  ].join(' '),
  reddit: [
    'This site is a Reddit-like discussion forum.',
    'Retrieve tasks usually ask about recent posts, usernames, comment counts, and voting patterns in a subreddit.',
    'The answer should be concise and directly grounded in the task description.',
  ].join(' '),
  wikipedia: [
    'This site is a Wikipedia-like encyclopedia.',
    'Retrieve tasks usually require locating facts from a long article page.',
    'The answer should be short and factual.',
  ].join(' '),
};

function avg(values: number[]): number {
  if (!values.length) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function quantile(values: number[], q: number): number {
  if (!values.length) return 0;
  const sorted = values.slice().sort((a, b) => a - b);
  const pos = Math.min(sorted.length - 1, Math.max(0, (sorted.length - 1) * q));
  const lower = Math.floor(pos);
  const upper = Math.ceil(pos);
  if (lower === upper) return sorted[lower];
  const weight = pos - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

function buildCdf(values: number[]): CdfPoint[] {
  if (!values.length) return [];
  const sorted = values.slice().sort((a, b) => a - b);
  return sorted.map((value, index) => ({
    value,
    cdf: (index + 1) / sorted.length,
  }));
}

function safeText(text: unknown): string {
  return String(text ?? '').replace(/\s+/g, ' ').trim();
}

function buildSiteSharedPrefix(site: string, pageContext?: string): string {
  const siteProfile = SITE_PROFILES[site] ?? 'This is a WebArena browser environment task.';
  const sections = [
    'You are solving a WebArena Verified information-seeking task.',
    `Current site: ${site}.`,
    siteProfile,
  ];
  if (safeText(pageContext)) {
    sections.push(
      'The following page context was preloaded in the background before the user asked the question. Treat it as already-available page text extracted from the current site:',
      safeText(pageContext),
    );
  }
  sections.push(
    'You are given WebArena task metadata rather than a live browser trace.',
    'Use the task description as the primary signal and produce a short final answer.',
    'If the task requests multiple named fields, answer with compact JSON and no explanation.',
    'If the task asks for a count, return only the count or a compact JSON value.',
  );
  return sections.join('\n');
}

function buildTaskPrompt(task: WebArenaTask): string {
  const lines = [
    `Task ID: ${task.taskId}`,
    `Site: ${task.site}`,
    `Start URLs: ${task.startUrls.join(' | ')}`,
    `Intent Template ID: ${task.intentTemplateId ?? 'n/a'}`,
    'Intent:',
    safeText(task.intent),
  ];
  if (task.intentTemplate) {
    lines.push('Intent Template:');
    lines.push(safeText(task.intentTemplate));
  }
  if (task.instantiationDict && Object.keys(task.instantiationDict).length > 0) {
    lines.push('Instantiation Arguments:');
    lines.push(JSON.stringify(task.instantiationDict));
  }
  lines.push('Return only the final answer in one concise line or compact JSON. Do not explain.');
  return lines.join('\n');
}

function buildPromptChars(sharedPrefix: string, taskPrompt: string): number {
  return `${sharedPrefix}\n\n${taskPrompt}`.length;
}

function getSharedContextKey(task: WebArenaTask): string {
  return task.pageContextKey
    ?? task.renderedStartUrls?.join(' | ')
    ?? task.startUrls.join(' | ')
    ?? task.site;
}

type FailureCounterKey =
  | 'timeoutFailureCount'
  | 'abortFailureCount'
  | 'disposedFailureCount'
  | 'otherFailureCount';

function classifyFailure(err: unknown): FailureCounterKey {
  const msg = err instanceof Error ? err.message.toLowerCase() : String(err).toLowerCase();
  if (msg.includes('abort')) return 'abortFailureCount';
  if (msg.includes('timeout')) return 'timeoutFailureCount';
  if (msg.includes('disposed') || msg.includes('terminated')) return 'disposedFailureCount';
  return 'otherFailureCount';
}

async function createLoadedWllama(config: BenchConfig, onLog?: (e: BenchLogEvent) => void): Promise<Wllama> {
  const logger = {
    debug: (...args: unknown[]) => console.debug(...args),
    log: (...args: unknown[]) => console.log(...args),
    warn: (...args: unknown[]) => console.warn(...args),
    error: (...args: unknown[]) => console.error(...args),
  };
  const wllama = new Wllama(WLLAMA_CONFIG_PATHS, {
    preferWebGPU: true,
    noPerf: false,
    suppressNativeLog: false,
    logger,
  });
  onLog?.({ text: '[Wllama] loading model...' });
  await wllama.loadModelFromUrl(config.modelUrl, {
    useCache: true,
    n_ctx: config.nCtx,
    n_batch: config.nBatch,
    embeddings: false,
    offload_kqv: true,
    flash_attn: true,
  });
  onLog?.({ text: '[Wllama] model loaded.' });
  return wllama;
}

async function runWllamaPrompt(
  runtime: Wllama,
  parentNodeId: number,
  prompt: string,
  maxOutputTokens: number,
  startedAt?: number
): Promise<{ assistantText: string; latencyMs: number; ttftMs: number; tokensPerSecond: number; tokenCount: number }> {
  const t0 = startedAt ?? performance.now();
  let firstTokenAt = 0;
  const out = await runtime.chatFromNode(parentNodeId, prompt, {
    nPredict: maxOutputTokens,
    sampling: { temp: 0.1, top_p: 0.9, top_k: 40 },
    onChunk: () => {
      if (!firstTokenAt) {
        firstTokenAt = performance.now();
      }
    },
  });
  const latencyMs = Math.max(0, performance.now() - t0);
  const ttftMs = Math.max(0, (firstTokenAt || performance.now()) - t0);
  const answerText = out.assistantText || '';
  const tokenCount = Math.max(1, (await runtime.tokenize(answerText, true)).length);
  const tokensPerSecond = (tokenCount * 1000) / Math.max(latencyMs, 1e-6);
  return { assistantText: answerText, latencyMs, ttftMs, tokensPerSecond, tokenCount };
}

async function runWebllmPrompt(
  engine: any,
  prompt: string,
  maxOutputTokens: number
): Promise<{ assistantText: string; latencyMs: number; ttftMs: number; tokensPerSecond: number; tokenCount: number }> {
  const t0 = performance.now();
  let firstTokenAt = 0;
  let assistantText = '';
  let completionTokens = 0;
  await engine.resetChat();
  const stream = await engine.chat.completions.create({
    messages: [{ role: 'user', content: prompt }],
    temperature: 0.1,
    top_p: 0.9,
    max_tokens: maxOutputTokens,
    stream: true,
    stream_options: { include_usage: true },
  });
  for await (const chunk of stream) {
    const c = chunk as {
      choices?: Array<{ delta?: { content?: string | null } }>;
      usage?: { completion_tokens?: number };
    };
    const delta = c.choices?.[0]?.delta?.content ?? '';
    if (delta) {
      if (!firstTokenAt) {
        firstTokenAt = performance.now();
      }
      assistantText += delta;
    }
    if (typeof c.usage?.completion_tokens === 'number') {
      completionTokens = c.usage.completion_tokens;
    }
  }
  const latencyMs = Math.max(0, performance.now() - t0);
  const ttftMs = Math.max(0, (firstTokenAt || performance.now()) - t0);
  if (completionTokens <= 0) {
    completionTokens = Math.max(1, safeText(assistantText).length > 0 ? 1 : 0);
  }
  const tokensPerSecond = (completionTokens * 1000) / Math.max(latencyMs, 1e-6);
  return { assistantText, latencyMs, ttftMs, tokensPerSecond, tokenCount: completionTokens };
}

function buildMetric(
  task: WebArenaTask,
  mode: 'flat' | 'tree' | 'web-llm',
  sharedPrefix: string,
  taskPrompt: string,
  result: { assistantText: string; latencyMs: number; ttftMs: number; tokensPerSecond: number; tokenCount: number }
): BenchSampleMetric {
  return {
    benchmark: 'WebArena',
    id: task.id,
    subject: task.site,
    mode,
    latencyMs: result.latencyMs,
    ttftMs: result.ttftMs,
    tokensPerSecond: result.tokensPerSecond,
    tokenCount: result.tokenCount,
    promptChars: buildPromptChars(sharedPrefix, taskPrompt),
    outputChars: result.assistantText.length,
  };
}

function logSampleMetric(onLog: ((e: BenchLogEvent) => void) | undefined, metric: BenchSampleMetric): void {
  onLog?.({
    text: [
      `[Metric/${metric.benchmark}/${metric.mode}]`,
      `id=${metric.id}`,
      `site=${metric.subject ?? ''}`,
      `latencyMs=${metric.latencyMs.toFixed(2)}`,
      `ttftMs=${metric.ttftMs.toFixed(2)}`,
      `tokens=${metric.tokenCount}`,
      `tps=${metric.tokensPerSecond.toFixed(3)}`,
      `promptChars=${metric.promptChars}`,
      `outputChars=${metric.outputChars}`,
    ].join(' '),
  });
}

function summarizeSeries(metrics: BenchSampleMetric[]): BenchSeriesStats {
  const latency = metrics.map((m) => m.latencyMs);
  const ttft = metrics.map((m) => m.ttftMs);
  const tps = metrics.map((m) => m.tokensPerSecond);
  return {
    sampleCount: metrics.length,
    avgLatencyMs: avg(latency),
    p50LatencyMs: quantile(latency, 0.5),
    p95LatencyMs: quantile(latency, 0.95),
    p99LatencyMs: quantile(latency, 0.99),
    avgTtftMs: avg(ttft),
    p50TtftMs: quantile(ttft, 0.5),
    p95TtftMs: quantile(ttft, 0.95),
    p99TtftMs: quantile(ttft, 0.99),
    avgTokensPerSecond: avg(tps),
  };
}

function summarizeDualPath(tasks: WebArenaTask[], flatMetrics: BenchSampleMetric[], treeMetrics: BenchSampleMetric[]): BenchSummary {
  const flatStats = summarizeSeries(flatMetrics);
  const treeStats = summarizeSeries(treeMetrics);
  const siteBreakdown: Record<string, number> = {};
  for (const task of tasks) {
    siteBreakdown[task.site] = (siteBreakdown[task.site] ?? 0) + 1;
  }
  return {
    benchmark: 'WebArena',
    evalCount: tasks.length,
    successCountFlat: flatMetrics.length,
    successCountTree: treeMetrics.length,
    failureCountFlat: Math.max(0, tasks.length - flatMetrics.length),
    failureCountTree: Math.max(0, tasks.length - treeMetrics.length),
    siteBreakdown,
    avgTtftMsFlat: flatStats.avgTtftMs,
    avgTtftMsTree: treeStats.avgTtftMs,
    ttftSpeedupPct: flatStats.avgTtftMs > 0 ? ((flatStats.avgTtftMs - treeStats.avgTtftMs) / flatStats.avgTtftMs) * 100 : 0,
    avgTokensPerSecondFlat: flatStats.avgTokensPerSecond,
    avgTokensPerSecondTree: treeStats.avgTokensPerSecond,
    tpsGainPct: flatStats.avgTokensPerSecond > 0 ? ((treeStats.avgTokensPerSecond - flatStats.avgTokensPerSecond) / flatStats.avgTokensPerSecond) * 100 : 0,
    avgLatencyMsFlat: flatStats.avgLatencyMs,
    avgLatencyMsTree: treeStats.avgLatencyMs,
    speedupPct: flatStats.avgLatencyMs > 0 ? ((flatStats.avgLatencyMs - treeStats.avgLatencyMs) / flatStats.avgLatencyMs) * 100 : 0,
    sampleMetricsFlat: flatMetrics,
    sampleMetricsTree: treeMetrics,
    latencyCdfFlat: buildCdf(flatMetrics.map((m) => m.latencyMs)),
    latencyCdfTree: buildCdf(treeMetrics.map((m) => m.latencyMs)),
    ttftCdfFlat: buildCdf(flatMetrics.map((m) => m.ttftMs)),
    ttftCdfTree: buildCdf(treeMetrics.map((m) => m.ttftMs)),
  };
}

function summarizeSinglePath(tasks: WebArenaTask[], metrics: BenchSampleMetric[]): BenchSummary {
  const stats = summarizeSeries(metrics);
  const siteBreakdown: Record<string, number> = {};
  for (const task of tasks) {
    siteBreakdown[task.site] = (siteBreakdown[task.site] ?? 0) + 1;
  }
  return {
    benchmark: 'WebArena',
    evalCount: tasks.length,
    successCountFlat: 0,
    successCountTree: metrics.length,
    failureCountFlat: 0,
    failureCountTree: Math.max(0, tasks.length - metrics.length),
    siteBreakdown,
    avgTtftMsFlat: 0,
    avgTtftMsTree: stats.avgTtftMs,
    ttftSpeedupPct: 0,
    avgTokensPerSecondFlat: 0,
    avgTokensPerSecondTree: stats.avgTokensPerSecond,
    tpsGainPct: 0,
    avgLatencyMsFlat: 0,
    avgLatencyMsTree: stats.avgLatencyMs,
    speedupPct: 0,
    sampleMetricsFlat: [],
    sampleMetricsTree: metrics,
    latencyCdfFlat: [],
    latencyCdfTree: buildCdf(metrics.map((m) => m.latencyMs)),
    ttftCdfFlat: [],
    ttftCdfTree: buildCdf(metrics.map((m) => m.ttftMs)),
  };
}

function emptyCacheProfile(): CacheProfile {
  return {
    maintenanceMs: 0,
    runTotalMs: 0,
    maintenancePct: 0,
    maintenanceBreakdownMs: {
      sessionInitMs: 0,
      stateReadMs: 0,
      prefixSetupMs: 0,
      otherMs: 0,
    },
    snapshotTokenBytes: 0,
    tierL1Tokens: 0,
    tierL2Tokens: 0,
    tierL3Tokens: 0,
    occupancyStats: {
      sampleCount: 0,
      avgSnapshotTokenBytes: 0,
      peakSnapshotTokenBytes: 0,
      avgTierL1Tokens: 0,
      avgTierL2Tokens: 0,
      avgTierL3Tokens: 0,
      peakTierL1Tokens: 0,
      peakTierL2Tokens: 0,
      peakTierL3Tokens: 0,
      avgTierL1Slots: 0,
      avgTierL2Slots: 0,
      avgTierL3Slots: 0,
      peakTierL1Slots: 0,
      peakTierL2Slots: 0,
      peakTierL3Slots: 0,
    },
    occupancySamples: [],
  };
}

function emptyDiagnostics(): BenchDiagnostics {
  return {
    runtimeRestartCount: 0,
    timeoutFailureCount: 0,
    abortFailureCount: 0,
    disposedFailureCount: 0,
    otherFailureCount: 0,
    failureRecords: [],
  };
}

type CacheMaintenanceCategory = 'sessionInit' | 'stateRead' | 'prefixSetup' | 'other';

type CacheMaintenanceRecorder = {
  totalMs: number;
  breakdownMs: Record<CacheMaintenanceCategory, number>;
  occupancySampleCount: number;
  snapshotTokenBytesSum: number;
  snapshotTokenBytesPeak: number;
  tierL1TokensSum: number;
  tierL2TokensSum: number;
  tierL3TokensSum: number;
  tierL1TokensPeak: number;
  tierL2TokensPeak: number;
  tierL3TokensPeak: number;
  tierL1SlotsSum: number;
  tierL2SlotsSum: number;
  tierL3SlotsSum: number;
  tierL1SlotsPeak: number;
  tierL2SlotsPeak: number;
  tierL3SlotsPeak: number;
  samples: CacheOccupancySample[];
};

type MinimalTreeState = {
  totalSnapshotTokenBytes?: number;
  tierStats?: {
    l1Tokens?: number;
    l2Tokens?: number;
    l3Tokens?: number;
    l1Slots?: number;
    l2Slots?: number;
    l3Slots?: number;
  };
};

function createCacheMaintenanceRecorder(): CacheMaintenanceRecorder {
  return {
    totalMs: 0,
    breakdownMs: {
      sessionInit: 0,
      stateRead: 0,
      prefixSetup: 0,
      other: 0,
    },
    occupancySampleCount: 0,
    snapshotTokenBytesSum: 0,
    snapshotTokenBytesPeak: 0,
    tierL1TokensSum: 0,
    tierL2TokensSum: 0,
    tierL3TokensSum: 0,
    tierL1TokensPeak: 0,
    tierL2TokensPeak: 0,
    tierL3TokensPeak: 0,
    tierL1SlotsSum: 0,
    tierL2SlotsSum: 0,
    tierL3SlotsSum: 0,
    tierL1SlotsPeak: 0,
    tierL2SlotsPeak: 0,
    tierL3SlotsPeak: 0,
    samples: [],
  };
}

function observeCacheOccupancy(
  recorder: CacheMaintenanceRecorder,
  state: MinimalTreeState | null | undefined,
  meta: {
    label: string;
    mode: 'flat' | 'tree' | 'web-llm' | 'runtime';
    site: string;
    groupKey: string;
    taskId?: string;
  }
): void {
  if (!state) return;
  const snapshotTokenBytes = state.totalSnapshotTokenBytes ?? 0;
  const l1Tokens = state.tierStats?.l1Tokens ?? 0;
  const l2Tokens = state.tierStats?.l2Tokens ?? 0;
  const l3Tokens = state.tierStats?.l3Tokens ?? 0;
  const l1Slots = state.tierStats?.l1Slots ?? 0;
  const l2Slots = state.tierStats?.l2Slots ?? 0;
  const l3Slots = state.tierStats?.l3Slots ?? 0;
  recorder.occupancySampleCount += 1;
  recorder.snapshotTokenBytesSum += snapshotTokenBytes;
  recorder.snapshotTokenBytesPeak = Math.max(recorder.snapshotTokenBytesPeak, snapshotTokenBytes);
  recorder.tierL1TokensSum += l1Tokens;
  recorder.tierL2TokensSum += l2Tokens;
  recorder.tierL3TokensSum += l3Tokens;
  recorder.tierL1TokensPeak = Math.max(recorder.tierL1TokensPeak, l1Tokens);
  recorder.tierL2TokensPeak = Math.max(recorder.tierL2TokensPeak, l2Tokens);
  recorder.tierL3TokensPeak = Math.max(recorder.tierL3TokensPeak, l3Tokens);
  recorder.tierL1SlotsSum += l1Slots;
  recorder.tierL2SlotsSum += l2Slots;
  recorder.tierL3SlotsSum += l3Slots;
  recorder.tierL1SlotsPeak = Math.max(recorder.tierL1SlotsPeak, l1Slots);
  recorder.tierL2SlotsPeak = Math.max(recorder.tierL2SlotsPeak, l2Slots);
  recorder.tierL3SlotsPeak = Math.max(recorder.tierL3SlotsPeak, l3Slots);
  recorder.samples.push({
    label: meta.label,
    mode: meta.mode,
    site: meta.site,
    groupKey: meta.groupKey,
    taskId: meta.taskId,
    snapshotTokenBytes,
    tierL1Tokens: l1Tokens,
    tierL2Tokens: l2Tokens,
    tierL3Tokens: l3Tokens,
    tierL1Slots: l1Slots,
    tierL2Slots: l2Slots,
    tierL3Slots: l3Slots,
  });
}

function buildCacheProfile(
  recorder: CacheMaintenanceRecorder,
  runStartedAt: number,
  state?: MinimalTreeState | null
): CacheProfile {
  if (state) {
    observeCacheOccupancy(recorder, state, {
      label: 'final-state',
      mode: 'runtime',
      site: 'all',
      groupKey: 'final',
    });
  }
  const runTotalMs = Math.max(0, performance.now() - runStartedAt);
  const tier = state?.tierStats;
  const sampleCount = recorder.occupancySampleCount;
  const avgOf = (sum: number) => (sampleCount > 0 ? sum / sampleCount : 0);
  return {
    maintenanceMs: recorder.totalMs,
    runTotalMs,
    maintenancePct: runTotalMs > 0 ? (recorder.totalMs / runTotalMs) * 100 : 0,
    maintenanceBreakdownMs: {
      sessionInitMs: recorder.breakdownMs.sessionInit,
      stateReadMs: recorder.breakdownMs.stateRead,
      prefixSetupMs: recorder.breakdownMs.prefixSetup,
      otherMs: recorder.breakdownMs.other,
    },
    snapshotTokenBytes: state?.totalSnapshotTokenBytes ?? 0,
    tierL1Tokens: tier?.l1Tokens ?? 0,
    tierL2Tokens: tier?.l2Tokens ?? 0,
    tierL3Tokens: tier?.l3Tokens ?? 0,
    occupancyStats: {
      sampleCount,
      avgSnapshotTokenBytes: avgOf(recorder.snapshotTokenBytesSum),
      peakSnapshotTokenBytes: recorder.snapshotTokenBytesPeak,
      avgTierL1Tokens: avgOf(recorder.tierL1TokensSum),
      avgTierL2Tokens: avgOf(recorder.tierL2TokensSum),
      avgTierL3Tokens: avgOf(recorder.tierL3TokensSum),
      peakTierL1Tokens: recorder.tierL1TokensPeak,
      peakTierL2Tokens: recorder.tierL2TokensPeak,
      peakTierL3Tokens: recorder.tierL3TokensPeak,
      avgTierL1Slots: avgOf(recorder.tierL1SlotsSum),
      avgTierL2Slots: avgOf(recorder.tierL2SlotsSum),
      avgTierL3Slots: avgOf(recorder.tierL3SlotsSum),
      peakTierL1Slots: recorder.tierL1SlotsPeak,
      peakTierL2Slots: recorder.tierL2SlotsPeak,
      peakTierL3Slots: recorder.tierL3SlotsPeak,
    },
    occupancySamples: recorder.samples,
  };
}

function capToBytes(mb: number): number {
  return Math.max(0, Math.floor(mb * 1024 * 1024));
}

function tieredCacheOptions(config: BenchConfig) {
  return config.trueTreeTieredCacheEnabled ? {
    enabled: true,
    l1TokenCap: config.trueTreeTierL1TokenCap,
    l2TokenCap: config.trueTreeTierL2TokenCap,
    l3TokenCap: config.trueTreeTierL3TokenCap,
    pruneL1L2TokenThreshold: config.trueTreePruneL1L2TokenThreshold,
    pruneL2L3TokenThreshold: config.trueTreePruneL2L3TokenThreshold,
    replacementPolicy: config.trueTreeReplacementPolicy,
  } : { enabled: false };
}

export async function runWebArenaBench(
  config: BenchConfig,
  tasks: WebArenaTask[],
  onLog?: (e: BenchLogEvent) => void,
  onProgress?: (e: BenchProgressEvent) => void,
  signal?: AbortSignal
): Promise<BenchReport> {
  const selectedTasks = tasks.slice(0, Math.min(config.evalCount, tasks.length));
  const diagnostics = emptyDiagnostics();
  const tRun0 = performance.now();

  if (config.backend === 'web-llm') {
    const webllm = await import('@mlc-ai/web-llm');
    onLog?.({ text: `[web-llm] loading model=${config.webllmModelId}` });
    const engine = await webllm.CreateMLCEngine(config.webllmModelId);
    onLog?.({ text: '[web-llm] model loaded.' });
    try {
      const sampleMetrics: BenchSampleMetric[] = [];
      onProgress?.({ current: 0, total: selectedTasks.length, label: 'WebArena web-llm' });
      for (let i = 0; i < selectedTasks.length; i += 1) {
        const task = selectedTasks[i];
        if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
        const sharedPrefix = buildSiteSharedPrefix(
          task.site,
          config.usePreloadedPageContext ? task.pageContext : '',
        );
        const taskPrompt = buildTaskPrompt(task);
        const prompt = `${sharedPrefix}\n\n${taskPrompt}`;
        try {
          const result = await runWebllmPrompt(engine, prompt, config.maxOutputTokens);
          const metric = buildMetric(task, 'web-llm', sharedPrefix, taskPrompt, result);
          sampleMetrics.push(metric);
          logSampleMetric(onLog, metric);
        } catch (err) {
          diagnostics[classifyFailure(err)] += 1;
          diagnostics.failureRecords.push({
            mode: 'web-llm',
            taskId: task.id,
            site: task.site,
            groupKey: getSharedContextKey(task),
            stage: 'taskRun',
            message: err instanceof Error ? err.message : String(err),
          });
          onLog?.({ text: `[WebArena/web-llm] ${task.id} failed: ${err instanceof Error ? err.message : String(err)}` });
        }
        onProgress?.({ current: i + 1, total: selectedTasks.length, label: 'WebArena web-llm' });
      }
      const summary = summarizeSinglePath(selectedTasks, sampleMetrics);
      return {
        modelUrl: config.modelUrl,
        config,
        webarena: summary,
        cacheProfile: emptyCacheProfile(),
        diagnostics,
      };
    } finally {
      try {
        await engine.unload();
      } catch {
        // best effort
      }
    }
  }

  const runtime = await createLoadedWllama(config, onLog);
  const cacheRecorder = createCacheMaintenanceRecorder();
  const flatMetrics: BenchSampleMetric[] = [];
  const treeMetrics: BenchSampleMetric[] = [];
  const grouped = new Map<string, WebArenaTask[]>();
  for (const task of selectedTasks) {
    const groupKey = getSharedContextKey(task);
    const bucket = grouped.get(groupKey) ?? [];
    bucket.push(task);
    grouped.set(groupKey, bucket);
  }
  const orderedGroups = [...grouped.entries()].sort((a, b) => a[0].localeCompare(b[0]));

  let progressCurrent = 0;
  onProgress?.({ current: 0, total: selectedTasks.length * 2, label: 'WebArena flat' });
  let finalState: MinimalTreeState | null = null;

  try {
    for (const [groupKey, siteTasks] of orderedGroups) {
      const site = siteTasks[0]?.site ?? 'unknown';
      const sharedPrefix = buildSiteSharedPrefix(
        site,
        config.usePreloadedPageContext ? siteTasks[0]?.pageContext : '',
      );
      onLog?.({ text: `[WebArena/flat] site=${site} group=${groupKey} shared prefix chars=${sharedPrefix.length}` });
      for (const task of siteTasks) {
        if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
        const taskPrompt = buildTaskPrompt(task);
        const requestStart = performance.now();
        let stage = 'sessionInit';
        try {
          const startSession = performance.now();
          await runtime.chatSessionInit(capToBytes(config.trueTreeMemoryCapMB), tieredCacheOptions(config));
          const sessionMs = performance.now() - startSession;
          cacheRecorder.totalMs += sessionMs;
          cacheRecorder.breakdownMs.sessionInit += sessionMs;

          stage = 'stateRead-before-setup';
          const stateReadStart = performance.now();
          const state = await runtime.chatGetState();
          const stateMs = performance.now() - stateReadStart;
          cacheRecorder.totalMs += stateMs;
          cacheRecorder.breakdownMs.stateRead += stateMs;
          observeCacheOccupancy(cacheRecorder, state, {
            label: 'flat-before-setup',
            mode: 'flat',
            site,
            groupKey,
            taskId: task.id,
          });

          stage = 'prefixSetup';
          const setupStart = performance.now();
          const setup = await runtime.chatFromNode(state.rootId, sharedPrefix, { nPredict: 0 });
          const setupMs = performance.now() - setupStart;
          cacheRecorder.totalMs += setupMs;
          cacheRecorder.breakdownMs.prefixSetup += setupMs;

          stage = 'taskRun';
          const result = await runWllamaPrompt(runtime, setup.nodeId, taskPrompt, config.maxOutputTokens, requestStart);
          const metric = buildMetric(task, 'flat', sharedPrefix, taskPrompt, result);
          flatMetrics.push(metric);
          logSampleMetric(onLog, metric);

          stage = 'stateRead-after-task';
          const afterTaskStateReadStart = performance.now();
          const afterTaskState = await runtime.chatGetState();
          const afterTaskStateMs = performance.now() - afterTaskStateReadStart;
          cacheRecorder.totalMs += afterTaskStateMs;
          cacheRecorder.breakdownMs.stateRead += afterTaskStateMs;
          observeCacheOccupancy(cacheRecorder, afterTaskState, {
            label: 'flat-after-task',
            mode: 'flat',
            site,
            groupKey,
            taskId: task.id,
          });
        } catch (err) {
          diagnostics[classifyFailure(err)] += 1;
          diagnostics.failureRecords.push({
            mode: 'flat',
            taskId: task.id,
            site,
            groupKey,
            stage,
            message: err instanceof Error ? err.message : String(err),
          });
          onLog?.({ text: `[WebArena/flat] ${task.id} failed: ${err instanceof Error ? err.message : String(err)}` });
        }
        progressCurrent += 1;
        onProgress?.({ current: progressCurrent, total: selectedTasks.length * 2, label: 'WebArena flat' });
      }
    }

    for (const [groupKey, siteTasks] of orderedGroups) {
      const site = siteTasks[0]?.site ?? 'unknown';
      const sharedPrefix = buildSiteSharedPrefix(
        site,
        config.usePreloadedPageContext ? siteTasks[0]?.pageContext : '',
      );
      onLog?.({ text: `[WebArena/tree] site=${site} group=${groupKey} shared prefix chars=${sharedPrefix.length}` });
      let sharedNodeId = -1;
      let groupInitFailed = false;
      const sessionInitStart = performance.now();
      try {
        await runtime.chatSessionInit(capToBytes(config.trueTreeMemoryCapMB), tieredCacheOptions(config));
        const sessionMs = performance.now() - sessionInitStart;
        cacheRecorder.totalMs += sessionMs;
        cacheRecorder.breakdownMs.sessionInit += sessionMs;

        const stateReadStart = performance.now();
        const state = await runtime.chatGetState();
        const stateMs = performance.now() - stateReadStart;
        cacheRecorder.totalMs += stateMs;
        cacheRecorder.breakdownMs.stateRead += stateMs;
        observeCacheOccupancy(cacheRecorder, state, {
          label: 'tree-before-setup',
          mode: 'tree',
          site,
          groupKey,
        });

        const setupStart = performance.now();
        const setup = await runtime.chatFromNode(state.rootId, sharedPrefix, { nPredict: 0 });
        const setupMs = performance.now() - setupStart;
        cacheRecorder.totalMs += setupMs;
        cacheRecorder.breakdownMs.prefixSetup += setupMs;
        sharedNodeId = setup.nodeId;
      } catch (err) {
        diagnostics[classifyFailure(err)] += 1;
        for (const task of siteTasks) {
          diagnostics.failureRecords.push({
            mode: 'tree',
            taskId: task.id,
            site,
            groupKey,
            stage: 'shared-prefix-setup',
            message: err instanceof Error ? err.message : String(err),
          });
          onLog?.({ text: `[WebArena/tree] ${task.id} failed: ${err instanceof Error ? err.message : String(err)}` });
          progressCurrent += 1;
          onProgress?.({ current: progressCurrent, total: selectedTasks.length * 2, label: 'WebArena tree' });
        }
        groupInitFailed = true;
      }

      if (groupInitFailed) {
        continue;
      }

      for (const task of siteTasks) {
        if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
        const taskPrompt = buildTaskPrompt(task);
        let stage = 'taskRun';
        try {
          const result = await runWllamaPrompt(runtime, sharedNodeId, taskPrompt, config.maxOutputTokens);
          const metric = buildMetric(task, 'tree', sharedPrefix, taskPrompt, result);
          treeMetrics.push(metric);
          logSampleMetric(onLog, metric);

          stage = 'stateRead-after-task';
          const afterTaskStateReadStart = performance.now();
          const afterTaskState = await runtime.chatGetState();
          const afterTaskStateMs = performance.now() - afterTaskStateReadStart;
          cacheRecorder.totalMs += afterTaskStateMs;
          cacheRecorder.breakdownMs.stateRead += afterTaskStateMs;
          observeCacheOccupancy(cacheRecorder, afterTaskState, {
            label: 'tree-after-task',
            mode: 'tree',
            site,
            groupKey,
            taskId: task.id,
          });
        } catch (err) {
          diagnostics[classifyFailure(err)] += 1;
          diagnostics.failureRecords.push({
            mode: 'tree',
            taskId: task.id,
            site,
            groupKey,
            stage,
            message: err instanceof Error ? err.message : String(err),
          });
          onLog?.({ text: `[WebArena/tree] ${task.id} failed: ${err instanceof Error ? err.message : String(err)}` });
        }
        progressCurrent += 1;
        onProgress?.({ current: progressCurrent, total: selectedTasks.length * 2, label: 'WebArena tree' });
      }
    }
  } finally {
    try {
      const finalStateReadStart = performance.now();
      finalState = await runtime.chatGetState();
      const finalStateMs = performance.now() - finalStateReadStart;
      cacheRecorder.totalMs += finalStateMs;
      cacheRecorder.breakdownMs.stateRead += finalStateMs;
    } catch {
      finalState = null;
    }
    try {
      await runtime.exit();
    } catch {
      // best effort
    }
  }

  const cacheProfile = buildCacheProfile(cacheRecorder, tRun0, finalState);
  return {
    modelUrl: config.modelUrl,
    config,
    webarena: summarizeDualPath(selectedTasks, flatMetrics, treeMetrics),
    cacheProfile,
    diagnostics,
  };
}
