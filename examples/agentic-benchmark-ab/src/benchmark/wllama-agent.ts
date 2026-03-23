import { Wllama } from '@wllama/wllama';
import type {
  ABExperimentResult,
  ABProgress,
  ABTaskOutputEvent,
  ABTaskDelta,
  BenchmarkTask,
  PerfMetrics,
  PrefixConfig,
} from './types';
import {
  buildPrefixAnchorPrompt,
  buildPrefixKey,
  buildSharedPrefix,
  buildTaskPrompt,
} from './prefix-builder';
import websiteConfig from '../../../../AgenticBrowserBenchmark/browser_test_windows/Webarena_website.json';
import wllamaSingle from '@wllama/wllama/src/single-thread/wllama.wasm?url';
import wllamaMulti from '@wllama/wllama/src/multi-thread/wllama.wasm?url';

const WLLAMA_CONFIG_PATHS = {
  'single-thread/wllama.wasm': wllamaSingle,
  'multi-thread/wllama.wasm': wllamaMulti,
};

const SAMPLING = {
  temp: 0.7,
  top_k: 40,
  top_p: 0.9,
};

const URL_FETCH_TIMEOUT_MS = 3000;
const MAX_PAGE_SNIPPET_CHARS = 600;
const MAX_OUTPUT_PREVIEW_CHARS = 180;
const MIN_VALID_OUTPUT_TOKENS = 4;
const MIN_VALID_OUTPUT_TOKENS_SHORT_TASK = 1;
const MIN_VALID_TPS = 1e-6;

const URL_PLACEHOLDERS: Record<string, keyof typeof websiteConfig> = {
  __SHOPPING__: 'SHOPPING_URL',
  __SHOPPING_ADMIN__: 'SHOPPING_ADMIN_URL',
  __GITLAB__: 'GITLAB_URL',
  __REDDIT__: 'REDDIT_URL',
  __MAP__: 'MAP_URL',
  __WIKI__: 'WIKI_URL',
};

type UrlContext = {
  rawUrl: string;
  resolvedUrl: string;
  summary: string;
  source: 'fetched' | 'fallback';
};

function cleanSpaces(input: string): string {
  return input.replace(/\s+/g, ' ').trim();
}

function resolveBenchmarkUrl(rawUrl: string): string {
  let out = rawUrl;
  for (const [token, cfgKey] of Object.entries(URL_PLACEHOLDERS)) {
    const replacement = websiteConfig[cfgKey];
    if (replacement) {
      out = out.split(token).join(replacement);
    }
  }
  return out;
}

function fallbackSummaryFromUrl(rawUrl: string, resolvedUrl: string): string {
  const tail = resolvedUrl.split('/').filter(Boolean).pop() || resolvedUrl;
  const guessedTitle = cleanSpaces(
    tail
      .replace(/\.html?$/i, '')
      .replace(/[-_]+/g, ' ')
      .replace(/\b\w/g, (c) => c.toUpperCase())
  );
  return [
    `Resolved URL: ${resolvedUrl}`,
    `Heuristic title: ${guessedTitle || 'Unknown'}`,
    `Note: Live page fetch unavailable, using URL-derived context for ${rawUrl}.`,
  ].join('\n');
}

function summarizeHtml(html: string, resolvedUrl: string): string {
  const doc = new DOMParser().parseFromString(html, 'text/html');
  doc.querySelectorAll('script,style,noscript,svg').forEach((n) => n.remove());

  const title = cleanSpaces(doc.title || '');
  const bodyText = cleanSpaces(doc.body?.textContent || '');
  const clippedBody = bodyText.slice(0, MAX_PAGE_SNIPPET_CHARS);
  const prices = Array.from(new Set(bodyText.match(/\$\s?\d+(?:\.\d{1,2})?/g) || [])).slice(0, 6);

  const lines = [
    `Resolved URL: ${resolvedUrl}`,
    `Title: ${title || 'N/A'}`,
    `Price hints: ${prices.length ? prices.join(', ') : 'none found'}`,
    `Content snippet: ${clippedBody || 'N/A'}`,
  ];
  return lines.join('\n');
}

async function fetchWithTimeout(
  url: string,
  timeoutMs: number,
  signal?: AbortSignal
): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  const onAbort = () => controller.abort();
  signal?.addEventListener('abort', onAbort, { once: true });

  try {
    return await fetch(url, {
      method: 'GET',
      signal: controller.signal,
      redirect: 'follow',
      credentials: 'omit',
      cache: 'no-store',
    });
  } finally {
    clearTimeout(timeoutId);
    signal?.removeEventListener('abort', onAbort);
  }
}

async function getUrlContext(rawUrl: string, signal?: AbortSignal): Promise<UrlContext> {
  const resolvedUrl = resolveBenchmarkUrl(rawUrl);

  if (!/^https?:\/\//i.test(resolvedUrl)) {
    return {
      rawUrl,
      resolvedUrl,
      summary: fallbackSummaryFromUrl(rawUrl, resolvedUrl),
      source: 'fallback',
    };
  }

  try {
    const response = await fetchWithTimeout(resolvedUrl, URL_FETCH_TIMEOUT_MS, signal);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const html = await response.text();
    return {
      rawUrl,
      resolvedUrl: response.url || resolvedUrl,
      summary: summarizeHtml(html, response.url || resolvedUrl),
      source: 'fetched',
    };
  } catch {
    return {
      rawUrl,
      resolvedUrl,
      summary: fallbackSummaryFromUrl(rawUrl, resolvedUrl),
      source: 'fallback',
    };
  }
}

async function buildSharedPrefixWithPageContext(
  task: BenchmarkTask,
  cfg: PrefixConfig,
  urlContextCache: Map<string, Promise<UrlContext>>,
  signal?: AbortSignal
): Promise<string> {
  const basePrefix = buildSharedPrefix(task, cfg);

  const rawUrls: string[] = [];
  if (cfg.includeStartUrl && task.startUrl) {
    rawUrls.push(task.startUrl);
  }
  if (cfg.includeOpenUrls && task.openUrls.length) {
    rawUrls.push(...task.openUrls);
  }

  const uniqueRawUrls = Array.from(new Set(rawUrls));
  const contexts = await Promise.all(
    uniqueRawUrls.map((rawUrl) => {
      const existing = urlContextCache.get(rawUrl);
      if (existing) {
        return existing;
      }
      const next = getUrlContext(rawUrl, signal);
      urlContextCache.set(rawUrl, next);
      return next;
    })
  );

  if (!contexts.length) {
    return basePrefix;
  }

  const contextByRaw = new Map(contexts.map((c) => [c.rawUrl, c]));
  const detailChunks: string[] = [];

  if (cfg.includeStartUrl && task.startUrl) {
    const startCtx = contextByRaw.get(task.startUrl);
    if (startCtx) {
      detailChunks.push(
        [
          'START_URL_CONTEXT:',
          `SOURCE: ${startCtx.source}`,
          startCtx.summary,
        ].join('\n')
      );
    }
  }

  if (cfg.includeOpenUrls && task.openUrls.length) {
    const openTabChunks = task.openUrls
      .map((rawUrl, idx) => {
        const c = contextByRaw.get(rawUrl);
        if (!c) {
          return '';
        }
        return [
          `TAB_${idx + 1}_CONTEXT:`,
          `SOURCE: ${c.source}`,
          c.summary,
        ].join('\n');
      })
      .filter(Boolean);
    if (openTabChunks.length) {
      detailChunks.push(['OPEN_TABS_CONTEXT:', ...openTabChunks].join('\n\n'));
    }
  }

  return [basePrefix, ...detailChunks].filter(Boolean).join('\n\n');
}

function mean(values: number[]): number {
  if (!values.length) {
    return 0;
  }
  return values.reduce((acc, v) => acc + v, 0) / values.length;
}

function finiteMean(values: number[]): number {
  const finite = values.filter((v) => Number.isFinite(v));
  if (!finite.length) {
    return 0;
  }
  return mean(finite);
}

function safePercentGain(base: number, target: number): number {
  if (!Number.isFinite(base) || !Number.isFinite(target)) {
    return 0;
  }
  const EPS = 1e-3;
  if (Math.abs(base) < EPS) {
    return 0;
  }
  return ((target - base) / base) * 100;
}

function isFiniteMetric(m: PerfMetrics): boolean {
  return (
    Number.isFinite(m.ttftMs)
    && Number.isFinite(m.tokensPerSecond)
    && Number.isFinite(m.outputTokens)
    && Number.isFinite(m.latencyMs)
  );
}

function isLikelyShortAnswerTask(task: BenchmarkTask): boolean {
  const intent = task.intent.toLowerCase();
  const shortIntentPatterns = [
    /single integer/,
    /only return/,
    /no extra text/,
    /yes\s*\/?\s*no/,
    /yes or no/,
    /command and its arguments/,
    /output the count/,
    /round.*%/,
  ];

  if (shortIntentPatterns.some((p) => p.test(intent))) {
    return true;
  }

  if (task.evalTypes.includes('string_match')) {
    const shortHint = task.referenceHints.some((h) => /^(yes|no|none|\d+(\.\d+)?%?)$/i.test(h.trim()));
    if (shortHint) {
      return true;
    }
  }

  return false;
}

function minValidOutputTokensForTask(task: BenchmarkTask): number {
  return isLikelyShortAnswerTask(task)
    ? MIN_VALID_OUTPUT_TOKENS_SHORT_TASK
    : MIN_VALID_OUTPUT_TOKENS;
}

function sanitizeGeneratedAnswer(rawText: string): string {
  let out = rawText.replace(/\u0000/g, '');
  out = out.replace(/^\s*CONTEXT_READY\b[:\-\s]*/i, '');

  const finalAnswerMatch = out.match(/FINAL_ANSWER\s*:\s*([\s\S]*)$/i);
  if (finalAnswerMatch?.[1]) {
    out = finalAnswerMatch[1];
  }

  return cleanSpaces(out);
}

function collectInvalidReasons(mode: 'A' | 'B', task: BenchmarkTask, m: PerfMetrics): string[] {
  const reasons: string[] = [];
  if (!isFiniteMetric(m)) {
    reasons.push(`${mode}: non_finite_metric`);
    return reasons;
  }
  const minOutputTokens = minValidOutputTokensForTask(task);
  if (m.outputTokens < minOutputTokens) {
    reasons.push(`${mode}: low_output_tokens(${m.outputTokens})`);
  }
  if (m.outputTokens > 0 && m.tokensPerSecond <= MIN_VALID_TPS) {
    reasons.push(`${mode}: near_zero_tps(${m.tokensPerSecond.toFixed(6)})`);
  }
  if (m.latencyMs + 1e-6 < m.ttftMs) {
    reasons.push(`${mode}: latency_lt_ttft`);
  }
  return reasons;
}

function averageMetrics(rows: PerfMetrics[]): PerfMetrics {
  return {
    ttftMs: mean(rows.map((r) => r.ttftMs)),
    tokensPerSecond: mean(rows.map((r) => r.tokensPerSecond)),
    outputTokens: mean(rows.map((r) => r.outputTokens)),
    sharedPrefixChars: mean(rows.map((r) => r.sharedPrefixChars)),
    nReused: mean(rows.map((r) => r.nReused)),
    latencyMs: mean(rows.map((r) => r.latencyMs)),
  };
}

type TaskInferenceResult = {
  metrics: PerfMetrics;
  outputPreview: string;
};

function toOutputPreview(text: string): string {
  const normalized = cleanSpaces(text);
  if (!normalized) {
    return '(empty)';
  }
  if (normalized.length <= MAX_OUTPUT_PREVIEW_CHARS) {
    return normalized;
  }
  return `${normalized.slice(0, MAX_OUTPUT_PREVIEW_CHARS)}...`;
}

function assertNotAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw new DOMException('Aborted', 'AbortError');
  }
}

async function collectStreamMetrics(
  stream: AsyncIterable<{ piece: Uint8Array; currentText: string }>,
  startedAt: number,
  signal?: AbortSignal
): Promise<{ text: string; ttftMs: number; latencyMs: number }> {
  let text = '';
  let firstTokenAt = 0;

  for await (const chunk of stream) {
    assertNotAborted(signal);
    if (!firstTokenAt) {
      firstTokenAt = performance.now();
    }
    text = chunk.currentText;
  }

  const doneAt = performance.now();
  return {
    text,
    ttftMs: Math.max((firstTokenAt || doneAt) - startedAt, 0),
    latencyMs: Math.max(doneAt - startedAt, 0),
  };
}

async function measureA(
  wllama: Wllama,
  task: BenchmarkTask,
  cfg: PrefixConfig,
  urlContextCache: Map<string, Promise<UrlContext>>,
  signal?: AbortSignal
): Promise<TaskInferenceResult> {
  assertNotAborted(signal);
  const sharedPrefix = await buildSharedPrefixWithPageContext(task, cfg, urlContextCache, signal);
  const userPrompt = `${sharedPrefix}\n\n${buildTaskPrompt(task)}`;
  const messages = [
    {
      role: 'user' as const,
      content: userPrompt,
    },
  ];

  await wllama.resetPerfContext();
  const startedAt = performance.now();
  const stream = (await wllama.createChatCompletion(messages, {
    stream: true,
    useCache: false,
    nPredict: cfg.nPredict,
    sampling: SAMPLING,
    abortSignal: signal,
  })) as AsyncIterable<{ piece: Uint8Array; currentText: string }>;

  const streamRes = await collectStreamMetrics(stream, startedAt, signal);
  const answerText = sanitizeGeneratedAnswer(streamRes.text);
  const perf = await wllama.getPerfContext();
  const outTokens = (await wllama.tokenize(answerText, true)).length;
  const decodeSec = Math.max((streamRes.latencyMs - streamRes.ttftMs) / 1000, 1e-6);

  return {
    metrics: {
      ttftMs: streamRes.ttftMs,
      tokensPerSecond: outTokens / decodeSec,
      outputTokens: outTokens,
      sharedPrefixChars: sharedPrefix.length,
      nReused: perf.n_reused,
      latencyMs: streamRes.latencyMs,
    },
    outputPreview: toOutputPreview(answerText),
  };
}

async function warmPrefixNodes(
  wllama: Wllama,
  tasks: BenchmarkTask[],
  cfg: PrefixConfig,
  urlContextCache: Map<string, Promise<UrlContext>>,
  onProgress?: (p: ABProgress) => void,
  signal?: AbortSignal
): Promise<Map<string, number>> {
  const state = await wllama.chatGetState();
  const rootId = state.rootId;
  const keyToNode = new Map<string, number>();
  const uniqueKeys = Array.from(new Set(tasks.map((t) => buildPrefixKey(t, cfg))));

  for (let i = 0; i < uniqueKeys.length; i += 1) {
    assertNotAborted(signal);
    const key = uniqueKeys[i];
    const sampleTask = tasks.find((t) => buildPrefixKey(t, cfg) === key);
    if (!sampleTask) {
      continue;
    }
    const sharedPrefix = await buildSharedPrefixWithPageContext(
      sampleTask,
      cfg,
      urlContextCache,
      signal
    );
    const anchorPrompt = buildPrefixAnchorPrompt(sharedPrefix);
    const res = await wllama.chatFromNode(rootId, anchorPrompt, {
      nPredict: 1,
      sampling: SAMPLING,
      abortSignal: signal,
    });
    keyToNode.set(key, res.nodeId);

    onProgress?.({
      modelId: '',
      mode: 'warmup',
      phase: 'warming-prefix-nodes',
      current: i + 1,
      total: uniqueKeys.length,
      taskId: sampleTask.id,
    });
  }

  return keyToNode;
}

async function measureB(
  wllama: Wllama,
  task: BenchmarkTask,
  cfg: PrefixConfig,
  keyToNode: Map<string, number>,
  urlContextCache: Map<string, Promise<UrlContext>>,
  signal?: AbortSignal
): Promise<TaskInferenceResult> {
  assertNotAborted(signal);
  const key = buildPrefixKey(task, cfg);
  const sharedPrefix = await buildSharedPrefixWithPageContext(task, cfg, urlContextCache, signal);

  const state = await wllama.chatGetState();
  const rootId = state.rootId;
  const parentNodeId = keyToNode.get(key) ?? rootId;

  await wllama.resetPerfContext();
  const startedAt = performance.now();
  let firstTokenAt = 0;
  const result = await wllama.chatFromNode(parentNodeId, buildTaskPrompt(task), {
    nPredict: cfg.nPredict,
    sampling: SAMPLING,
    useCache: true,
    abortSignal: signal,
    onChunk: () => {
      if (!firstTokenAt) {
        firstTokenAt = performance.now();
      }
    },
  });
  const doneAt = performance.now();

  const answerText = sanitizeGeneratedAnswer(result.assistantText);
  const perf = await wllama.getPerfContext();
  const outTokens = (await wllama.tokenize(answerText, true)).length;
  const ttftMs = Math.max((firstTokenAt || doneAt) - startedAt, 0);
  const latencyMs = Math.max(doneAt - startedAt, 0);
  const decodeSec = Math.max((latencyMs - ttftMs) / 1000, 1e-6);

  return {
    metrics: {
      ttftMs,
      tokensPerSecond: outTokens / decodeSec,
      outputTokens: outTokens,
      sharedPrefixChars: sharedPrefix.length,
      nReused: perf.n_reused,
      latencyMs,
    },
    outputPreview: toOutputPreview(answerText),
  };
}

export async function runABExperimentWithWllama(
  modelId: string,
  modelUrl: string,
  tasks: BenchmarkTask[],
  cfg: PrefixConfig,
  onProgress?: (p: ABProgress) => void,
  signal?: AbortSignal,
  onTaskOutput?: (event: ABTaskOutputEvent) => void
): Promise<ABExperimentResult> {
  assertNotAborted(signal);
  const wllama = new Wllama(WLLAMA_CONFIG_PATHS, { preferWebGPU: true, noPerf: false });
  const urlContextCache = new Map<string, Promise<UrlContext>>();

  try {
    onProgress?.({
      modelId,
      mode: 'A',
      phase: 'loading-model',
      current: 0,
      total: tasks.length,
    });

    await wllama.loadModelFromUrl(modelUrl, {
      useCache: true,
      n_ctx: cfg.nCtx,
      n_batch: cfg.nBatch,
      n_seq_max: 1,
      kv_unified: true,
    });

    const aRuns: TaskInferenceResult[] = [];
    for (let i = 0; i < tasks.length; i += 1) {
      assertNotAborted(signal);
      onProgress?.({
        modelId,
        mode: 'A',
        phase: 'running-task',
        current: i + 1,
        total: tasks.length,
        taskId: tasks[i].id,
      });
      const run = await measureA(wllama, tasks[i], cfg, urlContextCache, signal);
      aRuns.push(run);
      onTaskOutput?.({
        modelId,
        taskId: tasks[i].id,
        dataset: tasks[i].dataset,
        mode: 'A',
        outputTokens: run.metrics.outputTokens,
        outputPreview: run.outputPreview,
      });
    }

    const bRuns: TaskInferenceResult[] = new Array(tasks.length);
    for (let i = 0; i < tasks.length; i += 1) {
      assertNotAborted(signal);
      onProgress?.({
        modelId,
        mode: 'B',
        phase: 'running-task',
        current: i + 1,
        total: tasks.length,
        taskId: tasks[i].id,
      });

      await wllama.chatReset();
      await wllama.chatSessionInit(cfg.memoryCapBytes);

      const keyToNode = await warmPrefixNodes(
        wllama,
        [tasks[i]],
        cfg,
        urlContextCache,
        (p) => onProgress?.({ ...p, modelId }),
        signal
      );

      const run = await measureB(wllama, tasks[i], cfg, keyToNode, urlContextCache, signal);
      bRuns[i] = run;
      onTaskOutput?.({
        modelId,
        taskId: tasks[i].id,
        dataset: tasks[i].dataset,
        mode: 'B',
        outputTokens: run.metrics.outputTokens,
        outputPreview: run.outputPreview,
      });
      await wllama.chatReset();
    }

    const aMetrics = aRuns.map((r) => r.metrics);
    const bMetrics = bRuns.map((r) => r.metrics);

    const deltas: ABTaskDelta[] = tasks.map((task, i) => {
      const a = aMetrics[i];
      const b = bMetrics[i];
      const ttftGainPct = safePercentGain(a.ttftMs, b.ttftMs) * -1;
      const tpsGainPct = safePercentGain(a.tokensPerSecond, b.tokensPerSecond);
      const invalidReasons = [
        ...collectInvalidReasons('A', task, a),
        ...collectInvalidReasons('B', task, b),
      ];
      return {
        task,
        a,
        b,
        aOutputPreview: aRuns[i].outputPreview,
        bOutputPreview: bRuns[i].outputPreview,
        ttftGainPct,
        tpsGainPct,
        isValid: invalidReasons.length === 0,
        invalidReasons,
      };
    });

    const invalidTasks = deltas
      .filter((d) => !d.isValid)
      .map((d) => ({
        taskId: d.task.id,
        dataset: d.task.dataset,
        reasons: d.invalidReasons,
        aOutputPreview: d.aOutputPreview,
        bOutputPreview: d.bOutputPreview,
      }));

    onProgress?.({
      modelId,
      mode: 'B',
      phase: 'done',
      current: tasks.length,
      total: tasks.length,
    });

    return {
      modelId,
      modelPath: modelUrl,
      totalTasks: tasks.length,
      avgA: averageMetrics(aMetrics),
      avgB: averageMetrics(bMetrics),
      avgTtftGainPct: finiteMean(deltas.map((d) => d.ttftGainPct)),
      avgTpsGainPct: finiteMean(deltas.map((d) => d.tpsGainPct)),
      validTaskCount: deltas.length - invalidTasks.length,
      invalidTaskCount: invalidTasks.length,
      invalidTaskIds: invalidTasks.map((t) => t.taskId),
      invalidTasks,
      deltas,
    };
  } finally {
    await wllama.exit();
  }
}
