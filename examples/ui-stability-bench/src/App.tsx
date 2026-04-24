import { useEffect, useMemo, useRef, useState } from 'react';
import { Wllama } from '@wllama/wllama';
import wllamaSingle from '@wllama/wllama/src/single-thread/wllama.wasm?url';
import wllamaMulti from '@wllama/wllama/src/multi-thread/wllama.wasm?url';
import { WebGpuUiStressRenderer, type RendererLiveStats, type RendererWindowStats } from './uiRenderer';

const MODEL_BASE_DIR = '/Users/rambo/Desktop/wllama-webgpu/examples/ui-stability-bench/model/';
const MODEL_FILE = 'Llama-3.2-1B-Instruct-Q4_0.gguf';

const WLLAMA_PATHS = {
  'single-thread/wllama.wasm': wllamaSingle,
  'multi-thread/wllama.wasm': wllamaMulti,
};

const NO_TECH_COLOR = '#ff9b73';
const FULL_TECH_COLOR = '#33d8b0';

type ModeName = 'ui-only' | 'no-tech' | 'full-tech';

type BenchConfig = {
  modelUrl: string;
  nCtx: number;
  nBatch: number;
  requestCount: number;
  outputTokens: number;
  uiInstanceCount: number;
  sharedPrefixRepeats: number;
  fullTechQueueMaxPending: number;
  fullTechSliceTokenBudget: number;
  fullTechPrefillSliceMaxMs: number;
  fullTechWarmupRequests: number;
  fullTechSampleWindow: number;
  treeMemoryCapMB: number;
  scanPointCount: number;
  scanMinMultiplier: number;
  scanMaxMultiplier: number;
};

type RequestResult = {
  id: number;
  ok: boolean;
  reason: 'completed' | 'timeout' | 'aborted' | 'error';
  latencyMs: number;
  ttftMs: number;
  tokenCount: number;
  tokensPerSecond: number;
  message?: string;
};

type PhaseReport = {
  mode: ModeName;
  requestCount: number;
  completedCount: number;
  failedCount: number;
  completionRatePct: number;
  avgLatencyMs: number;
  medianLatencyMs: number;
  p95LatencyMs: number;
  maxLatencyMs: number;
  avgTtftMs: number;
  avgTokensPerSecond: number;
  wallClockMs: number;
  uiStats: RendererWindowStats;
  failuresByReason: Record<string, number>;
  requests: RequestResult[];
  runtimeDebug?: unknown;
};

type EnvironmentSnapshot = {
  href: string;
  userAgent: string;
  language: string;
  hardwareConcurrency: number;
};

type BenchReport = {
  createdAt: string;
  config: BenchConfig;
  environment: EnvironmentSnapshot;
  phases: PhaseReport[];
};

type PressurePoint = {
  level: number;
  multiplier: number;
  pressureScore: number;
  config: BenchConfig;
};

type ScanPointReport = PressurePoint & {
  noTech: PhaseReport;
  fullTech: PhaseReport;
};

type ScanReport = {
  createdAt: string;
  baseConfig: BenchConfig;
  environment: EnvironmentSnapshot;
  points: ScanPointReport[];
};

type ChartPoint = {
  pressureScore: number;
  label: string;
  noTech: number;
  fullTech: number;
};

type LineChartProps = {
  title: string;
  subtitle: string;
  yLabel: string;
  points: ChartPoint[];
  valueFormatter: (value: number) => string;
  yMin?: number;
  yMax?: number;
};

const DEFAULT_CONFIG: BenchConfig = {
  modelUrl: `${window.location.origin}/@fs${encodeURI(`${MODEL_BASE_DIR}/${MODEL_FILE}`)}`,
  nCtx: 16384,
  nBatch: 512,
  requestCount: 10,
  outputTokens: 16,
  uiInstanceCount: 90000,
  sharedPrefixRepeats: 48,
  fullTechQueueMaxPending: 64,
  fullTechSliceTokenBudget: 32,
  fullTechPrefillSliceMaxMs: 100,
  fullTechWarmupRequests: 4,
  fullTechSampleWindow: 64,
  treeMemoryCapMB: 2048,
  scanPointCount: 5,
  scanMinMultiplier: 1,
  scanMaxMultiplier: 1.6,
};

function avg(values: number[]): number {
  if (!values.length) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function percentile(values: number[], p: number): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.floor((sorted.length - 1) * p)));
  return sorted[index];
}

function lerp(start: number, end: number, t: number): number {
  return start + (end - start) * t;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function fmtMs(value: number): string {
  return `${value.toFixed(1)} ms`;
}

function fmtPct(value: number): string {
  return `${value.toFixed(1)}%`;
}

function fmtTps(value: number): string {
  return value.toFixed(2);
}

function downloadText(filename: string, text: string): void {
  const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function downloadJson(filename: string, data: unknown): void {
  downloadText(filename, JSON.stringify(data, null, 2));
}

function createLogger(appendLog: (text: string) => void) {
  return {
    debug: (...args: unknown[]) => {
      const text = args.map((x) => (typeof x === 'string' ? x : JSON.stringify(x))).join(' ');
      if (text.includes('[EngineChatTrace]') || text.includes('@@ERROR@@') || text.includes('@@WARN@@')) {
        appendLog(text);
      }
    },
    log: () => undefined,
    warn: (...args: unknown[]) => {
      appendLog(args.map((x) => (typeof x === 'string' ? x : JSON.stringify(x))).join(' '));
    },
    error: (...args: unknown[]) => {
      appendLog(args.map((x) => (typeof x === 'string' ? x : JSON.stringify(x))).join(' '));
    },
  };
}

async function safeExit(runtime: Wllama | null): Promise<void> {
  if (!runtime) return;
  try {
    await runtime.exit();
  } catch {
    // Best-effort cleanup.
  }
}

async function getRuntimeDebug(runtime: Wllama | null): Promise<unknown | null> {
  if (!runtime) return null;
  try {
    return await (runtime as unknown as { _getDebugInfo: () => Promise<unknown> })._getDebugInfo();
  } catch (err) {
    return {
      error: err instanceof Error ? err.message : String(err),
    };
  }
}

function classifyError(err: unknown): RequestResult['reason'] {
  const msg = err instanceof Error ? err.message : String(err);
  if (/abort/i.test(msg)) return 'aborted';
  if (/timeout/i.test(msg)) return 'timeout';
  return 'error';
}

function getEnvironmentSnapshot(): EnvironmentSnapshot {
  return {
    href: window.location.href,
    userAgent: navigator.userAgent,
    language: navigator.language,
    hardwareConcurrency: navigator.hardwareConcurrency,
  };
}

function buildSharedRules(repeats: number): string {
  const section = [
    'Benchmark policy block.',
    'You are evaluating a browser app that runs UI rendering and local inference on the same GPU.',
    'Treat smooth camera motion, stable chart animation, low input latency, and predictable completion time as top priorities.',
    'Prefer short concrete phrasing.',
    'Never use markdown tables.',
    'Always end with a line in the format "Final Answer: <label>".',
    'When uncertain, prefer the safer diagnosis for interactivity.',
  ].join(' ');
  const parts: string[] = [];
  for (let i = 0; i < repeats; i += 1) {
    parts.push(
      `Rule Block ${i + 1}: ${section} ` +
      `Signals: frame pacing, queue buildup, burst latency, thermal pressure, cache locality, prompt reuse, recoverability.`
    );
  }
  return parts.join('\n\n');
}

function buildScenarioPrompt(index: number, targetTokens: number): string {
  const cases = [
    'A note-taking web app drops frames when background summarization spikes.',
    'A CAD viewer shares GPU time with local inference and stalls camera motion.',
    'A dashboard mixes streaming charts with on-device report generation.',
    'An educational app animates particles while answering tutoring prompts.',
    'A local coding assistant re-renders syntax highlights while generating explanations.',
  ];
  const scenario = cases[index % cases.length];
  const wordBudget = Math.max(18, Math.round(targetTokens * 1.2));
  return [
    `Scenario ${index + 1}: ${scenario}`,
    `Load level: ${(index % 5) + 1}/5.`,
    'Observed symptoms: animation stutter, delayed first token, and queue buildup under bursty load.',
    'Return exactly 2 short numbered observations.',
    'Then return exactly 1 short mitigation bullet.',
    `Keep the whole answer under ${wordBudget} words.`,
    'Choose one final label from: Stable, Degraded, Critical, Recovered.',
  ].join('\n');
}

function buildDirectPrompt(sharedRules: string, index: number, targetTokens: number): string {
  return [
    'Benchmark Instructions:',
    sharedRules,
    '',
    'New Request:',
    buildScenarioPrompt(index, targetTokens),
  ].join('\n');
}

async function createRuntime(
  config: BenchConfig,
  appendLog: (text: string) => void,
  mode: 'no-tech' | 'full-tech'
): Promise<Wllama> {
  const runtime = new Wllama(WLLAMA_PATHS, {
    preferWebGPU: true,
    noPerf: false,
    suppressNativeLog: false,
    logger: createLogger(appendLog),
    engineChatTraceEnabled: mode === 'full-tech',
    engineChatQueueMaxPending: config.fullTechQueueMaxPending,
    engineChatSliceTokenBudget: config.fullTechSliceTokenBudget,
    engineChatPrefillSliceMaxMs: config.fullTechPrefillSliceMaxMs,
    engineChatCostWarmupRequests: config.fullTechWarmupRequests,
    engineChatCostSampleWindow: config.fullTechSampleWindow,
  });
  appendLog(`[${mode}] loading model...`);
  await runtime.loadModelFromUrl(config.modelUrl, {
    useCache: true,
    n_ctx: config.nCtx,
    n_batch: config.nBatch,
    n_seq_max: 1,
    kv_unified: true,
  });
  appendLog(`[${mode}] model loaded.`);
  return runtime;
}

async function runUiOnlyPhase(
  renderer: WebGpuUiStressRenderer,
  appendLog: (text: string) => void
): Promise<PhaseReport> {
  appendLog('[ui-only] measuring renderer without inference...');
  renderer.beginMeasurement('ui-only');
  const startedAt = performance.now();
  await new Promise((resolve) => setTimeout(resolve, 2000));
  const { stats } = renderer.endMeasurement();
  const wallClockMs = performance.now() - startedAt;
  return {
    mode: 'ui-only',
    requestCount: 0,
    completedCount: 0,
    failedCount: 0,
    completionRatePct: 100,
    avgLatencyMs: 0,
    medianLatencyMs: 0,
    p95LatencyMs: 0,
    maxLatencyMs: 0,
    avgTtftMs: 0,
    avgTokensPerSecond: 0,
    wallClockMs,
    uiStats: stats,
    failuresByReason: {},
    requests: [],
  };
}

async function runNoTechPhase(
  renderer: WebGpuUiStressRenderer,
  config: BenchConfig,
  appendLog: (text: string) => void
): Promise<PhaseReport> {
  const runtime = await createRuntime(config, appendLog, 'no-tech');
  const sharedRules = buildSharedRules(config.sharedPrefixRepeats);
  const results: RequestResult[] = [];
  const batchStartedAt = performance.now();
  renderer.beginMeasurement('no-tech');
  appendLog(`[no-tech] batchStart requestCount=${config.requestCount} dispatch=fcfs deadline=none`);

  try {
    for (let i = 0; i < config.requestCount; i += 1) {
      const prompt = buildDirectPrompt(sharedRules, i, config.outputTokens);
      const arrivalAt = batchStartedAt;
      let firstTokenAt = 0;
      try {
        const text = await runtime.createCompletion(prompt, {
          nPredict: config.outputTokens,
          useCache: false,
          sampling: { temp: 0.2, top_k: 40, top_p: 0.92 },
          onNewToken: () => {
            if (!firstTokenAt) {
              firstTokenAt = performance.now();
            }
          },
        });
        const latencyMs = performance.now() - arrivalAt;
        const ttftMs = (firstTokenAt || performance.now()) - arrivalAt;
        const tokenCount = Math.max(1, (await runtime.tokenize(text, true)).length);
        const tokensPerSecond = (tokenCount * 1000) / Math.max(1e-6, latencyMs);
        results.push({
          id: i,
          ok: true,
          reason: 'completed',
          latencyMs,
          ttftMs,
          tokenCount,
          tokensPerSecond,
        });
        appendLog(`[no-tech] req=${i} done ttft=${ttftMs.toFixed(1)}ms latency=${latencyMs.toFixed(1)}ms tokens=${tokenCount}`);
      } catch (err) {
        const reason = classifyError(err);
        results.push({
          id: i,
          ok: false,
          reason,
          latencyMs: performance.now() - arrivalAt,
          ttftMs: firstTokenAt ? firstTokenAt - arrivalAt : 0,
          tokenCount: 0,
          tokensPerSecond: 0,
          message: err instanceof Error ? err.message : String(err),
        });
        appendLog(`[no-tech] req=${i} failed reason=${reason} msg=${err instanceof Error ? err.message : String(err)}`);
      }
    }
  } finally {
    const { stats } = renderer.endMeasurement();
    const wallClockMs = performance.now() - batchStartedAt;
    const runtimeDebug = await getRuntimeDebug(runtime);
    await safeExit(runtime);
    return summarizePhase('no-tech', results, config.requestCount, wallClockMs, stats, runtimeDebug);
  }
}

async function runFullTechPhase(
  renderer: WebGpuUiStressRenderer,
  config: BenchConfig,
  appendLog: (text: string) => void
): Promise<PhaseReport> {
  const runtime = await createRuntime(config, appendLog, 'full-tech');
  const sharedRules = buildSharedRules(config.sharedPrefixRepeats);
  const results: RequestResult[] = [];
  const batchStartedAt = performance.now();
  renderer.beginMeasurement('full-tech');
  appendLog('[full-tech] initializing tree session...');
  await runtime.chatSessionInit(config.treeMemoryCapMB * 1024 * 1024, {
    enabled: true,
    l1TokenCap: 8192,
    l2TokenCap: 32768,
    l3TokenCap: 131072,
    pruneL1L2TokenThreshold: 1024,
    pruneL2L3TokenThreshold: 8192,
    replacementPolicy: 'hybrid',
  });
  const setup = await runtime.chatFromNode(0, sharedRules, {
    nPredict: 0,
  });
  const sharedNodeId = setup.nodeId;
  appendLog(`[full-tech] shared prefix ready node=${sharedNodeId}`);
  appendLog(`[full-tech] batchStart requestCount=${config.requestCount} submit=all-at-once deadline=none`);

  try {
    await Promise.all(
      Array.from({ length: config.requestCount }, async (_, i) => {
        const arrivalAt = batchStartedAt;
        let firstTokenAt = 0;
        try {
          const out = await runtime.chatFromNode(sharedNodeId, buildScenarioPrompt(i, config.outputTokens), {
            nPredict: config.outputTokens,
            sampling: { temp: 0.2, top_k: 40, top_p: 0.92 },
            onChunk: () => {
              if (!firstTokenAt) {
                firstTokenAt = performance.now();
              }
            },
          });
          const latencyMs = performance.now() - arrivalAt;
          const ttftMs = (firstTokenAt || performance.now()) - arrivalAt;
          const tokenCount = Math.max(1, (await runtime.tokenize(out.assistantText, true)).length);
          const tokensPerSecond = (tokenCount * 1000) / Math.max(1e-6, latencyMs);
          results.push({
            id: i,
            ok: true,
            reason: 'completed',
            latencyMs,
            ttftMs,
            tokenCount,
            tokensPerSecond,
          });
          appendLog(`[full-tech] req=${i} done ttft=${ttftMs.toFixed(1)}ms latency=${latencyMs.toFixed(1)}ms tokens=${tokenCount}`);
        } catch (err) {
          const reason = classifyError(err);
          results.push({
            id: i,
            ok: false,
            reason,
            latencyMs: performance.now() - arrivalAt,
            ttftMs: firstTokenAt ? firstTokenAt - arrivalAt : 0,
            tokenCount: 0,
            tokensPerSecond: 0,
            message: err instanceof Error ? err.message : String(err),
          });
          appendLog(`[full-tech] req=${i} failed reason=${reason} msg=${err instanceof Error ? err.message : String(err)}`);
        }
      })
    );
  } finally {
    const { stats } = renderer.endMeasurement();
    const wallClockMs = performance.now() - batchStartedAt;
    const runtimeDebug = await getRuntimeDebug(runtime);
    await safeExit(runtime);
    return summarizePhase('full-tech', results, config.requestCount, wallClockMs, stats, runtimeDebug);
  }
}

function summarizePhase(
  mode: ModeName,
  results: RequestResult[],
  requestCount: number,
  wallClockMs: number,
  uiStats: RendererWindowStats,
  runtimeDebug?: unknown
): PhaseReport {
  const completed = results.filter((result) => result.ok);
  const completedLatencies = completed.map((result) => result.latencyMs);
  const failuresByReason: Record<string, number> = {};
  for (const result of results) {
    if (result.ok) continue;
    failuresByReason[result.reason] = (failuresByReason[result.reason] ?? 0) + 1;
  }
  return {
    mode,
    requestCount,
    completedCount: completed.length,
    failedCount: results.length - completed.length,
    completionRatePct: requestCount > 0 ? (completed.length / requestCount) * 100 : 100,
    avgLatencyMs: avg(completedLatencies),
    medianLatencyMs: percentile(completedLatencies, 0.5),
    p95LatencyMs: percentile(completedLatencies, 0.95),
    maxLatencyMs: completedLatencies.length ? Math.max(...completedLatencies) : 0,
    avgTtftMs: avg(completed.map((result) => result.ttftMs)),
    avgTokensPerSecond: avg(completed.map((result) => result.tokensPerSecond)),
    wallClockMs,
    uiStats,
    failuresByReason,
    requests: [...results].sort((a, b) => a.id - b.id),
    runtimeDebug,
  };
}

function buildPressurePoints(baseConfig: BenchConfig): PressurePoint[] {
  const pointCount = Math.max(2, Math.floor(baseConfig.scanPointCount));
  const minMultiplier = Math.max(0.25, baseConfig.scanMinMultiplier);
  const maxMultiplier = Math.max(minMultiplier, baseConfig.scanMaxMultiplier);
  const points: PressurePoint[] = [];
  for (let i = 0; i < pointCount; i += 1) {
    const t = pointCount === 1 ? 0 : i / (pointCount - 1);
    const multiplier = lerp(minMultiplier, maxMultiplier, t);
    const nextConfig: BenchConfig = {
      ...baseConfig,
      requestCount: Math.max(1, Math.round(baseConfig.requestCount * multiplier)),
      outputTokens: Math.max(12, Math.round(baseConfig.outputTokens + (multiplier - 1) * 4)),
      uiInstanceCount: Math.max(1000, Math.round(baseConfig.uiInstanceCount * (1 + (multiplier - 1) * 0.55))),
      sharedPrefixRepeats: Math.max(8, Math.round(baseConfig.sharedPrefixRepeats * (1 + (multiplier - 1) * 0.2))),
      fullTechPrefillSliceMaxMs: Math.max(60, Math.round(baseConfig.fullTechPrefillSliceMaxMs * (1 - (multiplier - 1) * 0.2))),
    };
    const requestRatio = nextConfig.requestCount / Math.max(1, baseConfig.requestCount);
    const tokenRatio = nextConfig.outputTokens / Math.max(1, baseConfig.outputTokens);
    const uiRatio = nextConfig.uiInstanceCount / Math.max(1, baseConfig.uiInstanceCount);
    const prefixRatio = nextConfig.sharedPrefixRepeats / Math.max(1, baseConfig.sharedPrefixRepeats);
    const pressureScore = 100 * (0.4 * requestRatio + 0.2 * tokenRatio + 0.25 * uiRatio + 0.15 * prefixRatio);
    points.push({
      level: i + 1,
      multiplier,
      pressureScore,
      config: nextConfig,
    });
  }
  return points;
}

function formatPressureLabel(point: PressurePoint | ScanPointReport): string {
  return `P${point.pressureScore.toFixed(0)}`;
}

function createLinePath(
  points: ChartPoint[],
  valueSelector: (point: ChartPoint) => number,
  xMin: number,
  xMax: number,
  yMin: number,
  yMax: number,
  width: number,
  height: number,
  marginLeft: number,
  marginTop: number,
  plotWidth: number,
  plotHeight: number
): string {
  const scaleX = (x: number) => {
    if (xMax <= xMin) return marginLeft + plotWidth / 2;
    return marginLeft + ((x - xMin) / (xMax - xMin)) * plotWidth;
  };
  const scaleY = (y: number) => {
    if (yMax <= yMin) return marginTop + plotHeight / 2;
    return marginTop + plotHeight - ((y - yMin) / (yMax - yMin)) * plotHeight;
  };
  return points
    .map((point, index) => {
      const cmd = index === 0 ? 'M' : 'L';
      return `${cmd} ${scaleX(point.pressureScore).toFixed(2)} ${scaleY(valueSelector(point)).toFixed(2)}`;
    })
    .join(' ');
}

function LineChart(props: LineChartProps) {
  const { title, subtitle, yLabel, points, valueFormatter } = props;
  const width = 680;
  const height = 300;
  const marginLeft = 56;
  const marginRight = 20;
  const marginTop = 24;
  const marginBottom = 44;
  const plotWidth = width - marginLeft - marginRight;
  const plotHeight = height - marginTop - marginBottom;

  const xValues = points.map((point) => point.pressureScore);
  const yValues = points.flatMap((point) => [point.noTech, point.fullTech]);
  const xMin = xValues.length ? Math.min(...xValues) : 0;
  const xMax = xValues.length ? Math.max(...xValues) : 1;
  const inferredYMin = yValues.length ? Math.min(...yValues) : 0;
  const inferredYMax = yValues.length ? Math.max(...yValues) : 1;
  const ySpan = Math.max(1, inferredYMax - inferredYMin);
  const yMin = props.yMin ?? Math.max(0, inferredYMin - ySpan * 0.15);
  const yMax = props.yMax ?? inferredYMax + ySpan * 0.15;

  const scaleX = (x: number) => {
    if (xMax <= xMin) return marginLeft + plotWidth / 2;
    return marginLeft + ((x - xMin) / (xMax - xMin)) * plotWidth;
  };
  const scaleY = (y: number) => {
    if (yMax <= yMin) return marginTop + plotHeight / 2;
    return marginTop + plotHeight - ((y - yMin) / (yMax - yMin)) * plotHeight;
  };

  const noTechPath = createLinePath(
    points,
    (point) => point.noTech,
    xMin,
    xMax,
    yMin,
    yMax,
    width,
    height,
    marginLeft,
    marginTop,
    plotWidth,
    plotHeight
  );
  const fullTechPath = createLinePath(
    points,
    (point) => point.fullTech,
    xMin,
    xMax,
    yMin,
    yMax,
    width,
    height,
    marginLeft,
    marginTop,
    plotWidth,
    plotHeight
  );

  const gridTicks = Array.from({ length: 5 }, (_, index) => {
    const t = index / 4;
    return yMin + (yMax - yMin) * t;
  });

  if (!points.length) {
    return null;
  }

  return (
    <article className="chart-card">
      <div className="chart-head">
        <div>
          <h3>{title}</h3>
          <p>{subtitle}</p>
        </div>
        <div className="chart-legend">
          <span><i style={{ background: NO_TECH_COLOR }} />No-Tech</span>
          <span><i style={{ background: FULL_TECH_COLOR }} />Full-Tech</span>
        </div>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} className="chart-svg" role="img" aria-label={title}>
        {gridTicks.map((tick) => (
          <g key={tick}>
            <line
              x1={marginLeft}
              y1={scaleY(tick)}
              x2={width - marginRight}
              y2={scaleY(tick)}
              stroke="rgba(170, 195, 230, 0.14)"
              strokeWidth="1"
            />
            <text x={marginLeft - 8} y={scaleY(tick) + 4} textAnchor="end" className="chart-axis-text">
              {valueFormatter(tick)}
            </text>
          </g>
        ))}
        <line
          x1={marginLeft}
          y1={marginTop}
          x2={marginLeft}
          y2={height - marginBottom}
          stroke="rgba(170, 195, 230, 0.25)"
          strokeWidth="1"
        />
        <line
          x1={marginLeft}
          y1={height - marginBottom}
          x2={width - marginRight}
          y2={height - marginBottom}
          stroke="rgba(170, 195, 230, 0.25)"
          strokeWidth="1"
        />
        <path d={noTechPath} fill="none" stroke={NO_TECH_COLOR} strokeWidth="3" strokeLinecap="round" />
        <path d={fullTechPath} fill="none" stroke={FULL_TECH_COLOR} strokeWidth="3" strokeLinecap="round" />
        {points.map((point) => (
          <g key={`pt-${point.label}`}>
            <circle cx={scaleX(point.pressureScore)} cy={scaleY(point.noTech)} r="4.5" fill={NO_TECH_COLOR} />
            <circle cx={scaleX(point.pressureScore)} cy={scaleY(point.fullTech)} r="4.5" fill={FULL_TECH_COLOR} />
            <text x={scaleX(point.pressureScore)} y={height - marginBottom + 18} textAnchor="middle" className="chart-axis-text">
              {point.label}
            </text>
          </g>
        ))}
        <text x="16" y={marginTop + 8} className="chart-axis-title">{yLabel}</text>
        <text x={width - marginRight} y={height - 10} textAnchor="end" className="chart-axis-title">Pressure Score</text>
      </svg>
    </article>
  );
}

export default function App() {
  const [config, setConfig] = useState(DEFAULT_CONFIG);
  const [logs, setLogs] = useState<string[]>([]);
  const [report, setReport] = useState<BenchReport | null>(null);
  const [scanReport, setScanReport] = useState<ScanReport | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState('');
  const [statusText, setStatusText] = useState('Idle');
  const [rendererReady, setRendererReady] = useState(false);
  const [liveStats, setLiveStats] = useState<RendererLiveStats | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rendererRef = useRef<WebGpuUiStressRenderer | null>(null);
  const logsRef = useRef<string[]>([]);

  const appendLog = (text: string) => {
    const line = `[${new Date().toISOString()}] ${text}`;
    logsRef.current = [...logsRef.current, line];
    setLogs(logsRef.current.slice(-600));
  };

  useEffect(() => {
    let disposed = false;
    const boot = async () => {
      if (!canvasRef.current) return;
      try {
        const renderer = new WebGpuUiStressRenderer(canvasRef.current);
        await renderer.init({ instanceCount: config.uiInstanceCount });
        renderer.start();
        rendererRef.current = renderer;
        if (!disposed) {
          setRendererReady(true);
          setLiveStats(renderer.getLiveStats());
          appendLog('[ui] WebGPU stress renderer ready.');
        }
      } catch (err) {
        if (!disposed) {
          const message = err instanceof Error ? err.message : String(err);
          setError(message);
          appendLog(`[ui] renderer init failed: ${message}`);
        }
      }
    };
    void boot();
    return () => {
      disposed = true;
      rendererRef.current?.destroy();
      rendererRef.current = null;
    };
    // Initialize once on mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const id = window.setInterval(() => {
      const renderer = rendererRef.current;
      if (!renderer) return;
      setLiveStats(renderer.getLiveStats());
    }, 250);
    return () => window.clearInterval(id);
  }, []);

  const resetRunState = () => {
    setError('');
    setStatusText('Preparing run');
    logsRef.current = [];
    setLogs([]);
  };

  const runBench = async () => {
    const renderer = rendererRef.current;
    if (!renderer) {
      setError('Renderer is not ready yet.');
      return;
    }

    setRunning(true);
    setReport(null);
    setScanReport(null);
    resetRunState();

    try {
      renderer.setInstanceCount(config.uiInstanceCount);
      setStatusText('Running single A/B benchmark');
      appendLog(`[bench] config requestCount=${config.requestCount} outputTokens=${config.outputTokens} uiInstances=${config.uiInstanceCount} deadline=none`);

      const phases: PhaseReport[] = [];
      phases.push(await runUiOnlyPhase(renderer, appendLog));
      phases.push(await runNoTechPhase(renderer, config, appendLog));
      phases.push(await runFullTechPhase(renderer, config, appendLog));

      const nextReport: BenchReport = {
        createdAt: new Date().toISOString(),
        config,
        environment: getEnvironmentSnapshot(),
        phases,
      };
      setReport(nextReport);
      setStatusText('Single A/B benchmark complete');
      appendLog('[bench] all phases complete.');
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      setStatusText('Single A/B benchmark failed');
      appendLog(`[bench] failed: ${message}`);
    } finally {
      setRunning(false);
    }
  };

  const runPressureScan = async () => {
    const renderer = rendererRef.current;
    if (!renderer) {
      setError('Renderer is not ready yet.');
      return;
    }

    setRunning(true);
    setReport(null);
    setScanReport(null);
    resetRunState();

    try {
      const points = buildPressurePoints(config);
      const baseReport: ScanReport = {
        createdAt: new Date().toISOString(),
        baseConfig: config,
        environment: getEnvironmentSnapshot(),
        points: [],
      };
      setScanReport(baseReport);
      appendLog(`[scan] start pointCount=${points.length} minMultiplier=${config.scanMinMultiplier} maxMultiplier=${config.scanMaxMultiplier}`);

      const completedPoints: ScanPointReport[] = [];
      for (const point of points) {
        renderer.setInstanceCount(point.config.uiInstanceCount);
        setStatusText(`Scanning ${formatPressureLabel(point)} (${point.level}/${points.length})`);
        appendLog(
          `[scan] ${formatPressureLabel(point)} multiplier=${point.multiplier.toFixed(2)} requestCount=${point.config.requestCount} outputTokens=${point.config.outputTokens} uiInstances=${point.config.uiInstanceCount}`
        );

        const noTech = await runNoTechPhase(renderer, point.config, appendLog);
        const fullTech = await runFullTechPhase(renderer, point.config, appendLog);
        const pointReport: ScanPointReport = {
          ...point,
          noTech,
          fullTech,
        };
        completedPoints.push(pointReport);
        setScanReport({
          ...baseReport,
          points: [...completedPoints],
        });
        appendLog(
          `[scan] ${formatPressureLabel(point)} summary noTechFPS=${noTech.uiStats.avgFps.toFixed(1)} fullTechFPS=${fullTech.uiStats.avgFps.toFixed(1)} noTechMakespan=${(noTech.wallClockMs / 1000).toFixed(1)}s fullTechMakespan=${(fullTech.wallClockMs / 1000).toFixed(1)}s`
        );
      }

      setStatusText('Pressure scan complete');
      appendLog('[scan] all pressure points complete.');
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      setStatusText('Pressure scan failed');
      appendLog(`[scan] failed: ${message}`);
    } finally {
      setRunning(false);
    }
  };

  const summaryLines = useMemo(() => {
    if (!report) return [];
    return report.phases.map((phase) => {
      return `${phase.mode}: completion=${phase.completedCount}/${phase.requestCount} (${phase.completionRatePct.toFixed(1)}%), uiP95=${phase.uiStats.p95FrameMs.toFixed(1)}ms, uiLong=${phase.uiStats.longFrameRatioPct.toFixed(1)}%, avgLatency=${phase.avgLatencyMs.toFixed(1)}ms`;
    });
  }, [report]);

  const scanChartPoints = useMemo<ChartPoint[]>(() => {
    if (!scanReport) return [];
    return scanReport.points.map((point) => ({
      pressureScore: point.pressureScore,
      label: formatPressureLabel(point),
      noTech: point.noTech.uiStats.avgFps,
      fullTech: point.fullTech.uiStats.avgFps,
    }));
  }, [scanReport]);

  const uiP95ChartPoints = useMemo<ChartPoint[]>(() => {
    if (!scanReport) return [];
    return scanReport.points.map((point) => ({
      pressureScore: point.pressureScore,
      label: formatPressureLabel(point),
      noTech: point.noTech.uiStats.p95FrameMs,
      fullTech: point.fullTech.uiStats.p95FrameMs,
    }));
  }, [scanReport]);

  const completionChartPoints = useMemo<ChartPoint[]>(() => {
    if (!scanReport) return [];
    return scanReport.points.map((point) => ({
      pressureScore: point.pressureScore,
      label: formatPressureLabel(point),
      noTech: point.noTech.wallClockMs,
      fullTech: point.fullTech.wallClockMs,
    }));
  }, [scanReport]);

  return (
    <main className="page">
      <section className="panel hero">
        <div>
          <p className="eyebrow">New Example</p>
          <h1>UI Stability Bench</h1>
          <p className="lede">
            同页持续渲染 WebGPU UI，同时对比无技术点的 FCFS 直跑与带队列、切片、树缓存的推理路径。
            现在支持单次 A/B 运行，也支持自动扫描多档压力并生成 `FPS-Pressure` 与 `Makespan-Pressure` 曲线。
          </p>
        </div>
        <div className="hero-actions">
          <button onClick={() => void runBench()} disabled={running || !rendererReady}>
            {running ? 'Running...' : 'Run A/B Bench'}
          </button>
          <button onClick={() => void runPressureScan()} disabled={running || !rendererReady}>
            {running ? 'Running...' : 'Run Pressure Scan'}
          </button>
          <button
            onClick={() => downloadText(`ui-stability-bench-logs-${Date.now()}.log`, logsRef.current.join('\n'))}
            disabled={!logsRef.current.length}
          >
            Download Logs
          </button>
          <button
            onClick={() => downloadJson(`ui-stability-bench-report-${Date.now()}.json`, report)}
            disabled={!report}
          >
            Download Bench Report
          </button>
          <button
            onClick={() => downloadJson(`ui-stability-pressure-scan-${Date.now()}.json`, scanReport)}
            disabled={!scanReport}
          >
            Download Scan Report
          </button>
        </div>
      </section>

      <section className="panel layout">
        <div className="canvas-wrap">
          <canvas ref={canvasRef} className="gpu-canvas" width={960} height={540} />
          <div className="canvas-overlay">
            <div>Renderer: {rendererReady ? 'Ready' : 'Booting'}</div>
            <div>Status: {statusText}</div>
            <div>FPS: {liveStats ? liveStats.avgFps.toFixed(1) : '0.0'}</div>
            <div>P95 frame: {liveStats ? fmtMs(liveStats.p95FrameMs) : '0.0 ms'}</div>
            <div>Long frames: {liveStats ? fmtPct(liveStats.longFrameRatioPct) : '0.0%'}</div>
            <div>Instances: {liveStats?.instanceCount ?? config.uiInstanceCount}</div>
          </div>
        </div>

        <div className="controls">
          <h2>Controls</h2>
          <label>
            <span>Model URL</span>
            <input
              className="text-input"
              value={config.modelUrl}
              onChange={(e) => setConfig((prev) => ({ ...prev, modelUrl: e.target.value }))}
              disabled={running}
            />
          </label>
          <div className="grid2">
            <label>
              <span>n_ctx</span>
              <input
                className="text-input"
                type="number"
                value={config.nCtx}
                onChange={(e) => setConfig((prev) => ({ ...prev, nCtx: Math.max(1024, Number(e.target.value) || 8192) }))}
                disabled={running}
              />
            </label>
            <label>
              <span>n_batch</span>
              <input
                className="text-input"
                type="number"
                value={config.nBatch}
                onChange={(e) => setConfig((prev) => ({ ...prev, nBatch: Math.max(64, Number(e.target.value) || 512) }))}
                disabled={running}
              />
            </label>
            <label>
              <span>Request Count</span>
              <input
                className="text-input"
                type="number"
                value={config.requestCount}
                onChange={(e) => setConfig((prev) => ({ ...prev, requestCount: Math.max(1, Number(e.target.value) || 12) }))}
                disabled={running}
              />
            </label>
            <label>
              <span>Execution Mode</span>
              <input
                className="text-input"
                value="Run until all requests finish"
                disabled
              />
            </label>
            <label>
              <span>Output Tokens</span>
              <input
                className="text-input"
                type="number"
                value={config.outputTokens}
                onChange={(e) => setConfig((prev) => ({ ...prev, outputTokens: Math.max(16, Number(e.target.value) || 96) }))}
                disabled={running}
              />
            </label>
            <label>
              <span>UI Instances</span>
              <input
                className="text-input"
                type="number"
                value={config.uiInstanceCount}
                onChange={(e) => setConfig((prev) => ({ ...prev, uiInstanceCount: Math.max(1000, Number(e.target.value) || 22000) }))}
                disabled={running}
              />
            </label>
            <label>
              <span>Shared Prefix Repeats</span>
              <input
                className="text-input"
                type="number"
                value={config.sharedPrefixRepeats}
                onChange={(e) => setConfig((prev) => ({ ...prev, sharedPrefixRepeats: Math.max(1, Number(e.target.value) || 12) }))}
                disabled={running}
              />
            </label>
            <label>
              <span>Tree Cap (MB)</span>
              <input
                className="text-input"
                type="number"
                value={config.treeMemoryCapMB}
                onChange={(e) => setConfig((prev) => ({ ...prev, treeMemoryCapMB: Math.max(256, Number(e.target.value) || 2048) }))}
                disabled={running}
              />
            </label>
            <label>
              <span>Queue Max Pending</span>
              <input
                className="text-input"
                type="number"
                value={config.fullTechQueueMaxPending}
                onChange={(e) => setConfig((prev) => ({ ...prev, fullTechQueueMaxPending: Math.max(8, Number(e.target.value) || 64) }))}
                disabled={running}
              />
            </label>
            <label>
              <span>Slice Token Budget</span>
              <input
                className="text-input"
                type="number"
                value={config.fullTechSliceTokenBudget}
                onChange={(e) => setConfig((prev) => ({ ...prev, fullTechSliceTokenBudget: Math.max(8, Number(e.target.value) || 24) }))}
                disabled={running}
              />
            </label>
            <label>
              <span>Prefill Slice Max (ms)</span>
              <input
                className="text-input"
                type="number"
                value={config.fullTechPrefillSliceMaxMs}
                onChange={(e) => setConfig((prev) => ({ ...prev, fullTechPrefillSliceMaxMs: Math.max(50, Number(e.target.value) || 220) }))}
                disabled={running}
              />
            </label>
            <label>
              <span>Warmup Requests</span>
              <input
                className="text-input"
                type="number"
                value={config.fullTechWarmupRequests}
                onChange={(e) => setConfig((prev) => ({ ...prev, fullTechWarmupRequests: Math.max(0, Number(e.target.value) || 4) }))}
                disabled={running}
              />
            </label>
            <label>
              <span>Scan Points</span>
              <input
                className="text-input"
                type="number"
                value={config.scanPointCount}
                onChange={(e) => setConfig((prev) => ({ ...prev, scanPointCount: Math.max(2, Number(e.target.value) || 6) }))}
                disabled={running}
              />
            </label>
            <label>
              <span>Scan Min Multiplier</span>
              <input
                className="text-input"
                type="number"
                step="0.05"
                value={config.scanMinMultiplier}
                onChange={(e) => setConfig((prev) => ({ ...prev, scanMinMultiplier: Math.max(0.25, Number(e.target.value) || 0.75) }))}
                disabled={running}
              />
            </label>
            <label>
              <span>Scan Max Multiplier</span>
              <input
                className="text-input"
                type="number"
                step="0.05"
                value={config.scanMaxMultiplier}
                onChange={(e) => setConfig((prev) => ({ ...prev, scanMaxMultiplier: Math.max(prev.scanMinMultiplier, Number(e.target.value) || 2.5) }))}
                disabled={running}
              />
            </label>
          </div>
          {error ? <p className="error">{error}</p> : null}
          {summaryLines.length ? (
            <div className="summary-box">
              {summaryLines.map((line) => (
                <div key={line}>{line}</div>
              ))}
            </div>
          ) : null}
        </div>
      </section>

      {report ? (
        <section className="panel">
          <h2>Phase Summary</h2>
          <div className="phase-grid">
            {report.phases.map((phase) => (
              <article key={phase.mode} className="phase-card">
                <h3>{phase.mode}</h3>
                <div>Completion: {phase.completedCount}/{phase.requestCount} ({fmtPct(phase.completionRatePct)})</div>
                <div>Avg latency: {fmtMs(phase.avgLatencyMs)}</div>
                <div>P95 completion: {fmtMs(phase.p95LatencyMs)}</div>
                <div>Avg TTFT: {fmtMs(phase.avgTtftMs)}</div>
                <div>Avg tokens/s: {fmtTps(phase.avgTokensPerSecond)}</div>
                <div>Makespan: {fmtMs(phase.wallClockMs)}</div>
                <div>UI avg FPS: {phase.uiStats.avgFps.toFixed(1)}</div>
                <div>UI P95 frame: {fmtMs(phase.uiStats.p95FrameMs)}</div>
                <div>UI long-frame ratio: {fmtPct(phase.uiStats.longFrameRatioPct)}</div>
                <div>UI max frame: {fmtMs(phase.uiStats.maxFrameMs)}</div>
                <div>Failures: {JSON.stringify(phase.failuresByReason)}</div>
              </article>
            ))}
          </div>
        </section>
      ) : null}

      {scanReport && scanReport.points.length ? (
        <section className="panel">
          <h2>Pressure Scan</h2>
          <div className="scan-grid">
            <LineChart
              title="FPS vs Pressure"
              subtitle="Higher is better. Pressure score combines request count, output tokens, UI load, and shared-prefix size."
              yLabel="Avg FPS"
              points={scanChartPoints}
              valueFormatter={(value) => value.toFixed(0)}
            />
            <LineChart
              title="UI P95 Frame vs Pressure"
              subtitle="Lower is better. This is usually more sensitive than average FPS on 120Hz displays."
              yLabel="P95 Frame"
              points={uiP95ChartPoints}
              valueFormatter={(value) => `${value.toFixed(1)}ms`}
              yMin={0}
            />
            <LineChart
              title="Makespan vs Pressure"
              subtitle="Lower is better. Each point waits until all requests finish and plots total batch completion time."
              yLabel="Makespan"
              points={completionChartPoints}
              valueFormatter={(value) => `${(value / 1000).toFixed(1)}s`}
              yMin={0}
            />
          </div>
          <div className="scan-table-wrap">
            <table className="scan-table">
              <thead>
                <tr>
                  <th>Point</th>
                  <th>Pressure</th>
                  <th>Req</th>
                  <th>OutTok</th>
                  <th>UI Inst</th>
                  <th>Shared Rep</th>
                  <th>No-Tech FPS</th>
                  <th>Full-Tech FPS</th>
                  <th>No-Tech UI P95</th>
                  <th>Full-Tech UI P95</th>
                  <th>No-Tech Makespan</th>
                  <th>Full-Tech Makespan</th>
                  <th>No-Tech P95</th>
                  <th>Full-Tech P95</th>
                </tr>
              </thead>
              <tbody>
                {scanReport.points.map((point) => (
                  <tr key={`${point.level}-${point.pressureScore}`}>
                    <td>{point.level}</td>
                    <td>{formatPressureLabel(point)}</td>
                    <td>{point.config.requestCount}</td>
                    <td>{point.config.outputTokens}</td>
                    <td>{point.config.uiInstanceCount}</td>
                    <td>{point.config.sharedPrefixRepeats}</td>
                    <td>{point.noTech.uiStats.avgFps.toFixed(1)}</td>
                    <td>{point.fullTech.uiStats.avgFps.toFixed(1)}</td>
                    <td>{point.noTech.uiStats.p95FrameMs.toFixed(1)}ms</td>
                    <td>{point.fullTech.uiStats.p95FrameMs.toFixed(1)}ms</td>
                    <td>{(point.noTech.wallClockMs / 1000).toFixed(1)}s</td>
                    <td>{(point.fullTech.wallClockMs / 1000).toFixed(1)}s</td>
                    <td>{(point.noTech.p95LatencyMs / 1000).toFixed(1)}s</td>
                    <td>{(point.fullTech.p95LatencyMs / 1000).toFixed(1)}s</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      <section className="panel">
        <h2>Logs</h2>
        <div className="log-box">
          {logs.map((line, index) => (
            <div key={`${index}-${line}`} className="log-line">{line}</div>
          ))}
        </div>
      </section>
    </main>
  );
}
