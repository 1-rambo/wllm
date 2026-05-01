import { useMemo, useRef, useState } from 'react';
import {
  getAllLocalMmluSubjects,
  getHellaSwagLineCount,
  getLocalMmluSubjectCounts,
  loadRealHellaSwag,
  loadRealMmluFromLocal,
} from './bench/data-real';
import { runSglangStyleBench } from './bench/runner';
import type { BenchBackend, BenchConfig, BenchProgressEvent, BenchReport, BenchTarget, ExperimentRunMode, QAResult } from './bench/types';

const MODEL_BASE_DIR = '/Users/rambo/Desktop/wllama-webgpu/examples/sglang-style-mmlu-hellaswag/model/';
const MODEL_FILE = 'Llama-3.2-1B-Instruct-Q4_0.gguf';

const DEFAULT_CONFIG: BenchConfig = {
  backend: 'wllama',
  webllmModelId: 'Llama-3.2-1B-Instruct-q4f32_1-MLC',
  modelUrl: `${window.location.origin}/@fs${encodeURI(`${MODEL_BASE_DIR}/${MODEL_FILE}`)}`,
  nCtx: 8192,
  nBatch: 512,
  engineChatServiceUpperBoundMs: 30000,
  engineChatQueueMaxPending: 512,
  engineChatSliceTokenBudget: 64,
  engineChatPrefillSliceMaxMs: 1500,
  engineChatCostWarmupRequests: 8,
  engineChatCostSampleWindow: 128,
  engineChatTraceEnabled: true,
  treeBackend: 'true-tree',
  target: 'mmlu',
  experimentRunMode: 'exp1',
  mmluExperimentMode: 'exp1-sequential-once',
  randomSeed: 42,
  trueTreeMemoryCapMB: 4096,
  trueTreeTieredCacheEnabled: true,
  trueTreeTierL1TokenCap: 8192,
  trueTreeTierL2TokenCap: 32768,
  trueTreeTierL3TokenCap: 131072,
  trueTreePruneL1L2TokenThreshold: 1024,
  trueTreePruneL2L3TokenThreshold: 8192,
  trueTreeReplacementPolicy: 'hybrid',
  mmluShots: 5,
  hellaShots: 20,
  mmluEvalCount: 1,
  hellaEvalCount: 1,
  exp4Concurrency: 4,
  exp4SampleCount: 256,
  exp4OutputTokens: 96,
};

const DEFAULT_HELLA_URL = '/datasets/hellaswag/hellaswag_val.jsonl';
const DEFAULT_MMLU_SUBJECT = 'abstract_algebra';

function pct01(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

function ms(v: number): string {
  return `${v.toFixed(1)} ms`;
}

function speed(v: number): string {
  return `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`;
}

function tps(v: number): string {
  return v.toFixed(3);
}

function saveJson(name: string, data: unknown): void {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

function saveText(name: string, text: string): void {
  const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

function rowsMismatched(rows: QAResult[]): QAResult[] {
  return rows.filter((r) => r.correctFlat !== r.correctTree);
}

function escapeCsv(value: string | number | boolean | null | undefined): string {
  const text = value == null ? '' : String(value);
  if (/[",\n]/.test(text)) {
    return `"${text.replace(/"/g, '""')}"`;
  }
  return text;
}

function collectAllSampleMetrics(report: BenchReport) {
  return [report.mmlu, report.hella]
    .filter((summary): summary is NonNullable<typeof summary> => Boolean(summary))
    .flatMap((summary) => [...summary.sampleMetricsFlat, ...summary.sampleMetricsTree]);
}

function buildSampleMetricsCsv(report: BenchReport): string {
  const header = [
    'benchmark',
    'mode',
    'id',
    'subject',
    'correct',
    'visitOrdinal',
    'isRepeatVisit',
    'restoreSource',
    'hadParentRecover',
    'tierL1TokensAfter',
    'tierL2TokensAfter',
    'tierL3TokensAfter',
    'tierL1SlotsAfter',
    'tierL2SlotsAfter',
    'tierL3SlotsAfter',
    'latencyMs',
    'ttftMs',
    'tokensPerSecond',
    'tokenCount',
  ];
  const rows = collectAllSampleMetrics(report).map((metric) => ([
    metric.benchmark,
    metric.mode,
    metric.id,
    metric.subject ?? '',
    metric.correct ? 1 : 0,
    metric.visitOrdinal ?? '',
    typeof metric.isRepeatVisit === 'boolean' ? (metric.isRepeatVisit ? 1 : 0) : '',
    metric.restoreSource ?? '',
    typeof metric.hadParentRecover === 'boolean' ? (metric.hadParentRecover ? 1 : 0) : '',
    metric.tierTokensAfter?.l1 ?? '',
    metric.tierTokensAfter?.l2 ?? '',
    metric.tierTokensAfter?.l3 ?? '',
    metric.tierSlotsAfter?.l1 ?? '',
    metric.tierSlotsAfter?.l2 ?? '',
    metric.tierSlotsAfter?.l3 ?? '',
    metric.latencyMs.toFixed(4),
    metric.ttftMs.toFixed(4),
    metric.tokensPerSecond.toFixed(6),
    metric.tokenCount,
  ].map(escapeCsv).join(',')));
  return [header.join(','), ...rows].join('\n');
}

function buildCdfCsv(report: BenchReport): string {
  const rows: string[] = [['benchmark', 'metric', 'mode', 'value', 'cdf'].join(',')];
  const append = (
    benchmark: 'MMLU' | 'HellaSwag',
    metric: 'latencyMs' | 'ttftMs',
    mode: 'flat' | 'tree',
    points: Array<{ value: number; cdf: number }>
  ) => {
    for (const point of points) {
      rows.push([
        benchmark,
        metric,
        mode,
        point.value.toFixed(4),
        point.cdf.toFixed(6),
      ].map(escapeCsv).join(','));
    }
  };

  if (report.mmlu) {
    append('MMLU', 'latencyMs', 'flat', report.mmlu.latencyCdfFlat);
    append('MMLU', 'latencyMs', 'tree', report.mmlu.latencyCdfTree);
    append('MMLU', 'ttftMs', 'flat', report.mmlu.ttftCdfFlat);
    append('MMLU', 'ttftMs', 'tree', report.mmlu.ttftCdfTree);
  }
  if (report.hella) {
    append('HellaSwag', 'latencyMs', 'flat', report.hella.latencyCdfFlat);
    append('HellaSwag', 'latencyMs', 'tree', report.hella.latencyCdfTree);
    append('HellaSwag', 'ttftMs', 'flat', report.hella.ttftCdfFlat);
    append('HellaSwag', 'ttftMs', 'tree', report.hella.ttftCdfTree);
  }
  return rows.join('\n');
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

function buildSeriesStats(metrics: ReturnType<typeof collectAllSampleMetrics>) {
  const latency = metrics.map((metric) => metric.latencyMs);
  const ttft = metrics.map((metric) => metric.ttftMs);
  const tps = metrics.map((metric) => metric.tokensPerSecond);
  const correctCount = metrics.filter((metric) => metric.correct).length;
  return {
    sampleCount: metrics.length,
    accuracy: metrics.length > 0 ? correctCount / metrics.length : 0,
    avgLatencyMs: latency.length ? latency.reduce((sum, value) => sum + value, 0) / latency.length : 0,
    p50LatencyMs: quantile(latency, 0.5),
    p95LatencyMs: quantile(latency, 0.95),
    p99LatencyMs: quantile(latency, 0.99),
    avgTtftMs: ttft.length ? ttft.reduce((sum, value) => sum + value, 0) / ttft.length : 0,
    p50TtftMs: quantile(ttft, 0.5),
    p95TtftMs: quantile(ttft, 0.95),
    p99TtftMs: quantile(ttft, 0.99),
    avgTokensPerSecond: tps.length ? tps.reduce((sum, value) => sum + value, 0) / tps.length : 0,
  };
}

function stripPerQuestionAnswers(report: BenchReport): BenchReport {
  const stripSummary = (summary: BenchReport['mmlu']) => {
    if (!summary) return summary;
    return {
      ...summary,
      results: [],
    };
  };
  return {
    ...report,
    mmlu: stripSummary(report.mmlu),
    hella: stripSummary(report.hella),
  };
}

function buildAnalysisExport(
  report: BenchReport,
  context: {
    timestamp: string;
    requestedConfig: BenchConfig;
    effectiveConfig: BenchConfig;
    runtimeInputs: {
      runFullDataset: boolean;
      mmluSubject: string;
      hellaDataUrl: string;
      loadedMmluItems: number;
      loadedHellaItems: number;
    };
    summaryLines: string[];
    sampleMetricLines: string[];
    logs: string[];
  }
) {
  const sampleMetrics = collectAllSampleMetrics(report);
  const groupedEntries = Array.from(
    sampleMetrics.reduce((map, metric) => {
      const key = `${metric.benchmark}/${metric.mode}`;
      const bucket = map.get(key) ?? [];
      bucket.push(metric);
      map.set(key, bucket);
      return map;
    }, new Map<string, typeof sampleMetrics>())
  );

  return {
    timestamp: context.timestamp,
    requestedConfig: context.requestedConfig,
    effectiveConfig: context.effectiveConfig,
    runtimeInputs: context.runtimeInputs,
    summaryLines: context.summaryLines,
    sampleMetricLineCount: context.sampleMetricLines.length,
    sampleMetricLines: context.sampleMetricLines,
    sampleMetrics,
    sampleMetricSeries: groupedEntries.map(([seriesKey, metrics]) => ({
      seriesKey,
      benchmark: metrics[0]?.benchmark ?? '',
      mode: metrics[0]?.mode ?? '',
      stats: buildSeriesStats(metrics),
      samples: metrics,
    })),
    cdf: {
      mmlu: report.mmlu
        ? {
          latencyFlat: report.mmlu.latencyCdfFlat,
          latencyTree: report.mmlu.latencyCdfTree,
          ttftFlat: report.mmlu.ttftCdfFlat,
          ttftTree: report.mmlu.ttftCdfTree,
        }
        : null,
      hella: report.hella
        ? {
          latencyFlat: report.hella.latencyCdfFlat,
          latencyTree: report.hella.latencyCdfTree,
          ttftFlat: report.hella.ttftCdfFlat,
          ttftTree: report.hella.ttftCdfTree,
        }
        : null,
    },
    report: stripPerQuestionAnswers(report),
    logs: context.logs,
  };
}

function buildCdfPath(points: Array<{ value: number; cdf: number }>, width: number, height: number, padding: number): string {
  if (!points.length) return '';
  const minX = points[0]?.value ?? 0;
  const maxX = points[points.length - 1]?.value ?? minX + 1;
  const usableWidth = Math.max(1, width - padding * 2);
  const usableHeight = Math.max(1, height - padding * 2);
  return points.map((point, index) => {
    const ratioX = maxX > minX ? (point.value - minX) / (maxX - minX) : 0;
    const x = padding + ratioX * usableWidth;
    const y = height - padding - point.cdf * usableHeight;
    return `${index === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`;
  }).join(' ');
}

function CdfChart({
  title,
  flatPoints,
  treePoints,
  flatLabel,
  treeLabel,
}: {
  title: string;
  flatPoints: Array<{ value: number; cdf: number }>;
  treePoints: Array<{ value: number; cdf: number }>;
  flatLabel: string;
  treeLabel: string;
}) {
  const width = 420;
  const height = 240;
  const padding = 28;
  const flatPath = buildCdfPath(flatPoints, width, height, padding);
  const treePath = buildCdfPath(treePoints, width, height, padding);

  if (!flatPath && !treePath) {
    return null;
  }

  return (
    <div>
      <div style={{ marginBottom: 8, fontWeight: 600 }}>{title}</div>
      <svg viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', maxWidth: 420, height: 'auto', border: '1px solid #d0d7de', background: '#fff' }}>
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="#9aa4b2" strokeWidth="1" />
        <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="#9aa4b2" strokeWidth="1" />
        {flatPath ? <path d={flatPath} fill="none" stroke="#2563eb" strokeWidth="2" /> : null}
        {treePath ? <path d={treePath} fill="none" stroke="#dc2626" strokeWidth="2" /> : null}
      </svg>
      <div className="hint" style={{ marginTop: 8 }}>
        <span style={{ color: '#2563eb' }}>{flatLabel}</span>
        {' / '}
        <span style={{ color: '#dc2626' }}>{treeLabel}</span>
      </div>
    </div>
  );
}

function reportSummaryLines(report: BenchReport): string[] {
  const lines: string[] = [];
  const INT32_MAX = 2147483647;
  const isWebllmNoCache = report.config.backend === 'web-llm';
  if (report.mmlu) {
    const isMmluExp2 = report.config.mmluExperimentMode === 'exp2-random-twice';
    if (isWebllmNoCache) {
      lines.push(`[MMLU/web-llm-no-cache] acc=${(report.mmlu.accTree * 100).toFixed(2)}%`);
      lines.push(`[MMLU/web-llm-no-cache] ttft=${report.mmlu.avgTtftMsTree.toFixed(2)}ms`);
      lines.push(`[MMLU/web-llm-no-cache] latency=${report.mmlu.avgLatencyMsTree.toFixed(2)}ms`);
      lines.push(`[MMLU/web-llm-no-cache] tokens/s=${report.mmlu.avgTokensPerSecondTree.toFixed(2)}`);
    } else if (isMmluExp2) {
      lines.push(`[MMLU/Exp2] acc(tree)=${(report.mmlu.accTree * 100).toFixed(2)}%`);
      lines.push(`[MMLU/Exp2] ttft(tree)=${report.mmlu.avgTtftMsTree.toFixed(2)}ms`);
      lines.push(`[MMLU/Exp2] latency(tree)=${report.mmlu.avgLatencyMsTree.toFixed(2)}ms`);
      lines.push(`[MMLU/Exp2] tokens/s(tree)=${report.mmlu.avgTokensPerSecondTree.toFixed(2)}`);
    } else {
      lines.push(`[MMLU] acc(flat/tree)=${(report.mmlu.accFlat * 100).toFixed(2)}%/${(report.mmlu.accTree * 100).toFixed(2)}%`);
      lines.push(`[MMLU] ttft(flat/tree)=${report.mmlu.avgTtftMsFlat.toFixed(2)}ms/${report.mmlu.avgTtftMsTree.toFixed(2)}ms`);
      lines.push(`[MMLU] latency(flat/tree)=${report.mmlu.avgLatencyMsFlat.toFixed(2)}ms/${report.mmlu.avgLatencyMsTree.toFixed(2)}ms`);
      lines.push(`[MMLU] speedup=${report.mmlu.speedupPct.toFixed(2)}% ttftSpeedup=${report.mmlu.ttftSpeedupPct.toFixed(2)}%`);
    }
  }
  if (report.hella) {
    if (isWebllmNoCache) {
      lines.push(`[Hella/web-llm-no-cache] acc=${(report.hella.accTree * 100).toFixed(2)}%`);
      lines.push(`[Hella/web-llm-no-cache] ttft=${report.hella.avgTtftMsTree.toFixed(2)}ms`);
      lines.push(`[Hella/web-llm-no-cache] latency=${report.hella.avgLatencyMsTree.toFixed(2)}ms`);
      lines.push(`[Hella/web-llm-no-cache] tokens/s=${report.hella.avgTokensPerSecondTree.toFixed(2)}`);
    } else {
      lines.push(`[Hella] acc(flat/tree)=${(report.hella.accFlat * 100).toFixed(2)}%/${(report.hella.accTree * 100).toFixed(2)}%`);
      lines.push(`[Hella] ttft(flat/tree)=${report.hella.avgTtftMsFlat.toFixed(2)}ms/${report.hella.avgTtftMsTree.toFixed(2)}ms`);
      lines.push(`[Hella] latency(flat/tree)=${report.hella.avgLatencyMsFlat.toFixed(2)}ms/${report.hella.avgLatencyMsTree.toFixed(2)}ms`);
      lines.push(`[Hella] speedup=${report.hella.speedupPct.toFixed(2)}% ttftSpeedup=${report.hella.ttftSpeedupPct.toFixed(2)}%`);
    }
  }
  if (report.cacheProfile) {
    lines.push(`[Exp3] maintenance=${report.cacheProfile.maintenanceMs.toFixed(2)}ms total=${report.cacheProfile.runTotalMs.toFixed(2)}ms ratio=${report.cacheProfile.maintenancePct.toFixed(4)}%`);
    lines.push(
      `[Exp3] breakdown(sessionInit/stateRead/prefixSetup/other)=${report.cacheProfile.maintenanceBreakdownMs.sessionInitMs.toFixed(2)}/${report.cacheProfile.maintenanceBreakdownMs.stateReadMs.toFixed(2)}/${report.cacheProfile.maintenanceBreakdownMs.prefixSetupMs.toFixed(2)}/${report.cacheProfile.maintenanceBreakdownMs.otherMs.toFixed(2)}ms`
    );
    const snapshotSuffix = report.cacheProfile.snapshotTokenBytes >= INT32_MAX ? ' (clamped-int32)' : '';
    lines.push(`[Exp3] snapshotBytes=${report.cacheProfile.snapshotTokenBytes}${snapshotSuffix} tier(L1/L2/L3)=${report.cacheProfile.tierL1Tokens}/${report.cacheProfile.tierL2Tokens}/${report.cacheProfile.tierL3Tokens}`);
    lines.push(
      `[Exp3] occupancy samples=${report.cacheProfile.occupancyStats.sampleCount} avgSnapshotBytes=${report.cacheProfile.occupancyStats.avgSnapshotTokenBytes.toFixed(2)} peakSnapshotBytes=${report.cacheProfile.occupancyStats.peakSnapshotTokenBytes}`
    );
    lines.push(
      `[Exp3] occupancy avgTokens(L1/L2/L3)=${report.cacheProfile.occupancyStats.avgTierL1Tokens.toFixed(2)}/${report.cacheProfile.occupancyStats.avgTierL2Tokens.toFixed(2)}/${report.cacheProfile.occupancyStats.avgTierL3Tokens.toFixed(2)} peakTokens=${report.cacheProfile.occupancyStats.peakTierL1Tokens}/${report.cacheProfile.occupancyStats.peakTierL2Tokens}/${report.cacheProfile.occupancyStats.peakTierL3Tokens}`
    );
    lines.push(
      `[Exp3] occupancy avgSlots(L1/L2/L3)=${report.cacheProfile.occupancyStats.avgTierL1Slots.toFixed(2)}/${report.cacheProfile.occupancyStats.avgTierL2Slots.toFixed(2)}/${report.cacheProfile.occupancyStats.avgTierL3Slots.toFixed(2)} peakSlots=${report.cacheProfile.occupancyStats.peakTierL1Slots}/${report.cacheProfile.occupancyStats.peakTierL2Slots}/${report.cacheProfile.occupancyStats.peakTierL3Slots}`
    );
  }
  if (report.queueVsDirect) {
    lines.push(`[Exp4] requestCount=${report.queueVsDirect.requestCount} failed(queue/direct)=${report.queueVsDirect.failedCountQueue}/${report.queueVsDirect.failedCountDirect}`);
    lines.push(`[Exp4] ttft(queue/direct)=${report.queueVsDirect.avgTtftMsQueue.toFixed(2)}ms/${report.queueVsDirect.avgTtftMsDirect.toFixed(2)}ms`);
    lines.push(`[Exp4] latency(queue/direct)=${report.queueVsDirect.avgLatencyMsQueue.toFixed(2)}ms/${report.queueVsDirect.avgLatencyMsDirect.toFixed(2)}ms`);
    lines.push(`[Exp4] avgReqTps(queue/direct)=${report.queueVsDirect.avgTokensPerSecondQueue.toFixed(3)}/${report.queueVsDirect.avgTokensPerSecondDirect.toFixed(3)}`);
    lines.push(`[Exp4] batchTps(queue/direct)=${report.queueVsDirect.batchTokensPerSecondQueue.toFixed(3)}/${report.queueVsDirect.batchTokensPerSecondDirect.toFixed(3)} wallClockMs(queue/direct)=${report.queueVsDirect.batchWallClockMsQueue.toFixed(2)}/${report.queueVsDirect.batchWallClockMsDirect.toFixed(2)}`);
    if (report.queueVsDirect.queueEngineChat) {
      const q = report.queueVsDirect.queueEngineChat;
      lines.push(`[Exp4/QueueDiag] pendingMax=${q.maxPendingCount} overdueMax=${q.maxOverduePendingCount} overdueSamples=${q.snapshotsWithOverdue} sliceMax=${q.maxSliceCount} promptTokMax=${q.maxEstimatedPromptTokens}`);
      lines.push(`[Exp4/QueueDiag] learned(prefill/decode/rebuild)=${q.learnedPrefillCostPerTokenMs?.toFixed(2) ?? 'n/a'}/${q.learnedDecodeCostPerTokenMs?.toFixed(2) ?? 'n/a'}/${q.learnedRebuildCostPerTokenMs?.toFixed(2) ?? 'n/a'} observed=${q.costModelObservedCount}`);
    }
    if (report.queueVsDirect.directEngineChat) {
      const d = report.queueVsDirect.directEngineChat;
      lines.push(`[Exp4/DirectDiag] pendingMax=${d.maxPendingCount} overdueMax=${d.maxOverduePendingCount} sliceMax=${d.maxSliceCount} promptTokMax=${d.maxEstimatedPromptTokens}`);
      lines.push(`[Exp4/DirectDiag] learned(prefill/decode/rebuild)=${d.learnedPrefillCostPerTokenMs?.toFixed(2) ?? 'n/a'}/${d.learnedDecodeCostPerTokenMs?.toFixed(2) ?? 'n/a'}/${d.learnedRebuildCostPerTokenMs?.toFixed(2) ?? 'n/a'} observed=${d.costModelObservedCount}`);
    }
  }
  if (report.schedulingStress) {
    lines.push(
      `[Exp4] scenario=${report.schedulingStress.scenario} long/short/medium=${report.schedulingStress.scenarioConfig.longHeadCount}/${report.schedulingStress.scenarioConfig.shortTailCount}/${report.schedulingStress.scenarioConfig.mediumTailCount} prefix(seed/examples medium/long)=${report.schedulingStress.scenarioConfig.mediumPrefixSeedShots}/${report.schedulingStress.scenarioConfig.mediumPrefixExampleCount}|${report.schedulingStress.scenarioConfig.longPrefixSeedShots}/${report.schedulingStress.scenarioConfig.longPrefixExampleCount} promptBudget(medium/long)=${report.schedulingStress.scenarioConfig.mediumPromptTokenBudget}/${report.schedulingStress.scenarioConfig.longPromptTokenBudget} promptEst(medium/long)=${report.schedulingStress.scenarioConfig.estimatedMediumPromptTokens}/${report.schedulingStress.scenarioConfig.estimatedLongPromptTokens} shortStream(start/intervalMs)=${report.schedulingStress.scenarioConfig.shortStreamStartMs}/${report.schedulingStress.scenarioConfig.shortInterArrivalMs} dualBudget(scale/capMs)=${report.schedulingStress.scenarioConfig.dualQueueWaitBudgetScale}/${report.schedulingStress.scenarioConfig.dualQueueWaitBudgetCapMs}`
    );
    for (const arm of report.schedulingStress.arms) {
      const short = arm.byClass.find((row) => row.workloadClass === 'short');
      const medium = arm.byClass.find((row) => row.workloadClass === 'medium');
      lines.push(
        `[Exp4/${arm.arm}] completion=${(arm.completionRate * 100).toFixed(1)}% timeout=${(arm.timeoutRate * 100).toFixed(1)}% wait(p95)=${arm.p95WaitMs.toFixed(2)}ms ttft(p95)=${arm.p95TtftMs.toFixed(2)}ms sojourn(p95)=${arm.p95SojournMs.toFixed(2)}ms throughputReq/s=${arm.throughputReqPerSec.toFixed(3)}`
      );
      if (short) {
        lines.push(
          `[Exp4/${arm.arm}/short] wait(p95/p99)=${short.p95WaitMs.toFixed(2)}/${short.p99WaitMs.toFixed(2)}ms ttft(p95)=${short.p95TtftMs.toFixed(2)}ms sla>${short.waitSlaMs}ms=${(short.waitSlaViolationRate * 100).toFixed(1)}%`
        );
      }
      if (medium) {
        lines.push(
          `[Exp4/${arm.arm}/medium] wait(p95/p99/max)=${medium.p95WaitMs.toFixed(2)}/${medium.p99WaitMs.toFixed(2)}/${medium.maxWaitMs.toFixed(2)}ms sla>${medium.waitSlaMs}ms=${(medium.waitSlaViolationRate * 100).toFixed(1)}%`
        );
      }
    }
  }
  if (report.diagnostics) {
    lines.push(`[Diag] restarts=${report.diagnostics.runtimeRestartCount} failures(timeout/abort/disposed/other)=${report.diagnostics.timeoutFailureCount}/${report.diagnostics.abortFailureCount}/${report.diagnostics.disposedFailureCount}/${report.diagnostics.otherFailureCount}`);
    if (report.diagnostics.exp2CacheStats) {
      const s = report.diagnostics.exp2CacheStats;
      if (
        typeof s.nodeCacheAttempts === 'number'
        && typeof s.sharedNodeCacheHits === 'number'
        && typeof s.sharedNodeCacheMisses === 'number'
        && typeof s.questionNodeCacheHits === 'number'
        && typeof s.questionNodeCacheMisses === 'number'
      ) {
        const sharedHitRate = typeof s.sharedNodeCacheHitRatePct === 'number'
          ? s.sharedNodeCacheHitRatePct
          : (s.nodeCacheAttempts > 0 ? (s.sharedNodeCacheHits / s.nodeCacheAttempts) * 100 : 0);
        const questionHitRate = typeof s.questionNodeCacheHitRatePct === 'number'
          ? s.questionNodeCacheHitRatePct
          : (s.nodeCacheAttempts > 0 ? (s.questionNodeCacheHits / s.nodeCacheAttempts) * 100 : 0);
        lines.push(`[Exp2/Cache] questionHit/miss=${s.questionNodeCacheHits}/${s.questionNodeCacheMisses}(${questionHitRate.toFixed(1)}%) attempts=${s.nodeCacheAttempts}`);
        lines.push(`[Exp2/Cache] sharedHit/miss=${s.sharedNodeCacheHits}/${s.sharedNodeCacheMisses}(${sharedHitRate.toFixed(1)}%) attempts=${s.nodeCacheAttempts}`);
      }
      lines.push(`[Exp2/Internal] restoreHit=${s.restoreHits}/${s.restoreAttempts}(${s.restoreHitRatePct.toFixed(1)}%) byTier(L1/L2/L3)=${s.restoreHitsL1}/${s.restoreHitsL2}/${s.restoreHitsL3}`);
      lines.push(`[Exp2/Internal] replacements promo/demote=${s.promotions}/${s.demotions} diskR/W=${s.diskReads}/${s.diskWrites} l3Overflow=${s.l3OverflowEvents} misses=${s.restoreMisses} rebuilds=${s.restoreRebuilds}`);
      lines.push(`[Exp2/Internal] parentRecover success/attempt=${s.parentRecoverSuccesses}/${s.parentRecoverAttempts} failures=${s.parentRecoverFailures}`);
      lines.push(`[Exp2/Internal] slotAlloc noEvict/needEvict=${s.slotAllocHits}/${s.slotAllocMisses} evict(L1/L2/L3)=${s.slotEvictL1}/${s.slotEvictL2}/${s.slotEvictL3} fallbackReplay=${s.fallbackReplays}`);
    }
    if (report.diagnostics.timeoutPhaseCounts && Object.keys(report.diagnostics.timeoutPhaseCounts).length > 0) {
      const topTimeouts = Object.entries(report.diagnostics.timeoutPhaseCounts)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 3)
        .map(([phase, count]) => `${phase}:${count}`)
        .join(', ');
      lines.push(`[Diag] timeoutTopPhases=${topTimeouts}`);
    }
  }
  return lines;
}

export default function App() {
  const [cfg, setCfg] = useState(DEFAULT_CONFIG);
  const [hellaDataUrl, setHellaDataUrl] = useState(DEFAULT_HELLA_URL);
  const [mmluSubject, setMmluSubject] = useState(DEFAULT_MMLU_SUBJECT);
  const [runFullDataset, setRunFullDataset] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState('');
  const [report, setReport] = useState<BenchReport | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [progress, setProgress] = useState<BenchProgressEvent>({ current: 0, total: 0, label: 'Idle' });
  const logsRef = useRef<string[]>([]);
  const metricLinesRef = useRef<string[]>([]);
  const effectiveTarget = cfg.experimentRunMode === 'exp1'
    ? cfg.target
    : cfg.experimentRunMode === 'exp4'
      ? 'both'
      : 'mmlu';
  const isHellaOnlyMode = effectiveTarget === 'hella';
  const isExp4Mode = cfg.experimentRunMode === 'exp4';
  const progressPct = progress.total > 0
    ? Math.min(100, Math.max(0, (progress.current / progress.total) * 100))
    : 0;

  const canRun = useMemo(() => {
    const needHella = effectiveTarget === 'hella' || effectiveTarget === 'both';
    const hasBackendModel = cfg.backend === 'wllama'
      ? cfg.modelUrl.trim().length > 0
      : cfg.webllmModelId.trim().length > 0;
    return !running
      && hasBackendModel
      && (!needHella || hellaDataUrl.trim().length > 0);
  }, [running, cfg.backend, cfg.modelUrl, cfg.webllmModelId, effectiveTarget, hellaDataUrl]);

  const appendLog = (text: string) => {
    const line = `[${new Date().toISOString()}] ${text}`;
    logsRef.current = [...logsRef.current, line];
    if (/^\[Metric\/(MMLU|HellaSwag)\/.+\]/.test(text)) {
      metricLinesRef.current = [...metricLinesRef.current, line];
    }
    setLogs(logsRef.current.slice(-300));
  };

  const runBench = async () => {
    setRunning(true);
    setError('');
    logsRef.current = [];
    metricLinesRef.current = [];
    setProgress({ current: 0, total: 0, label: 'Preparing' });
    setLogs([]);
    setReport(null);

    try {
      const effectiveCfg: BenchConfig = { ...cfg };
      if (effectiveCfg.backend === 'web-llm') {
        if (effectiveCfg.experimentRunMode !== 'exp1') {
          appendLog('[Mode] web-llm backend only supports Exp1-style no-cache path; run mode forced to exp1.');
        }
        effectiveCfg.experimentRunMode = 'exp1';
        effectiveCfg.mmluExperimentMode = 'exp1-sequential-once';
      }
      if (effectiveCfg.experimentRunMode === 'exp2') {
        if (effectiveCfg.target !== 'mmlu') {
          appendLog('[Mode] Exp2 runs MMLU only; target forced to mmlu.');
        }
        effectiveCfg.target = 'mmlu';
        effectiveCfg.mmluExperimentMode = 'exp2-random-twice';
      } else if (effectiveCfg.experimentRunMode === 'exp4') {
        if (effectiveCfg.target !== 'both') {
          appendLog('[Mode] Exp4 uses mixed MMLU + HellaSwag workload; target forced to both.');
        }
        effectiveCfg.target = 'both';
        effectiveCfg.mmluExperimentMode = 'exp1-sequential-once';
      } else {
        effectiveCfg.mmluExperimentMode = 'exp1-sequential-once';
      }

      const runMmluTarget = effectiveCfg.target === 'mmlu' || effectiveCfg.target === 'both';
      const runHellaTarget = effectiveCfg.target === 'hella' || effectiveCfg.target === 'both';
      let mmluSubjectsToRun = [mmluSubject];
      let mmluCountsBySubject = new Map<string, { valCount: number; testCount: number }>();
      if (runFullDataset) {
        appendLog('Resolving full dataset counts from local files (all MMLU subjects + HellaSwag)...');
        const mmluCountsPromise = runMmluTarget
          ? Promise.all(
            getAllLocalMmluSubjects().map(async (subject) => [subject, await getLocalMmluSubjectCounts(subject)] as const)
          )
          : Promise.resolve([] as Array<readonly [string, { valCount: number; testCount: number }]>);
        const hellaCountPromise = runHellaTarget ? getHellaSwagLineCount(hellaDataUrl) : Promise.resolve(0);
        const [mmluCountPairs, hellaTotal] = await Promise.all([mmluCountsPromise, hellaCountPromise]);

        if (runMmluTarget) {
          const eligible = mmluCountPairs.filter(([, c]) => c.valCount >= cfg.mmluShots && c.testCount > 0);
          const skipped = mmluCountPairs.filter(([, c]) => c.valCount < cfg.mmluShots || c.testCount <= 0);

          mmluSubjectsToRun = eligible.map(([subject]) => subject);
          if (!mmluSubjectsToRun.length) {
            throw new Error('No local MMLU subjects found.');
          }
          mmluCountsBySubject = new Map(mmluCountPairs);
          const totalTestCount = eligible.reduce((sum, [, c]) => sum + c.testCount, 0);

          // Keep strict K-shot semantics in full mode.
          effectiveCfg.mmluShots = Math.max(0, cfg.mmluShots);
          // Per-subject full eval is handled by loader; runner will consume all loaded rows.
          effectiveCfg.mmluEvalCount = Number.MAX_SAFE_INTEGER;

          if (skipped.length) {
            appendLog(
              `Full mode: skipped ${skipped.length} MMLU subjects that cannot satisfy ${effectiveCfg.mmluShots}-shot or have no test rows.`
            );
          }

          appendLog(
            `Full mode: MMLU subjects=${mmluSubjectsToRun.length}, per-subject shots=${effectiveCfg.mmluShots}, total MMLU eval=${totalTestCount}, Hella shots=${effectiveCfg.hellaShots}, Hella eval=${effectiveCfg.hellaEvalCount}`
          );
        }
        if (runHellaTarget) {
          effectiveCfg.hellaShots = Math.max(1, Math.min(cfg.hellaShots, Math.max(1, hellaTotal - 1)));
          effectiveCfg.hellaEvalCount = Math.max(1, hellaTotal - effectiveCfg.hellaShots);
        } else {
          effectiveCfg.hellaShots = 0;
          effectiveCfg.hellaEvalCount = 0;
        }
      } else if (!runHellaTarget) {
        effectiveCfg.hellaShots = 0;
        effectiveCfg.hellaEvalCount = 0;
      }

      if (!runMmluTarget) {
        effectiveCfg.mmluEvalCount = 0;
      } else if (effectiveCfg.experimentRunMode === 'exp2' && !runFullDataset) {
        appendLog('Exp2 is using a single MMLU subject; random order is mostly intra-subject. Enable Run Full Dataset for cross-subject mixing.');
      } else if (effectiveCfg.experimentRunMode === 'exp4' && !runFullDataset) {
        const counts = await getLocalMmluSubjectCounts(mmluSubject);
        const desiredSamples = Math.max(1, effectiveCfg.exp4SampleCount);
        const autoMmluEvalCount = Math.max(4, Math.min(desiredSamples, counts.testCount));
        effectiveCfg.mmluEvalCount = autoMmluEvalCount;
        if (autoMmluEvalCount < desiredSamples) {
          appendLog(
            `[Exp4] MMLU sampleCount=${desiredSamples} exceeds available test rows=${counts.testCount} for subject=${mmluSubject}; clamped to ${autoMmluEvalCount}.`
          );
        } else {
          appendLog(
            `[Exp4] auto-set mmluEvalCount=${autoMmluEvalCount} from exp4SampleCount=${desiredSamples}.`
          );
        }
        effectiveCfg.hellaEvalCount = Math.max(4, desiredSamples);
      }

      let mmlu: Awaited<ReturnType<typeof loadRealMmluFromLocal>> = [];
      if (runMmluTarget && runFullDataset) {
        appendLog(`Loading full local MMLU set across ${mmluSubjectsToRun.length} subjects ...`);
        for (const subject of mmluSubjectsToRun) {
          const counts = mmluCountsBySubject.get(subject) ?? await getLocalMmluSubjectCounts(subject);
          const rows = await loadRealMmluFromLocal(subject, effectiveCfg.mmluShots, counts.testCount);
          mmlu.push(...rows);
          appendLog(
            `MMLU subject loaded: ${subject} shots=${Math.min(effectiveCfg.mmluShots, counts.valCount)} eval=${counts.testCount} rows=${rows.length}`
          );
        }
        appendLog(`MMLU loaded: ${mmlu.length} items`);
        if (mmlu.length === 0) {
          throw new Error('MMLU items not enough: loaded 0 rows for full dataset mode.');
        }
      } else if (runMmluTarget && effectiveCfg.mmluEvalCount > 0) {
        const mmluRequired = effectiveCfg.mmluShots + effectiveCfg.mmluEvalCount;
        appendLog(`Loading local MMLU CSV (subject=${mmluSubject}) ...`);
        mmlu = await loadRealMmluFromLocal(mmluSubject, effectiveCfg.mmluShots, effectiveCfg.mmluEvalCount);
        appendLog(`MMLU loaded: ${mmlu.length} items`);
        if (mmlu.length < mmluRequired) {
          throw new Error(`MMLU items not enough: need >= ${mmluRequired}, got ${mmlu.length}`);
        }
      } else {
        appendLog('MMLU disabled (mmluEvalCount=0). Running HellaSwag only.');
      }

      const hellaRequired = runHellaTarget ? effectiveCfg.hellaShots + effectiveCfg.hellaEvalCount : 0;
      let hella: Awaited<ReturnType<typeof loadRealHellaSwag>> = [];
      if (hellaRequired > 0) {
        appendLog(`Loading HellaSwag dataset: ${hellaDataUrl}`);
        hella = await loadRealHellaSwag(hellaDataUrl, hellaRequired);
        appendLog(`HellaSwag loaded: ${hella.length} items`);
        if (hella.length < hellaRequired) {
          throw new Error(`HellaSwag items not enough: need >= ${hellaRequired}, got ${hella.length}`);
        }
      } else {
        appendLog('HellaSwag disabled. Running MMLU only.');
      }

      const out = await runSglangStyleBench(
        effectiveCfg,
        mmlu,
        hella,
        (e) => appendLog(e.text),
        (e) => setProgress(e)
      );
      setReport(out);
      const nowIso = new Date().toISOString();
      const stamp = nowIso.replace(/[:.]/g, '-');
      const runtimeInputs = {
        runFullDataset,
        mmluSubject,
        hellaDataUrl,
        loadedMmluItems: mmlu.length,
        loadedHellaItems: hella.length,
      };
      const summaryLines = reportSummaryLines(out);
      const analysisExport = buildAnalysisExport(out, {
        timestamp: nowIso,
        requestedConfig: cfg,
        effectiveConfig: effectiveCfg,
        runtimeInputs,
        summaryLines,
        sampleMetricLines: metricLinesRef.current,
        logs: logsRef.current,
      });
      saveJson(`sglang-style-report-${Date.now()}.json`, stripPerQuestionAnswers(out));
      saveText(`sglang-style-latency-samples-${stamp}.csv`, buildSampleMetricsCsv(out));
      saveText(`sglang-style-cdf-points-${stamp}.csv`, buildCdfCsv(out));
      saveJson(`sglang-style-run-detail-${stamp}.json`, analysisExport);
      const successHeader = [
        `timestamp: ${nowIso}`,
        'status: success',
        '',
        '==== REQUESTED CONFIG ==== ',
        JSON.stringify(cfg, null, 2),
        '',
        '==== EFFECTIVE CONFIG ==== ',
        JSON.stringify(effectiveCfg, null, 2),
        '',
        '==== RUNTIME INPUTS ==== ',
        JSON.stringify(runtimeInputs, null, 2),
        '',
        '==== SUMMARY ==== ',
        ...summaryLines,
        '',
        '==== SAMPLE METRIC LINES ==== ',
        ...metricLinesRef.current,
        '',
        '==== LOGS ==== ',
      ].join('\n');
      saveText(`sglang-style-run-logs-${stamp}.log`, `${successHeader}\n${logsRef.current.join('\n')}\n`);
      appendLog(`Run detail exported: sglang-style-run-detail-${stamp}.json`);
      appendLog(`Sample latency CSV exported: sglang-style-latency-samples-${stamp}.csv`);
      appendLog(`CDF point CSV exported: sglang-style-cdf-points-${stamp}.csv`);
      appendLog(`Run logs exported: sglang-style-run-logs-${stamp}.log`);
      appendLog('Benchmark finished and report exported.');
      setProgress((prev) => ({
        current: Math.max(prev.current, prev.total),
        total: prev.total,
        label: 'Done',
      }));
    } catch (e) {
      const errText = (e as Error).message || String(e);
      setError(errText);
      appendLog(`Error: ${errText}`);

      const stamp = new Date().toISOString().replace(/[:.]/g, '-');
      const header = [
        `timestamp: ${new Date().toISOString()}`,
        `error: ${errText}`,
        `backend: ${cfg.backend}`,
        `webllmModelId: ${cfg.webllmModelId}`,
        `modelUrl: ${cfg.modelUrl}`,
        `nCtx: ${cfg.nCtx}`,
        `nBatch: ${cfg.nBatch}`,
        `engineChatServiceUpperBoundMs: ${cfg.engineChatServiceUpperBoundMs}`,
        `engineChatQueueMaxPending: ${cfg.engineChatQueueMaxPending}`,
        `engineChatSliceTokenBudget: ${cfg.engineChatSliceTokenBudget}`,
        `engineChatPrefillSliceMaxMs: ${cfg.engineChatPrefillSliceMaxMs}`,
        `engineChatCostWarmupRequests: ${cfg.engineChatCostWarmupRequests}`,
        `engineChatCostSampleWindow: ${cfg.engineChatCostSampleWindow}`,
        `engineChatTraceEnabled: ${cfg.engineChatTraceEnabled}`,
        `treeBackend: ${cfg.treeBackend}`,
        `target: ${cfg.target}`,
        `experimentRunMode: ${cfg.experimentRunMode}`,
        `mmluExperimentMode: ${cfg.mmluExperimentMode}`,
        `randomSeed: ${cfg.randomSeed}`,
        `trueTreeMemoryCapMB: ${cfg.trueTreeMemoryCapMB}`,
        `trueTreeTieredCacheEnabled: ${cfg.trueTreeTieredCacheEnabled}`,
        `trueTreeTierL1TokenCap: ${cfg.trueTreeTierL1TokenCap}`,
        `trueTreeTierL2TokenCap: ${cfg.trueTreeTierL2TokenCap}`,
        `trueTreeTierL3TokenCap: ${cfg.trueTreeTierL3TokenCap}`,
        `trueTreePruneL1L2TokenThreshold: ${cfg.trueTreePruneL1L2TokenThreshold}`,
        `trueTreePruneL2L3TokenThreshold: ${cfg.trueTreePruneL2L3TokenThreshold}`,
        `trueTreeReplacementPolicy: ${cfg.trueTreeReplacementPolicy}`,
        `mmluShots: ${cfg.mmluShots}`,
        `hellaShots: ${cfg.hellaShots}`,
        `mmluEvalCount: ${cfg.mmluEvalCount}`,
        `hellaEvalCount: ${cfg.hellaEvalCount}`,
        `exp4Concurrency: ${cfg.exp4Concurrency}`,
        `exp4SampleCount: ${cfg.exp4SampleCount}`,
        `exp4OutputTokens: ${cfg.exp4OutputTokens}`,
        `runFullDataset: ${runFullDataset}`,
        `mmluSubject: ${mmluSubject}`,
        `hellaDataUrl: ${hellaDataUrl}`,
        '',
        '==== LOGS ====',
      ].join('\n');
      const payload = `${header}\n${logsRef.current.join('\n')}\n`;
      saveText(`sglang-style-error-logs-${stamp}.log`, payload);
      saveJson(`sglang-style-error-metrics-${stamp}.json`, {
        timestamp: new Date().toISOString(),
        error: errText,
        config: {
          backend: cfg.backend,
          webllmModelId: cfg.webllmModelId,
          modelUrl: cfg.modelUrl,
          nCtx: cfg.nCtx,
          nBatch: cfg.nBatch,
          engineChatServiceUpperBoundMs: cfg.engineChatServiceUpperBoundMs,
          engineChatQueueMaxPending: cfg.engineChatQueueMaxPending,
          engineChatSliceTokenBudget: cfg.engineChatSliceTokenBudget,
          engineChatPrefillSliceMaxMs: cfg.engineChatPrefillSliceMaxMs,
          engineChatCostWarmupRequests: cfg.engineChatCostWarmupRequests,
          engineChatCostSampleWindow: cfg.engineChatCostSampleWindow,
          engineChatTraceEnabled: cfg.engineChatTraceEnabled,
          treeBackend: cfg.treeBackend,
          target: cfg.target,
          experimentRunMode: cfg.experimentRunMode,
          mmluExperimentMode: cfg.mmluExperimentMode,
          randomSeed: cfg.randomSeed,
          trueTreeMemoryCapMB: cfg.trueTreeMemoryCapMB,
          trueTreeTieredCacheEnabled: cfg.trueTreeTieredCacheEnabled,
          trueTreeTierL1TokenCap: cfg.trueTreeTierL1TokenCap,
          trueTreeTierL2TokenCap: cfg.trueTreeTierL2TokenCap,
          trueTreeTierL3TokenCap: cfg.trueTreeTierL3TokenCap,
          trueTreePruneL1L2TokenThreshold: cfg.trueTreePruneL1L2TokenThreshold,
          trueTreePruneL2L3TokenThreshold: cfg.trueTreePruneL2L3TokenThreshold,
          trueTreeReplacementPolicy: cfg.trueTreeReplacementPolicy,
          mmluShots: cfg.mmluShots,
          hellaShots: cfg.hellaShots,
          mmluEvalCount: cfg.mmluEvalCount,
          hellaEvalCount: cfg.hellaEvalCount,
          exp4Concurrency: cfg.exp4Concurrency,
          exp4SampleCount: cfg.exp4SampleCount,
          exp4OutputTokens: cfg.exp4OutputTokens,
          runFullDataset,
          mmluSubject,
          hellaDataUrl,
        },
        metricLineCount: metricLinesRef.current.length,
        metricLines: metricLinesRef.current,
      });
      appendLog(`Error logs exported: sglang-style-error-logs-${stamp}.log`);
      appendLog(`Error metrics exported: sglang-style-error-metrics-${stamp}.json`);
    } finally {
      setRunning(false);
    }
  };

  const isMmluExp2 = (report?.config.experimentRunMode ?? cfg.experimentRunMode) === 'exp2';
  const isWebllmNoCache = (report?.config.backend ?? cfg.backend) === 'web-llm';
  const mmluDiff = isMmluExp2 ? [] : rowsMismatched(report?.mmlu?.results ?? []);
  const hellaDiff = rowsMismatched(report?.hella?.results ?? []);

  return (
    <div className="page">
      <header>
        <h1>SGLang-style MMLU / HellaSwag Experiments</h1>
        <p>
          设计对齐 SGLang 思路：MMLU 用单-token 选项概率，HellaSwag 用候选 continuation 概率选择。
          同时对比 Flat（无复用）与 Tree（True-Tree + 前缀复用）以展示缓存收益。
        </p>
      </header>

      <section className="panel">
        <h2>配置</h2>
        <label>
          <span>Backend</span>
          <select
            value={cfg.backend}
            onChange={(e) => setCfg((p) => ({ ...p, backend: e.target.value as BenchBackend }))}
            disabled={running}
          >
            <option value="wllama">wllama</option>
            <option value="web-llm">web-llm (no cache)</option>
          </select>
        </label>
        <label>
          <span>Model URL</span>
          <input
            className="text-input"
            value={cfg.modelUrl}
            onChange={(e) => setCfg((p) => ({ ...p, modelUrl: e.target.value }))}
            disabled={running || cfg.backend !== 'wllama'}
          />
        </label>
        <label>
          <span>web-llm Model ID</span>
          <input
            className="text-input"
            value={cfg.webllmModelId}
            onChange={(e) => setCfg((p) => ({ ...p, webllmModelId: e.target.value }))}
            disabled={running || cfg.backend !== 'web-llm'}
          />
        </label>
        <label>
          <span>Run Full Dataset</span>
          <input
            type="checkbox"
            checked={runFullDataset}
            onChange={(e) => setRunFullDataset(e.target.checked)}
            disabled={running}
          />
        </label>
        <label>
          <span>Run Mode</span>
          <select
            value={cfg.experimentRunMode}
            onChange={(e) => setCfg((p) => ({ ...p, experimentRunMode: e.target.value as ExperimentRunMode }))}
            disabled={running}
          >
            <option value="exp1">Exp1: Flat vs Tree</option>
            <option value="exp2">Exp2: Random x2 (Tree-only)</option>
            <option value="exp4">Exp4: Head-of-Line Mixed Stress</option>
          </select>
        </label>
        <label>
          <span>Target</span>
          <select
            value={cfg.target}
            onChange={(e) => setCfg((p) => ({ ...p, target: e.target.value as BenchTarget }))}
            disabled={running || cfg.experimentRunMode !== 'exp1'}
          >
            <option value="mmlu">MMLU</option>
            <option value="hella">HellaSwag</option>
            <option value="both">MMLU + HellaSwag</option>
          </select>
        </label>
        <label>
          <span>Effective Target</span>
          <input className="text-input" value={effectiveTarget} disabled />
        </label>
        <label>
          <span>MMLU Subject (local CSV)</span>
          <input
            className="text-input"
            value={mmluSubject}
            onChange={(e) => setMmluSubject(e.target.value.trim())}
            disabled={running || isHellaOnlyMode || (cfg.mmluEvalCount === 0 && !runFullDataset)}
          />
        </label>
        <label>
          <span>HellaSwag JSONL URL</span>
          <input
            className="text-input"
            value={hellaDataUrl}
            onChange={(e) => setHellaDataUrl(e.target.value)}
            disabled={running || effectiveTarget === 'mmlu'}
          />
        </label>

        <div className="row4">
          <label>
            <span>True-Tree Memory Cap (MB)</span>
            <input
              className="text-input"
              type="number"
              value={cfg.trueTreeMemoryCapMB}
              onChange={(e) => setCfg((p) => ({
                ...p,
                trueTreeMemoryCapMB: Math.max(64, Number(e.target.value) || 1024),
              }))}
              disabled={running || cfg.treeBackend !== 'true-tree'}
            />
          </label>
          <label>
            <span>True-Tree Tiered Cache</span>
            <input
              type="checkbox"
              checked={cfg.trueTreeTieredCacheEnabled}
              onChange={(e) => setCfg((p) => ({ ...p, trueTreeTieredCacheEnabled: e.target.checked }))}
              disabled={running || cfg.treeBackend !== 'true-tree'}
            />
          </label>
        </div>

        <div className="row4">
          <label>
            <span>Tier L1 Token Cap</span>
            <input
              className="text-input"
              type="number"
              value={cfg.trueTreeTierL1TokenCap}
              onChange={(e) => setCfg((p) => ({
                ...p,
                trueTreeTierL1TokenCap: Math.max(0, Number(e.target.value) || 0),
              }))}
              disabled={running || cfg.treeBackend !== 'true-tree' || !cfg.trueTreeTieredCacheEnabled}
            />
          </label>
          <label>
            <span>Tier L2 Token Cap</span>
            <input
              className="text-input"
              type="number"
              value={cfg.trueTreeTierL2TokenCap}
              onChange={(e) => setCfg((p) => ({
                ...p,
                trueTreeTierL2TokenCap: Math.max(0, Number(e.target.value) || 0),
              }))}
              disabled={running || cfg.treeBackend !== 'true-tree' || !cfg.trueTreeTieredCacheEnabled}
            />
          </label>
          <label>
            <span>Tier L3 Token Cap</span>
            <input
              className="text-input"
              type="number"
              value={cfg.trueTreeTierL3TokenCap}
              onChange={(e) => setCfg((p) => ({
                ...p,
                trueTreeTierL3TokenCap: Math.max(0, Number(e.target.value) || 0),
              }))}
              disabled={running || cfg.treeBackend !== 'true-tree' || !cfg.trueTreeTieredCacheEnabled}
            />
          </label>
          <label>
            <span>Prune Threshold L1-&gt;L2</span>
            <input
              className="text-input"
              type="number"
              value={cfg.trueTreePruneL1L2TokenThreshold}
              onChange={(e) => setCfg((p) => ({
                ...p,
                trueTreePruneL1L2TokenThreshold: Math.max(0, Number(e.target.value) || 0),
              }))}
              disabled={running || cfg.treeBackend !== 'true-tree' || !cfg.trueTreeTieredCacheEnabled}
            />
          </label>
        </div>

        <div className="row4">
          <label>
            <span>Prune Threshold L2-&gt;L3</span>
            <input
              className="text-input"
              type="number"
              value={cfg.trueTreePruneL2L3TokenThreshold}
              onChange={(e) => setCfg((p) => ({
                ...p,
                trueTreePruneL2L3TokenThreshold: Math.max(0, Number(e.target.value) || 0),
              }))}
              disabled={running || cfg.treeBackend !== 'true-tree' || !cfg.trueTreeTieredCacheEnabled}
            />
          </label>
          <label>
            <span>Replacement Policy</span>
            <select
              value={cfg.trueTreeReplacementPolicy}
              onChange={(e) => setCfg((p) => ({
                ...p,
                trueTreeReplacementPolicy: e.target.value as BenchConfig['trueTreeReplacementPolicy'],
              }))}
              disabled={running || cfg.treeBackend !== 'true-tree' || !cfg.trueTreeTieredCacheEnabled}
            >
              <option value="hybrid">Hybrid</option>
              <option value="lru">LRU</option>
              <option value="lfu">LFU</option>
              <option value="size-only">Size-only</option>
              <option value="random">Random</option>
            </select>
          </label>
        </div>

        <div className="row4">
          <label>
            <span>n_ctx</span>
            <input
              className="text-input"
              type="number"
              value={cfg.nCtx}
              onChange={(e) => setCfg((p) => ({ ...p, nCtx: Math.max(1024, Number(e.target.value) || 8192) }))}
              disabled={running || cfg.backend !== 'wllama'}
            />
          </label>
          <label>
            <span>n_batch</span>
            <input
              className="text-input"
              type="number"
              value={cfg.nBatch}
              onChange={(e) => setCfg((p) => ({ ...p, nBatch: Math.max(64, Number(e.target.value) || 512) }))}
              disabled={running || cfg.backend !== 'wllama'}
            />
          </label>
          <label>
            <span>MMLU shots</span>
            <input
              className="text-input"
              type="number"
              value={cfg.mmluShots}
              onChange={(e) => setCfg((p) => ({ ...p, mmluShots: Math.max(0, Number(e.target.value) || 0) }))}
              disabled={running}
            />
          </label>
          <label>
            <span>Hella shots</span>
            <input
              className="text-input"
              type="number"
              value={cfg.hellaShots}
              onChange={(e) => setCfg((p) => ({ ...p, hellaShots: Math.max(1, Number(e.target.value) || 20) }))}
              disabled={running || effectiveTarget === 'mmlu'}
            />
          </label>
        </div>

        <div className="row4">
          <label>
            <span>Queue Max Pending</span>
            <input
              className="text-input"
              type="number"
              value={cfg.engineChatQueueMaxPending}
              onChange={(e) => setCfg((p) => ({ ...p, engineChatQueueMaxPending: Math.max(1, Number(e.target.value) || 128) }))}
              disabled={running || cfg.backend !== 'wllama'}
            />
          </label>
          <label>
            <span>Slice Token Budget</span>
            <input
              className="text-input"
              type="number"
              value={cfg.engineChatSliceTokenBudget}
              onChange={(e) => setCfg((p) => ({ ...p, engineChatSliceTokenBudget: Math.max(1, Number(e.target.value) || 64) }))}
              disabled={running || cfg.backend !== 'wllama'}
            />
          </label>
          <label>
            <span>Prefill Slice Max (ms)</span>
            <input
              className="text-input"
              type="number"
              value={cfg.engineChatPrefillSliceMaxMs}
              onChange={(e) => setCfg((p) => ({ ...p, engineChatPrefillSliceMaxMs: Math.max(1, Number(e.target.value) || 1500) }))}
              disabled={running || cfg.backend !== 'wllama'}
            />
          </label>
          <label>
            <span>Warmup Requests</span>
            <input
              className="text-input"
              type="number"
              value={cfg.engineChatCostWarmupRequests}
              onChange={(e) => setCfg((p) => ({ ...p, engineChatCostWarmupRequests: Math.max(0, Number(e.target.value) || 0) }))}
              disabled={running || cfg.backend !== 'wllama'}
            />
          </label>
        </div>

        <div className="row4">
          <label>
            <span>Cost Sample Window</span>
            <input
              className="text-input"
              type="number"
              value={cfg.engineChatCostSampleWindow}
              onChange={(e) => setCfg((p) => ({ ...p, engineChatCostSampleWindow: Math.max(8, Number(e.target.value) || 128) }))}
              disabled={running || cfg.backend !== 'wllama'}
            />
          </label>
          <label>
            <span>Service Upper Bound (ms)</span>
            <input
              className="text-input"
              type="number"
              value={cfg.engineChatServiceUpperBoundMs}
              onChange={(e) => setCfg((p) => ({ ...p, engineChatServiceUpperBoundMs: Math.max(1000, Number(e.target.value) || 30000) }))}
              disabled={running || cfg.backend !== 'wllama'}
            />
          </label>
          <label>
            <span>Engine Trace</span>
            <input
              type="checkbox"
              checked={cfg.engineChatTraceEnabled}
              onChange={(e) => setCfg((p) => ({ ...p, engineChatTraceEnabled: e.target.checked }))}
              disabled={running || cfg.backend !== 'wllama'}
            />
          </label>
        </div>

        <div className="row4">
          <label>
            <span>MMLU eval count</span>
            <input
              className="text-input"
              type="number"
              value={cfg.mmluEvalCount}
              onChange={(e) => setCfg((p) => ({ ...p, mmluEvalCount: Math.max(0, Number(e.target.value) || 0) }))}
              disabled={running || isExp4Mode}
            />
          </label>
          <label>
            <span>Hella eval count</span>
            <input
              className="text-input"
              type="number"
              value={cfg.hellaEvalCount}
              onChange={(e) => setCfg((p) => ({ ...p, hellaEvalCount: Math.max(1, Number(e.target.value) || 4) }))}
              disabled={running || effectiveTarget === 'mmlu'}
            />
          </label>
        </div>

        <div className="row4">
          <label>
            <span>Random Seed (MMLU Exp)</span>
            <input
              className="text-input"
              type="number"
              value={cfg.randomSeed}
              onChange={(e) => setCfg((p) => ({ ...p, randomSeed: Math.max(1, Number(e.target.value) || 42) }))}
              disabled={running}
            />
          </label>
          <label>
            <span>Exp4 Concurrency</span>
            <input
              className="text-input"
              type="number"
              value={cfg.exp4Concurrency}
              onChange={(e) => setCfg((p) => ({ ...p, exp4Concurrency: Math.max(1, Number(e.target.value) || 4) }))}
              disabled={running || !isExp4Mode}
            />
          </label>
          <label>
            <span>Exp4 Sample Count</span>
            <input
              className="text-input"
              type="number"
              value={cfg.exp4SampleCount}
              onChange={(e) => setCfg((p) => ({ ...p, exp4SampleCount: Math.max(1, Number(e.target.value) || 256) }))}
              disabled={running || !isExp4Mode}
            />
          </label>
          <label>
            <span>Exp4 Output Tokens</span>
            <input
              className="text-input"
              type="number"
              value={cfg.exp4OutputTokens}
              onChange={(e) => setCfg((p) => ({ ...p, exp4OutputTokens: Math.max(1, Number(e.target.value) || 32) }))}
              disabled={running || !isExp4Mode}
            />
          </label>
        </div>

        <div className="run-row">
          <button onClick={() => void runBench()} disabled={!canRun}>
            {running ? 'Running...' : 'Run SGLang-style Bench'}
          </button>
        </div>

        {error ? <p className="error">{error}</p> : null}
        {(running || progress.total > 0) ? (
          <div className="progress-wrap">
            <div className="progress-meta">
              <span>当前进度：第 {Math.min(progress.current, progress.total)} / {progress.total} 题</span>
              <span>{progressPct.toFixed(1)}%</span>
            </div>
            <div className="progress-track" role="progressbar" aria-valuemin={0} aria-valuemax={progress.total || 1} aria-valuenow={Math.min(progress.current, progress.total)}>
              <div className="progress-fill" style={{ width: `${progressPct}%` }} />
            </div>
            <div className="hint">{progress.label}</div>
          </div>
        ) : null}
        <div className="log-box">
          {logs.map((l, i) => (
            <div key={`${i}-${l}`} className="log-line">{l}</div>
          ))}
        </div>
      </section>

      {report ? (
        <>
          {report.mmlu ? (
            <section className="panel">
              <h2>MMLU Summary</h2>
              <div className="grid3">
                <div>{isWebllmNoCache ? 'web-llm(no-cache) acc' : 'Tree acc'}: {pct01(report.mmlu.accTree)}</div>
                <div>{isWebllmNoCache ? 'web-llm(no-cache) TTFT' : 'Tree TTFT'}: {ms(report.mmlu.avgTtftMsTree)}</div>
                <div>{isWebllmNoCache ? 'web-llm(no-cache) tokens/s' : 'Tree tokens/s'}: {tps(report.mmlu.avgTokensPerSecondTree)}</div>
                <div>{isWebllmNoCache ? 'web-llm(no-cache) avg latency' : 'Tree avg latency'}: {ms(report.mmlu.avgLatencyMsTree)}</div>
                <div>Eval count: {report.mmlu.evalCount}</div>
                {isMmluExp2 || isWebllmNoCache ? null : (
                  <>
                    <div>Flat acc: {pct01(report.mmlu.accFlat)}</div>
                    <div>Tree latency speedup: {speed(report.mmlu.speedupPct)}</div>
                    <div>Flat TTFT: {ms(report.mmlu.avgTtftMsFlat)}</div>
                    <div>Tree TTFT speedup: {speed(report.mmlu.ttftSpeedupPct)}</div>
                    <div>Flat tokens/s: {tps(report.mmlu.avgTokensPerSecondFlat)}</div>
                    <div>Tree tokens/s gain: {speed(report.mmlu.tpsGainPct)}</div>
                    <div>Flat avg latency: {ms(report.mmlu.avgLatencyMsFlat)}</div>
                  </>
                )}
              </div>
              <div style={{ marginTop: 20 }}>
                <CdfChart
                  title="MMLU Latency CDF"
                  flatPoints={report.mmlu.latencyCdfFlat}
                  treePoints={report.mmlu.latencyCdfTree}
                  flatLabel="Flat"
                  treeLabel={isWebllmNoCache ? 'web-llm(no-cache)' : 'Tree'}
                />
              </div>
            </section>
          ) : null}

          {report.hella ? (
            <section className="panel">
              <h2>HellaSwag Summary</h2>
              <div className="grid3">
                {isWebllmNoCache ? (
                  <>
                    <div>web-llm(no-cache) acc: {pct01(report.hella.accTree)}</div>
                    <div>web-llm(no-cache) TTFT: {ms(report.hella.avgTtftMsTree)}</div>
                    <div>web-llm(no-cache) tokens/s: {tps(report.hella.avgTokensPerSecondTree)}</div>
                    <div>web-llm(no-cache) avg latency: {ms(report.hella.avgLatencyMsTree)}</div>
                  </>
                ) : (
                  <>
                    <div>Flat acc: {pct01(report.hella.accFlat)}</div>
                    <div>Tree acc: {pct01(report.hella.accTree)}</div>
                    <div>Tree latency speedup: {speed(report.hella.speedupPct)}</div>
                    <div>Flat TTFT: {ms(report.hella.avgTtftMsFlat)}</div>
                    <div>Tree TTFT: {ms(report.hella.avgTtftMsTree)}</div>
                    <div>Tree TTFT speedup: {speed(report.hella.ttftSpeedupPct)}</div>
                    <div>Flat tokens/s: {tps(report.hella.avgTokensPerSecondFlat)}</div>
                    <div>Tree tokens/s: {tps(report.hella.avgTokensPerSecondTree)}</div>
                    <div>Tree tokens/s gain: {speed(report.hella.tpsGainPct)}</div>
                    <div>Flat avg latency: {ms(report.hella.avgLatencyMsFlat)}</div>
                    <div>Tree avg latency: {ms(report.hella.avgLatencyMsTree)}</div>
                  </>
                )}
                <div>Eval count: {report.hella.evalCount}</div>
              </div>
              <div style={{ marginTop: 20 }}>
                <CdfChart
                  title="HellaSwag Latency CDF"
                  flatPoints={report.hella.latencyCdfFlat}
                  treePoints={report.hella.latencyCdfTree}
                  flatLabel="Flat"
                  treeLabel={isWebllmNoCache ? 'web-llm(no-cache)' : 'Tree'}
                />
              </div>
            </section>
          ) : null}

          {report.cacheProfile ? (
            <section className="panel">
              <h2>Exp3 Cache Maintenance Profile</h2>
              <div className="grid3">
                <div>Maintenance time: {ms(report.cacheProfile.maintenanceMs)}</div>
                <div>Total run time: {ms(report.cacheProfile.runTotalMs)}</div>
                <div>Maintenance ratio: {report.cacheProfile.maintenancePct.toFixed(2)}%</div>
                <div>Session-init time: {ms(report.cacheProfile.maintenanceBreakdownMs.sessionInitMs)}</div>
                <div>State-read time: {ms(report.cacheProfile.maintenanceBreakdownMs.stateReadMs)}</div>
                <div>Prefix-setup time: {ms(report.cacheProfile.maintenanceBreakdownMs.prefixSetupMs)}</div>
                <div>Snapshot bytes: {report.cacheProfile.snapshotTokenBytes}</div>
                <div>L1 tokens: {report.cacheProfile.tierL1Tokens}</div>
                <div>L2 tokens: {report.cacheProfile.tierL2Tokens}</div>
                <div>L3 tokens: {report.cacheProfile.tierL3Tokens}</div>
                <div>Occupancy samples: {report.cacheProfile.occupancyStats.sampleCount}</div>
                <div>Avg snapshot bytes: {report.cacheProfile.occupancyStats.avgSnapshotTokenBytes.toFixed(2)}</div>
                <div>Peak snapshot bytes: {report.cacheProfile.occupancyStats.peakSnapshotTokenBytes}</div>
                <div>Avg L1/L2/L3 tokens: {report.cacheProfile.occupancyStats.avgTierL1Tokens.toFixed(1)} / {report.cacheProfile.occupancyStats.avgTierL2Tokens.toFixed(1)} / {report.cacheProfile.occupancyStats.avgTierL3Tokens.toFixed(1)}</div>
                <div>Peak L1/L2/L3 tokens: {report.cacheProfile.occupancyStats.peakTierL1Tokens} / {report.cacheProfile.occupancyStats.peakTierL2Tokens} / {report.cacheProfile.occupancyStats.peakTierL3Tokens}</div>
                <div>Avg L1/L2/L3 slots: {report.cacheProfile.occupancyStats.avgTierL1Slots.toFixed(1)} / {report.cacheProfile.occupancyStats.avgTierL2Slots.toFixed(1)} / {report.cacheProfile.occupancyStats.avgTierL3Slots.toFixed(1)}</div>
                <div>Peak L1/L2/L3 slots: {report.cacheProfile.occupancyStats.peakTierL1Slots} / {report.cacheProfile.occupancyStats.peakTierL2Slots} / {report.cacheProfile.occupancyStats.peakTierL3Slots}</div>
              </div>
            </section>
          ) : null}

          {report.queueVsDirect ? (
            <section className="panel">
              <h2>Exp4 Queue vs Direct</h2>
              <div className="grid3">
                <div>Request count: {report.queueVsDirect.requestCount}</div>
                <div>Queue failed: {report.queueVsDirect.failedCountQueue}</div>
                <div>Direct failed: {report.queueVsDirect.failedCountDirect}</div>
                <div>Queue avg TTFT: {ms(report.queueVsDirect.avgTtftMsQueue)}</div>
                <div>Direct avg TTFT: {ms(report.queueVsDirect.avgTtftMsDirect)}</div>
                <div>Queue avg latency: {ms(report.queueVsDirect.avgLatencyMsQueue)}</div>
                <div>Direct avg latency: {ms(report.queueVsDirect.avgLatencyMsDirect)}</div>
                <div>Queue avg request tokens/s: {tps(report.queueVsDirect.avgTokensPerSecondQueue)}</div>
                <div>Direct avg request tokens/s: {tps(report.queueVsDirect.avgTokensPerSecondDirect)}</div>
                <div>Queue batch tokens/s: {tps(report.queueVsDirect.batchTokensPerSecondQueue)}</div>
                <div>Direct batch tokens/s: {tps(report.queueVsDirect.batchTokensPerSecondDirect)}</div>
                <div>Queue batch wall-clock: {ms(report.queueVsDirect.batchWallClockMsQueue)}</div>
                <div>Direct batch wall-clock: {ms(report.queueVsDirect.batchWallClockMsDirect)}</div>
                {report.queueVsDirect.queueEngineChat ? (
                  <>
                    <div>Queue pending max: {report.queueVsDirect.queueEngineChat.maxPendingCount}</div>
                    <div>Queue overdue max: {report.queueVsDirect.queueEngineChat.maxOverduePendingCount}</div>
                    <div>Queue slice max: {report.queueVsDirect.queueEngineChat.maxSliceCount}</div>
                    <div>Queue prompt tokens max: {report.queueVsDirect.queueEngineChat.maxEstimatedPromptTokens}</div>
                    <div>Queue cost obs count: {report.queueVsDirect.queueEngineChat.costModelObservedCount}</div>
                    <div>Queue learned prefill/decode: {report.queueVsDirect.queueEngineChat.learnedPrefillCostPerTokenMs?.toFixed(2) ?? 'n/a'} / {report.queueVsDirect.queueEngineChat.learnedDecodeCostPerTokenMs?.toFixed(2) ?? 'n/a'}</div>
                  </>
                ) : null}
                {report.queueVsDirect.directEngineChat ? (
                  <>
                    <div>Direct pending max: {report.queueVsDirect.directEngineChat.maxPendingCount}</div>
                    <div>Direct overdue max: {report.queueVsDirect.directEngineChat.maxOverduePendingCount}</div>
                    <div>Direct slice max: {report.queueVsDirect.directEngineChat.maxSliceCount}</div>
                    <div>Direct prompt tokens max: {report.queueVsDirect.directEngineChat.maxEstimatedPromptTokens}</div>
                    <div>Direct cost obs count: {report.queueVsDirect.directEngineChat.costModelObservedCount}</div>
                    <div>Direct learned prefill/decode: {report.queueVsDirect.directEngineChat.learnedPrefillCostPerTokenMs?.toFixed(2) ?? 'n/a'} / {report.queueVsDirect.directEngineChat.learnedDecodeCostPerTokenMs?.toFixed(2) ?? 'n/a'}</div>
                  </>
                ) : null}
              </div>
            </section>
          ) : null}

          {report.schedulingStress ? (
            <section className="panel">
              <h2>Exp4 Head-of-Line Mixed Stress</h2>
              <div className="grid3">
                <div>Scenario: {report.schedulingStress.scenario}</div>
                <div>Long head count: {report.schedulingStress.scenarioConfig.longHeadCount}</div>
                <div>Short tail count: {report.schedulingStress.scenarioConfig.shortTailCount}</div>
                <div>Medium tail count: {report.schedulingStress.scenarioConfig.mediumTailCount}</div>
                <div>Long arrival gap: {ms(report.schedulingStress.scenarioConfig.longArrivalGapMs)}</div>
                <div>Medium burst delay: {ms(report.schedulingStress.scenarioConfig.mediumBurstDelayMs)}</div>
                <div>Short stream start: {ms(report.schedulingStress.scenarioConfig.shortStreamStartMs)}</div>
                <div>Short inter-arrival: {ms(report.schedulingStress.scenarioConfig.shortInterArrivalMs)}</div>
                <div>Medium prefix seed shots: {report.schedulingStress.scenarioConfig.mediumPrefixSeedShots}</div>
                <div>Medium prefix repeated examples: {report.schedulingStress.scenarioConfig.mediumPrefixExampleCount}</div>
                <div>Long prefix seed shots: {report.schedulingStress.scenarioConfig.longPrefixSeedShots}</div>
                <div>Long prefix repeated examples: {report.schedulingStress.scenarioConfig.longPrefixExampleCount}</div>
                <div>Medium prompt token budget: {report.schedulingStress.scenarioConfig.mediumPromptTokenBudget}</div>
                <div>Medium estimated prompt tokens: {report.schedulingStress.scenarioConfig.estimatedMediumPromptTokens}</div>
                <div>Long prompt token budget: {report.schedulingStress.scenarioConfig.longPromptTokenBudget}</div>
                <div>Long estimated prompt tokens: {report.schedulingStress.scenarioConfig.estimatedLongPromptTokens}</div>
                <div>Dual wait budget scale: {report.schedulingStress.scenarioConfig.dualQueueWaitBudgetScale}</div>
                <div>Dual wait budget cap: {ms(report.schedulingStress.scenarioConfig.dualQueueWaitBudgetCapMs)}</div>
              </div>
              <table>
                <thead>
                  <tr>
                    <th>Arm</th>
                    <th>Completion</th>
                    <th>Timeout</th>
                    <th>Wait p95</th>
                    <th>TTFT p95</th>
                    <th>Sojourn p95</th>
                    <th>Req/s</th>
                    <th>Tok/s</th>
                  </tr>
                </thead>
                <tbody>
                  {report.schedulingStress.arms.map((arm) => (
                    <tr key={arm.arm}>
                      <td>{arm.arm}</td>
                      <td>{pct01(arm.completionRate)}</td>
                      <td>{pct01(arm.timeoutRate)}</td>
                      <td>{ms(arm.p95WaitMs)}</td>
                      <td>{ms(arm.p95TtftMs)}</td>
                      <td>{ms(arm.p95SojournMs)}</td>
                      <td>{arm.throughputReqPerSec.toFixed(3)}</td>
                      <td>{arm.throughputTokensPerSec.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div style={{ marginTop: 18, fontWeight: 600 }}>Class-Focused View</div>
              <table>
                <thead>
                  <tr>
                    <th>Arm</th>
                    <th>Class</th>
                    <th>Wait p95</th>
                    <th>Wait p99</th>
                    <th>Max Wait</th>
                    <th>TTFT p95</th>
                    <th>Wait SLA</th>
                  </tr>
                </thead>
                <tbody>
                  {report.schedulingStress.arms.flatMap((arm) =>
                    arm.byClass
                      .filter((row) => row.workloadClass === 'short' || row.workloadClass === 'medium')
                      .map((row) => (
                        <tr key={`${arm.arm}-${row.workloadClass}`}>
                          <td>{arm.arm}</td>
                          <td>{row.workloadClass}</td>
                          <td>{ms(row.p95WaitMs)}</td>
                          <td>{ms(row.p99WaitMs)}</td>
                          <td>{ms(row.maxWaitMs)}</td>
                          <td>{ms(row.p95TtftMs)}</td>
                          <td>{`${(row.waitSlaViolationRate * 100).toFixed(1)}% > ${ms(row.waitSlaMs)}`}</td>
                        </tr>
                      )))}
                </tbody>
              </table>
            </section>
          ) : null}

          {report.diagnostics ? (
            <section className="panel">
              <h2>Diagnostics</h2>
              <div className="grid3">
                <div>Runtime restarts: {report.diagnostics.runtimeRestartCount}</div>
                <div>Timeout failures: {report.diagnostics.timeoutFailureCount}</div>
                <div>Abort failures: {report.diagnostics.abortFailureCount}</div>
                <div>Disposed failures: {report.diagnostics.disposedFailureCount}</div>
                <div>Other failures: {report.diagnostics.otherFailureCount}</div>
              </div>
            </section>
          ) : null}

          {report.mmlu && !isMmluExp2 && !isWebllmNoCache ? (
            <section className="panel">
              <h2>MMLU Disagreement (Flat vs Tree)</h2>
              <p>Count: {mmluDiff.length}</p>
              {mmluDiff.length ? (
                <table>
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>GT</th>
                      <th>Flat</th>
                      <th>Tree</th>
                    </tr>
                  </thead>
                  <tbody>
                    {mmluDiff.map((r) => (
                      <tr key={r.id}>
                        <td>{r.id}</td>
                        <td>{r.gtIndex}</td>
                        <td>{r.predIndexFlat}</td>
                        <td>{r.predIndexTree}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : <p className="hint">No disagreement.</p>}
            </section>
          ) : null}

          {report.hella && !isWebllmNoCache ? (
            <section className="panel">
              <h2>HellaSwag Disagreement (Flat vs Tree)</h2>
              <p>Count: {hellaDiff.length}</p>
              {hellaDiff.length ? (
                <table>
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>GT</th>
                      <th>Flat</th>
                      <th>Tree</th>
                    </tr>
                  </thead>
                  <tbody>
                    {hellaDiff.map((r) => (
                      <tr key={r.id}>
                        <td>{r.id}</td>
                        <td>{r.gtIndex}</td>
                        <td>{r.predIndexFlat}</td>
                        <td>{r.predIndexTree}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : <p className="hint">No disagreement.</p>}
            </section>
          ) : null}
        </>
      ) : null}
    </div>
  );
}
