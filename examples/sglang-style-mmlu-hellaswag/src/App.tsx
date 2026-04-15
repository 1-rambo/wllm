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

const MODEL_BASE_DIR = '/Users/rambo/Desktop/wllama-webgpu/model';
const MODEL_FILE = 'Llama-3.2-1B-Instruct-Q4_0.gguf';

const DEFAULT_CONFIG: BenchConfig = {
  backend: 'wllama',
  webllmModelId: 'Llama-3.2-1B-Instruct-q4f32_1-MLC',
  modelUrl: `${window.location.origin}/@fs${encodeURI(`${MODEL_BASE_DIR}/${MODEL_FILE}`)}`,
  nCtx: 8192,
  nBatch: 512,
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
  exp4OutputTokens: 1,
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
    const snapshotSuffix = report.cacheProfile.snapshotTokenBytes >= INT32_MAX ? ' (clamped-int32)' : '';
    lines.push(`[Exp3] snapshotBytes=${report.cacheProfile.snapshotTokenBytes}${snapshotSuffix} tier(L1/L2/L3)=${report.cacheProfile.tierL1Tokens}/${report.cacheProfile.tierL2Tokens}/${report.cacheProfile.tierL3Tokens}`);
  }
  if (report.queueVsDirect) {
    lines.push(`[Exp4] requestCount=${report.queueVsDirect.requestCount} failed(queue/direct)=${report.queueVsDirect.failedCountQueue}/${report.queueVsDirect.failedCountDirect}`);
    lines.push(`[Exp4] ttft(queue/direct)=${report.queueVsDirect.avgTtftMsQueue.toFixed(2)}ms/${report.queueVsDirect.avgTtftMsDirect.toFixed(2)}ms`);
    lines.push(`[Exp4] latency(queue/direct)=${report.queueVsDirect.avgLatencyMsQueue.toFixed(2)}ms/${report.queueVsDirect.avgLatencyMsDirect.toFixed(2)}ms`);
    lines.push(`[Exp4] avgReqTps(queue/direct)=${report.queueVsDirect.avgTokensPerSecondQueue.toFixed(3)}/${report.queueVsDirect.avgTokensPerSecondDirect.toFixed(3)}`);
    lines.push(`[Exp4] batchTps(queue/direct)=${report.queueVsDirect.batchTokensPerSecondQueue.toFixed(3)}/${report.queueVsDirect.batchTokensPerSecondDirect.toFixed(3)} wallClockMs(queue/direct)=${report.queueVsDirect.batchWallClockMsQueue.toFixed(2)}/${report.queueVsDirect.batchWallClockMsDirect.toFixed(2)}`);
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
  const effectiveTarget = cfg.experimentRunMode === 'exp1' ? cfg.target : 'mmlu';
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
    if (/\[(MMLU|Hella)\/.+\] .+ pred=/.test(text)) {
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
        if (effectiveCfg.target !== 'mmlu') {
          appendLog('[Mode] Exp4 runs MMLU only; target forced to mmlu.');
        }
        effectiveCfg.target = 'mmlu';
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
        const autoEvalCount = Math.max(1, Math.min(desiredSamples, counts.testCount));
        effectiveCfg.mmluEvalCount = autoEvalCount;
        if (autoEvalCount < desiredSamples) {
          appendLog(
            `[Exp4] sampleCount=${desiredSamples} exceeds available test rows=${counts.testCount} for subject=${mmluSubject}; clamped to ${autoEvalCount}.`
          );
        } else {
          appendLog(
            `[Exp4] auto-set mmluEvalCount=${autoEvalCount} from exp4SampleCount=${desiredSamples}.`
          );
        }
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
      const stamp = new Date().toISOString().replace(/[:.]/g, '-');
      saveJson(`sglang-style-report-${Date.now()}.json`, out);
      saveJson(`sglang-style-run-detail-${stamp}.json`, {
        timestamp: new Date().toISOString(),
        requestedConfig: cfg,
        effectiveConfig: effectiveCfg,
        runtimeInputs: {
          runFullDataset,
          mmluSubject,
          hellaDataUrl,
          loadedMmluItems: mmlu.length,
          loadedHellaItems: hella.length,
        },
        summaryLines: reportSummaryLines(out),
        metricLineCount: metricLinesRef.current.length,
        metricLines: metricLinesRef.current,
        report: out,
        logs: logsRef.current,
      });
      const successHeader = [
        `timestamp: ${new Date().toISOString()}`,
        'status: success',
        '',
        '==== REQUESTED CONFIG ==== ',
        JSON.stringify(cfg, null, 2),
        '',
        '==== EFFECTIVE CONFIG ==== ',
        JSON.stringify(effectiveCfg, null, 2),
        '',
        '==== RUNTIME INPUTS ==== ',
        JSON.stringify({
          runFullDataset,
          mmluSubject,
          hellaDataUrl,
          loadedMmluItems: mmlu.length,
          loadedHellaItems: hella.length,
        }, null, 2),
        '',
        '==== SUMMARY ==== ',
        ...reportSummaryLines(out),
        '',
        '==== METRIC LINES ==== ',
        ...metricLinesRef.current,
        '',
        '==== LOGS ==== ',
      ].join('\n');
      saveText(`sglang-style-run-logs-${stamp}.log`, `${successHeader}\n${logsRef.current.join('\n')}\n`);
      appendLog(`Run detail exported: sglang-style-run-detail-${stamp}.json`);
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
            <option value="exp4">Exp4: Queue vs Direct</option>
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
            </section>
          ) : null}

          {report.cacheProfile ? (
            <section className="panel">
              <h2>Exp3 Cache Maintenance Profile</h2>
              <div className="grid3">
                <div>Maintenance time: {ms(report.cacheProfile.maintenanceMs)}</div>
                <div>Total run time: {ms(report.cacheProfile.runTotalMs)}</div>
                <div>Maintenance ratio: {report.cacheProfile.maintenancePct.toFixed(2)}%</div>
                <div>Snapshot bytes: {report.cacheProfile.snapshotTokenBytes}</div>
                <div>L1 tokens: {report.cacheProfile.tierL1Tokens}</div>
                <div>L2 tokens: {report.cacheProfile.tierL2Tokens}</div>
                <div>L3 tokens: {report.cacheProfile.tierL3Tokens}</div>
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
              </div>
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
