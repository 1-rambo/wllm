import { useMemo, useRef, useState } from 'react';
import { filterTasksBySites, loadWebArenaRetrieveSubset, summarizeSiteCounts } from './bench/data';
import { runWebArenaBench } from './bench/runner';
import type { BenchConfig, BenchProgressEvent, BenchReport, BenchSampleMetric, BenchSeriesStats, WebArenaTask } from './bench/types';

const MODEL_BASE_DIR = '/Users/rambo/Desktop/wllama-webgpu/examples/prefix-tree-chat/dist/';
const MODEL_FILE = 'Llama-3.2-1B-Instruct-Q4_0.gguf';
const DATASET_URL = '/datasets/webarena/retrieve_info_subset.json';
const SITE_OPTIONS = ['shopping_admin', 'shopping', 'gitlab', 'reddit'] as const;

const DEFAULT_CONFIG: BenchConfig = {
  backend: 'wllama',
  webllmModelId: 'Llama-3.2-1B-Instruct-q4f32_1-MLC',
  modelUrl: `${window.location.origin}/@fs${encodeURI(`${MODEL_BASE_DIR}/${MODEL_FILE}`)}`,
  nCtx: 8192,
  nBatch: 512,
  usePreloadedPageContext: true,
  trueTreeMemoryCapMB: 4096,
  trueTreeTieredCacheEnabled: true,
  trueTreeTierL1TokenCap: 8192,
  trueTreeTierL2TokenCap: 32768,
  trueTreeTierL3TokenCap: 131072,
  trueTreePruneL1L2TokenThreshold: 1024,
  trueTreePruneL2L3TokenThreshold: 8192,
  trueTreeReplacementPolicy: 'hybrid',
  evalCount: 120,
  maxOutputTokens: 48,
  includeSites: [...SITE_OPTIONS],
};

function ms(v: number): string {
  return `${v.toFixed(1)} ms`;
}

function pct(v: number): string {
  return `${v.toFixed(1)}%`;
}

function tps(v: number): string {
  return v.toFixed(3);
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

function escapeCsv(value: string | number | boolean | null | undefined): string {
  const text = value == null ? '' : String(value);
  if (/[",\n]/.test(text)) {
    return `"${text.replace(/"/g, '""')}"`;
  }
  return text;
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

function buildSeriesStats(metrics: BenchSampleMetric[]): BenchSeriesStats {
  const latency = metrics.map((m) => m.latencyMs);
  const ttftVals = metrics.map((m) => m.ttftMs);
  const tpsVals = metrics.map((m) => m.tokensPerSecond);
  return {
    sampleCount: metrics.length,
    avgLatencyMs: latency.length ? latency.reduce((s, v) => s + v, 0) / latency.length : 0,
    p50LatencyMs: quantile(latency, 0.5),
    p95LatencyMs: quantile(latency, 0.95),
    p99LatencyMs: quantile(latency, 0.99),
    avgTtftMs: ttftVals.length ? ttftVals.reduce((s, v) => s + v, 0) / ttftVals.length : 0,
    p50TtftMs: quantile(ttftVals, 0.5),
    p95TtftMs: quantile(ttftVals, 0.95),
    p99TtftMs: quantile(ttftVals, 0.99),
    avgTokensPerSecond: tpsVals.length ? tpsVals.reduce((s, v) => s + v, 0) / tpsVals.length : 0,
  };
}

function buildSampleMetricsCsv(report: BenchReport): string {
  const metrics = report.webarena
    ? [...report.webarena.sampleMetricsFlat, ...report.webarena.sampleMetricsTree]
    : [];
  const header = [
    'benchmark', 'mode', 'id', 'site', 'latencyMs', 'ttftMs', 'tokensPerSecond', 'tokenCount', 'promptChars', 'outputChars',
  ];
  const rows = metrics.map((metric) => ([
    metric.benchmark,
    metric.mode,
    metric.id,
    metric.subject ?? '',
    metric.latencyMs.toFixed(4),
    metric.ttftMs.toFixed(4),
    metric.tokensPerSecond.toFixed(6),
    metric.tokenCount,
    metric.promptChars,
    metric.outputChars,
  ].map(escapeCsv).join(',')));
  return [header.join(','), ...rows].join('\n');
}

function buildCdfCsv(report: BenchReport): string {
  const rows = [['benchmark', 'metric', 'mode', 'value', 'cdf'].join(',')];
  const summary = report.webarena;
  if (!summary) return rows.join('\n');
  const append = (metric: 'latencyMs' | 'ttftMs', mode: 'flat' | 'tree', points: Array<{ value: number; cdf: number }>) => {
    for (const point of points) {
      rows.push(['WebArena', metric, mode, point.value.toFixed(4), point.cdf.toFixed(6)].map(escapeCsv).join(','));
    }
  };
  append('latencyMs', 'flat', summary.latencyCdfFlat);
  append('latencyMs', 'tree', summary.latencyCdfTree);
  append('ttftMs', 'flat', summary.ttftCdfFlat);
  append('ttftMs', 'tree', summary.ttftCdfTree);
  return rows.join('\n');
}

function buildFailureCsv(report: BenchReport): string {
  const header = ['mode', 'taskId', 'site', 'groupKey', 'stage', 'message'];
  const rows = report.diagnostics.failureRecords.map((record) => ([
    record.mode,
    record.taskId,
    record.site,
    record.groupKey,
    record.stage,
    record.message,
  ].map(escapeCsv).join(',')));
  return [header.join(','), ...rows].join('\n');
}

function buildOccupancyCsv(report: BenchReport): string {
  const header = [
    'label',
    'mode',
    'site',
    'groupKey',
    'taskId',
    'snapshotTokenBytes',
    'tierL1Tokens',
    'tierL2Tokens',
    'tierL3Tokens',
    'tierL1Slots',
    'tierL2Slots',
    'tierL3Slots',
  ];
  const rows = report.cacheProfile.occupancySamples.map((sample) => ([
    sample.label,
    sample.mode,
    sample.site,
    sample.groupKey,
    sample.taskId ?? '',
    sample.snapshotTokenBytes,
    sample.tierL1Tokens,
    sample.tierL2Tokens,
    sample.tierL3Tokens,
    sample.tierL1Slots,
    sample.tierL2Slots,
    sample.tierL3Slots,
  ].map(escapeCsv).join(',')));
  return [header.join(','), ...rows].join('\n');
}

function buildDetailsExport(report: BenchReport, tasks: WebArenaTask[], logs: string[], summaryLines: string[]) {
  const sampleMetrics = report.webarena
    ? [...report.webarena.sampleMetricsFlat, ...report.webarena.sampleMetricsTree]
    : [];
  const grouped = Array.from(sampleMetrics.reduce((map, metric) => {
    const key = `WebArena/${metric.mode}`;
    const bucket = map.get(key) ?? [];
    bucket.push(metric);
    map.set(key, bucket);
    return map;
  }, new Map<string, BenchSampleMetric[]>()).entries()).map(([seriesKey, metrics]) => ({
    seriesKey,
    benchmark: 'WebArena',
    mode: metrics[0]?.mode ?? 'tree',
    stats: buildSeriesStats(metrics),
    samples: metrics,
  }));

  return {
    report,
    tasks,
    sampleMetrics,
    sampleMetricSeries: grouped,
    cdf: {
      webarena: {
        latencyFlat: report.webarena?.latencyCdfFlat ?? [],
        latencyTree: report.webarena?.latencyCdfTree ?? [],
        ttftFlat: report.webarena?.ttftCdfFlat ?? [],
        ttftTree: report.webarena?.ttftCdfTree ?? [],
      },
    },
    failures: report.diagnostics.failureRecords,
    cacheOccupancySamples: report.cacheProfile.occupancySamples,
    summaryLines,
    logs,
  };
}

export default function App() {
  const [config, setConfig] = useState<BenchConfig>(DEFAULT_CONFIG);
  const [tasks, setTasks] = useState<WebArenaTask[] | null>(null);
  const [report, setReport] = useState<BenchReport | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [progress, setProgress] = useState<BenchProgressEvent | null>(null);
  const [running, setRunning] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const appendLog = (text: string) => setLogs((prev) => [...prev, text]);

  const filteredTasks = useMemo(() => {
    if (!tasks) return [];
    return filterTasksBySites(tasks, config.includeSites);
  }, [tasks, config.includeSites]);

  const summaryLines = useMemo(() => {
    if (!report?.webarena) return [];
    const summary = report.webarena;
    if (config.backend === 'web-llm') {
      return [
        `[WebArena] tasks=${summary.evalCount} success=${summary.successCountTree} failed=${summary.failureCountTree} sites=${Object.entries(summary.siteBreakdown).map(([k, v]) => `${k}:${v}`).join(', ')}`,
        `[WebArena] web-llm Avg TTFT=${summary.avgTtftMsTree.toFixed(2)} ms P95 TTFT=${quantile(summary.sampleMetricsTree.map((m) => m.ttftMs), 0.95).toFixed(2)} ms`,
        `[WebArena] web-llm Avg Latency=${summary.avgLatencyMsTree.toFixed(2)} ms P95 Latency=${quantile(summary.sampleMetricsTree.map((m) => m.latencyMs), 0.95).toFixed(2)} ms`,
        `[WebArena] web-llm Avg Tokens/s=${summary.avgTokensPerSecondTree.toFixed(3)}`,
      ];
    }
    return [
      `[WebArena] tasks=${summary.evalCount} success flat/tree=${summary.successCountFlat}/${summary.successCountTree} failed flat/tree=${summary.failureCountFlat}/${summary.failureCountTree} sites=${Object.entries(summary.siteBreakdown).map(([k, v]) => `${k}:${v}`).join(', ')}`,
      `[WebArena] TTFT flat/tree=${summary.avgTtftMsFlat.toFixed(2)}/${summary.avgTtftMsTree.toFixed(2)} ms speedup=${summary.ttftSpeedupPct.toFixed(1)}%`,
      `[WebArena] Latency flat/tree=${summary.avgLatencyMsFlat.toFixed(2)}/${summary.avgLatencyMsTree.toFixed(2)} ms speedup=${summary.speedupPct.toFixed(1)}%`,
      `[WebArena] Tokens/s flat/tree=${summary.avgTokensPerSecondFlat.toFixed(3)}/${summary.avgTokensPerSecondTree.toFixed(3)} gain=${summary.tpsGainPct.toFixed(1)}%`,
      `[WebArena] Maintenance session/state/prefix=${report.cacheProfile.maintenanceBreakdownMs.sessionInitMs.toFixed(2)}/${report.cacheProfile.maintenanceBreakdownMs.stateReadMs.toFixed(2)}/${report.cacheProfile.maintenanceBreakdownMs.prefixSetupMs.toFixed(2)} ms; snapshot=${report.cacheProfile.snapshotTokenBytes}B l1/l2/l3=${report.cacheProfile.tierL1Tokens}/${report.cacheProfile.tierL2Tokens}/${report.cacheProfile.tierL3Tokens}`,
    ];
  }, [report, config.backend]);

  const loadTasks = async () => {
    const data = await loadWebArenaRetrieveSubset(DATASET_URL);
    setTasks(data);
    appendLog(`[Dataset] loaded ${data.length} WebArena retrieve tasks; sites=${JSON.stringify(summarizeSiteCounts(data))}`);
  };

  const runBench = async () => {
    setRunning(true);
    setReport(null);
    setLogs([]);
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const loadedTasks = tasks ?? await loadWebArenaRetrieveSubset(DATASET_URL);
      if (!tasks) {
        setTasks(loadedTasks);
      }
      const selected = filterTasksBySites(loadedTasks, config.includeSites);
      appendLog(`[Run] backend=${config.backend} selectedTasks=${selected.length} evalCount=${config.evalCount} pageContext=${config.usePreloadedPageContext ? 'on' : 'off'}`);
      const out = await runWebArenaBench(
        config,
        selected,
        (e) => appendLog(e.text),
        setProgress,
        controller.signal,
      );
      setReport(out);
    } catch (err) {
      appendLog(`[Error] ${err instanceof Error ? err.message : String(err)}`);
      throw err;
    } finally {
      setRunning(false);
      abortRef.current = null;
    }
  };

  const exportArtifacts = () => {
    if (!report) return;
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    saveJson(`webarena-exp1-report-${timestamp}.json`, report);
    const usedTasks = filteredTasks.slice(0, Math.min(config.evalCount, filteredTasks.length));
    saveJson(`webarena-exp1-details-${timestamp}.json`, buildDetailsExport(report, usedTasks, logs, summaryLines));
    saveText(`webarena-exp1-sample-metrics-${timestamp}.csv`, buildSampleMetricsCsv(report));
    saveText(`webarena-exp1-cdf-${timestamp}.csv`, buildCdfCsv(report));
    saveText(`webarena-exp1-failures-${timestamp}.csv`, buildFailureCsv(report));
    saveText(`webarena-exp1-occupancy-${timestamp}.csv`, buildOccupancyCsv(report));
    saveText(`webarena-exp1-logs-${timestamp}.log`, logs.join('\n'));
  };

  return (
    <div className="page">
      <div className="hero">
        <div>
          <h1>WebArena Latency Bench</h1>
          <p>Exp1-style end-to-end latency test on WebArena Verified retrieve tasks.</p>
        </div>
        <div className="hero-actions">
          <button onClick={loadTasks} disabled={running}>Load Dataset</button>
          <button onClick={() => void runBench()} disabled={running}>Run Exp1</button>
          <button onClick={() => abortRef.current?.abort()} disabled={!running}>Abort</button>
          <button onClick={exportArtifacts} disabled={!report}>Export</button>
        </div>
      </div>

      <section className="panel-grid two">
        <div className="panel">
          <h2>Config</h2>
          <label>Backend
            <select value={config.backend} onChange={(e) => setConfig((prev) => ({ ...prev, backend: e.target.value as BenchConfig['backend'] }))}>
              <option value="wllama">wllama (flat + tree)</option>
              <option value="web-llm">web-llm (single path)</option>
            </select>
          </label>
          <label>Eval Count
            <input type="number" value={config.evalCount} min={1} max={filteredTasks.length || 1} onChange={(e) => setConfig((prev) => ({ ...prev, evalCount: Number(e.target.value) }))} />
          </label>
          <label>Max Output Tokens
            <input type="number" value={config.maxOutputTokens} min={8} max={128} onChange={(e) => setConfig((prev) => ({ ...prev, maxOutputTokens: Number(e.target.value) }))} />
          </label>
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={config.usePreloadedPageContext}
              onChange={(e) => setConfig((prev) => ({ ...prev, usePreloadedPageContext: e.target.checked }))}
            />
            Use preloaded page context in shared prefix
          </label>
          <div className="site-list">
            <span>Sites</span>
            {SITE_OPTIONS.map((site) => (
              <label key={site} className="checkbox-row">
                <input
                  type="checkbox"
                  checked={config.includeSites.includes(site)}
                  onChange={(e) => {
                    setConfig((prev) => ({
                      ...prev,
                      includeSites: e.target.checked
                        ? [...prev.includeSites, site]
                        : prev.includeSites.filter((x) => x !== site),
                    }));
                  }}
                />
                {site}
              </label>
            ))}
          </div>
        </div>

        <div className="panel">
          <h2>Dataset</h2>
          <p>Total loaded tasks: {tasks?.length ?? 0}</p>
          <p>Filtered tasks: {filteredTasks.length}</p>
          <p>Site counts: {filteredTasks.length ? JSON.stringify(summarizeSiteCounts(filteredTasks)) : 'n/a'}</p>
          <p>Progress: {progress ? `${progress.current}/${progress.total} ${progress.label}` : 'idle'}</p>
        </div>
      </section>

      <section className="panel">
        <h2>Summary</h2>
        {summaryLines.length ? (
          <ul className="summary-list">
            {summaryLines.map((line) => <li key={line}>{line}</li>)}
          </ul>
        ) : <p>No report yet.</p>}
        {report?.webarena && (
          <table className="metrics-table">
            <thead>
              <tr>
                <th>Mode</th>
                <th>Avg TTFT</th>
                <th>P95 TTFT</th>
                <th>Avg Latency</th>
                <th>P95 Latency</th>
                <th>Avg Tokens/s</th>
              </tr>
            </thead>
            <tbody>
              {config.backend !== 'web-llm' && (
                <tr>
                  <td>flat</td>
                  <td>{ms(report.webarena.avgTtftMsFlat)}</td>
                  <td>{ms(report.webarena.ttftCdfFlat.length ? quantile(report.webarena.sampleMetricsFlat.map((m) => m.ttftMs), 0.95) : 0)}</td>
                  <td>{ms(report.webarena.avgLatencyMsFlat)}</td>
                  <td>{ms(report.webarena.latencyCdfFlat.length ? quantile(report.webarena.sampleMetricsFlat.map((m) => m.latencyMs), 0.95) : 0)}</td>
                  <td>{tps(report.webarena.avgTokensPerSecondFlat)}</td>
                </tr>
              )}
              <tr>
                <td>{config.backend === 'web-llm' ? 'web-llm' : 'tree'}</td>
                <td>{ms(report.webarena.avgTtftMsTree)}</td>
                <td>{ms(quantile(report.webarena.sampleMetricsTree.map((m) => m.ttftMs), 0.95))}</td>
                <td>{ms(report.webarena.avgLatencyMsTree)}</td>
                <td>{ms(quantile(report.webarena.sampleMetricsTree.map((m) => m.latencyMs), 0.95))}</td>
                <td>{tps(report.webarena.avgTokensPerSecondTree)}</td>
              </tr>
            </tbody>
          </table>
        )}
      </section>

      <section className="panel">
        <h2>Logs</h2>
        <pre className="log-view">{logs.join('\n')}</pre>
      </section>
    </div>
  );
}
