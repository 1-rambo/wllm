import { useMemo, useRef, useState } from 'react';
import {
  getAllLocalMmluSubjects,
  getHellaSwagLineCount,
  getLocalMmluSubjectCounts,
  loadRealHellaSwag,
  loadRealMmluFromLocal,
} from './bench/data-real';
import { runSglangStyleBench } from './bench/runner';
import type { BenchConfig, BenchReport, QAResult } from './bench/types';

const MODEL_BASE_DIR = '/Users/rambo/Desktop/wllama-webgpu/examples/agentic-benchmark-ab/public';
const MODEL_FILE = 'Llama-3.2-1B-Instruct-Q4_0.gguf';

const DEFAULT_CONFIG: BenchConfig = {
  modelUrl: `${window.location.origin}/@fs${encodeURI(`${MODEL_BASE_DIR}/${MODEL_FILE}`)}`,
  nCtx: 8192,
  nBatch: 512,
  mmluShots: 5,
  hellaShots: 20,
  mmluEvalCount: 1,
  hellaEvalCount: 1,
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

export default function App() {
  const [cfg, setCfg] = useState(DEFAULT_CONFIG);
  const [hellaDataUrl, setHellaDataUrl] = useState(DEFAULT_HELLA_URL);
  const [mmluSubject, setMmluSubject] = useState(DEFAULT_MMLU_SUBJECT);
  const [runFullDataset, setRunFullDataset] = useState(false);
  const [runHellaSwag, setRunHellaSwag] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState('');
  const [report, setReport] = useState<BenchReport | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const logsRef = useRef<string[]>([]);
  const metricLinesRef = useRef<string[]>([]);

  const canRun = useMemo(() => {
    return !running
      && cfg.modelUrl.trim().length > 0
      && (!runHellaSwag || hellaDataUrl.trim().length > 0);
  }, [running, cfg.modelUrl, hellaDataUrl, runHellaSwag]);

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
    setLogs([]);
    setReport(null);

    try {
      const effectiveCfg: BenchConfig = { ...cfg };
      let mmluSubjectsToRun = [mmluSubject];
      let mmluCountsBySubject = new Map<string, { valCount: number; testCount: number }>();
      if (runFullDataset) {
        appendLog('Resolving full dataset counts from local files (all MMLU subjects + HellaSwag)...');
        mmluSubjectsToRun = getAllLocalMmluSubjects();
        if (!mmluSubjectsToRun.length) {
          throw new Error('No local MMLU subjects found.');
        }

        const mmluCountsPromise = Promise.all(
          mmluSubjectsToRun.map(async (subject) => [subject, await getLocalMmluSubjectCounts(subject)] as const)
        );
        const hellaCountPromise = runHellaSwag ? getHellaSwagLineCount(hellaDataUrl) : Promise.resolve(0);
        const [mmluCountPairs, hellaTotal] = await Promise.all([mmluCountsPromise, hellaCountPromise]);

        mmluCountsBySubject = new Map(mmluCountPairs);
        const minValCount = Math.min(...mmluCountPairs.map(([, c]) => c.valCount));
        const totalTestCount = mmluCountPairs.reduce((sum, [, c]) => sum + c.testCount, 0);

        // Full mode should expand eval set, but still respect caller-configured few-shot count.
        effectiveCfg.mmluShots = Math.max(0, Math.min(cfg.mmluShots, minValCount));
        // Per-subject full eval is handled by loader; runner will consume all loaded rows.
        effectiveCfg.mmluEvalCount = Number.MAX_SAFE_INTEGER;
        if (effectiveCfg.mmluShots !== cfg.mmluShots) {
          appendLog(
            `MMLU shots clipped by local val sets: requested=${cfg.mmluShots}, min-available=${minValCount}, used=${effectiveCfg.mmluShots}`
          );
        }
        if (runHellaSwag) {
          effectiveCfg.hellaShots = Math.max(1, Math.min(cfg.hellaShots, Math.max(1, hellaTotal - 1)));
          effectiveCfg.hellaEvalCount = Math.max(1, hellaTotal - effectiveCfg.hellaShots);
        } else {
          effectiveCfg.hellaShots = 0;
          effectiveCfg.hellaEvalCount = 0;
        }

        appendLog(
          `Full mode: MMLU subjects=${mmluSubjectsToRun.length}, per-subject shots=${effectiveCfg.mmluShots}, total MMLU eval=${totalTestCount}, Hella shots=${effectiveCfg.hellaShots}, Hella eval=${effectiveCfg.hellaEvalCount}`
        );
      } else if (!runHellaSwag) {
        effectiveCfg.hellaShots = 0;
        effectiveCfg.hellaEvalCount = 0;
      }

      let mmlu: Awaited<ReturnType<typeof loadRealMmluFromLocal>> = [];
      if (runFullDataset) {
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
      } else if (effectiveCfg.mmluEvalCount > 0) {
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

      const hellaRequired = effectiveCfg.hellaShots + effectiveCfg.hellaEvalCount;
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
        (e) => appendLog(e.text)
      );
      setReport(out);
      saveJson(`sglang-style-report-${Date.now()}.json`, out);
      appendLog('Benchmark finished and report exported.');
    } catch (e) {
      const errText = (e as Error).message || String(e);
      setError(errText);
      appendLog(`Error: ${errText}`);

      const stamp = new Date().toISOString().replace(/[:.]/g, '-');
      const header = [
        `timestamp: ${new Date().toISOString()}`,
        `error: ${errText}`,
        `modelUrl: ${cfg.modelUrl}`,
        `nCtx: ${cfg.nCtx}`,
        `nBatch: ${cfg.nBatch}`,
        `mmluShots: ${cfg.mmluShots}`,
        `hellaShots: ${cfg.hellaShots}`,
        `mmluEvalCount: ${cfg.mmluEvalCount}`,
        `hellaEvalCount: ${cfg.hellaEvalCount}`,
        `runFullDataset: ${runFullDataset}`,
        `runHellaSwag: ${runHellaSwag}`,
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
          modelUrl: cfg.modelUrl,
          nCtx: cfg.nCtx,
          nBatch: cfg.nBatch,
          mmluShots: cfg.mmluShots,
          hellaShots: cfg.hellaShots,
          mmluEvalCount: cfg.mmluEvalCount,
          hellaEvalCount: cfg.hellaEvalCount,
          runFullDataset,
          runHellaSwag,
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

  const mmluDiff = rowsMismatched(report?.mmlu.results ?? []);
  const hellaDiff = rowsMismatched(report?.hella.results ?? []);

  return (
    <div className="page">
      <header>
        <h1>SGLang-style MMLU5 + HellaSwag20</h1>
        <p>
          设计对齐 SGLang 思路：MMLU 用单-token 选项概率，HellaSwag 用候选 continuation 概率选择。
          同时对比 Flat（无树缓存）与 Tree（KV slot restore）以展示前缀缓存收益。
        </p>
      </header>

      <section className="panel">
        <h2>配置</h2>
        <label>
          <span>Model URL</span>
          <input
            className="text-input"
            value={cfg.modelUrl}
            onChange={(e) => setCfg((p) => ({ ...p, modelUrl: e.target.value }))}
            disabled={running}
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
          <span>Run HellaSwag</span>
          <input
            type="checkbox"
            checked={runHellaSwag}
            onChange={(e) => setRunHellaSwag(e.target.checked)}
            disabled={running}
          />
        </label>
        <label>
          <span>MMLU Subject (local CSV)</span>
          <input
            className="text-input"
            value={mmluSubject}
            onChange={(e) => setMmluSubject(e.target.value.trim())}
            disabled={running || (cfg.mmluEvalCount === 0 && !runFullDataset)}
          />
        </label>
        <label>
          <span>HellaSwag JSONL URL</span>
          <input
            className="text-input"
            value={hellaDataUrl}
            onChange={(e) => setHellaDataUrl(e.target.value)}
            disabled={running || !runHellaSwag}
          />
        </label>

        <div className="row4">
          <label>
            <span>n_ctx</span>
            <input
              className="text-input"
              type="number"
              value={cfg.nCtx}
              onChange={(e) => setCfg((p) => ({ ...p, nCtx: Math.max(1024, Number(e.target.value) || 8192) }))}
              disabled={running}
            />
          </label>
          <label>
            <span>n_batch</span>
            <input
              className="text-input"
              type="number"
              value={cfg.nBatch}
              onChange={(e) => setCfg((p) => ({ ...p, nBatch: Math.max(64, Number(e.target.value) || 512) }))}
              disabled={running}
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
              disabled={running || !runHellaSwag}
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
              disabled={running}
            />
          </label>
          <label>
            <span>Hella eval count</span>
            <input
              className="text-input"
              type="number"
              value={cfg.hellaEvalCount}
              onChange={(e) => setCfg((p) => ({ ...p, hellaEvalCount: Math.max(1, Number(e.target.value) || 4) }))}
              disabled={running || !runHellaSwag}
            />
          </label>
        </div>

        <div className="run-row">
          <button onClick={() => void runBench()} disabled={!canRun}>
            {running ? 'Running...' : 'Run SGLang-style Bench'}
          </button>
        </div>

        {error ? <p className="error">{error}</p> : null}
        <div className="log-box">
          {logs.map((l, i) => (
            <div key={`${i}-${l}`} className="log-line">{l}</div>
          ))}
        </div>
      </section>

      {report ? (
        <>
          <section className="panel">
            <h2>MMLU Summary</h2>
            <div className="grid3">
              <div>Flat acc: {pct01(report.mmlu.accFlat)}</div>
              <div>Tree acc: {pct01(report.mmlu.accTree)}</div>
              <div>Tree latency speedup: {speed(report.mmlu.speedupPct)}</div>
              <div>Flat TTFT: {ms(report.mmlu.avgTtftMsFlat)}</div>
              <div>Tree TTFT: {ms(report.mmlu.avgTtftMsTree)}</div>
              <div>Tree TTFT speedup: {speed(report.mmlu.ttftSpeedupPct)}</div>
              <div>Flat tokens/s: {tps(report.mmlu.avgTokensPerSecondFlat)}</div>
              <div>Tree tokens/s: {tps(report.mmlu.avgTokensPerSecondTree)}</div>
              <div>Tree tokens/s gain: {speed(report.mmlu.tpsGainPct)}</div>
              <div>Flat avg latency: {ms(report.mmlu.avgLatencyMsFlat)}</div>
              <div>Tree avg latency: {ms(report.mmlu.avgLatencyMsTree)}</div>
              <div>Eval count: {report.mmlu.evalCount}</div>
            </div>
          </section>

          <section className="panel">
            <h2>HellaSwag Summary</h2>
            <div className="grid3">
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
              <div>Eval count: {report.hella.evalCount}</div>
            </div>
          </section>

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
        </>
      ) : null}
    </div>
  );
}
