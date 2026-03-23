import { useMemo, useRef, useState } from 'react';
import { loadAllBenchmarkTasks } from './benchmark/dataset-loader';
import { runABExperiment } from './benchmark/ab-runner';
import type { ABExperimentResult, ABProgress, ABTaskDelta, ModelSpec, PrefixConfig } from './benchmark/types';

const DEFAULT_PREFIX_CONFIG: PrefixConfig = {
  systemPrompt:
    'You are a browser agent. Solve user requests from currently opened pages first. Keep output concise and accurate.',
  includeOpenUrls: true,
  includeStartUrl: true,
  webContentTemplate:
    'WEB_CONTEXT(dataset={{DATASET}}, eval={{EVAL}}): infer useful facts from opened tabs before deciding actions.',
  nPredict: 128,
  nCtx: 4096,
  nBatch: 512,
  memoryCapBytes: 1024 * 1024 * 1024,
};

const MODEL_BASE_DIR = '/Users/rambo/Desktop/wllama-webgpu/examples/agentic-benchmark-ab/public';
const MODELS: ModelSpec[] = [
  { id: 'llama-q4', fileName: 'Llama-3.2-1B-Instruct-Q4_0.gguf' },
];

function pct(v: number): string {
  return `${v >= 0 ? '+' : ''}${v.toFixed(2)}%`;
}

function ms(v: number): string {
  return `${v.toFixed(1)} ms`;
}

function tps(v: number): string {
  return `${v.toFixed(2)} tok/s`;
}

function normalizeForMatch(text: string): string {
  return text.toLowerCase().replace(/\s+/g, ' ').trim();
}

function formatLogArg(arg: unknown): string {
  if (typeof arg === 'string') {
    return arg;
  }
  if (arg instanceof Error) {
    return `${arg.name}: ${arg.message}\n${arg.stack || ''}`;
  }
  try {
    return JSON.stringify(arg);
  } catch {
    return String(arg);
  }
}

function nowIsoCompact(): string {
  return new Date().toISOString().replace(/[:.]/g, '-');
}

function saveJsonToLocalFile(fileName: string, payload: unknown): void {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = fileName;
  a.click();
  URL.revokeObjectURL(url);
}

type StopCapture = () => void;

function startRuntimeLogCapture(append: (line: string) => void): StopCapture {
  const original = {
    debug: console.debug,
    log: console.log,
    warn: console.warn,
    error: console.error,
  };

  const patch =
    (level: 'debug' | 'log' | 'warn' | 'error') =>
    (...args: unknown[]) => {
      original[level](...args);
      append(`[console.${level}] ${args.map(formatLogArg).join(' ')}`);
    };

  console.debug = patch('debug');
  console.log = patch('log');
  console.warn = patch('warn');
  console.error = patch('error');

  const onWindowError = (event: ErrorEvent) => {
    append(
      `[window.error] ${event.message} @ ${event.filename || 'unknown'}:${event.lineno}:${event.colno}`
    );
  };

  const onUnhandledRejection = (event: PromiseRejectionEvent) => {
    append(`[unhandledrejection] ${formatLogArg(event.reason)}`);
  };

  window.addEventListener('error', onWindowError);
  window.addEventListener('unhandledrejection', onUnhandledRejection);

  return () => {
    console.debug = original.debug;
    console.log = original.log;
    console.warn = original.warn;
    console.error = original.error;
    window.removeEventListener('error', onWindowError);
    window.removeEventListener('unhandledrejection', onUnhandledRejection);
  };
}

function pickSamples(deltas: ABTaskDelta[]): ABTaskDelta[] {
  const validRows = deltas.filter((d) => d.isValid);
  if (!validRows.length) {
    return [];
  }
  if (validRows.length <= 8) {
    return validRows;
  }
  const sorted = [...validRows].sort((a, b) => b.ttftGainPct - a.ttftGainPct);
  return [
    sorted[0],
    sorted[Math.floor(sorted.length * 0.2)],
    sorted[Math.floor(sorted.length * 0.5)],
    sorted[Math.floor(sorted.length * 0.8)],
    sorted[sorted.length - 1],
  ];
}

export default function App() {
  const tasks = useMemo(() => loadAllBenchmarkTasks(), []);
  const [cfg, setCfg] = useState(DEFAULT_PREFIX_CONFIG);
  const [taskCountInput, setTaskCountInput] = useState('20');
  const [modelBaseDir, setModelBaseDir] = useState(MODEL_BASE_DIR);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<ABProgress | null>(null);
  const [error, setError] = useState('');
  const [results, setResults] = useState<ABExperimentResult[]>([]);
  const [activeModelId, setActiveModelId] = useState(MODELS[0].id);
  const [controller, setController] = useState<AbortController | null>(null);
  const [logLines, setLogLines] = useState<string[]>([]);
  const fullLogLinesRef = useRef<string[]>([]);

  const activeResult = results.find((r) => r.modelId === activeModelId) || null;

  const grouped = useMemo(() => {
    const total = activeResult?.deltas ?? [];
    const high = total.filter((d) => d.ttftGainPct >= 35).length;
    const medium = total.filter((d) => d.ttftGainPct >= 15 && d.ttftGainPct < 35).length;
    const low = total.filter((d) => d.ttftGainPct > 0 && d.ttftGainPct < 15).length;
    const none = total.filter((d) => d.ttftGainPct <= 0).length;
    return { high, medium, low, none };
  }, [activeResult]);

  const correctnessPanel = useMemo(() => {
    const deltas = activeResult?.deltas ?? [];
    const rows = deltas
      .filter((d) => d.task.evalTypes.includes('string_match') && d.task.referenceHints.length > 0)
      .map((d) => {
        const refs = d.task.referenceHints.map(normalizeForMatch);
        const aText = normalizeForMatch(d.aOutputPreview);
        const bText = normalizeForMatch(d.bOutputPreview);
        const aHit = refs.some((r) => aText.includes(r));
        const bHit = refs.some((r) => bText.includes(r));
        return {
          taskId: d.task.id,
          dataset: d.task.dataset,
          refs: d.task.referenceHints,
          aHit,
          bHit,
          aOutput: d.aOutputPreview,
          bOutput: d.bOutputPreview,
          isValid: d.isValid,
        };
      });

    const aHitCount = rows.filter((r) => r.aHit).length;
    const bHitCount = rows.filter((r) => r.bHit).length;
    const bothMiss = rows.filter((r) => !r.aHit && !r.bHit);
    return {
      total: rows.length,
      aHitCount,
      bHitCount,
      bothMiss,
    };
  }, [activeResult]);

  const selectedTaskCount = useMemo(() => {
    const n = Number(taskCountInput);
    if (!Number.isFinite(n) || n <= 0) {
      return tasks.length;
    }
    return Math.min(Math.floor(n), tasks.length);
  }, [taskCountInput, tasks.length]);

  const appendLog = (line: string) => {
    const stamped = `[${new Date().toISOString()}] ${line}`;
    fullLogLinesRef.current.push(stamped);
    setLogLines((prev) => {
      const next = [...prev, stamped];
      return next.slice(-80);
    });
  };

  const onRun = async (taskLimit?: number, fixedTaskId?: string) => {
    const nextController = new AbortController();
    setController(nextController);
    setRunning(true);
    setError('');
    setResults([]);
    setLogLines([]);
    fullLogLinesRef.current = [];

    const runId = `ab-run-${nowIsoCompact()}`;
    const startedAt = new Date().toISOString();
    let stopCapture: StopCapture | null = null;
    let finalStatus: 'success' | 'error' | 'aborted' = 'success';
    let finalError = '';
    let finalResults: ABExperimentResult[] = [];

    appendLog(`run_id=${runId}`);
    stopCapture = startRuntimeLogCapture(appendLog);

    const runTasks = fixedTaskId
      ? tasks.filter((t) => t.id === fixedTaskId)
      : taskLimit && taskLimit > 0
        ? tasks.slice(0, Math.min(taskLimit, tasks.length))
        : tasks;
    if (fixedTaskId && runTasks.length === 0) {
      throw new Error(`Task ${fixedTaskId} not found`);
    }
    appendLog(`start: tasks=${runTasks.length}, models=${MODELS.length}`);

    try {
      const out: ABExperimentResult[] = [];
      for (const model of MODELS) {
        const modelPath = `${modelBaseDir.replace(/\/$/, '')}/${model.fileName}`;
        const modelUrl = `${window.location.origin}/@fs${encodeURI(modelPath)}`;
        appendLog(`model=${model.id} load=${model.fileName}`);
        const result = await runABExperiment(
          model.id,
          modelUrl,
          runTasks,
          cfg,
          (p) => {
            setProgress(p);
            if (p.phase === 'running-task') {
              appendLog(
                `model=${p.modelId} mode=${p.mode} task=${p.current}/${p.total} id=${p.taskId || '-'}`
              );
            }
          },
          nextController.signal,
          (event) => {
            appendLog(
              `model=${event.modelId} mode=${event.mode} task=${event.taskId} dataset=${event.dataset} output_tokens=${event.outputTokens} preview=${event.outputPreview}`
            );
          }
        );
        out.push(result);
        setResults([...out]);
        appendLog(
          `done model=${model.id} ttft_gain=${result.avgTtftGainPct.toFixed(2)}% tps_gain=${result.avgTpsGainPct.toFixed(2)}% invalid=${result.invalidTaskCount}`
        );
        if (result.invalidTaskIds.length) {
          appendLog(`invalid_ids=${result.invalidTaskIds.join(',')}`);
        }
      }
      setActiveModelId(out[0]?.modelId || MODELS[0].id);
      finalResults = out;
      appendLog('all done');
    } catch (e) {
      if ((e as Error).name !== 'AbortError') {
        setError((e as Error).message || String(e));
        finalStatus = 'error';
        finalError = (e as Error).message || String(e);
        appendLog(`error: ${finalError}`);
      } else {
        finalStatus = 'aborted';
        appendLog('stopped by user');
      }
    } finally {
      stopCapture?.();
      const endedAt = new Date().toISOString();
      saveJsonToLocalFile(`${runId}.json`, {
        runId,
        startedAt,
        endedAt,
        status: finalStatus,
        error: finalError || undefined,
        config: {
          modelBaseDir,
          mode: fixedTaskId ? 'smoke' : taskLimit && taskLimit > 0 ? 'subset' : 'all',
          taskLimit: taskLimit ?? null,
          fixedTaskId: fixedTaskId ?? null,
          prefixConfig: cfg,
        },
        logs: fullLogLinesRef.current,
        results: finalResults,
      });
      appendLog(`saved_report=${runId}.json`);
      setRunning(false);
      setController(null);
      setProgress(null);
    }
  };

  const onStop = () => controller?.abort();

  const sampleRows = pickSamples(activeResult?.deltas ?? []);

  return (
    <div className="page">
      <header>
        <h1>Agentic Browser Benchmark - Tree Cache A/B</h1>
        <p>
          目标：全量任务只看性能指标（TTFT、token/s）。A 组 Flat(no-cache)，B 组 Tree(useCache)，
          公共前缀=系统提示词 + 网页内容。
        </p>
      </header>

      <section className="panel">
        <h2>配置</h2>
        <label>
          <span>Model Base Directory</span>
          <input
            className="text-input"
            value={modelBaseDir}
            onChange={(e) => setModelBaseDir(e.target.value)}
            disabled={running}
          />
        </label>
        <label>
          <span>System Prompt</span>
          <textarea
            value={cfg.systemPrompt}
            onChange={(e) => setCfg((prev) => ({ ...prev, systemPrompt: e.target.value }))}
          />
        </label>

        <label>
          <span>Web Context Template</span>
          <textarea
            value={cfg.webContentTemplate}
            onChange={(e) => setCfg((prev) => ({ ...prev, webContentTemplate: e.target.value }))}
          />
        </label>

        <div className="row">
          <label>
            <span>Run Task Count</span>
            <input
              className="text-input"
              type="number"
              min={1}
              max={tasks.length}
              value={taskCountInput}
              onChange={(e) => setTaskCountInput(e.target.value)}
              disabled={running}
            />
          </label>
          <label>
            <span>nPredict</span>
            <input
              className="text-input"
              type="number"
              min={16}
              max={1024}
              value={cfg.nPredict}
              onChange={(e) =>
                setCfg((prev) => ({ ...prev, nPredict: Math.max(16, Number(e.target.value) || 128) }))
              }
              disabled={running}
            />
          </label>
          <label>
            <span>n_ctx</span>
            <input
              className="text-input"
              type="number"
              min={1024}
              max={16384}
              value={cfg.nCtx}
              onChange={(e) =>
                setCfg((prev) => ({ ...prev, nCtx: Math.max(1024, Number(e.target.value) || 4096) }))
              }
              disabled={running}
            />
          </label>
          <label>
            <span>n_batch</span>
            <input
              className="text-input"
              type="number"
              min={64}
              max={2048}
              value={cfg.nBatch}
              onChange={(e) =>
                setCfg((prev) => ({ ...prev, nBatch: Math.max(64, Number(e.target.value) || 512) }))
              }
              disabled={running}
            />
          </label>
        </div>

        <div className="row">
          <label>
            <input
              type="checkbox"
              checked={cfg.includeStartUrl}
              onChange={(e) => setCfg((prev) => ({ ...prev, includeStartUrl: e.target.checked }))}
            />
            Include start_url
          </label>
          <label>
            <input
              type="checkbox"
              checked={cfg.includeOpenUrls}
              onChange={(e) => setCfg((prev) => ({ ...prev, includeOpenUrls: e.target.checked }))}
            />
            Include open_url tabs
          </label>
        </div>

        <div className="run-row">
          <button onClick={() => void onRun(selectedTaskCount)} disabled={running}>
            {running
              ? 'Running...'
              : `Run Real A/B on first ${selectedTaskCount} tasks x ${MODELS.length} models`}
          </button>
          <button onClick={() => void onRun()} disabled={running}>
            Run All Tasks ({tasks.length})
          </button>
          <button onClick={() => void onRun(undefined, '1307')} disabled={running}>
            Run Task 1307 Smoke
          </button>
          <button className="stop-btn" onClick={onStop} disabled={!running}>
            Stop
          </button>
        </div>

        {progress ? (
          <p className="hint">
            [{progress.modelId}] {progress.mode} / {progress.phase}: {progress.current}/{progress.total}
            {progress.taskId ? ` / task=${progress.taskId}` : ''}
          </p>
        ) : null}
        {error ? <p className="error">{error}</p> : null}

        <div className="log-box">
          {logLines.length === 0 ? (
            <p className="hint">日志会在这里按行滚动显示。</p>
          ) : (
            logLines.map((line, idx) => (
              <div key={`${idx}-${line}`} className="log-line">
                {line}
              </div>
            ))
          )}
        </div>
      </section>

      {results.length ? (
        <>
          <section className="panel">
            <h2>模型切换</h2>
            <div className="row">
              {results.map((r) => (
                <button
                  key={r.modelId}
                  className={r.modelId === activeModelId ? 'tab-btn active' : 'tab-btn'}
                  onClick={() => setActiveModelId(r.modelId)}
                >
                  {r.modelId}
                </button>
              ))}
            </div>
          </section>

          <section className="panel metrics">
            <h2>总体指标</h2>
            <div className="grid">
              <div>
                <h3>A 组 (Flat + no-cache)</h3>
                <p>Avg TTFT: {ms(activeResult!.avgA.ttftMs)}</p>
                <p>Avg token/s: {tps(activeResult!.avgA.tokensPerSecond)}</p>
                <p>Avg n_reused: {activeResult!.avgA.nReused.toFixed(1)}</p>
              </div>
              <div>
                <h3>B 组 (Tree + useCache)</h3>
                <p>Avg TTFT: {ms(activeResult!.avgB.ttftMs)}</p>
                <p>Avg token/s: {tps(activeResult!.avgB.tokensPerSecond)}</p>
                <p>Avg n_reused: {activeResult!.avgB.nReused.toFixed(1)}</p>
              </div>
              <div>
                <h3>增益</h3>
                <p>TTFT gain: {pct(activeResult!.avgTtftGainPct)}</p>
                <p>token/s gain: {pct(activeResult!.avgTpsGainPct)}</p>
              </div>
            </div>
          </section>

          <section className="panel">
            <h2>优势分层（按 TTFT）</h2>
            <div className="grid compact">
              <div>高优势 ({'>='}35%): {grouped.high}</div>
              <div>中优势 (15%-35%): {grouped.medium}</div>
              <div>低优势 (0%-15%): {grouped.low}</div>
              <div>无优势 ({'<='}0%): {grouped.none}</div>
            </div>
          </section>

          <section className="panel">
            <h2>有效性与异常样本</h2>
            <p>Valid tasks: {activeResult!.validTaskCount} / {activeResult!.totalTasks}</p>
            <p>Invalid tasks: {activeResult!.invalidTaskCount}</p>
            <p>
              Invalid Task IDs:{' '}
              {activeResult!.invalidTaskIds.length
                ? activeResult!.invalidTaskIds.join(', ')
                : 'None'}
            </p>
            {activeResult!.invalidTasks.length ? (
              <table>
                <thead>
                  <tr>
                    <th>Task ID</th>
                    <th>Dataset</th>
                    <th>Reason</th>
                    <th>A Output</th>
                    <th>B Output</th>
                  </tr>
                </thead>
                <tbody>
                  {activeResult!.invalidTasks.map((item) => (
                    <tr key={`${item.dataset}-${item.taskId}`}>
                      <td>{item.taskId}</td>
                      <td>{item.dataset}</td>
                      <td>{item.reasons.join('; ')}</td>
                      <td>{item.aOutputPreview}</td>
                      <td>{item.bOutputPreview}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : null}
          </section>

          <section className="panel">
            <h2>样例任务（展示优势有高有低）</h2>
            {!sampleRows.length ? <p className="hint">当前无有效样本可展示。</p> : null}
            <table>
              <thead>
                <tr>
                  <th>Task ID</th>
                  <th>Dataset</th>
                  <th>TTFT A</th>
                  <th>TTFT B</th>
                  <th>TTFT Gain</th>
                  <th>token/s A</th>
                  <th>token/s B</th>
                  <th>token/s Gain</th>
                </tr>
              </thead>
              <tbody>
                {sampleRows.map((row) => (
                  <tr key={row.task.id}>
                    <td>{row.task.id}</td>
                    <td>{row.task.dataset}</td>
                    <td>{ms(row.a.ttftMs)}</td>
                    <td>{ms(row.b.ttftMs)}</td>
                    <td>{pct(row.ttftGainPct)}</td>
                    <td>{tps(row.a.tokensPerSecond)}</td>
                    <td>{tps(row.b.tokensPerSecond)}</td>
                    <td>{pct(row.tpsGainPct)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>

          <section className="panel">
            <h2>轻量正确性面板（string_match）</h2>
            <p>Checked tasks: {correctnessPanel.total}</p>
            <p>
              A hit: {correctnessPanel.aHitCount} / {correctnessPanel.total}
            </p>
            <p>
              B hit: {correctnessPanel.bHitCount} / {correctnessPanel.total}
            </p>
            <p>Both miss: {correctnessPanel.bothMiss.length}</p>
            {correctnessPanel.bothMiss.length ? (
              <table>
                <thead>
                  <tr>
                    <th>Task ID</th>
                    <th>Dataset</th>
                    <th>Reference Hints</th>
                    <th>A Output</th>
                    <th>B Output</th>
                    <th>Valid</th>
                  </tr>
                </thead>
                <tbody>
                  {correctnessPanel.bothMiss.slice(0, 12).map((row) => (
                    <tr key={`${row.dataset}-${row.taskId}`}>
                      <td>{row.taskId}</td>
                      <td>{row.dataset}</td>
                      <td>{row.refs.join(' | ')}</td>
                      <td>{row.aOutput}</td>
                      <td>{row.bOutput}</td>
                      <td>{String(row.isValid)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : null}
          </section>
        </>
      ) : null}
    </div>
  );
}
