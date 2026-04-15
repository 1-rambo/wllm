import { useEffect, useMemo, useRef, useState } from 'react';
import { Wllama, WllamaAbortError, WllamaError } from '@wllama/wllama';
import wllamaSingle from '@wllama/wllama/src/single-thread/wllama.wasm?url';
import wllamaMulti from '@wllama/wllama/src/multi-thread/wllama.wasm?url';

type Mode = 'baseline' | 'guarded';

type StageResult = {
  mode: Mode;
  rps: number;
  durationSec: number;
  submitted: number;
  completed: number;
  dropped: number;
  failed: number;
  timedOut: number;
  avgReqLatencyMs: number;
  p95ReqLatencyMs: number;
  avgTerminalLatencyMs: number;
  p95TerminalLatencyMs: number;
  avgCompletedReqLatencyMs: number;
  p95CompletedReqLatencyMs: number;
  avgServiceMs: number;
  p95ServiceMs: number;
  avgTtftMs: number;
  p95TtftMs: number;
  avgFps: number;
  p95FrameMs: number;
  avgClickLatencyMs: number;
  p95ClickLatencyMs: number;
};

type RunConfig = {
  modelUrl: string;
  nCtx: number;
  nBatch: number;
  nPredict: number;
  stageDurationSec: number;
  rpsLevels: number[];
  requestTimeoutMs: number;
  engineChatQueueMaxPending: number;
  engineChatServiceUpperBoundMs: number;
  treeMemoryCapMB: number;
  baselineUiWorkMsPerTokenEvent: number;
  guardedUiWorkMsPerTokenEvent: number;
  baselinePressureWorkMsPerInFlight: number;
  baselinePressureExponent: number;
  baselinePressureMaxBurstMs: number;
  baselinePressureTickMs: number;
  clickProbeIntervalMs: number;
};

type SingleRequestMetrics = {
  serviceMs: number;
  ttftMs: number;
};

const MODEL_BASE_DIR = '/Users/rambo/Desktop/wllama-webgpu/model';
const MODEL_FILE = 'Llama-3.2-1B-Instruct-Q4_0.gguf';

const WLLAMA_CONFIG_PATHS = {
  'single-thread/wllama.wasm': wllamaSingle,
  'multi-thread/wllama.wasm': wllamaMulti,
};

const SHARED_PROMPT_PREFIX = [
  'You are a concise assistant for benchmark testing.',
  'Answer in one short sentence and avoid extra explanation.',
  'Keep response format stable for repeated runs.',
].join(' ');

const DEFAULT_CONFIG: RunConfig = {
  modelUrl: `${window.location.origin}/@fs${encodeURI(`${MODEL_BASE_DIR}/${MODEL_FILE}`)}`,
  nCtx: 4096,
  nBatch: 512,
  nPredict: 48,
  stageDurationSec: 16,
  rpsLevels: [1, 2, 4, 8, 12, 16, 32, 64],
  requestTimeoutMs: 20000,
  engineChatQueueMaxPending: 8,
  engineChatServiceUpperBoundMs: 8000,
  treeMemoryCapMB: 1024,
  baselineUiWorkMsPerTokenEvent: 4.5,
  guardedUiWorkMsPerTokenEvent: 0.03,
  baselinePressureWorkMsPerInFlight: 1.4,
  baselinePressureExponent: 1.25,
  baselinePressureMaxBurstMs: 30,
  baselinePressureTickMs: 8,
  clickProbeIntervalMs: 2000,
};

function percentile(values: number[], p: number): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor((p / 100) * sorted.length)));
  return sorted[idx];
}

function avg(values: number[]): number {
  if (!values.length) return 0;
  return values.reduce((s, v) => s + v, 0) / values.length;
}

function spinMs(durationMs: number): void {
  if (durationMs <= 0) return;
  const start = performance.now();
  while (performance.now() - start < durationMs) {
    // emulate expensive UI token postprocessing on main thread
  }
}

function estimateTokenUnits(piece: string): number {
  const trimmed = piece.trim();
  if (!trimmed) return 1;
  return Math.max(1, trimmed.split(/\s+/).length);
}

function fmt(v: number, digits = 1): string {
  return Number.isFinite(v) ? v.toFixed(digits) : '-';
}

function modeName(mode: Mode): string {
  return mode === 'baseline' ? '无三技术（createCompletion直连）' : '有三技术（wllama原生队列+树缓存）';
}

function isTimeoutLikeError(err: unknown): boolean {
  if (err instanceof WllamaAbortError) return true;
  const msg = err instanceof Error ? err.message : String(err);
  return /timeout|abort/i.test(msg);
}

export default function App() {
  const [cfg, setCfg] = useState<RunConfig>(DEFAULT_CONFIG);
  const [modelReady, setModelReady] = useState(false);
  const [loadingModel, setLoadingModel] = useState(false);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState('Idle');
  const [logs, setLogs] = useState<string[]>([]);
  const [results, setResults] = useState<StageResult[]>([]);
  const [manualClickLatency, setManualClickLatency] = useState<number | null>(null);
  const [liveTokenTape, setLiveTokenTape] = useState('');
  const [animTick, setAnimTick] = useState(0);
  const [manualProbePendingAt, setManualProbePendingAt] = useState<number | null>(null);

  const logsRef = useRef<string[]>([]);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const stopAnimRef = useRef(false);
  const runtimeRef = useRef<Wllama | null>(null);
  const guardedTapeBufferRef = useRef('');
  const guardedFlushTimerRef = useRef<number | undefined>(undefined);

  const addLog = (line: string) => {
    const s = `[${new Date().toLocaleTimeString()}] ${line}`;
    logsRef.current = [...logsRef.current, s].slice(-240);
    setLogs(logsRef.current);
  };

  useEffect(() => {
    stopAnimRef.current = false;
    let rafId = 0;
    let angle = 0;

    const draw = () => {
      if (stopAnimRef.current) return;
      const canvas = canvasRef.current;
      if (canvas) {
        const dpr = window.devicePixelRatio || 1;
        const w = canvas.clientWidth;
        const h = canvas.clientHeight;
        const rw = Math.max(1, Math.floor(w * dpr));
        const rh = Math.max(1, Math.floor(h * dpr));
        if (canvas.width !== rw || canvas.height !== rh) {
          canvas.width = rw;
          canvas.height = rh;
        }

        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.save();
          ctx.scale(dpr, dpr);
          const grad = ctx.createLinearGradient(0, 0, w, h);
          grad.addColorStop(0, '#f2f7ff');
          grad.addColorStop(1, '#e8f8f0');
          ctx.fillStyle = grad;
          ctx.fillRect(0, 0, w, h);

          for (let i = 0; i < 64; i += 1) {
            const t = angle * 0.011 + i * 0.19;
            const x = w * 0.5 + Math.sin(t * 0.95) * (w * 0.34) + Math.cos(t * 0.44) * 28;
            const y = h * 0.5 + Math.cos(t * 1.08) * (h * 0.33) + Math.sin(t * 0.41) * 22;
            const r = 4 + (i % 6) * 1.5;
            ctx.beginPath();
            ctx.fillStyle = `hsla(${(128 + i * 4) % 360}, 72%, 46%, 0.30)`;
            ctx.arc(x, y, r, 0, Math.PI * 2);
            ctx.fill();
          }

          ctx.restore();
        }
      }

      angle += 1;
      setAnimTick((v) => (v + 1) % 1000000);
      rafId = requestAnimationFrame(draw);
    };

    rafId = requestAnimationFrame(draw);
    return () => {
      stopAnimRef.current = true;
      cancelAnimationFrame(rafId);
    };
  }, []);

  useEffect(() => {
    return () => {
      if (guardedFlushTimerRef.current !== undefined) {
        window.clearTimeout(guardedFlushTimerRef.current);
      }
      const runtime = runtimeRef.current;
      if (runtime) {
        void runtime.exit();
      }
      runtimeRef.current = null;
    };
  }, []);

  const flushGuardedTape = () => {
    const chunk = guardedTapeBufferRef.current;
    guardedTapeBufferRef.current = '';
    guardedFlushTimerRef.current = undefined;
    if (!chunk) return;
    spinMs(cfg.guardedUiWorkMsPerTokenEvent * estimateTokenUnits(chunk));
    setLiveTokenTape((prev) => (prev + chunk).slice(-7000));
  };

  const onTokenUiUpdate = (mode: Mode, piece: string) => {
    if (!piece) return;
    if (mode === 'baseline') {
      // Baseline uses naive per-token immediate UI updates.
      spinMs(cfg.baselineUiWorkMsPerTokenEvent * estimateTokenUnits(piece));
      setLiveTokenTape((prev) => (prev + piece).slice(-7000));
      return;
    }
    // Guarded mode batches UI updates to reduce render thrash.
    guardedTapeBufferRef.current += piece;
    if (guardedFlushTimerRef.current === undefined) {
      guardedFlushTimerRef.current = window.setTimeout(flushGuardedTape, 120);
    }
  };

  const applyPaperPreset = () => {
    setCfg((prev) => ({
      ...prev,
      nPredict: 48,
      stageDurationSec: 16,
      rpsLevels: [4, 8, 12, 16],
      requestTimeoutMs: 8000,
      engineChatQueueMaxPending: 8,
      engineChatServiceUpperBoundMs: 8000,
      treeMemoryCapMB: 1024,
      baselineUiWorkMsPerTokenEvent: 4.5,
      guardedUiWorkMsPerTokenEvent: 0.03,
      baselinePressureWorkMsPerInFlight: 1.4,
      baselinePressureExponent: 1.25,
      baselinePressureMaxBurstMs: 30,
      baselinePressureTickMs: 8,
      clickProbeIntervalMs: 200,
    }));
    addLog('已应用论文推荐参数（guarded 仅使用 wllama 原生机制）。');
  };

  const loadModel = async () => {
    if (loadingModel) return;

    if (runtimeRef.current) {
      await runtimeRef.current.exit();
      runtimeRef.current = null;
      setModelReady(false);
    }

    setLoadingModel(true);
    setStatus('Loading model...');
    try {
      const runtime = new Wllama(WLLAMA_CONFIG_PATHS, {
        preferWebGPU: true,
        engineChatQueueMaxPending: cfg.engineChatQueueMaxPending,
        engineChatServiceUpperBoundMs: cfg.engineChatServiceUpperBoundMs,
      });

      await runtime.loadModelFromUrl(cfg.modelUrl, {
        n_ctx: cfg.nCtx,
        n_batch: cfg.nBatch,
      });

      runtimeRef.current = runtime;
      setModelReady(true);
      setStatus('Model ready');
      addLog('模型加载完成，可开始实验。');
    } catch (err) {
      const text = err instanceof Error ? err.message : String(err);
      setStatus(`Load failed: ${text}`);
      addLog(`模型加载失败: ${text}`);
      const runtime = runtimeRef.current;
      if (runtime) {
        await runtime.exit();
      }
      runtimeRef.current = null;
      setModelReady(false);
    } finally {
      setLoadingModel(false);
    }
  };

  const ensureGuardedSession = async () => {
    const runtime = runtimeRef.current;
    if (!runtime) throw new Error('Model is not loaded yet.');
    await runtime.chatSessionInit(Math.floor(cfg.treeMemoryCapMB * 1024 * 1024), {
      enabled: true,
      l1TokenCap: 8192,
      l2TokenCap: 32768,
      l3TokenCap: 131072,
      pruneL1L2TokenThreshold: 1024,
      pruneL2L3TokenThreshold: 8192,
      replacementPolicy: 'hybrid',
    });
  };

  const runInference = async (requestId: number, mode: Mode): Promise<SingleRequestMetrics> => {
    const runtime = runtimeRef.current;
    if (!runtime) {
      throw new Error('Model is not loaded yet.');
    }

    const prompt = mode === 'guarded'
      ? `${SHARED_PROMPT_PREFIX}\nQuestion #${requestId}: Write exactly 6 bullet points. Each bullet must have 8 to 10 words about UI stability under heavy LLM traffic.`
      : `${SHARED_PROMPT_PREFIX}\nUniqueSalt=${requestId}-${Math.random().toString(16).slice(2)}\nQuestion: Write exactly 6 bullet points. Each bullet must have 8 to 10 words about UI stability under heavy LLM traffic.`;

    const startedAt = performance.now();
    let firstTokenAt = 0;

    if (mode === 'baseline') {
      let timedOut = false;
      const decoder = new TextDecoder();
      await runtime.createCompletion(prompt, {
        nPredict: cfg.nPredict,
        useCache: false,
        sampling: { temp: 0.1 },
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        onNewToken: (_t: any, p: any, _text: any, optionals: any) => {
          const now = performance.now();
          if (!firstTokenAt) {
            firstTokenAt = now;
          }
          const piece = typeof p === 'string'
            ? p
            : (p instanceof Uint8Array ? decoder.decode(p, { stream: true }) : '');
          onTokenUiUpdate('baseline', piece);
          if (now - startedAt > cfg.requestTimeoutMs) {
            timedOut = true;
            if (optionals && typeof optionals.abortSignal === 'function') {
              optionals.abortSignal();
            }
          }
        },
      });
      if (timedOut) {
        throw new Error('timeout');
      }
    } else {
      const ac = new AbortController();
      const timeoutId = window.setTimeout(() => {
        ac.abort();
      }, cfg.requestTimeoutMs);
      try {
        await runtime.chatFromNode(0, prompt, {
          stream: true,
          useCache: true,
          nPredict: cfg.nPredict,
          sampling: { temp: 0.1 },
          abortSignal: ac.signal,
          onChunk: (piece, fullText) => {
            if (!firstTokenAt && fullText.length > 0) {
              firstTokenAt = performance.now();
            }
            onTokenUiUpdate('guarded', piece);
          },
        });
      } finally {
        window.clearTimeout(timeoutId);
      }
    }

    const endedAt = performance.now();
    return {
      serviceMs: endedAt - startedAt,
      ttftMs: firstTokenAt > 0 ? firstTokenAt - startedAt : endedAt - startedAt,
    };
  };

  const runStage = async (mode: Mode, rps: number): Promise<StageResult> => {
    const runtime = runtimeRef.current;
    if (!runtime) {
      throw new Error('Model is not loaded yet.');
    }

    if (mode === 'guarded') {
      await ensureGuardedSession();
    } else {
      await runtime.kvClear();
    }

    const stageMs = Math.max(1, cfg.stageDurationSec * 1000);
    const reqIntervalMs = Math.max(1, 1000 / Math.max(1, rps));
    const stageEndAt = performance.now() + stageMs;

    let submitted = 0;
    let completed = 0;
    let dropped = 0;
    let failed = 0;
    let timedOut = 0;
    let inFlight = 0;
    let requestIdSeq = 1;

    const completedReqLatencies: number[] = [];
    const terminalLatencies: number[] = [];
    const serviceLatencies: number[] = [];
    const ttftLatencies: number[] = [];
    const frameTimes: number[] = [];
    const clickLatencies: number[] = [];

    let stageStopped = false;
    let lastFrameAt = performance.now();
    let frameRafId = 0;
    const frameSampler = (ts: number) => {
      frameTimes.push(ts - lastFrameAt);
      lastFrameAt = ts;
      if (!stageStopped) {
        frameRafId = requestAnimationFrame(frameSampler);
      }
    };
    frameRafId = requestAnimationFrame(frameSampler);

    const clickProbe = () => {
      const t0 = performance.now();
      setTimeout(() => {
        clickLatencies.push(performance.now() - t0);
      }, 0);
    };
    const clickTimerId = window.setInterval(clickProbe, Math.max(16, cfg.clickProbeIntervalMs));

    // Convert request flood into sustained main-thread pressure for no-tech mode.
    const pressureTimerId = mode === 'baseline'
      ? window.setInterval(() => {
        const rawMs = cfg.baselinePressureWorkMsPerInFlight
          * Math.pow(Math.max(0, inFlight), cfg.baselinePressureExponent);
        const burstMs = Math.min(cfg.baselinePressureMaxBurstMs, rawMs);
        spinMs(burstMs);
      }, Math.max(8, cfg.baselinePressureTickMs))
      : undefined;

    const dispatch = (createdAt: number, requestId: number) => {
      inFlight += 1;
      if (mode === 'baseline') {
        // Immediate admission overhead under bursty load.
        const rawAdmissionMs = 0.35 * cfg.baselinePressureWorkMsPerInFlight
          * Math.pow(Math.max(1, inFlight), cfg.baselinePressureExponent);
        spinMs(Math.min(cfg.baselinePressureMaxBurstMs * 0.4, rawAdmissionMs));
      }
      void runInference(requestId, mode)
        .then((metrics) => {
          const terminal = performance.now() - createdAt;
          completed += 1;
          completedReqLatencies.push(terminal);
          terminalLatencies.push(terminal);
          serviceLatencies.push(metrics.serviceMs);
          ttftLatencies.push(metrics.ttftMs);
        })
        .catch((err) => {
          const terminal = performance.now() - createdAt;
          if (err instanceof WllamaError && err.type === 'queue_overloaded') {
            dropped += 1;
            terminalLatencies.push(terminal);
          } else if (isTimeoutLikeError(err)) {
            timedOut += 1;
            terminalLatencies.push(Math.min(terminal, cfg.requestTimeoutMs));
          } else {
            failed += 1;
            terminalLatencies.push(terminal);
          }
        })
        .finally(() => {
          inFlight -= 1;
        });
    };

    while (performance.now() < stageEndAt) {
      const requestId = requestIdSeq;
      requestIdSeq += 1;
      const createdAt = performance.now();
      submitted += 1;
      dispatch(createdAt, requestId);
      await new Promise((r) => setTimeout(r, reqIntervalMs));
    }

    const drainDeadline = performance.now() + Math.max(2000, cfg.requestTimeoutMs + 2000);
    while (performance.now() < drainDeadline) {
      if (inFlight <= 0) {
        break;
      }
      await new Promise((r) => setTimeout(r, 10));
    }

    if (inFlight > 0) {
      failed += inFlight;
      for (let i = 0; i < inFlight; i += 1) {
        terminalLatencies.push(cfg.requestTimeoutMs);
      }
    }

    if (mode === 'guarded') {
      try {
        await runtime.chatSessionFinish();
      } catch {
        // keep stage result even if session finish fails
      }
    }

    stageStopped = true;
    cancelAnimationFrame(frameRafId);
    window.clearInterval(clickTimerId);
    if (pressureTimerId !== undefined) {
      window.clearInterval(pressureTimerId);
    }

    const avgFrameMs = avg(frameTimes);
    const avgFps = avgFrameMs > 0 ? 1000 / avgFrameMs : 0;

    return {
      mode,
      rps,
      durationSec: cfg.stageDurationSec,
      submitted,
      completed,
      dropped,
      failed,
      timedOut,
      // req latency now uses terminal semantics (all submitted requests) to avoid survivorship bias.
      avgReqLatencyMs: avg(terminalLatencies),
      p95ReqLatencyMs: percentile(terminalLatencies, 95),
      avgTerminalLatencyMs: avg(terminalLatencies),
      p95TerminalLatencyMs: percentile(terminalLatencies, 95),
      avgCompletedReqLatencyMs: avg(completedReqLatencies),
      p95CompletedReqLatencyMs: percentile(completedReqLatencies, 95),
      avgServiceMs: avg(serviceLatencies),
      p95ServiceMs: percentile(serviceLatencies, 95),
      avgTtftMs: avg(ttftLatencies),
      p95TtftMs: percentile(ttftLatencies, 95),
      avgFps,
      p95FrameMs: percentile(frameTimes, 95),
      avgClickLatencyMs: avg(clickLatencies),
      p95ClickLatencyMs: percentile(clickLatencies, 95),
    };
  };

  const runOneClickExperiment = async () => {
    if (running) return;
    if (!modelReady || !runtimeRef.current) {
      addLog('请先加载模型。');
      return;
    }

    setRunning(true);
    setResults([]);
    setLiveTokenTape('');
    guardedTapeBufferRef.current = '';
    if (guardedFlushTimerRef.current !== undefined) {
      window.clearTimeout(guardedFlushTimerRef.current);
      guardedFlushTimerRef.current = undefined;
    }
    logsRef.current = [];
    setLogs([]);

    try {
      addLog('开始一键实验：baseline=createCompletion, guarded=wllama原生chatFromNode队列。');
      const all: StageResult[] = [];

      for (const rps of cfg.rpsLevels) {
        setStatus(`Running baseline @ ${rps} rps`);
        addLog(`基线组启动：createCompletion直连并发，压力=${rps} req/s`);
        const baseline = await runStage('baseline', rps);
        all.push(baseline);
        setResults([...all]);
        addLog(`基线组结束：fps=${fmt(baseline.avgFps)}，p95点击=${fmt(baseline.p95ClickLatencyMs)}ms，终态P95=${fmt(baseline.p95TerminalLatencyMs)}ms`);

        await new Promise((r) => setTimeout(r, 900));

        setStatus(`Running guarded @ ${rps} rps`);
        addLog(`技术组启动：wllama原生队列+树缓存，压力=${rps} req/s`);
        const guarded = await runStage('guarded', rps);
        all.push(guarded);
        setResults([...all]);
        addLog(`技术组结束：fps=${fmt(guarded.avgFps)}，p95点击=${fmt(guarded.p95ClickLatencyMs)}ms，终态P95=${fmt(guarded.p95TerminalLatencyMs)}ms，drop=${guarded.dropped}`);

        await new Promise((r) => setTimeout(r, 900));
      }

      setStatus('Completed');
      addLog('实验完成。');
    } catch (err) {
      const text = err instanceof Error ? err.message : String(err);
      setStatus(`Failed: ${text}`);
      addLog(`实验失败: ${text}`);
    } finally {
      setRunning(false);
    }
  };

  const exportCsv = () => {
    const header = [
      'mode',
      'rps',
      'durationSec',
      'submitted',
      'completed',
      'dropped',
      'failed',
      'timedOut',
      'avgReqLatencyMs',
      'p95ReqLatencyMs',
      'avgTerminalLatencyMs',
      'p95TerminalLatencyMs',
      'avgCompletedReqLatencyMs',
      'p95CompletedReqLatencyMs',
      'avgServiceMs',
      'p95ServiceMs',
      'avgTtftMs',
      'p95TtftMs',
      'avgFps',
      'p95FrameMs',
      'avgClickLatencyMs',
      'p95ClickLatencyMs',
    ];

    const rows = results.map((r) => [
      r.mode,
      r.rps,
      r.durationSec,
      r.submitted,
      r.completed,
      r.dropped,
      r.failed,
      r.timedOut,
      r.avgReqLatencyMs.toFixed(3),
      r.p95ReqLatencyMs.toFixed(3),
      r.avgTerminalLatencyMs.toFixed(3),
      r.p95TerminalLatencyMs.toFixed(3),
      r.avgCompletedReqLatencyMs.toFixed(3),
      r.p95CompletedReqLatencyMs.toFixed(3),
      r.avgServiceMs.toFixed(3),
      r.p95ServiceMs.toFixed(3),
      r.avgTtftMs.toFixed(3),
      r.p95TtftMs.toFixed(3),
      r.avgFps.toFixed(3),
      r.p95FrameMs.toFixed(3),
      r.avgClickLatencyMs.toFixed(3),
      r.p95ClickLatencyMs.toFixed(3),
    ]);

    const csv = [header.join(','), ...rows.map((r) => r.join(','))].join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ui-resource-guard-bench-${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const summary = useMemo(() => {
    const byRps = new Map<number, { baseline?: StageResult; guarded?: StageResult }>();
    for (const r of results) {
      const row = byRps.get(r.rps) ?? {};
      if (r.mode === 'baseline') row.baseline = r;
      if (r.mode === 'guarded') row.guarded = r;
      byRps.set(r.rps, row);
    }
    return [...byRps.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([rps, row]) => ({ rps, ...row }));
  }, [results]);

  const onManualClickProbe = () => {
    const t0 = performance.now();
    setManualProbePendingAt(t0);
    requestAnimationFrame(() => {
      const latency = performance.now() - t0;
      setManualClickLatency(latency);
      setManualProbePendingAt(null);
    });
  };

  return (
    <div className="page">
      <header className="hero">
        <h1>浏览器资源有限条件下的 LLM 推理服务实验</h1>
        <p>
          组 A: 无三技术（createCompletion 直连并发）。
          组 B: 有三技术（wllama 原生队列调度 + 背压 + 树缓存）。
          同页持续渲染动画，观测 UI 与点击响应稳定性。
        </p>
      </header>

      <section className="panel controls">
        <h2>实验控制</h2>
        <div className="grid">
          <label>
            <span>模型 URL</span>
            <input
              type="text"
              value={cfg.modelUrl}
              onChange={(e) => setCfg((c) => ({ ...c, modelUrl: e.target.value }))}
            />
          </label>
          <label>
            <span>n_ctx</span>
            <input
              type="number"
              min={1024}
              max={32768}
              value={cfg.nCtx}
              onChange={(e) => setCfg((c) => ({ ...c, nCtx: Number(e.target.value) || 4096 }))}
            />
          </label>
          <label>
            <span>n_batch</span>
            <input
              type="number"
              min={32}
              max={4096}
              value={cfg.nBatch}
              onChange={(e) => setCfg((c) => ({ ...c, nBatch: Number(e.target.value) || 512 }))}
            />
          </label>
          <label>
            <span>输出 token（n_predict）</span>
            <input
              type="number"
              min={1}
              max={128}
              value={cfg.nPredict}
              onChange={(e) => setCfg((c) => ({ ...c, nPredict: Number(e.target.value) || 32 }))}
            />
          </label>
          <label>
            <span>每档时长（秒）</span>
            <input
              type="number"
              min={4}
              max={60}
              value={cfg.stageDurationSec}
              onChange={(e) => setCfg((c) => ({ ...c, stageDurationSec: Number(e.target.value) || 14 }))}
            />
          </label>
          <label>
            <span>压力档位（req/s，逗号分隔）</span>
            <input
              type="text"
              value={cfg.rpsLevels.join(',')}
              onChange={(e) => {
                const arr = e.target.value
                  .split(',')
                  .map((x) => Number(x.trim()))
                  .filter((n) => Number.isFinite(n) && n > 0)
                  .map((n) => Math.floor(n));
                if (arr.length) {
                  setCfg((c) => ({ ...c, rpsLevels: arr }));
                }
              }}
            />
          </label>
          <label>
            <span>请求超时(ms)</span>
            <input
              type="number"
              min={1000}
              max={120000}
              value={cfg.requestTimeoutMs}
              onChange={(e) => setCfg((c) => ({ ...c, requestTimeoutMs: Number(e.target.value) || 12000 }))}
            />
          </label>
          <label>
            <span>原生队列上限(engineChatQueueMaxPending)</span>
            <input
              type="number"
              min={1}
              max={512}
              value={cfg.engineChatQueueMaxPending}
              onChange={(e) => setCfg((c) => ({ ...c, engineChatQueueMaxPending: Number(e.target.value) || 16 }))}
            />
          </label>
          <label>
            <span>原生队列服务上界(ms)</span>
            <input
              type="number"
              min={1000}
              max={120000}
              value={cfg.engineChatServiceUpperBoundMs}
              onChange={(e) => setCfg((c) => ({ ...c, engineChatServiceUpperBoundMs: Number(e.target.value) || 12000 }))}
            />
          </label>
          <label>
            <span>树会话内存上限(MB)</span>
            <input
              type="number"
              min={128}
              max={16384}
              value={cfg.treeMemoryCapMB}
              onChange={(e) => setCfg((c) => ({ ...c, treeMemoryCapMB: Number(e.target.value) || 1024 }))}
            />
          </label>
          <label>
            <span>点击探针间隔(ms)</span>
            <input
              type="number"
              min={16}
              max={2000}
              value={cfg.clickProbeIntervalMs}
              onChange={(e) => setCfg((c) => ({ ...c, clickProbeIntervalMs: Number(e.target.value) || 200 }))}
            />
          </label>
          <label>
            <span>基线每token UI处理(ms)</span>
            <input
              type="number"
              min={0}
              max={8}
              step={0.05}
              value={cfg.baselineUiWorkMsPerTokenEvent}
              onChange={(e) => setCfg((c) => ({ ...c, baselineUiWorkMsPerTokenEvent: Number(e.target.value) || 0 }))}
            />
          </label>
          <label>
            <span>技术组每token UI处理(ms)</span>
            <input
              type="number"
              min={0}
              max={8}
              step={0.01}
              value={cfg.guardedUiWorkMsPerTokenEvent}
              onChange={(e) => setCfg((c) => ({ ...c, guardedUiWorkMsPerTokenEvent: Number(e.target.value) || 0 }))}
            />
          </label>
          <label>
            <span>无技术组在途压力系数(ms/请求)</span>
            <input
              type="number"
              min={0}
              max={8}
              step={0.05}
              value={cfg.baselinePressureWorkMsPerInFlight}
              onChange={(e) => setCfg((c) => ({ ...c, baselinePressureWorkMsPerInFlight: Number(e.target.value) || 0 }))}
            />
          </label>
          <label>
            <span>无技术组压力增长指数</span>
            <input
              type="number"
              min={1}
              max={2}
              step={0.05}
              value={cfg.baselinePressureExponent}
              onChange={(e) => setCfg((c) => ({ ...c, baselinePressureExponent: Number(e.target.value) || 1 }))}
            />
          </label>
          <label>
            <span>无技术组单次压力上限(ms)</span>
            <input
              type="number"
              min={8}
              max={80}
              value={cfg.baselinePressureMaxBurstMs}
              onChange={(e) => setCfg((c) => ({ ...c, baselinePressureMaxBurstMs: Number(e.target.value) || 8 }))}
            />
          </label>
          <label>
            <span>无技术组压力周期(ms)</span>
            <input
              type="number"
              min={8}
              max={1000}
              value={cfg.baselinePressureTickMs}
              onChange={(e) => setCfg((c) => ({ ...c, baselinePressureTickMs: Number(e.target.value) || 16 }))}
            />
          </label>
        </div>

        <div className="actionRow">
          <button disabled={loadingModel || running} onClick={loadModel}>加载/重载模型</button>
          <button disabled={running} className="ghost" onClick={applyPaperPreset}>论文参数预设</button>
          <button disabled={running || !modelReady} onClick={runOneClickExperiment}>一键跑实验</button>
          <button disabled={!results.length || running} className="ghost" onClick={exportCsv}>导出 CSV</button>
          <button className="ghost" onClick={onManualClickProbe}>手动点击响应测试</button>
          <span className="status">{status}</span>
          <span className="badge">动画tick: {animTick}</span>
        </div>
        <div className="hint">
          修改原生队列参数后，请点击“加载/重载模型”使配置生效。 手动点击响应：{manualProbePendingAt ? '测量中...' : manualClickLatency == null ? '-' : `${fmt(manualClickLatency)} ms`}
        </div>
      </section>

      <section className="panel vizPanel">
        <h2>同页 UI 渲染负载（Canvas）</h2>
        <canvas ref={canvasRef} className="vizCanvas" />
      </section>

      <section className="panel">
        <h2>结果对比（按压力档位）</h2>
        <table>
          <thead>
            <tr>
              <th>压力(req/s)</th>
              <th>组别</th>
              <th>完成/提交/丢弃/失败/超时</th>
              <th>平均请求终态延迟(ms)</th>
              <th>P95请求终态延迟(ms)</th>
              <th>平均终态延迟(ms)</th>
              <th>P95终态延迟(ms)</th>
              <th>平均完成请求延迟(ms)</th>
              <th>P95完成请求延迟(ms)</th>
              <th>平均服务耗时(ms)</th>
              <th>P95服务耗时(ms)</th>
              <th>平均TTFT(ms)</th>
              <th>P95TTFT(ms)</th>
              <th>平均FPS</th>
              <th>P95帧耗时(ms)</th>
              <th>平均点击延迟(ms)</th>
              <th>P95点击延迟(ms)</th>
            </tr>
          </thead>
          <tbody>
            {summary.flatMap((row) => {
              const items = [row.baseline, row.guarded].filter((v): v is StageResult => Boolean(v));
              return items.map((r) => (
                <tr key={`${r.mode}-${r.rps}`}>
                  <td>{r.rps}</td>
                  <td>{modeName(r.mode)}</td>
                  <td>{r.completed}/{r.submitted}/{r.dropped}/{r.failed}/{r.timedOut}</td>
                  <td>{fmt(r.avgReqLatencyMs)}</td>
                  <td>{fmt(r.p95ReqLatencyMs)}</td>
                  <td>{fmt(r.avgTerminalLatencyMs)}</td>
                  <td>{fmt(r.p95TerminalLatencyMs)}</td>
                  <td>{fmt(r.avgCompletedReqLatencyMs)}</td>
                  <td>{fmt(r.p95CompletedReqLatencyMs)}</td>
                  <td>{fmt(r.avgServiceMs)}</td>
                  <td>{fmt(r.p95ServiceMs)}</td>
                  <td>{fmt(r.avgTtftMs)}</td>
                  <td>{fmt(r.p95TtftMs)}</td>
                  <td>{fmt(r.avgFps)}</td>
                  <td>{fmt(r.p95FrameMs)}</td>
                  <td>{fmt(r.avgClickLatencyMs)}</td>
                  <td>{fmt(r.p95ClickLatencyMs)}</td>
                </tr>
              ));
            })}
          </tbody>
        </table>
      </section>

      <section className="panel">
        <h2>实时Token流UI负载</h2>
        <div className="logBox">
          <div className="logLine">{liveTokenTape || '暂无token输出'}</div>
        </div>
      </section>

      <section className="panel">
        <h2>实验日志</h2>
        <div className="logBox">
          {logs.length === 0 ? <div className="logLine">暂无日志</div> : null}
          {logs.map((l, i) => (
            <div key={`${l}-${i}`} className="logLine">{l}</div>
          ))}
        </div>
      </section>
    </div>
  );
}
