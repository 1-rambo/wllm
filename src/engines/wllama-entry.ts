/**
 * Browser entry for the wllama baseline page (`/engines/wllama.html`).
 *
 * The page takes its knobs from the query string so the harness can pin the
 * model and context size per run, and it installs exactly one engine on
 * `window.__wllmEnginePage`.
 */

import wasmUrl from '@wllama/wllama/esm/wasm/wllama.wasm?url';
import compatWasmUrl from '@wllama/wllama-compat/wasm/wllama.wasm?url';
import compatWorkerCode from '@wllama/wllama-compat/wasm/wllama.js?raw';
import { Wllama } from '@wllama/wllama/esm/index.js';
import { observeFramePressure } from '../client/frame-pressure-monitor';
import { defaultModelPath, defaultRuntimeConfig } from '../config/demo-model';
import { readBooleanParam, readNumberParam, round2 } from './engine-page-api';
import {
  installWllamaEnginePage,
  type WllamaPageRuntime,
} from './wllama-page';

const params = new URLSearchParams(location.search);
type WllamaLoadModelParams = NonNullable<Parameters<Wllama['loadModel']>[1]>;
const scope = window as unknown as Record<string, unknown>;
const modelUrl = params.get('modelUrl') ?? defaultModelPath;
const modelLoadMode =
  params.get('modelLoadMode') === 'direct' ? 'direct' : 'cache';
const observeFrames = readBooleanParam(params, 'frameMonitor', true);
const queueWaitTimeoutMs = Math.max(
  30_000,
  Math.floor(readNumberParam(params, 'queueWaitTimeoutMs', 300_000))
);
// Keep each tab's WebGPU queue finite. The fork otherwise permits an
// unlimited number of submitted batches, so ten tabs can starve one another
// even when every tab is making forward progress on its own.
const webgpuCommandSubmitBatchSize = Math.max(
  1,
  Math.floor(readNumberParam(params, 'webgpuCommandSubmitBatchSize', 32))
);
const webgpuMaxInflightBatches = Math.max(
  1,
  Math.floor(readNumberParam(params, 'webgpuMaxInflightBatches', 4))
);
const statusLine = document.querySelector('#engine-status');

const runtimeLogTail: string[] = [];
function recordRuntimeLog(level: string, values: unknown[]): void {
  const text = values.map(formatRuntimeLogValue).join(' ').slice(0, 2000);
  runtimeLogTail.push(`[${level}] ${text}`);
  if (runtimeLogTail.length > 80) runtimeLogTail.shift();
}

function formatRuntimeLogValue(value: unknown): string {
  if (value instanceof Error) return value.stack ?? value.message;
  return typeof value === 'string' ? value : String(value);
}

const runtimeLogger = {
  debug: () => undefined,
  log: (...values: unknown[]) => recordRuntimeLog('log', values),
  warn: (...values: unknown[]) => recordRuntimeLog('warn', values),
  error: (...values: unknown[]) => recordRuntimeLog('error', values),
};

const frameMonitor = observeFrames
  ? observeFramePressure({
      canvas: () =>
        document.querySelector<HTMLCanvasElement>(
          '#gpu-render-probe, canvas[data-wllm-presentation]'
        ),
      onSample: () => undefined,
    })
  : null;

frameMonitor?.start();

const api = installWllamaEnginePage(scope, {
  runtimeFactory: () =>
    createRuntime(
      wasmUrl,
      compatWasmUrl,
      compatWorkerCode,
      queueWaitTimeoutMs
    ),
  engineVersion: __WLLM_VERSION__,
  modelUrl,
  nCtx: readNumberParam(params, 'nCtx', defaultRuntimeConfig.nCtx),
  nBatch: readNumberParam(params, 'nBatch', defaultRuntimeConfig.nBatch),
  nParallel: readNumberParam(params, 'nParallel', defaultRuntimeConfig.nParallel),
  temperature: readNumberParam(params, 'temperature', 0.8),
  seed: readNumberParam(params, 'seed', 0),
  useCache: readBooleanParam(params, 'useCache', true),
  modelLoadMode,
  cachePrompt: readBooleanParam(params, 'cachePrompt', true),
  webgpuSubmissionPolicy: {
    commandSubmitBatchSize: webgpuCommandSubmitBatchSize,
    maxInflightBatches: webgpuMaxInflightBatches,
  },
  nThreads: readNumberParam(params, 'nThreads', defaultThreads()),
  frameMetrics: () =>
    frameMonitor?.snapshot() ?? null,
  errorDetail: () =>
    runtimeLogTail.length === 0 ? null : runtimeLogTail.slice(-16).join('\n'),
});

function describeStatus(): string {
  const status = api.status();
  if (!status.loading && !status.ready && status.error === null) {
    return 'Engine page ready; waiting for prewarm.';
  }
  if (status.error !== null) {
    return `Engine page failed: ${status.error}`;
  }
  if (status.loading) {
    return `Loading model${status.loadProgress === null ? '' : ` ${round2(status.loadProgress * 100)}%`}`;
  }
  return `Model ready on ${status.modelUrl ?? 'unknown model'}.`;
}

if (statusLine) {
  statusLine.textContent = describeStatus();
  const timer = setInterval(() => {
    statusLine.textContent = describeStatus();
    if (api.status().ready || api.status().error !== null) {
      clearInterval(timer);
    }
  }, 250);
}

/**
 * The pinned fork needs the Asyncify build on browsers without JSPI, and the
 * fetching path is the same one the demo uses, so the baseline is not given a
 * different wasm deployment from the one wllama ships.
 */
function createRuntime(
  primaryWasmUrl: string,
  fallbackWasmUrl: string,
  fallbackWorkerCode: string,
  queueWaitTimeoutMs: number
): WllamaPageRuntime {
  const runtime = new Wllama(
    { default: primaryWasmUrl },
    {
      logger: runtimeLogger,
      allowOffline: true,
      parallelDownloads: 3,
      // The per-tab baseline can legitimately leave a large prefill queued
      // behind other browser GPU work beyond the upstream safety timeout.
      runtimeEnv: {
        GGML_WEBGPU_WAIT_TIMEOUT_MS: String(queueWaitTimeoutMs),
      },
    }
  );
  runtime.setCompat({
    wasm: fallbackWasmUrl,
    worker: { code: fallbackWorkerCode },
  });
  const runtimeWithDirectLoad = runtime as unknown as WllamaPageRuntime;
  runtimeWithDirectLoad.loadModelDirect = async (source, params = {}) => {
    const requestInit: RequestInit = {};
    const signal = params.signal;
    if (signal instanceof AbortSignal) requestInit.signal = signal;
    const response = await fetch(source, requestInit);
    if (!response.ok || !response.body) {
      throw new Error(`Failed to fetch ${source}: HTTP ${response.status}`);
    }
    const blob = await response.blob();
    const progressCallback = params.progressCallback as
      | ((progress: { loaded: number; total: number }) => void)
      | undefined;
    progressCallback?.({ loaded: blob.size, total: blob.size });
    await runtime.loadModel([blob], params as WllamaLoadModelParams);
  };
  return runtime as unknown as WllamaPageRuntime;
}

function defaultThreads(): number {
  const cores = globalThis.navigator?.hardwareConcurrency ?? 4;
  return Math.max(1, Math.min(8, cores));
}
