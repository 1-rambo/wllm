/**
 * Browser entry for the shared-coordinator engine page (`/engines/wllm.html`).
 *
 * This is the `ours-*` arm. It installs one coordinator page for a logical
 * application block; the experiments create multiple sessions against it.
 * The page takes the same execution and lease knobs as the demo so an arm is
 * selected by URL rather than by a hidden compile-time branch.
 */

import coordinatorWorkerAssetUrl from '../workers/coordinator.shared-worker.ts?sharedworker&url';
import { BrowserLLMClient } from '../client/browser-llm-client';
import {
  observeFramePressure,
  type FramePressureSample,
} from '../client/frame-pressure-monitor';
import { InferenceBridge, type InferenceBridgeOptions } from '../client/inference-bridge';
import { coordinatorWorkerName } from '../coordinator/worker-name';
import { defaultModelPath, defaultRuntimeConfig } from '../config/demo-model';
import {
  readBooleanParam,
  readNumberParam,
  round2,
} from './engine-page-api';
import {
  readCoordinatorNumericOptions,
  writeCoordinatorNumericOptions,
} from './coordinator-numeric-options';
import { installWllmEnginePage, type WllmEnginePageApi } from './wllm-page';
import type { ExecutionConfig } from '../contracts';

const params = new URLSearchParams(location.search);
const scope = window as unknown as Record<string, unknown>;
const statusLine = document.querySelector('#engine-status');
const options = readInferenceOptions(params);
const priorityMode = params.get('priorityMode') === 'disabled' ? 'disabled' : 'enabled';
const initialPriority = params.get('initialPriority') === 'normal' ? 'normal' : 'high';
const modelUrl = options.modelUrl ?? defaultModelPath;
const requestedCoordinatorUrl = params.get('coordinatorWorkerUrl');
const coordinatorUrl = requestedCoordinatorUrl === null
  ? new URL(coordinatorWorkerAssetUrl, import.meta.url)
  : new URL(requestedCoordinatorUrl, import.meta.url);
if (coordinatorUrl.origin !== location.origin) {
  throw new Error('coordinator worker URL must use the engine page origin');
}
if (requestedCoordinatorUrl === null) {
  applyCoordinatorWorkerOptions(coordinatorUrl, options);
}
const workerName = await coordinatorWorkerName(coordinatorUrl);
scope.__wllmCoordinatorWorkerUrl = coordinatorUrl.href;
scope.__wllmCoordinatorWorkerName = workerName;
const worker = new SharedWorker(coordinatorUrl, {
  name: workerName,
  type: 'module',
});

const bridge = new InferenceBridge(
  worker.port as unknown as ConstructorParameters<typeof InferenceBridge>[0],
  options
);
let api: WllmEnginePageApi | null = null;
const client = new BrowserLLMClient(
  worker.port as unknown as ConstructorParameters<typeof BrowserLLMClient>[0],
  {
    onUnit: (event) => api?.handleEvent(event),
    onRuntime: (event) => api?.handleRuntimeEvent(event),
  }
);

function forwardRenderPressure(sample: FramePressureSample): void {
  void api?.reportRenderPressure(sample).catch(() => undefined);
}
const frameMonitor = observeFramePressure({
  canvas: () =>
    document.querySelector<HTMLCanvasElement>(
      '#gpu-render-probe, canvas[data-wllm-presentation]'
    ),
  onSample: (sample) => {
    if (document.visibilityState !== 'visible') {
      return;
    }
    forwardRenderPressure(sample);
  },
});

api = installWllmEnginePage(scope, {
  client: () => client,
  bridge: () => bridge,
  engineVersion: __WLLM_VERSION__,
  modelUrl,
  priorityMode,
  initialPriority,
  executionMode: options.executionMode ?? 'bounded',
  frameMetrics: () => frameMonitor.snapshot(),
  onSessionCreated: () => {
    frameMonitor.start();
  },
  scope,
});


// A dedicated worker is owned by its page. Tell the SharedWorker before that
// page goes away so it can drop the dead port and accept a replacement worker
// from a surviving tab instead of leaving in-flight calls unresolved.
window.addEventListener('pagehide', () => bridge.release(), { once: true });

scope.__wllmCoordinatorDiagnosticsProvider = () => client.diagnostics();
scope.__wllmRenderAware = options.renderAware ?? true;
scope.__wllmBurstPolicyHistory = [];

if (statusLine) {
  const render = () => {
    const status = api!.status();
    if (status.error !== null) {
      statusLine.textContent = `Coordinator engine failed: ${status.error}`;
    } else if (status.ready) {
      statusLine.textContent = `Coordinator ready on ${status.modelUrl ?? 'unknown model'}.`;
    } else if (status.loading) {
      statusLine.textContent =
        status.loadProgress === null
          ? 'Loading model'
          : `Loading model ${round2(status.loadProgress * 100)}%`;
    } else {
      statusLine.textContent = 'Coordinator ready; waiting for prewarm.';
    }
  };
  render();
  const timer = setInterval(() => {
    render();
    if (api!.status().ready || api!.status().error !== null) {
      clearInterval(timer);
    }
  }, 250);
}

function readInferenceOptions(
  source: URLSearchParams
): InferenceBridgeOptions {
  const options: InferenceBridgeOptions = {
    modelUrl: source.get('modelUrl') ?? defaultModelPath,
    modelLoadMode: source.get('modelLoadMode') === 'direct' ? 'direct' : 'cache',
    nCtx: readNumberParam(source, 'nCtx', defaultRuntimeConfig.nCtx),
    nBatch: readNumberParam(source, 'nBatch', defaultRuntimeConfig.nBatch),
    nParallel: readNumberParam(source, 'nParallel', defaultRuntimeConfig.nParallel),
    temperature: readNumberParam(source, 'temperature', 0.8),
    seed: readNumberParam(source, 'seed', 0),
    renderAware: readBooleanParam(source, 'renderAware', true),
    ...(source.get('executionMode') === null
      ? {}
      : { executionMode: normalizeExecutionMode(source.get('executionMode')!) }),
  };
  if (source.get('flashAttn') !== null) {
    options.flashAttn = readBooleanParam(source, 'flashAttn', true);
  }
  if (source.get('leaseWorkConservingBarrier') !== null) {
    options.leaseWorkConservingBarrier = readBooleanParam(
      source,
      'leaseWorkConservingBarrier',
      false
    );
  }
  if (source.get('leaseBackgroundBarrier') !== null) {
    options.leaseBackgroundBarrier = readBooleanParam(
      source,
      'leaseBackgroundBarrier',
      true
    );
  }
  if (source.get('batchHandoff') !== null) {
    options.batchHandoff = readBooleanParam(source, 'batchHandoff', false);
  }
  if (source.get('expressHandoff') !== null) {
    options.expressHandoff = readBooleanParam(source, 'expressHandoff', false);
  }
  if (source.get('preemptOnActivation') !== null) {
    options.preemptOnActivation = readBooleanParam(source, 'preemptOnActivation', false);
  }
  const replacementPolicy = source.get('replacementPolicy');
  if (
    replacementPolicy === 'hybrid' ||
    replacementPolicy === 'lru' ||
    replacementPolicy === 'ttl'
  ) {
    options.replacementPolicy = replacementPolicy;
  }
  const admissionPolicy = source.get('admissionPolicy');
  if (
    admissionPolicy === 'monotonic-bounded' ||
    admissionPolicy === 'reactive' ||
    admissionPolicy === 'immediate-reject'
  ) {
    options.admissionPolicy = admissionPolicy;
  }
  if (source.get('lifecycleProtection') !== null) {
    options.lifecycleProtection = readBooleanParam(
      source,
      'lifecycleProtection',
      true
    );
  }
  if (source.get('evictOnPause') !== null) {
    options.evictOnPause = readBooleanParam(source, 'evictOnPause', false);
  }
  const staleShape = source.get('leaseStaleSignalShape');
  if (staleShape === 'barrier' || staleShape === 'bounded') {
    options.leaseStaleSignalShape = staleShape;
  }
  readCoordinatorNumericOptions(source, options);
  // Keep the same lower bound as the per-tab baseline. Values below it can
  // turn ordinary WebGPU backpressure into an avoidable wasm abort.
  options.queueWaitTimeoutMs = Math.max(
    30_000,
    Math.floor(options.queueWaitTimeoutMs ?? 300_000)
  );
  // Execution-shape knobs use their bare names (backgroundDecodeTokens,
  // backgroundPrefillTokens, ...). An arm page can pin them per run without
  // rebuilding the demo, which is what lets a harness calibrate a hidden
  // tab's prefill chunk separately from its decode quantum.
  const execution: Partial<ExecutionConfig> = {};
  for (const key of [
    'maxPrefillTokens',
    'baseDecodeTokens',
    'foregroundDecodeTokens',
    'backgroundDecodeTokens',
    'backgroundPrefillTokens',
    'prefillMsPerToken',
    'maxUnitWallTimeMs',
    'unitWallTimeGraceMs',
    'targetUnitWallTimeMs',
    'minDecodeTokens',
    'maxDecodeTokens',
  ] as const) {
    const raw = source.get(key);
    if (raw === null || raw.trim() === '') {
      continue;
    }
    const value = Number(raw);
    if (Number.isFinite(value) && value > 0) {
      execution[key] = value;
    }
  }
  if (Object.keys(execution).length > 0) {
    options.execution = execution;
  }
  if (source.get('parkCompleted') !== null) {
    options.parkCompleted = readBooleanParam(source, 'parkCompleted', true);
  }
  const completedTier = source.get('completedTier');
  if (
    completedTier === '1' ||
    completedTier === '2' ||
    completedTier === '3' ||
    completedTier === 'L1' ||
    completedTier === 'L2' ||
    completedTier === 'L3' ||
    completedTier === 'rebuild'
  ) {
    options.completedTier = completedTier;
  }
  return options;
}

function normalizeExecutionMode(value: string): NonNullable<InferenceBridgeOptions['executionMode']> {
  switch (value.trim().toLowerCase()) {
    case 'unsliced':
      return 'unsliced';
    case 'deadline':
    case 'lease':
      return 'deadline';
    default:
      return 'bounded';
  }
}

function applyCoordinatorWorkerOptions(
  url: URL,
  options: InferenceBridgeOptions
): void {
  if (options.modelUrl) {
    url.searchParams.set('modelUrl', options.modelUrl);
  }
  for (const key of [
    'nCtx',
    'nBatch',
    'nParallel',
    'nThreads',
    'temperature',
    'seed',
    'nGpuLayers',
    'l1TokenCap',
    'l1NodeCap',
    'l2TokenCap',
    'continuationTokenCap',
    'l3ByteCap',
    'repairBudget',
    'webgpuCommandSubmitBatchSize',
    'webgpuMaxInflightBatches',
  ] as const) {
    const value = options[key];
    if (value !== undefined) {
      url.searchParams.set(key, String(value));
    }
  }
  if (options.renderAware !== undefined) {
    url.searchParams.set('renderAware', options.renderAware ? '1' : '0');
  }
  if (options.parkCompleted !== undefined) {
    url.searchParams.set('parkCompleted', options.parkCompleted ? '1' : '0');
  }
  if (options.completedTier !== undefined) {
    url.searchParams.set('completedTier', options.completedTier);
  }
  if (options.replacementPolicy !== undefined) {
    url.searchParams.set('replacementPolicy', options.replacementPolicy);
  }
  if (options.admissionPolicy !== undefined) {
    url.searchParams.set('admissionPolicy', options.admissionPolicy);
  }
  if (options.lifecycleProtection !== undefined) {
    url.searchParams.set(
      'lifecycleProtection',
      options.lifecycleProtection ? '1' : '0'
    );
  }
  if (options.evictOnPause !== undefined) {
    url.searchParams.set('evictOnPause', options.evictOnPause ? '1' : '0');
  }
  if (options.flashAttn !== undefined) {
    url.searchParams.set('flashAttn', options.flashAttn ? '1' : '0');
  }
  if (options.executionMode !== undefined) {
    url.searchParams.set('executionMode', options.executionMode);
  }
  if (options.leaseWorkConservingBarrier !== undefined) {
    url.searchParams.set(
      'leaseWorkConservingBarrier',
      options.leaseWorkConservingBarrier ? '1' : '0'
    );
  }
  if (options.leaseBackgroundBarrier !== undefined) {
    url.searchParams.set(
      'leaseBackgroundBarrier',
      options.leaseBackgroundBarrier ? '1' : '0'
    );
  }
  if (options.batchHandoff !== undefined) {
    url.searchParams.set('batchHandoff', options.batchHandoff ? '1' : '0');
  }
  if (options.expressHandoff !== undefined) {
    url.searchParams.set('expressHandoff', options.expressHandoff ? '1' : '0');
  }
  if (options.preemptOnActivation !== undefined) {
    url.searchParams.set(
      'preemptOnActivation',
      options.preemptOnActivation ? '1' : '0'
    );
  }
  if (options.leaseStaleSignalShape !== undefined) {
    url.searchParams.set(
      'leaseStaleSignalShape',
      options.leaseStaleSignalShape
    );
  }
  writeCoordinatorNumericOptions(url, options);
  for (const key of [
    'maxPrefillTokens',
    'baseDecodeTokens',
    'foregroundDecodeTokens',
    'backgroundDecodeTokens',
    'prefillMsPerToken',
    'maxUnitWallTimeMs',
    'unitWallTimeGraceMs',
    'targetUnitWallTimeMs',
    'minDecodeTokens',
    'maxDecodeTokens',
  ] as const) {
    const value = options.execution?.[key];
    if (value !== undefined) {
      url.searchParams.set(key, String(value));
    }
  }
}
