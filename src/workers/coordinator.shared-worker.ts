import {
  SharedWorkerCoordinator,
  type CompletedTier,
} from '../coordinator/shared-worker-coordinator';
import { startCoordinatorPort } from '../coordinator/coordinator-host';
import {
  BrowserOpfsBlobStore,
  ResilientBlobStore,
} from '../state/checkpoint-store';
import { DeferredNativeEngine } from '../native/deferred-native-engine';
import type {
  ExecutionConfig,
  NativeRuntimeEvent,
  StateManagerConfig,
} from '../contracts';
import type { ExecutionMode } from '../contracts';
import type { FrameDeadlineLeaseOptions } from '../execution/frame-deadline-lease';
import type { MessagePortLike } from '../transport/protocol';
import {
  MessagePortRpcClient,
  NativeEngineProxy,
} from '../transport/native-worker';
import { modelIdentity } from '../utils/model-identity';
import {
  defaultModelPath,
  defaultRuntimeConfig,
} from '../config/demo-model';

const scope = self as unknown as SharedWorkerGlobalScope;
const params = new URL(scope.location.href).searchParams;
const coordinatorInstanceId =
  typeof crypto.randomUUID === 'function'
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2)}`;

function executionConfigForMode(
  mode: ExecutionMode | undefined
): Partial<ExecutionConfig> {
  if (mode === 'deadline') {
    return {
      // The clear-page shape is a plain one-shot decoder: one unit may
      // cover the whole response, exactly like a per-tab engine would.
      // These are ceilings, not quanta. The lease still hands a
      // pressure-shaped unit its own small token budget, and the WebGPU
      // burst policy still narrows the in-flight queue at a graph
      // boundary while frames are late.
      maxPrefillTokens: 4_096,
      baseDecodeTokens: 4_096,
      foregroundDecodeTokens: 4_096,
      backgroundDecodeTokens: 8,
      prefillMsPerToken: 2,
      // A work-conserving unit can cover a full response, so the host
      // watchdog has to sit above it while still bounding a stalled
      // graph. Pressure units stay short because the lease caps them.
      maxUnitWallTimeMs: 30_000,
      // One contended WebGPU token can finish only after the shared queue
      // drains. The lease uses the token boundary; this is the hard guard
      // for a genuinely stalled graph.
      unitWallTimeGraceMs: 5_000,
      adaptiveUnits: true,
      targetUnitWallTimeMs: 400,
      nativeTimeBudgetEnabled: true,
      minDecodeTokens: 1,
      maxDecodeTokens: 4_096,
    };
  }
  if (mode === 'unsliced') {
    return {
      maxPrefillTokens: 4_096,
      // The unsliced control has to let one request run to completion. The
      // previous fallback kept the bounded 12-token / 24 ms shape, which made
      // the ablation a second bounded implementation rather than a control.
      baseDecodeTokens: 4_096,
      foregroundDecodeTokens: 4_096,
      backgroundDecodeTokens: 4_096,
      prefillMsPerToken: 2,
      maxUnitWallTimeMs: 30_000,
      unitWallTimeGraceMs: 30_000,
      adaptiveUnits: false,
      targetUnitWallTimeMs: 30_000,
      nativeTimeBudgetEnabled: false,
      minDecodeTokens: 1,
      maxDecodeTokens: 4_096,
    };
  }
  return {
    maxPrefillTokens: 64,
    baseDecodeTokens: 12,
    foregroundDecodeTokens: 12,
    backgroundDecodeTokens: 4,
    prefillMsPerToken: 2,
    maxUnitWallTimeMs: 40,
    unitWallTimeGraceMs: 1_000,
    adaptiveUnits: true,
    targetUnitWallTimeMs: 24,
    nativeTimeBudgetEnabled: true,
    minDecodeTokens: 1,
    maxDecodeTokens: 64,
  };
}

export interface InferenceWorkerConfig {
  modelUrl: string;
  modelLoadMode?: 'cache' | 'direct';
  nCtx: number;
  nBatch: number;
  nParallel: number;
  nThreads?: number;
  temperature?: number;
  seed?: number;
  nGpuLayers?: number;
  flashAttn?: boolean;
  webgpuCommandSubmitBatchSize?: number;
  webgpuMaxInflightBatches?: number;
  queueWaitTimeoutMs?: number;
}
const inferenceConfig: InferenceWorkerConfig = {
  modelUrl: resolveModelUrl(
    params.get('modelUrl') ?? defaultModelPath,
    scope.location.origin
  ),
  modelLoadMode: params.get('modelLoadMode') === 'direct' ? 'direct' : 'cache',
  nCtx: numberParam(params.get('nCtx'), defaultRuntimeConfig.nCtx),
  nBatch: numberParam(params.get('nBatch'), defaultRuntimeConfig.nBatch),
  nParallel: numberParam(
    params.get('nParallel'),
    defaultRuntimeConfig.nParallel
  ),
  queueWaitTimeoutMs: numberParam(
    params.get('queueWaitTimeoutMs'),
    300_000
  ),
};
// Sampling settings must reach the engine through the same config channel as
// the load knobs, otherwise the arm silently samples at the runtime default.
const samplingTemperature = optionalNumber(params.get('temperature'));
if (samplingTemperature !== undefined) {
  inferenceConfig.temperature = samplingTemperature;
}
const samplingSeed = optionalNumber(params.get('seed'));
if (samplingSeed !== undefined) {
  inferenceConfig.seed = samplingSeed;
}
const nThreads = optionalNumber(params.get('nThreads'));
if (nThreads !== undefined) {
  inferenceConfig.nThreads = nThreads;
}
const nGpuLayers = optionalNumber(params.get('nGpuLayers'));
if (nGpuLayers !== undefined) {
  inferenceConfig.nGpuLayers = nGpuLayers;
}
const flashAttn = optionalBoolean(params.get('flashAttn'));
if (flashAttn !== undefined) {
  inferenceConfig.flashAttn = flashAttn;
}
const webgpuCommandSubmitBatchSize = optionalNumber(
  params.get('webgpuCommandSubmitBatchSize')
);
if (webgpuCommandSubmitBatchSize !== undefined) {
  inferenceConfig.webgpuCommandSubmitBatchSize = webgpuCommandSubmitBatchSize;
}
const webgpuMaxInflightBatches = optionalNumber(
  params.get('webgpuMaxInflightBatches')
);
if (webgpuMaxInflightBatches !== undefined) {
  inferenceConfig.webgpuMaxInflightBatches = webgpuMaxInflightBatches;
}

const native = new DeferredNativeEngine();
let lastRuntime: NativeRuntimeEvent | null = null;
let coordinator!: SharedWorkerCoordinator;
// A SharedWorker cannot spawn a Worker. The dedicated native engine is
// therefore owned by one page and connected over a transferred MessagePort.
const store = new ResilientBlobStore(
  new BrowserOpfsBlobStore('wllm-demo-checkpoints')
);
const executionMode = executionModeParam(params.get('executionMode'));
// `0` is meaningful here (one self contained turn per request), so the cap is
// only forwarded when the harness actually set it.
const continuationTokenCap = optionalNumber(
  params.get('continuationTokenCap')
);
const leaseBatchSize = optionalNumber(params.get('leaseBatchSize'));
const leaseHandoffBatchSize = optionalNumber(
  params.get('leaseHandoffBatchSize')
);
const handoffPrefillTokens = optionalNumber(
  params.get('handoffPrefillTokens')
);
const schedulerForegroundBurstUnits = optionalNumber(
  params.get('schedulerForegroundBurstUnits')
);
const schedulerResidentBurstUnits = optionalNumber(
  params.get('schedulerResidentBurstUnits')
);
const schedulerRetentionWindowMs = optionalNumber(
  params.get('schedulerRetentionWindowMs')
);
// The resident pool has to hold the working set the arm claims, otherwise
// every finished turn is parked and the next turn of that session re-prefills
// its whole prefix. Scale with the context and slot count, then clamp so a
// wide-context sweep cannot silently reserve gigabytes of KV.
const residentSlots = Math.max(
  1,
  Math.min(4, Math.floor(inferenceConfig.nParallel))
);
const l1TokenCap = numberParam(
  params.get('l1TokenCap'),
  Math.min(16_384, Math.max(4_096, inferenceConfig.nCtx * residentSlots))
);
const l2TokenCap = numberParam(
  params.get('l2TokenCap'),
  Math.max(4_096, l1TokenCap * 2)
);
// One resident session needs one serving slot, so the pool is capped by the
// native slot count as well as by tokens.
const l1NodeCap = numberParam(params.get('l1NodeCap'), residentSlots);
// A stateless turn has no KV prefix to reuse. Retaining the completed slot
// would only create parking pressure under concurrent long-lived
// sessions, so the default completion policy is to rebuild that turn.
// A stateless turn still shares a prefix with the turn before it in the same
// session, so the finished slot stays resident and the next turn reuses that
// prefix. Dropping the slot is an explicit lifecycle choice, not the default.
const initialCompletedTier = completedTierParam(params.get('completedTier'));
// Lease knobs a harness can sweep without rebuilding the demo. Only values
// that were actually set are forwarded, so the controller keeps its own
// defaults everywhere else.
const leaseParamMap = {
  leaseMaxBudgetMs: 'maxBudgetMs',
  leaseMaxDecodeTokens: 'maxDecodeTokens',
  leaseMaxPauseMs: 'maxPauseMs',
  leasePausePerDebtMs: 'pausePerDebtMs',
  leaseMinPressureDecodeTokens: 'minPressureDecodeTokens',
  leaseBackgroundDecodeTokens: 'backgroundDecodeTokens',
  leaseMinPressurePrefillTokens: 'minPressurePrefillTokens',
  leaseMaxPrefillTokens: 'maxPrefillTokens',
  leaseMaxPressurePrefillTokens: 'maxPressurePrefillTokens',
  leasePrefillChunkTokens: 'prefillChunkOverride',
  leaseMaxCommandBatchSize: 'maxCommandBatchSize',
  leaseMinPressureCommandBatchSize: 'minPressureCommandBatchSize',
  leaseInflightCap: 'inflightCap',
  leaseMildPressureFramePeriods: 'mildPressureFramePeriods',
  leaseSeverePressureFramePeriods: 'severePressureFramePeriods',
  leaseQueueWaitToleranceMs: 'queueWaitToleranceMs',
  leaseQueueBackpressureSamples: 'queueBackpressureSamples',
  leaseQueueReliefSamples: 'queueReliefSamples',
  leaseWorkConservingQuantumMs: 'workConservingQuantumMs',
} as const satisfies Record<string, keyof FrameDeadlineLeaseOptions>;
const leaseOptions: Partial<FrameDeadlineLeaseOptions> = {};
for (const [param, optionKey] of Object.entries(leaseParamMap)) {
  const parsed = optionalNumber(params.get(param));
  if (parsed !== undefined && parsed >= 0) {
    leaseOptions[optionKey] = parsed;
  }
}
// Boolean lease switches cannot ride the numeric sweep table above.
const leaseWorkConservingBarrier = optionalBoolean(
  params.get('leaseWorkConservingBarrier')
);
if (leaseWorkConservingBarrier !== undefined) {
  leaseOptions.workConservingBarrier = leaseWorkConservingBarrier;
}
const leaseBackgroundBarrier = optionalBoolean(
  params.get('leaseBackgroundBarrier')
);
if (leaseBackgroundBarrier !== undefined) {
  leaseOptions.backgroundBarrier = leaseBackgroundBarrier;
}
const leaseStaleSignalShape = params.get('leaseStaleSignalShape');
if (leaseStaleSignalShape === 'barrier' || leaseStaleSignalShape === 'bounded') {
  leaseOptions.staleSignalShape = leaseStaleSignalShape;
}
// The priority lane leaves the local barrier without exceeding the queue
// shape the engine actually deployed. Otherwise a 64/unlimited lease can
// silently differ from the configured 32/4 submission path under GPU
// contention.
if (
  leaseOptions.maxCommandBatchSize === undefined &&
  webgpuCommandSubmitBatchSize !== undefined
) {
  leaseOptions.maxCommandBatchSize = webgpuCommandSubmitBatchSize;
}
if (
  leaseOptions.inflightCap === undefined &&
  webgpuMaxInflightBatches !== undefined
) {
  leaseOptions.inflightCap = webgpuMaxInflightBatches;
}
coordinator = new SharedWorkerCoordinator(native, store, {
  // Keeping a finished turn resident is what lets the next turn of the same
  // session reuse its KV prefix instead of paying the whole prefill again.
  // Parking to L2 stays the eviction and lifecycle path, not the default.
  parkCompleted: booleanParam(params.get('parkCompleted'), false),
  ...(executionMode === undefined ? {} : { executionMode }),
  renderAware: booleanParam(params.get('renderAware'), true),
  renderGate: booleanParam(params.get('renderGate'), false),
  ...(initialCompletedTier ? { completedTier: initialCompletedTier } : {}),
  ...(continuationTokenCap === undefined ? {} : { continuationTokenCap }),
  ...(leaseBatchSize === undefined ? {} : { leaseBatchSize }),
  batchHandoff: booleanParam(params.get('batchHandoff'), false),
  expressHandoff: booleanParam(params.get('expressHandoff'), false),
  preemptOnActivation: booleanParam(
    params.get('preemptOnActivation'),
    false
  ),
  ...(leaseHandoffBatchSize === undefined
    ? {}
    : { handoffBatchSize: leaseHandoffBatchSize }),
  ...(handoffPrefillTokens === undefined
    ? {}
    : { handoffPrefillTokens }),
  ...(Object.keys(leaseOptions).length === 0
    ? {}
    : { lease: leaseOptions }),
  modelHash: modelIdentity(inferenceConfig.modelUrl),
  nativeWorkerHost: 'page',
  coordinatorInstanceId,
  state: {
    // Size the resident pool for the shape this deployment claims: every
    // serving slot keeps its session's KV, so a 2048-token context with four
    // parallel slots fits without parking. A harness can still shrink it to
    // force the park/resume lifecycle paths.
    l1TokenCap: l1TokenCap,
    l2TokenCap: l2TokenCap,
    maxResidentNodes: Math.max(1, Math.floor(l1NodeCap)),
    replacementPolicy:
      replacementPolicyParam(params.get('replacementPolicy')) ?? 'hybrid',
    lifecycleProtection: booleanParam(
      params.get('lifecycleProtection'),
      true
    ),
    admissionPolicy:
      admissionPolicyParam(params.get('admissionPolicy')) ??
      'monotonic-bounded',
  },
  evictOnPause: booleanParam(params.get('evictOnPause'), false),
  scheduler: {
    maxPending: 64,
    graceMs: 0,
    retentionWindowMs: 30_000,
    maxProtectedHeads: 4,
    ...(schedulerForegroundBurstUnits === undefined
      ? {}
      : { highPriorityBurstUnits: schedulerForegroundBurstUnits }),
    ...(schedulerResidentBurstUnits === undefined
      ? {}
      : { residentBurstUnits: schedulerResidentBurstUnits }),
    ...(schedulerRetentionWindowMs === undefined
      ? {}
      : { retentionWindowMs: schedulerRetentionWindowMs }),
  },
  // Deadline mode spends a long work-conserving unit when frames are clear,
  // so its watchdog has to sit above the lease's own ceiling; the fixed
  // shapes keep the short bounded unit they were measured with.
  execution: executionConfigForMode(executionMode),
});

let boundWorkerLabel = '';
let nativeOwnerPort: MessagePortLike | null = null;
let nativeRpc: MessagePortRpcClient | null = null;

const clients = new Set<MessagePortLike>();

scope.addEventListener('connect', (event) => {
  const port = event.ports[0] as unknown as MessagePortLike | undefined;
  if (!port) {
    return;
  }
  clients.add(port);
  port.addEventListener('message', async (messageEvent) => {
    const data = (messageEvent as MessageEvent).data as
      | {
          type?: unknown;
          callbackId?: unknown;
          label?: unknown;
          config?: unknown;
        }
      | null;
    if (!data || typeof data.type !== 'string') {
      return;
    }
    switch (data.type) {
      case 'native.probe':
        await applyInferenceConfig(data.config);
        clients.add(port);
        port.postMessage({
          type: 'native.status',
          callbackId: data.callbackId,
          bound: native.bound,
          worker: boundWorkerLabel,
          inference: inferenceConfig,
        });
        // A tab that arrives after the model is warm still needs the current
        // runtime state, otherwise its UI would wait for a load event forever.
        if (native.bound && lastRuntime) {
          coordinator.publishRuntime(lastRuntime);
        }
        return;
      case 'native.offer': {
        const offered = (messageEvent as MessageEvent).ports[0] as
          | MessagePortLike
          | undefined;
        if (!offered || native.bound) {
          offered?.close?.();
          port.postMessage({
            type: 'native.offer.result',
            callbackId: data.callbackId,
            accepted: false,
            reason: native.bound ? 'already_bound' : 'missing_port',
          });
          return;
        }
        nativeRpc = new MessagePortRpcClient(offered, (event) => {
          if (event.type === 'native.chunk') {
            coordinator.publishTokenChunk(event.event);
            return;
          }
          lastRuntime = event.event;
          coordinator.publishRuntime(event.event);
        });
        native.attach(new NativeEngineProxy(nativeRpc));
        nativeOwnerPort = port;
        boundWorkerLabel =
          typeof data.label === 'string' && data.label.length > 0
            ? data.label
            : 'page';
        port.postMessage({
          type: 'native.offer.result',
          callbackId: data.callbackId,
          accepted: true,
        });
        schedulePump();
        return;
      }
      case 'native.release':
        clients.delete(port);
        if (nativeOwnerPort === port) {
          dropNative('native host released');
        }
        return;
      case 'native.goodbye':
        clients.delete(port);
        if (nativeOwnerPort === port) {
          dropNative('native host disconnected');
        }
        return;
      default:
        return;
    }
  });
  port.start?.();
  startCoordinatorPort(port, coordinator, { onActivity: schedulePump });
});

let pumping = false;
let pumpScheduled = false;
const wakePump = createImmediateWake(() => {
  pumpScheduled = false;
  void runPumpLoop();
});

function schedulePump(): void {
  if (pumping || pumpScheduled || !native.bound) {
    return;
  }
  pumpScheduled = true;
  wakePump();
}

async function runPumpLoop(): Promise<void> {
  if (pumping || !native.bound) {
    return;
  }
  pumping = true;
  let ran = 0;
  try {
    ran = await coordinator.pump(1);
  } catch (error: unknown) {
    console.error('[wllm] coordinator pump failed', error);
  } finally {
    pumping = false;
  }
  if (ran > 0 && native.bound) {
    schedulePump();
  }
}

const HEARTBEAT_INTERVAL_MS = 2_000;
const HEARTBEAT_TIMEOUT_MS = 1_500;
let heartbeatAt = 0;
setInterval(() => {
  if (!native.bound || pumping) {
    return;
  }
  const now = Date.now();
  if (now - heartbeatAt < HEARTBEAT_INTERVAL_MS) {
    return;
  }
  heartbeatAt = now;
  void withTimeout(native.slotStats(), HEARTBEAT_TIMEOUT_MS).catch(() => {
    dropNative('inference worker stopped responding');
  });
}, HEARTBEAT_INTERVAL_MS);

function createImmediateWake(callback: () => void): () => void {
  if (typeof MessageChannel === 'undefined') {
    return () => {
      setTimeout(callback, 0);
    };
  }
  const channel = new MessageChannel();
  channel.port1.onmessage = () => callback();
  channel.port1.start();
  return () => channel.port2.postMessage(null);
}

function dropNative(reason: string): void {
  if (!native.bound) {
    return;
  }
  coordinator.handleNativeWorkerLoss(reason);
  lastRuntime = null;
  boundWorkerLabel = '';
  native.detach();
  nativeRpc?.dispose(new Error(reason));
  nativeRpc = null;
  nativeOwnerPort?.close?.();
  nativeOwnerPort = null;
  coordinator.publishRuntime({
    type: 'runtime',
    modelUrl: inferenceConfig.modelUrl,
    phase: 'error',
    message: reason,
  });
  for (const client of clients) {
    client.postMessage({ type: 'native.request' });
  }
}

function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(
      () => reject(new Error('inference worker heartbeat timed out')),
      ms
    );
    promise.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (error: unknown) => {
        clearTimeout(timer);
        reject(error instanceof Error ? error : new Error(String(error)));
      }
    );
  });
}

async function applyInferenceConfig(raw: unknown): Promise<void> {
  if (typeof raw !== 'object' || raw === null) {
    return;
  }
  const candidate = raw as Record<string, unknown>;
  if (typeof candidate.modelUrl === 'string' && candidate.modelUrl.length > 0) {
    const requestedModelUrl = resolveModelUrl(
      candidate.modelUrl,
      scope.location.origin
    );
    if (modelIdentity(requestedModelUrl) !== modelIdentity(inferenceConfig.modelUrl)) {
      console.warn(
        '[wllm] ignoring model change after coordinator startup',
        requestedModelUrl
      );
    } else {
      inferenceConfig.modelUrl = requestedModelUrl;
    }
  }
  if (candidate.modelLoadMode === 'cache' || candidate.modelLoadMode === 'direct') {
    inferenceConfig.modelLoadMode = candidate.modelLoadMode;
  }
  for (const key of [
    'nCtx',
    'nBatch',
    'nParallel',
    'nThreads',
    'webgpuCommandSubmitBatchSize',
    'webgpuMaxInflightBatches',
    'queueWaitTimeoutMs',
  ] as const) {
    const value = candidate[key];
    if (typeof value === 'number' && Number.isFinite(value) && value > 0) {
      inferenceConfig[key] = Math.floor(value);
    }
  }
  // Temperature 0 is meaningful (greedy), so it cannot ride the > 0 loop.
  const temperature = candidate.temperature;
  if (typeof temperature === 'number' && Number.isFinite(temperature) && temperature >= 0) {
    inferenceConfig.temperature = temperature;
  }
  const seed = candidate.seed;
  if (typeof seed === 'number' && Number.isFinite(seed)) {
    inferenceConfig.seed = Math.floor(seed);
  }
  const gpuLayers = candidate.nGpuLayers;
  if (
    typeof gpuLayers === 'number' &&
    Number.isFinite(gpuLayers) &&
    gpuLayers >= 0
  ) {
    inferenceConfig.nGpuLayers = Math.floor(gpuLayers);
  }
  if (typeof candidate.flashAttn === 'boolean') {
    inferenceConfig.flashAttn = candidate.flashAttn;
  }
  const stateConfig: Partial<StateManagerConfig> = {};
  for (const key of [
    'l1TokenCap',
    'l2TokenCap',
    'l3ByteCap',
    'repairBudget',
  ] as const) {
    const value = candidate[key];
    if (typeof value === 'number' && Number.isFinite(value) && value > 0) {
      stateConfig[key] = Math.floor(value);
    }
  }
  const residentNodeCap = candidate.l1NodeCap;
  if (
    typeof residentNodeCap === 'number' &&
    Number.isFinite(residentNodeCap) &&
    residentNodeCap > 0
  ) {
    stateConfig.maxResidentNodes = Math.floor(residentNodeCap);
  }
  const replacementPolicy = replacementPolicyParam(
    candidate.replacementPolicy
  );
  if (replacementPolicy !== undefined) {
    stateConfig.replacementPolicy = replacementPolicy;
  }
  const admissionPolicy = admissionPolicyParam(candidate.admissionPolicy);
  if (admissionPolicy !== undefined) {
    stateConfig.admissionPolicy = admissionPolicy;
  }
  if (typeof candidate.lifecycleProtection === 'boolean') {
    stateConfig.lifecycleProtection = candidate.lifecycleProtection;
  }
  if (typeof candidate.evictOnPause === 'boolean') {
    stateConfig.evictOnPause = candidate.evictOnPause;
  }
  coordinator.configureState(
    stateConfig,
    typeof candidate.parkCompleted === 'boolean'
      ? candidate.parkCompleted
      : undefined,
    parseCompletedTier(candidate.completedTier)
  );
  const executionConfig = parseExecutionConfig(candidate.execution);
  if (Object.keys(executionConfig).length > 0) {
    coordinator.configureExecution(executionConfig);
  }
  const renderAware = candidate.renderAware;
  const executionMode = parseExecutionMode(candidate.executionMode);
  if (executionMode !== undefined) {
    await coordinator.configureExecutionMode(executionMode);
  }
  if (typeof renderAware === 'boolean') {
    await coordinator.configureRenderAware(renderAware);
  } else if (renderAware === 0 || renderAware === 1) {
    await coordinator.configureRenderAware(renderAware === 1);
  }
}

function parseExecutionMode(raw: unknown): ExecutionMode | undefined {
  if (typeof raw !== 'string') {
    return undefined;
  }
  switch (raw.trim().toLowerCase()) {
    case 'bounded':
      return 'bounded';
    case 'unsliced':
      return 'unsliced';
    case 'deadline':
    case 'lease':
      return 'deadline';
    default:
      return undefined;
  }
}

function replacementPolicyParam(
  raw: unknown
): StateManagerConfig['replacementPolicy'] | undefined {
  if (typeof raw !== 'string') {
    return undefined;
  }
  switch (raw.trim().toLowerCase()) {
    case 'hybrid':
      return 'hybrid';
    case 'lru':
      return 'lru';
    case 'ttl':
      return 'ttl';
    default:
      return undefined;
  }
}

function admissionPolicyParam(
  raw: unknown
): NonNullable<StateManagerConfig['admissionPolicy']> | undefined {
  if (typeof raw !== 'string') {
    return undefined;
  }
  switch (raw.trim().toLowerCase()) {
    case 'monotonic-bounded':
    case 'monotonic':
      return 'monotonic-bounded';
    case 'reactive':
      return 'reactive';
    case 'immediate-reject':
    case 'reject':
      return 'immediate-reject';
    default:
      return undefined;
  }
}

function parseExecutionConfig(raw: unknown): Partial<ExecutionConfig> {
  if (typeof raw !== 'object' || raw === null) {
    return {};
  }
  const candidate = raw as Record<string, unknown>;
  const config: Partial<ExecutionConfig> = {};
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
    const value = candidate[key];
    if (typeof value === 'number' && Number.isFinite(value) && value > 0) {
      config[key] = value;
    }
  }
  const adaptive = candidate.adaptiveUnits;
  if (typeof adaptive === 'boolean') {
    config.adaptiveUnits = adaptive;
  } else if (adaptive === 0 || adaptive === 1) {
    config.adaptiveUnits = adaptive === 1;
  }
  const nativeTimeBudget = candidate.nativeTimeBudgetEnabled;
  if (typeof nativeTimeBudget === 'boolean') {
    config.nativeTimeBudgetEnabled = nativeTimeBudget;
  } else if (nativeTimeBudget === 0 || nativeTimeBudget === 1) {
    config.nativeTimeBudgetEnabled = nativeTimeBudget === 1;
  }
  return config;
}
function executionModeParam(value: string | null): ExecutionMode | undefined {
  switch (value?.trim().toLowerCase()) {
    case 'bounded':
      return 'bounded';
    case 'unsliced':
      return 'unsliced';
    case 'deadline':
    case 'lease':
      return 'deadline';
    default:
      return undefined;
  }
}

function completedTierParam(value: string | null): CompletedTier | undefined {
  return parseCompletedTier(value);
}

function parseCompletedTier(value: unknown): CompletedTier | undefined {
  if (typeof value !== 'string') {
    return undefined;
  }
  switch (value.trim().toLowerCase()) {
    case '1':
    case 'l1':
      return 'L1';
    case '2':
    case 'l2':
      return 'L2';
    case '3':
    case 'l3':
      return 'L3';
    case 'rebuild':
    case 'cold':
    case 'none':
      return 'rebuild';
    default:
      return undefined;
  }
}

function resolveModelUrl(value: string, baseUrl: string): string {
  try {
    return new URL(value, baseUrl).href;
  } catch {
    return value;
  }
}

function numberParam(value: string | null, fallback: number): number {
  return optionalNumber(value) ?? fallback;
}

function booleanParam(value: string | null, fallback: boolean): boolean {
  if (value === null || value.trim() === '') {
    return fallback;
  }
  return !['0', 'false', 'off', 'no'].includes(value.toLowerCase());
}

function optionalNumber(value: string | null): number | undefined {
  if (value === null || value.trim() === '') {
    return undefined;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function optionalBoolean(value: string | null): boolean | undefined {
  if (value === null || value.trim() === '') {
    return undefined;
  }
  return !['0', 'false', 'off', 'no'].includes(value.toLowerCase());
}
