import wasmUrl from '@wllama/wllama/esm/wasm/wllama.wasm?url';
import {
  WllamaNativeEngine,
  type WllamaNativeEngineConfig,
} from '../native/wllama-native-engine';
import { NativeEngineWorkerServer } from '../transport/native-worker';
import type { MessagePortLike, NativeWorkerEvent } from '../transport/protocol';
import type { NativeRuntimeEvent } from '../contracts';
import {
  defaultModelPath,
  defaultRuntimeConfig,
} from '../config/demo-model';

const scope = self as unknown as DedicatedWorkerGlobalScope;
const params = new URL(scope.location.href).searchParams;
const modelUrl = new URL(
  params.get('modelUrl') ?? defaultModelPath,
  scope.location.origin
).href;
const runtimeUrl = params.get('wasmUrl') ?? wasmUrl;
const modelLoadMode =
  params.get('modelLoadMode') === 'direct' ? 'direct' : 'cache';

// The coordinator owns this worker through a transferred MessagePort.
// Native chunk events use the same port as runtime events.
let rpcPort: MessagePortLike | null = null;
let started = false;

const engineConfig: WllamaNativeEngineConfig = {
  wasmUrl: runtimeUrl,
  modelUrl,
  modelLoadMode,
  nCtx: numberParam(params.get('nCtx'), defaultRuntimeConfig.nCtx),
  nBatch: numberParam(params.get('nBatch'), defaultRuntimeConfig.nBatch),
  nParallel: numberParam(
    params.get('nParallel'),
    defaultRuntimeConfig.nParallel
  ),
  queueWaitTimeoutMs: Math.max(
    30_000,
    Math.floor(numberParam(params.get('queueWaitTimeoutMs'), 300_000))
  ),
  temperature: numberParam(params.get('temperature'), 0.8),
  seed: numberParam(params.get('seed'), 0),
  onTokenChunk: (chunk) => {
    rpcPort?.postMessage({
      type: 'native.chunk',
      event: chunk,
    } satisfies NativeWorkerEvent);
  },
  onRuntimeEvent: (event) => emitRuntime(event),
};
const nThreads = optionalNumber(params.get('nThreads'));
const nGpuLayers = optionalNumber(params.get('nGpuLayers'));
const flashAttn = optionalBoolean(params.get('flashAttn'));
const webgpuCommandSubmitBatchSize = optionalNumber(
  params.get('webgpuCommandSubmitBatchSize')
);
const webgpuMaxInflightBatches = optionalNumber(
  params.get('webgpuMaxInflightBatches')
);
if (nThreads !== undefined) {
  engineConfig.nThreads = nThreads;
}
if (nGpuLayers !== undefined) {
  engineConfig.nGpuLayers = nGpuLayers;
}
if (flashAttn !== undefined) {
  engineConfig.flashAttn = flashAttn;
}
if (webgpuCommandSubmitBatchSize !== undefined) {
  engineConfig.webgpuCommandSubmitBatchSize = webgpuCommandSubmitBatchSize;
}
if (webgpuMaxInflightBatches !== undefined) {
  engineConfig.webgpuMaxInflightBatches = webgpuMaxInflightBatches;
}
const engine = new WllamaNativeEngine(engineConfig);

// The coordinator runs inside a SharedWorker, which cannot create a dedicated
// worker. The owning page therefore spawns this worker and transfers a
// MessagePort in; all RPC traffic arrives on that port.

// A bounded unit can abort an in-flight completion when the wall-clock guard
// fires. wllama surfaces those as unhandled rejections; they are expected.
scope.addEventListener('unhandledrejection', (event) => {
  const reason = (event as unknown as { reason?: unknown }).reason;
  if (isAbortLike(reason)) {
    (event as unknown as { preventDefault?: () => void }).preventDefault?.();
  }
});

scope.addEventListener('message', (event) => {
  const data = (event as MessageEvent).data as { type?: unknown } | null;
  if (!data || typeof data !== 'object' || data.type !== 'native.attach') {
    return;
  }
  if (started) {
    return;
  }
  const port = (event as MessageEvent).ports[0] as unknown as
    | MessagePortLike
    | undefined;
  if (!port) {
    return;
  }
  started = true;
  rpcPort = port;
  new NativeEngineWorkerServer(port, engine);
  port.start?.();
  void engine.prewarm().catch((error: unknown) => {
    console.error('[wllm] inference prewarm failed', error);
  });
});

function emitRuntime(event: NativeRuntimeEvent): void {
  rpcPort?.postMessage({
    type: 'native.event',
    event,
  } satisfies NativeWorkerEvent);
}

function isAbortLike(error: unknown): boolean {
  return (
    typeof error === 'object' &&
    error !== null &&
    'name' in error &&
    (error as { name?: unknown }).name === 'AbortError'
  );
}

function numberParam(value: string | null, fallback: number): number {
  return optionalNumber(value) ?? fallback;
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
