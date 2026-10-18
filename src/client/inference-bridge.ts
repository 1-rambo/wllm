import wasmUrl from '@wllama/wllama/esm/wasm/wllama.wasm?url';
import inferenceWorkerUrl from '../workers/inference.worker.ts?worker&url';
import type { MessagePortLike } from '../transport/protocol';
import type { ExecutionConfig } from '../contracts';

export interface InferenceWorkerConfig {
  modelUrl: string;
  modelLoadMode?: 'cache' | 'direct';
  nCtx: number;
  nBatch: number;
  nParallel: number;
  temperature?: number;
  seed?: number;
  nThreads?: number;
  nGpuLayers?: number;
  // Fused attention. wllama reads `true` as "use it when the backend
  // supports it" and `false` as the explicit fallback path; leaving it
  // undefined keeps the pinned default.
  flashAttn?: boolean;
  webgpuCommandSubmitBatchSize?: number;
  webgpuMaxInflightBatches?: number;
  queueWaitTimeoutMs?: number;
}

export type InferenceBridgeState =
  | 'idle'
  | 'waiting'
  | 'offering'
  | 'owner'
  | 'released';

export interface InferenceBridgeEvents {
  onState?: (state: InferenceBridgeState, detail?: string) => void;
}

export interface InferenceBridgeOptions {
  modelUrl?: string;
  modelLoadMode?: 'cache' | 'direct';
  nCtx?: number;
  nBatch?: number;
  nParallel?: number;
  temperature?: number;
  seed?: number;
  nThreads?: number;
  nGpuLayers?: number;
  flashAttn?: boolean;
  webgpuCommandSubmitBatchSize?: number;
  webgpuMaxInflightBatches?: number;
  // Maximum time a WebGPU submission may wait for queue completion before
  // the native runtime aborts. Both ours and the per-tab baseline use the
  // same explicit ceiling so this is not an accidental arm difference.
  queueWaitTimeoutMs?: number;
  l1TokenCap?: number;
  // how many sessions may hold a serving slot at once; defaults to the
  // engine's parallel slot count
  l1NodeCap?: number;
  l2TokenCap?: number;
  replacementPolicy?: 'hybrid' | 'lru' | 'ttl';
  lifecycleProtection?: boolean;
  evictOnPause?: boolean;
  admissionPolicy?:
    | 'monotonic-bounded'
    | 'reactive'
    | 'immediate-reject';
  // bound on the history a session replays into its next turn; 0 keeps every
  // turn self contained
  continuationTokenCap?: number;
  l3ByteCap?: number;
  repairBudget?: number;
  parkCompleted?: boolean;
  completedTier?: '1' | '2' | '3' | 'L1' | 'L2' | 'L3' | 'rebuild';
  execution?: Partial<ExecutionConfig>;
  // coordinator-level render-pressure feedback switch
  renderAware?: boolean;
  // which controller shapes cooperative units; deadline uses the lease
  executionMode?: 'bounded' | 'unsliced' | 'deadline';
  // Allow an unanswered promotion to share a batch with ordinary work
  // instead of running as a lone high-priority-shaped graph.
  batchHandoff?: boolean;
  // Total graph size (head included) when a pending hand-off leads the
  // admission. Only meaningful while batchHandoff is enabled.
  leaseHandoffBatchSize?: number;
  // Submit a promoted session's next-token unit immediately instead of
  // waiting for the ordinary work-conserving graph to return to the
  // scheduler.
  expressHandoff?: boolean;
  // Cut the current work-conserving unit at its next token boundary when a
  // new session is promoted.
  preemptOnActivation?: boolean;
  // Minimum prompt tokens a hand-off admission may evaluate.
  handoffPrefillTokens?: number;
  // How many consecutive units the high-priority lane may take after its
  // first token before ordinary work gets a unit.
  schedulerForegroundBurstUnits?: number;
  // How many consecutive units the current resident sequence may take before
  // overdue cold work can preempt it.
  schedulerResidentBurstUnits?: number;
  // How long a recently promoted session remains protected from eviction.
  schedulerRetentionWindowMs?: number;
  // FrameDeadlineLease knobs a harness sweeps. Unset values keep the
  // controller's own defaults.
  leaseBatchSize?: number;
  // Wall-time quantum for work-conserving decode units. The coordinator maps
  // this URL-level lease knob onto FrameDeadlineLease.workConservingQuantumMs.
  leaseWorkConservingQuantumMs?: number;
  leaseMaxBudgetMs?: number;
  leaseMaxPauseMs?: number;
  leasePausePerDebtMs?: number;
  leaseMinPressureDecodeTokens?: number;
  leaseBackgroundDecodeTokens?: number;
  leaseMinPressurePrefillTokens?: number;
  leaseMaxPrefillTokens?: number;
  // ceiling for one native decode unit; the lease still shrinks the
  // pressure-shaped unit from its own presented-window budget.
  leaseMaxDecodeTokens?: number;
  leaseMaxPressurePrefillTokens?: number;
  // Evaluation-only fixed prefill graph length for the chunk study.
  leasePrefillChunkTokens?: number;
  leaseMaxCommandBatchSize?: number;
  leaseMinPressureCommandBatchSize?: number;
  leaseInflightCap?: number;
  leaseMildPressureFramePeriods?: number;
  leaseSeverePressureFramePeriods?: number;
  leaseQueueWaitToleranceMs?: number;
  leaseQueueBackpressureSamples?: number;
  leaseQueueReliefSamples?: number;
  // Keep the continuous work-conserving decode but submit through the local
  // barrier instead of the deep queue, so a renderer sharing the device is
  // never parked behind more than one command buffer.
  leaseWorkConservingBarrier?: boolean;
  leaseBackgroundBarrier?: boolean;
  // Shape used before fresh frame evidence exists: the sliced bounded
  // fallback, or the same continuous decode with a local barrier.
  leaseStaleSignalShape?: 'bounded' | 'barrier';
  // hold new units while render pressure sits at its worst level
  renderGate?: boolean;
}

interface PendingCall {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}

const REQUEST_TIMEOUT_MS = 30_000;

/**
 * A SharedWorker cannot spawn a dedicated worker, so a page has to host the
 * wllama inference worker and hand its MessagePort to the shared coordinator.
 * Only the first page that offers a port becomes the owner; the rest keep using
 * the coordinator-bound engine. If the owner page goes away, the coordinator
 * asks the remaining pages for a replacement.
 */
export class InferenceBridge {
  private nextCallback = 1;
  private readonly pending = new Map<string, PendingCall>();
  private bootstrap: { worker: Worker; port: MessagePort } | null = null;
  private ensuring: Promise<void> | null = null;

  constructor(
    private readonly sharedPort: MessagePortLike,
    private readonly options: InferenceBridgeOptions = {},
    private readonly events: InferenceBridgeEvents = {}
  ) {
    sharedPort.addEventListener('message', (event) =>
      this.onMessage(event.data)
    );
    sharedPort.start?.();
  }

  get ownsWorker(): boolean {
    return this.bootstrap !== null;
  }

  ensure(): Promise<void> {
    if (this.ensuring) {
      return this.ensuring;
    }
    this.ensuring = this.ensureOnce().finally(() => {
      this.ensuring = null;
    });
    return this.ensuring;
  }

  release(): void {
    if (!this.bootstrap) {
      return;
    }
    this.sharedPort.postMessage({ type: 'native.release' });
    this.terminate();
    this.events.onState?.('released');
  }

  private async ensureOnce(): Promise<void> {
    const status = await this.call<{
      bound: boolean;
      inference: InferenceWorkerConfig;
    }>('native.probe', (callbackId) => ({
      type: 'native.probe',
      callbackId,
      config: this.options,
    }));
    if (status.bound) {
      this.events.onState?.('waiting', 'session already served');
      return;
    }
    if (this.bootstrap) {
      return;
    }
    this.events.onState?.('offering', 'starting inference worker');
    const bootstrap = createInferenceWorker(status.inference);
    this.bootstrap = bootstrap;
    const result = await this.call<{ accepted: boolean; reason?: string }>(
      'native.offer',
      (callbackId) => ({
        type: 'native.offer',
        callbackId,
        label: window.location.pathname,
      }),
      [bootstrap.port as unknown as Transferable]
    );
    if (!result.accepted) {
      this.terminate();
      this.events.onState?.('waiting', result.reason ?? 'already bound');
      return;
    }
    this.events.onState?.('owner', 'hosting inference worker');
  }

  private onMessage(data: unknown): void {
    if (typeof data !== 'object' || data === null) {
      return;
    }
    const message = data as { type?: unknown; callbackId?: unknown };
    if (message.type === 'native.request') {
      void this.ensure().catch(() => undefined);
      return;
    }
    if (typeof message.callbackId !== 'string') {
      return;
    }
    const pending = this.pending.get(message.callbackId);
    if (!pending) {
      return;
    }
    this.pending.delete(message.callbackId);
    clearTimeout(pending.timer);
    if (
      message.type === 'native.status' ||
      message.type === 'native.offer.result'
    ) {
      pending.resolve(message);
      return;
    }
    pending.reject(new Error('unexpected native bridge response'));
  }

  private call<T>(
    label: string,
    build: (callbackId: string) => Record<string, unknown>,
    transfer: Transferable[] = []
  ): Promise<T> {
    const callbackId = `${label}:${++this.nextCallback}`;
    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(callbackId);
        reject(new Error(`native bridge ${label} timed out`));
      }, REQUEST_TIMEOUT_MS);
      this.pending.set(callbackId, {
        resolve: resolve as (value: unknown) => void,
        reject,
        timer,
      });
      try {
        this.sharedPort.postMessage(build(callbackId), transfer);
      } catch (error) {
        clearTimeout(timer);
        this.pending.delete(callbackId);
        reject(error instanceof Error ? error : new Error(String(error)));
      }
    });
  }

  private terminate(): void {
    this.bootstrap?.worker.terminate();
    this.bootstrap?.port.close();
    this.bootstrap = null;
  }
}

export function createInferenceWorker(
  config: InferenceWorkerConfig
): { worker: Worker; port: MessagePort } {
  const url = new URL(inferenceWorkerUrl, window.location.href);
  url.searchParams.set('modelUrl', config.modelUrl);
  if (config.modelLoadMode !== undefined) {
    url.searchParams.set('modelLoadMode', config.modelLoadMode);
  }
  url.searchParams.set('wasmUrl', new URL(wasmUrl, window.location.href).href);
  url.searchParams.set('nCtx', String(config.nCtx));
  url.searchParams.set('nBatch', String(config.nBatch));
  url.searchParams.set('nParallel', String(config.nParallel));
  if (config.temperature !== undefined) {
    url.searchParams.set('temperature', String(config.temperature));
  }
  if (config.seed !== undefined) {
    url.searchParams.set('seed', String(config.seed));
  }
  if (config.nThreads !== undefined) {
    url.searchParams.set('nThreads', String(config.nThreads));
  }
  if (config.nGpuLayers !== undefined) {
    url.searchParams.set('nGpuLayers', String(config.nGpuLayers));
  }
  if (config.flashAttn !== undefined) {
    url.searchParams.set('flashAttn', config.flashAttn ? '1' : '0');
  }
  if (config.webgpuCommandSubmitBatchSize !== undefined) {
    url.searchParams.set(
      'webgpuCommandSubmitBatchSize',
      String(config.webgpuCommandSubmitBatchSize)
    );
  }
  if (config.webgpuMaxInflightBatches !== undefined) {
    url.searchParams.set(
      'webgpuMaxInflightBatches',
      String(config.webgpuMaxInflightBatches)
    );
  }
  if (config.queueWaitTimeoutMs !== undefined) {
    url.searchParams.set(
      'queueWaitTimeoutMs',
      String(config.queueWaitTimeoutMs)
    );
  }
  const worker = new Worker(url, {
    type: 'module',
    name: 'wllm-inference',
  });
  const channel = new MessageChannel();
  worker.postMessage({ type: 'native.attach' }, [channel.port1]);
  return { worker, port: channel.port2 };
}
