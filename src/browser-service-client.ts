import CacheManager, { type DownloadOptions } from './cache-manager';
import { ModelManager, Model } from './model-manager';
import type {
  AssetsPathConfig,
  LoadModelConfig,
  WllamaChatFromNodeOptions,
  WllamaChatFromNodeResult,
  WllamaConfig,
  WllamaTieredCacheOptions,
  WllamaTreeState,
} from './wllama';
import { absoluteUrl, sortFileByShard } from './utils';
import type {
  BrowserClientLifecycle,
  BrowserServiceEvent,
  BrowserServiceMessage,
  BrowserServiceRequest,
  BrowserServiceStatus,
} from './browser-service-protocol';
import {
  BROWSER_SERVICE_SCOPE_CODE,
  LIBLLAMA_VERSION,
  LLAMA_CPP_WORKER_CODE,
  WLLAMA_SINGLE_THREAD_CODE,
} from './workers-code/generated';

const HEARTBEAT_MS = 1000;
const CONNECT_TIMEOUT_MS = 5000;

const hashCode = (text: string): string => {
  let hash = 5381;
  for (let i = 0; i < text.length; i++) {
    hash = ((hash << 5) + hash + text.charCodeAt(i)) | 0;
  }
  return (hash >>> 0).toString(36);
};

const VERSION_TAG = `${LIBLLAMA_VERSION}-${hashCode(BROWSER_SERVICE_SCOPE_CODE)}`;
const WORKER_NAME = `wllm-browser-service-${VERSION_TAG}`;
const REGISTRY_KEY = `${WORKER_NAME}-url`;
const CREATE_LOCK = `${WORKER_NAME}-create`;
const DOWNLOAD_LOCK_PREFIX = `${WORKER_NAME}-download-`;

const modelConfigKey = (config: LoadModelConfig): string =>
  JSON.stringify({
    n_ctx: config.n_ctx ?? 1024,
    n_batch: config.n_batch ?? null,
    n_seq_max: config.n_seq_max ?? null,
    kv_unified: config.kv_unified ?? true,
    embeddings: config.embeddings ?? false,
    offload_kqv: config.offload_kqv ?? null,
    pooling_type: config.pooling_type ?? null,
    cache_type_k: config.cache_type_k ?? null,
    cache_type_v: config.cache_type_v ?? null,
    flash_attn: config.flash_attn ?? null,
  });

interface PendingRequest {
  resolve: (value: any) => void;
  reject: (reason: unknown) => void;
}

interface PendingChat {
  onChunk?: WllamaChatFromNodeOptions['onChunk'];
  cleanupAbort?: () => void;
}

export interface BrowserServiceRuntimeAssets {
  moduleCode?: string;
  llamaCppCode?: string;
}

export const isBrowserLLMServiceSupported = (): boolean =>
  typeof SharedWorker !== 'undefined' &&
  typeof localStorage !== 'undefined' &&
  typeof navigator !== 'undefined' &&
  !!navigator.locks;

export class BrowserLLMClient {
  readonly clientId: string;

  private readonly pathConfig: AssetsPathConfig;
  private readonly config: WllamaConfig;
  private readonly cacheManager: CacheManager;
  private readonly modelManager: ModelManager;
  private readonly runtimeAssets: BrowserServiceRuntimeAssets;
  private worker: SharedWorker | undefined;
  private port: MessagePort | undefined;
  private heartbeat: ReturnType<typeof setInterval> | undefined;
  private rpcId = 1;
  private chatId = 1;
  private lifecycleEpoch = 0;
  private lifecycle: BrowserClientLifecycle;
  private pending = new Map<number, PendingRequest>();
  private chats = new Map<number, PendingChat>();
  private listeners = new Set<(event: BrowserServiceEvent) => void>();
  private lifecycleCleanup: Array<() => void> = [];

  constructor(
    pathConfig: AssetsPathConfig,
    config: WllamaConfig = {},
    runtimeAssets: BrowserServiceRuntimeAssets = {}
  ) {
    if (!isBrowserLLMServiceSupported()) {
      throw new Error('SharedWorker browser LLM service is not supported');
    }
    this.pathConfig = pathConfig;
    this.config = config;
    this.runtimeAssets = runtimeAssets;
    this.clientId =
      typeof crypto !== 'undefined' && 'randomUUID' in crypto
        ? crypto.randomUUID()
        : `client-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    this.cacheManager = config.cacheManager ?? new CacheManager();
    this.modelManager =
      config.modelManager ??
      new ModelManager({
        cacheManager: this.cacheManager,
        logger: config.logger ?? console,
        parallelDownloads: config.parallelDownloads,
        allowOffline: config.allowOffline,
      });
    this.lifecycle = this.readLifecycle();
  }

  async connect(): Promise<BrowserServiceStatus> {
    if (this.port) return await this.status();

    const knownUrl = localStorage.getItem(REGISTRY_KEY);
    if (knownUrl) {
      try {
        await this.attach(knownUrl);
      } catch {
        localStorage.removeItem(REGISTRY_KEY);
      }
    }

    if (!this.port) {
      await navigator.locks.request(CREATE_LOCK, async () => {
        const currentUrl = localStorage.getItem(REGISTRY_KEY);
        if (currentUrl) {
          try {
            await this.attach(currentUrl);
            return;
          } catch {
            localStorage.removeItem(REGISTRY_KEY);
          }
        }
        const url = URL.createObjectURL(
          new Blob([BROWSER_SERVICE_SCOPE_CODE], { type: 'text/javascript' })
        );
        await this.attach(url);
        localStorage.setItem(REGISTRY_KEY, url);
      });
    }

    this.heartbeat = setInterval(() => {
      void this.rpc('heartbeat').catch(() => {});
    }, HEARTBEAT_MS);
    this.installLifecycleObservers();
    await this.updateLifecycle();
    return await this.status();
  }

  async status(): Promise<BrowserServiceStatus> {
    return await this.rpc('status');
  }

  async loadModelFromUrl(
    modelUrl: string | string[],
    config: LoadModelConfig & DownloadOptions & { useCache?: boolean } = {}
  ): Promise<void> {
    await this.connect();
    const urls = ModelManager.parseModelUrl(modelUrl);
    const modelKey = `${urls.join(';')}#${modelConfigKey(config)}`;
    const status = await this.status();
    if (status.status === 'ready') {
      if (status.modelKey !== modelKey) {
        throw new Error(
          `browser LLM service already loaded model ${status.modelKey}`
        );
      }
      return;
    }
    if (status.status === 'initializing') {
      if (status.modelKey !== modelKey) {
        throw new Error(
          `browser LLM service is loading model ${status.modelKey}`
        );
      }
      await this.waitUntilReady(modelKey);
      return;
    }

    const model = await navigator.locks.request(
      `${DOWNLOAD_LOCK_PREFIX}${hashCode(urls.join(';'))}`,
      async () => {
        const downloadOptions: DownloadOptions = {};
        if (config.progressCallback) {
          downloadOptions.progressCallback = config.progressCallback;
        }
        if (config.headers) downloadOptions.headers = config.headers;
        if (config.signal) downloadOptions.signal = config.signal;
        return config.useCache === false
          ? await this.modelManager.downloadModel(urls[0], downloadOptions)
          : await this.modelManager.getModelOrDownload(
              urls[0],
              downloadOptions
            );
      }
    );
    await this.loadModel(model, config, modelKey);
  }

  async loadModel(
    modelOrBlobs: Model | Blob[],
    config: LoadModelConfig = {},
    modelKey?: string
  ): Promise<void> {
    await this.connect();
    const blobs =
      modelOrBlobs instanceof Model
        ? await modelOrBlobs.open()
        : [...modelOrBlobs];
    sortFileByShard(blobs);
    const files = blobs.map((blob, index) => ({
      name: (blob as File).name || `model-${index}.gguf`,
      blob,
    }));
    const key =
      modelKey ??
      `${files.map((file) => `${file.name}:${file.blob.size}`).join(';')}#${modelConfigKey(config)}`;
    await navigator.locks.request(
      `${WORKER_NAME}-model-${hashCode(key)}`,
      async () => {
        const status = await this.status();
        if (status.status === 'ready') {
          if (status.modelKey !== key) {
            throw new Error(
              `browser LLM service already loaded model ${status.modelKey}`
            );
          }
          return;
        }
        if (status.status === 'initializing') {
          if (status.modelKey !== key) {
            throw new Error(
              `browser LLM service is loading model ${status.modelKey}`
            );
          }
          await this.waitUntilReady(key);
          return;
        }
        await this.rpc('init', {
          modelKey: key,
          pathConfig: {
            'single-thread/wllama.wasm': absoluteUrl(
              this.pathConfig['single-thread/wllama.wasm']
            ),
            ...(this.pathConfig['multi-thread/wllama.wasm']
              ? {
                  'multi-thread/wllama.wasm': absoluteUrl(
                    this.pathConfig['multi-thread/wllama.wasm']
                  ),
                }
              : {}),
          },
          config: this.serializableConfig(),
          loadConfig: this.serializableLoadConfig(config),
          modelFiles: files,
          moduleCode:
            this.runtimeAssets.moduleCode ?? WLLAMA_SINGLE_THREAD_CODE,
          llamaCppCode:
            this.runtimeAssets.llamaCppCode ?? LLAMA_CPP_WORKER_CODE,
        });
      }
    );
  }

  async chatSessionInit(
    memoryCapBytes: number = 1024 * 1024 * 1024,
    tieredCache: WllamaTieredCacheOptions = {}
  ): Promise<WllamaTreeState> {
    return await this.rpc('session-init', { memoryCapBytes, tieredCache });
  }

  async chatGetState(): Promise<WllamaTreeState> {
    return await this.rpc('state');
  }

  async chatReset(): Promise<WllamaTreeState> {
    return await this.rpc('session-reset');
  }

  async chatDeleteNode(nodeId: number): Promise<WllamaTreeState> {
    return await this.rpc('session-delete', { nodeId });
  }

  async chatFromNode(
    parentId: number,
    userText: string,
    options: WllamaChatFromNodeOptions = {}
  ): Promise<WllamaChatFromNodeResult> {
    const requestId = this.chatId++;
    const { onChunk, abortSignal, ...serializableOptions } = options;
    const pendingChat: PendingChat = { onChunk };
    if (abortSignal) {
      const cancel = () => {
        void this.rpc('cancel', { requestId }).catch(() => {});
      };
      abortSignal.addEventListener('abort', cancel, { once: true });
      pendingChat.cleanupAbort = () =>
        abortSignal.removeEventListener('abort', cancel);
    }
    this.chats.set(requestId, pendingChat);
    try {
      return await this.rpc('chat', {
        requestId,
        parentId,
        userText,
        options: serializableOptions,
      });
    } finally {
      pendingChat.cleanupAbort?.();
      this.chats.delete(requestId);
    }
  }

  async updateLifecycle(
    update?: Partial<
      Omit<BrowserClientLifecycle, 'lifecycleEpoch' | 'changedAt'>
    >
  ): Promise<void> {
    this.lifecycleEpoch += 1;
    this.lifecycle = {
      ...this.readLifecycle(),
      ...update,
      lifecycleEpoch: this.lifecycleEpoch,
      changedAt: Date.now(),
    };
    if (this.port) {
      await this.rpc('lifecycle', this.lifecycle);
    }
  }

  async flushLifecyclePolicy(): Promise<void> {
    await this.rpc('flush-lifecycle');
  }

  onTelemetry(listener: (event: BrowserServiceEvent) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  async getDebugInfo(): Promise<unknown> {
    return await this.rpc('debug');
  }

  async getTelemetry(clear: boolean = false): Promise<BrowserServiceEvent[]> {
    return await this.rpc('telemetry', { clear });
  }

  async exportTelemetryJsonl(clear: boolean = false): Promise<string> {
    const records = await this.getTelemetry(clear);
    return records.map((record) => JSON.stringify(record)).join('\n');
  }

  async recordEvent(
    type: string,
    data?: Record<string, unknown>
  ): Promise<void> {
    await this.rpc('record-event', { type, data });
  }

  async exit(): Promise<void> {
    if (this.port) {
      await this.rpc('detach').catch(() => {});
    }
    this.shutdown();
  }

  async destroyService(): Promise<void> {
    if (this.port) {
      await this.rpc('destroy').catch(() => {});
    }
    this.shutdown();
    localStorage.removeItem(REGISTRY_KEY);
  }

  private serializableConfig(): WllamaConfig {
    const {
      logger: _logger,
      cacheManager: _cacheManager,
      modelManager: _modelManager,
      engineChatEventListener: _engineChatEventListener,
      ...config
    } = this.config;
    return {
      ...config,
      preferWebGPU: config.preferWebGPU ?? true,
      suppressNativeLog: config.suppressNativeLog ?? true,
      engineChatSchedulingPolicy:
        config.engineChatSchedulingPolicy ?? 'foreground-aware',
    };
  }

  private async waitUntilReady(modelKey: string): Promise<void> {
    const deadline = Date.now() + 5 * 60 * 1000;
    while (Date.now() < deadline) {
      const status = await this.status();
      if (status.status === 'ready') {
        if (status.modelKey !== modelKey) {
          throw new Error(
            `browser LLM service loaded model ${status.modelKey}`
          );
        }
        return;
      }
      if (status.status === 'error') {
        throw new Error(
          status.lastError ?? 'browser LLM service initialization failed'
        );
      }
      await new Promise((resolve) => setTimeout(resolve, 50));
    }
    throw new Error('timed out waiting for browser LLM service initialization');
  }

  private serializableLoadConfig(config: LoadModelConfig): LoadModelConfig {
    const keys: Array<keyof LoadModelConfig> = [
      'seed',
      'n_ctx',
      'n_batch',
      'n_seq_max',
      'kv_unified',
      'n_threads',
      'embeddings',
      'offload_kqv',
      'pooling_type',
      'rope_scaling_type',
      'rope_freq_base',
      'rope_freq_scale',
      'yarn_ext_factor',
      'yarn_attn_factor',
      'yarn_beta_fast',
      'yarn_beta_slow',
      'yarn_orig_ctx',
      'cache_type_k',
      'cache_type_v',
      'flash_attn',
    ];
    const output: Record<string, unknown> = { n_threads: 1 };
    for (const key of keys) {
      const value = config[key];
      if (value !== undefined) output[key] = value;
    }
    return output as LoadModelConfig;
  }

  private readLifecycle(): BrowserClientLifecycle {
    const visible =
      typeof document === 'undefined' || document.visibilityState === 'visible';
    const focused =
      typeof document === 'undefined' || typeof document.hasFocus !== 'function'
        ? visible
        : document.hasFocus();
    return {
      foreground: visible && focused,
      lifecycleState: !visible ? 'hidden' : focused ? 'active' : 'passive',
      lifecycleEpoch: this.lifecycleEpoch,
      changedAt: Date.now(),
    };
  }

  private installLifecycleObservers(): void {
    if (typeof document === 'undefined' || this.lifecycleCleanup.length > 0)
      return;
    const update = () => void this.updateLifecycle().catch(() => {});
    const freeze = () =>
      void this.updateLifecycle({
        foreground: false,
        lifecycleState: 'frozen',
      }).catch(() => {});
    const pageHide = (event: PageTransitionEvent) => {
      if (event.persisted) {
        freeze();
      } else {
        void this.rpc('detach').catch(() => {});
      }
    };
    document.addEventListener('visibilitychange', update);
    globalThis.addEventListener('focus', update);
    globalThis.addEventListener('blur', update);
    globalThis.addEventListener('pageshow', update);
    globalThis.addEventListener('pagehide', pageHide);
    document.addEventListener('freeze', freeze);
    document.addEventListener('resume', update);
    this.lifecycleCleanup.push(
      () => document.removeEventListener('visibilitychange', update),
      () => globalThis.removeEventListener('focus', update),
      () => globalThis.removeEventListener('blur', update),
      () => globalThis.removeEventListener('pageshow', update),
      () => globalThis.removeEventListener('pagehide', pageHide),
      () => document.removeEventListener('freeze', freeze),
      () => document.removeEventListener('resume', update)
    );
  }

  private attach(url: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const worker = new SharedWorker(url, { name: WORKER_NAME });
      const timer = setTimeout(() => {
        cleanup();
        reject(new Error('timed out while connecting to browser LLM service'));
      }, CONNECT_TIMEOUT_MS);
      const cleanup = () => {
        clearTimeout(timer);
        worker.onerror = null;
      };
      worker.onerror = () => {
        cleanup();
        reject(new Error('browser LLM SharedWorker could not be loaded'));
      };
      worker.port.onmessage = (event: MessageEvent<BrowserServiceMessage>) => {
        if (!('event' in event.data) || event.data.event !== 'hello') return;
        cleanup();
        this.worker = worker;
        this.port = worker.port;
        worker.port.onmessage = (message) => this.onMessage(message);
        resolve();
      };
      worker.port.start();
    });
  }

  private onMessage(event: MessageEvent<BrowserServiceMessage>): void {
    const message = event.data;
    if ('event' in message) {
      if (message.event === 'chunk' && message.clientId === this.clientId) {
        this.chats
          .get(message.requestId)
          ?.onChunk?.(message.piece, message.fullText);
      } else if (message.event === 'telemetry') {
        for (const listener of this.listeners) listener(message.record);
      } else if (message.event === 'dying') {
        this.shutdown(new Error(message.reason));
      }
      return;
    }
    const pending = this.pending.get(message.id);
    if (!pending) return;
    this.pending.delete(message.id);
    if ('error' in message) pending.reject(new Error(message.error));
    else pending.resolve(message.result);
  }

  private rpc(
    verb: BrowserServiceRequest['verb'],
    payload?: unknown
  ): Promise<any> {
    if (!this.port)
      return Promise.reject(new Error('browser LLM service is not connected'));
    return new Promise((resolve, reject) => {
      const id = this.rpcId++;
      this.pending.set(id, { resolve, reject });
      this.port!.postMessage({ id, verb, payload, clientId: this.clientId });
    });
  }

  private shutdown(reason = new Error('browser LLM client detached')): void {
    if (this.heartbeat) clearInterval(this.heartbeat);
    for (const cleanup of this.lifecycleCleanup) cleanup();
    this.lifecycleCleanup = [];
    for (const pending of this.pending.values()) pending.reject(reason);
    this.pending.clear();
    this.port?.close();
    this.port = undefined;
    this.worker = undefined;
  }
}
