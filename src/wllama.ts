import { ProxyToWorker } from './worker';
import {
  absoluteUrl,
  bufToText,
  cbToAsyncIter,
  checkEnvironmentCompatible,
  isString,
  isSupportMultiThread,
  joinBuffers,
  sortFileByShard,
  isValidGgufFile,
} from './utils';
import CacheManager, { type DownloadOptions } from './cache-manager';
import { ModelManager, Model } from './model-manager';
import type {
  GlueMsgChatFormatRes,
  GlueMsgDecodeRes,
  GlueMsgDetokenizeRes,
  GlueMsgGetEmbeddingsRes,
  GlueMsgGetKvClearRes,
  GlueMsgGetKvRemoveRes,
  GlueMsgGetLogitsRes,
  GlueMsgGetVocabRes,
  GlueMsgLoadRes,
  GlueMsgLookupTokenRes,
  GlueMsgPerfContextRes,
  GlueMsgPerfResetRes,
  GlueMsgSamplingAcceptRes,
  GlueMsgSamplingSampleRes,
  GlueMsgSetOptionsRes,
  GlueMsgStatusRes,
  GlueMsgTestBenchmarkRes,
  GlueMsgTestPerplexityRes,
  GlueMsgTokenizeRes,
  GlueMsgTreeDeleteRes,
  GlueMsgTreeChatFinishRes,
  GlueMsgTreeChatStartRes,
  GlueMsgTreeInitRes,
  GlueMsgTreeResetRes,
  GlueMsgTreeStateRes,
  GlueMsgTreeSwitchRes,
} from './glue/messages';
import { LIBLLAMA_VERSION } from './workers-code/generated';

const HF_MODEL_ID_REGEX = /^([a-zA-Z0-9_\-\.]+)\/([a-zA-Z0-9_\-\.]+)$/;
const HF_MODEL_ID_REGEX_EXPLAIN =
  "Hugging Face model ID is incorrect. Only regular alphanumeric characters, '-', '.' and '_' supported";
const MAX_SAFE_N_SEQ_MAX = 256;
const MIN_CTX_PER_SEQUENCE = 1024;
const DEFAULT_ENGINE_CHAT_SERVICE_UPPER_BOUND_MS = 30000;
const DEFAULT_ENGINE_CHAT_QUEUE_MAX_PENDING = 128;

export interface WllamaLogger {
  debug: typeof console.debug;
  log: typeof console.log;
  warn: typeof console.warn;
  error: typeof console.error;
}

// TODO: bring back useCache
export interface WllamaConfig {
  /**
   * If true, suppress all log messages from native CPP code
   */
  suppressNativeLog?: boolean;
  /**
   * Custom logger functions
   */
  logger?: WllamaLogger;
  /**
   * Maximum number of parallel files to be downloaded
   *
   * Default: parallelDownloads = 3
   */
  parallelDownloads?: number;
  /**
   * Allow offline mode. If true, the model will be loaded from cache if it's available.
   *
   * Default: allowOffline = false
   */
  allowOffline?: boolean;
  /**
   * Custom cache manager (only for advanced usage)
   */
  cacheManager?: CacheManager;
  /**
   * Custom model manager (only for advanced usage)
   */
  modelManager?: ModelManager;
  /**
   * Use the WebGPU backend if available.
   */
  preferWebGPU?: boolean;
  /**
   * Disable llama.cpp performance metrics.
   *
   * Default: noPerf = false
   */
  noPerf?: boolean;
  /**
   * Unified upper bound of per-request service time used by queue scheduling budget (ms).
   *
   * Default: 30000
   */
  engineChatServiceUpperBoundMs?: number;
  /**
   * Maximum number of pending chat requests allowed in engine queue.
   * Requests above this bound are rejected immediately for backpressure.
   *
   * Default: 128
   */
  engineChatQueueMaxPending?: number;
  /**
   * Enable detailed engine-chat tracing logs for diagnosing random stalls/timeouts.
   *
   * Default: false
   */
  engineChatTraceEnabled?: boolean;
}

export interface WllamaChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface AssetsPathConfig {
  'single-thread/wllama.wasm': string;
  'multi-thread/wllama.wasm'?: string;
}

export interface LoadModelConfig {
  seed?: number;
  n_ctx?: number;
  n_batch?: number;
  n_seq_max?: number;
  kv_unified?: boolean;
  // by default, on multi-thread build, we take half number of available threads (hardwareConcurrency / 2)
  n_threads?: number;
  embeddings?: boolean;
  offload_kqv?: boolean;
  pooling_type?:
    | 'LLAMA_POOLING_TYPE_UNSPECIFIED'
    | 'LLAMA_POOLING_TYPE_NONE'
    | 'LLAMA_POOLING_TYPE_MEAN'
    | 'LLAMA_POOLING_TYPE_CLS';
  // context extending
  rope_scaling_type?:
    | 'LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED'
    | 'LLAMA_ROPE_SCALING_TYPE_NONE'
    | 'LLAMA_ROPE_SCALING_TYPE_LINEAR'
    | 'LLAMA_ROPE_SCALING_TYPE_YARN';
  rope_freq_base?: number;
  rope_freq_scale?: number;
  yarn_ext_factor?: number;
  yarn_attn_factor?: number;
  yarn_beta_fast?: number;
  yarn_beta_slow?: number;
  yarn_orig_ctx?: number;
  // TODO: add group attention
  // optimizations
  cache_type_k?: 'f32' | 'f16' | 'q8_0' | 'q5_1' | 'q5_0' | 'q4_1' | 'q4_0';
  cache_type_v?: 'f32' | 'f16' | 'q8_0' | 'q5_1' | 'q5_0' | 'q4_1' | 'q4_0';
  flash_attn?: boolean; // true is auto, false is disabled
}

export interface PerfContextData {
  success: boolean;
  t_start_ms: number;
  t_load_ms: number;
  t_p_eval_ms: number;
  t_eval_ms: number;
  n_p_eval: number;
  n_eval: number;
  n_reused: number;
}

export interface SamplingConfig {
  // See sampling.h for more details
  mirostat?: number | undefined; // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
  mirostat_eta?: number | undefined;
  mirostat_tau?: number | undefined;
  samplers_sequence?: string[] | undefined; // unused for now
  temp?: number | undefined; // temperature
  top_p?: number | undefined;
  top_k?: number | undefined;
  penalty_last_n?: number | undefined;
  penalty_repeat?: number | undefined;
  penalty_freq?: number | undefined;
  penalty_present?: number | undefined;
  dynatemp_range?: number | undefined;
  dynatemp_exponent?: number | undefined;
  grammar?: string;
  n_prev?: number | undefined;
  n_probs?: number | undefined;
  min_p?: number | undefined;
  typ_p?: number | undefined;
  typical_p?: number | undefined;
  logit_bias?: { token: number; bias: number }[] | undefined;
}

export interface CompletionChunk {
  token: number;
  piece: Uint8Array;
  currentText: string;
}

export interface CompletionOptions {
  /**
   * When processing input prompt, we don't need to get output tokens. Only used by llama_decode()
   * Default: false
   */
  skipLogits?: boolean;
  /**
   * Optional abort signal to stop the generation.
   * This can also be used to stop during prompt processing. In this case, it will throw WllamaAbortError.
   */
  abortSignal?: AbortSignal;
  /**
   * If true, return an AsyncIterable instead of a string
   */
  stream?: boolean;
}

export interface ChatCompletionOptions {
  nPredict?: number;
  onNewToken?(
    token: number,
    piece: Uint8Array,
    currentText: string,
    optionals: {
      /**
       * DEPRECATED, use ChatCompletionOptions["abortSignal"] instead
       */
      abortSignal: () => any;
    }
  ): any;
  sampling?: SamplingConfig;
  /**
   * List of custom token IDs for stopping the generation.
   * Note: To convert from text to token ID, use lookupToken()
   */
  stopTokens?: number[];
  /**
   * Equivalent to `cache_prompt` option in llama.cpp server.
   * Useful for chat, because it skip evaluating the history part of the conversation.
   */
  useCache?: boolean;
  /**
   * Optional abort signal to stop the generation.
   * This can also be used to stop during prompt processing (with a bit of delay.)
   */
  abortSignal?: AbortSignal;
  /**
   * If true, return an AsyncIterable instead of a string
   */
  stream?: boolean;
}

export interface ModelMetadata {
  hparams: {
    nVocab: number;
    nCtxTrain: number;
    nEmbd: number;
    nLayer: number;
  };
  meta: Record<string, string>;
}

export interface ContextOptions {
  /**
   * Allow switching between embeddings / generation mode. Useful for models like GritLM.
   */
  embeddings: boolean;
}

export interface LoadedContextInfo {
  n_vocab: number;
  n_ctx: number;
  n_batch: number;
  n_ubatch: number;
  n_ctx_train: number;
  n_embd: number;
  n_layer: number;
  metadata: Record<string, string>;
  token_bos: number;
  token_eos: number;
  token_eot: number;
  list_tokens_eog: number[];
  has_encoder: boolean;
  token_decoder_start: number;
  add_bos_token: boolean;
  add_eos_token: boolean;
}

export interface WllamaTreeNode {
  id: number;
  parentId: number | null;
  childIds: number[];
  turn: {
    user: string;
    assistant: string;
  };
  status: string;
  prefixTokenCount: number;
  generationTimeMs: number;
  cachedTokenCount: number;
  snapshotTokenBytes: number;
  createdAt: number;
  lastAccessedAt: number;
}

export interface WllamaTreeState {
  nodes: Map<number, WllamaTreeNode>;
  rootId: number;
  activeNodeId: number;
  nextId: number;
  contextMemoryBytes: number;
  memoryCapBytes: number;
  totalSnapshotTokenBytes: number;
  lastPrunedNodeIds: number[];
  lastPrunedAt: number;
  tieredCacheEnabled: boolean;
  tierStats: {
    l1Tokens: number;
    l2Tokens: number;
    l3Tokens: number;
    l1Slots: number;
    l2Slots: number;
    l3Slots: number;
    promotions: number;
    demotions: number;
    diskReads: number;
    diskWrites: number;
    l3OverflowEvents: number;
    restoreAttempts: number;
    restoreHitsL1: number;
    restoreHitsL2: number;
    restoreHitsL3: number;
    restoreMisses: number;
    restoreRebuilds: number;
    parentRecoverAttempts: number;
    parentRecoverSuccesses: number;
    parentRecoverFailures: number;
    slotAllocHits: number;
    slotAllocMisses: number;
    slotEvictL1: number;
    slotEvictL2: number;
    slotEvictL3: number;
    fallbackReplays: number;
  };
}

export interface WllamaTieredCacheOptions {
  enabled?: boolean;
  l1TokenCap?: number;
  l2TokenCap?: number;
  l3TokenCap?: number;
  pruneL1L2TokenThreshold?: number;
  pruneL2L3TokenThreshold?: number;
  replacementPolicy?: 'hybrid' | 'lru' | 'lfu' | 'size-only' | 'random';
  l3Path?: string;
}

export interface WllamaChatFromNodeOptions extends ChatCompletionOptions {
  onChunk?: (piece: string, fullText: string) => void;
}

export interface WllamaChatSessionOptions extends WllamaChatFromNodeOptions {}

interface EngineChatRequest {
  id: number;
  parentId: number;
  userText: string;
  options: WllamaChatFromNodeOptions;
  enqueuedAt: number;
  nAheadAtEnqueue: number;
  baselineWorkMs: number;
  waitBudgetMs: number;
  processingStartedAt?: number;
  resolve: (value: { nodeId: number; assistantText: string; state: WllamaTreeState }) => void;
  reject: (reason?: unknown) => void;
  cleanupAbort?: () => void;
}

/**
 * Logger preset with debug messages suppressed
 */
export const LoggerWithoutDebug = {
  ...console,
  debug: () => {},
};

export type WllamaErrorType =
  | 'model_not_loaded'
  | 'download_error'
  | 'load_error'
  | 'kv_cache_full'
  | 'queue_overloaded'
  | 'unknown_error'
  | 'inference_error';

export class WllamaError extends Error {
  type: WllamaErrorType;
  constructor(message: string, type: WllamaErrorType = 'unknown_error') {
    super(message);
    this.type = type;
  }
}

/**
 * AbortError is thrown when the user wants to abort the current operation.
 * This is equivalent to AbortError in Fetch API.
 */
export class WllamaAbortError extends Error {
  override name: string = 'AbortError';
  constructor() {
    super('Operation aborted');
  }
}

export class Wllama {
  // The CacheManager and ModelManager are singleton, can be accessed by user
  public cacheManager: CacheManager;
  public modelManager: ModelManager;

  private proxy: ProxyToWorker = null as any;
  private config: WllamaConfig;
  private pathConfig: AssetsPathConfig;
  private useMultiThread: boolean = false;
  private useWebGPU: boolean = false;
  private nbThreads: number = 1;
  private useEmbeddings: boolean = false;
  // available when loaded
  private loadedContextInfo: LoadedContextInfo = null as any;
  private bosToken: number = -1;
  private eosToken: number = -1;
  private eotToken: number = -1;
  private eogTokens: Set<number> = new Set();
  private addBosToken: boolean = false;
  private addEosToken: boolean = false;
  private chatTemplate?: string;
  private metadata?: ModelMetadata;
  private samplingConfig: SamplingConfig = {};
  private hasEncoder: boolean = false;
  private decoderStartToken: number = -1;
  private nCachedTokens: number = 0;
  private nextEngineChatRequestId: number = 1;
  private engineChatQueue: EngineChatRequest[] = [];
  private engineChatQueueRunning: boolean = false;

  constructor(pathConfig: AssetsPathConfig, wllamaConfig: WllamaConfig = {}) {
    checkEnvironmentCompatible();
    if (!pathConfig) throw new WllamaError('AssetsPathConfig is required');
    this.pathConfig = pathConfig;
    this.config = wllamaConfig;
    this.cacheManager = wllamaConfig.cacheManager ?? new CacheManager();
    this.modelManager =
      wllamaConfig.modelManager ??
      new ModelManager({
        cacheManager: this.cacheManager,
        logger: wllamaConfig.logger ?? console,
        parallelDownloads: wllamaConfig.parallelDownloads,
        allowOffline: wllamaConfig.allowOffline,
      });
  }

  private logger() {
    return this.config.logger ?? console;
  }

  private isEngineChatTraceEnabled(): boolean {
    return this.config.engineChatTraceEnabled ?? false;
  }

  private traceEngineChat(message: string, level: 'debug' | 'warn' = 'debug') {
    if (!this.isEngineChatTraceEnabled()) {
      return;
    }
    if (level === 'warn') {
      this.logger().warn(`[EngineChatTrace] ${message}`);
      return;
    }
    this.logger().debug(`[EngineChatTrace] ${message}`);
  }

  private checkModelLoaded() {
    if (!this.isModelLoaded()) {
      throw new WllamaError(
        'loadModel() is not yet called',
        'model_not_loaded'
      );
    }
  }

  /**
   * Get the libllama version string, e.g. "b6327-4d74393".
   *
   * @returns version string embedded at build time.
   */
  static getLibllamaVersion(): string {
    return LIBLLAMA_VERSION;
  }

  /**
   * Check if the model is loaded via `loadModel()`
   */
  isModelLoaded(): boolean {
    return !!this.proxy && !!this.metadata;
  }

  /**
   * Get token ID associated to BOS (begin of sentence) token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getBOS(): number {
    return this.bosToken;
  }

  /**
   * Get token ID associated to EOS (end of sentence) token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getEOS(): number {
    return this.eosToken;
  }

  /**
   * Get token ID associated to EOT (end of turn) token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getEOT(): number {
    return this.eotToken;
  }

  /**
   * Check if a given token is end-of-generation token (e.g. EOS, EOT, etc.)
   *
   * @param token the token ID to be checked
   * @returns true if the token is EOS, EOT, or any other end-of-generation tokens
   */
  isTokenEOG(token: number): boolean {
    return (
      token === this.eosToken ||
      token === this.eotToken ||
      this.eogTokens.has(token)
    );
  }

  /**
   * Get token ID associated to token used by decoder, to start generating output sequence(only usable for encoder-decoder architecture). In other words, encoder uses normal BOS and decoder uses this token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getDecoderStartToken(): number {
    return this.decoderStartToken;
  }

  /**
   * Get model hyper-parameters and metadata
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns ModelMetadata
   */
  getModelMetadata(): ModelMetadata {
    this.checkModelLoaded();
    return this.metadata!;
  }

  /**
   * Check if we're currently using multi-thread build.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if multi-thread is used.
   */
  isMultithread(): boolean {
    this.checkModelLoaded();
    return this.useMultiThread;
  }

  /**
   * Get number of threads used in the current context.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns number of threads
   */
  getNumThreads(): number {
    this.checkModelLoaded();
    return this.nbThreads;
  }

  usingWebGPU(): boolean {
    this.checkModelLoaded();
    return this.useWebGPU;
  }

  /**
   * Check if the current model uses encoder-decoder architecture
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if multi-thread is used.
   */
  isEncoderDecoderArchitecture(): boolean {
    this.checkModelLoaded();
    return this.hasEncoder;
  }

  /**
   * Must we add BOS token to the tokenized sequence?
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if BOS token must be added to the sequence
   */
  mustAddBosToken(): boolean {
    this.checkModelLoaded();
    return this.addBosToken;
  }

  /**
   * Must we add EOS token to the tokenized sequence?
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if EOS token must be added to the sequence
   */
  mustAddEosToken(): boolean {
    this.checkModelLoaded();
    return this.addEosToken;
  }

  /**
   * Get the jinja chat template comes with the model. It only available if the original model (before converting to gguf) has the template in `tokenizer_config.json`
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns the jinja template. null if there is no template in gguf
   */
  getChatTemplate(): string | null {
    this.checkModelLoaded();
    return this.chatTemplate ?? null;
  }

  /**
   * Load model from a given URL (or a list of URLs, in case the model is splitted into smaller files)
   * - If the model already been downloaded (via `downloadModel()`), then we will use the cached model
   * - Else, we download the model from internet
   * @param modelUrl URL to the GGUF file. If the model is splitted, pass the URL to the first shard.
   * @param config
   */
  async loadModelFromUrl(
    modelUrl: string | string[],
    config: LoadModelConfig & DownloadOptions & { useCache?: boolean } = {}
  ): Promise<void> {
    const url: string = isString(modelUrl) ? (modelUrl as string) : modelUrl[0];
    const useCache = config.useCache ?? true;
    const model = useCache
      ? await this.modelManager.getModelOrDownload(url, config)
      : await this.modelManager.downloadModel(url, config);
    const blobs = await model.open();
    return await this.loadModel(blobs, config);
  }

  /**
   * Load model from a given Hugging Face model ID and file path.
   *
   * @param modelId The HF model ID, for example: 'ggml-org/models'
   * @param filePath The GGUF file path, for example: 'tinyllamas/stories15M-q4_0.gguf'
   * @param config
   */
  async loadModelFromHF(
    modelId: string,
    filePath: string,
    config: LoadModelConfig & DownloadOptions & { useCache?: boolean } = {}
  ) {
    if (!modelId.match(HF_MODEL_ID_REGEX)) {
      throw new WllamaError(HF_MODEL_ID_REGEX_EXPLAIN, 'download_error');
    }
    if (!isValidGgufFile(filePath)) {
      throw new WllamaError('Only GGUF file is supported', 'download_error');
    }
    return await this.loadModelFromUrl(
      `https://huggingface.co/${modelId}/resolve/main/${filePath}`,
      config
    );
  }

  /**
   * Load model from a given list of Blob.
   *
   * You can pass multiple buffers into the function (in case the model contains multiple shards).
   *
   * @param ggufBlobsOrModel Can be either list of Blobs (in case you use local file), or a Model object (in case you use ModelManager)
   * @param config LoadModelConfig
   */
  async loadModel(
    ggufBlobsOrModel: Blob[] | Model,
    config: LoadModelConfig = {}
  ): Promise<void> {
    const blobs: Blob[] =
      ggufBlobsOrModel instanceof Model
        ? await ggufBlobsOrModel.open()
        : [...(ggufBlobsOrModel as Blob[])]; // copy array
    if (blobs.some((b) => b.size === 0)) {
      throw new WllamaError(
        'Input model (or splits) must be non-empty Blob or File',
        'load_error'
      );
    }
    sortFileByShard(blobs);
    if (this.proxy) {
      throw new WllamaError('Module is already initialized', 'load_error');
    }
    if (this.config.preferWebGPU) {
      if (navigator.gpu) {
        this.useWebGPU = true;
      } else {
        this.logger().warn(
          'WebGPU backend requested but WebGPU is not available, falling back to CPU'
        );
      }
    }
    // detect if we can use multi-thread
    if (await isSupportMultiThread()) {
      if (this.pathConfig['multi-thread/wllama.wasm']) {
        const hwConcurrency = Math.floor((navigator.hardwareConcurrency || 1) / 2);
        this.nbThreads = config.n_threads ?? hwConcurrency;
        if (this.nbThreads > 1) {
          this.useMultiThread = true;
        } else {
          this.logger().warn(
            'Falling back single-thread due to n_threads configuration or limited hardware concurrency'
          );
        }
      } else {
        this.logger().warn(
          'Missing paths to "multi-thread/wllama.wasm", falling back to single-thread'
        );
      }
    } else {
      this.logger().warn(
        'Multi-threads are not supported in this environment, falling back to single-thread'
      );
    }

    // TODO: investigate why WebGPU + multi-threading causes performance issues
    if (this.useWebGPU) {
      this.logger().warn(
        'Disabling multi-threading when using WebGPU backend'
      );
      this.useMultiThread = false;
      this.nbThreads = 1;
    }

    const mPathConfig = this.useMultiThread
      ? {
          'wllama.wasm': absoluteUrl(
            this.pathConfig['multi-thread/wllama.wasm']!!
          ),
        }
      : {
          'wllama.wasm': absoluteUrl(
            this.pathConfig['single-thread/wllama.wasm']
          ),
        };
    this.proxy = new ProxyToWorker(
      mPathConfig,
      this.nbThreads,
      this.config.suppressNativeLog ?? false,
      this.logger()
    );
    const modelFiles = blobs.map((blob, i) => ({
      name: `model-${i}.gguf`,
      blob,
    }));
    await this.proxy.moduleInit(modelFiles);
    // run it
    const startResult: any = await this.proxy.wllamaStart();
    if (!startResult.success) {
      throw new WllamaError(
        `Error while calling start function, result = ${startResult}`
      );
    }
    // load the model
    const resolvedNCtx = config.n_ctx || 1024;
    const maxNSeqFromCtx = Math.max(1, Math.floor(resolvedNCtx / MIN_CTX_PER_SEQUENCE));
    const hardNSeqMax = Math.min(MAX_SAFE_N_SEQ_MAX, maxNSeqFromCtx);
    const requestedNSeqMax = config.n_seq_max ?? hardNSeqMax;
    const safeNSeqMax = Math.min(Math.max(1, Math.floor(requestedNSeqMax)), hardNSeqMax);
    if (safeNSeqMax !== requestedNSeqMax) {
      this.logger().warn(
        `n_seq_max=${requestedNSeqMax} is not supported by current runtime/context, using ${safeNSeqMax} instead`
      );
    }

    const loadResult: GlueMsgLoadRes = await this.proxy.wllamaAction('load', {
      _name: 'load_req',
      use_mmap: true,
      use_mlock: true,
      use_webgpu: this.useWebGPU,
      n_gpu_layers: this.useWebGPU ? 999 : 0,
      no_perf: this.config.noPerf ?? false,
      seed: config.seed || Math.floor(Math.random() * 100000),
      n_ctx: config.n_ctx || 1024,
      n_threads: this.nbThreads,
      n_ctx_auto: false, // not supported for now
      model_paths: modelFiles.map((f) => `models/${f.name}`),
      embeddings: config.embeddings,
      offload_kqv: config.offload_kqv,
      n_batch: config.n_batch,
      pooling_type: config.pooling_type as string,
      rope_scaling_type: config.rope_scaling_type as string,
      rope_freq_base: config.rope_freq_base,
      rope_freq_scale: config.rope_freq_scale,
      yarn_ext_factor: config.yarn_ext_factor,
      yarn_attn_factor: config.yarn_attn_factor,
      yarn_beta_fast: config.yarn_beta_fast,
      yarn_beta_slow: config.yarn_beta_slow,
      yarn_orig_ctx: config.yarn_orig_ctx,
      cache_type_k: config.cache_type_k as string,
      cache_type_v: config.cache_type_v as string,
      n_seq_max: safeNSeqMax,
      kv_unified: config.kv_unified ?? true,
      flash_attn: config.flash_attn,
      swa_full: true, // TODO: properly support SWA
    });
    const loadedCtxInfo: LoadedContextInfo = {
      ...loadResult,
      metadata: {},
    };
    for (let i = 0; i < loadResult.metadata_key.length; i++) {
      loadedCtxInfo.metadata[loadResult.metadata_key[i]] =
        loadResult.metadata_val[i];
    }
    this.bosToken = loadedCtxInfo.token_bos;
    this.eosToken = loadedCtxInfo.token_eos;
    this.eotToken = loadedCtxInfo.token_eot;
    this.useEmbeddings = !!config.embeddings;
    this.metadata = {
      hparams: {
        nVocab: loadedCtxInfo.n_vocab,
        nCtxTrain: loadedCtxInfo.n_ctx_train,
        nEmbd: loadedCtxInfo.n_embd,
        nLayer: loadedCtxInfo.n_layer,
      },
      meta: loadedCtxInfo.metadata,
    };
    this.hasEncoder = !!loadedCtxInfo.has_encoder;
    this.decoderStartToken = loadedCtxInfo.token_decoder_start;
    this.addBosToken = loadedCtxInfo.add_bos_token;
    this.addEosToken = loadedCtxInfo.add_eos_token;
    this.chatTemplate = loadedCtxInfo.metadata['tokenizer.chat_template'];
    this.loadedContextInfo = loadedCtxInfo;
    this.eogTokens = new Set(loadedCtxInfo.list_tokens_eog);
    this.logger().debug({ loadedCtxInfo });
  }

  getLoadedContextInfo(): LoadedContextInfo {
    this.checkModelLoaded();
    if (!this.loadedContextInfo) {
      throw new WllamaError('Loaded context info is not available');
    }
    // copy object
    return { ...this.loadedContextInfo };
  }

  //////////////////////////////////////////////
  // High level API

  /**
   * Calculate embedding vector for a given text.
   * By default, BOS and EOS tokens will be added automatically. You can use the "skipBOS" and "skipEOS" option to disable it.
   * @param text Input text
   * @returns An embedding vector
   */
  async createEmbedding(
    text: string,
    options: {
      skipBOS?: boolean;
      skipEOS?: boolean;
    } = {}
  ): Promise<number[]> {
    this.checkModelLoaded();
    const opt = {
      skipBOS: false,
      skipEOS: false,
      ...options,
    };
    await this.samplingInit(this.samplingConfig);
    await this.kvClear();
    const tokens = await this.tokenize(text);
    if (this.bosToken && !opt.skipBOS) {
      tokens.unshift(this.bosToken);
    }
    if (this.eosToken && !opt.skipEOS) {
      tokens.push(this.eosToken);
    }
    const result = await this.embeddings(tokens);
    return result;
  }

  /**
   * Make completion for a given chat messages.
   *
   * NOTE: this function uses the chat template (if available) to format the chat messages. If the template is not available, it will use the default format (chatml). It can throw an error if the chat template is not compatible.
   *
   * @param messages Chat messages
   * @param options
   * @returns Output completion text (only the completion part)
   */
  async createChatCompletion(
    messages: WllamaChatMessage[],
    options: ChatCompletionOptions & { stream?: false }
  ): Promise<string>;
  async createChatCompletion(
    messages: WllamaChatMessage[],
    options: ChatCompletionOptions & { stream: true }
  ): Promise<AsyncIterable<CompletionChunk>>;
  async createChatCompletion(
    messages: WllamaChatMessage[],
    options: ChatCompletionOptions
  ): Promise<string | AsyncIterable<CompletionChunk>> {
    const prompt = await this.formatChat(messages, true);
    return options.stream
      ? await this.createCompletionGenerator(prompt, options)
      : await this.createCompletion(prompt, { ...options, stream: false });
  }

  /**
   * Make completion for a given text.
   * @param prompt Input text
   * @param options
   * @returns Output completion text (only the completion part)
   */
  async createCompletion(
    prompt: string,
    options: ChatCompletionOptions & { stream?: false }
  ): Promise<string>;
  async createCompletion(
    prompt: string,
    options: ChatCompletionOptions & { stream: true }
  ): Promise<AsyncIterable<CompletionChunk>>;
  async createCompletion(
    prompt: string,
    options: ChatCompletionOptions
  ): Promise<string | AsyncIterable<CompletionChunk>> {
    return options.stream
      ? await this.createCompletionGenerator(prompt, options)
      : await this.createCompletionImpl(prompt, { ...options, stream: false });
  }

  /**
   * Private implementation of createCompletion
   */
  private async createCompletionImpl(
    prompt: string,
    options: ChatCompletionOptions
  ): Promise<string> {
    this.checkModelLoaded();
    this.samplingConfig = options.sampling ?? {};
    await this.samplingInit(this.samplingConfig);
    const stopTokens = new Set(options.stopTokens ?? []);
    // process prompt
    let tokens = await this.tokenize(prompt, true);
    if (this.addBosToken && tokens[0] !== this.bosToken) {
      tokens.unshift(this.bosToken);
    }
    // maybe reuse KV cache
    if (options.useCache) {
      tokens = await this.computeNonCachedTokens(tokens);
    } else {
      await this.kvClear();
    }
    // decode/encode tokens
    await this.samplingAccept(tokens);
    if (this.isEncoderDecoderArchitecture()) {
      await this.encode(tokens);
      await this.decode([this.getDecoderStartToken()], {});
    } else {
      await this.decode(tokens, {});
    }
    let outBuf = new Uint8Array();
    // abort signal
    let abort = false;
    // abortSignalFn is a legacy function, use options.abortSignal instead
    const abortSignalFn = () => {
      abort = true;
    };
    // predict next tokens
    for (let i = 0; i < (options.nPredict ?? Infinity); i++) {
      const sampled = await this.samplingSample();
      if (this.isTokenEOG(sampled.token) || stopTokens.has(sampled.token)) {
        break; // stop token
      }
      // @ts-ignore Type 'Uint8Array<ArrayBufferLike>' is not assignable to type 'Uint8Array<ArrayBuffer>'
      outBuf = joinBuffers([outBuf, sampled.piece]);
      if (options.onNewToken) {
        options.onNewToken(sampled.token, sampled.piece, bufToText(outBuf), {
          abortSignal: abortSignalFn, // legacy
        });
      }
      if (abort || options.abortSignal?.aborted) {
        break; // abort signal is set
      }
      // decode next token
      await this.samplingAccept([sampled.token]);
      await this.decode([sampled.token], {});
    }
    return bufToText(outBuf);
  }

  /**
   * Same with `createCompletion`, but returns an async iterator instead.
   */
  private createCompletionGenerator(
    prompt: string,
    options: Exclude<ChatCompletionOptions, 'onNewToken'>
  ): Promise<AsyncIterable<CompletionChunk>> {
    return new Promise((resolve, reject) => {
      const createGenerator = cbToAsyncIter(
        (callback: (val?: CompletionChunk, done?: boolean) => void) => {
          this.createCompletionImpl(prompt, {
            ...options,
            onNewToken: (token, piece, currentText) => {
              callback({ token, piece, currentText }, false);
            },
          })
            .catch(reject)
            .then(() => {
              callback(undefined, true);
            });
        }
      );
      resolve(createGenerator());
    });
  }

  //////////////////////////////////////////////
  // Low level API

  /**
   * Create or reset the ctx_sampling
   * @param config
   * @param pastTokens In case re-initializing the ctx_sampling, you can re-import past tokens into the new context
   */
  async samplingInit(
    config: SamplingConfig,
    pastTokens: number[] = []
  ): Promise<void> {
    this.checkModelLoaded();
    this.samplingConfig = config;
    const logitBias = config.logit_bias ?? [];
    const logitBiasTok = logitBias.map((b) => b.token);
    const logitBiasVal = logitBias.map((b) => b.bias);
    const result = await this.proxy.wllamaAction<GlueMsgSamplingAcceptRes>(
      'sampling_init',
      {
        _name: 'sint_req',
        ...config,
        logit_bias_toks: logitBiasTok,
        logit_bias_vals: logitBiasVal,
        tokens: pastTokens,
      }
    );
    if (!result.success) {
      throw new WllamaError('Failed to initialize sampling');
    }
  }

  /**
   * Get a list of pieces in vocab.
   * NOTE: This function is slow, should only be used once.
   * @returns A list of Uint8Array. The nth element in the list associated to nth token in vocab
   */
  async getVocab(): Promise<Uint8Array[]> {
    this.checkModelLoaded();
    const result = await this.proxy.wllamaAction<GlueMsgGetVocabRes>(
      'get_vocab',
      {
        _name: 'gvoc_req',
      }
    );
    return result.vocab;
  }

  /**
   * Lookup to see if a token exist in vocab or not. Useful for searching special tokens like "<|im_start|>"
   * NOTE: It will match the whole token, so do not use it as a replacement for tokenize()
   * @param piece
   * @returns Token ID associated to the given piece. Returns -1 if cannot find the token.
   */
  async lookupToken(piece: string): Promise<number> {
    this.checkModelLoaded();
    const result = await this.proxy.wllamaAction<GlueMsgLookupTokenRes>(
      'lookup_token',
      {
        _name: 'lkup_req',
        piece,
      }
    );
    if (!result.success) {
      return -1;
    } else {
      return result.token as number;
    }
  }

  /**
   * Convert a given text to list of tokens
   * @param text
   * @param special Should split special tokens?
   * @returns List of token ID
   */
  async tokenize(text: string, special: boolean = true): Promise<number[]> {
    this.checkModelLoaded();
    const result = await this.proxy.wllamaAction<GlueMsgTokenizeRes>(
      'tokenize',
      {
        _name: 'tokn_req',
        text,
        special: !!special,
      }
    );
    return result.tokens;
  }

  /**
   * Convert a list of tokens to text
   * @param tokens
   * @param returnString Return a string instead of Uint8Array
   * @returns Uint8Array, which maybe an unfinished unicode
   */
  async detokenize(tokens: number[], returnString?: false): Promise<Uint8Array>;
  async detokenize(tokens: number[], returnString: true): Promise<string>;
  async detokenize(
    tokens: number[],
    returnString: true | false = false
  ): Promise<Uint8Array | string> {
    this.checkModelLoaded();
    const result = await this.proxy.wllamaAction<GlueMsgDetokenizeRes>(
      'detokenize',
      {
        _name: 'dtkn_req',
        tokens,
      }
    );
    return returnString ? bufToText(result.buffer) : result.buffer;
  }

  /**
   * Run llama_decode()
   * @param tokens A list of tokens to be decoded
   * @param options Additional options
   * @returns n_past (number of tokens so far in the sequence)
   */
  async decode(
    tokens: number[],
    options: CompletionOptions
  ): Promise<{ nPast: number }> {
    this.checkModelLoaded();
    if (this.useEmbeddings) {
      throw new WllamaError(
        'embeddings is enabled. Use wllama.setOptions({ embeddings: false }) to disable it.'
      );
    }
    if (tokens.length === 0) {
      // do not call llama_decode if list of tokens is empty
      return {
        nPast: this.nCachedTokens,
      };
    }
    if (this.nCachedTokens + tokens.length > this.loadedContextInfo.n_ctx) {
      throw new WllamaError(
        'Running out of context cache. Please increase n_ctx when loading the model',
        'kv_cache_full'
      );
    }
    const batches = this.breakTokensIntoBatches(
      tokens,
      this.loadedContextInfo.n_batch
    );
    let result: any;
    for (let i = 0; i < batches.length; i++) {
      if (options?.abortSignal?.aborted) {
        throw new WllamaAbortError();
      }
      const isNotLast = batches.length > 1 && i < batches.length - 1;
      result = await this.proxy.wllamaAction<GlueMsgDecodeRes>('decode', {
        _name: 'deco_req',
        tokens: batches[i],
        skip_logits: options.skipLogits || isNotLast,
      });
      if (result.error) {
        throw new WllamaError(result.error);
      } else if (!result.success) {
        throw new WllamaError('Cannot encode, unknown error');
      }
    }
    this.nCachedTokens = result.n_past;
    return { nPast: result.n_past };
  }

  /**
   * Run llama_encode()
   * @param tokens A list of tokens to be encoded
   * @param options Additional options
   * @returns n_past (number of tokens so far in the sequence)
   */
  async encode(
    tokens: number[],
    options?: CompletionOptions
  ): Promise<{ nPast: number }> {
    this.checkModelLoaded();
    if (!this.hasEncoder) {
      throw new WllamaError(
        'This model does not use encoder-decoder architecture.',
        'inference_error'
      );
    }
    if (this.useEmbeddings) {
      throw new WllamaError(
        'embeddings is enabled. Use wllama.setOptions({ embeddings: false }) to disable it.',
        'inference_error'
      );
    }
    if (tokens.length === 0) {
      // do not call llama_encode if list of tokens is empty
      return {
        nPast: this.nCachedTokens,
      };
    }
    if (this.nCachedTokens + tokens.length > this.loadedContextInfo.n_ctx) {
      throw new WllamaError(
        'Running out of context cache. Please increase n_ctx when loading the model',
        'kv_cache_full'
      );
    }
    const batches = this.breakTokensIntoBatches(
      tokens,
      this.loadedContextInfo.n_batch
    );
    let result: any;
    for (let i = 0; i < batches.length; i++) {
      if (options?.abortSignal?.aborted) {
        throw new WllamaAbortError();
      }
      result = await this.proxy.wllamaAction<GlueMsgDecodeRes>('encode', {
        _name: 'enco_req',
        tokens: batches[i],
      });
      if (result.error) {
        throw new WllamaError(result.error);
      } else if (!result.success) {
        throw new WllamaError('Cannot encode, unknown error');
      }
    }
    this.nCachedTokens = result.n_past;
    return { nPast: result.n_past };
  }

  private breakTokensIntoBatches(
    tokens: number[],
    maxBatchSize: number
  ): number[][] {
    const batches: number[][] = [];
    for (let i = 0; i < tokens.length; i += maxBatchSize) {
      batches.push(tokens.slice(i, i + maxBatchSize));
    }
    return batches;
  }

  /**
   * Sample a new token (remember to samplingInit() at least once before calling this function)
   * @returns the token ID and its detokenized value (which maybe an unfinished unicode)
   */
  async samplingSample(): Promise<{ piece: Uint8Array; token: number }> {
    this.checkModelLoaded();
    const result = await this.proxy.wllamaAction<GlueMsgSamplingSampleRes>(
      'sampling_sample',
      {
        _name: 'ssam_req',
      }
    );
    return {
      piece: result.piece,
      token: result.token,
    };
  }

  /**
   * Accept and save a new token to ctx_sampling
   * @param tokens
   */
  async samplingAccept(tokens: number[]): Promise<void> {
    this.checkModelLoaded();
    const result = await this.proxy.wllamaAction<GlueMsgSamplingAcceptRes>(
      'sampling_accept',
      {
        _name: 'sacc_req',
        tokens,
      }
    );
    if (!result.success) {
      throw new WllamaError('samplingAccept unknown error');
    }
  }

  /**
   * Get softmax-ed probability of logits, can be used for custom sampling
   * @param topK Get top K tokens having highest logits value. If topK == -1, we return all n_vocab logits, but this is not recommended because it's slow.
   */
  async getLogits(topK: number = 40): Promise<{ token: number; p: number }[]> {
    this.checkModelLoaded();
    const result = await this.proxy.wllamaAction<GlueMsgGetLogitsRes>(
      'get_logits',
      {
        _name: 'glog_req',
        top_k: topK,
      }
    );
    const logits: { token: number; p: number }[] = [];
    for (let i = 0; i < result.tokens.length; i++) {
      logits.push({
        token: result.tokens[i],
        p: result.probs[i],
      });
    }
    return logits;
  }

  /**
   * Calculate embeddings for a given list of tokens. Output vector is always normalized
   * @param tokens
   * @returns A list of number represents an embedding vector of N dimensions
   */
  async embeddings(tokens: number[]): Promise<number[]> {
    this.checkModelLoaded();
    if (!this.useEmbeddings) {
      throw new WllamaError(
        'embeddings is disabled. Use wllama.setOptions({ embeddings: true }) to enable it.',
        'inference_error'
      );
    }
    if (this.nCachedTokens > 0) {
      this.logger().warn(
        'Embeddings: KV cache is not empty, this may produce incorrect results'
      );
    }
    if (this.nCachedTokens + tokens.length > this.loadedContextInfo.n_ctx) {
      throw new WllamaError(
        'Running out of context cache. Please increase n_ctx when loading the model',
        'kv_cache_full'
      );
    }
    if (tokens.length > this.loadedContextInfo.n_batch) {
      throw new WllamaError(
        'Embedding tokens does not fit into batch. Please increase n_batch when loading the model',
        'inference_error'
      );
    }
    if (tokens.length > this.loadedContextInfo.n_ubatch) {
      throw new WllamaError(
        'Embedding tokens does not fit into physical batch. Please increase n_ubatch when loading the model',
        'inference_error'
      );
    }
    const result = await this.proxy.wllamaAction<GlueMsgGetEmbeddingsRes>(
      'embeddings',
      {
        _name: 'gemb_req',
        tokens,
      }
    );
    if (!result.success) {
      throw new WllamaError('embeddings unknown error');
    } else {
      return result.embeddings;
    }
  }

  /**
   * Remove and shift some tokens from KV cache.
   * Keep n_keep, remove n_discard then shift the rest
   * @param nKeep
   * @param nDiscard
   */
  async kvRemove(nKeep: number, nDiscard: number): Promise<void> {
    this.checkModelLoaded();
    if (nDiscard === 0) return;
    const result = await this.proxy.wllamaAction<GlueMsgGetKvRemoveRes>(
      'kv_remove',
      {
        _name: 'kvcr_req',
        n_keep: nKeep,
        n_discard: nDiscard,
      }
    );
    if (!result.success) {
      throw new WllamaError('kvRemove unknown error');
    }
    // When nDiscard is negative (-1), it means remove everything after nKeep
    if (nDiscard < 0) {
      this.nCachedTokens = nKeep;
    } else {
      this.nCachedTokens -= nDiscard;
    }
  }

  /**
   * Clear all tokens in KV cache
   */
  async kvClear(): Promise<void> {
    this.checkModelLoaded();
    const result = await this.proxy.wllamaAction<GlueMsgGetKvClearRes>(
      'kv_clear',
      {
        _name: 'kvcc_req',
      }
    );
    if (!result.success) {
      throw new WllamaError('kvClear unknown error');
    }
    this.nCachedTokens = 0;
  }

  private mapTreeState(result: GlueMsgTreeStateRes): WllamaTreeState {
    const nodes = new Map<number, WllamaTreeNode>();
    for (let i = 0; i < result.ids.length; i++) {
      const childStart = result.child_offsets[i] ?? 0;
      const childEnd = result.child_offsets[i + 1] ?? childStart;
      nodes.set(result.ids[i], {
        id: result.ids[i],
        parentId: result.parent_ids[i] < 0 ? null : result.parent_ids[i],
        childIds: result.child_ids.slice(childStart, childEnd),
        turn: {
          user: result.user_texts[i] ?? '',
          assistant: result.assistant_texts[i] ?? '',
        },
        status: result.statuses[i] ?? 'pending',
        prefixTokenCount: result.prefix_token_counts[i] ?? -1,
        generationTimeMs: result.generation_time_ms[i] ?? -1,
        cachedTokenCount: result.cached_token_counts[i] ?? 0,
        snapshotTokenBytes: result.snapshot_token_bytes[i] ?? 0,
        createdAt: (result.created_at_s[i] ?? 0) * 1000,
        lastAccessedAt: (result.last_accessed_at_s[i] ?? 0) * 1000,
      });
    }

    return {
      nodes,
      rootId: result.root_id,
      activeNodeId: result.active_node_id,
      nextId: result.next_id,
      contextMemoryBytes: result.context_memory_bytes,
      memoryCapBytes: result.memory_cap_bytes,
      totalSnapshotTokenBytes: result.total_snapshot_token_bytes,
      lastPrunedNodeIds: result.last_pruned_node_ids,
      lastPrunedAt: result.last_pruned_at_s * 1000,
      tieredCacheEnabled: result.tiered_cache_enabled,
      tierStats: {
        l1Tokens: result.tier_l1_tokens,
        l2Tokens: result.tier_l2_tokens,
        l3Tokens: result.tier_l3_tokens,
        l1Slots: result.tier_l1_slots,
        l2Slots: result.tier_l2_slots,
        l3Slots: result.tier_l3_slots,
        promotions: result.tier_promotions,
        demotions: result.tier_demotions,
        diskReads: result.tier_disk_reads,
        diskWrites: result.tier_disk_writes,
        l3OverflowEvents: result.tier_l3_overflow_events,
        restoreAttempts: (result as unknown as Record<string, number>).tier_restore_attempts ?? 0,
        restoreHitsL1: (result as unknown as Record<string, number>).tier_restore_hits_l1 ?? 0,
        restoreHitsL2: (result as unknown as Record<string, number>).tier_restore_hits_l2 ?? 0,
        restoreHitsL3: (result as unknown as Record<string, number>).tier_restore_hits_l3 ?? 0,
        restoreMisses: (result as unknown as Record<string, number>).tier_restore_misses ?? 0,
        restoreRebuilds: (result as unknown as Record<string, number>).tier_restore_rebuilds ?? 0,
        parentRecoverAttempts: (result as unknown as Record<string, number>).tier_parent_recover_attempts ?? 0,
        parentRecoverSuccesses: (result as unknown as Record<string, number>).tier_parent_recover_successes ?? 0,
        parentRecoverFailures: (result as unknown as Record<string, number>).tier_parent_recover_failures ?? 0,
        slotAllocHits: (result as unknown as Record<string, number>).tier_slot_alloc_hits ?? 0,
        slotAllocMisses: (result as unknown as Record<string, number>).tier_slot_alloc_misses ?? 0,
        slotEvictL1: (result as unknown as Record<string, number>).tier_slot_evict_l1 ?? 0,
        slotEvictL2: (result as unknown as Record<string, number>).tier_slot_evict_l2 ?? 0,
        slotEvictL3: (result as unknown as Record<string, number>).tier_slot_evict_l3 ?? 0,
        fallbackReplays: (result as unknown as Record<string, number>).tier_fallback_replays ?? 0,
      },
    };
  }

  private async treeInit(
    memoryCapBytes: number,
    tieredCache: WllamaTieredCacheOptions = {}
  ): Promise<WllamaTreeState> {
    this.checkModelLoaded();
    const result = await this.proxy.wllamaAction<GlueMsgTreeInitRes>('chat_init', {
      _name: 'trin_req',
      memory_cap_bytes: memoryCapBytes,
      tiered_cache_enabled: tieredCache.enabled ?? false,
      tier_l1_token_cap: tieredCache.l1TokenCap ?? 0,
      tier_l2_token_cap: tieredCache.l2TokenCap ?? 0,
      tier_l3_token_cap: tieredCache.l3TokenCap ?? 0,
      tier_prune_l1_l2_token_threshold: tieredCache.pruneL1L2TokenThreshold ?? 0,
      tier_prune_l2_l3_token_threshold: tieredCache.pruneL2L3TokenThreshold ?? 0,
      tier_replacement_policy: tieredCache.replacementPolicy ?? 'hybrid',
      tier_l3_path: tieredCache.l3Path ?? '/tmp/wllama-tier-cache',
    });
    if (!result.success) {
      throw new WllamaError(`treeInit failed: ${result.message}`);
    }
    this.nCachedTokens = 0;
    return this.treeGetState();
  }

  // High-level chat session APIs (SGLang-style) that hide tree internals.
  async chatEnsureReady(
    memoryCapBytes: number = 1024 * 1024 * 1024,
    tieredCache: WllamaTieredCacheOptions = {}
  ): Promise<WllamaTreeState> {
    this.checkModelLoaded();
    try {
      return await this.treeGetState();
    } catch (_) {
      return await this.treeInit(memoryCapBytes, tieredCache);
    }
  }

  async chatGetState(): Promise<WllamaTreeState> {
    await this.chatEnsureReady();
    return await this.treeGetState();
  }

  async chatSetActiveNode(nodeId: number): Promise<WllamaTreeState> {
    await this.chatEnsureReady();
    return await this.treeSwitch(nodeId);
  }

  async chatDeleteNode(nodeId: number): Promise<WllamaTreeState> {
    await this.chatEnsureReady();
    return await this.treeDelete(nodeId);
  }

  async chatReset(): Promise<WllamaTreeState> {
    await this.chatEnsureReady();
    return await this.treeReset();
  }

  async chatSessionInit(
    memoryCapBytes: number = 1024 * 1024 * 1024,
    tieredCache: WllamaTieredCacheOptions = {}
  ): Promise<WllamaTreeState> {
    // Reinitialize tree session so new memory/tier settings always take effect.
    // This matches UI semantics: "apply and reset session".
    return this.treeInit(memoryCapBytes, tieredCache);
  }

  async chatSessionChat(
    history: WllamaChatMessage[],
    userText: string,
    options: WllamaChatSessionOptions = {}
  ): Promise<{ nodeId: number; assistantText: string; state: WllamaTreeState }> {
    const state = await this.chatEnsureReady();
    const parentId = this.resolveNodeIdByHistory(state, history);
    return this.chatFromNode(parentId, userText, options);
  }

  async chatSessionFinish(): Promise<WllamaTreeState> {
    return this.chatReset();
  }

  async chatFromNode(
    parentId: number,
    userText: string,
    options: WllamaChatFromNodeOptions = {}
  ): Promise<{ nodeId: number; assistantText: string; state: WllamaTreeState }> {
    await this.chatEnsureReady();

    return this.enqueueEngineChat(parentId, userText, options);
  }

  private enqueueEngineChat(
    parentId: number,
    userText: string,
    options: WllamaChatFromNodeOptions
  ): Promise<{ nodeId: number; assistantText: string; state: WllamaTreeState }> {
    if (options.abortSignal?.aborted) {
      return Promise.reject(new WllamaAbortError());
    }

    const maxPending = this.getEngineChatQueueMaxPending();
    if (this.engineChatQueue.length >= maxPending) {
      this.traceEngineChat(
        `reject queue_overloaded pending=${this.engineChatQueue.length} max=${maxPending} parent=${parentId}`,
        'warn'
      );
      return Promise.reject(
        new WllamaError(
          `engine chat queue is full (pending=${this.engineChatQueue.length}, max=${maxPending})`,
          'queue_overloaded'
        )
      );
    }

    return new Promise((resolve, reject) => {
      const nAheadAtEnqueue = this.engineChatQueue.length;
      const pMaxMs = this.getEngineChatServiceUpperBoundMs();
      const baselineWorkMs = nAheadAtEnqueue * pMaxMs;
      const waitBudgetMs = baselineWorkMs + pMaxMs;
      const req: EngineChatRequest = {
        id: this.nextEngineChatRequestId++,
        parentId,
        userText,
        options,
        enqueuedAt: Date.now(),
        nAheadAtEnqueue,
        baselineWorkMs,
        waitBudgetMs,
        resolve,
        reject,
      };

      const abortSignal = options.abortSignal;
      if (abortSignal) {
        const onAbort = () => {
          const idx = this.engineChatQueue.findIndex((x) => x.id === req.id);
          if (idx >= 0) {
            this.engineChatQueue.splice(idx, 1);
            req.reject(new WllamaAbortError());
          }
        };
        req.cleanupAbort = () => abortSignal.removeEventListener('abort', onAbort);
        abortSignal.addEventListener('abort', onAbort, { once: true });
      }

      this.engineChatQueue.push(req);
      this.traceEngineChat(
        `enqueue req=${req.id} parent=${parentId} pending=${this.engineChatQueue.length} nAhead=${req.nAheadAtEnqueue} budgetMs=${req.waitBudgetMs}`
      );
      void this.processEngineChatQueue();
    });
  }

  private async processEngineChatQueue(): Promise<void> {
    if (this.engineChatQueueRunning) {
      return;
    }

    this.engineChatQueueRunning = true;
    this.traceEngineChat(`queue loop start pending=${this.engineChatQueue.length}`);
    try {
      while (this.engineChatQueue.length > 0) {
        const req = await this.pickNextEngineChatRequest();
        if (!req) {
          break;
        }

        if (req.options.abortSignal?.aborted) {
          req.cleanupAbort?.();
          this.traceEngineChat(`drop aborted req=${req.id} before execute`);
          req.reject(new WllamaAbortError());
          continue;
        }

        try {
          req.processingStartedAt = Date.now();
          this.traceEngineChat(
            `start req=${req.id} waitMs=${req.processingStartedAt - req.enqueuedAt} pendingAfterPick=${this.engineChatQueue.length}`
          );
          const result = await this.chatFromNodeDirect(
            req.parentId,
            req.userText,
            ({ ...req.options, __engineReqId: req.id } as unknown as WllamaChatFromNodeOptions)
          );
          const doneAt = Date.now();
          this.traceEngineChat(
            `finish req=${req.id} serviceMs=${doneAt - (req.processingStartedAt ?? doneAt)} totalMs=${doneAt - req.enqueuedAt}`
          );
          req.resolve(result);
        } catch (err) {
          const failAt = Date.now();
          const msg = err instanceof Error ? err.message : String(err);
          this.traceEngineChat(
            `fail req=${req.id} serviceMs=${req.processingStartedAt ? failAt - req.processingStartedAt : -1} totalMs=${failAt - req.enqueuedAt} err=${msg}`,
            'warn'
          );
          req.reject(err);
        } finally {
          req.cleanupAbort?.();
        }
      }
    } finally {
      this.engineChatQueueRunning = false;
      this.traceEngineChat(`queue loop end pending=${this.engineChatQueue.length}`);
      if (this.engineChatQueue.length > 0) {
        void this.processEngineChatQueue();
      }
    }
  }

  private async pickNextEngineChatRequest(): Promise<EngineChatRequest | undefined> {
    if (this.engineChatQueue.length === 0) {
      return undefined;
    }
    if (this.engineChatQueue.length === 1) {
      return this.engineChatQueue.shift();
    }

    // RFG scheduling:
    // 1) If overdue set is non-empty, pick from overdue set.
    // 2) Otherwise pick by reuse-first lexicographic order.
    let state: WllamaTreeState | null = null;
    try {
      state = await this.treeGetState();
    } catch {
      state = null;
    }

    interface CandidateMetrics {
      req: EngineChatRequest;
      idx: number;
      waitMs: number;
      overdue: boolean;
      reuseDepth: number;
      newPrefillWork: number;
    }

    const now = Date.now();
    const candidates: CandidateMetrics[] = [];

    for (let i = 0; i < this.engineChatQueue.length; i++) {
      const req = this.engineChatQueue[i];
      const waitMs = Math.max(0, now - req.enqueuedAt);

      let reuseDepth = 0;
      let newPrefillWork = 0;
      if (state) {
        reuseDepth = this.sharedPrefixDepthInTree(state, state.activeNodeId, req.parentId);
        const promptDepth = this.pathNodeIdsInTree(state, req.parentId).length;
        newPrefillWork = Math.max(0, promptDepth - reuseDepth);
      }

      candidates.push({
        req,
        idx: i,
        waitMs,
        overdue: waitMs >= req.waitBudgetMs,
        reuseDepth,
        newPrefillWork,
      });
    }

    const overdueCandidates = candidates.filter((x) => x.overdue);
    const activeSet = overdueCandidates.length > 0 ? overdueCandidates : candidates;
    const useOverdueOrder = overdueCandidates.length > 0;

    let best = activeSet[0];
    for (let i = 1; i < activeSet.length; i++) {
      const cur = activeSet[i];
      if (this.isBetterEngineChatCandidate(cur, best, useOverdueOrder)) {
        best = cur;
      }
    }

    const [picked] = this.engineChatQueue.splice(best.idx, 1);
    return picked;
  }

  private getEngineChatServiceUpperBoundMs(): number {
    const configured = this.config.engineChatServiceUpperBoundMs;
    if (typeof configured !== 'number' || !Number.isFinite(configured) || configured <= 0) {
      return DEFAULT_ENGINE_CHAT_SERVICE_UPPER_BOUND_MS;
    }
    return Math.floor(configured);
  }

  private getEngineChatQueueMaxPending(): number {
    const configured = this.config.engineChatQueueMaxPending;
    if (typeof configured !== 'number' || !Number.isFinite(configured) || configured <= 0) {
      return DEFAULT_ENGINE_CHAT_QUEUE_MAX_PENDING;
    }
    return Math.floor(configured);
  }

  private isBetterEngineChatCandidate(
    a: {
      req: EngineChatRequest;
      waitMs: number;
      reuseDepth: number;
      newPrefillWork: number;
    },
    b: {
      req: EngineChatRequest;
      waitMs: number;
      reuseDepth: number;
      newPrefillWork: number;
    },
    useOverdueOrder: boolean
  ): boolean {
    if (useOverdueOrder) {
      // overdue order: (wait, reuse, -newPrefillWork, enqueuedAt, id)
      if (a.waitMs !== b.waitMs) return a.waitMs > b.waitMs;
      if (a.reuseDepth !== b.reuseDepth) return a.reuseDepth > b.reuseDepth;
      if (a.newPrefillWork !== b.newPrefillWork) return a.newPrefillWork < b.newPrefillWork;
      if (a.req.enqueuedAt !== b.req.enqueuedAt) return a.req.enqueuedAt < b.req.enqueuedAt;
      return a.req.id < b.req.id;
    }

    // non-overdue order: (reuse, -newPrefillWork, wait, enqueuedAt, id)
    if (a.reuseDepth !== b.reuseDepth) return a.reuseDepth > b.reuseDepth;
    if (a.newPrefillWork !== b.newPrefillWork) return a.newPrefillWork < b.newPrefillWork;
    if (a.waitMs !== b.waitMs) return a.waitMs > b.waitMs;
    if (a.req.enqueuedAt !== b.req.enqueuedAt) return a.req.enqueuedAt < b.req.enqueuedAt;
    return a.req.id < b.req.id;
  }

  private sharedPrefixDepthInTree(state: WllamaTreeState, aNodeId: number, bNodeId: number): number {
    const aPath = this.pathNodeIdsInTree(state, aNodeId);
    const bPath = this.pathNodeIdsInTree(state, bNodeId);
    const n = Math.min(aPath.length, bPath.length);
    let depth = 0;
    for (let i = 0; i < n; i++) {
      if (aPath[i] !== bPath[i]) {
        break;
      }
      depth += 1;
    }
    return depth;
  }

  private pathNodeIdsInTree(state: WllamaTreeState, nodeId: number): number[] {
    const ids: number[] = [];
    let cur = state.nodes.get(nodeId);
    while (cur) {
      ids.push(cur.id);
      if (cur.parentId === null) {
        break;
      }
      cur = state.nodes.get(cur.parentId);
    }
    ids.reverse();
    return ids;
  }

  private async chatFromNodeDirect(
    parentId: number,
    userText: string,
    options: WllamaChatFromNodeOptions = {}
  ): Promise<{ nodeId: number; assistantText: string; state: WllamaTreeState }> {
    if (options.abortSignal?.aborted) {
      throw new WllamaAbortError();
    }

    const reqId = (options as unknown as { __engineReqId?: number }).__engineReqId;
    const traceKey = `req=${reqId ?? 'na'} parent=${parentId}`;
    const opStartedAt = Date.now();
    let stage: 'chat_start' | 'create_completion' | 'streaming' | 'chat_finish' | 'rollback' = 'chat_start';
    let chunkCount = 0;
    let lastProgressAt = opStartedAt;
    let firstTokenAt = 0;
    const heartbeatId = this.isEngineChatTraceEnabled()
      ? setInterval(() => {
        const now = Date.now();
        this.traceEngineChat(
          `${traceKey} heartbeat stage=${stage} elapsedMs=${now - opStartedAt} chunks=${chunkCount} sinceProgressMs=${now - lastProgressAt}`,
          now - lastProgressAt > 10000 ? 'warn' : 'debug'
        );
      }, 5000)
      : null;

    const started = await this.treeChatStart(parentId, userText);
    const t0 = Date.now();
    let assistantText = '';

    try {
      stage = 'create_completion';
      const prompt = started.formattedPrompt || (await this.formatChat(started.messages, true));
      const stream = await this.createCompletion(prompt, {
        ...options,
        stream: true,
      });

      stage = 'streaming';
      for await (const chunk of stream) {
        if (options.abortSignal?.aborted) {
          this.traceEngineChat(`${traceKey} abort observed in streaming loop`);
          break;
        }
        chunkCount += 1;
        lastProgressAt = Date.now();
        if (firstTokenAt === 0) {
          firstTokenAt = lastProgressAt;
          this.traceEngineChat(`${traceKey} first-token ttftMs=${firstTokenAt - t0}`);
        }
        const piece = new TextDecoder().decode(chunk.piece, { stream: true });
        assistantText = chunk.currentText;
        options.onChunk?.(piece, assistantText);
      }

      stage = 'chat_finish';
      const state = await this.treeChatFinish(
        started.nodeId,
        assistantText,
        Date.now() - t0,
        false
      );
      this.traceEngineChat(
        `${traceKey} done node=${started.nodeId} totalMs=${Date.now() - opStartedAt} chunks=${chunkCount}`
      );
      return { nodeId: started.nodeId, assistantText, state };
    } catch (err) {
      try {
        stage = 'rollback';
        await this.treeChatFinish(started.nodeId, '', Date.now() - t0, true);
      } catch {
        // Ignore rollback failure and rethrow original inference error.
      }
      const msg = err instanceof Error ? err.message : String(err);
      this.traceEngineChat(
        `${traceKey} error stage=${stage} totalMs=${Date.now() - opStartedAt} chunks=${chunkCount} err=${msg}`,
        'warn'
      );
      throw err;
    } finally {
      if (heartbeatId) {
        clearInterval(heartbeatId);
      }
    }
  }

  private resolveNodeIdByHistory(state: WllamaTreeState, history: WllamaChatMessage[]): number {
    const turns = this.historyToTurns(history);
    let currentId = state.rootId;

    for (const turn of turns) {
      const current = state.nodes.get(currentId);
      if (!current) {
        throw new WllamaError(`Node ${currentId} not found while resolving history`);
      }

      const nextId = current.childIds.find((childId) => {
        const child = state.nodes.get(childId);
        return !!child
          && child.turn.user === turn.user
          && child.turn.assistant === turn.assistant;
      });

      if (nextId === undefined) {
        throw new WllamaError('History does not map to an existing conversation path');
      }
      currentId = nextId;
    }

    return currentId;
  }

  private historyToTurns(history: WllamaChatMessage[]): Array<{ user: string; assistant: string }> {
    const turns: Array<{ user: string; assistant: string }> = [];
    let pendingUser = '';

    for (const msg of history) {
      if (msg.role === 'system') {
        continue;
      }
      if (msg.role === 'user') {
        if (pendingUser) {
          throw new WllamaError('Invalid history: consecutive user messages are not supported');
        }
        pendingUser = msg.content;
        continue;
      }
      if (msg.role === 'assistant') {
        if (!pendingUser) {
          throw new WllamaError('Invalid history: assistant message without preceding user message');
        }
        turns.push({ user: pendingUser, assistant: msg.content });
        pendingUser = '';
      }
    }

    if (pendingUser) {
      throw new WllamaError('Invalid history: trailing user message is not allowed in base history');
    }

    return turns;
  }

  private async treeGetState(): Promise<WllamaTreeState> {
    this.checkModelLoaded();
    const result = await this.proxy.wllamaAction<GlueMsgTreeStateRes>('chat_state', {
      _name: 'trst_req',
    });
    if (!result.success) {
      throw new WllamaError(`treeGetState failed: ${result.message}`);
    }
    return this.mapTreeState(result);
  }

  private async treeSwitch(nodeId: number): Promise<WllamaTreeState> {
    this.checkModelLoaded();
    const result = await this.proxy.wllamaAction<GlueMsgTreeSwitchRes>('chat_set_active', {
      _name: 'trsw_req',
      node_id: nodeId,
    });
    if (!result.success) {
      throw new WllamaError(`treeSwitch(${nodeId}) failed: ${result.message}`);
    }
    const state = await this.treeGetState();
    this.nCachedTokens = state.nodes.get(state.activeNodeId)?.prefixTokenCount ?? 0;
    return state;
  }

  private async treeChatStart(parentId: number, userText: string): Promise<{ nodeId: number; messages: WllamaChatMessage[]; formattedPrompt: string; state: WllamaTreeState }> {
    this.checkModelLoaded();
    const result = await this.proxy.wllamaAction<GlueMsgTreeChatStartRes>('chat_start', {
      _name: 'tchs_req',
      parent_id: parentId,
      user_text: userText,
    });
    if (!result.success) {
      throw new WllamaError(`treeChatStart(${parentId}) failed: ${result.message}`);
    }

    const messages: WllamaChatMessage[] = [];
    const len = Math.min(result.roles.length, result.contents.length);
    for (let i = 0; i < len; i++) {
      const role = result.roles[i];
      if (role === 'system' || role === 'user' || role === 'assistant') {
        messages.push({ role, content: result.contents[i] });
      }
    }

    const state = await this.treeGetState();
    const parentNode = state.nodes.get(parentId);
    this.nCachedTokens = parentNode?.prefixTokenCount ?? 0;
    return {
      nodeId: result.node_id,
      messages,
      formattedPrompt: result.formatted_chat ?? '',
      state,
    };
  }

  private async treeChatFinish(
    nodeId: number,
    assistantText: string,
    generationTimeMs: number,
    abortedOrError: boolean
  ): Promise<WllamaTreeState> {
    this.checkModelLoaded();
    const result = await this.proxy.wllamaAction<GlueMsgTreeChatFinishRes>('chat_finish', {
      _name: 'tchf_req',
      node_id: nodeId,
      assistant_text: assistantText,
      generation_time_ms: generationTimeMs,
      aborted_or_error: abortedOrError,
    });
    if (!result.success) {
      throw new WllamaError(`treeChatFinish(${nodeId}) failed: ${result.message}`);
    }

    const state = await this.treeGetState();
    this.nCachedTokens = state.nodes.get(state.activeNodeId)?.prefixTokenCount ?? 0;
    return state;
  }

  private async treeDelete(nodeId: number): Promise<WllamaTreeState> {
    this.checkModelLoaded();
    const result = await this.proxy.wllamaAction<GlueMsgTreeDeleteRes>('chat_delete', {
      _name: 'trde_req',
      node_id: nodeId,
    });
    if (!result.success) {
      throw new WllamaError(`treeDelete(${nodeId}) failed: ${result.message}`);
    }
    return this.treeGetState();
  }

  private async treeReset(): Promise<WllamaTreeState> {
    this.checkModelLoaded();
    const result = await this.proxy.wllamaAction<GlueMsgTreeResetRes>('chat_reset', {
      _name: 'trrs_req',
    });
    if (!result.success) {
      throw new WllamaError(`treeReset failed: ${result.message}`);
    }
    this.nCachedTokens = 0;
    return this.treeGetState();
  }

  /**
   * Save session to file (virtual file system)
   * TODO: add ability to download the file
   * @param filePath
   * @returns List of tokens saved to the file
   */
  // async sessionSave(filePath: string): Promise<{ tokens: number[] }> {
  //   this.checkModelLoaded();
  //   const result = await this.proxy.wllamaAction('session_save', {
  //     session_path: filePath,
  //   });
  //   return result;
  // }

  /**
   * Load session from file (virtual file system)
   * TODO: add ability to download the file
   * @param filePath
   */
  // async sessionLoad(filePath: string): Promise<void> {
  //   this.checkModelLoaded();
  //   const result = await this.proxy.wllamaAction('session_load', {
  //     session_path: filePath,
  //   });
  //   if (result.error) {
  //     throw new WllamaError(result.error);
  //   } else if (!result.success) {
  //     throw new WllamaError('sessionLoad unknown error');
  //   }
  //   const cachedTokens = await this.getCachedTokens();
  //   this.nCachedTokens = cachedTokens.length;
  // }

  /**
   * Apply chat template to a list of messages
   *
   * @param messages list of messages
   * @param addAssistant whether to add assistant prompt at the end
   * @param template (optional) custom template, see llama-server --chat-template argument for more details
   * @returns formatted chat
   */
  async formatChat(
    messages: WllamaChatMessage[],
    addAssistant: boolean,
    template?: string
  ): Promise<string> {
    this.checkModelLoaded();
    const roles = messages.map((m) => m.role);
    const contents = messages.map((m) => m.content);
    const result = await this.proxy.wllamaAction<GlueMsgChatFormatRes>(
      'chat_format',
      {
        _name: 'cfmt_req',
        roles,
        contents,
        tmpl: template,
        add_ass: addAssistant,
      }
    );
    if (!result.success) {
      throw new WllamaError('formatChat unknown error');
    }
    return result.formatted_chat;
  }

  /**
   * Set options for underlaying llama_context
   */
  async setOptions(opt: ContextOptions): Promise<void> {
    this.checkModelLoaded();
    await this.proxy.wllamaAction<GlueMsgSetOptionsRes>('set_options', {
      _name: 'opti_req',
      ...opt,
    });
    this.useEmbeddings = opt.embeddings;
  }

  /**
   * Unload the model and free all memory.
   *
   * Note: This function will NOT crash if model is not yet loaded
   */
  async exit(): Promise<void> {
    await this.proxy?.wllamaExit();
    this.proxy = null as any;
  }

  /**
   * get debug info
   */
  async _getDebugInfo(): Promise<any> {
    this.checkModelLoaded();
    const nativeDebug = await this.proxy.wllamaDebug();
    const now = Date.now();
    const pending = this.engineChatQueue.map((req) => ({
      id: req.id,
      parentId: req.parentId,
      enqueuedForMs: now - req.enqueuedAt,
      processingForMs: req.processingStartedAt ? now - req.processingStartedAt : 0,
      nAheadAtEnqueue: req.nAheadAtEnqueue,
      waitBudgetMs: req.waitBudgetMs,
    }));

    if (nativeDebug && typeof nativeDebug === 'object') {
      return {
        ...(nativeDebug as Record<string, unknown>),
        engineChat: {
          running: this.engineChatQueueRunning,
          pendingCount: this.engineChatQueue.length,
          pending,
        },
      };
    }

    return {
      nativeDebug,
      engineChat: {
        running: this.engineChatQueueRunning,
        pendingCount: this.engineChatQueue.length,
        pending,
      },
    };
  }

  /**
   * Get llama.cpp performance counters for the current context.
   */
  async getPerfContext(): Promise<PerfContextData> {
    this.checkModelLoaded();
    return await this.proxy.wllamaAction<GlueMsgPerfContextRes>(
      'perf_context',
      {
        _name: 'pctx_req',
      }
    );
  }

  /**
   * Reset llama.cpp performance counters for the current context.
   */
  async resetPerfContext(): Promise<{ success: boolean }> {
    this.checkModelLoaded();
    return await this.proxy.wllamaAction<GlueMsgPerfResetRes>('perf_reset', {
      _name: 'prst_req',
    });
  }

  /**
   * benchmark function, only used internally
   */
  async _testBenchmark(
    type: 'tg' | 'pp',
    nSamples: number
  ): Promise<{ t_ms: number }> {
    this.checkModelLoaded();
    return await this.proxy.wllamaAction<GlueMsgTestBenchmarkRes>(
      'test_benchmark',
      {
        _name: 'tben_req',
        type,
        n_samples: nSamples,
      }
    );
  }

  /**
   * perplexity function, only used internally
   */
  async _testPerplexity(tokens: number[]): Promise<{ ppl: number }> {
    this.checkModelLoaded();
    return await this.proxy.wllamaAction<GlueMsgTestPerplexityRes>(
      'test_perplexity',
      {
        _name: 'tper_req',
        tokens,
      }
    );
  }

  ///// Prompt cache utils /////
  private async getCachedTokens(): Promise<number[]> {
    this.checkModelLoaded();
    const result = await this.proxy.wllamaAction<GlueMsgStatusRes>(
      'current_status',
      {
        _name: 'stat_req',
      }
    );
    return result.tokens;
  }

  /**
   * Compare the input sequence and cachedToken, then return the part that is not in cache.
   * This function also remove mismatch part in cache (via kvRemove)
   */
  private async computeNonCachedTokens(seq: number[]): Promise<number[]> {
    const cachedTokens = await this.getCachedTokens();
    let nKeep = 0;
    for (; nKeep < Math.min(cachedTokens.length, seq.length); nKeep++) {
      if (cachedTokens[nKeep] !== seq[nKeep]) {
        break;
      }
    }
    this.logger().debug(`Cache nKeep=${nKeep}`);
    try {
      await this.kvRemove(nKeep, -1);
      return seq.slice(nKeep, seq.length);
    } catch (e) {
      this.logger().warn('Failed to rollback KV cache, clearing it instead');
      await this.kvClear();
      return seq;
    }
  }

  // TODO: add current_status
}
