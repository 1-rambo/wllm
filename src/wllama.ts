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
  GlueMsgTreeChatCheckpointRes,
  GlueMsgTreeChatFinishRes,
  GlueMsgTreeCacheHintRes,
  GlueMsgTreeChatResumeRes,
  GlueMsgTreeChatStartHistRes,
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
const DEFAULT_ENGINE_CHAT_SLICE_TOKEN_BUDGET = 128;
const DEFAULT_ENGINE_CHAT_EVICTION_HINT_TOP_K = 4;
const DEFAULT_ENGINE_CHAT_COST_PREFILL_PER_TOKEN_MS = 3;
const DEFAULT_ENGINE_CHAT_COST_DECODE_PER_TOKEN_MS = 4;
const DEFAULT_ENGINE_CHAT_COST_POST_MS = 2;
const DEFAULT_ENGINE_CHAT_PREFILL_SLICE_MAX_MS = 5000;
const DEFAULT_ENGINE_CHAT_COST_RESTORE_L1_PER_TOKEN_MS = 0.02;
const DEFAULT_ENGINE_CHAT_COST_RESTORE_L2_PER_TOKEN_MS = 0.5;
const DEFAULT_ENGINE_CHAT_COST_RESTORE_L3_PER_TOKEN_MS = 2;
const DEFAULT_ENGINE_CHAT_COST_REBUILD_PER_TOKEN_MS = 4;
const DEFAULT_ENGINE_CHAT_COST_PARENT_RECOVER_MS = 1;
const DEFAULT_ENGINE_CHAT_COST_WARMUP_REQUESTS = 20;
const DEFAULT_ENGINE_CHAT_COST_SAMPLE_WINDOW = 256;

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
   * Maximum token budget per serve slice for long generation requests.
   *
  * Default: 128
   */
  engineChatSliceTokenBudget?: number;
  /**
   * Number of queued parent nodes sent to native cache-hint path.
   *
   * Default: 4
   */
  engineChatEvictionHintTopK?: number;
  /**
   * Estimated prefill cost per prompt token (ms).
   *
   * Default: 3
   */
  engineChatPrefillCostPerTokenMs?: number;
  /**
   * Legacy alias of `engineChatPrefillCostPerTokenMs`.
   */
  engineChatPrefillCostPerUnitMs?: number;
  /**
   * Estimated decode cost per generated token (ms).
   *
   * Default: 4
   */
  engineChatDecodeCostPerTokenMs?: number;
  /**
   * Estimated post-processing cost (ms).
   *
   * Default: 2
   */
  engineChatPostCostMs?: number;
  /**
   * Maximum wall time budget of one prefill round for very long prompts (ms).
   *
   * Default: 5000
   */
  engineChatPrefillSliceMaxMs?: number;
  /**
   * Estimated L1 restore cost per token (ms).
   */
  engineChatRestoreL1CostPerTokenMs?: number;
  /**
   * Estimated L2 restore/replay cost per token (ms).
   */
  engineChatRestoreL2CostPerTokenMs?: number;
  /**
   * Estimated L3 restore/readback cost per token (ms).
   */
  engineChatRestoreL3CostPerTokenMs?: number;
  /**
   * Estimated rebuild cost per token (ms).
   */
  engineChatRebuildCostPerTokenMs?: number;
  /**
   * Legacy alias of `engineChatRestoreL2CostPerTokenMs`.
   */
  engineChatCacheMoveCostPerUnitMs?: number;
  /**
   * Legacy alias of `engineChatRebuildCostPerTokenMs`.
   */
  engineChatRebuildCostPerUnitMs?: number;
  /**
   * Estimated parent-recover fixed overhead cost (ms).
   *
   * Default: 1
   */
  engineChatParentRecoverCostMs?: number;
  /**
   * Number of completed requests recorded before enabling time estimation.
   * During warmup, queue records observations only and uses conservative
   * upper-bound service time for budgeting.
   *
   * Default: 20
   */
  engineChatCostWarmupRequests?: number;
  /**
   * Max number of recent observations retained for online fitting.
   *
   * Default: 256
   */
  engineChatCostSampleWindow?: number;
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
  cacheTierLevel: number;
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

type EngineChatRestoreSource = 'none' | 'l1' | 'l2' | 'l3' | 'rebuild';

interface EngineChatPreparedPrompt {
  prompt: string;
  promptTokens: number[];
  promptTokenCount: number;
  parentNodeId?: number | undefined;
  parentPrefixTokens: number;
  restoreSource: EngineChatRestoreSource;
  restoredPrefixTokens: number;
  rebuiltPrefixTokens: number;
  promptTailTokens: number;
  estimatedParentRecoverMs: number;
}

interface EngineChatRuntime {
  nodeId: number;
  stage: 'prefill' | 'decode';
  prompt: string;
  promptTokens: number[];
  remainingPromptTokens: number[];
  samplingHistoryTokens: number[];
  assistantText: string;
  generatedTokenIds: number[];
  generatedTokensLimit: number;
  generatedTokensSoFar: number;
  startedAt: number;
  firstTokenAt?: number;
  cacheLoadMs: number;
  rebuildMs: number;
  parentRecoverMs: number;
  startOverheadMs: number;
  accumulatedPrefillMs: number;
  accumulatedDecodeMs: number;
  restoredPrefixTokens: number;
  rebuiltPrefixTokens: number;
  restoreSource: EngineChatRestoreSource;
  promptTokenCount: number;
  promptTailTokenCount: number;
  resumedPrefixTokenCount: number;
}

interface EngineChatRequest {
  id: number;
  parentId?: number | undefined;
  baseHistory?: WllamaChatMessage[] | undefined;
  userText: string;
  options: WllamaChatFromNodeOptions;
  queueType: 'normal' | 'overdue';
  enqueuedAt: number;
  nAheadAtEnqueue: number;
  estimatedServiceMs: number;
  estimatedPrefillMs: number;
  estimatedDecodeMs: number;
  estimatedPostMs: number;
  estimatedCacheMoveMs: number;
  estimatedRebuildMs: number;
  estimatedParentRecoverMs: number;
  estimatedPromptTokens: number;
  estimatedPromptTailTokens: number;
  estimatedRestoredPrefixTokens: number;
  estimatedRebuiltPrefixTokens: number;
  estimatedRestoreSource: EngineChatRestoreSource;
  targetPredictTokens: number;
  generatedTokens: number;
  sliceCount: number;
  baselineWorkMs: number;
  waitBudgetMs: number;
  processingStartedAt?: number | undefined;
  preparedPrompt?: EngineChatPreparedPrompt | undefined;
  runtime?: EngineChatRuntime | undefined;
  resolve: (value: { nodeId: number; assistantText: string; state: WllamaTreeState }) => void;
  reject: (reason?: unknown) => void;
  cleanupAbort?: (() => void) | undefined;
}

interface EngineChatDetailedResult {
  text: string;
  generatedTokens: number;
  stoppedByEog: boolean;
  stoppedByStopToken: boolean;
  aborted: boolean;
  prefillMs: number;
  decodeMs: number;
}

interface EngineChatCostObservation {
  promptTailTokens: number;
  decodeTokens: number;
  restoredPrefixTokens: number;
  restoredFromTier: 0 | 1 | 2 | 3;
  rebuiltPrefixTokens: number;
  prefillMs: number;
  decodeMs: number;
  postMs: number;
  cacheLoadMs: number;
  rebuildMs: number;
  parentRecoverMs: number;
  totalServiceMs: number;
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
  private engineChatNormalQueue: EngineChatRequest[] = [];
  private engineChatOverdueQueue: EngineChatRequest[] = [];
  private engineChatQueueRunning: boolean = false;
  private engineChatCostObservations: EngineChatCostObservation[] = [];
  private engineChatCostObservedCount: number = 0;
  private engineChatLearnedPrefillCostPerTokenMs?: number;
  private engineChatLearnedDecodeCostPerTokenMs?: number;
  private engineChatLearnedPostCostMs?: number;
  private engineChatLearnedRestoreL1CostPerTokenMs?: number;
  private engineChatLearnedRestoreL2CostPerTokenMs?: number;
  private engineChatLearnedRestoreL3CostPerTokenMs?: number;
  private engineChatLearnedRebuildCostPerTokenMs?: number;
  private engineChatLearnedParentRecoverCostMs?: number;

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
    const detail = await this.createCompletionDetailed(prompt, options);
    return detail.text;
  }

  private async createCompletionDetailed(
    prompt: string,
    options: ChatCompletionOptions
  ): Promise<EngineChatDetailedResult> {
    this.checkModelLoaded();
    const stagePrefillWallStart = Date.now();
    this.samplingConfig = options.sampling ?? {};
    await this.samplingInit(this.samplingConfig);
    await this.resetPerfContext();
    const stopTokens = new Set(options.stopTokens ?? []);

    let tokens = await this.tokenize(prompt, true);
    if (this.addBosToken && tokens[0] !== this.bosToken) {
      tokens.unshift(this.bosToken);
    }

    if (options.useCache) {
      tokens = await this.computeNonCachedTokens(tokens);
    } else {
      await this.kvClear();
    }

    await this.prefillPromptWithTimeSlices(tokens, options.abortSignal);

    let outBuf = new Uint8Array();
    let generatedTokens = 0;
    let aborted = false;
    let stoppedByEog = false;
    let stoppedByStopToken = false;

    const abortSignalFn = () => {
      aborted = true;
    };

    const stageDecodeWallStart = Date.now();

    const maxPredict = options.nPredict ?? Infinity;
    for (let i = 0; i < maxPredict; i++) {
      const sampled = await this.samplingSample();
      if (this.isTokenEOG(sampled.token)) {
        stoppedByEog = true;
        break;
      }
      if (stopTokens.has(sampled.token)) {
        stoppedByStopToken = true;
        break;
      }

      // @ts-ignore Type 'Uint8Array<ArrayBufferLike>' is not assignable to type 'Uint8Array<ArrayBuffer>'
      outBuf = joinBuffers([outBuf, sampled.piece]);
      generatedTokens += 1;

      if (options.onNewToken) {
        options.onNewToken(sampled.token, sampled.piece, bufToText(outBuf), {
          abortSignal: abortSignalFn,
        });
      }
      if (aborted || options.abortSignal?.aborted) {
        aborted = true;
        break;
      }

      await this.samplingAccept([sampled.token]);
      await this.decode([sampled.token], {});
    }

    const perf = await this.getPerfContext();
    const measuredPrefillMs =
      Number.isFinite(perf?.t_p_eval_ms) && (perf?.t_p_eval_ms ?? 0) > 0
        ? perf.t_p_eval_ms
        : Math.max(0, stageDecodeWallStart - stagePrefillWallStart);
    const measuredDecodeMs =
      Number.isFinite(perf?.t_eval_ms) && (perf?.t_eval_ms ?? 0) > 0
        ? perf.t_eval_ms
        : Math.max(0, Date.now() - stageDecodeWallStart);

    return {
      text: bufToText(outBuf),
      generatedTokens,
      stoppedByEog,
      stoppedByStopToken,
      aborted,
      prefillMs: measuredPrefillMs,
      decodeMs: measuredDecodeMs,
    };
  }

  private async prefillPromptWithTimeSlices(
    tokens: number[],
    abortSignal?: AbortSignal
  ): Promise<void> {
    if (tokens.length === 0) {
      return;
    }

    const roundBudgetMs = this.getEngineChatPrefillSliceMaxMs();
    const chunkSize = Math.max(1, this.loadedContextInfo?.n_batch ?? 1);
    const completionOpts = abortSignal ? { abortSignal } : {};
    let cursor = 0;

    while (cursor < tokens.length) {
      const roundStartedAt = Date.now();
      while (cursor < tokens.length) {
        if (abortSignal?.aborted) {
          throw new WllamaAbortError();
        }

        const end = Math.min(tokens.length, cursor + chunkSize);
        const chunk = tokens.slice(cursor, end);
        await this.samplingAccept(chunk);

        if (this.isEncoderDecoderArchitecture()) {
          await this.encode(chunk, completionOpts);
        } else {
          await this.decode(chunk, completionOpts);
        }

        cursor = end;
        if (cursor < tokens.length && Date.now() - roundStartedAt >= roundBudgetMs) {
          break;
        }
      }
    }

    if (this.isEncoderDecoderArchitecture()) {
      await this.decode([this.getDecoderStartToken()], completionOpts);
    }
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
        cacheTierLevel: result.cache_tier_levels[i] ?? -1,
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
    await this.chatEnsureReady();
    return this.enqueueEngineChatByHistory(history, userText, options);
  }

  async chatSessionFinish(): Promise<WllamaTreeState> {
    return this.chatReset();
  }

  async chatFromNode(
    parentId: number,
    userText: string,
    options: WllamaChatFromNodeOptions = {}
  ): Promise<{ nodeId: number; assistantText: string; state: WllamaTreeState }> {
    // await this.chatEnsureReady();
    return this.enqueueEngineChat(parentId, userText, options);
  }

  private enqueueEngineChat(
    parentId: number,
    userText: string,
    options: WllamaChatFromNodeOptions
  ): Promise<{ nodeId: number; assistantText: string; state: WllamaTreeState }> {
    return this.enqueueEngineChatInternal(parentId, undefined, userText, options);
  }

  private enqueueEngineChatByHistory(
    history: WllamaChatMessage[],
    userText: string,
    options: WllamaChatFromNodeOptions
  ): Promise<{ nodeId: number; assistantText: string; state: WllamaTreeState }> {
    return this.enqueueEngineChatInternal(undefined, history, userText, options);
  }

  private async enqueueEngineChatInternal(
    parentId: number | undefined,
    baseHistory: WllamaChatMessage[] | undefined,
    userText: string,
    options: WllamaChatFromNodeOptions
  ): Promise<{ nodeId: number; assistantText: string; state: WllamaTreeState }> {
    await this.chatEnsureReady();
    if (options.abortSignal?.aborted) {
      throw new WllamaAbortError();
    }

    const maxPending = this.getEngineChatQueueMaxPending();
    const pendingNow = this.getEngineChatPendingCount();
    if (pendingNow >= maxPending) {
      this.traceEngineChat(
        `reject queue_overloaded pending=${pendingNow} max=${maxPending} parent=${parentId ?? 'history'}`,
        'warn'
      );
      throw new WllamaError(
        `engine chat queue is full (pending=${pendingNow}, max=${maxPending})`,
        'queue_overloaded'
      );
    }

    const state = await this.treeGetState();
    const preparedPrompt = await this.prepareEngineChatPrompt(state, parentId, baseHistory, userText);

    return new Promise((resolve, reject) => {
      const nAheadAtEnqueue = this.getEngineChatPendingCount();
      const req: EngineChatRequest = {
        id: this.nextEngineChatRequestId++,
        parentId,
        baseHistory: baseHistory ? [...baseHistory] : undefined,
        userText,
        options,
        queueType: 'normal',
        enqueuedAt: Date.now(),
        nAheadAtEnqueue,
        estimatedServiceMs: 0,
        estimatedPrefillMs: 0,
        estimatedDecodeMs: 0,
        estimatedPostMs: 0,
        estimatedCacheMoveMs: 0,
        estimatedRebuildMs: 0,
        estimatedParentRecoverMs: 0,
        estimatedPromptTokens: preparedPrompt.promptTokenCount,
        estimatedPromptTailTokens: preparedPrompt.promptTailTokens,
        estimatedRestoredPrefixTokens: preparedPrompt.restoredPrefixTokens,
        estimatedRebuiltPrefixTokens: preparedPrompt.rebuiltPrefixTokens,
        estimatedRestoreSource: preparedPrompt.restoreSource,
        targetPredictTokens: 0,
        generatedTokens: 0,
        sliceCount: 0,
        baselineWorkMs: 0,
        waitBudgetMs: 0,
        preparedPrompt,
        resolve,
        reject,
      };
      this.refreshEngineChatRequestEstimate(req, state, 0, nAheadAtEnqueue);

      const abortSignal = options.abortSignal;
      if (abortSignal) {
        const onAbort = () => {
          const removed = this.removeRequestFromQueues(req.id);
          if (removed) {
            req.reject(new WllamaAbortError());
          }
        };
        req.cleanupAbort = () => abortSignal.removeEventListener('abort', onAbort);
        abortSignal.addEventListener('abort', onAbort, { once: true });
      }

      this.engineChatNormalQueue.push(req);
      this.traceEngineChat(
        `enqueue req=${req.id} parent=${parentId ?? preparedPrompt.parentNodeId ?? 'history'} pending=${this.getEngineChatPendingCount()} nAhead=${req.nAheadAtEnqueue} budgetMs=${req.waitBudgetMs} estMs=${Math.round(req.estimatedServiceMs)}`
      );
      void this.processEngineChatQueue();
    });
  }

  private async processEngineChatQueue(): Promise<void> {
    if (this.engineChatQueueRunning) {
      return;
    }

    this.engineChatQueueRunning = true;
    this.traceEngineChat(`queue loop start pending=${this.getEngineChatPendingCount()}`);
    try {
      while (this.getEngineChatPendingCount() > 0) {
        this.promoteOverdueEngineChatRequests();
        await this.sendEngineChatCacheHint();

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
          if (!req.processingStartedAt) {
            req.processingStartedAt = Date.now();
          }
          this.traceEngineChat(
            `start req=${req.id} q=${req.queueType} waitMs=${Date.now() - req.enqueuedAt} pendingAfterPick=${this.getEngineChatPendingCount()} slices=${req.sliceCount}`
          );
          const slice = await this.executeEngineChatSlice(req);
          if (slice.done) {
            const doneAt = Date.now();
            const serviceMs = doneAt - (req.processingStartedAt ?? doneAt);
            this.recordEngineChatCostObservation(req, slice.phases, serviceMs);
            this.traceEngineChat(
              `finish req=${req.id} serviceMs=${serviceMs} totalMs=${doneAt - req.enqueuedAt} slices=${req.sliceCount} generatedTokens=${req.generatedTokens}`
            );
            req.cleanupAbort?.();
            req.resolve(slice.result);
          } else {
            this.requeueEngineChatRequest(req);
            this.traceEngineChat(
              `yield req=${req.id} nextStage=${req.runtime?.stage ?? 'na'} pending=${this.getEngineChatPendingCount()} slices=${req.sliceCount}`
            );
          }
        } catch (err) {
          const failAt = Date.now();
          const msg = err instanceof Error ? err.message : String(err);
          this.traceEngineChat(
            `fail req=${req.id} serviceMs=${req.processingStartedAt ? failAt - req.processingStartedAt : -1} totalMs=${failAt - req.enqueuedAt} err=${msg}`,
            'warn'
          );
          req.cleanupAbort?.();
          req.reject(err);
        }
      }
    } finally {
      this.engineChatQueueRunning = false;
      this.traceEngineChat(`queue loop end pending=${this.getEngineChatPendingCount()}`);
      if (this.getEngineChatPendingCount() > 0) {
        void this.processEngineChatQueue();
      }
    }
  }

  private getEngineChatPendingCount(): number {
    return this.engineChatNormalQueue.length + this.engineChatOverdueQueue.length;
  }

  private removeRequestFromQueues(reqId: number): boolean {
    let idx = this.engineChatNormalQueue.findIndex((x) => x.id === reqId);
    if (idx >= 0) {
      this.engineChatNormalQueue.splice(idx, 1);
      return true;
    }
    idx = this.engineChatOverdueQueue.findIndex((x) => x.id === reqId);
    if (idx >= 0) {
      this.engineChatOverdueQueue.splice(idx, 1);
      return true;
    }
    return false;
  }

  private promoteOverdueEngineChatRequests(): void {
    if (this.engineChatNormalQueue.length === 0) {
      return;
    }

    const now = Date.now();
    for (let i = this.engineChatNormalQueue.length - 1; i >= 0; i--) {
      const req = this.engineChatNormalQueue[i];
      const waitMs = Math.max(0, now - req.enqueuedAt);
      if (waitMs < req.waitBudgetMs) {
        continue;
      }
      this.engineChatNormalQueue.splice(i, 1);
      req.queueType = 'overdue';
      this.engineChatOverdueQueue.push(req);
      this.traceEngineChat(
        `promote-overdue req=${req.id} waitMs=${waitMs} budgetMs=${req.waitBudgetMs} overdueSize=${this.engineChatOverdueQueue.length}`
      );
    }
  }

  private async sendEngineChatCacheHint(): Promise<void> {
    const hotNodeIds = this.collectHotParentNodeIdsForHint();
    if (hotNodeIds.length === 0) {
      return;
    }
    const pending = this.getEngineChatPendingCount();
    const maxPending = Math.max(1, this.getEngineChatQueueMaxPending());
    const queuePressure = Math.max(0, Math.min(1, pending / maxPending));
    try {
      await this.treeCacheHint(hotNodeIds, queuePressure);
    } catch {
      // Non-critical best-effort hint path.
    }
  }

  private collectHotParentNodeIdsForHint(): number[] {
    const k = Math.max(1, this.getEngineChatEvictionHintTopK());
    const ids: number[] = [];
    const seen = new Set<number>();
    const appendFrom = (queue: EngineChatRequest[]) => {
      for (const req of queue) {
        if (ids.length >= k) {
          break;
        }
        const candidates = [req.runtime?.nodeId, req.parentId, req.preparedPrompt?.parentNodeId];
        for (const candidate of candidates) {
          if (ids.length >= k) {
            break;
          }
          if (typeof candidate !== 'number' || candidate < 0 || seen.has(candidate)) {
            continue;
          }
          seen.add(candidate);
          ids.push(candidate);
        }
      }
    };
    appendFrom(this.engineChatOverdueQueue);
    appendFrom(this.engineChatNormalQueue);
    return ids;
  }

  private async pickNextEngineChatRequest(): Promise<EngineChatRequest | undefined> {
    const pending = this.getEngineChatPendingCount();
    if (pending === 0) {
      return undefined;
    }

    let state: WllamaTreeState | null = null;
    try {
      state = await this.treeGetState();
    } catch {
      state = null;
    }

    interface CandidateMetrics {
      req: EngineChatRequest;
      idx: number;
      queueType: 'normal' | 'overdue';
      waitMs: number;
      estimatedServiceMs: number;
      blockedByPendingNodeId?: number | undefined;
      pendingDependencyDepth: number;
    }

    const buildCandidates = (
      queue: EngineChatRequest[],
      queueType: 'normal' | 'overdue'
    ): CandidateMetrics[] => {
      const now = Date.now();
      const candidates: CandidateMetrics[] = [];
      for (let i = 0; i < queue.length; i++) {
        const req = queue[i];
        const waitMs = Math.max(0, now - req.enqueuedAt);
        this.refreshEngineChatRequestEstimate(req, state, waitMs);
        const pendingDependency = this.findPendingDependencyForEngineChatRequest(req, state);
        candidates.push({
          req,
          idx: i,
          queueType,
          waitMs,
          estimatedServiceMs: req.estimatedServiceMs,
          blockedByPendingNodeId: pendingDependency?.nodeId,
          pendingDependencyDepth: pendingDependency?.depth ?? 0,
        });
      }
      return candidates;
    };

    const pickBest = (
      candidates: CandidateMetrics[],
      useOverdueOrder: boolean
    ): CandidateMetrics | undefined => {
      if (candidates.length === 0) {
        return undefined;
      }
      let best = candidates[0];
      for (let i = 1; i < candidates.length; i++) {
        const cur = candidates[i];
        if (this.isBetterEngineChatCandidate(cur, best, useOverdueOrder)) {
          best = cur;
        }
      }
      return best;
    };

    const overdueCandidates = buildCandidates(this.engineChatOverdueQueue, 'overdue');
    const normalCandidates = buildCandidates(this.engineChatNormalQueue, 'normal');
    const runnableOverdue = overdueCandidates.filter((candidate) => candidate.blockedByPendingNodeId === undefined);
    const runnableNormal = normalCandidates.filter((candidate) => candidate.blockedByPendingNodeId === undefined);
    const blockedOverdue = overdueCandidates.filter((candidate) => candidate.blockedByPendingNodeId !== undefined);
    const blockedNormal = normalCandidates.filter((candidate) => candidate.blockedByPendingNodeId !== undefined);

    const best =
      pickBest(runnableOverdue, true)
      ?? pickBest(runnableNormal, false)
      ?? pickBest(blockedOverdue, true)
      ?? pickBest(blockedNormal, false);

    if (!best) {
      return undefined;
    }

    if (best.blockedByPendingNodeId !== undefined) {
      this.traceEngineChat(
        `pick blocked req=${best.req.id} q=${best.queueType} blockedByPending=${best.blockedByPendingNodeId} depth=${best.pendingDependencyDepth}`,
        'warn'
      );
    }

    const targetQueue = best.queueType === 'overdue'
      ? this.engineChatOverdueQueue
      : this.engineChatNormalQueue;
    const [picked] = targetQueue.splice(best.idx, 1);
    if (picked) {
      picked.queueType = best.queueType;
    }
    return picked;
  }

  private findPendingDependencyForEngineChatRequest(
    req: EngineChatRequest,
    state: WllamaTreeState | null
  ): { nodeId: number; depth: number } | undefined {
    if (!state) {
      return undefined;
    }

    let cursorId: number | null | undefined;
    if (req.runtime) {
      const runtimeNode = state.nodes.get(req.runtime.nodeId);
      cursorId = runtimeNode?.parentId;
    } else {
      cursorId = req.preparedPrompt?.parentNodeId ?? req.parentId;
    }

    let depth = 0;
    while (typeof cursorId === 'number' && cursorId >= 0) {
      const node = state.nodes.get(cursorId);
      if (!node) {
        break;
      }
      depth += 1;
      if (node.status === 'pending') {
        return { nodeId: node.id, depth };
      }
      cursorId = node.parentId;
    }

    return undefined;
  }

  private async prepareEngineChatPrompt(
    state: WllamaTreeState,
    parentId: number | undefined,
    baseHistory: WllamaChatMessage[] | undefined,
    userText: string
  ): Promise<EngineChatPreparedPrompt> {
    let resolvedParentId = parentId;
    let messages: WllamaChatMessage[];
    if (typeof parentId === 'number' && parentId >= 0) {
      messages = this.buildPromptMessagesForNode(state, parentId, userText);
    } else {
      messages = [...(baseHistory ?? []), { role: 'user', content: userText }];
      if (Array.isArray(baseHistory) && baseHistory.length > 0) {
        try {
          resolvedParentId = this.resolveNodeIdByHistory(state, baseHistory);
        } catch {
          resolvedParentId = undefined;
        }
      }
    }

    const prompt = await this.formatChat(messages, true);
    const promptTokens = await this.tokenizePromptForEngineChat(prompt);
    const parentNode = typeof resolvedParentId === 'number'
      ? state.nodes.get(resolvedParentId)
      : undefined;
    const parentPrefixTokens =
      parentNode && resolvedParentId !== state.rootId
        ? Math.max(0, parentNode.prefixTokenCount)
        : 0;
    const restoreSource = this.classifyEngineChatRestoreSource(parentNode, parentPrefixTokens);
    const restoredPrefixTokens =
      restoreSource === 'l1' || restoreSource === 'l2' || restoreSource === 'l3'
        ? parentPrefixTokens
        : 0;
    const rebuiltPrefixTokens = restoreSource === 'rebuild' ? parentPrefixTokens : 0;

    return {
      prompt,
      promptTokens,
      promptTokenCount: promptTokens.length,
      parentNodeId: resolvedParentId,
      parentPrefixTokens,
      restoreSource,
      restoredPrefixTokens,
      rebuiltPrefixTokens,
      promptTailTokens: Math.max(0, promptTokens.length - parentPrefixTokens),
      estimatedParentRecoverMs: 0,
    };
  }

  private buildPromptMessagesForNode(
    state: WllamaTreeState,
    parentId: number,
    userText: string
  ): WllamaChatMessage[] {
    const messages: WllamaChatMessage[] = [];
    for (const nodeId of this.pathNodeIdsInTree(state, parentId)) {
      const node = state.nodes.get(nodeId);
      if (!node || node.id === state.rootId) {
        continue;
      }
      if (node.turn.user) {
        messages.push({ role: 'user', content: node.turn.user });
      }
      if (node.turn.assistant) {
        messages.push({ role: 'assistant', content: node.turn.assistant });
      }
    }
    messages.push({ role: 'user', content: userText });
    return messages;
  }

  private async tokenizePromptForEngineChat(prompt: string): Promise<number[]> {
    let tokens = await this.tokenize(prompt, true);
    if (this.addBosToken && tokens[0] !== this.bosToken) {
      tokens = [this.bosToken, ...tokens];
    }
    return tokens;
  }

  private classifyEngineChatRestoreSource(
    node: WllamaTreeNode | undefined,
    prefixTokens: number
  ): EngineChatRestoreSource {
    if (!node || prefixTokens <= 0) {
      return 'none';
    }
    if (node.cacheTierLevel === 1) {
      return 'l1';
    }
    if (node.cacheTierLevel === 2) {
      return 'l2';
    }
    if (node.cacheTierLevel === 3) {
      return 'l3';
    }
    return 'rebuild';
  }

  private refreshEngineChatPreparedPrompt(
    req: EngineChatRequest,
    state: WllamaTreeState | null
  ): EngineChatPreparedPrompt {
    const fallback = req.preparedPrompt!;
    const runtime = req.runtime;
    if (!state) {
      if (!runtime) {
        return fallback;
      }
      return {
        ...fallback,
        parentNodeId: runtime.nodeId,
        parentPrefixTokens: Math.max(0, runtime.resumedPrefixTokenCount),
        restoreSource: runtime.restoreSource,
        restoredPrefixTokens: runtime.restoreSource === 'l1' || runtime.restoreSource === 'l2' || runtime.restoreSource === 'l3'
          ? Math.max(0, runtime.resumedPrefixTokenCount)
          : 0,
        rebuiltPrefixTokens: runtime.restoreSource === 'rebuild'
          ? Math.max(0, runtime.resumedPrefixTokenCount)
          : 0,
        promptTailTokens: Math.max(0, runtime.remainingPromptTokens.length),
      };
    }

    const parentNodeId = runtime?.nodeId ?? fallback.parentNodeId;
    const parentNode = typeof parentNodeId === 'number' ? state.nodes.get(parentNodeId) : undefined;
    const parentPrefixTokens = runtime
      ? Math.max(0, runtime.resumedPrefixTokenCount)
      : (parentNode && parentNodeId !== state.rootId ? Math.max(0, parentNode.prefixTokenCount) : 0);
    const restoreSource = this.classifyEngineChatRestoreSource(parentNode, parentPrefixTokens);

    return {
      ...fallback,
      parentNodeId,
      parentPrefixTokens,
      restoreSource,
      restoredPrefixTokens:
        restoreSource === 'l1' || restoreSource === 'l2' || restoreSource === 'l3'
          ? parentPrefixTokens
          : 0,
      rebuiltPrefixTokens: restoreSource === 'rebuild' ? parentPrefixTokens : 0,
      promptTailTokens: runtime
        ? Math.max(0, runtime.remainingPromptTokens.length)
        : Math.max(0, fallback.promptTokenCount - parentPrefixTokens),
    };
  }

  private refreshEngineChatRequestEstimate(
    req: EngineChatRequest,
    state: WllamaTreeState | null,
    waitMs: number = 0,
    nAheadAtEnqueue?: number
  ): void {
    const prepared = this.refreshEngineChatPreparedPrompt(req, state);
    req.preparedPrompt = prepared;
    const est = this.estimateEngineChatServiceCost(prepared, req.options.nPredict, req.runtime, waitMs);
    req.estimatedServiceMs = est.totalMs;
    req.estimatedPrefillMs = est.prefillMs;
    req.estimatedDecodeMs = est.decodeMs;
    req.estimatedPostMs = est.postMs;
    req.estimatedCacheMoveMs = est.cacheMoveMs;
    req.estimatedRebuildMs = est.rebuildMs;
    req.estimatedParentRecoverMs = est.parentRecoverMs;
    req.estimatedPromptTokens = prepared.promptTokenCount;
    req.estimatedPromptTailTokens = prepared.promptTailTokens;
    req.estimatedRestoredPrefixTokens = prepared.restoredPrefixTokens;
    req.estimatedRebuiltPrefixTokens = prepared.rebuiltPrefixTokens;
    req.estimatedRestoreSource = prepared.restoreSource;
    req.targetPredictTokens = est.predictTokens;
    const nAhead = nAheadAtEnqueue ?? req.nAheadAtEnqueue;
    const pMaxMs = Math.max(this.getEngineChatServiceUpperBoundMs(), Math.ceil(est.totalMs));
    req.baselineWorkMs = nAhead * pMaxMs;
    req.waitBudgetMs = req.baselineWorkMs + pMaxMs;
  }

  private estimateEngineChatServiceCost(
    prepared: EngineChatPreparedPrompt,
    nPredict: number | undefined,
    runtime?: EngineChatRuntime,
    waitMs: number = 0
  ): {
      prefillMs: number;
      decodeMs: number;
      postMs: number;
      cacheMoveMs: number;
      rebuildMs: number;
      parentRecoverMs: number;
      totalMs: number;
      predictTokens: number;
    } {
    const warmupRequests = this.getEngineChatCostWarmupRequests();
    if (this.engineChatCostObservedCount < warmupRequests) {
      return {
        prefillMs: 0,
        decodeMs: 0,
        postMs: 0,
        cacheMoveMs: 0,
        rebuildMs: 0,
        parentRecoverMs: 0,
        totalMs: this.getEngineChatServiceUpperBoundMs(),
        predictTokens: runtime ? this.getEngineChatNextDecodeSliceTokens(runtime) : this.getEngineChatInitialDecodeSliceTokens(nPredict),
      };
    }

    const prefillPerToken = this.engineChatLearnedPrefillCostPerTokenMs
      ?? this.getEngineChatPrefillCostPerTokenMs();
    const decodePerToken = this.engineChatLearnedDecodeCostPerTokenMs
      ?? this.getEngineChatDecodeCostPerTokenMs();
    const postMs = this.engineChatLearnedPostCostMs ?? this.getEngineChatPostCostMs();
    const rebuildPerToken = this.engineChatLearnedRebuildCostPerTokenMs
      ?? this.getEngineChatRebuildCostPerTokenMs();
    const parentRecoverMs = this.engineChatLearnedParentRecoverCostMs
      ?? prepared.estimatedParentRecoverMs
      ?? this.getEngineChatParentRecoverCostMs();

    const prefillTokens =
      runtime?.stage === 'prefill'
        ? Math.min(prepared.promptTailTokens, this.getEngineChatPrefillSliceTokenBudget())
        : !runtime && prepared.promptTailTokens > 0
          ? Math.min(prepared.promptTailTokens, this.getEngineChatPrefillSliceTokenBudget())
          : 0;
    const predictTokens =
      runtime?.stage === 'decode'
        ? this.getEngineChatNextDecodeSliceTokens(runtime)
        : !runtime && prepared.promptTailTokens === 0
          ? this.getEngineChatInitialDecodeSliceTokens(nPredict)
          : 0;

    const restoreCostPerToken =
      prepared.restoreSource === 'l1'
        ? (this.engineChatLearnedRestoreL1CostPerTokenMs ?? this.getEngineChatRestoreL1CostPerTokenMs())
        : prepared.restoreSource === 'l2'
          ? (this.engineChatLearnedRestoreL2CostPerTokenMs ?? this.getEngineChatRestoreL2CostPerTokenMs())
          : prepared.restoreSource === 'l3'
            ? (this.engineChatLearnedRestoreL3CostPerTokenMs ?? this.getEngineChatRestoreL3CostPerTokenMs())
            : 0;

    const prefillMs = prefillTokens * prefillPerToken;
    const decodeMs = predictTokens * decodePerToken;
    const cacheMoveMs = prepared.restoredPrefixTokens * restoreCostPerToken;
    const rebuildMs = prepared.rebuiltPrefixTokens * rebuildPerToken;
    const queuePenaltyMs = waitMs > 0 ? Math.min(waitMs * 0.05, prefillMs + decodeMs) : 0;
    const totalMs = Math.max(
      1,
      prefillMs + decodeMs + postMs + cacheMoveMs + rebuildMs + parentRecoverMs + queuePenaltyMs
    );

    return {
      prefillMs,
      decodeMs,
      postMs,
      cacheMoveMs,
      rebuildMs,
      parentRecoverMs,
      totalMs,
      predictTokens,
    };
  }

  private requeueEngineChatRequest(req: EngineChatRequest): void {
    req.queueType = 'normal';
    req.enqueuedAt = Date.now();
    req.nAheadAtEnqueue = this.getEngineChatPendingCount();
    this.refreshEngineChatRequestEstimate(req, null, 0, req.nAheadAtEnqueue);
    this.engineChatNormalQueue.push(req);
  }

  private async ensureEngineChatRuntimeReady(req: EngineChatRequest): Promise<void> {
    if (!req.runtime) {
      await this.initializeEngineChatRuntime(req);
    } else {
      await this.resumeEngineChatRuntime(req);
    }
    if (!req.runtime) {
      throw new WllamaError(`engine chat runtime missing for req=${req.id}`);
    }
    await this.samplingInit(req.options.sampling ?? {}, req.runtime.samplingHistoryTokens);
  }

  private async initializeEngineChatRuntime(req: EngineChatRequest): Promise<void> {
    const started =
      Array.isArray(req.baseHistory)
        ? await this.treeChatStartFromHistory(req.baseHistory, req.userText)
        : await this.treeChatStart(req.parentId ?? 0, req.userText);
    const prompt = started.formattedPrompt || req.preparedPrompt?.prompt || (await this.formatChat(started.messages, true));
    const promptTokens = req.preparedPrompt?.promptTokens ?? (await this.tokenizePromptForEngineChat(prompt));
    const { cachedPrefixTokens, promptTailTokens } = await this.alignPromptWithCurrentCache(promptTokens);
    const rebuildUsed = started.timing.rebuildPromptMs > 0;
    req.runtime = {
      nodeId: started.nodeId,
      stage: promptTailTokens.length > 0 ? 'prefill' : 'decode',
      prompt,
      promptTokens,
      remainingPromptTokens: promptTailTokens,
      samplingHistoryTokens: [],
      assistantText: '',
      generatedTokenIds: [],
      generatedTokensLimit: Number.isFinite(req.options.nPredict as number)
        ? Math.max(0, Math.floor(req.options.nPredict as number))
        : Number.POSITIVE_INFINITY,
      generatedTokensSoFar: 0,
      startedAt: Date.now(),
      cacheLoadMs: started.timing.restoreCacheMs,
      rebuildMs: started.timing.rebuildPromptMs,
      parentRecoverMs: started.timing.parentRecoverMs,
      startOverheadMs: started.timing.startOverheadMs,
      accumulatedPrefillMs: 0,
      accumulatedDecodeMs: 0,
      restoredPrefixTokens: rebuildUsed ? 0 : cachedPrefixTokens,
      rebuiltPrefixTokens: rebuildUsed ? cachedPrefixTokens : 0,
      restoreSource: rebuildUsed ? 'rebuild' : (req.preparedPrompt?.restoreSource ?? 'none'),
      promptTokenCount: promptTokens.length,
      promptTailTokenCount: promptTailTokens.length,
      resumedPrefixTokenCount: cachedPrefixTokens,
    };
  }

  private async resumeEngineChatRuntime(req: EngineChatRequest): Promise<void> {
    if (!req.runtime) {
      return;
    }
    const resumed = await this.treeChatResume(req.runtime.nodeId);
    req.runtime.cacheLoadMs += resumed.restoreCacheMs;
    req.runtime.rebuildMs += resumed.rebuildPromptMs;
    req.runtime.resumedPrefixTokenCount = resumed.resumedPrefixTokenCount;
  }

  private async rollbackEngineChatRuntime(req: EngineChatRequest): Promise<void> {
    if (!req.runtime) {
      return;
    }
    try {
      await this.treeChatFinish(
        req.runtime.nodeId,
        req.runtime.assistantText,
        Math.max(0, Date.now() - req.runtime.startedAt),
        true
      );
    } catch {
      // Ignore rollback failure and rethrow the original error.
    } finally {
      req.runtime = undefined;
    }
  }

  private async alignPromptWithCurrentCache(promptTokens: number[]): Promise<{
    cachedPrefixTokens: number;
    promptTailTokens: number[];
  }> {
    const cachedTokens = await this.getCachedTokens();
    let cachedPrefixTokens = 0;
    for (; cachedPrefixTokens < Math.min(cachedTokens.length, promptTokens.length); cachedPrefixTokens++) {
      if (cachedTokens[cachedPrefixTokens] !== promptTokens[cachedPrefixTokens]) {
        break;
      }
    }
    try {
      await this.kvRemove(cachedPrefixTokens, -1);
      return {
        cachedPrefixTokens,
        promptTailTokens: promptTokens.slice(cachedPrefixTokens),
      };
    } catch {
      await this.kvClear();
      return {
        cachedPrefixTokens: 0,
        promptTailTokens: promptTokens,
      };
    }
  }

  private async executeEngineChatPrefillSlice(req: EngineChatRequest): Promise<number> {
    if (!req.runtime) {
      throw new WllamaError(`engine chat runtime missing for prefill req=${req.id}`);
    }
    const runtime = req.runtime;
    if (runtime.remainingPromptTokens.length === 0) {
      return 0;
    }

    await this.resetPerfContext();
    const roundBudgetMs = this.getEngineChatPrefillSliceMaxMs();
    const tokenBudget = this.getEngineChatPrefillSliceTokenBudget();
    const chunkSize = Math.max(1, this.loadedContextInfo?.n_batch ?? 1);
    const completionOpts = req.options.abortSignal ? { abortSignal: req.options.abortSignal } : {};
    const startedAt = Date.now();
    let processed = 0;

    while (runtime.remainingPromptTokens.length > 0 && processed < tokenBudget) {
      if (req.options.abortSignal?.aborted) {
        throw new WllamaAbortError();
      }
      const take = Math.min(chunkSize, tokenBudget - processed, runtime.remainingPromptTokens.length);
      const chunk = runtime.remainingPromptTokens.slice(0, take);
      const isFinalOverallChunk = take === runtime.remainingPromptTokens.length;
      await this.samplingAccept(chunk);
      runtime.samplingHistoryTokens.push(...chunk);
      if (this.isEncoderDecoderArchitecture()) {
        await this.encode(chunk, completionOpts);
      } else {
        await this.decode(chunk, {
          ...completionOpts,
          skipLogits: !isFinalOverallChunk,
        });
      }
      runtime.remainingPromptTokens.splice(0, take);
      processed += take;
      if (runtime.remainingPromptTokens.length > 0 && Date.now() - startedAt >= roundBudgetMs) {
        break;
      }
    }

    if (runtime.remainingPromptTokens.length === 0 && this.isEncoderDecoderArchitecture()) {
      await this.decode([this.getDecoderStartToken()], completionOpts);
    }

    const perf = await this.getPerfContext();
    return Number.isFinite(perf?.t_p_eval_ms) && (perf?.t_p_eval_ms ?? 0) > 0
      ? perf.t_p_eval_ms
      : Math.max(0, Date.now() - startedAt);
  }

  private async executeEngineChatDecodeSlice(req: EngineChatRequest): Promise<{
    decodeMs: number;
    aborted: boolean;
    finished: boolean;
  }> {
    if (!req.runtime) {
      throw new WllamaError(`engine chat runtime missing for decode req=${req.id}`);
    }
    const runtime = req.runtime;
    const stopTokens = new Set(req.options.stopTokens ?? []);
    const maxTokens = this.getEngineChatNextDecodeSliceTokens(runtime);
    if (maxTokens <= 0) {
      return { decodeMs: 0, aborted: false, finished: true };
    }

    await this.resetPerfContext();
    const stageStartedAt = Date.now();
    let aborted = false;
    let finished = false;
    for (let i = 0; i < maxTokens; i++) {
      const sampled = await this.samplingSample();
      if (this.isTokenEOG(sampled.token) || stopTokens.has(sampled.token)) {
        finished = true;
        break;
      }

      const piece = new TextDecoder().decode(sampled.piece, { stream: true });
      runtime.generatedTokenIds.push(sampled.token);
      runtime.generatedTokensSoFar += 1;
      runtime.assistantText += piece;
      runtime.firstTokenAt ??= Date.now();
      req.options.onChunk?.(piece, runtime.assistantText);

      if (req.options.abortSignal?.aborted) {
        aborted = true;
        break;
      }

      await this.samplingAccept([sampled.token]);
      runtime.samplingHistoryTokens.push(sampled.token);
      await this.decode([sampled.token], {});

      if (Number.isFinite(runtime.generatedTokensLimit)
        && runtime.generatedTokensSoFar >= runtime.generatedTokensLimit) {
        finished = true;
        break;
      }
    }

    const perf = await this.getPerfContext();
    const decodeMs =
      Number.isFinite(perf?.t_eval_ms) && (perf?.t_eval_ms ?? 0) > 0
        ? perf.t_eval_ms
        : Math.max(0, Date.now() - stageStartedAt);
    return { decodeMs, aborted, finished };
  }

  private async executeEngineChatSlice(
    req: EngineChatRequest
  ): Promise<
    | {
      done: true;
      result: { nodeId: number; assistantText: string; state: WllamaTreeState };
      phases: {
        prefillMs: number;
        decodeMs: number;
        postMs: number;
        cacheLoadMs: number;
        rebuildMs: number;
        parentRecoverMs: number;
      };
    }
    | { done: false }
  > {
    try {
      await this.ensureEngineChatRuntimeReady(req);
      if (!req.runtime) {
        throw new WllamaError(`engine chat runtime missing for req=${req.id}`);
      }

      if (req.runtime.stage === 'prefill') {
        const prefillMs = await this.executeEngineChatPrefillSlice(req);
        req.runtime.accumulatedPrefillMs += prefillMs;
        req.sliceCount += 1;
        req.runtime.resumedPrefixTokenCount = this.getEngineChatRuntimeLiveTokenCount(req.runtime);
        if (req.runtime.remainingPromptTokens.length > 0) {
          await this.treeChatCheckpoint(
            req.runtime.nodeId,
            req.runtime.assistantText,
            Date.now() - req.runtime.startedAt
          );
          return { done: false };
        }
        req.runtime.stage = 'decode';
        await this.treeChatCheckpoint(
          req.runtime.nodeId,
          req.runtime.assistantText,
          Date.now() - req.runtime.startedAt
        );
        return { done: false };
      }

      const detail = await this.executeEngineChatDecodeSlice(req);
      req.runtime.accumulatedDecodeMs += detail.decodeMs;
      req.generatedTokens = req.runtime.generatedTokensSoFar;
      req.sliceCount += 1;
      req.runtime.resumedPrefixTokenCount = this.getEngineChatRuntimeLiveTokenCount(req.runtime);

      if (detail.aborted) {
        const state = await this.treeChatFinish(
          req.runtime.nodeId,
          req.runtime.assistantText,
          Date.now() - req.runtime.startedAt,
          true
        );
        return {
          done: true,
          result: {
            nodeId: req.runtime.nodeId,
            assistantText: req.runtime.assistantText,
            state,
          },
          phases: {
            prefillMs: req.runtime.accumulatedPrefillMs,
            decodeMs: req.runtime.accumulatedDecodeMs,
            postMs: 0,
            cacheLoadMs: req.runtime.cacheLoadMs,
            rebuildMs: req.runtime.rebuildMs,
            parentRecoverMs: req.runtime.parentRecoverMs,
          },
        };
      }

      if (!detail.finished) {
        await this.treeChatCheckpoint(
          req.runtime.nodeId,
          req.runtime.assistantText,
          Date.now() - req.runtime.startedAt
        );
        return { done: false };
      }

      const postStartedAt = Date.now();
      const state = await this.treeChatFinish(
        req.runtime.nodeId,
        req.runtime.assistantText,
        Date.now() - req.runtime.startedAt,
        false
      );
      const postMs = Math.max(0, Date.now() - postStartedAt);
      return {
        done: true,
        result: {
          nodeId: req.runtime.nodeId,
          assistantText: req.runtime.assistantText,
          state,
        },
        phases: {
          prefillMs: req.runtime.accumulatedPrefillMs,
          decodeMs: req.runtime.accumulatedDecodeMs,
          postMs,
          cacheLoadMs: req.runtime.cacheLoadMs,
          rebuildMs: req.runtime.rebuildMs,
          parentRecoverMs: req.runtime.parentRecoverMs,
        },
      };
    } catch (err) {
      await this.rollbackEngineChatRuntime(req);
      throw err;
    }
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

  private getEngineChatSliceTokenBudget(): number {
    const configured = this.config.engineChatSliceTokenBudget;
    if (typeof configured !== 'number' || !Number.isFinite(configured) || configured <= 0) {
      return DEFAULT_ENGINE_CHAT_SLICE_TOKEN_BUDGET;
    }
    return Math.max(1, Math.floor(configured));
  }

  private getEngineChatEvictionHintTopK(): number {
    const configured = this.config.engineChatEvictionHintTopK;
    if (typeof configured !== 'number' || !Number.isFinite(configured) || configured <= 0) {
      return DEFAULT_ENGINE_CHAT_EVICTION_HINT_TOP_K;
    }
    return Math.max(1, Math.floor(configured));
  }

  private getEngineChatPrefillCostPerTokenMs(): number {
    const configured =
      this.config.engineChatPrefillCostPerTokenMs ?? this.config.engineChatPrefillCostPerUnitMs;
    if (typeof configured !== 'number' || !Number.isFinite(configured) || configured <= 0) {
      return DEFAULT_ENGINE_CHAT_COST_PREFILL_PER_TOKEN_MS;
    }
    return configured;
  }

  private getEngineChatDecodeCostPerTokenMs(): number {
    const configured = this.config.engineChatDecodeCostPerTokenMs;
    if (typeof configured !== 'number' || !Number.isFinite(configured) || configured <= 0) {
      return DEFAULT_ENGINE_CHAT_COST_DECODE_PER_TOKEN_MS;
    }
    return configured;
  }

  private getEngineChatPostCostMs(): number {
    const configured = this.config.engineChatPostCostMs;
    if (typeof configured !== 'number' || !Number.isFinite(configured) || configured <= 0) {
      return DEFAULT_ENGINE_CHAT_COST_POST_MS;
    }
    return configured;
  }

  private getEngineChatPrefillSliceMaxMs(): number {
    const configured = this.config.engineChatPrefillSliceMaxMs;
    if (typeof configured !== 'number' || !Number.isFinite(configured) || configured <= 0) {
      return DEFAULT_ENGINE_CHAT_PREFILL_SLICE_MAX_MS;
    }
    return Math.max(100, Math.floor(configured));
  }

  private getEngineChatPrefillSliceTokenBudget(): number {
    const perTokenMs = Math.max(0.1, this.getEngineChatPrefillCostPerTokenMs());
    return Math.max(1, Math.floor(this.getEngineChatPrefillSliceMaxMs() / perTokenMs));
  }

  private getEngineChatRestoreL1CostPerTokenMs(): number {
    const configured = this.config.engineChatRestoreL1CostPerTokenMs;
    if (typeof configured !== 'number' || !Number.isFinite(configured) || configured < 0) {
      return DEFAULT_ENGINE_CHAT_COST_RESTORE_L1_PER_TOKEN_MS;
    }
    return configured;
  }

  private getEngineChatRestoreL2CostPerTokenMs(): number {
    const configured =
      this.config.engineChatRestoreL2CostPerTokenMs ?? this.config.engineChatCacheMoveCostPerUnitMs;
    if (typeof configured !== 'number' || !Number.isFinite(configured) || configured < 0) {
      return DEFAULT_ENGINE_CHAT_COST_RESTORE_L2_PER_TOKEN_MS;
    }
    return configured;
  }

  private getEngineChatRestoreL3CostPerTokenMs(): number {
    const configured = this.config.engineChatRestoreL3CostPerTokenMs;
    if (typeof configured !== 'number' || !Number.isFinite(configured) || configured < 0) {
      return DEFAULT_ENGINE_CHAT_COST_RESTORE_L3_PER_TOKEN_MS;
    }
    return configured;
  }

  private getEngineChatRebuildCostPerTokenMs(): number {
    const configured =
      this.config.engineChatRebuildCostPerTokenMs ?? this.config.engineChatRebuildCostPerUnitMs;
    if (typeof configured !== 'number' || !Number.isFinite(configured) || configured < 0) {
      return DEFAULT_ENGINE_CHAT_COST_REBUILD_PER_TOKEN_MS;
    }
    return configured;
  }

  private getEngineChatParentRecoverCostMs(): number {
    const configured = this.config.engineChatParentRecoverCostMs;
    if (typeof configured !== 'number' || !Number.isFinite(configured) || configured < 0) {
      return DEFAULT_ENGINE_CHAT_COST_PARENT_RECOVER_MS;
    }
    return configured;
  }

  private getEngineChatCostWarmupRequests(): number {
    const configured = this.config.engineChatCostWarmupRequests;
    if (typeof configured !== 'number' || !Number.isFinite(configured) || configured <= 0) {
      return DEFAULT_ENGINE_CHAT_COST_WARMUP_REQUESTS;
    }
    return Math.max(1, Math.floor(configured));
  }

  private getEngineChatCostSampleWindow(): number {
    const configured = this.config.engineChatCostSampleWindow;
    if (typeof configured !== 'number' || !Number.isFinite(configured) || configured <= 0) {
      return DEFAULT_ENGINE_CHAT_COST_SAMPLE_WINDOW;
    }
    return Math.max(16, Math.floor(configured));
  }

  private recordEngineChatCostObservation(
    req: EngineChatRequest,
    phases: {
      prefillMs: number;
      decodeMs: number;
      postMs: number;
      cacheLoadMs: number;
      rebuildMs: number;
      parentRecoverMs: number;
    },
    totalServiceMs: number
  ): void {
    if (!Number.isFinite(totalServiceMs) || totalServiceMs <= 0) {
      return;
    }

    const decodeTokens = Math.max(0, Math.floor(req.generatedTokens));
    const promptTailTokens = Math.max(
      0,
      Math.floor(req.runtime?.promptTailTokenCount ?? req.estimatedPromptTailTokens)
    );
    const restoredPrefixTokens = Math.max(
      0,
      Math.floor(req.runtime?.restoredPrefixTokens ?? req.estimatedRestoredPrefixTokens)
    );
    const rebuiltPrefixTokens = Math.max(
      0,
      Math.floor(req.runtime?.rebuiltPrefixTokens ?? req.estimatedRebuiltPrefixTokens)
    );
    const restoredFromTier =
      req.runtime?.restoreSource === 'l1' || req.estimatedRestoreSource === 'l1'
        ? 1
        : req.runtime?.restoreSource === 'l2' || req.estimatedRestoreSource === 'l2'
          ? 2
          : req.runtime?.restoreSource === 'l3' || req.estimatedRestoreSource === 'l3'
            ? 3
            : 0;

    const obs: EngineChatCostObservation = {
      promptTailTokens,
      decodeTokens,
      restoredPrefixTokens,
      restoredFromTier: restoredFromTier as 0 | 1 | 2 | 3,
      rebuiltPrefixTokens,
      prefillMs: Math.max(0, phases.prefillMs),
      decodeMs: Math.max(0, phases.decodeMs),
      postMs: Math.max(0, phases.postMs),
      cacheLoadMs: Math.max(0, phases.cacheLoadMs),
      rebuildMs: Math.max(0, phases.rebuildMs),
      parentRecoverMs: Math.max(0, phases.parentRecoverMs),
      totalServiceMs: Math.max(0, totalServiceMs),
    };

    this.engineChatCostObservedCount += 1;
    this.engineChatCostObservations.push(obs);

    const maxWindow = this.getEngineChatCostSampleWindow();
    if (this.engineChatCostObservations.length > maxWindow) {
      this.engineChatCostObservations.splice(0, this.engineChatCostObservations.length - maxWindow);
    }

    this.maybeUpdateEngineChatLearnedCostModel();
  }

  private maybeUpdateEngineChatLearnedCostModel(): void {
    if (this.engineChatCostObservedCount < this.getEngineChatCostWarmupRequests()) {
      return;
    }
    if (this.engineChatCostObservations.length < 5) {
      return;
    }

    let prefillWeightedMs = 0;
    let prefillWeight = 0;
    let decodeWeightedMs = 0;
    let decodeWeight = 0;
    let postMsSum = 0;
    let restoreL1WeightedMs = 0;
    let restoreL1Weight = 0;
    let restoreL2WeightedMs = 0;
    let restoreL2Weight = 0;
    let restoreL3WeightedMs = 0;
    let restoreL3Weight = 0;
    let rebuildWeightedMs = 0;
    let rebuildWeight = 0;
    let parentRecoverMsSum = 0;

    for (const obs of this.engineChatCostObservations) {
      if (obs.promptTailTokens > 0) {
        prefillWeightedMs += obs.prefillMs;
        prefillWeight += obs.promptTailTokens;
      }
      if (obs.decodeTokens > 0) {
        decodeWeightedMs += obs.decodeMs;
        decodeWeight += obs.decodeTokens;
      }
      if (obs.restoredPrefixTokens > 0) {
        if (obs.restoredFromTier === 1) {
          restoreL1WeightedMs += obs.cacheLoadMs;
          restoreL1Weight += obs.restoredPrefixTokens;
        } else if (obs.restoredFromTier === 2) {
          restoreL2WeightedMs += obs.cacheLoadMs;
          restoreL2Weight += obs.restoredPrefixTokens;
        } else if (obs.restoredFromTier === 3) {
          restoreL3WeightedMs += obs.cacheLoadMs;
          restoreL3Weight += obs.restoredPrefixTokens;
        }
      }
      if (obs.rebuiltPrefixTokens > 0) {
        rebuildWeightedMs += obs.rebuildMs;
        rebuildWeight += obs.rebuiltPrefixTokens;
      }
      postMsSum += obs.postMs;
      parentRecoverMsSum += obs.parentRecoverMs;
    }

    const learnedPrefill = prefillWeight > 0
      ? prefillWeightedMs / prefillWeight
      : this.getEngineChatPrefillCostPerTokenMs();
    const learnedDecode = decodeWeight > 0
      ? decodeWeightedMs / decodeWeight
      : this.getEngineChatDecodeCostPerTokenMs();
    const learnedRestoreL1 = restoreL1Weight > 0
      ? restoreL1WeightedMs / restoreL1Weight
      : this.getEngineChatRestoreL1CostPerTokenMs();
    const learnedRestoreL2 = restoreL2Weight > 0
      ? restoreL2WeightedMs / restoreL2Weight
      : this.getEngineChatRestoreL2CostPerTokenMs();
    const learnedRestoreL3 = restoreL3Weight > 0
      ? restoreL3WeightedMs / restoreL3Weight
      : this.getEngineChatRestoreL3CostPerTokenMs();
    const learnedRebuild = rebuildWeight > 0
      ? rebuildWeightedMs / rebuildWeight
      : this.getEngineChatRebuildCostPerTokenMs();
    const learnedPost = postMsSum / this.engineChatCostObservations.length;
    const learnedParentRecover = parentRecoverMsSum / this.engineChatCostObservations.length;

    const boundedPrefill = Math.min(200, Math.max(0.1, learnedPrefill));
    const boundedDecode = Math.min(200, Math.max(0.1, learnedDecode));
    const boundedRestoreL1 = Math.min(50, Math.max(0, learnedRestoreL1));
    const boundedRestoreL2 = Math.min(200, Math.max(0, learnedRestoreL2));
    const boundedRestoreL3 = Math.min(400, Math.max(0, learnedRestoreL3));
    const boundedRebuild = Math.min(400, Math.max(0, learnedRebuild));
    const boundedPost = Math.min(2000, Math.max(0, learnedPost));
    const boundedParentRecover = Math.min(2000, Math.max(0, learnedParentRecover));

    const alpha = 0.2;
    this.engineChatLearnedPrefillCostPerTokenMs =
      this.engineChatLearnedPrefillCostPerTokenMs === undefined
        ? boundedPrefill
        : (1 - alpha) * this.engineChatLearnedPrefillCostPerTokenMs + alpha * boundedPrefill;
    this.engineChatLearnedDecodeCostPerTokenMs =
      this.engineChatLearnedDecodeCostPerTokenMs === undefined
        ? boundedDecode
        : (1 - alpha) * this.engineChatLearnedDecodeCostPerTokenMs + alpha * boundedDecode;
    this.engineChatLearnedPostCostMs =
      this.engineChatLearnedPostCostMs === undefined
        ? boundedPost
        : (1 - alpha) * this.engineChatLearnedPostCostMs + alpha * boundedPost;
    this.engineChatLearnedRestoreL1CostPerTokenMs =
      this.engineChatLearnedRestoreL1CostPerTokenMs === undefined
        ? boundedRestoreL1
        : (1 - alpha) * this.engineChatLearnedRestoreL1CostPerTokenMs + alpha * boundedRestoreL1;
    this.engineChatLearnedRestoreL2CostPerTokenMs =
      this.engineChatLearnedRestoreL2CostPerTokenMs === undefined
        ? boundedRestoreL2
        : (1 - alpha) * this.engineChatLearnedRestoreL2CostPerTokenMs + alpha * boundedRestoreL2;
    this.engineChatLearnedRestoreL3CostPerTokenMs =
      this.engineChatLearnedRestoreL3CostPerTokenMs === undefined
        ? boundedRestoreL3
        : (1 - alpha) * this.engineChatLearnedRestoreL3CostPerTokenMs + alpha * boundedRestoreL3;
    this.engineChatLearnedRebuildCostPerTokenMs =
      this.engineChatLearnedRebuildCostPerTokenMs === undefined
        ? boundedRebuild
        : (1 - alpha) * this.engineChatLearnedRebuildCostPerTokenMs + alpha * boundedRebuild;
    this.engineChatLearnedParentRecoverCostMs =
      this.engineChatLearnedParentRecoverCostMs === undefined
        ? boundedParentRecover
        : (1 - alpha) * this.engineChatLearnedParentRecoverCostMs + alpha * boundedParentRecover;

    this.traceEngineChat(
      `cost-model update n=${this.engineChatCostObservedCount} coeff={prefill:${this.engineChatLearnedPrefillCostPerTokenMs.toFixed(2)}, decode:${this.engineChatLearnedDecodeCostPerTokenMs.toFixed(2)}, post:${this.engineChatLearnedPostCostMs.toFixed(2)}, restoreL1:${this.engineChatLearnedRestoreL1CostPerTokenMs.toFixed(2)}, restoreL2:${this.engineChatLearnedRestoreL2CostPerTokenMs.toFixed(2)}, restoreL3:${this.engineChatLearnedRestoreL3CostPerTokenMs.toFixed(2)}, rebuild:${this.engineChatLearnedRebuildCostPerTokenMs.toFixed(2)}, parentRecover:${this.engineChatLearnedParentRecoverCostMs.toFixed(2)}}`
    );
  }

  private isBetterEngineChatCandidate(
    a: {
      req: EngineChatRequest;
      waitMs: number;
      estimatedServiceMs: number;
    },
    b: {
      req: EngineChatRequest;
      waitMs: number;
      estimatedServiceMs: number;
    },
    useOverdueOrder: boolean
  ): boolean {
    if (useOverdueOrder) {
      // overdue order: (wait, estimatedService, enqueuedAt, id)
      if (a.waitMs !== b.waitMs) return a.waitMs > b.waitMs;
      if (a.estimatedServiceMs !== b.estimatedServiceMs) {
        return a.estimatedServiceMs < b.estimatedServiceMs;
      }
      if (a.req.enqueuedAt !== b.req.enqueuedAt) return a.req.enqueuedAt < b.req.enqueuedAt;
      return a.req.id < b.req.id;
    }

    // non-overdue order: (estimatedService, wait, enqueuedAt, id)
    if (a.estimatedServiceMs !== b.estimatedServiceMs) {
      return a.estimatedServiceMs < b.estimatedServiceMs;
    }
    if (a.waitMs !== b.waitMs) return a.waitMs > b.waitMs;
    if (a.req.enqueuedAt !== b.req.enqueuedAt) return a.req.enqueuedAt < b.req.enqueuedAt;
    return a.req.id < b.req.id;
  }

  private getEngineChatInitialDecodeSliceTokens(nPredict: number | undefined): number {
    if (typeof nPredict !== 'number' || !Number.isFinite(nPredict)) {
      return this.getEngineChatSliceTokenBudget();
    }
    return Math.max(0, Math.floor(nPredict));
  }

  private getEngineChatRuntimeLiveTokenCount(runtime: EngineChatRuntime): number {
    return Math.max(
      0,
      runtime.promptTokenCount - runtime.remainingPromptTokens.length + runtime.generatedTokensSoFar
    );
  }

  private getEngineChatNextDecodeSliceTokens(runtime: EngineChatRuntime): number {
    if (!Number.isFinite(runtime.generatedTokensLimit)) {
      return this.getEngineChatSliceTokenBudget();
    }
    return Math.max(
      0,
      Math.max(0, runtime.generatedTokensLimit - runtime.generatedTokensSoFar)
    );
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
    parentId: number | undefined,
    userText: string,
    options: WllamaChatFromNodeOptions = {},
    baseHistory?: WllamaChatMessage[]
  ): Promise<{ nodeId: number; assistantText: string; state: WllamaTreeState }> {
    if (options.abortSignal?.aborted) {
      throw new WllamaAbortError();
    }

    const reqId = (options as unknown as { __engineReqId?: number }).__engineReqId;
    const parentKey =
      typeof parentId === 'number' && parentId >= 0
        ? `parent=${parentId}`
        : `historyLen=${baseHistory?.length ?? 0}`;
    const traceKey = `req=${reqId ?? 'na'} ${parentKey}`;
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

    const started =
      Array.isArray(baseHistory)
        ? await this.treeChatStartFromHistory(baseHistory, userText)
        : await this.treeChatStart(parentId ?? 0, userText);
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

  private async treeCacheHint(hotNodeIds: number[], queuePressure: number): Promise<void> {
    this.checkModelLoaded();
    const result = await this.proxy.wllamaAction<GlueMsgTreeCacheHintRes>('chat_cache_hint', {
      _name: 'tchi_req',
      hot_node_ids: hotNodeIds,
      queue_pressure: queuePressure,
    });
    if (!result.success) {
      throw new WllamaError(`treeCacheHint failed: ${result.message}`);
    }
  }

  private async treeChatResume(nodeId: number): Promise<{
    restoreCacheMs: number;
    rebuildPromptMs: number;
    resumedPrefixTokenCount: number;
  }> {
    this.checkModelLoaded();
    const result = await this.proxy.wllamaAction<GlueMsgTreeChatResumeRes>('chat_resume', {
      _name: 'tchr_req',
      node_id: nodeId,
    });
    if (!result.success) {
      throw new WllamaError(`treeChatResume(${nodeId}) failed: ${result.message}`);
    }
    this.nCachedTokens = result.resumed_prefix_token_count ?? 0;
    return {
      restoreCacheMs: result.restore_cache_ms ?? 0,
      rebuildPromptMs: result.rebuild_prompt_ms ?? 0,
      resumedPrefixTokenCount: result.resumed_prefix_token_count ?? 0,
    };
  }

  private async treeChatCheckpoint(
    nodeId: number,
    assistantText: string,
    generationTimeMs: number
  ): Promise<void> {
    this.checkModelLoaded();
    const result = await this.proxy.wllamaAction<GlueMsgTreeChatCheckpointRes>('chat_checkpoint', {
      _name: 'tchp_req',
      node_id: nodeId,
      assistant_text: assistantText,
      generation_time_ms: generationTimeMs,
    });
    if (!result.success) {
      throw new WllamaError(`treeChatCheckpoint(${nodeId}) failed: ${result.message}`);
    }
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

  private async treeChatStart(parentId: number, userText: string): Promise<{
    nodeId: number;
    messages: WllamaChatMessage[];
    formattedPrompt: string;
    state: WllamaTreeState;
    timing: {
      restoreCacheMs: number;
      rebuildPromptMs: number;
      parentRecoverMs: number;
      startOverheadMs: number;
    };
  }> {
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
      timing: {
        restoreCacheMs: result.restore_cache_ms ?? 0,
        rebuildPromptMs: result.rebuild_prompt_ms ?? 0,
        parentRecoverMs: result.parent_recover_ms ?? 0,
        startOverheadMs: result.start_overhead_ms ?? 0,
      },
    };
  }

  private async treeChatStartFromHistory(
    history: WllamaChatMessage[],
    userText: string
  ): Promise<{
    nodeId: number;
    messages: WllamaChatMessage[];
    formattedPrompt: string;
    state: WllamaTreeState;
    timing: {
      restoreCacheMs: number;
      rebuildPromptMs: number;
      parentRecoverMs: number;
      startOverheadMs: number;
    };
  }> {
    this.checkModelLoaded();
    const roles = history.map((m) => m.role);
    const contents = history.map((m) => m.content);
    const result = await this.proxy.wllamaAction<GlueMsgTreeChatStartHistRes>('chat_start_hist', {
      _name: 'tchh_req',
      roles,
      contents,
      user_text: userText,
    });
    if (!result.success) {
      throw new WllamaError(`treeChatStartFromHistory failed: ${result.message}`);
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
    this.nCachedTokens = state.nodes.get(state.activeNodeId)?.prefixTokenCount ?? 0;
    return {
      nodeId: result.node_id,
      messages,
      formattedPrompt: result.formatted_chat ?? '',
      state,
      timing: {
        restoreCacheMs: result.restore_cache_ms ?? 0,
        rebuildPromptMs: result.rebuild_prompt_ms ?? 0,
        parentRecoverMs: result.parent_recover_ms ?? 0,
        startOverheadMs: result.start_overhead_ms ?? 0,
      },
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
    const toDebugReq = (req: EngineChatRequest) => ({
      id: req.id,
      queueType: req.queueType,
      parentId: req.parentId ?? null,
      historyLength: req.baseHistory?.length ?? 0,
      enqueuedForMs: now - req.enqueuedAt,
      processingForMs: req.processingStartedAt ? now - req.processingStartedAt : 0,
      nAheadAtEnqueue: req.nAheadAtEnqueue,
      waitBudgetMs: req.waitBudgetMs,
      estimatedServiceMs: req.estimatedServiceMs,
      estimatedPrefillMs: req.estimatedPrefillMs,
      estimatedDecodeMs: req.estimatedDecodeMs,
      estimatedPostMs: req.estimatedPostMs,
      estimatedCacheMoveMs: req.estimatedCacheMoveMs,
      estimatedRebuildMs: req.estimatedRebuildMs,
      estimatedParentRecoverMs: req.estimatedParentRecoverMs,
      estimatedPromptTokens: req.estimatedPromptTokens,
      estimatedPromptTailTokens: req.estimatedPromptTailTokens,
      estimatedRestoredPrefixTokens: req.estimatedRestoredPrefixTokens,
      estimatedRebuiltPrefixTokens: req.estimatedRebuiltPrefixTokens,
      estimatedRestoreSource: req.estimatedRestoreSource,
      targetPredictTokens: req.targetPredictTokens,
      generatedTokens: req.generatedTokens,
      sliceCount: req.sliceCount,
      runtimeNodeId: req.runtime?.nodeId ?? null,
      runtimeStage: req.runtime?.stage ?? null,
      runtimeAssistantChars: req.runtime?.assistantText.length ?? 0,
    });
    const normalPending = this.engineChatNormalQueue.map(toDebugReq);
    const overduePending = this.engineChatOverdueQueue.map(toDebugReq);

    if (nativeDebug && typeof nativeDebug === 'object') {
      return {
        ...(nativeDebug as Record<string, unknown>),
        engineChat: {
          running: this.engineChatQueueRunning,
          pendingCount: this.getEngineChatPendingCount(),
          normalPendingCount: normalPending.length,
          overduePendingCount: overduePending.length,
          costModel: {
            observedCount: this.engineChatCostObservedCount,
            warmupRequests: this.getEngineChatCostWarmupRequests(),
            sampleWindow: this.getEngineChatCostSampleWindow(),
            sampleCount: this.engineChatCostObservations.length,
            learnedPrefillCostPerTokenMs: this.engineChatLearnedPrefillCostPerTokenMs ?? null,
            learnedDecodeCostPerTokenMs: this.engineChatLearnedDecodeCostPerTokenMs ?? null,
            learnedPostCostMs: this.engineChatLearnedPostCostMs ?? null,
            learnedRestoreL1CostPerTokenMs: this.engineChatLearnedRestoreL1CostPerTokenMs ?? null,
            learnedRestoreL2CostPerTokenMs: this.engineChatLearnedRestoreL2CostPerTokenMs ?? null,
            learnedRestoreL3CostPerTokenMs: this.engineChatLearnedRestoreL3CostPerTokenMs ?? null,
            learnedRebuildCostPerTokenMs: this.engineChatLearnedRebuildCostPerTokenMs ?? null,
            learnedParentRecoverCostMs: this.engineChatLearnedParentRecoverCostMs ?? null,
          },
          normalPending,
          overduePending,
        },
      };
    }

    return {
      nativeDebug,
      engineChat: {
        running: this.engineChatQueueRunning,
        pendingCount: this.getEngineChatPendingCount(),
        normalPendingCount: normalPending.length,
        overduePendingCount: overduePending.length,
        costModel: {
          observedCount: this.engineChatCostObservedCount,
          warmupRequests: this.getEngineChatCostWarmupRequests(),
          sampleWindow: this.getEngineChatCostSampleWindow(),
          sampleCount: this.engineChatCostObservations.length,
          learnedPrefillCostPerTokenMs: this.engineChatLearnedPrefillCostPerTokenMs ?? null,
          learnedDecodeCostPerTokenMs: this.engineChatLearnedDecodeCostPerTokenMs ?? null,
          learnedPostCostMs: this.engineChatLearnedPostCostMs ?? null,
          learnedRestoreL1CostPerTokenMs: this.engineChatLearnedRestoreL1CostPerTokenMs ?? null,
          learnedRestoreL2CostPerTokenMs: this.engineChatLearnedRestoreL2CostPerTokenMs ?? null,
          learnedRestoreL3CostPerTokenMs: this.engineChatLearnedRestoreL3CostPerTokenMs ?? null,
          learnedRebuildCostPerTokenMs: this.engineChatLearnedRebuildCostPerTokenMs ?? null,
          learnedParentRecoverCostMs: this.engineChatLearnedParentRecoverCostMs ?? null,
        },
        normalPending,
        overduePending,
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
