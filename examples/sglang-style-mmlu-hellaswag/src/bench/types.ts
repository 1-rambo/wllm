export interface MMLUItem {
  id: string;
  subject: string;
  question: string;
  choices: [string, string, string, string];
  answerIndex: number;
}

export interface HellaSwagItem {
  id: string;
  ctx: string;
  endings: [string, string, string, string];
  label: number;
}

export type TreeBackend = 'true-tree';

export type BenchTarget = 'mmlu' | 'hella' | 'both';
export type MmluExperimentMode = 'exp1-sequential-once' | 'exp2-random-twice';
export type TrueTreeReplacementPolicy = 'hybrid' | 'lru' | 'lfu' | 'size-only' | 'random';

export interface BenchConfig {
  modelUrl: string;
  nCtx: number;
  nBatch: number;
  treeBackend: TreeBackend;
  target: BenchTarget;
  mmluExperimentMode: MmluExperimentMode;
  randomSeed: number;
  trueTreeMemoryCapMB: number;
  trueTreeTieredCacheEnabled: boolean;
  trueTreeTierL1TokenCap: number;
  trueTreeTierL2TokenCap: number;
  trueTreeTierL3TokenCap: number;
  trueTreePruneL1L2TokenThreshold: number;
  trueTreePruneL2L3TokenThreshold: number;
  trueTreeReplacementPolicy: TrueTreeReplacementPolicy;
  mmluShots: number;
  hellaShots: number;
  mmluEvalCount: number;
  hellaEvalCount: number;
  runExp4: boolean;
  exp4Concurrency: number;
  exp4OutputTokens: number;
}

export interface BenchLogEvent {
  text: string;
}

export interface BenchProgressEvent {
  current: number;
  total: number;
  label: string;
}

export interface QAResult {
  id: string;
  predIndexFlat: number;
  predIndexTree: number;
  gtIndex: number;
  correctFlat: boolean;
  correctTree: boolean;
  explainFlat: string;
  explainTree: string;
}

export interface BenchSummary {
  benchmark: 'MMLU' | 'HellaSwag';
  shots: number;
  evalCount: number;
  accFlat: number;
  accTree: number;
  avgTtftMsFlat: number;
  avgTtftMsTree: number;
  ttftSpeedupPct: number;
  avgTokensPerSecondFlat: number;
  avgTokensPerSecondTree: number;
  tpsGainPct: number;
  avgLatencyMsFlat: number;
  avgLatencyMsTree: number;
  speedupPct: number;
  results: QAResult[];
}

export interface CacheProfile {
  maintenanceMs: number;
  runTotalMs: number;
  maintenancePct: number;
  snapshotTokenBytes: number;
  tierL1Tokens: number;
  tierL2Tokens: number;
  tierL3Tokens: number;
}

export interface QueueVsDirectSummary {
  requestCount: number;
  failedCountQueue: number;
  failedCountDirect: number;
  avgTtftMsQueue: number;
  avgTtftMsDirect: number;
  avgLatencyMsQueue: number;
  avgLatencyMsDirect: number;
  avgTokensPerSecondQueue: number;
  avgTokensPerSecondDirect: number;
}

export interface BenchDiagnostics {
  runtimeRestartCount: number;
  timeoutFailureCount: number;
  abortFailureCount: number;
  disposedFailureCount: number;
  otherFailureCount: number;
  timeoutPhaseCounts?: Record<string, number>;
  exp2CacheStats?: {
    restoreAttempts: number;
    restoreHits: number;
    restoreHitRatePct: number;
    restoreHitsL1: number;
    restoreHitsL2: number;
    restoreHitsL3: number;
    restoreMisses: number;
    restoreRebuilds: number;
    promotions: number;
    demotions: number;
    diskReads: number;
    diskWrites: number;
    l3OverflowEvents: number;
    parentRecoverAttempts: number;
    parentRecoverSuccesses: number;
    parentRecoverFailures: number;
    slotAllocHits: number;
    slotAllocMisses: number;
    slotEvictL1: number;
    slotEvictL2: number;
    slotEvictL3: number;
    fallbackReplays: number;
    nodeCacheAttempts?: number;
    sharedNodeCacheHits?: number;
    sharedNodeCacheMisses?: number;
    sharedNodeCacheHitRatePct?: number;
    questionNodeCacheHits?: number;
    questionNodeCacheMisses?: number;
    questionNodeCacheHitRatePct?: number;
  };
}

export interface BenchReport {
  modelUrl: string;
  config: BenchConfig;
  mmlu?: BenchSummary;
  hella?: BenchSummary;
  cacheProfile?: CacheProfile;
  queueVsDirect?: QueueVsDirectSummary;
  diagnostics?: BenchDiagnostics;
}
