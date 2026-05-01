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
export type BenchBackend = 'wllama' | 'web-llm';

export type BenchTarget = 'mmlu' | 'hella' | 'both';
export type MmluExperimentMode = 'exp1-sequential-once' | 'exp2-random-twice';
export type ExperimentRunMode = 'exp1' | 'exp2' | 'exp4';
export type TrueTreeReplacementPolicy = 'hybrid' | 'lru' | 'lfu' | 'size-only' | 'random';

export interface BenchConfig {
  backend: BenchBackend;
  webllmModelId: string;
  modelUrl: string;
  nCtx: number;
  nBatch: number;
  engineChatServiceUpperBoundMs: number;
  engineChatQueueMaxPending: number;
  engineChatSliceTokenBudget: number;
  engineChatPrefillSliceMaxMs: number;
  engineChatCostWarmupRequests: number;
  engineChatCostSampleWindow: number;
  engineChatTraceEnabled: boolean;
  treeBackend: TreeBackend;
  target: BenchTarget;
  experimentRunMode: ExperimentRunMode;
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
  exp4Concurrency: number;
  exp4SampleCount: number;
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

export interface BenchSampleMetric {
  benchmark: 'MMLU' | 'HellaSwag';
  id: string;
  subject?: string;
  mode: 'flat' | 'tree' | 'web-llm';
  latencyMs: number;
  ttftMs: number;
  tokensPerSecond: number;
  tokenCount: number;
  correct: boolean;
  visitOrdinal?: number;
  isRepeatVisit?: boolean;
  restoreSource?: 'L1' | 'L2' | 'L3' | 'MISS' | 'REBUILD' | 'UNKNOWN';
  hadParentRecover?: boolean;
  tierTokensAfter?: {
    l1: number;
    l2: number;
    l3: number;
  };
  tierSlotsAfter?: {
    l1: number;
    l2: number;
    l3: number;
  };
}

export interface CdfPoint {
  value: number;
  cdf: number;
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
  sampleMetricsFlat: BenchSampleMetric[];
  sampleMetricsTree: BenchSampleMetric[];
  latencyCdfFlat: CdfPoint[];
  latencyCdfTree: CdfPoint[];
  ttftCdfFlat: CdfPoint[];
  ttftCdfTree: CdfPoint[];
}

export interface CacheProfile {
  maintenanceMs: number;
  runTotalMs: number;
  maintenancePct: number;
  maintenanceBreakdownMs: {
    sessionInitMs: number;
    stateReadMs: number;
    prefixSetupMs: number;
    otherMs: number;
  };
  snapshotTokenBytes: number;
  tierL1Tokens: number;
  tierL2Tokens: number;
  tierL3Tokens: number;
  occupancyStats: {
    sampleCount: number;
    avgSnapshotTokenBytes: number;
    peakSnapshotTokenBytes: number;
    avgTierL1Tokens: number;
    avgTierL2Tokens: number;
    avgTierL3Tokens: number;
    peakTierL1Tokens: number;
    peakTierL2Tokens: number;
    peakTierL3Tokens: number;
    avgTierL1Slots: number;
    avgTierL2Slots: number;
    avgTierL3Slots: number;
    peakTierL1Slots: number;
    peakTierL2Slots: number;
    peakTierL3Slots: number;
  };
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
  batchWallClockMsQueue: number;
  batchWallClockMsDirect: number;
  batchTokensPerSecondQueue: number;
  batchTokensPerSecondDirect: number;
  queueEngineChat?: EngineChatBatchDiagnostics;
  directEngineChat?: EngineChatBatchDiagnostics;
}

export type SchedulingStressArm = 'fcfs-no-slice' | 'single-size-aware' | 'dual-queue';
export type SchedulingWorkloadClass = 'short' | 'medium' | 'long';

export interface SchedulingStressRequestMetric {
  arm: SchedulingStressArm;
  workloadClass: SchedulingWorkloadClass;
  benchmark: 'MMLU' | 'HellaSwag';
  itemId: string;
  subject?: string;
  arrivalOffsetMs: number;
  estimatedServiceMs: number;
  estimatedPromptTokens: number;
  waitBudgetMs: number;
  promptChars: number;
  outputTokenBudget: number;
  completed: boolean;
  timedOut: boolean;
  waitMs: number;
  ttftMs: number;
  serviceToFirstTokenMs: number;
  serviceMs: number;
  sojournMs: number;
  tokenCount: number;
  generatedTokens: number;
  tokensPerSecond: number;
  sliceCount: number;
  promotedToOverdueCount: number;
  hadPendingDependency: boolean;
  finalQueueType?: 'normal' | 'overdue';
}

export interface SchedulingStressClassSummary {
  workloadClass: SchedulingWorkloadClass;
  requestCount: number;
  completedCount: number;
  timeoutCount: number;
  completionRate: number;
  timeoutRate: number;
  waitSlaMs: number;
  waitSlaViolationCount: number;
  waitSlaViolationRate: number;
  avgWaitMs: number;
  p95WaitMs: number;
  p99WaitMs: number;
  maxWaitMs: number;
  avgTtftMs: number;
  p95TtftMs: number;
  p99TtftMs: number;
  avgSojournMs: number;
  p95SojournMs: number;
  p99SojournMs: number;
}

export interface SchedulingStressArmSummary {
  arm: SchedulingStressArm;
  requestCount: number;
  completedCount: number;
  timeoutCount: number;
  completionRate: number;
  timeoutRate: number;
  batchWallClockMs: number;
  throughputReqPerSec: number;
  throughputTokensPerSec: number;
  avgWaitMs: number;
  p50WaitMs: number;
  p95WaitMs: number;
  p99WaitMs: number;
  avgTtftMs: number;
  p50TtftMs: number;
  p95TtftMs: number;
  p99TtftMs: number;
  avgSojournMs: number;
  p50SojournMs: number;
  p95SojournMs: number;
  p99SojournMs: number;
  byClass: SchedulingStressClassSummary[];
  engineChat?: EngineChatBatchDiagnostics;
}

export interface SchedulingStressSummary {
  scenario: 'head-of-line-mixed';
  scenarioConfig: {
    longHeadCount: number;
    shortTailCount: number;
    mediumTailCount: number;
    longArrivalGapMs: number;
    mediumBurstDelayMs: number;
    shortStreamStartMs: number;
    shortInterArrivalMs: number;
    mediumPrefixSeedShots: number;
    longPrefixSeedShots: number;
    mediumPrefixExampleCount: number;
    longPrefixExampleCount: number;
    mediumPromptTokenBudget: number;
    longPromptTokenBudget: number;
    estimatedMediumPromptTokens: number;
    estimatedLongPromptTokens: number;
    dualQueueWaitBudgetScale: number;
    dualQueueWaitBudgetCapMs: number;
  };
  requestMetrics: SchedulingStressRequestMetric[];
  arms: SchedulingStressArmSummary[];
}

export interface EngineChatBatchDiagnostics {
  sampleCount: number;
  maxPendingCount: number;
  maxNormalPendingCount: number;
  maxOverduePendingCount: number;
  snapshotsWithOverdue: number;
  maxEstimatedServiceMs: number;
  maxEstimatedPromptTokens: number;
  maxSliceCount: number;
  maxGeneratedTokens: number;
  maxRuntimeAssistantChars: number;
  costModelObservedCount: number;
  costModelSampleCount: number;
  learnedPrefillCostPerTokenMs: number | null;
  learnedDecodeCostPerTokenMs: number | null;
  learnedPostCostMs: number | null;
  learnedRestoreL1CostPerTokenMs: number | null;
  learnedRestoreL2CostPerTokenMs: number | null;
  learnedRestoreL3CostPerTokenMs: number | null;
  learnedRebuildCostPerTokenMs: number | null;
  learnedParentRecoverCostMs: number | null;
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
  schedulingStress?: SchedulingStressSummary;
  diagnostics?: BenchDiagnostics;
}
