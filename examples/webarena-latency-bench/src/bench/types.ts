export interface WebArenaTask {
  id: string;
  taskId: number;
  site: 'shopping_admin' | 'shopping' | 'gitlab' | 'reddit' | 'wikipedia' | string;
  sites: string[];
  startUrls: string[];
  renderedStartUrls?: string[];
  intent: string;
  intentTemplateId?: number;
  intentTemplate?: string;
  instantiationDict?: Record<string, unknown>;
  revision?: unknown;
  pageContext?: string;
  pageContextKey?: string;
}

export type BenchBackend = 'wllama' | 'web-llm';
export type BenchMode = 'flat' | 'tree' | 'web-llm';

export interface BenchConfig {
  backend: BenchBackend;
  webllmModelId: string;
  modelUrl: string;
  nCtx: number;
  nBatch: number;
  usePreloadedPageContext: boolean;
  trueTreeMemoryCapMB: number;
  trueTreeTieredCacheEnabled: boolean;
  trueTreeTierL1TokenCap: number;
  trueTreeTierL2TokenCap: number;
  trueTreeTierL3TokenCap: number;
  trueTreePruneL1L2TokenThreshold: number;
  trueTreePruneL2L3TokenThreshold: number;
  trueTreeReplacementPolicy: 'hybrid' | 'lru' | 'lfu' | 'size-only' | 'random';
  evalCount: number;
  maxOutputTokens: number;
  includeSites: string[];
}

export interface BenchLogEvent {
  text: string;
}

export interface BenchProgressEvent {
  current: number;
  total: number;
  label: string;
}

export interface BenchSampleMetric {
  benchmark: 'WebArena';
  id: string;
  subject?: string;
  mode: BenchMode;
  latencyMs: number;
  ttftMs: number;
  tokensPerSecond: number;
  tokenCount: number;
  promptChars: number;
  outputChars: number;
}

export interface CdfPoint {
  value: number;
  cdf: number;
}

export interface BenchSeriesStats {
  sampleCount: number;
  avgLatencyMs: number;
  p50LatencyMs: number;
  p95LatencyMs: number;
  p99LatencyMs: number;
  avgTtftMs: number;
  p50TtftMs: number;
  p95TtftMs: number;
  p99TtftMs: number;
  avgTokensPerSecond: number;
}

export interface BenchSummary {
  benchmark: 'WebArena';
  evalCount: number;
  successCountFlat: number;
  successCountTree: number;
  failureCountFlat: number;
  failureCountTree: number;
  siteBreakdown: Record<string, number>;
  avgTtftMsFlat: number;
  avgTtftMsTree: number;
  ttftSpeedupPct: number;
  avgTokensPerSecondFlat: number;
  avgTokensPerSecondTree: number;
  tpsGainPct: number;
  avgLatencyMsFlat: number;
  avgLatencyMsTree: number;
  speedupPct: number;
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
  occupancySamples: CacheOccupancySample[];
}

export interface CacheOccupancySample {
  label: string;
  mode: BenchMode | 'runtime';
  site: string;
  groupKey: string;
  taskId?: string;
  snapshotTokenBytes: number;
  tierL1Tokens: number;
  tierL2Tokens: number;
  tierL3Tokens: number;
  tierL1Slots: number;
  tierL2Slots: number;
  tierL3Slots: number;
}

export interface BenchFailureRecord {
  mode: BenchMode;
  taskId: string;
  site: string;
  groupKey: string;
  stage: string;
  message: string;
}

export interface BenchDiagnostics {
  runtimeRestartCount: number;
  timeoutFailureCount: number;
  abortFailureCount: number;
  disposedFailureCount: number;
  otherFailureCount: number;
  failureRecords: BenchFailureRecord[];
}

export interface BenchReport {
  modelUrl: string;
  config: BenchConfig;
  webarena?: BenchSummary;
  cacheProfile: CacheProfile;
  diagnostics: BenchDiagnostics;
}
