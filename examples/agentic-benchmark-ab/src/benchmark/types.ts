export type EvalType = 'string_match' | 'program_html' | 'url_match' | 'unknown';

export interface BenchmarkTask {
  id: string;
  dataset: string;
  intent: string;
  startUrl?: string;
  openUrls: string[];
  requireReset: boolean;
  templateId?: string;
  evalTypes: EvalType[];
  referenceHints: string[];
}

export interface PerfMetrics {
  ttftMs: number;
  tokensPerSecond: number;
  outputTokens: number;
  sharedPrefixChars: number;
  nReused: number;
  latencyMs: number;
}

export interface TaskRunResult {
  task: BenchmarkTask;
  mode: 'A' | 'B';
  metrics: PerfMetrics;
}

export interface ABTaskDelta {
  task: BenchmarkTask;
  a: PerfMetrics;
  b: PerfMetrics;
  aOutputPreview: string;
  bOutputPreview: string;
  ttftGainPct: number;
  tpsGainPct: number;
  isValid: boolean;
  invalidReasons: string[];
}

export interface ABInvalidTask {
  taskId: string;
  dataset: string;
  reasons: string[];
  aOutputPreview: string;
  bOutputPreview: string;
}

export interface ABTaskOutputEvent {
  modelId: string;
  taskId: string;
  dataset: string;
  mode: 'A' | 'B';
  outputTokens: number;
  outputPreview: string;
}

export interface ABExperimentResult {
  modelId: string;
  modelPath: string;
  totalTasks: number;
  avgA: PerfMetrics;
  avgB: PerfMetrics;
  avgTtftGainPct: number;
  avgTpsGainPct: number;
  validTaskCount: number;
  invalidTaskCount: number;
  invalidTaskIds: string[];
  invalidTasks: ABInvalidTask[];
  deltas: ABTaskDelta[];
}

export interface ModelSpec {
  id: string;
  fileName: string;
}

export interface ABProgress {
  modelId: string;
  mode: 'A' | 'B' | 'warmup';
  phase: 'loading-model' | 'warming-prefix-nodes' | 'running-task' | 'done';
  current: number;
  total: number;
  taskId?: string;
}

export interface PrefixConfig {
  systemPrompt: string;
  includeOpenUrls: boolean;
  includeStartUrl: boolean;
  webContentTemplate: string;
  nPredict: number;
  nCtx: number;
  nBatch: number;
  memoryCapBytes: number;
}
