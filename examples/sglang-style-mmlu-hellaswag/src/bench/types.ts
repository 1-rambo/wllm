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

export interface BenchConfig {
  modelUrl: string;
  nCtx: number;
  nBatch: number;
  mmluShots: number;
  hellaShots: number;
  mmluEvalCount: number;
  hellaEvalCount: number;
}

export interface BenchLogEvent {
  text: string;
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

export interface BenchReport {
  modelUrl: string;
  config: BenchConfig;
  mmlu: BenchSummary;
  hella: BenchSummary;
}
