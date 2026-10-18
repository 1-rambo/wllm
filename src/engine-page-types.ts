/**
 * Browser-facing engine page contracts.
 *
 * These types describe the narrow API a page exposes to an application
 * adapter or browser smoke. They are deliberately independent of any
 * benchmark harness so production engine pages do not depend on evaluation
 * code.
 */

import type { LeaseDiagnostics } from './contracts';

export type BrowserEngineName = 'wllm' | 'wllama';

export interface BrowserEngineFrameMetrics {
  source: string;
  frameCount: number;
  pageOwned?: boolean;
  p50Ms: number;
  p95Ms: number;
  p99Ms: number;
  maxMs: number;
  over33Ratio: number;
  over50Ratio: number;
  longestStallMs: number;
  refreshHz?: number;
}

export interface BrowserEngineSessionOptions {
  sessionId: string;
  /** Optional legacy workload label retained for existing engine-page tests. */
  site?: string;
  pageId?: string;
}

export interface BrowserEngineGenerateRequest {
  requestId: string;
  promptText: string;
  maxNewTokens: number;
  seed?: number;
  temperature?: number;
  nodeId?: string;
  forkFromNodeId?: string;
  signal?: AbortSignal;
}

export type BrowserEngineStopReason = 'stop' | 'length' | 'cancelled';

export interface BrowserEngineGeneration {
  requestId: string;
  text: string;
  ttftMs: number | null;
  decodeMs: number;
  decodeTokens: number;
  firstChunkTokens?: number | null;
  prefillTokens: number | null;
  cacheTokens?: number | null;
  prefillProcessedTokens?: number | null;
  units: number | null;
  stopReason: BrowserEngineStopReason;
}

export interface BrowserEngineMetrics {
  frame?: BrowserEngineFrameMetrics | null;
  engine: BrowserEngineName;
  decodeTokensPerSecond: number | null;
  prefillTokensPerSecond: number | null;
  ttftMs: number | null;
  unitsPerRequest: number | null;
  usedJsHeapBytes: number | null;
  deviceMemoryBytes: number | null;
  lease?: LeaseDiagnostics | null;
}
