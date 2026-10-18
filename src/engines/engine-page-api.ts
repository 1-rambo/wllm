/**
 * Page-side contract for browser engine pages.
 *
 * A per-tab engine exposes only this narrow surface. The shared coordinator
 * remains behind wllm's page API, so a per-tab baseline cannot accidentally
 * reach multi-session scheduling or state policy.
 */

import type {
  BrowserEngineGeneration,
  BrowserEngineGenerateRequest,
  BrowserEngineMetrics,
  BrowserEngineName,
  BrowserEngineSessionOptions,
} from '../engine-page-types';
import type { StateTransitionEvent } from '../transport/protocol';

export interface EnginePageStatus {
  engine: BrowserEngineName;
  engineVersion: string;
  ready: boolean;
  loading: boolean;
  error: string | null;
  loadProgress: number | null;
  modelUrl: string | null;
  sessions: readonly string[];
  // Sessions with a request running right now. The harness uses this to know
  // that a cancellation really interrupted work instead of arriving early.
  activeRequests: readonly string[];
  /** Shared-coordinator globals exposed on this page, if any. */
  coordinatorGlobals: readonly string[];
  /** Optional requested priority mode; disabled means writes are audited but ignored. */
  priorityMode?: 'enabled' | 'disabled';
  /** Initial explicit priority for sessions created on this page. */
  initialPriority?: SessionPriority;
}

export interface EngineProgressEvent {
  sessionId: string;
  requestId: string;
  type: 'token' | 'complete' | 'failed' | 'state';
  state?: StateTransitionEvent;
  /** Text emitted by this event. Present on token events; absent on state,
   *  completion, and failure events, whose payload is reported elsewhere. */
  text?: string;
  decodeGenerated: number;
  timestampMs: number;
  error?: string;
}

export interface EnginePageApi {
  status(): EnginePageStatus;
  prewarm(): Promise<void>;
  createSession(options: BrowserEngineSessionOptions): Promise<void>;
  generate(
    sessionId: string,
    request: BrowserEngineGenerateRequest
  ): Promise<BrowserEngineGeneration>;
  cancel(sessionId: string): Promise<void>;
  metrics(): Promise<BrowserEngineMetrics>;
  disposeSession(sessionId: string): Promise<void>;
  /**
   * Optional engine-neutral progress stream. The harness subscribes before
   * starting a request so click-to-next-token comes from a real token event.
   */
  subscribeProgress?(
    listener: (event: EngineProgressEvent) => void
  ): () => void;
  /**
   * Session-level priority is the scheduler-facing contract. Callers decide
   * which sessions are `high`; the runtime never derives the level from tab
   * visibility, and any number of sessions may hold `high` at once.
   */
  setSessionPriority?(
    sessionId: string,
    level: SessionPriority
  ): Promise<PriorityUpdate>;
  /** Optional bounded audit log for explicit priority writes. */
  priorityLog?(): readonly PriorityAuditEntry[];
}

export type SessionPriority = 'high' | 'normal';

export interface PriorityUpdate {
  accepted: boolean;
  /** False when the page accepted the write but policy intentionally ignored it. */
  applied?: boolean;
  epoch: number;
  level: SessionPriority;
  atMs: number;
}

export interface PriorityAuditEntry extends PriorityUpdate {
  sessionId: string;
}

export const ENGINE_PAGE_GLOBAL = '__wllmEnginePage';

/**
 * Globals the serving demo installs on its own page. A baseline page that
 * carries any of them is not a baseline, so the check is explicit.
 */
export const COORDINATOR_GLOBALS: readonly string[] = Object.freeze([
  '__wllmCoordinatorDiagnostics',
  '__wllmCoordinatorDiagnosticsProvider',
  '__wllmBurstPolicy',
  '__wllmBurstPolicyHistory',
  '__wllmRenderAware',
  '__wllmRenderPressure',
  '__wllmFrameStats',
]);

export function presentCoordinatorGlobals(
  scope: Record<string, unknown>
): string[] {
  return COORDINATOR_GLOBALS.filter((name) => scope[name] !== undefined);
}

export function installEnginePage(
  scope: Record<string, unknown>,
  api: EnginePageApi
): void {
  scope[ENGINE_PAGE_GLOBAL] = api;
}

export function readNumberParam(
  params: URLSearchParams,
  key: string,
  fallback: number
): number {
  const raw = params.get(key);
  if (raw === null || raw === '') {
    return fallback;
  }
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export function readBooleanParam(
  params: URLSearchParams,
  key: string,
  fallback: boolean
): boolean {
  const raw = params.get(key);
  if (raw === null || raw === '') {
    return fallback;
  }
  return raw !== '0' && raw !== 'false';
}

export function round2(value: number): number {
  return Math.round(value * 100) / 100;
}
