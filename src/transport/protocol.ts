import type {
  ClientId,
  GpuBurstPolicy,
  LifecycleRecord,
  ExecutionMode,
  LeaseDiagnostics,
  NativeRuntimeEvent,
  NativeTokenChunk,
  NativeUnitProgress,
  NativeUnitBudget,
  NativeUnitStop,
  RenderPressureSignal,
  RequestId,
  RequestRecord,
  ResidencyRecord,
  SessionKey,
} from '../contracts';

export interface MessagePortLike {
  postMessage(message: unknown, transfer?: Transferable[]): void;
  addEventListener(
    type: 'message',
    listener: (event: { data: unknown }) => void
  ): void;
  start?(): void;
  close?(): void;
}

export type CoordinatorRequest =
  | {
      type: 'attach';
      callbackId: string;
      clientId: ClientId;
      lifecycle: LifecycleRecord;
    }
  | {
      type: 'detach';
      callbackId: string;
      clientId: ClientId;
    }
  | {
      type: 'lifecycle';
      callbackId: string;
      record: LifecycleRecord;
    }
  | {
      type: 'render.signal';
      callbackId: string;
      signal: RenderPressureSignal;
    }
  | {
      type: 'submit';
      callbackId: string;
      request: RequestRecord;
    }
  | {
      type: 'cancel';
      callbackId: string;
      requestId: RequestId;
    }
  | {
      type: 'status';
      callbackId: string;
      requestId: RequestId;
    }
  | {
      type: 'pump';
      callbackId: string;
      maxSteps?: number;
    }
  | {
      type: 'diagnostics';
      callbackId: string;
    };

export interface CoordinatorDiagnostics {
  completedTier: 'L1' | 'L2' | 'L3' | 'rebuild';
  runtime: {
    nativeBound: boolean;
    // Where the native inference runtime lives. Page-hosted workers can be
    // deprioritized with their owner tab; the shared worker remains attached
    // to whichever app tab currently has a visible client.
    nativeWorkerHost?: 'shared-worker' | 'page';
    // Stable identity of the SharedWorker instance, used to prove that all
    // application pages reached the same coordinator.
    coordinatorInstanceId?: string;
    inFlight: boolean;
    activeRequestId: string | null;
    pendingRequests: number;
    pendingForeground: boolean;
    renderAware: boolean;
    gpuBurstPolicy: GpuBurstPolicy;
    // which controller shapes units: bounded | unsliced | deadline
    executionMode: ExecutionMode;
    // active deadline lease, null for the fixed modes
    lease: LeaseDiagnostics | null;
    // Native counters from the most recently completed unit, available in
    // every execution mode. This is what proves whether a warm session
    // actually reused its prefix instead of only looking fast.
    lastNativeUnit: {
      batchSize: number;
      elapsedMs: number;
      promptTokens: number | null;
      cacheTokens: number | null;
      promptMs: number | null;
      predictedTokens: number | null;
      predictedMs: number | null;
      unitStop: NativeUnitStop | null;
    } | null;
    // the gate is a separate lever from the policy level, so the experiment
    // needs to read it independently of the burst level it is derived from
    renderGate: { enabled: boolean; open: boolean };
    // the effective unit shape, so a harness can tell when a reconfiguration
    // has actually reached the coordinator it shares with other tabs
    execution: {
      maxPrefillTokens: number;
      maxDecodeTokens: number;
      foregroundDecodeTokens: number;
      backgroundDecodeTokens: number;
      backgroundPrefillTokens: number | null;
      targetUnitWallTimeMs: number;
      adaptiveUnits: boolean;
      nativeTimeBudgetEnabled: boolean;
    };
    completedRequests: string[];
    cancelledRequests: string[];
    lastError: string | null;
  };
  recovery: {
    scanned: number;
    recovered: number;
    skipped: number;
    errors: string[];
  };
  // Native-heap accounting sampled when diagnostics are read. This is the
  // bound the memory regression checks look at: live KV tokens, parked KV
  // tokens, parked bytes, and tier-three bytes.
  state: {
    l1Tokens: number;
    l2Tokens: number;
    l3Bytes: number;
    residentNodes: number;
    trackedNodes?: number;
    parkedNodes: number;
    parkedBytes: number;
  };
  // Complete current residency snapshot. Unlike stateHistory this is not a
  // bounded event window, so a harness can still inspect a node after long
  // multi-session pressure has pushed older transitions out of the log.
  residencies: ResidencyRecord[];
  stateHistory: StateTransitionEvent[];
}

export interface StateTransitionEvent {
  sequence: number;
  timestampMs: number;
  clientId: ClientId;
  sessionId: string;
  nodeId: string;
  action:
    | 'account'
    | 'admit'
    | 'restore'
    | 'park'
    | 'evict-on-pause'
    | 'drop';
  level: 'L1' | 'L2' | 'L3' | 'rebuild';
  previousLevel?: 'L1' | 'L2' | 'L3' | 'rebuild';
  admissionPolicy?:
    | 'monotonic-bounded'
    | 'reactive'
    | 'immediate-reject';
  admissionSuccess?: boolean;
  admissionReason?: string;
  steps?: number;
  victims?: string[];
  tokenCount: number;
  bufferBytes: number;
  hostBytes: number;
  totals: { l1Tokens: number; l2Tokens: number; l3Bytes: number };
  residentNodes: number;
  trackedNodes?: number;
  parkedNodes: number;
  parkedBytes: number;
}

export type CoordinatorResponse =
  | {
      type: 'response';
      callbackId: string;
      ok: true;
      result: unknown;
    }
  | {
      type: 'response';
      callbackId: string;
      ok: false;
      error: string;
    }
  | {
      type: 'event';
      event: CoordinatorEvent;
    };

export type CoordinatorEvent =
  | NativeRuntimeEvent
  | {
      type: 'burst';
      clientId: ClientId;
      policy: GpuBurstPolicy;
      reason: 'pressure' | 'recovery' | 'stale' | 'initial';
    }
  | {
      type: 'unit';
      clientId: ClientId;
      sessionId: string;
      requestId: RequestId;
      progress: NativeUnitProgress;
      // the per-unit budget this outcome ran with, after adaptive sizing
      budget: NativeUnitBudget;
    }
  | {
      // A native streamed text chunk. This arrives while the unit is still
      // running, so the page can timestamp the first token and render
      // progressively without changing the unit-level scheduling boundary.
      type: 'token';
      clientId: ClientId;
      sessionId: string;
      requestId: RequestId;
      text: string;
      decodeGenerated: number;
      elapsedMs: number;
    }
  | {
      type: 'complete';
      clientId: ClientId;
      sessionId: string;
      requestId: RequestId;
      output: string;
    }
  | {
      type: 'fenced';
      clientId: ClientId;
      sessionId: string;
      requestId: RequestId;
    }
  | {
      type: 'failed';
      clientId: ClientId;
      sessionId: string;
      requestId: RequestId;
      error: string;
    }
  | {
      // bounded serving: where a session node currently lives (L1 live KV,
      // L2 parked in memory, L3 parked in storage)
      type: 'residency';
      clientId: ClientId;
      sessionId: string;
      requestId: RequestId;
      nodeId: string;
      level: 'L1' | 'L2' | 'L3' | 'rebuild';
      slotId: number | null;
      tokenCount: number;
      bufferBytes: number;
      // of the bytes above, the part the coordinator holds as a host copy.
      // Tier two parks state in the native heap, so this stays zero.
      hostBytes: number;
      totals: { l1Tokens: number; l2Tokens: number; l3Bytes: number };
      state: StateTransitionEvent;
    };

export interface CoordinatorApi {
  attach(clientId: ClientId, lifecycle: LifecycleRecord): Promise<void>;
  detach(clientId: ClientId): Promise<void>;
  updateLifecycle(record: LifecycleRecord): Promise<boolean>;
  updateRenderPressure(
    signal: RenderPressureSignal
  ): Promise<GpuBurstPolicy>;
  submit(request: RequestRecord): Promise<{ accepted: boolean; reason?: string }>;
  cancel(requestId: RequestId): Promise<boolean>;
  status(requestId: RequestId): Promise<RequestRecord | null>;
  pump(maxSteps?: number): Promise<number>;
  diagnostics(): Promise<CoordinatorDiagnostics>;
}

export interface NativeWorkerRequest {
  type: 'native.call';
  callbackId: string;
  method:
    | 'submit'
    | 'prewarm'
    | 'setGpuBurstPolicy'
    | 'advance'
    | 'status'
    | 'cancel'
    | 'interrupt'
    | 'slotExport'
    | 'slotSnapshot'
    | 'slotImport'
    | 'slotPark'
    | 'slotParkCheckpoint'
    | 'slotReadParked'
    | 'slotReleaseParked'
    | 'slotDrop'
    | 'slotStats';
  args: unknown[];
}

export interface NativeWorkerResponse {
  type: 'native.result';
  callbackId: string;
  ok: boolean;
  result?: unknown;
  error?: string;
}

export type NativeWorkerEvent =
  | {
      type: 'native.event';
      event: NativeRuntimeEvent;
    }
  | {
      type: 'native.chunk';
      event: NativeTokenChunk;
    };

export interface SessionProtectionSnapshot {
  protectedSessions: SessionKey[];
}
