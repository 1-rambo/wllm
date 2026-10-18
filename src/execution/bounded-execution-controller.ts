import type {
  ExecutionConfig,
  ExecutionOutcome,
  NativeUnitBudget,
  NativeUnitProgress,
  RequestRecord,
} from '../contracts';
import type { ExecutionLease } from '../contracts';
import { WllmError } from '../contracts';
import type { NativeEngine } from '../native/native-engine';

const DEFAULT_CONFIG: ExecutionConfig = {
  maxPrefillTokens: 128,
  baseDecodeTokens: 32,
  foregroundDecodeTokens: 32,
  backgroundDecodeTokens: 8,
  prefillMsPerToken: 2,
  maxUnitWallTimeMs: 50,
  unitWallTimeGraceMs: 120,
  adaptiveUnits: true,
  targetUnitWallTimeMs: 30,
  nativeTimeBudgetEnabled: true,
  minDecodeTokens: 1,
  maxDecodeTokens: 128,
};

export type MacrotaskYield = () => Promise<void>;

export interface ExecutionControllerHooks {
  // Called after a request has a native id but before its first advance.
  // This is the first point where an activation preemption can target a
  // batch that was already admitted while the promotion was arriving.
  onNativeSubmit?: (request: RequestRecord) => void;
  onBoundary?: (
    request: RequestRecord,
    progress: Awaited<ReturnType<NativeEngine['advance']>>
  ) => Promise<void>;
}

export class BoundedExecutionController {
  readonly config: ExecutionConfig;

  constructor(
    private readonly native: NativeEngine,
    config: Partial<ExecutionConfig> = {},
    private readonly yieldMacrotask: MacrotaskYield = defaultYield,
    private readonly hooks: ExecutionControllerHooks = {}
  ) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  configure(config: Partial<ExecutionConfig>): void {
    Object.assign(this.config, config);
  }

  async runUnit(
    request: RequestRecord,
    foregroundPending: boolean,
    // FrameDeadlineLease output for this unit. When absent the controller
    // uses the deployment's fixed shape, which is what the bounded arm uses.
    lease?: ExecutionLease
    ,
    // A hidden tab whose activation still owes a token: it keeps the
    // foreground-style ceiling instead of the hidden slice.
    handoff = false
    ,
    // An answered promotion still inside its priority serving window: it
    // keeps the deadline-shaped unit budget while it is not foreground.
    priority = false
  ): Promise<ExecutionOutcome> {
    if (!request.nativeRequestId) {
      const replayText = request.replayText;
      const promptText =
        replayText && request.promptText
          ? appendPromptText(replayText, request.promptText)
          : request.promptText;
      const promptTokens = replayText
        ? [...(request.replayTokens ?? []), ...request.promptTokens]
        : request.promptTokens;
      await this.native.submit({
        requestId: request.requestId,
        promptTokens,
        maxNewTokens: request.maxNewTokens,
        ...(promptText === undefined ? {} : { promptText }),
        ...(request.seed === undefined ? {} : { seed: request.seed }),
        ...(request.temperature === undefined
          ? {}
          : { temperature: request.temperature }),
        ...(request.resumeSlotId === undefined || request.resumeSlotId === null
          ? {}
          : { resumeSlotId: request.resumeSlotId }),
        ...(request.replaceContext === true
          ? { replaceContext: true }
          : {}),
        // a request that has not done any native work yet and resumes a
        // parked slot is a new turn, not a continuation of an old one
        ...(request.resumeSlotId !== undefined &&
        request.resumeSlotId !== null &&
        request.prefillProcessed === 0 &&
        request.decodeGenerated === 0
          ? { newTurn: true }
          : {}),
      });
      request.nativeRequestId = request.requestId;
      request.resumeSlotId = null;
      this.hooks.onNativeSubmit?.(request);
    }

    const budget = this.makeBudget(
      request,
      foregroundPending,
      lease,
      handoff,
      priority
    );
    const decodedBefore = request.decodeGenerated;
    const prefillBefore = request.prefillProcessed;
    const progress = await this.native.advance(request.nativeRequestId, budget);
    if (progress.isError) {
      throw new WllmError(
        `native advance failed for ${request.requestId}`,
        'native_error'
      );
    }

    this.observeUnit(request, progress, decodedBefore, prefillBefore);

    request.prefillProcessed = progress.prefillProcessed;
    if (progress.prefillCached !== undefined) {
      request.prefillCached = progress.prefillCached;
    }
    if (progress.residentTokens !== undefined) {
      request.residentTokens = progress.residentTokens;
    }
    if (progress.boundedStops !== undefined) {
      request.boundedStops = progress.boundedStops;
    }
    if (progress.unitStop !== undefined) {
      request.unitStop = progress.unitStop;
    }
    request.decodeGenerated = progress.decodeGenerated;
    request.tokenAccounting = progress.tokenAccounting;
    if (progress.counters) {
      request.counters = { ...progress.counters };
    }
    request.stage = progress.stage === 'done' ? 'decode' : progress.stage;
    request.serviceMs += progress.elapsedMs;
    request.outputChunks.push(...progress.outputChunks);
    request.slotId = progress.slotId;

    const completed = progress.done && !progress.cancelled;
    const requeued = !completed;
    if (requeued) {
      await this.hooks.onBoundary?.(request, progress);
      await this.yieldMacrotask();
    }

    return {
      request,
      progress,
      budget,
      requeued,
      completed,
    };
  }

  // Turn the throughput llama.cpp reported for this unit into an estimate the
  // next unit sizes itself against. A unit that also evaluated prompt tokens
  // mixes prefill and decode, so it only contributes a decode sample when the
  // native counters isolate the prediction time.
  private observeUnit(
    request: RequestRecord,
    progress: NativeUnitProgress,
    decodedBefore: number,
    prefillBefore: number
  ): void {
    const decoded = progress.decodeGenerated - decodedBefore;
    const counters = progress.counters;
    if (decoded > 0) {
      const nativePredictedMs = counters?.predictedMs;
      const hasNativeTiming =
        (counters?.predictedN ?? 0) > 0 &&
        nativePredictedMs !== undefined &&
        nativePredictedMs > 0;
      const evaluatedPrompt =
        (counters?.promptN ?? 0) > 0 ||
        progress.prefillProcessed > prefillBefore;
      const decodeMs = hasNativeTiming
        ? nativePredictedMs
        : evaluatedPrompt
          ? 0
          : progress.elapsedMs;
      if (decodeMs > 0) {
        request.estimatedDecodeTokensPerMs = ema(
          request.estimatedDecodeTokensPerMs,
          decoded / decodeMs
        );
      }
    }
    if (
      counters !== undefined &&
      counters.promptN > 0 &&
      counters.promptMs !== undefined &&
      counters.promptMs > 0
    ) {
      request.estimatedPrefillMsPerToken = ema(
        request.estimatedPrefillMsPerToken,
        counters.promptMs / counters.promptN
      );
    }
  }

  private makeBudget(
    request: RequestRecord,
    foregroundPending: boolean,
    lease?: ExecutionLease,
    handoff = false
    ,
    priority = false
  ): NativeUnitBudget {
    // A lease replaces the deployment's fixed target for this unit: its
    // budget was derived from presented frames and measured token cost, so
    // the controller only has to keep it inside the watchdog ceiling.
    const targetMs = Math.max(
      1,
      Math.min(
        lease?.budgetMs ?? this.config.targetUnitWallTimeMs,
        this.config.maxUnitWallTimeMs
      )
    );
    const backgroundPrefillClamp =
      foregroundPending &&
      !request.lifecycle.foreground &&
      !handoff &&
      !priority
        ? this.config.backgroundPrefillTokens ?? this.config.backgroundDecodeTokens
        : this.config.maxPrefillTokens;
    // measured prefill cost wins over the configured estimate once a request
    // has evaluated at least one prompt chunk
    const prefillMsPerToken = Math.max(
      0.01,
      request.estimatedPrefillMsPerToken ?? this.config.prefillMsPerToken
    );
    const timePrefillCap = Math.max(1, Math.floor(targetMs / prefillMsPerToken));
    const prefillCeiling = Math.min(
      this.config.maxPrefillTokens,
      lease?.maxPrefillTokens ?? this.config.maxPrefillTokens
    );
    const maxPrefillTokens = Math.max(
      1,
      lease === undefined
        ? Math.min(prefillCeiling, backgroundPrefillClamp, timePrefillCap)
        : Math.min(prefillCeiling, backgroundPrefillClamp)
    );

    // the lifecycle class sets the ceiling; adaptation only moves below it
    const lifecycleCeiling = Math.max(
      1,
      request.lifecycle.foreground || handoff || priority
        ? this.config.foregroundDecodeTokens
        : foregroundPending
          ? this.config.backgroundDecodeTokens
          : this.config.baseDecodeTokens
    );
    const decodeCeiling = Math.max(
      1,
      Math.min(decodeCeilingBase(lease, lifecycleCeiling), this.config.maxDecodeTokens)
    );
    let maxDecodeTokens = decodeCeiling;
    const rate = request.estimatedDecodeTokensPerMs;
    // A lease already sized its token budget from the measured rate, so the
    // adaptive shrink only applies to the fixed bounded shape.
    if (
      lease === undefined &&
      this.config.adaptiveUnits !== false &&
      rate !== undefined &&
      rate > 0
    ) {
      maxDecodeTokens = Math.min(
        decodeCeiling,
        Math.max(this.config.minDecodeTokens, Math.round(rate * targetMs))
      );
    }

    return {
      maxPrefillTokens,
      maxDecodeTokens: Math.max(1, maxDecodeTokens),
      ...(lease ? { leaseId: lease.leaseId, leaseSource: lease.source } : {}),
      // A leased unit is already bounded by maxDecodeTokens. Passing the
      // frame window as a native wall-clock boundary cut batches after the
      // first token once queue/host latency exceeded that window, then the
      // next lease measured that one-token unit and kept shrinking itself.
      // Keep the deployment watchdog as the hard guard, but let the lease's
      // token budget be the cooperative boundary.
      ...(this.config.nativeTimeBudgetEnabled && lease === undefined
        ? { nativeTimeBudgetMs: targetMs }
        : {}),
      // one slow batch must not turn the watchdog into the unit boundary
      wallTimeMs:
        this.config.maxUnitWallTimeMs +
        Math.max(0, this.config.unitWallTimeGraceMs),
    };
  }
}

async function defaultYield(): Promise<void> {
  await macrotaskYield();
}

// A MessageChannel turn preserves the event-loop boundary the coordinator
// needs without paying the timer clamp that setTimeout(0) introduces in a
// busy page. The lease can return at graph cadence, so those milliseconds
// are on the throughput path; tests and embedded runtimes without the DOM
// channel keep the timer fallback.
const macrotaskYield = createMacrotaskYield();

function createMacrotaskYield(): MacrotaskYield {
  if (typeof MessageChannel === 'undefined') {
    return async () => {
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
    };
  }
  const channel = new MessageChannel();
  const pending: Array<() => void> = [];
  channel.port1.onmessage = () => pending.shift()?.();
  channel.port1.start();
  return () =>
    new Promise<void>((resolve) => {
      pending.push(resolve);
      channel.port2.postMessage(null);
    });
}

// The lease caps how much decode work one unit may admit. The lifecycle class
// stays a hard ceiling on top of it, so a background session never inherits a
// foreground-sized lease.
function decodeCeilingBase(
  lease: ExecutionLease | undefined,
  lifecycleCeiling: number
): number {
  if (!lease) {
    return lifecycleCeiling;
  }
  return Math.min(lease.maxDecodeTokens, lifecycleCeiling);
}

// First sample wins; later samples move the estimate by half. Unit cost moves
// with thermal state and backend scheduling, so the estimate has to track
// that without letting one slow unit collapse the budget to one token.
function ema(previous: number | undefined, sample: number): number {
  return previous === undefined ? sample : previous * 0.5 + sample * 0.5;
}

function appendPromptText(prefix: string, next: string): string {
  if (!prefix) {
    return next;
  }
  if (!next) {
    return prefix;
  }
  // Chat-template markers are boundaries, not words. Adding a space before
  // `<|eot_id|>` or `<|im_end|>` changes the tokenized recovery prompt.
  const separator = next.startsWith(' ') || next.startsWith('<|') ? '' : ' ';
  return `${prefix}${separator}${next}`;
}
