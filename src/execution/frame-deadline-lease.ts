import type {
  ExecutionLease,
  ExecutionLeaseSource,
  FrameHealthTail,
  LeaseDiagnostics,
  LeaseReason,
  QueueLease,
  RenderPressureSignal,
} from '../contracts';

/**
 * FrameDeadlineLease turns presented-frame health and measured native cost
 * into the shape of the next cooperative unit.
 *
 * It converts display cadence and the presented-frame tail into a deadline
 * severity:
 *
 *   frameBudgetMs  = 1000 / refreshHz
 *   tailDeadlineMs = tailBudgetFrameMultiplier * frameBudgetMs
 *   tailSeverity   = (presentedP99 - tailDeadlineMs) / tailDeadlineMs
 *   windowPeriods  = mildPressureFramePeriods
 *                  + (severePressureFramePeriods - mildPressureFramePeriods)
 *                    * clamp(tailSeverity, 0, 0.9)
 *   budgetMs       = min(deploymentBudgetMs * (1 - clamp(tailSeverity, 0, 0.9)),
 *                      windowPeriods * frameBudgetMs)
 *
 * The deployment-wide watchdog is the ceiling, not a scheduling quantum.
 * With no render pressure the lease is work-conserving and spends that full
 * ceiling. Once a deadline is active the window is counted in display
 * periods, not milliseconds: the same lease admits half the wall time on a
 * 120 Hz panel, and the unit returns to the coordinator at frame cadence so
 * the queue it leaves behind is short enough for the next presented frame.
 * Native token cost converts that window into a token budget, and a token
 * floor keeps progress going when one native token already costs more than
 * the whole window.
 *
 * Everything here is pure with respect to the native engine: the controller
 * only consumes samples and produces a lease. That keeps it unit-testable and
 * keeps the failure path (stale signal, background page, queue timeout) a
 * matter of returning a different lease rather than aborting work.
 */

export interface FrameDeadlineLeaseOptions {
  // Display budget used when a signal cannot measure its own refresh rate.
  fallbackRefreshHz: number;
  // Floor and ceiling of a derived budget. The floor is what keeps the lease
  // from starving a request: the native unit still returns at its next token
  // boundary, so a smaller budget only costs unit overhead, never progress.
  minBudgetMs: number;
  maxBudgetMs: number;
  // Safety shape used when the frame signal goes stale or a unit overruns the
  // queue proof. This is the bounded deployment shape, not a fixed quantum.
  fallbackBudgetMs: number;
  // Slice admitted while the page is not foreground.
  backgroundBudgetMs: number;
  // Pressure test: any of these turns the lease into deadline mode.
  pressureP95Headroom: number;
  pressureOver33Ratio: number;
  pressureP99Multiplier: number;
  // Presented tail, in display periods, at which the deadline counts as
  // violated. It scales the severity that shrinks the admitted window and
  // the rest interval that hands the renderer an empty queue.
  tailBudgetFrameMultiplier: number;
  // Display periods admitted per unit while a deadline is active. A tail
  // that is late but still inside the multi-frame deadline admits the mild
  // value; a violated deadline collapses towards the severe value. Both are
  // measured in display periods rather than milliseconds, so the window
  // follows the panel and not a scheduling constant.
  mildPressureFramePeriods: number;
  severePressureFramePeriods: number;
  // Clear test: all of these must hold before the lease goes work-conserving.
  clearP95Headroom: number;
  clearOver33Ratio: number;
  // The page's own frame cadence floor, learned from recent samples, moves
  // the thresholds too: a page that idles at 20 ms P95 on a 60 Hz display
  // is not under GPU contention even though 20 ms is above the panel period.
  pressureRelativeHeadroom: number;
  clearRelativeHeadroom: number;
  baselineSamples: number;
  minBaselineSamples: number;
  // Hysteresis. Downgrade is fast, release needs sustained calm so one lucky
  // sample cannot flip the lease back to the work-conserving shape.
  downgradeSamples: number;
  recoverSamples: number;
  // No frame sample for this long means the page is not reporting; the lease
  // returns to the bounded safety shape instead of guessing.
  staleSignalMs: number;
  // Queue lease. `inflightCap` is the deployment's own maximum; the lease
  // never exceeds it, and derives a smaller value from the budget share.
  inflightCap: number;
  minCommandBatchSize: number;
  maxCommandBatchSize: number;
  // A unit that overshoots its own budget by more than this means GPU queue
  // wait, not token cost, is pacing the unit.
  queueWaitToleranceMs: number;
  // How a presented deadline becomes a rest interval between units. A
  // multi-token pressure batch amortises queue and host round trips, but it
  // can still leave a late tail behind. The default is therefore a short
  // bounded yield derived from the presented debt. Setting this to zero keeps
  // the original no-rest ablation available without changing the batch floor.
  pausePerDebtMs: number;
  maxPauseMs: number;
  // Ablation of the clear-page queue shape. The default keeps the deep
  // submission queue while frames are clear, which is what the presented
  // window is computed against. Setting this keeps the same continuous
  // decode (no rest interval, unit ceiling unchanged) but submits through
  // the local barrier, so the renderer never waits behind more than one
  // command buffer.
  workConservingBarrier: boolean;
  backgroundBarrier: boolean;
  // Shape used while no fresh presented-frame evidence is available. The
  // bounded fallback cannot hold the renderer for long, but it also pays a
  // host round trip every one or two tokens, so a cold start with a silent
  // signal decodes an order of magnitude below the native rate. `barrier`
  // keeps the work-conserving token ceilings and bounds the renderer wait
  // with the local barrier instead, which is the shape a backend without
  // exposed queue completion has to use anyway.
  staleSignalShape: 'bounded' | 'barrier';
  queueBackpressureSamples: number;
  queueReliefSamples: number;
  // Token budgets derived from the lease budget and the measured native cost.
  minDecodeTokens: number;
  // An unanswered click only needs one visible token to prove that the
  // hand-off worked. Keeping the hand-off unit at one token prevents a
  // rapid click from monopolising a full foreground-shaped batch.
  handoffDecodeTokens: number;
  // Background tabs do not need the foreground attention quantum, but a
  // one-token graph pays the host/queue round trip on every token. Keep
  // this floor specific to hidden work so the visible/hand-off path can
  // still return at the attention boundary.
  backgroundDecodeTokens: number;
  maxDecodeTokens: number;
  // A high-priority decode unit must amortise native graph and stream
  // setup. Without this floor, a slow pressure-shaped sample can teach the
  // rate estimator to issue ever-smaller units and make the slowdown
  // self-sustaining. It applies only to the explicit priority lane while
  // frames are healthy; pressure and hand-off leases keep their bounds.
  priorityDecodeTokens: number;
  maxPrefillTokens: number;
  // Lower bound on the pressure decode cap: the lease never admits fewer
  // native tokens than this, so the unit makes progress even when one token
  // costs more than the display window.
  minPressureDecodeTokens: number;
  // Pressure-only caps. A pressured lease stops at graph boundaries: the
  // window divided by the measured native token cost gives the cap, and
  // these bounds keep the unit from vanishing or growing without limit.
  // A prefill graph is atomic once submitted. The motivation study found
  // the fixed graph/host/queue overhead dominates below one 64-token batch,
  // so the pressure path must not collapse this floor to a single token.
  minPressurePrefillTokens: number;
  // Upper bound for a pressure-shaped prefill unit. Within these bounds
  // the lease divides the wall-time window by the measured prefill cost
  // per token, so a bulk prefill can amortise graph overhead without
  // holding a late frame behind a longer command buffer than the window.
  maxPressurePrefillTokens: number;
  // Longest wall time one graph may hold the coordinator before it returns
  // to the scheduler. A lifecycle (attention) change can only be honoured
  // at a unit boundary, so a work-conserving unit that covers the whole
  // response makes that signal unobservable for seconds: the lease is also
  // the mechanism's attention-reaction bound, not only a frame bound.
  attentionQuantumMs: number;
  // Clear pages can afford a longer cooperative burst: activation
  // preemption still cuts the native graph at a token boundary, while the
  // shorter attention quantum is reserved for frame-deadline units.
  workConservingQuantumMs: number;
  // Floor for the quantum converted into tokens. A backend that has not
  // reported a rate yet, or one whose tokens are genuinely expensive, still
  // has to make progress instead of returning at every token.
  minQuantumDecodeTokens: number;
  // Prefill graph floor outside the pressure branches. The 2026-09-24
  // chunk study put the knee at 64 uncached prompt tokens; below it a
  // prefill unit pays the graph and queue round trip per token, which is
  // what made a ten-tab background prefill take seconds instead of one
  // unit. The pressure branches keep their own, measured floor.
  minPrefillTokens: number;
  // Evaluation-only control for the prefill-chunk motivation study. When
  // set, a foreground lease admits exactly this many prompt tokens (up to
  // the deployment ceiling) instead of applying the measured cost cap.
  // Unset in serving, so production behaviour is unchanged unless an
  // experiment explicitly selects a chunk size.
  prefillChunkOverride?: number;
  // Smallest command submission admitted while a deadline is active. Very
  // small submissions cut how long the renderer waits, but every submission
  // is a host round trip; the measured frontier sits near 32.
  minPressureCommandBatchSize: number;
  // Estimates used until a request has reported native counters.
  fallbackDecodeTokensPerMs: number;
  fallbackPrefillMsPerToken: number;
}

type FrameDeadlineLeaseDefaults = Omit<FrameDeadlineLeaseOptions, 'maxBudgetMs'>;

const DEFAULTS: FrameDeadlineLeaseDefaults = {
  fallbackRefreshHz: 60,
  minBudgetMs: 8,
  fallbackBudgetMs: 24,
  backgroundBudgetMs: 8,
  pressureP95Headroom: 0.35,
  pressureOver33Ratio: 0.25,
  pressureP99Multiplier: 2,
  // Five display periods at 60 Hz is 83.33 ms. The pressure trigger still
  // reacts to the shorter two-frame tail, but a unit only becomes more
  // aggressive once the presented tail reaches this wider deadline.
  tailBudgetFrameMultiplier: 5,
  // One display period of GPU work per unit. The mechanism screen fixes
  // this: with five paired trials of the 900-iteration contended schedule,
  // one period measured gpu P95 -1.04 ms against the bounded quantum
  // (CI high +1.23) and -2.51 ms against unsliced (CI high -1.41, i.e.
  // better on every pair), while a two-period window lost its significance
  // against unsliced (CI high +0.04). The knobs stay separate so the
  // wider-window ablation stays a parameter rather than a patch, and they
  // are counted in periods so a 120 Hz panel receives half the wall time
  // without a new constant.
  // Screen 21 re-measured the window with the presented-frame cadence
  // fixed to the real panel. One period is shorter than a native batch on
  // the anchor device (a token costs 25-45 ms under contention), so every
  // unit overran its own window, the occupancy test read that as queue
  // backpressure, and the lease paid unit overhead without buying frames.
  // Two periods is the smallest window that can contain a batch: 0.89-0.95
  // of unsliced clear-phase throughput against 0.80, gpu P95 +0.8 ms
  // against bounded (CI high 1.4) and -2.4 ms against unsliced.
  mildPressureFramePeriods: 2,
  severePressureFramePeriods: 1,
  clearP95Headroom: 0.1,
  // The helper-presented stream commonly has a small number of 30-40 ms
  // intervals while the GPU queue drains. Keep those in the indeterminate
  // band instead of latching deadline mode indefinitely.
  clearOver33Ratio: 0.15,
  // A presented-frame helper occasionally reports one 25-30 ms tail while
  // the page is otherwise clear. That is above the 60 Hz period but not
  // enough evidence to enter deadline mode; the pressure branch also has
  // the over-33-ms and P99 tests for genuinely bad frames.
  pressureRelativeHeadroom: 0.75,
  clearRelativeHeadroom: 0.15,
  baselineSamples: 20,
  minBaselineSamples: 6,
  downgradeSamples: 2,
  recoverSamples: 2,
  staleSignalMs: 1_500,
  inflightCap: 2,
  minCommandBatchSize: 32,
  maxCommandBatchSize: 64,
  // A sub-frame overshoot is scheduling jitter, not proof that the
  // renderer is waiting behind a deep queue. Arming the local barrier on it
  // drains the queue and adds a round trip in the light-pressure regime, so
  // the browser path waits for a half-frame overrun before collapsing to one
  // in-flight batch.
  queueWaitToleranceMs: 8,
  pausePerDebtMs: 1,
  maxPauseMs: 16,
  workConservingBarrier: false,
  backgroundBarrier: true,
  staleSignalShape: 'barrier',
  queueBackpressureSamples: 2,
  queueReliefSamples: 3,
  minDecodeTokens: 1,
  handoffDecodeTokens: 1,
  backgroundDecodeTokens: 1,
  minPressureDecodeTokens: 2,
  priorityDecodeTokens: 64,
  // The 2026-09-24 prefill chunk study measured the knee at 64 uncached
  // prompt tokens: 16 -> 32 -> 64 cut TTFT from 1111 -> 681 -> 509 ms at
  // 2000 triangles, while 128/256 were a plateau. Keep decode frame-sliced;
  // this floor only prevents a prefill graph from paying that overhead per
  // token.
  minPressurePrefillTokens: 64,
  // A larger window can spend the deployment's full prefill chunk; every
  // extra chunk boundary costs a graph restart and a queue round trip. The
  // 64-token floor still keeps a late tail from paying that overhead per
  // token.
  maxPressurePrefillTokens: 256,
  attentionQuantumMs: 200,
  minQuantumDecodeTokens: 4,
  workConservingQuantumMs: 1000,
  minPrefillTokens: 64,
  minPressureCommandBatchSize: 32,
  // Work-conserving ceilings. A clear page is the regime where this engine
  // has to look like a plain one-shot decoder: one native graph may cover
  // the whole response instead of returning to the coordinator every few
  // tokens. The pressure branches derive their own caps from the presented
  // window, so these ceilings say nothing about how much work a contended
  // unit admits.
  maxDecodeTokens: 4_096,
  maxPrefillTokens: 4_096,
  fallbackDecodeTokensPerMs: 0.04,
  fallbackPrefillMsPerToken: 2,
};

export interface NativeUnitCostSample {
  // Lease the unit ran under. Pressure-shaped units are what teach the lease
  // what a native batch costs while the renderer shares the GPU queue.
  reason?: LeaseReason;
  // Native counters for the unit that just finished.
  decodeTokens?: number;
  predictedMs?: number;
  promptTokens?: number;
  promptMs?: number;
  // Native prompt time with queue wait removed. A backend that can report
  // this may teach the prefill cost even while frames are under pressure.
  promptComputeMs?: number;
  // Explicit queue component; when compute time is not supplied it can be
  // subtracted from promptMs to recover the computable portion.
  promptQueueWaitMs?: number;
  // Host-side wall clock for the unit, used for the queue-wait estimate.
  // Overshoot measured by the caller; the lease derives it from elapsedMs
  // and budgetMs when it is absent.
  occupancyOvershootMs?: number;
  elapsedMs?: number;
  budgetMs?: number;
}

export interface LeaseRequest {
  now: number;
  foreground: boolean;
  // Explicit background status. In the priority API, a `normal` session is
  // still runnable work; only an actually hidden/frozen context should take
  // the background slice. When omitted, keep the legacy foreground-derived
  // behavior for callers that have not migrated yet.
  background?: boolean;
  // true when another foreground session is waiting, used to keep background
  // work from taking the whole budget.
  backgroundPending?: boolean;
  // A recently activated tab that has not produced its next token yet. It
  // is not visible any more, but the human is still waiting on the click
  // that opened it, so it gets the foreground-style unit shape instead of
  // the hidden tab's small slice.
  handoff?: boolean;
  // An answered promotion can outlive the foreground edge that created it.
  // It remains in the priority lane until its serving window ends, so it
  // must retain the deadline-shaped work budget instead of falling back to
  // the one-token background slice.
  priority?: boolean;
}

type LeaseRegime = 'work-conserving' | 'deadline';

interface FrameSample {
  at: number;
  p95Ms: number;
  p99Ms: number;
  over33Ratio: number;
  over50Ratio: number;
  longestStallMs: number;
  refreshHz: number | null;
  frameCount: number;
}

const MAX_SAMPLES = 5;

// A one-token or otherwise tiny prompt graph is dominated by its fixed
// graph/host cost. Dividing that sample by one token teaches the lease a
// fictitious per-token cost, so only a real batch may update the estimate.
const MIN_PREFILL_COST_SAMPLE_TOKENS = 16;

export class FrameDeadlineLease {
  readonly options: FrameDeadlineLeaseOptions;
  private readonly samples: FrameSample[] = [];
  private readonly baselineP95: number[] = [];
  private pressureSamples = 0;
  private clearSamples = 0;
  private regime: LeaseRegime = 'work-conserving';
  private lastSignalAt = 0;
  private lastLease: ExecutionLease | null = null;
  private nextLeaseId = 1;
  private decodeTokensPerMs: number | null = null;
  // Per-token wall cost measured under a pressure lease. The next window has
  // to cover at least one native batch at this cost, so it is kept apart
  // from the clear-conditioned rate above.
  private pressureMsPerToken: number | null = null;
  private prefillMsPerToken: number | null = null;
  private queueWaitMs: number | null = null;
  // Rolling GPU residency overshoot: how long pressure-shaped units kept the
  // queue past the window they were admitted under. The browser path cannot
  // read a completion-wait counter, so this is the occupancy signal the queue
  // barrier is decided from.
  private occupancyOvershootMs: number | null = null;
  private queueWaitStreak = 0;
  private queueReliefStreak = 0;
  private queueBarrier = false;
  // Whether the newest page-owned frame sample was late. A unit that overran
  // its own window is only queue evidence while the page is reporting a late
  // presented frame; without that corroboration the local barrier would arm
  // on the engine's own queue depth.
  private lastSampleStressed = false;
  // Whether the newest sample was clear enough to release the barrier. The
  // regime label has two-sample hysteresis; without this separate bit the
  // barrier stayed armed after the page had already recovered, serializing
  // decode units and capping throughput.
  private lastSampleClear = false;
  private recoveries = 0;
  private lastRecoveryAt: number | null = null;
  private lastRecoveryReason: LeaseReason | null = null;

  constructor(
    options: Partial<Omit<FrameDeadlineLeaseOptions, 'maxBudgetMs'>> & {
      // The deployment watchdog, not a scheduling quantum.
      maxBudgetMs: number;
    }
  ) {
    this.options = {
      ...DEFAULTS,
      ...options,
      maxBudgetMs: Math.max(1, options.maxBudgetMs),
    };
  }

  /**
   * Record one presented-frame sample. Called while a unit is running, so the
   * next graph boundary sees the newest health without waiting for the
   * request to finish.
   */
  observe(signal: RenderPressureSignal, now: number): void {
    if (
      !Number.isFinite(signal.p95Ms) ||
      !Number.isFinite(signal.p99Ms ?? signal.p95Ms)
    ) {
      return;
    }
    const idle = signal.frameCount < 2;
    // A sample the monitor produced from its own helper canvas keeps the
    // stall detector alive, but it is not evidence that a renderer is
    // competing: with no application canvas presenting, the helper can only
    // show how deep this engine's own queue is, which is what the
    // work-conserving regime exists for. Only a page-owned sample feeds the
    // pressure decision.
    this.lastSignalAt = now;
    if (signal.pageOwned === false) {
      this.pressureSamples = 0;
      this.clearSamples += 1;
      // The page canvas is not presenting at all, so any earlier late-frame
      // evidence is no longer corroborated by what the monitor can see.
      this.lastSampleStressed = false;
      this.lastSampleClear = true;
      if (this.clearSamples >= this.options.recoverSamples) {
        if (this.regime !== 'work-conserving') {
          this.pressureMsPerToken = null;
        }
        this.regime = 'work-conserving';
      }
      return;
    }
    const tail: FrameHealthTail = idle
      ? {
          p99Ms: 0,
          over33Ratio: 0,
          over50Ratio: 0,
          longestStallMs: 0,
          ...(signal.refreshHz === undefined
            ? {}
            : { refreshHz: signal.refreshHz }),
        }
      : extendedTail(signal);
    this.samples.push({
      at: now,
      p95Ms: idle ? 0 : signal.p95Ms,
      p99Ms: tail.p99Ms,
      over33Ratio: tail.over33Ratio,
      over50Ratio: tail.over50Ratio,
      longestStallMs: tail.longestStallMs,
      refreshHz: finiteHz(signal.refreshHz),
      frameCount: signal.frameCount,
    });
    while (this.samples.length > MAX_SAMPLES) {
      this.samples.shift();
    }

    const frameBudgetMs = this.frameBudgetMs();
    const smoothed = smoothedSample(this.samples);
    if (!smoothed) {
      return;
    }
    const baselineP95Ms = this.learnedBaselineP95Ms();
    // The threshold is the larger of the display budget and the page's own
    // cadence floor, so a busy-but-uncontended page is not misread as
    // contention and a fast panel is not excused by its small frame period.
    const pressureP95Ms = Math.max(
      frameBudgetMs * (1 + this.options.pressureP95Headroom),
      baselineP95Ms * (1 + this.options.pressureRelativeHeadroom)
    );
    const clearP95Ms = Math.max(
      frameBudgetMs * (1 + this.options.clearP95Headroom),
      baselineP95Ms * (1 + this.options.clearRelativeHeadroom)
    );
    // Fast attack, slow release: the stress test looks at the worst of the
    // last two samples, so a step into contention is caught on the second
    // sample, while the clear test uses the median and needs sustained calm.
    const attack = worstOfLast(this.samples, 2);
    // A late p99 alone is not contention evidence: a single 30-40 ms outlier
    // from the page's own presenter lands in the tail while the page is
    // otherwise presenting at panel cadence. Entering deadline mode on that
    // sample slices decode and inserts a rest interval the baseline never
    // pays, which reads as a decoder gap that is really a scheduler gap.
    // The p99 branch therefore needs corroboration: the same attack window
    // has to show a raised median or over-33 share, or a stall that spans
    // the whole multi-frame deadline.
    const p99Late =
      attack.p99Ms >
      Math.max(
        frameBudgetMs * this.options.pressureP99Multiplier,
        pressureP95Ms
      );
    const p99Corroborated =
      attack.p95Ms > frameBudgetMs * (1 + this.options.clearP95Headroom) ||
      attack.over33Ratio > this.options.clearOver33Ratio ||
      attack.longestStallMs >
        frameBudgetMs * this.options.tailBudgetFrameMultiplier;
    const stressed =
      !idle &&
      (attack.p95Ms > pressureP95Ms ||
        attack.over33Ratio > this.options.pressureOver33Ratio ||
        (p99Late && p99Corroborated));
    const clear =
      idle ||
      (smoothed.p95Ms <= clearP95Ms &&
        smoothed.over33Ratio <= this.options.clearOver33Ratio);

    if (!idle) {
      // A page can need its own cadence floor even when it is not GPU
      // contended: captureStream and compositor jitter can put a few frames
      // slightly above the 60 Hz frame period without an over-33 tail. Accept such a sample
      // only when its over-33 share and P99 stay below the pressure tests,
      // so a genuinely slow cold-start load still cannot teach itself as
      // the page's uncontended baseline.
      const baselineEligible =
        signal.p95Ms <=
          frameBudgetMs * (1 + this.options.pressureRelativeHeadroom) &&
        tail.over33Ratio <= this.options.pressureOver33Ratio &&
        tail.p99Ms <=
          Math.max(
            frameBudgetMs * this.options.pressureP99Multiplier,
            frameBudgetMs * (1 + this.options.pressureP95Headroom)
          );
      if (baselineEligible) {
        this.baselineP95.push(signal.p95Ms);
        while (this.baselineP95.length > this.options.baselineSamples) {
          this.baselineP95.shift();
        }
      }
    }
    this.lastSampleStressed = stressed;
    this.lastSampleClear = clear;

    if (stressed) {
      this.clearSamples = 0;
      this.pressureSamples += 1;
      if (this.pressureSamples >= this.options.downgradeSamples) {
        this.regime = 'deadline';
      }
    } else if (clear) {
      this.pressureSamples = 0;
      this.clearSamples += 1;
      if (this.clearSamples >= this.options.recoverSamples) {
        if (this.regime !== 'work-conserving') {
          // The contended cost describes a queue the renderer is sharing. A
          // fresh pressure episode starts from a different queue state, so
          // the next window is measured again instead of inheriting it.
          this.pressureMsPerToken = null;
        }
        this.regime = 'work-conserving';
      }
    } else {
      // indeterminate band: hold the current regime rather than oscillate
      this.clearSamples = 0;
    }
  }

  /**
   * Feed the native counters of one finished unit back into the cost estimate
   * and the queue-wait detector.
   */
  observeUnit(sample: NativeUnitCostSample): void {
    const decoded = sample.decodeTokens ?? 0;
    const pressureShaped =
      sample.reason === 'frame-deadline' ||
      sample.reason === 'queue-backpressure';
    const reliableNativeTiming = hasReliableNativeTiming(
      decoded,
      sample.predictedMs,
      sample.elapsedMs
    );
    if (
      pressureShaped &&
      reliableNativeTiming &&
      sample.predictedMs !== undefined &&
      sample.predictedMs > 0
    ) {
      // What one native batch costs while the renderer is on the queue.
      // This is a measured completion cost, not the clear-conditioned rate:
      // a contended queue inflates the same batch by its wait time, and the
      // next window has to cover it or the unit is cut mid-batch.
      // Native timings occasionally report a counter-reset value such as
      // 0.001 ms for a token after a bounded stop. Feeding that into an EMA
      // makes the next unit look free and can admit the full decode ceiling.
      // Accept only timings that are physically plausible against the
      // host-observed unit duration; otherwise retain the previous estimate.
      this.pressureMsPerToken = ema(
        this.pressureMsPerToken,
        sample.predictedMs / decoded
      );
    }
    if (reliableNativeTiming && sample.predictedMs !== undefined) {
      this.decodeTokensPerMs = ema(
        this.decodeTokensPerMs,
        decoded / sample.predictedMs
      );
    }
    if (
      sample.promptTokens !== undefined &&
      sample.promptTokens >= MIN_PREFILL_COST_SAMPLE_TOKENS
    ) {
      // Pressure-shaped prompt timing includes the queue state that the
      // renderer shares. Treating it as a per-token compute cost made a
      // one-token chunk look cheaper than a real batch and collapsed the
      // next prefill graph to one token. Only clear samples, or a backend
      // that explicitly separates compute from queue wait, may update this
      // estimate.
      const isolatedPromptComputeMs =
        sample.promptComputeMs ??
        (sample.promptQueueWaitMs !== undefined &&
          sample.promptMs !== undefined
          ? Math.max(0, sample.promptMs - sample.promptQueueWaitMs)
          : undefined);
      const prefillComputeMs =
        isolatedPromptComputeMs ??
        (pressureShaped ? undefined : sample.promptMs);
      if (prefillComputeMs !== undefined && prefillComputeMs > 0) {
        this.prefillMsPerToken = ema(
          this.prefillMsPerToken,
          prefillComputeMs / sample.promptTokens
        );
      }
    }
    if (
      sample.elapsedMs !== undefined &&
      sample.budgetMs !== undefined &&
      sample.budgetMs > 0
    ) {
      // The unit's GPU residency, not a completion-wait counter: elapsed
      // wall time minus the window the lease admitted.
      this.noteOccupancyOvershoot(
        sample.occupancyOvershootMs ??
          Math.max(0, sample.elapsedMs - sample.budgetMs)
      );
    }
  }

  /**
   * Explicit queue wait, when a backend can report how long it waited for GPU
   * completion. Hosts without that counter should call observeUnit instead,
   * which derives the wait from the unit's overshoot.
   */
  noteQueueWait(waitMs: number): void {
    if (!Number.isFinite(waitMs) || waitMs < 0) {
      return;
    }
    this.queueWaitMs = ema(this.queueWaitMs, waitMs);
    this.noteGpuPressure(waitMs);
  }

  /**
   * GPU occupancy overshoot of one unit: elapsed wall time minus the window
   * the lease admitted. A unit that runs far past its own deadline is still
   * holding the queue while the compositor wants it, whatever the backend can
   * or cannot report about completion, so this is what the local barrier is
   * decided from when no wait counter exists.
   */
  noteOccupancyOvershoot(overshootMs: number): void {
    if (!Number.isFinite(overshootMs) || overshootMs < 0) {
      return;
    }
    this.occupancyOvershootMs = ema(this.occupancyOvershootMs, overshootMs);
    // The overshoot is a soft-budget overrun: a unit ends at its next token
    // boundary, so a work-conserving unit that runs past its own ceiling
    // says nothing about what a renderer is waiting for. It becomes queue
    // evidence only once the frame signal already reports a late presented
    // frame; otherwise the local barrier would arm on this engine's own
    // queue depth and drain the device queue on every submission.
    // A clear newest sample releases the barrier even while the regime label
    // is still in its release hysteresis. In that state an overshoot says the
    // engine waited behind the renderer, not that the renderer is waiting
    // behind the engine. Indeterminate samples keep the conservative path.
    if (
      this.lastSampleStressed ||
      (this.regime === 'deadline' && !this.lastSampleClear)
    ) {
      this.noteGpuPressure(overshootMs);
    } else {
      this.noteGpuRelief();
    }
  }

  private noteGpuPressure(waitMs: number): void {
    if (waitMs > this.options.queueWaitToleranceMs) {
      this.queueWaitStreak += 1;
      this.queueReliefStreak = 0;
      if (this.queueWaitStreak >= this.options.queueBackpressureSamples) {
        this.queueBarrier = true;
      }
      return;
    }
    this.noteGpuRelief();
  }

  /**
   * A unit that finished inside its window is the signal that the local
   * barrier can be released: with no completion-wait counter to read, relief
   * is the absence of an overshoot, not a reported wait of zero.
   */
  private noteGpuRelief(): void {
    this.queueWaitStreak = 0;
    this.queueReliefStreak += 1;
    if (this.queueReliefStreak >= this.options.queueReliefSamples) {
      this.queueBarrier = false;
    }
  }

  /**
   * The lease for the next unit. Safe to call at every graph boundary; the
   * returned lease carries its own id so policy pushes can be correlated with
   * the unit they shaped.
   */
  lease(input: LeaseRequest): ExecutionLease {
    const now = input.now;
    const frameBudgetMs = this.frameBudgetMs();
    // The debt uses the worst of the last two samples, so the very next unit
    // already reflects a step into contention instead of waiting for the
    // smoothing window to turn over.
    const health =
      this.samples.length > 0 ? worstOfLast(this.samples, 2) : null;
    const debtMs = health ? Math.max(0, health.p95Ms - frameBudgetMs) : 0;
    const tailSeverity = this.tailSeverity(health, frameBudgetMs);
    const signalAgeMs = this.lastSignalAt > 0 ? now - this.lastSignalAt : null;

    // A page that says it is hidden gets the background slice even if no
    // frame sample arrived: the lifecycle update is itself fresh evidence.
    // ...unless the human is still waiting on the click that activated it.
    // An unanswered activation keeps the foreground shape until it produces
    // a token, so the prefill chunk it is in the middle of can finish.
    const background =
      input.background ??
      (!input.foreground && input.handoff !== true && input.priority !== true);
    const highPriorityLane =
      input.foreground || input.handoff === true || input.priority === true;
    if (background) {
      return this.emit({
        source: 'background',
        reason: 'background',
        budgetMs: this.options.backgroundBudgetMs,
        frameBudgetMs,
        frameDebtMs: debtMs,
        now,
        forceBarrier: this.options.backgroundBarrier,
        pauseMs: 0,
      });
    }

    if (signalAgeMs === null || signalAgeMs > this.options.staleSignalMs) {
      if (
        signalAgeMs !== null &&
        signalAgeMs > this.options.staleSignalMs &&
        this.lastRecoveryReason !== 'stale-signal'
      ) {
        this.noteRecovery('stale-signal', now);
      }
      // Without fresh evidence the lease cannot prove that a long unit is
      // safe for the renderer, but the bounded fallback pays a host round
      // trip every one or two tokens, which turns a cold start into a
      // decoder that runs far below the native rate. The barrier shape
      // keeps the native token ceilings and bounds the renderer wait
      // instead: at most one command batch is in flight.
      const barrierFallback = this.options.staleSignalShape === 'barrier';
      return this.emit({
        source: 'fallback',
        handoff: input.handoff === true,
        reason: 'stale-signal',
        budgetMs: barrierFallback
          ? this.options.maxBudgetMs
          : this.options.fallbackBudgetMs,
        frameBudgetMs,
        frameDebtMs: debtMs,
        now,
        forceBarrier: barrierFallback,
        continuous: barrierFallback,
        pauseMs: 0,
      });
    }

    // The regime label has hysteresis so that one late sample cannot latch
    // deadline mode. The first stressed sample still has to constrain the
    // next graph boundary: otherwise a work-conserving unit can carry the
    // whole response while the renderer is already missing frames.
    if (this.regime === 'work-conserving' && !this.lastSampleStressed) {
      return this.emit({
        source: 'deadline',
        handoff: input.handoff === true,
        reason: 'work-conserving',
        budgetMs: this.options.maxBudgetMs,
        frameBudgetMs,
        frameDebtMs: debtMs,
        now,
        forceBarrier: false,
        priorityLane: highPriorityLane,
        pauseMs: 0,
      });
    }

    const frameDebtMs = debtMs;
    const budgetMs = this.pressureBudgetMs(health, frameBudgetMs);
    const reason: LeaseReason = this.queueBarrier
      ? 'queue-backpressure'
      : 'frame-deadline';
    if (reason !== this.lastRecoveryReason && this.queueBarrier) {
      this.noteRecovery(reason, now);
    }
    return this.emit({
      source: 'deadline',
      handoff: input.handoff === true,
      reason,
      budgetMs,
      frameBudgetMs,
      frameDebtMs,
      now,
      forceBarrier: this.queueBarrier,
      tailSeverity,
      pauseMs: this.pressurePauseMs(health, frameBudgetMs),
    });
  }

  diagnostics(input: { now: number } = { now: this.lastSignalAt }): LeaseDiagnostics {
    const lease = this.lastLease;
    const frameBudgetMs = this.frameBudgetMs();
    return {
      source: lease?.source ?? 'deadline',
      reason: lease?.reason ?? 'work-conserving',
      leaseId: lease?.leaseId ?? 0,
      budgetMs: lease?.budgetMs ?? this.options.maxBudgetMs,
      maxDecodeTokens: lease?.maxDecodeTokens ?? this.options.maxDecodeTokens,
      maxPrefillTokens: lease?.maxPrefillTokens ?? this.options.maxPrefillTokens,
      queue: lease
        ? {
            commandSubmitBatchSize: lease.commandSubmitBatchSize,
            maxInflightBatches: lease.maxInflightBatches,
            basis: lease.basis,
          }
        : {
            commandSubmitBatchSize: this.options.maxCommandBatchSize,
            maxInflightBatches: 0,
            basis: 'unlimited',
          },
      frameBudgetMs,
      frameDebtMs: lease?.frameDebtMs ?? 0,
      refreshHz: this.refreshHz(),
      signalAgeMs:
        this.lastSignalAt > 0 ? Math.max(0, input.now - this.lastSignalAt) : null,
      occupancyOvershootMs:
        this.occupancyOvershootMs === null
          ? null
          : round2(this.occupancyOvershootMs),
      queueWaitMs: this.queueWaitMs,
      submissionPauseMs: lease?.submissionPauseMs ?? 0,
      measuredTokenCostMs:
        this.pressureMsPerToken === null
          ? null
          : round2(this.pressureMsPerToken),
      measuredPrefillMsPerToken:
        this.prefillMsPerToken === null
          ? null
          : round2(this.prefillMsPerToken),
      lastUnit: null,
      recoveries: this.recoveries,
      lastRecoveryAt: this.lastRecoveryAt,
      lastRecoveryReason: this.lastRecoveryReason,
    };
  }

  reset(): void {
    this.samples.length = 0;
    this.baselineP95.length = 0;
    this.pressureSamples = 0;
    this.clearSamples = 0;
    this.regime = 'work-conserving';
    this.lastSignalAt = 0;
    this.lastLease = null;
    this.decodeTokensPerMs = null;
    this.pressureMsPerToken = null;
    this.prefillMsPerToken = null;
    this.queueWaitMs = null;
    this.occupancyOvershootMs = null;
    this.queueWaitStreak = 0;
    this.queueReliefStreak = 0;
    this.queueBarrier = false;
    this.lastSampleStressed = false;
    this.lastSampleClear = false;
  }

  private emit(input: {
    source: ExecutionLeaseSource;
    reason: LeaseReason;
    budgetMs: number;
    frameBudgetMs: number;
    frameDebtMs: number;
    now: number;
    forceBarrier: boolean;
    pauseMs: number;
    // Keeps the work-conserving token ceilings for a lease that is not the
    // work-conserving regime itself, so a cold start can decode a full
    // native unit while the queue stays barriered.
    continuous?: boolean;
    handoff?: boolean;
    priorityLane?: boolean;
    tailSeverity?: number;
  }): ExecutionLease {
    const pressureLease =
      input.reason === 'frame-deadline' ||
      input.reason === 'queue-backpressure';
    // Pressure leases already carry the formula's wall-time window; other
    // regimes keep the runtime floor that keeps a fallback unit alive.
    const budgetMs = pressureLease
      ? Math.min(input.budgetMs, this.options.maxBudgetMs)
      : clamp(
          input.budgetMs,
          Math.min(this.options.minBudgetMs, this.options.fallbackBudgetMs),
          this.options.maxBudgetMs
        );
    const continuous =
      input.reason === 'work-conserving' || input.continuous === true;
    const workConserving = input.reason === 'work-conserving';
    const queue = this.deriveQueueLease(
      continuous,
      input.forceBarrier,
      input.tailSeverity ?? 0,
      input.priorityLane === true
    );
    // The token budget comes from the native rate (tokens per millisecond of
    // prediction time). It is deliberately not derived from the contended
    // wall cost: that cost includes queue wait, and a cap built from it would
    // truncate the unit before the window the cost term just admitted.
    const decodeRate =
      this.decodeTokensPerMs ?? this.options.fallbackDecodeTokensPerMs;
    const prefillMsPerToken =
      this.prefillMsPerToken ?? this.options.fallbackPrefillMsPerToken;
    // A native prefill graph cannot be interrupted at a token boundary the
    // way decode can. Its token cap is the lease window divided by the
    // measured prefill cost per token, clamped to the phase bounds below.
    const measuredPrefillCap = Math.max(
      1,
      Math.floor(budgetMs / Math.max(0.01, prefillMsPerToken))
    );
    const decodeFloor = pressureLease
      ? Math.max(1, Math.floor(this.options.minPressureDecodeTokens))
      : input.reason === 'background'
        ? this.options.backgroundDecodeTokens
        : this.options.minDecodeTokens;
    const minPressurePrefillTokens = Math.max(
      1,
      this.options.minPressurePrefillTokens
    );
    const maxPressurePrefillTokens = Math.max(
      minPressurePrefillTokens,
      this.options.maxPressurePrefillTokens
    );
    const minPrefillTokens = pressureLease
      ? minPressurePrefillTokens
      : Math.max(1, Math.floor(this.options.minPrefillTokens));
    // Attention quantum: every lease returns to the scheduler inside this
    // wall time. Without it the work-conserving shape covers a whole
    // response in one graph, and a lifecycle change that arrives in the
    // middle of it cannot be honoured until that graph ends.
    // Decode burst: deadline-shaped leases return at the short attention
    // quantum; a work-conserving page may use the longer cooperative burst
    // because activation preemption can still cut the native graph at the
    // next token boundary.
    const quantumDecodeTokens = clamp(
      Math.round(
        decodeRate *
          (workConserving
            ? this.options.workConservingQuantumMs
            : this.options.attentionQuantumMs)
      ),
      Math.max(1, Math.floor(this.options.minQuantumDecodeTokens)),
      this.options.maxDecodeTokens
    );
    const maxDecodeTokens =
      input.handoff
        ? Math.min(this.options.handoffDecodeTokens, quantumDecodeTokens)
        : continuous
        ? Math.min(quantumDecodeTokens, this.options.maxDecodeTokens)
        : Math.min(
            quantumDecodeTokens,
            clamp(
              Math.floor(decodeRate * budgetMs),
              decodeFloor,
              this.options.maxDecodeTokens
            )
          );
    const priorityDecodeCap =
      input.priorityLane === true && input.handoff !== true
        ? Math.min(
            this.options.maxDecodeTokens,
            Math.max(this.options.priorityDecodeTokens, maxDecodeTokens)
          )
        : maxDecodeTokens;
    // The same bound applies to a prefill graph: a prompt chunk that costs
    // more than the quantum would hold the coordinator for longer than the
    // attention signal is allowed to wait.
    // A work-conserving page can amortise the prefill graph over the same
    // longer cooperative burst used for decode. Pressure and stale-signal
    // leases keep the short attention quantum so a late frame or a lifecycle
    // change is still observed at the next boundary.
    const prefillQuantumMs = workConserving
      ? this.options.workConservingQuantumMs
      : this.options.attentionQuantumMs;
    const quantumPrefillTokens = clamp(
      Math.floor(prefillQuantumMs / Math.max(0.01, prefillMsPerToken)),
      1,
      this.options.maxPrefillTokens
    );
    // The 64-token knee applies to every regime: a prompt chunk below it
    // pays the graph and queue round trip per token, which is what made a
    // foreground click in the middle of a prefill return without a token.
    // The work-conserving branch keeps that floor even when the measured
    // quantum is smaller, and the quantum still caps any longer chunk.
    const maxPrefillTokens =
      continuous
        ? clamp(
            quantumPrefillTokens,
            minPrefillTokens,
            this.options.maxPrefillTokens
          )
        : pressureLease
          ? clamp(
              measuredPrefillCap,
              minPressurePrefillTokens,
              maxPressurePrefillTokens
            )
          : clamp(
              Math.min(
                measuredPrefillCap,
                Math.floor(budgetMs / Math.max(0.01, prefillMsPerToken))
              ),
              minPrefillTokens,
              this.options.maxPrefillTokens
            );

    // The motivation harness pins the prefill graph length while keeping the
    // same decode and queue policy. It is deliberately ignored for stale or
    // background signals, where the safety fallback must still win.
    const forcedPrefillTokens =
      this.options.prefillChunkOverride === undefined ||
      input.reason === 'stale-signal' ||
      input.reason === 'background'
        ? null
        : clamp(
            Math.floor(this.options.prefillChunkOverride),
            1,
            this.options.maxPrefillTokens
          );
    const effectiveMaxPrefillTokens = forcedPrefillTokens ?? maxPrefillTokens;

    const lease: ExecutionLease = {
      leaseId: this.nextLeaseId,
      source: input.source,
      reason: input.reason,
      budgetMs: round2(budgetMs),
      maxDecodeTokens: priorityDecodeCap,
      maxPrefillTokens: effectiveMaxPrefillTokens,
      commandSubmitBatchSize: queue.commandSubmitBatchSize,
      maxInflightBatches: queue.maxInflightBatches,
      basis: queue.basis,
      submissionPauseMs: clamp(
        Math.round(input.pauseMs),
        0,
        Math.max(0, this.options.maxPauseMs)
      ),
      frameBudgetMs: round2(input.frameBudgetMs),
      frameDebtMs: round2(input.frameDebtMs),
      queueWaitMs: this.queueWaitMs === null ? null : round2(this.queueWaitMs),
      createdAt: input.now,
    };
    const previous = this.lastLease;
    if (
      !previous ||
      previous.budgetMs !== lease.budgetMs ||
      previous.maxDecodeTokens !== lease.maxDecodeTokens ||
      previous.maxInflightBatches !== lease.maxInflightBatches ||
      previous.commandSubmitBatchSize !== lease.commandSubmitBatchSize ||
      previous.submissionPauseMs !== lease.submissionPauseMs ||
      previous.reason !== lease.reason
    ) {
      this.nextLeaseId += 1;
    }
    this.lastLease = lease;
    return lease;
  }

  /**
   * Queue capacity follows the budget share of one frame: a lease that only
   * admits a fraction of a frame's work may keep only that fraction of the
   * deployment's in-flight batches outstanding. When the backend cannot report
   * GPU completion the caller forces the barrier, which is the same limit
   * expressed as "one batch at a time".
   */
  private pressureBudgetMs(
    health: FrameSample | null,
    frameBudgetMs: number
  ): number {
    if (!health) {
      return this.options.maxBudgetMs;
    }
    // The window is the tighter of the deployment ceiling left after
    // severity and the display-period window. It is a cooperative target
    // enforced at token boundaries: the phase token floor in emit() keeps
    // progress going when one native token costs more than the window.
    const tailSeverity = this.pressurePrefillSeverity(
      health,
      frameBudgetMs
    );
    // Two independent bounds on the window:
    //
    //  deadlineBudgetMs is what the presented tail leaves of the deployment
    //  ceiling: a lease that is already multi-frame late shrinks towards the
    //  floor of the deployment shape as the excess grows.
    //
    //  frameCadenceBudgetMs counts the window in displayed periods: a mild
    //  tail spends the mild number of periods, a violated deadline collapses
    //  towards a single period. Display periods scale with the panel, so the
    //  same lease admits half the wall time on a 120 Hz screen.
    const deadlineBudgetMs = this.options.maxBudgetMs * (1 - tailSeverity);
    const periods =
      this.options.mildPressureFramePeriods +
      (this.options.severePressureFramePeriods -
        this.options.mildPressureFramePeriods) *
        tailSeverity;
    const frameCadenceBudgetMs = Math.max(0, frameBudgetMs * periods);
    return Math.min(deadlineBudgetMs, frameCadenceBudgetMs);
  }

  /**
   * Presented-tail severity shared by the wall-clock window and the
   * pressure prefill cap. The p99 excess protects frames from a long
   * command buffer.
   */
  private pressurePrefillSeverity(
    health: FrameSample | null,
    frameBudgetMs: number
  ): number {
    if (!health) {
      return 0;
    }
    const tailDeadlineMs =
      frameBudgetMs * this.options.tailBudgetFrameMultiplier;
    const tailExcess = Math.max(0, health.p99Ms - tailDeadlineMs);
    return Math.min(
      0.9,
      tailDeadlineMs > 0 ? tailExcess / tailDeadlineMs : 0
    );
  }

  private deriveQueueLease(
    workConserving: boolean,
    forceBarrier: boolean,
    tailSeverity: number,
    priorityLane: boolean
  ): QueueLease {
    if (workConserving && !forceBarrier) {
      if (priorityLane) {
        // A foreground/high-priority unit may use the deployment's deep
        // submission queue while the presented frame is still healthy.
        // The local barrier is a protection for ordinary work; applying it
        // to the priority lane serializes every token behind the renderer
        // and turns a short response into a multi-second completion.
        // Match the deployment queue rather than the unlimited clear-page
        // shape. Under rendering contention an unbounded priority queue lets
        // the compositor and inference contend without a useful completion
        // signal; the bounded deployment queue keeps the promoted request on
        // the fast path without serializing it at a local barrier.
        return {
          commandSubmitBatchSize: this.options.maxCommandBatchSize,
          maxInflightBatches: clamp(this.options.inflightCap, 1, 64),
          basis: 'budget-share',
        };
      }
      if (this.options.workConservingBarrier) {
        return {
          commandSubmitBatchSize: clamp(
            this.options.minPressureCommandBatchSize,
            this.options.minCommandBatchSize,
            this.options.maxCommandBatchSize
          ),
          maxInflightBatches: 1,
          basis: 'local-barrier',
        };
      }
      return {
        commandSubmitBatchSize: this.options.maxCommandBatchSize,
        maxInflightBatches: 0,
        basis: 'unlimited',
      };
    }
    // Two separate limits, because they buy different things. The command
    // submission size bounds how long the renderer waits once its work is
    // behind ours, so it follows the presented deadline: a mild tail keeps
    // the deployment's submission size, a violated deadline halves it, and a
    // proven queue wait drops to the smallest submission that still pays for
    // the host round trip.
    // A proven queue wait is the strongest evidence that the renderer is
    // behind our command buffer. In that state the smallest useful batch is
    // the only bound that shortens the wait itself; a larger "pressure" floor
    // can still hold the compositor for the whole batch even with one buffer
    // in flight. The moderate-tail branch keeps its larger floor.
    const submissionSize = forceBarrier
      ? this.options.minCommandBatchSize
      : tailSeverity >= 0.5
        ? Math.max(
            this.options.minPressureCommandBatchSize,
            this.options.maxCommandBatchSize / 2
          )
        : this.options.maxCommandBatchSize;
    const commandSubmitBatchSize = clamp(
      nearestPowerOfTwo(submissionSize),
      this.options.minCommandBatchSize,
      this.options.maxCommandBatchSize
    );
    // In-flight batches bound how deep that backlog may get. While the
    // renderer is only mildly late, one submission may be encoded while the
    // previous one runs, which is what keeps the GPU fed without letting the
    // queue run away. Measured queue wait (forceBarrier) collapses that to a
    // local barrier, which is also the shape a backend without completion
    // telemetry falls back to.
    if (forceBarrier) {
      return {
        commandSubmitBatchSize,
        maxInflightBatches: 1,
        basis: 'local-barrier',
      };
    }
    return {
      commandSubmitBatchSize,
      maxInflightBatches: clamp(this.options.inflightCap, 1, 64),
      basis: 'budget-share',
    };
  }

  // Convert the presented tail into a 0..1 severity around a three-frame
  // deadline. The queue lease uses it to choose between a shallow bounded
  // queue and a local barrier; a two-frame tail alone is not severe enough.
  private tailSeverity(health: FrameSample | null, frameBudgetMs: number): number {
    if (!health || frameBudgetMs <= 0) {
      return 0;
    }
    return clamp(health.p95Ms / frameBudgetMs - 2.5, 0, 1);
  }

  private frameBudgetMs(): number {
    return 1000 / this.refreshHz();
  }

  private refreshHz(): number {
    for (let index = this.samples.length - 1; index >= 0; index -= 1) {
      const hz = this.samples[index]?.refreshHz;
      if (hz !== null && hz !== undefined) {
        return hz;
      }
    }
    return this.options.fallbackRefreshHz;
  }

  // The floor of the recent window: whatever the page manages to present
  // when nothing is contending. Until the window has enough samples the
  // lease falls back to the display budget alone, so a cold start cannot
  // learn a contended window as its own baseline.
  private learnedBaselineP95Ms(): number {
    if (this.baselineP95.length < this.options.minBaselineSamples) {
      return 0;
    }
    return Math.min(...this.baselineP95);
  }

  /**
   * Rest interval between units, derived from how far the presented tail has
   * passed the multi-frame deadline. The controller is closed-loop on the
   * measured tail: a page whose frames recover needs no pause, and a page
   * that is still late gets progressively longer empty-queue windows until
   * the measured tail recovers or the cap is reached. The cap is what keeps
   * this a throttle instead of the gate-based stop that failed earlier.
   */
  private pressurePauseMs(
    health: FrameSample | null,
    frameBudgetMs: number
  ): number {
    if (!health) {
      return 0;
    }
    const tailDeadlineMs =
      frameBudgetMs * this.options.tailBudgetFrameMultiplier;
    const debtMs = Math.max(0, health.p99Ms - tailDeadlineMs);
    return clamp(
      debtMs * this.options.pausePerDebtMs,
      0,
      this.options.maxPauseMs
    );
  }

  private noteRecovery(reason: LeaseReason, now: number): void {
    this.recoveries += 1;
    this.lastRecoveryAt = now;
    this.lastRecoveryReason = reason;
  }
}

// Native timers are sampled from the backend and can report a reset or a
// partial interval after a bounded stop. A plausible token cost is at least
// 0.5 ms and cannot exceed the host unit duration by more than a small
// scheduling margin.
function hasReliableNativeTiming(
  decoded: number,
  predictedMs: number | undefined,
  elapsedMs: number | undefined
): boolean {
  if (decoded <= 0 || predictedMs === undefined || predictedMs <= 0) {
    return false;
  }
  const perTokenMs = predictedMs / decoded;
  if (perTokenMs < 0.5 || perTokenMs > 1_000) {
    return false;
  }
  if (
    elapsedMs !== undefined &&
    elapsedMs > 0 &&
    predictedMs > elapsedMs * 1.5 + 5
  ) {
    return false;
  }
  return true;
}

function extendedTail(signal: RenderPressureSignal): FrameHealthTail {
  return {
    p99Ms: finite(signal.p99Ms, signal.maxMs),
    over33Ratio: finite(signal.over33Ratio, signal.overBudgetRatio),
    over50Ratio: finite(signal.over50Ratio, 0),
    longestStallMs: finite(signal.longestStallMs, 0),
    ...(signal.refreshHz === undefined ? {} : { refreshHz: signal.refreshHz }),
  };
}

// Worst of the most recent N samples: the attack half of the hysteresis.
function worstOfLast(
  samples: readonly FrameSample[],
  count: number
): FrameSample {
  const recent = samples.slice(-Math.max(1, count));
  const last = recent[recent.length - 1]!;
  return {
    at: last.at,
    p95Ms: Math.max(...recent.map((sample) => sample.p95Ms)),
    p99Ms: Math.max(...recent.map((sample) => sample.p99Ms)),
    over33Ratio: Math.max(...recent.map((sample) => sample.over33Ratio)),
    over50Ratio: Math.max(...recent.map((sample) => sample.over50Ratio)),
    longestStallMs: Math.max(...recent.map((sample) => sample.longestStallMs)),
    refreshHz: last.refreshHz,
    frameCount: recent.reduce((sum, sample) => sum + sample.frameCount, 0),
  };
}

function smoothedSample(samples: readonly FrameSample[]): FrameSample | null {
  if (samples.length === 0) {
    return null;
  }
  const recent = samples.slice(-3);
  return {
    at: recent[recent.length - 1]!.at,
    p95Ms: medianOf(recent.map((sample) => sample.p95Ms)),
    p99Ms: medianOf(recent.map((sample) => sample.p99Ms)),
    over33Ratio: medianOf(recent.map((sample) => sample.over33Ratio)),
    over50Ratio: medianOf(recent.map((sample) => sample.over50Ratio)),
    longestStallMs: Math.max(...recent.map((sample) => sample.longestStallMs)),
    refreshHz: recent[recent.length - 1]!.refreshHz,
    frameCount: recent.reduce((sum, sample) => sum + sample.frameCount, 0),
  };
}

function medianOf(values: readonly number[]): number {
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 1) {
    return sorted[middle] ?? 0;
  }
  const lower = sorted[middle - 1] ?? 0;
  const upper = sorted[middle] ?? 0;
  return (lower + upper) / 2;
}

function finite(value: number | undefined, fallback: number): number {
  return value !== undefined && Number.isFinite(value) ? value : fallback;
}

function finiteHz(value: number | undefined): number | null {
  if (value === undefined || !Number.isFinite(value) || value <= 0) {
    return null;
  }
  return clamp(Math.round(value), 24, 240);
}

function nearestPowerOfTwo(value: number): number {
  if (!Number.isFinite(value) || value <= 1) {
    return 1;
  }
  return 2 ** Math.round(Math.log2(value));
}

function ema(previous: number | null, sample: number): number {
  if (previous === null) {
    return sample;
  }
  return previous * 0.5 + sample * 0.5;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function round2(value: number): number {
  return Math.round(value * 100) / 100;
}
