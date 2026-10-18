import { describe, expect, it } from 'vitest';
import { FrameDeadlineLease } from '../../src/execution/frame-deadline-lease';
import type { RenderPressureSignal } from '../../src/contracts';

function frame(
  overrides: Partial<RenderPressureSignal> = {}
): RenderPressureSignal {
  return {
    clientId: 'c1',
    epoch: 1,
    sampledAt: 0,
    source: 'presented-frame',
    frameCount: 12,
    p50Ms: 16,
    p95Ms: 18,
    p99Ms: 20,
    maxMs: 24,
    overBudgetRatio: 0.1,
    over33Ratio: 0,
    over50Ratio: 0,
    longestStallMs: 0,
    refreshHz: 60,
    ...overrides,
  };
}

describe('FrameDeadlineLease', () => {
  const maxBudgetMs = 400;
  const newLease = () => new FrameDeadlineLease({ maxBudgetMs });

  it('is work-conserving while frames are clear', () => {
    const lease = newLease();
    lease.observe(frame(), 1_000);
    lease.observe(frame(), 1_500);

    const active = lease.lease({ now: 1_600, foreground: true });

    expect(active.reason).toBe('work-conserving');
    expect(active.source).toBe('deadline');
    expect(active.budgetMs).toBe(lease.options.maxBudgetMs);
    expect(active.maxInflightBatches).toBe(lease.options.inflightCap);
    expect(active.basis).toBe('budget-share');
    expect(active.commandSubmitBatchSize).toBe(
      lease.options.maxCommandBatchSize
    );
  });

  it('does not treat normal priority as a hidden background page', () => {
    const lease = newLease();
    lease.observe(frame(), 1_000);
    lease.observe(frame(), 1_500);

    const active = lease.lease({
      now: 1_600,
      foreground: false,
      background: false,
    });

    expect(active.source).toBe('deadline');
    expect(active.reason).toBe('work-conserving');
    expect(active.maxInflightBatches).toBe(0);
    expect(active.basis).toBe('unlimited');
    expect(active.maxDecodeTokens).toBeGreaterThan(1);
  });

  it('keeps the priority lane queued when ordinary work is barriered', () => {
    const lease = new FrameDeadlineLease({
      maxBudgetMs,
      workConservingBarrier: true,
    });
    lease.observe(frame(), 1_000);
    lease.observe(frame(), 1_500);

    const priority = lease.lease({ now: 1_600, foreground: true });
    expect(priority.reason).toBe('work-conserving');
    expect(priority.maxInflightBatches).toBe(lease.options.inflightCap);
    expect(priority.basis).toBe('budget-share');

    const ordinary = lease.lease({
      now: 1_700,
      foreground: false,
      background: false,
    });
    expect(ordinary.reason).toBe('work-conserving');
    expect(ordinary.maxInflightBatches).toBe(1);
    expect(ordinary.basis).toBe('local-barrier');
  });

  it('keeps a high-priority decode unit large enough to amortise native setup', () => {
    const lease = newLease();
    lease.observe(frame(), 1_000);
    lease.observe(frame(), 1_500);
    // A contended 16-token unit reports 5 s of native prediction time.
    // Converting that rate directly into the next unit would make the slow
    // unit shape self-sustaining instead of giving priority work a fair run.
    lease.observeUnit({
      reason: 'work-conserving',
      decodeTokens: 16,
      predictedMs: 5_000,
      elapsedMs: 5_000,
      budgetMs: lease.options.maxBudgetMs,
    });

    const ordinary = lease.lease({
      now: 1_600,
      foreground: false,
      background: false,
    });
    const priority = lease.lease({ now: 1_700, foreground: true });

    expect(ordinary.maxDecodeTokens).toBeLessThan(
      lease.options.priorityDecodeTokens
    );
    expect(priority.maxDecodeTokens).toBe(
      lease.options.priorityDecodeTokens
    );
  });

  it('uses the longer cooperative burst while frames stay clear', () => {
    const lease = newLease();
    lease.observe(frame(), 1_000);
    lease.observe(frame(), 1_500);
    // 20 tokens in 100 ms of native decode time: 0.2 tokens/ms
    lease.observeUnit({
      reason: 'work-conserving',
      decodeTokens: 20,
      predictedMs: 100,
      elapsedMs: 100,
      budgetMs: lease.options.maxBudgetMs,
    });

    const active = lease.lease({ now: 1_600, foreground: true });

    expect(active.reason).toBe('work-conserving');
    expect(active.maxDecodeTokens).toBe(
      Math.round(0.2 * lease.options.workConservingQuantumMs)
    );
    expect(active.maxDecodeTokens).toBeLessThan(
      lease.options.maxDecodeTokens
    );
  });

  it('keeps a prefill floor for background and silent-signal leases', () => {
    const lease = newLease();
    lease.observe(frame(), 1_000);
    lease.observeUnit({
      reason: 'background',
      decodeTokens: 1,
      predictedMs: 30,
      promptTokens: 2,
      promptMs: 6,
      elapsedMs: 55,
      budgetMs: 8,
    });

    const background = lease.lease({ now: 1_600, foreground: false });

    // The millisecond budget alone would admit a two-token prompt chunk, and
    // a graph/queue round trip per two tokens is what turned a ten-tab
    // background prefill into seconds of wall time.
    expect(background.reason).toBe('background');
    expect(background.maxPrefillTokens).toBe(lease.options.minPrefillTokens);
  });

  it('derives the budget from frame debt once pressure persists', () => {
    // Explicit period pair so the debt term is visible: the shipped default
    // spends one period at every severity, and the wider-window ablation is
    // exactly this option.
    const lease = new FrameDeadlineLease({
      maxBudgetMs,
      mildPressureFramePeriods: 2,
      severePressureFramePeriods: 1,
    });
    lease.observe(frame({ p95Ms: 34, p99Ms: 120, over33Ratio: 0.4 }), 1_000);
    // The first stressed sample already constrains the next graph boundary;
    // the second sample is what changes the persistent regime label.
    const provisional = lease.lease({ now: 1_100, foreground: true });
    expect(provisional.reason).toBe('frame-deadline');
    expect(provisional.budgetMs).toBeLessThan(lease.options.maxBudgetMs);
    expect(provisional.maxDecodeTokens).toBeGreaterThanOrEqual(
      lease.options.minPressureDecodeTokens
    );

    lease.observe(frame({ p95Ms: 34, p99Ms: 120, over33Ratio: 0.4 }), 1_500);
    const active = lease.lease({ now: 1_600, foreground: true });

    expect(active.reason).toBe('frame-deadline');
    // The default five-frame deadline is 83.33 ms at 60 Hz. A 120 ms P99
    // is 36.67 ms over it, so severity is 36.67 / 83.33 = 0.44.
    const tailDeadlineMs =
      active.frameBudgetMs * lease.options.tailBudgetFrameMultiplier;
    const severity = Math.min(
      0.9,
      Math.max(0, (120 - tailDeadlineMs) / tailDeadlineMs)
    );
    const periods =
      lease.options.mildPressureFramePeriods +
      (lease.options.severePressureFramePeriods -
        lease.options.mildPressureFramePeriods) *
        severity;
    expect(active.budgetMs).toBeCloseTo(active.frameBudgetMs * periods, 1);
    expect(active.budgetMs).toBeLessThan(lease.options.maxBudgetMs);
    expect(active.maxDecodeTokens).toBeGreaterThanOrEqual(
      lease.options.minPressureDecodeTokens
    );
    expect(active.maxInflightBatches).toBeGreaterThanOrEqual(1);
    expect(active.maxInflightBatches).toBeLessThanOrEqual(
      lease.options.inflightCap
    );
    // A P95 near two frame periods is not enough to shrink the
    // submission itself: the in-flight bound is what keeps the renderer's
    // wait bounded, and shrinking the batch would add a host round trip per
    // submission without shortening that wait.
    expect(active.commandSubmitBatchSize).toBe(
      lease.options.maxCommandBatchSize
    );
    expect(active.maxInflightBatches).toBe(lease.options.inflightCap);
    expect(active.basis).toBe('budget-share');
  });

  it('does not learn a cold pressure episode as its own baseline', () => {
    const lease = newLease();
    const pressured = frame({
      p95Ms: 39,
      p99Ms: 60,
      over33Ratio: 0.2,
    });

    // The smoke can start the load before the lease has seen a clear
    // window. Those samples must not raise the pressure threshold after
    // the baseline ring fills; otherwise the controller enters deadline
    // mode briefly and then mistakes the sustained load for its own floor.
    for (let sample = 0; sample <= lease.options.baselineSamples; sample += 1) {
      lease.observe(pressured, 1_000 + sample * 100);
    }

    expect(lease.lease({ now: 4_000, foreground: true }).reason).toBe(
      'frame-deadline'
    );
  });

  it('learns an uncontended capture cadence without learning pressure', () => {
    const lease = newLease();
    const cadenceJitter = frame({
      p95Ms: 25,
      p99Ms: 26,
      over33Ratio: 0,
      over50Ratio: 0,
    });

    for (let sample = 0; sample < 8; sample += 1) {
      lease.observe(cadenceJitter, 1_000 + sample * 100);
    }

    expect(lease.lease({ now: 1_900, foreground: true }).reason).toBe(
      'work-conserving'
    );
  });

  it('derives the pressure decode cap from the lease window', () => {
    const fast = { maxBudgetMs, fallbackDecodeTokensPerMs: 0.2 };
    const mild = new FrameDeadlineLease(fast);
    const severe = new FrameDeadlineLease(fast);
    const mildFrame = frame({ p95Ms: 34, p99Ms: 40, over33Ratio: 0.3 });
    // 160 ms is past the point where severity reaches its cap for the
    // 83.33 ms tail deadline at 60 Hz.
    const severeFrame = frame({
      p95Ms: 90,
      p99Ms: 160,
      over33Ratio: 0.9,
    });

    for (const at of [1_000, 1_500]) {
      mild.observe(mildFrame, at);
      severe.observe(severeFrame, at);
    }

    const mildLease = mild.lease({ now: 1_600, foreground: true });
    const severeLease = severe.lease({ now: 1_600, foreground: true });

    // k = clamp(floor(c_hat / tau), k_l, k_u): the narrower severe window
    // admits fewer tokens at the same measured decode rate.
    expect(mildLease.maxDecodeTokens).toBe(
      Math.floor(0.2 * mildLease.budgetMs)
    );
    expect(severeLease.maxDecodeTokens).toBe(
      Math.floor(0.2 * severeLease.budgetMs)
    );
    expect(severeLease.maxDecodeTokens).toBeLessThan(
      mildLease.maxDecodeTokens
    );

    // At the default slow rate the computed cap falls below k_l, so the
    // pressure floor keeps the unit moving.
    const slow = newLease();
    for (const at of [1_000, 1_500]) {
      slow.observe(severeFrame, at);
    }
    const slowLease = slow.lease({ now: 1_600, foreground: true });
    expect(slowLease.maxDecodeTokens).toBe(
      slow.options.minPressureDecodeTokens
    );
  });
  it('shrinks the budget as frame debt grows', () => {


    // The wider-window ablation is the configuration where the debt term
    // still has room to move: at the shipped single-period default the
    // window is already at its floor.
    const wideOptions = {
      maxBudgetMs,
      mildPressureFramePeriods: 2,
      severePressureFramePeriods: 1,
    };
    const mild = new FrameDeadlineLease(wideOptions);
    const severe = new FrameDeadlineLease(wideOptions);
    const mildFrame = frame({ p95Ms: 28, p99Ms: 40, over33Ratio: 0.3 });
    const severeFrame = frame({ p95Ms: 90, p99Ms: 160, over33Ratio: 0.9 });

    for (const at of [1_000, 1_500]) {
      mild.observe(mildFrame, at);
      severe.observe(severeFrame, at);
    }

    const mildLease = mild.lease({ now: 1_600, foreground: true });
    const severeLease = severe.lease({ now: 1_600, foreground: true });

    expect(severeLease.budgetMs).toBeLessThan(mildLease.budgetMs);
    expect(severeLease.maxDecodeTokens).toBeLessThanOrEqual(
      mildLease.maxDecodeTokens
    );
  });

  it('ignores helper-canvas samples even when they look like a stall', () => {
    const lease = newLease();
    // The monitor's own helper canvas presents at a low, load-independent
    // cadence, so a late window from it says nothing about the page's
    // renderer. Only a page-owned sample may shrink the lease.
    const helper = frame({
      source: 'presented-frame',
      pageOwned: false,
      frameCount: 20,
      p50Ms: 30,
      p95Ms: 35,
      p99Ms: 52,
      over33Ratio: 0.5,
      over50Ratio: 0.2,
      longestStallMs: 120,
    });
    for (const at of [1_000, 1_500, 2_000, 2_500]) {
      lease.observe(helper, at);
    }

    const active = lease.lease({ now: 2_600, foreground: true });

    expect(active.reason).toBe('work-conserving');
    expect(active.budgetMs).toBe(lease.options.maxBudgetMs);
    expect(active.maxInflightBatches).toBe(lease.options.inflightCap);
    // the work-conserving prefill chunk is a quantum, not the whole prompt
    expect(active.maxPrefillTokens).toBe(
      Math.floor(
        lease.options.workConservingQuantumMs /
          lease.options.fallbackPrefillMsPerToken
      )
    );
    // A helper sample still counts as a live monitor, so the lease keeps its
    // work-conserving decision instead of falling back to the bounded shape.
    expect(lease.diagnostics({ now: 2_600 }).signalAgeMs).toBe(100);
  });

  it('returns to work-conserving when only the helper canvas is left', () => {
    const lease = newLease();
    for (const at of [1_000, 1_500]) {
      lease.observe(frame({ p95Ms: 40, p99Ms: 70, over33Ratio: 0.5 }), at);
    }
    expect(lease.lease({ now: 1_600, foreground: true }).reason).toBe(
      'frame-deadline'
    );

    // clear phase: the page canvas stops presenting and the helper takes
    // over. That is release evidence, not contention.
    for (const at of [1_700, 2_200]) {
      lease.observe(frame({ pageOwned: false, p95Ms: 35 }), at);
    }

    const recovered = lease.lease({ now: 2_300, foreground: true });
    expect(recovered.reason).toBe('work-conserving');
    expect(recovered.budgetMs).toBe(lease.options.maxBudgetMs);
  });

  it('does not arm the local barrier from a work-conserving overshoot', () => {
    const lease = newLease();
    // No frame sample has reported a late presented frame, so a unit that
    // ran past the deployment's bounded target is this engine queueing its
    // own work, not a renderer waiting behind it.
    lease.observe(frame(), 1_000);
    lease.observe(frame(), 1_500);
    for (const at of [1_000, 1_500, 2_000]) {
      lease.observeUnit({ elapsedMs: 60, budgetMs: 24 });
    }

    const active = lease.lease({ now: 2_100, foreground: true });

    expect(active.reason).toBe('work-conserving');
    expect(active.maxInflightBatches).toBe(lease.options.inflightCap);
    expect(active.basis).toBe('budget-share');
    expect(lease.diagnostics({ now: 2_100 }).occupancyOvershootMs).toBe(36);
  });

  it('never lets the lease starve a request', () => {
    const lease = newLease();
    const stalled = frame({
      p95Ms: 500,
      p99Ms: 900,
      maxMs: 1_200,
      longestStallMs: 1_200,
      over33Ratio: 1,
      over50Ratio: 1,
    });
    for (const at of [1_000, 1_500, 2_000]) {
      lease.observe(stalled, at);
    }

    const active = lease.lease({ now: 2_100, foreground: true });

    expect(active.budgetMs).toBeGreaterThanOrEqual(lease.options.minBudgetMs);
    expect(active.maxDecodeTokens).toBeGreaterThanOrEqual(1);
    expect(active.maxInflightBatches).toBeGreaterThanOrEqual(1);
    expect(active.maxPrefillTokens).toBeGreaterThanOrEqual(
      lease.options.minPressurePrefillTokens
    );
  });

  it('bounds the pressure prefill cap by the lease window and measured prompt cost', () => {
    const lease = newLease();
    // k = clamp(floor(c_hat / tau_prefill), k_l, k_u).
    const mild = new FrameDeadlineLease({
      maxBudgetMs,
      fallbackPrefillMsPerToken: 0.3,
    });
    const severe = new FrameDeadlineLease({
      maxBudgetMs,
      fallbackPrefillMsPerToken: 0.3,
    });
    const mildFrame = frame({ p95Ms: 34, p99Ms: 52, over33Ratio: 0.4 });
    const severeFrame = frame({ p95Ms: 90, p99Ms: 160, over33Ratio: 0.9 });

    for (const at of [1_000, 1_500]) {
      mild.observe(mildFrame, at);
      severe.observe(severeFrame, at);
    }

    const mildLease = mild.lease({ now: 1_600, foreground: true });
    const severeLease = severe.lease({ now: 1_600, foreground: true });

    expect(mildLease.maxPrefillTokens).toBe(
      Math.floor(mildLease.budgetMs / 0.3)
    );
    expect(severeLease.maxPrefillTokens).toBe(
      lease.options.minPressurePrefillTokens
    );
    expect(severeLease.maxPrefillTokens).toBeLessThan(
      mildLease.maxPrefillTokens
    );
  });

  it('bounds prefill graph length by measured prompt cost', () => {
    const lease = new FrameDeadlineLease({ maxBudgetMs });
    for (const at of [1_000, 1_500]) {
      lease.observe(frame({ p95Ms: 34, p99Ms: 52, over33Ratio: 0.4 }), at);
    }

    const first = lease.lease({ now: 1_600, foreground: true });
    expect(first.maxPrefillTokens).toBe(
      lease.options.minPressurePrefillTokens
    );

    // Queue-inflated prompt timing is not a per-token compute estimate.
    // Ignore it and keep the batch floor instead of collapsing to one token.
    lease.observeUnit({
      reason: 'frame-deadline',
      decodeTokens: 1,
      predictedMs: 8,
      promptTokens: 10,
      promptMs: 800,
      elapsedMs: 900,
      budgetMs: first.budgetMs,
    });
    const measured = lease.lease({ now: 1_700, foreground: true });
    expect(measured.maxPrefillTokens).toBe(
      lease.options.minPressurePrefillTokens
    );
    expect(lease.diagnostics({ now: 1_700 }).measuredPrefillMsPerToken).toBeNull();
  });

  it('does not learn a tiny prompt graph as a per-token prefill cost', () => {
    const lease = new FrameDeadlineLease({ maxBudgetMs });
    lease.observeUnit({
      reason: 'work-conserving',
      decodeTokens: 1,
      predictedMs: 8,
      promptTokens: 1,
      promptMs: 84,
      elapsedMs: 100,
      budgetMs: maxBudgetMs,
    });
    expect(lease.diagnostics({ now: 1_000 }).measuredPrefillMsPerToken).toBeNull();
  });

  it('accepts an isolated prompt compute time under pressure', () => {
    const lease = new FrameDeadlineLease({ maxBudgetMs });
    for (const at of [1_000, 1_500]) {
      lease.observe(frame({ p95Ms: 34, p99Ms: 52, over33Ratio: 0.4 }), at);
    }
    const first = lease.lease({ now: 1_600, foreground: true });
    lease.observeUnit({
      reason: 'frame-deadline',
      decodeTokens: 1,
      predictedMs: 8,
      promptTokens: 64,
      promptMs: 640,
      promptComputeMs: 320,
      promptQueueWaitMs: 320,
      elapsedMs: 700,
      budgetMs: first.budgetMs,
    });
    expect(lease.diagnostics({ now: 1_700 }).measuredPrefillMsPerToken).toBe(
      5
    );
  });

  it('can pin the prefill chunk for the motivation study', () => {
    const clear = newLease();
    clear.observe(frame(), 1_000);
    clear.observe(frame(), 1_500);
    expect(clear.lease({ now: 1_600, foreground: true }).maxPrefillTokens).toBe(
      // a clear-page prefill chunk is the work-conserving quantum, not the
      // prompt: the pinned override below is what the study varies
      Math.floor(
        clear.options.workConservingQuantumMs /
          clear.options.fallbackPrefillMsPerToken
      )
    );

    const pinned = new FrameDeadlineLease({
      maxBudgetMs,
      prefillChunkOverride: 32,
    });
    pinned.observe(frame(), 1_000);
    pinned.observe(frame(), 1_500);
    expect(pinned.lease({ now: 1_600, foreground: true }).maxPrefillTokens).toBe(
      32
    );

    pinned.observe(frame({ p95Ms: 90, p99Ms: 160, over33Ratio: 0.9 }), 2_000);
    pinned.observe(frame({ p95Ms: 90, p99Ms: 160, over33Ratio: 0.9 }), 2_500);
    expect(pinned.lease({ now: 2_600, foreground: true }).maxPrefillTokens).toBe(
      32
    );
  });

  it('sizes the token budget from the measured native rate', () => {
    const lease = newLease();
    for (const at of [1_000, 1_500]) {
      lease.observe(frame({ p95Ms: 34, over33Ratio: 0.4 }), at);
    }
    // 20 tokens in 100 ms of native decode time: 0.2 tokens/ms
    lease.observeUnit({
      decodeTokens: 20,
      predictedMs: 100,
      elapsedMs: 150,
      budgetMs: 30,
    });

    const active = lease.lease({ now: 1_600, foreground: true });

    // The rate sizes the unit, and the pressure floor keeps it from
    // degenerating when the frame-cadence window is shorter than the
    // tokens it would buy.
    expect(active.maxDecodeTokens).toBe(
      Math.max(
        lease.options.minPressureDecodeTokens,
        Math.min(
          lease.options.maxDecodeTokens,
          Math.floor(active.budgetMs * 0.2)
        )
      )
    );
  });

  it('drops to a local barrier when GPU occupancy breaks the proof', () => {
    const lease = newLease();
    for (const at of [1_000, 1_500]) {
      lease.observe(frame({ p95Ms: 40, over33Ratio: 0.5 }), at);
      lease.observeUnit({ elapsedMs: 30, budgetMs: 18 });
    }

    const active = lease.lease({ now: 1_600, foreground: true });

    expect(active.reason).toBe('queue-backpressure');
    expect(active.maxInflightBatches).toBe(1);
    expect(active.basis).toBe('local-barrier');
    expect(active.commandSubmitBatchSize).toBe(
      lease.options.minCommandBatchSize
    );
    // The browser path has no completion-wait counter, so what the barrier
    // reacts to is residency: the unit ran 12 ms past an 18 ms window.
    const diagnostics = lease.diagnostics({ now: 1_600 });
    expect(diagnostics.occupancyOvershootMs).toBeGreaterThan(
      lease.options.queueWaitToleranceMs
    );
    expect(diagnostics.queueWaitMs).toBeNull();

    // sustained relief releases the barrier instead of latching it
    for (let sample = 0; sample < 3; sample += 1) {
      lease.observe(frame({ p95Ms: 40, over33Ratio: 0.5 }), 1_700 + sample * 100);
      lease.observeUnit({ elapsedMs: 10, budgetMs: 18 });
    }
    const relieved = lease.lease({ now: 2_100, foreground: true });
    expect(relieved.reason).toBe('frame-deadline');
    expect(relieved.maxInflightBatches).toBeGreaterThanOrEqual(1);
  });

  it('releases the barrier when the newest frame samples are clear but the regime is still latched', () => {
    // A long release hysteresis makes the regime label lag behind the frame
    // stream. The barrier must still release once the page is clear; keeping
    // it armed through that lag serialized every decode unit.
    const lease = new FrameDeadlineLease({ maxBudgetMs, recoverSamples: 5 });
    for (const at of [1_000, 1_500]) {
      lease.observe(frame({ p95Ms: 40, over33Ratio: 0.5 }), at);
      lease.observeUnit({ elapsedMs: 30, budgetMs: 18 });
    }
    expect(lease.lease({ now: 1_600, foreground: true }).reason).toBe(
      'queue-backpressure'
    );

    // The first clear sample is still inside the two-sample attack window;
    // the second one clears the smoothed test while the regime remains
    // latched because recovery needs five samples in this lease.
    lease.observe(frame({ p95Ms: 17, over33Ratio: 0 }), 1_700);
    lease.observe(frame({ p95Ms: 17, over33Ratio: 0 }), 1_800);
    for (let unit = 0; unit < 3; unit += 1) {
      lease.observeUnit({ elapsedMs: 60, budgetMs: 18 });
    }

    const relieved = lease.lease({ now: 1_900, foreground: true });
    expect(relieved.reason).toBe('frame-deadline');
    expect(relieved.maxInflightBatches).toBeGreaterThanOrEqual(1);
    expect(relieved.basis).toBe('budget-share');
  });

  it('returns the bounded safety shape when explicitly selected for stale signals', () => {
    const lease = new FrameDeadlineLease({
      maxBudgetMs,
      staleSignalShape: 'bounded',
    });
    lease.observe(frame({ p95Ms: 40, over33Ratio: 0.6 }), 1_000);

    const active = lease.lease({ now: 4_000, foreground: true });

    expect(active.source).toBe('fallback');
    expect(active.reason).toBe('stale-signal');
    expect(active.budgetMs).toBe(lease.options.fallbackBudgetMs);
    expect(lease.diagnostics({ now: 4_000 }).recoveries).toBe(1);
    expect(lease.diagnostics({ now: 4_000 }).lastRecoveryReason).toBe(
      'stale-signal'
    );

    // a fresh signal recovers the deadline path
    for (const at of [4_100, 4_200]) {
      lease.observe(frame({ p95Ms: 40, over33Ratio: 0.6 }), at);
    }
    expect(lease.lease({ now: 4_300, foreground: true }).reason).toBe(
      'frame-deadline'
    );
  });

  it('uses the barriered fallback by default and keeps continuous ceilings', () => {
    const lease = newLease();
    lease.observe(frame({ p95Ms: 40, over33Ratio: 0.6 }), 1_000);

    const active = lease.lease({ now: 4_000, foreground: true });

    expect(active.source).toBe('fallback');
    expect(active.reason).toBe('stale-signal');
    // the native ceilings survive, so a silent signal no longer forces a
    // one-token host round trip per lease
    expect(active.budgetMs).toBe(lease.options.maxBudgetMs);
    // the ceilings survive as ceilings; what a unit may actually hold is the
    // attention quantum, so a lifecycle change is still honoured promptly
    expect(active.maxDecodeTokens).toBe(
      Math.round(
        lease.options.fallbackDecodeTokensPerMs *
          lease.options.attentionQuantumMs
      )
    );
    expect(active.maxPrefillTokens).toBe(
      Math.floor(
        lease.options.attentionQuantumMs /
          lease.options.fallbackPrefillMsPerToken
      )
    );
    // the renderer wait is bounded by the local barrier instead
    expect(active.maxInflightBatches).toBe(1);
    expect(active.basis).toBe('local-barrier');
    expect(active.commandSubmitBatchSize).toBe(
      lease.options.minCommandBatchSize
    );
  });

  it('sidelines a background page without stopping the request', () => {
    const lease = newLease();
    lease.observe(frame(), 1_000);

    const active = lease.lease({ now: 1_100, foreground: false });

    expect(active.source).toBe('background');
    expect(active.reason).toBe('background');
    expect(active.budgetMs).toBe(lease.options.backgroundBudgetMs);
    expect(active.maxInflightBatches).toBe(1);
    expect(active.maxDecodeTokens).toBeGreaterThanOrEqual(1);
  });

  it('keeps an answered priority request in the deadline lane after demotion', () => {
    const lease = newLease();
    lease.observe(frame(), 1_000);

    const answered = lease.lease({
      now: 1_100,
      foreground: false,
      priority: true,
    });

    expect(answered.source).toBe('deadline');
    expect(answered.reason).toBe('work-conserving');
    expect(answered.maxDecodeTokens).toBeGreaterThan(1);

    const firstToken = lease.lease({
      now: 1_101,
      foreground: false,
      handoff: true,
      priority: true,
    });
    expect(firstToken.maxDecodeTokens).toBe(lease.options.handoffDecodeTokens);
  });

  it('needs sustained calm before returning to the work-conserving shape', () => {
    const lease = newLease();
    for (const at of [1_000, 1_500]) {
      lease.observe(frame({ p95Ms: 40, over33Ratio: 0.5 }), at);
    }
    expect(lease.lease({ now: 1_600, foreground: true }).reason).toBe(
      'frame-deadline'
    );

    lease.observe(frame({ p95Ms: 17, over33Ratio: 0 }), 2_000);
    expect(lease.lease({ now: 2_100, foreground: true }).reason).toBe(
      'frame-deadline'
    );

    // the smoothed window still carries the stressed sample until three
    // clear samples have arrived, which is the hysteresis this design
    // wants: one good sample cannot flip the lease back on its own
    lease.observe(frame({ p95Ms: 17, over33Ratio: 0 }), 2_300);
    expect(lease.lease({ now: 2_400, foreground: true }).reason).toBe(
      'frame-deadline'
    );

    lease.observe(frame({ p95Ms: 17, over33Ratio: 0 }), 2_500);
    expect(lease.lease({ now: 2_600, foreground: true }).reason).toBe(
      'work-conserving'
    );
  });

  it('counts the pressured window in display periods, not milliseconds', () => {
    const stressed = frame({
      refreshHz: 120,
      p95Ms: 40,
      p99Ms: 60,
      over33Ratio: 0.5,
    });
    const cadenceOnly = {
      maxBudgetMs,
      // Isolate the display-period term; progress-floor behavior has its
      // own tests below.
      fallbackDecodeTokensPerMs: 1,
    };
    const observed = (lease: FrameDeadlineLease, signal: RenderPressureSignal) => {
      for (const at of [1_000, 1_500]) {
        lease.observe(signal, at);
      }
      return lease.lease({ now: 1_600, foreground: true });
    };

    // A 120 Hz panel receives half the wall time for the same lease: the
    // window is a number of display periods, so it follows the panel and not
    // a configured millisecond.
    const fast = new FrameDeadlineLease(cadenceOnly);
    // Pin the window so the comparison isolates the panel period rather
    // than the tail severity, which differs between the two panels.
    const fastLease = observed(fast, stressed);
    expect(fastLease.frameBudgetMs).toBeCloseTo(8.33, 1);
    // A violated deadline collapses the window towards the severe value,
    // and the result is always a number of display periods between the
    // severe and mild windows.
    expect(fastLease.budgetMs).toBeGreaterThanOrEqual(
      fastLease.frameBudgetMs * fast.options.severePressureFramePeriods - 0.01
    );
    expect(fastLease.budgetMs).toBeLessThanOrEqual(
      fastLease.frameBudgetMs * fast.options.mildPressureFramePeriods + 0.01
    );
    const fixed = (signalFrame: RenderPressureSignal) =>
      observed(
        new FrameDeadlineLease({
          ...cadenceOnly,
          mildPressureFramePeriods: 2,
          severePressureFramePeriods: 2,
        }),
        signalFrame
      );
    const fastFixed = fixed(stressed);
    const slowFixed = fixed(frame({ refreshHz: 60, p95Ms: 40, p99Ms: 60, over33Ratio: 0.5 }));
    expect(fastFixed.budgetMs).toBeCloseTo(fastFixed.frameBudgetMs * 2, 1);
    expect(slowFixed.budgetMs).toBeCloseTo(fastFixed.budgetMs * 2, 1);


    // A wider-window ablation still collapses with the debt, which is what
    // makes the window a deadline rather than a configured quantum.
    const wide = new FrameDeadlineLease({
      ...cadenceOnly,
      mildPressureFramePeriods: 2,
      severePressureFramePeriods: 1,
    });
    const wideStressed = observed(wide, stressed);
    const wideMild = observed(
      new FrameDeadlineLease({
        ...cadenceOnly,
        mildPressureFramePeriods: 2,
        severePressureFramePeriods: 1,
      }),
      frame({ refreshHz: 120, p95Ms: 40, p99Ms: 20, over33Ratio: 0.4 })
    );
    expect(wideMild.reason).toBe('frame-deadline');
    expect(wideMild.budgetMs).toBeCloseTo(
      wideMild.frameBudgetMs * 2,
      1
    );
    expect(wideStressed.budgetMs).toBeGreaterThan(
      wideMild.frameBudgetMs
    );
    expect(wideStressed.budgetMs).toBeLessThan(wideMild.budgetMs);
  });

  it('keeps a shallow bounded queue for mild pressure', () => {
    const lease = newLease();
    for (const at of [1_000, 1_500]) {
      lease.observe(frame({ p95Ms: 34, p99Ms: 40, over33Ratio: 0.3 }), at);
    }

    const active = lease.lease({ now: 1_600, foreground: true });

    expect(active.reason).toBe('frame-deadline');
    expect(active.maxInflightBatches).toBe(2);
    expect(active.commandSubmitBatchSize).toBeGreaterThanOrEqual(8);
  });

  it('does not widen the window to the cost measured while contended', () => {
    const lease = newLease();
    const stalled = frame({
      p95Ms: 80,
      p99Ms: 160,
      over33Ratio: 0.9,
      over50Ratio: 0.8,
    });
    for (const at of [1_000, 1_500]) {
      lease.observe(stalled, at);
    }
    lease.observeUnit({
      reason: 'frame-deadline',
      decodeTokens: 1,
      elapsedMs: 60,
      budgetMs: 40,
    });

    const active = lease.lease({ now: 1_600, foreground: true });

    // The window is the display-period bound of the formula; a host-only
    // sample is not a native cost estimate and cannot widen it.
    expect(active.budgetMs).toBeCloseTo(18.33, 1);
    expect(active.maxDecodeTokens).toBeGreaterThanOrEqual(
      lease.options.minPressureDecodeTokens
    );
    // Host-only samples are not allowed to become a native cost estimate.
    expect(lease.diagnostics({ now: 1_600 }).measuredTokenCostMs).toBeNull();
  });

  it('uses native token cost instead of queue-inclusive host time', () => {
    const lease = newLease();
    for (const at of [1_000, 1_500]) {
      lease.observe(frame({ p95Ms: 34, p99Ms: 40, over33Ratio: 0.3 }), at);
    }
    lease.observeUnit({
      reason: 'frame-deadline',
      decodeTokens: 2,
      predictedMs: 40,
      elapsedMs: 120,
      budgetMs: 30,
    });

    const active = lease.lease({ now: 1_600, foreground: true });

    // The host unit spent 120 ms because it waited on the renderer, but the
    // native prediction cost was 40 ms. The next frame window must not grow
    // from queue wait, or the lease would make its own backlog permanent.
    expect(active.budgetMs).toBeLessThan(50);
    expect(lease.diagnostics({ now: 1_600 }).measuredTokenCostMs).toBe(20);
  });

  it('derives the pressure window and token cap from the formula', () => {
    const lease = newLease();
    for (const at of [1_000, 1_500]) {
      lease.observe(frame({ p95Ms: 34, p99Ms: 40, over33Ratio: 0.3 }), at);
    }
    // Two native tokens took 60 ms under the contended queue: 30 ms/token.
    // The measured cost is telemetry: the wall-time window still comes from
    // the display-period term, and the measured rate only converts it into
    // a token cap.
    lease.observeUnit({
      reason: 'frame-deadline',
      decodeTokens: 2,
      predictedMs: 60,
      elapsedMs: 60,
      budgetMs: 30,
    });

    const active = lease.lease({ now: 1_600, foreground: true });

    expect(active.reason).toBe('frame-deadline');
    expect(active.budgetMs).toBeCloseTo(
      active.frameBudgetMs * lease.options.mildPressureFramePeriods,
      1
    );
    expect(active.maxDecodeTokens).toBe(
      Math.max(
        lease.options.minPressureDecodeTokens,
        Math.floor(active.budgetMs / 30)
      )
    );
  });
  it('returns the GPU to the renderer for as long as the tail is late', () => {
    // The rest interval is an ablation now, so state the cap explicitly.
    const lease = new FrameDeadlineLease({ maxBudgetMs, maxPauseMs: 48 });
    // tail deadline is five frame periods (83.33 ms at 60 Hz); a 90 ms P99
    // is 6.67 ms past it, so the rounded pause is 7 ms of empty-queue time.
    for (const at of [1_000, 1_500]) {
      lease.observe(
        frame({ p95Ms: 70, p99Ms: 90, over33Ratio: 0.6, over50Ratio: 0.3 }),
        at
      );
    }

    const active = lease.lease({ now: 1_600, foreground: true });

    expect(active.reason).toBe('frame-deadline');
    expect(active.submissionPauseMs).toBe(7);
    expect(active.submissionPauseMs).toBeLessThanOrEqual(
      lease.options.maxPauseMs
    );
    expect(lease.diagnostics({ now: 1_600 }).submissionPauseMs).toBe(
      active.submissionPauseMs
    );
  });

  it('bounds the default window by display periods with a short bounded rest', () => {
    const lease = newLease();
    for (const at of [1_000, 1_500]) {
      lease.observe(
        frame({ p95Ms: 70, p99Ms: 90, over33Ratio: 0.6, over50Ratio: 0.3 }),
        at
      );
    }

    const active = lease.lease({ now: 1_600, foreground: true });

    // The severe tail collapses the window towards the one-period floor but
    // never below it, and the mild default is two periods: a window shorter
    // than one native batch only adds unit overhead. A short bounded rest
    // gives the compositor an empty-queue window without stopping progress.
    expect(active.reason).toBe('frame-deadline');
    expect(active.budgetMs).toBeGreaterThan(active.frameBudgetMs);
    expect(active.budgetMs).toBeLessThanOrEqual(
      active.frameBudgetMs * lease.options.mildPressureFramePeriods + 0.01
    );
    expect(active.submissionPauseMs).toBeGreaterThan(0);
    expect(active.submissionPauseMs).toBeLessThanOrEqual(
      lease.options.maxPauseMs
    );
    expect(lease.options.maxPauseMs).toBe(16);

    // The one-period arm stays reachable as an explicit ablation.
    const narrow = new FrameDeadlineLease({
      maxBudgetMs,
      mildPressureFramePeriods: 1,
      severePressureFramePeriods: 1,
    });
    for (const at of [1_000, 1_500]) {
      narrow.observe(
        frame({ p95Ms: 70, p99Ms: 90, over33Ratio: 0.6, over50Ratio: 0.3 }),
        at
      );
    }
    const narrowLease = narrow.lease({ now: 1_600, foreground: true });
    expect(narrowLease.budgetMs).toBeCloseTo(
      narrowLease.frameBudgetMs,
      1
    );
  });

  it('does not pause a page whose frames are inside the deadline', () => {
    const lease = newLease();
    lease.observe(frame(), 1_000);
    lease.observe(frame(), 1_500);

    const active = lease.lease({ now: 1_600, foreground: true });

    expect(active.reason).toBe('work-conserving');
    expect(active.submissionPauseMs).toBe(0);
  });

  it('shrinks the submission once the deadline is clearly violated', () => {
    const lease = newLease();
    for (const at of [1_000, 1_500]) {
      lease.observe(
        frame({ p95Ms: 90, p99Ms: 120, over33Ratio: 0.9, over50Ratio: 0.7 }),
        at
      );
    }

    const active = lease.lease({ now: 1_600, foreground: true });

    expect(active.reason).toBe('frame-deadline');
    expect(active.commandSubmitBatchSize).toBeLessThanOrEqual(
      lease.options.maxCommandBatchSize / 2
    );
    expect(active.commandSubmitBatchSize).toBeGreaterThanOrEqual(
      lease.options.minCommandBatchSize
    );
  });

  it('keeps the pressure token floor when a token costs more than the window', () => {
    const lease = newLease();
    const stalled = frame({
      p95Ms: 80,
      p99Ms: 160,
      over33Ratio: 0.9,
      over50Ratio: 0.8,
    });
    for (const at of [1_000, 1_500]) {
      lease.observe(stalled, at);
    }

    const active = lease.lease({ now: 1_600, foreground: true });

    // The window itself is shorter than the fallback cost of one token
    // (1 / 0.04 = 25 ms); the token floor is what keeps the unit from
    // degenerating, and the native budget closes it at the token boundary.
    expect(active.budgetMs).toBeCloseTo(18.33, 1);
    expect(active.maxDecodeTokens).toBeGreaterThanOrEqual(
      lease.options.minPressureDecodeTokens
    );
  });

  it('reports lease diagnostics for the harness', () => {
    const lease = newLease();
    lease.observe(frame({ p95Ms: 40, over33Ratio: 0.5 }), 1_000);
    lease.lease({ now: 1_100, foreground: true });

    const diagnostics = lease.diagnostics({ now: 1_200 });

    expect(diagnostics.leaseId).toBeGreaterThan(0);
    expect(diagnostics.refreshHz).toBe(60);
    expect(diagnostics.signalAgeMs).toBe(200);
    expect(diagnostics.frameBudgetMs).toBeCloseTo(16.67, 1);
    expect(diagnostics.queue).toEqual({
      commandSubmitBatchSize: expect.any(Number),
      maxInflightBatches: expect.any(Number),
      basis: expect.any(String),
    });
  });
});
