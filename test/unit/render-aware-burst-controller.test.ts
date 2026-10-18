import { describe, expect, it } from 'vitest';
import {
  RenderAwareBurstController,
  unitSizingForLevel,
} from '../../src/execution/render-aware-burst-controller';
import type { RenderPressureSignal } from '../../src/contracts';

function pressure(
  p95Ms: number,
  overBudgetRatio: number,
  frameCount = 8
): RenderPressureSignal {
  return {
    clientId: 'c1',
    epoch: 1,
    sampledAt: 0,
    source: 'presented-frame',
    frameCount,
    p50Ms: 16,
    p95Ms,
    maxMs: p95Ms,
    overBudgetRatio,
    p99Ms: p95Ms,
    over33Ratio: overBudgetRatio,
    over50Ratio: overBudgetRatio,
    longestStallMs: p95Ms,
  };
}

describe('RenderAwareBurstController', () => {
  it('shrinks the GPU submission shape as pressure persists', () => {
    const controller = new RenderAwareBurstController();
    expect(controller.policy).toEqual({
      level: 'normal',
      commandSubmitBatchSize: 64,
      maxInflightBatches: 0,
      submissionPauseMs: 0,
    });

    controller.observe(pressure(50, 0.4), 1_000);
    controller.observe(pressure(50, 0.4), 1_500);
    expect(controller.policy).toEqual({
      level: 'guarded',
      commandSubmitBatchSize: 32,
      maxInflightBatches: 4,
      submissionPauseMs: 0,
    });

    controller.observe(pressure(50, 0.4), 2_000);
    controller.observe(pressure(50, 0.4), 2_500);
    expect(controller.policy).toEqual({
      level: 'constrained',
      commandSubmitBatchSize: 8,
      maxInflightBatches: 2,
      submissionPauseMs: 0,
    });
  });

  it('bounds unit shape harder as the level tightens', () => {
    const base = {
      maxPrefillTokens: 512,
      baseDecodeTokens: 12,
      foregroundDecodeTokens: 12,
      backgroundDecodeTokens: 4,
      prefillMsPerToken: 2,
      maxUnitWallTimeMs: 40,
      unitWallTimeGraceMs: 1_000,
      adaptiveUnits: true,
      targetUnitWallTimeMs: 24,
      nativeTimeBudgetEnabled: true,
      minDecodeTokens: 1,
      maxDecodeTokens: 1024,
    };
    const normal = unitSizingForLevel('normal', base);
    const guarded = unitSizingForLevel('guarded', base);
    const constrained = unitSizingForLevel('constrained', base);

    expect(normal).toEqual({});
    // an unsliced base is capped rather than merely halved
    expect(guarded.maxPrefillTokens).toBe(16);
    expect(constrained.maxPrefillTokens).toBe(8);
    expect(guarded.maxDecodeTokens).toBe(16);
    expect(constrained.maxDecodeTokens).toBe(8);
    expect(constrained.foregroundDecodeTokens!).toBeLessThan(
      guarded.foregroundDecodeTokens!
    );
    expect(constrained.maxPrefillTokens!).toBeLessThan(
      guarded.maxPrefillTokens!
    );
    expect(constrained.targetUnitWallTimeMs!).toBeLessThan(
      guarded.targetUnitWallTimeMs!
    );
  });

  it('keeps pacing inert unless it is explicitly enabled', () => {
    const inert = new RenderAwareBurstController();
    inert.observe(pressure(50, 0.4), 1_000);
    inert.observe(pressure(50, 0.4), 1_500);
    expect(inert.policy.submissionPauseMs).toBe(0);

    const paced = new RenderAwareBurstController({ pauseOnPressure: true });
    paced.observe(pressure(50, 0.4), 1_000);
    paced.observe(pressure(50, 0.4), 1_500);
    expect(paced.policy.submissionPauseMs).toBe(32);
  });

  it('stays at normal through the monitor noise band', () => {
    const controller = new RenderAwareBurstController();
    // Anchor-device ambient readings: a page that is only holding its own
    // animation reports this band. Treating it as pressure pinned the gate
    // shut before the first trial and held one request for 284 s.
    for (let sample = 0; sample < 20; sample += 1) {
      expect(
        controller.observe(pressure(26, 0.31), 1_000 + sample * 500).level
      ).toBe('normal');
    }

    expect(controller.observe(pressure(52, 0.8), 20_000).level).toBe(
      'normal'
    );
    expect(controller.observe(pressure(52, 0.8), 20_500).level).toBe(
      'guarded'
    );
  });

  it('requires two stressed samples before downgrading', () => {
    const controller = new RenderAwareBurstController();

    expect(controller.observe(pressure(50, 0.4), 1_000).level).toBe('normal');
    expect(controller.observe(pressure(50, 0.4), 1_500).level).toBe('guarded');
    expect(controller.observe(pressure(50, 0.4), 2_000).level).toBe('guarded');
    expect(controller.observe(pressure(50, 0.4), 2_500).level).toBe(
      'constrained'
    );
  });

  it('recovers one level after two clear samples', () => {
    const controller = new RenderAwareBurstController();
    controller.observe(pressure(50, 0.4), 1_000);
    controller.observe(pressure(50, 0.4), 1_500);

    // The stressed samples stay inside the median-of-three window, so the
    // first clear sample is still read as stressed and does not count.
    expect(controller.observe(pressure(18, 0), 2_000).level).toBe('guarded');
    expect(controller.observe(pressure(18, 0), 2_500).level).toBe('guarded');
    expect(controller.observe(pressure(18, 0), 3_000).level).toBe('normal');
  });

  it('returns to the fast shape inside one clear window', () => {
    const controller = new RenderAwareBurstController();
    for (let sample = 0; sample < 4; sample += 1) {
      controller.observe(pressure(50, 0.8), 1_000 + sample * 500);
    }
    expect(controller.policy.level).toBe('constrained');

    // The contention schedule holds pressure for 4 s and then clears for
    // 6 s, so a recovery that outlives the clear window leaves the next
    // window measured against a shrunken unit shape.
    let now = 3_000;
    const clearWindowEnds = 9_000;
    while (controller.policy.level !== 'normal' && now < clearWindowEnds) {
      now += 500;
      controller.observe(pressure(19, 0), now);
    }
    expect(controller.policy.level).toBe('normal');
    expect(now).toBeLessThan(clearWindowEnds);
  });

  it('resets to normal after the signal becomes stale', () => {
    const controller = new RenderAwareBurstController();
    controller.observe(pressure(50, 0.4), 1_000);
    controller.observe(pressure(50, 0.4), 1_500);
    expect(controller.policy.level).toBe('guarded');

    expect(controller.release(2_999).level).toBe('guarded');
    expect(controller.release(3_001).level).toBe('normal');
  });

  it('ignores malformed or too-small samples', () => {
    const controller = new RenderAwareBurstController();
    const invalid = [
      pressure(Number.NaN, 0.4),
      pressure(50, Number.POSITIVE_INFINITY),
      pressure(50, 0.4, 2),
    ];

    for (const [index, signal] of invalid.entries()) {
      expect(controller.observe(signal, 1_000 + index).level).toBe('normal');
    }
    expect(controller.release(10_000).level).toBe('normal');
  });

  it('engages on frame-budget pressure instead of half cadence', () => {
    const controller = new RenderAwareBurstController();

    // The monitor reports 19-27 ms P95 with up to a 0.31 over-budget ratio on
    // an idle anchor-device page, so 26 ms is inside the noise band rather
    // than evidence of contention. The earlier 25 ms trigger engaged here and
    // held one request for 284 s.
    expect(controller.observe(pressure(26, 0.2), 1_000).level).toBe('normal');
    expect(controller.observe(pressure(26, 0.2), 1_500).level).toBe('normal');

    // Contention measured in the same probe reaches 50-63 ms P95.
    const stressed = new RenderAwareBurstController();
    expect(stressed.observe(pressure(52, 0.8), 1_000).level).toBe('normal');
    expect(stressed.observe(pressure(52, 0.8), 1_500).level).toBe('guarded');

    const settled = new RenderAwareBurstController();
    expect(settled.observe(pressure(19, 0), 1_000).level).toBe('normal');
  });
});
