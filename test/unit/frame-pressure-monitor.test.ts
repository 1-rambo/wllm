import { describe, expect, it } from 'vitest';
import {
  adoptRefreshCadence,
  estimateRefreshHz,
  summarizeFramePressure,
  type RefreshCadenceState,
} from '../../src/client/frame-pressure-monitor';

describe('summarizeFramePressure', () => {
  it('summarizes presented-frame intervals inside the sampling window', () => {
    const sample = summarizeFramePressure(
      [800, 1000, 1016, 1032, 1064],
      1100,
      200,
      33,
      'presented-frame'
    );

    expect(sample).toMatchObject({
      source: 'presented-frame',
      frameCount: 3,
      p50Ms: 16,
      p95Ms: 32,
      maxMs: 32,
      overBudgetRatio: 0,
      p99Ms: 32,
      over33Ratio: 0,
      over50Ratio: 0,
      longestStallMs: 0,
    });
  });

  it('reports the over-budget ratio from presented intervals', () => {
    const sample = summarizeFramePressure(
      [1000, 1016, 1032, 1082],
      1100,
      500,
      33,
      'presented-frame'
    );

    expect(sample?.overBudgetRatio).toBeCloseTo(1 / 3, 6);
    expect(sample?.p95Ms).toBe(50);
    // the 50 ms frame is one over-budget stall, and it is also over 33 ms
    expect(sample?.over33Ratio).toBeCloseTo(1 / 3, 6);
    expect(sample?.over50Ratio).toBe(0);
    expect(sample?.longestStallMs).toBe(50);
  });

  it('returns null for an empty rAF window', () => {
    expect(
      summarizeFramePressure([], 1100, 500, 33, 'animation-frame')
    ).toBeNull();
  });

  it('treats a stalled presented stream as pressure, not silence', () => {
    expect(
      summarizeFramePressure([], 1100, 500, 33, 'presented-frame')
    ).toMatchObject({
      frameCount: 1,
      p95Ms: 500,
      overBudgetRatio: 1,
      longestStallMs: 500,
    });
  });
});

const windowOf = (intervalMs: number, count: number): number[] =>
  Array.from({ length: count }, () => intervalMs);

describe('estimateRefreshHz', () => {
  it('reads the panel cadence, not a read burst of queued frames', () => {
    // A stalled MediaStreamTrackProcessor drains its queue back to back, so
    // a 60 Hz window can contain a handful of near-zero intervals. The
    // fastest tail is not evidence that the display is 240 Hz.
    const intervals = [...windowOf(16.7, 20), 0.4, 0.6, 0.9].sort(
      (a, b) => a - b
    );

    expect(estimateRefreshHz(intervals)).toBe(60);
  });

  it('reports a high-refresh panel when the window sits on its period', () => {
    expect(estimateRefreshHz(windowOf(8.33, 24))).toBe(120);
  });

  it('reports no cadence when most frames were dropped', () => {
    // 33 ms intervals are two 60 Hz periods. Reading them as a 30 Hz panel
    // would double the frame budget exactly when frames are being missed.
    expect(estimateRefreshHz(windowOf(33.3, 30))).toBeNull();
  });

  it('needs a window long enough to be a distribution', () => {
    expect(estimateRefreshHz(windowOf(16.7, 4))).toBeNull();
  });
});

describe('adoptRefreshCadence', () => {
  it('ignores a single window and adopts a repeated cadence', () => {
    const empty = { estimate: null, candidate: null, count: 0 };
    const once = adoptRefreshCadence(empty, 240);
    expect(once.estimate).toBeNull();

    const twice = adoptRefreshCadence(once, 240);
    expect(twice.estimate).toBe(240);
  });

  it('keeps the estimate through a window with no measurement', () => {
    const settled = { estimate: 60, candidate: null, count: 0 };
    expect(adoptRefreshCadence(settled, null).estimate).toBe(60);
  });

  it('revises the estimate instead of ratcheting to the fastest rate', () => {
    let state: RefreshCadenceState = { estimate: 60, candidate: null, count: 0 };
    // A burst window hints at 240 Hz once; adopting it would pin the frame
    // budget at a rate the display never presented at.
    state = adoptRefreshCadence(state, 240);
    state = adoptRefreshCadence(state, 120);
    expect(state.estimate).toBe(60);

    state = adoptRefreshCadence(state, 120);
    expect(state.estimate).toBe(120);
  });
});

describe('frame sample ownership', () => {
  it('marks a page-canvas window as page owned', () => {
    const sample = summarizeFramePressure(
      [1000, 1016, 1032, 1064],
      1100,
      500,
      33,
      'presented-frame',
      60,
      'stall',
      true
    );

    expect(sample?.pageOwned).toBe(true);
  });

  it('leaves the helper canvas and the rAF fallback unowned', () => {
    const helper = summarizeFramePressure([], 1100, 500, 33, 'presented-frame');
    expect(helper).toMatchObject({ pageOwned: false, frameCount: 1 });

    const raf = summarizeFramePressure(
      [1000, 1016, 1032],
      1100,
      500,
      33,
      'animation-frame'
    );
    expect(raf?.pageOwned).toBe(false);
  });
});
