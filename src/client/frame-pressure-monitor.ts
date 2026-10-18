import type { RenderPressureSource } from '../contracts';

// A sample carries the frame tail, not just the median: one 80 ms stall is
// invisible in a P95 but is exactly the event a deadline lease has to react
// to. refreshHz is the monitor's best estimate of the display cadence.
export interface FramePressureSample {
  source: RenderPressureSource;
  frameCount: number;
  p50Ms: number;
  p95Ms: number;
  maxMs: number;
  overBudgetRatio: number;
  p99Ms: number;
  over33Ratio: number;
  over50Ratio: number;
  longestStallMs: number;
  refreshHz?: number;
  // False when the sample came from the monitor's own helper canvas or the
  // requestAnimationFrame fallback. A controller must not read the page's
  // own queueing back out of a canvas the monitor paints by itself.
  pageOwned: boolean;
}

export interface FramePressureMonitorOptions {
  onSample: (sample: FramePressureSample) => void;
  intervalMs?: number;
  overBudgetMs?: number;
  // Prefer the page's own render canvas. The helper canvas is a second,
  // continuously painted source used only when the page canvas is static.
  canvas?: () => HTMLCanvasElement | null | undefined;
  // Retained for hosts that explicitly know an empty page canvas is idle.
  // The helper channel normally makes this unnecessary.
  idlePresentationWhenNoFrames?: boolean;
}

export interface FramePressureMonitor {
  start(): void;
  // Whole-run tail over every retained frame interval. `onSample` is the
  // rolling controller signal; the experiment record needs this longer
  // view so a lucky final 500 ms cannot hide a stalled middle of a run.
  snapshot(): FramePressureSample | null;
  stop(): void;
}

interface VideoFrameLike {
  close(): void;
}

interface VideoFrameReader {
  read(): Promise<{ done: boolean; value?: VideoFrameLike }>;
  cancel(): Promise<void>;
}

interface VideoFrameProcessorConstructor {
  new (options: { track: MediaStreamTrack }): {
    readable: { getReader(): VideoFrameReader };
  };
}

interface CaptureChannel {
  target: HTMLCanvasElement;
  pageOwned: boolean;
  stream: MediaStream | null;
  reader: VideoFrameReader | null;
  timestamps: number[];
  source: RenderPressureSource;
  startedAt: number;
  generation: number;
}

/**
 * Samples compositor presentation from the page canvas and a continuously
 * painted helper canvas. The page canvas wins whenever it presents frames;
 * the helper keeps the signal alive when a real application canvas
 * is intentionally static between paints. A stalled presented stream is
 * reported as pressure rather than silently disappearing. requestAnimation
 * Frame is only the final fallback when MediaStreamTrackProcessor is absent.
 */
export function observeFramePressure(
  options: FramePressureMonitorOptions
): FramePressureMonitor {
  // The lease requires two pressure samples before it downgrades, so a
  // 500 ms aggregation window made a step into contention take almost a
  // second to reach the native queue policy. Keep enough frames for a
  // cadence estimate while bounding that control latency.
  const intervalMs = options.intervalMs ?? 100;
  // 20 ms is one 60 Hz frame plus tolerance. A 33 ms budget only flagged a
  // display that had already halved its cadence.
  const overBudgetMs = options.overBudgetMs ?? 20;
  let running = false;
  let raf = 0;
  let refreshCadence: RefreshCadenceState = {
    estimate: null,
    candidate: null,
    count: 0,
  };
  // A cadence only becomes the estimate once consecutive windows agree on
  // it. A single window can contain a read burst (a stalled reader drains
  // several queued VideoFrames back to back), and adopting that burst as a
  // fast display would shrink the frame budget for the rest of the run.
  let timer: ReturnType<typeof setInterval> | null = null;
  let helperCanvas: HTMLCanvasElement | null = null;
  let helperContext: CanvasRenderingContext2D | null = null;
  let pageChannel: CaptureChannel | null = null;
  let helperChannel: CaptureChannel | null = null;
  const rafTimestamps: number[] = [];
  let phase = 0;

  const adoptRefresh = (observed: number | null): void => {
    refreshCadence = adoptRefreshCadence(refreshCadence, observed);
  };

  const ensureHelperCanvas = (): HTMLCanvasElement => {
    if (helperCanvas) {
      return helperCanvas;
    }
    helperCanvas = document.createElement('canvas');
    helperCanvas.width = 2;
    helperCanvas.height = 2;
    helperCanvas.setAttribute('aria-hidden', 'true');
    helperCanvas.style.cssText =
      'position:fixed;right:0;bottom:0;width:2px;height:2px;opacity:0.01;pointer-events:none;z-index:2147483646;';
    document.body.append(helperCanvas);
    helperContext = helperCanvas.getContext('2d');
    return helperCanvas;
  };

  const detachChannel = (channel: CaptureChannel): void => {
    channel.generation += 1;
    void channel.reader?.cancel().catch(() => undefined);
    channel.reader = null;
    channel.stream?.getTracks().forEach((track) => track.stop());
    channel.stream = null;
    channel.timestamps = [];
    channel.startedAt = 0;
  };

  const readPresentedFrames = async (
    channel: CaptureChannel,
    frameReader: VideoFrameReader,
    generation: number
  ): Promise<void> => {
    try {
      while (running && generation === channel.generation) {
        const { done, value } = await frameReader.read();
        if (done || generation !== channel.generation) {
          break;
        }
        channel.timestamps.push(performance.now());
        capTimestamps(channel.timestamps);
        value?.close();
      }
    } catch {
      if (generation === channel.generation) {
        channel.source = 'animation-frame';
        channel.reader = null;
      }
    }
  };

  const attachChannel = (
    channel: CaptureChannel,
    target: HTMLCanvasElement
  ): void => {
    detachChannel(channel);
    channel.target = target;
    channel.startedAt = performance.now();
    channel.source = 'animation-frame';
    const processor = (
      globalThis as unknown as {
        MediaStreamTrackProcessor?: VideoFrameProcessorConstructor;
      }
    ).MediaStreamTrackProcessor;
    if (!processor || typeof target.captureStream !== 'function') {
      return;
    }
    try {
      channel.stream = target.captureStream(60);
      const track = channel.stream.getVideoTracks()[0];
      if (!track) {
        channel.stream = null;
        return;
      }
      const generation = channel.generation;
      channel.reader = new processor({ track }).readable.getReader();
      channel.source = 'presented-frame';
      void readPresentedFrames(channel, channel.reader, generation);
    } catch {
      channel.reader = null;
      channel.stream?.getTracks().forEach((track) => track.stop());
      channel.stream = null;
      channel.source = 'animation-frame';
    }
  };

  const createChannel = (
    target: HTMLCanvasElement,
    pageOwned: boolean
  ): CaptureChannel => {
    const channel: CaptureChannel = {
      target,
      pageOwned,
      stream: null,
      reader: null,
      timestamps: [],
      source: 'animation-frame',
      startedAt: 0,
      generation: 0,
    };
    attachChannel(channel, target);
    return channel;
  };

  const ensureChannels = (): void => {
    const pageTarget = options.canvas?.() ?? null;
    if (pageTarget) {
      if (!pageChannel || pageChannel.target !== pageTarget) {
        if (pageChannel) {
          detachChannel(pageChannel);
        }
        pageChannel = createChannel(pageTarget, true);
      }
    } else if (pageChannel) {
      detachChannel(pageChannel);
      pageChannel = null;
    }

    const helper = ensureHelperCanvas();
    if (!helperChannel) {
      helperChannel = createChannel(helper, false);
    } else if (helperChannel.target !== helper) {
      detachChannel(helperChannel);
      helperChannel = createChannel(helper, false);
    }
  };

  const start = (): void => {
    if (running || typeof document === 'undefined') {
      return;
    }
    running = true;
    refreshCadence = { estimate: null, candidate: null, count: 0 };
    rafTimestamps.length = 0;
    ensureChannels();

    const draw = (now: number): void => {
      if (!running) {
        return;
      }
      phase = (phase + 1) % 4;
      if (helperContext) {
        helperContext.fillStyle =
          phase % 2 === 0 ? 'rgb(20,100,180)' : 'rgb(80,180,120)';
        helperContext.fillRect(0, 0, 2, 2);
      }
      if (
        (pageChannel?.source ?? 'animation-frame') === 'animation-frame' &&
        (helperChannel?.source ?? 'animation-frame') === 'animation-frame'
      ) {
        rafTimestamps.push(now);
        capTimestamps(rafTimestamps);
      }
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);

    timer = setInterval(() => {
      emitSample(performance.now());
    }, intervalMs);
  };

  const selectChannel = (now: number): CaptureChannel | null => {
    const pageFrames = pageChannel ? recentFrameCount(pageChannel, now) : 0;
    const helperFrames = helperChannel
      ? recentFrameCount(helperChannel, now)
      : 0;
    if (pageChannel && pageFrames >= 2) {
      return pageChannel;
    }
    if (helperChannel && helperFrames >= 2) {
      return helperChannel;
    }
    // A single page frame is still evidence that the real canvas is alive.
    if (pageChannel?.source === 'presented-frame' && pageFrames > 0) {
      return pageChannel;
    }
    // A helper with a presented stream but no frames is a genuine stall. This
    // is the last fallback before returning an empty rAF sample.
    if (helperChannel?.source === 'presented-frame') {
      return helperChannel;
    }
    if (pageChannel?.source === 'presented-frame') {
      return pageChannel;
    }
    return null;
  };

  const emitSample = (now: number): void => {
    ensureChannels();
    const selected = selectChannel(now);
    let sample: FramePressureSample | null = null;
    if (selected) {
      adoptRefresh(estimateRefreshHzFromTimestamps(selected.timestamps));
      sample = summarizeFramePressure(
        selected.timestamps,
        now,
        intervalMs,
        overBudgetMs,
        selected.source,
        refreshCadence.estimate,
        options.idlePresentationWhenNoFrames === true && selected.pageOwned
          ? 'idle'
          : 'stall',
        selected.pageOwned
      );
    } else {
      adoptRefresh(estimateRefreshHzFromTimestamps(rafTimestamps));
      sample = summarizeFramePressure(
        rafTimestamps,
        now,
        intervalMs,
        overBudgetMs,
        'animation-frame',
        refreshCadence.estimate
      );
    }
    if (sample) {
      options.onSample(sample);
    }
  };

  const snapshot = (): FramePressureSample | null => {
    const timestamped = [pageChannel, helperChannel]
      .filter((channel): channel is CaptureChannel => channel !== null)
      .map((channel) => ({ channel, timestamps: channel.timestamps }))
      .filter((candidate) => candidate.timestamps.length >= 2)
      .sort((left, right) => right.timestamps.length - left.timestamps.length);
    const selected = timestamped[0]?.channel ?? null;
    if (selected !== null) {
      adoptRefresh(estimateRefreshHzFromTimestamps(selected.timestamps));
      return summarizeFramePressure(
        selected.timestamps,
        performance.now(),
        Number.MAX_SAFE_INTEGER,
        overBudgetMs,
        selected.source,
        refreshCadence.estimate,
        'stall',
        selected.pageOwned
      );
    }
    if (rafTimestamps.length < 2) {
      return null;
    }
    return summarizeFramePressure(
      rafTimestamps,
      performance.now(),
      Number.MAX_SAFE_INTEGER,
      overBudgetMs,
      'animation-frame',
      refreshCadence.estimate
    );
  };

  const stop = (): void => {
    running = false;
    cancelAnimationFrame(raf);
    if (timer !== null) {
      clearInterval(timer);
      timer = null;
    }
    if (pageChannel) {
      detachChannel(pageChannel);
      pageChannel = null;
    }
    if (helperChannel) {
      detachChannel(helperChannel);
      helperChannel = null;
    }
    rafTimestamps.length = 0;
    helperCanvas?.remove();
    helperCanvas = null;
    helperContext = null;
  };

  return { start, snapshot, stop };
}

export const FRAME_BUDGET_33_MS = 33.3;
export const FRAME_BUDGET_50_MS = 50;

// How many consecutive sampling windows have to report the same cadence
// before the monitor replaces its estimate, and how close two reports have
// to be to count as the same rate.
export interface RefreshCadenceState {
  estimate: number | null;
  candidate: number | null;
  count: number;
}

// A candidate cadence is adopted only after consecutive windows agree on
// it. Replacing the estimate in place, rather than ratcheting to the fastest
// rate ever seen, is also what lets a panel that changes cadence be
// re-measured instead of pinning the frame budget at a stale estimate.
export function adoptRefreshCadence(
  state: RefreshCadenceState,
  observed: number | null
): RefreshCadenceState {
  if (observed === null) {
    return { ...state, candidate: null, count: 0 };
  }
  if (
    state.estimate !== null &&
    Math.abs(observed - state.estimate) <= REFRESH_CONFIRM_HZ
  ) {
    return { ...state, candidate: null, count: 0 };
  }
  const count =
    state.candidate !== null &&
    Math.abs(observed - state.candidate) <= REFRESH_CONFIRM_HZ
      ? state.count + 1
      : 1;
  if (count >= REFRESH_CONFIRMATIONS) {
    return { estimate: observed, candidate: null, count: 0 };
  }
  return { ...state, candidate: observed, count };
}
const REFRESH_CONFIRMATIONS = 2;
const REFRESH_CONFIRM_HZ = 2;

export function summarizeFramePressure(
  timestamps: readonly number[],
  now: number,
  intervalMs: number,
  overBudgetMs: number,
  source: RenderPressureSource,
  refreshHz?: number | null,
  emptyPresentation: 'stall' | 'idle' = 'stall',
  pageOwned = false
): FramePressureSample | null {
  const recent = timestamps.filter(
    (value) => value >= now - intervalMs && value <= now
  );
  const intervals: number[] = [];
  for (let index = 1; index < recent.length; index += 1) {
    const delta = recent[index]! - recent[index - 1]!;
    if (delta > 0 && delta < 2_000) {
      intervals.push(delta);
    }
  }

  if (intervals.length === 0) {
    if (source !== 'presented-frame') {
      return null;
    }
    if (emptyPresentation === 'idle') {
      return {
        source,
        pageOwned,
        frameCount: 0,
        p50Ms: 0,
        p95Ms: 0,
        maxMs: 0,
        overBudgetRatio: 0,
        p99Ms: 0,
        over33Ratio: 0,
        over50Ratio: 0,
        longestStallMs: 0,
        ...(refreshHz ? { refreshHz } : {}),
      };
    }
    return summarizeFrameIntervals(
      [intervalMs],
      overBudgetMs,
      source,
      refreshHz,
      pageOwned
    );
  }

  return summarizeFrameIntervals(
    intervals,
    overBudgetMs,
    source,
    refreshHz,
    pageOwned
  );
}

// Summarize already-computed frame intervals. The evaluation relay delivers
// intervals rather than timestamps because it samples another
// browser context; keeping one summarizer makes its pressure decision and the
// page monitor's decision use the same percentiles and stall rules.
export function summarizeFrameIntervals(
  intervals: readonly number[],
  overBudgetMs: number,
  source: RenderPressureSource,
  refreshHz?: number | null,
  pageOwned = false
): FramePressureSample | null {
  if (intervals.length === 0) {
    return null;
  }
  const sorted = [...intervals].sort((a, b) => a - b);
  const estimatedHz = refreshHz ?? estimateRefreshHz(sorted);
  return {
    source,
    pageOwned,
    frameCount: intervals.length,
    p50Ms: percentile(sorted, 0.5),
    p95Ms: percentile(sorted, 0.95),
    p99Ms: percentile(sorted, 0.99),
    maxMs: sorted.at(-1) ?? 0,
    overBudgetRatio:
      intervals.filter((value) => value > overBudgetMs).length /
      intervals.length,
    over33Ratio:
      intervals.filter((value) => value > FRAME_BUDGET_33_MS).length /
      intervals.length,
    over50Ratio:
      intervals.filter((value) => value > FRAME_BUDGET_50_MS).length /
      intervals.length,
    longestStallMs: longestStallMs(intervals, overBudgetMs),
    ...(estimatedHz === null ? {} : { refreshHz: estimatedHz }),
  };
}

// The longest uninterrupted stretch of over-budget frames, measured as the
// wall time it covers. A run of five 30 ms frames is one 150 ms stall; a
// single spike is one frame long.
export function longestStallMs(
  intervals: readonly number[],
  overBudgetMs: number
): number {
  let longest = 0;
  let run = 0;
  for (const value of intervals) {
    if (value > overBudgetMs) {
      run += value;
      longest = Math.max(longest, run);
    } else {
      run = 0;
    }
  }
  return longest;
}

// The display period is the cadence the page presents most of its frames at,
// so the estimate reads a low quantile of the frame intervals rather than the
// fastest one. The fastest tail is not evidence about the panel: a stalled
// reader drains several queued VideoFrames back to back when it catches up,
// which shows up as a few sub-millisecond intervals and, taken as the panel
// rate, shrinks the frame budget for the rest of the run.
//
// A candidate is only reported when enough of the window actually sits on
// that period. A page that drops most of its frames has intervals that are
// whole multiples of the period, and the support test is what stops those
// multiples from being read as a slower display: without support the window
// reports no cadence, which leaves the controller on its configured fallback
// instead of widening the deadline.
export function estimateRefreshHz(
  sortedIntervals: readonly number[]
): number | null {
  if (sortedIntervals.length < 6) {
    return null;
  }
  const index = Math.min(
    sortedIntervals.length - 1,
    Math.floor(sortedIntervals.length * 0.2)
  );
  const fast = sortedIntervals[index];
  if (fast === undefined || fast <= 0) {
    return null;
  }
  const raw = 1_000 / fast;
  let best = raw;
  let bestDelta = Number.POSITIVE_INFINITY;
  for (const candidate of [60, 75, 90, 100, 120, 144, 165, 240]) {
    const delta = Math.abs(candidate - raw);
    if (delta < bestDelta) {
      bestDelta = delta;
      best = candidate;
    }
  }
  if (bestDelta / best > 0.12) {
    best = Math.round(raw);
  }
  const clamped = Math.min(240, Math.max(60, best));
  const periodMs = 1_000 / clamped;
  const toleranceMs = periodMs * 0.15;
  const support = sortedIntervals.filter(
    (interval) => Math.abs(interval - periodMs) <= toleranceMs
  ).length;
  if (support / sortedIntervals.length < 0.2) {
    return null;
  }
  return clamped;
}

function estimateRefreshHzFromTimestamps(
  timestamps: readonly number[]
): number | null {
  if (timestamps.length < 5) {
    return null;
  }
  const intervals: number[] = [];
  for (let index = 1; index < timestamps.length; index += 1) {
    const delta = timestamps[index]! - timestamps[index - 1]!;
    if (delta > 0 && delta < 2_000) {
      intervals.push(delta);
    }
  }
  return estimateRefreshHz(intervals.sort((a, b) => a - b));
}

function recentFrameCount(channel: CaptureChannel, now: number): number {
  let count = 0;
  for (let index = channel.timestamps.length - 1; index >= 0; index -= 1) {
    const timestamp = channel.timestamps[index]!;
    if (timestamp < now - 500) {
      break;
    }
    count += 1;
  }
  return count;
}

function capTimestamps(timestamps: number[]): void {
  // The rolling controller only reads the last 500 ms. The same retained
  // history also feeds the whole-run experiment snapshot, so 120k frames
  // covers a bounded run at 60 Hz without unbounded page growth.
  if (timestamps.length > 120_000) {
    timestamps.splice(0, timestamps.length - 120_000);
  }
}


function percentile(sorted: number[], quantile: number): number {
  if (sorted.length === 0) {
    return 0;
  }
  const index = Math.min(
    sorted.length - 1,
    Math.floor(sorted.length * quantile)
  );
  return sorted[index] ?? 0;
}
