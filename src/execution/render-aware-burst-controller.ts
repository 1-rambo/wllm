import type {
  ExecutionConfig,
  GpuBurstLevel,
  GpuBurstPolicy,
  RenderPressureSignal,
} from '../contracts';

export const GPU_BURST_POLICIES: readonly GpuBurstPolicy[] = Object.freeze([
  {
    level: 'normal',
    commandSubmitBatchSize: 64,
    maxInflightBatches: 0,
    submissionPauseMs: 0,
  },
  {
    level: 'guarded',
    commandSubmitBatchSize: 32,
    maxInflightBatches: 4,
    submissionPauseMs: 0,
  },
  {
    level: 'constrained',
    commandSubmitBatchSize: 8,
    maxInflightBatches: 2,
    submissionPauseMs: 0,
  },
]);

// The same level also bounds how much token work one cooperative unit admits.
// Unit shape is the lever the moderated render regime actually responds to: a
// large unit keeps the GPU busy across several frame deadlines, while a small
// one returns to the coordinator at the next token boundary so it can decide
// again before the next frame is due.
// Units are derived from the deployment's own fast shape rather than from
// fixed constants, so clearing pressure restores exactly what was configured
// instead of leaving a shrunken unit behind. The absolute caps keep a fast
// unsliced base from shrinking to something still too large for a frame.
export function unitSizingForLevel(
  level: GpuBurstLevel,
  base: ExecutionConfig
): Partial<ExecutionConfig> {
  if (level === 'normal') {
    return {};
  }
  const guarded = level === 'guarded';
  const ratio = guarded ? 2 : 4;
  const tokenCap = guarded ? 16 : 8;
  const targetCap = guarded ? 12 : 6;
  const shrink = (value: number): number =>
    Math.max(1, Math.min(tokenCap, Math.ceil(value / ratio)));
  return {
    maxPrefillTokens: shrink(base.maxPrefillTokens),
    baseDecodeTokens: shrink(base.baseDecodeTokens),
    foregroundDecodeTokens: shrink(base.foregroundDecodeTokens),
    backgroundDecodeTokens: shrink(base.backgroundDecodeTokens),
    maxDecodeTokens: shrink(base.maxDecodeTokens),
    targetUnitWallTimeMs: Math.max(
      1,
      Math.min(targetCap, Math.ceil(base.targetUnitWallTimeMs / ratio))
    ),
  };
}

export interface RenderAwareBurstControllerOptions {
  pressureP95Ms?: number;
  clearP95Ms?: number;
  pressureOverBudgetRatio?: number;
  clearOverBudgetRatio?: number;
  // Number of recent samples the stress and clear tests are applied to. One
  // sample is raw; the default takes the median of the last three.
  smoothingSamples?: number;
  downgradeSamples?: number;
  recoverSamples?: number;
  staleSignalMs?: number;
  minFrameCount?: number;
  guardedPauseMinMs?: number;
  guardedPauseMaxMs?: number;
  constrainedPauseMinMs?: number;
  constrainedPauseMaxMs?: number;
  // Pacing is off by default: the evidence is that a worker-side pause cannot
  // preempt work already in the GPU process, so the level changes shape first.
  pauseOnPressure?: boolean;
}

const DEFAULTS: Required<RenderAwareBurstControllerOptions> = {
  // The trigger has to sit above the measurement's own noise floor, not just
  // above the 16.67 ms frame budget. On the anchor device the presented-frame
  // monitor reports 19-34 ms P95 with over-budget ratios up to 0.44 while the
  // page does nothing but hold its own animation, and that band moves with
  // window occlusion and run-to-run timing. Two earlier calibrations failed on
  // that band: a 33 ms trigger never fired under real contention that reached
  // 40 ms, and a 25 ms trigger fired on ambient noise, pinned the gate shut,
  // and held one request for 284 s. Contention measured in the same probe
  // reaches 50-86 ms P95, so the pressure bar sits in the gap between the two
  // populations and is applied to a median of three samples, which rejects the
  // single-sample spikes that both regimes produce.
  pressureP95Ms: 42,
  clearP95Ms: 30,
  pressureOverBudgetRatio: 0.5,
  clearOverBudgetRatio: 0.2,
  smoothingSamples: 3,
  downgradeSamples: 2,
  // Release is bounded by wall time, not by a sample count: the clear window
  // in the contention schedule is 6 s, and the earlier six-samples-per-level
  // release needed more than 10 s because the gate stayed shut while the
  // controller climbed back down. Two clear samples per level keeps a full
  // two-level recovery inside one clear window.
  recoverSamples: 2,
  staleSignalMs: 1_500,
  minFrameCount: 6,
  guardedPauseMinMs: 8,
  guardedPauseMaxMs: 32,
  constrainedPauseMinMs: 32,
  constrainedPauseMaxMs: 96,
  pauseOnPressure: false,
};

export class RenderAwareBurstController {
  readonly config: Required<RenderAwareBurstControllerOptions>;
  private levelIndex = 0;
  private pressureSamples = 0;
  private clearSamples = 0;
  private lastSignalAt = 0;
  private submissionPauseMs = 0;
  private readonly recent: Array<{ p95Ms: number; overBudgetRatio: number }> = [];

  constructor(options: RenderAwareBurstControllerOptions = {}) {
    this.config = { ...DEFAULTS, ...options };
  }

  get policy(): GpuBurstPolicy {
    const base = GPU_BURST_POLICIES[this.levelIndex] ?? GPU_BURST_POLICIES[0]!;
    return { ...base, submissionPauseMs: this.submissionPauseMs };
  }

  observe(signal: RenderPressureSignal, now: number): GpuBurstPolicy {
    if (
      !Number.isFinite(signal.p95Ms) ||
      !Number.isFinite(signal.overBudgetRatio) ||
      signal.frameCount < this.config.minFrameCount
    ) {
      return this.policy;
    }
    this.lastSignalAt = now;

    this.recent.push({
      p95Ms: signal.p95Ms,
      overBudgetRatio: signal.overBudgetRatio,
    });
    while (this.recent.length > this.config.smoothingSamples) {
      this.recent.shift();
    }
    const p95Values = this.recent.map((sample) => sample.p95Ms);
    const ratioValues = this.recent.map((sample) => sample.overBudgetRatio);
    // Fast attack, slow release. The stress test looks at the worst of the
    // last two samples, so a step into contention is not delayed by the
    // smoothing window, while a lone spike still cannot satisfy the
    // two-sample downgrade counter. The clear test uses the median so release
    // needs sustained quiet rather than a single good sample.
    const attackP95Ms = maxOf(p95Values.slice(-2));
    const attackRatio = maxOf(ratioValues.slice(-2));
    const smoothedP95Ms = medianOf(p95Values);
    const smoothedRatio = medianOf(ratioValues);

    const stressed =
      attackP95Ms >= this.config.pressureP95Ms ||
      attackRatio >= this.config.pressureOverBudgetRatio;
    const clear =
      smoothedP95Ms <= this.config.clearP95Ms &&
      smoothedRatio <= this.config.clearOverBudgetRatio;

    if (stressed) {
      this.clearSamples = 0;
      this.pressureSamples += 1;
      if (
        this.pressureSamples >= this.config.downgradeSamples &&
        this.levelIndex < GPU_BURST_POLICIES.length - 1
      ) {
        this.levelIndex += 1;
        this.pressureSamples = 0;
      }
    } else if (clear) {
      this.pressureSamples = 0;
      this.clearSamples += 1;
      if (
        this.clearSamples >= this.config.recoverSamples &&
        this.levelIndex > 0
      ) {
        this.levelIndex -= 1;
        this.clearSamples = 0;
      }
    } else {
      this.pressureSamples = 0;
      this.clearSamples = 0;
    }

    this.updateSubmissionPause(smoothedP95Ms, stressed, clear);
    return this.policy;
  }

  private updateSubmissionPause(
    p95Ms: number,
    stressed: boolean,
    clear: boolean
  ): void {
    if (!this.config.pauseOnPressure) {
      this.submissionPauseMs = 0;
      return;
    }
    if (stressed) {
      if (this.levelIndex === 1) {
        this.submissionPauseMs = clampPause(
          Math.round(p95Ms - 16.67),
          this.config.guardedPauseMinMs,
          this.config.guardedPauseMaxMs
        );
      } else if (this.levelIndex >= 2) {
        this.submissionPauseMs = clampPause(
          Math.round(p95Ms),
          this.config.constrainedPauseMinMs,
          this.config.constrainedPauseMaxMs
        );
      } else {
        this.submissionPauseMs = 0;
      }
      return;
    }
    if (clear) {
      this.submissionPauseMs = Math.max(0, this.submissionPauseMs - 8);
    }
  }

  release(now: number): GpuBurstPolicy {
    if (
      this.lastSignalAt > 0 &&
      now - this.lastSignalAt > this.config.staleSignalMs
    ) {
      this.levelIndex = 0;
      this.pressureSamples = 0;
      this.clearSamples = 0;
      this.lastSignalAt = 0;
      this.submissionPauseMs = 0;
      this.recent.length = 0;
    }
    return this.policy;
  }

  reset(): void {
    this.levelIndex = 0;
    this.pressureSamples = 0;
    this.clearSamples = 0;
    this.lastSignalAt = 0;
    this.submissionPauseMs = 0;
    this.recent.length = 0;
  }
}

function clampPause(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function medianOf(values: readonly number[]): number {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.floor(sorted.length / 2)] ?? 0;
}

function maxOf(values: readonly number[]): number {
  return values.length === 0 ? 0 : Math.max(...values);
}
