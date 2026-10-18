import { describe, expect, it, vi } from 'vitest';
import { BoundedExecutionController } from '../../src/execution/bounded-execution-controller';
import { FakeNativeEngine } from '../../src/native/fake-native-engine';
import { request } from '../helpers/factories';
import type {
  NativeUnitBudget,
  NativeUnitProgress,
} from '../../src/contracts';

// A backend that reports exact native counters, so the controller's estimates
// can be driven deterministically instead of relying on wall-clock timing.
class CounterScriptedEngine extends FakeNativeEngine {
  private index = 0;

  constructor(
    private readonly samples: Array<{ decoded: number; predictedMs: number }>
  ) {
    super();
  }

  override async advance(
    requestId: string,
_budget: NativeUnitBudget
  ): Promise<NativeUnitProgress> {
    const sample = this.samples[Math.min(this.index, this.samples.length - 1)];
    this.index += 1;
    const base = await super.advance(requestId, {
      maxPrefillTokens: 4_096,
      maxDecodeTokens: sample?.decoded ?? 1,
      wallTimeMs: 60_000,
    });
    return {
      ...base,
      counters: {
        promptN: 0,
        cacheN: 0,
        predictedN: sample?.decoded ?? 0,
        predictedMs: sample?.predictedMs ?? 1,
      },
    };
  }
}

describe('BoundedExecutionController', () => {
  it('bounds background prefill and decode while foreground work is pending', async () => {
    const native = new FakeNativeEngine({
      prefillMsPerToken: 1,
      decodeMsPerToken: 1,
    });
    const yieldMacrotask = vi.fn(async () => undefined);
    const controller = new BoundedExecutionController(
      native,
      {
        maxPrefillTokens: 32,
        baseDecodeTokens: 16,
        foregroundDecodeTokens: 16,
        backgroundDecodeTokens: 2,
        prefillMsPerToken: 1,
        maxUnitWallTimeMs: 64,
      },
      yieldMacrotask
    );
    const background = request('bg:r1', {
      promptTokens: [1, 2, 3, 4, 5, 6, 7, 8],
      maxNewTokens: 8,
      lifecycle: {
        clientId: 'bg',
        epoch: 1,
        state: 'hidden',
        foreground: false,
        changedAt: 0,
      },
    });

    const first = await controller.runUnit(background, true);
    expect(first.budget.maxPrefillTokens).toBe(2);
    expect(first.progress.prefillProcessed).toBe(2);
    expect(first.requeued).toBe(true);

    let outcome = first;
    for (let step = 0; step < 8 && !outcome.completed; step += 1) {
      outcome = await controller.runUnit(background, true);
    }
    expect(outcome.completed).toBe(true);
    expect(background.decodeGenerated).toBe(8);
    expect(yieldMacrotask).toHaveBeenCalled();
  });

  it('gives a hidden tab a prefill chunk of its own without widening its decode slice', async () => {
    const native = new FakeNativeEngine({
      prefillMsPerToken: 1,
      decodeMsPerToken: 1,
    });
    const controller = new BoundedExecutionController(native, {
      maxPrefillTokens: 32,
      baseDecodeTokens: 16,
      foregroundDecodeTokens: 16,
      backgroundDecodeTokens: 2,
      backgroundPrefillTokens: 8,
      prefillMsPerToken: 1,
      maxUnitWallTimeMs: 64,
    });
    const background = request('bg:r2', {
      promptTokens: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
      maxNewTokens: 4,
      lifecycle: {
        clientId: 'bg',
        epoch: 1,
        state: 'hidden',
        foreground: false,
        changedAt: 0,
      },
    });

    const prefill = await controller.runUnit(background, true);
    expect(prefill.budget.maxPrefillTokens).toBe(8);
    expect(prefill.budget.maxDecodeTokens).toBe(2);
    expect(prefill.progress.prefillProcessed).toBe(8);
    expect(prefill.requeued).toBe(true);
  });

  it('does not insert a prose separator before a chat-template marker', async () => {
    const native = new FakeNativeEngine();
    const submit = vi.spyOn(native, 'submit');
    const controller = new BoundedExecutionController(native);
    const delta = '<|eot_id|><|start_header_id|>user<|end_header_id|>';
    const pending = request('replay:r1', {
      promptText: delta,
      promptTokens: [5, 6],
      replayText: 'assistant answer',
      replayTokens: [1, 2, 3, 4],
      maxNewTokens: 1,
    });

    await controller.runUnit(pending, false);

    expect(submit.mock.calls[0]?.[0].promptText).toBe(`assistant answer${delta}`);
  });

  it('returns a native error instead of reporting a successful completion', async () => {
    const native = new FakeNativeEngine();
    const controller = new BoundedExecutionController(native);
    const pending = request('c1:r1', { maxNewTokens: 1 });

    await controller.runUnit(pending, false);
    native.failNextAdvance(pending.requestId);
    await expect(controller.runUnit(pending, false)).rejects.toMatchObject({
      code: 'native_error',
    });
  });
});

describe('BoundedExecutionController adaptive unit sizing', () => {
  const adaptiveConfig = {
    maxPrefillTokens: 16,
    baseDecodeTokens: 32,
    foregroundDecodeTokens: 32,
    backgroundDecodeTokens: 32,
    prefillMsPerToken: 1,
    maxUnitWallTimeMs: 240,
    adaptiveUnits: true,
    targetUnitWallTimeMs: 30,
    minDecodeTokens: 1,
    maxDecodeTokens: 32,
  };

  it('shrinks the next unit when native counters report a slow decode', async () => {
    const native = new CounterScriptedEngine([
      { decoded: 4, predictedMs: 400 },
    ]);
    const controller = new BoundedExecutionController(native, adaptiveConfig);
    const foreground = request('fg:slow', { maxNewTokens: 32 });

    const first = await controller.runUnit(foreground, false);
    // without an estimate the lifecycle ceiling applies
    expect(first.budget.maxDecodeTokens).toBe(32);
    expect(foreground.estimatedDecodeTokensPerMs).toBeCloseTo(0.01, 6);

    const second = await controller.runUnit(foreground, false);
    // 0.01 tokens/ms against a 30 ms target rounds below one token
    expect(second.budget.maxDecodeTokens).toBe(1);
  });

  it('grows the unit back toward the ceiling when decode speeds up', async () => {
    const native = new CounterScriptedEngine([
      { decoded: 4, predictedMs: 400 },
      { decoded: 20, predictedMs: 10 },
    ]);
    const controller = new BoundedExecutionController(native, adaptiveConfig);
    const foreground = request('fg:fast', { maxNewTokens: 64 });

    await controller.runUnit(foreground, false);
    const shrunk = await controller.runUnit(foreground, false);
    expect(shrunk.budget.maxDecodeTokens).toBe(1);

    // the second unit blended 0.01 and 2 tokens/ms into 1.005, which the
    // third unit sizes against: 30 ms x 1.005 tokens/ms, clamped by target
    expect(foreground.estimatedDecodeTokensPerMs).toBeCloseTo(1.005, 6);
    const grown = await controller.runUnit(foreground, false);
    expect(grown.budget.maxDecodeTokens).toBe(30);
  });

  it('keeps the fixed lifecycle budget when adaptation is disabled', async () => {
    const native = new CounterScriptedEngine([
      { decoded: 4, predictedMs: 400 },
    ]);
    const controller = new BoundedExecutionController(native, {
      ...adaptiveConfig,
      adaptiveUnits: false,
    });
    const foreground = request('fg:fixed', { maxNewTokens: 32 });

    await controller.runUnit(foreground, false);
    const second = await controller.runUnit(foreground, false);
    expect(second.budget.maxDecodeTokens).toBe(32);
  });

  it('adds watchdog grace on top of the unit budget', async () => {
    const native = new FakeNativeEngine();
    const controller = new BoundedExecutionController(native, {
      ...adaptiveConfig,
      maxUnitWallTimeMs: 40,
      unitWallTimeGraceMs: 200,
    });
    const foreground = request('fg:watchdog', { maxNewTokens: 8 });

    const outcome = await controller.runUnit(foreground, false);
    // one slow native batch must not let the watchdog pre-empt the token budget
    expect(outcome.budget.wallTimeMs).toBe(240);
    // the native runtime gets the cooperative quantum, while the host guard
    // remains above it and only covers a genuinely stalled batch
    expect(outcome.budget.nativeTimeBudgetMs).toBe(30);
  });

  it('disables the native time quantum for the unsliced control', async () => {
    const native = new FakeNativeEngine();
    const controller = new BoundedExecutionController(native, {
      ...adaptiveConfig,
      maxUnitWallTimeMs: 30_000,
      nativeTimeBudgetEnabled: false,
    });
    const foreground = request('fg:unsliced', { maxNewTokens: 8 });

    const outcome = await controller.runUnit(foreground, false);
    expect(outcome.budget.nativeTimeBudgetMs).toBeUndefined();
    expect(outcome.budget.wallTimeMs).toBe(30_120);
  });
});
