import { describe, expect, it } from 'vitest';
import {
  WllamaNativeEngine,
  type WllamaCompletionChunk,
  type WllamaRuntime,
} from '../../src/native/wllama-native-engine';

/**
 * A long-lived session carries its conversation across turns, so the sequence
 * resident in a slot grows every turn. llama.cpp splits `n_ctx` across
 * `n_parallel` slots, which is why a run configured with `nCtx=32768` and
 * `nParallel=4` died at turn seven with "available context size (8192
 * tokens)". These tests pin the engine's own accounting: a prompt that cannot
 * fit is refused with the numbers that were configured, and a retained prefix
 * that no longer fits keeps its newest turns instead of failing mid-request.
 */

class IdleRuntime implements WllamaRuntime {
  readonly seenOptions: Record<string, unknown>[] = [];
  private loaded = false;

  isModelLoaded(): boolean {
    return this.loaded;
  }

  async loadModelFromUrl(): Promise<void> {
    this.loaded = true;
  }

  async exit(): Promise<void> {
    this.loaded = false;
  }

  async createCompletion(
    options: Record<string, unknown>
  ): Promise<AsyncIterable<WllamaCompletionChunk>> {
    this.seenOptions.push(options);
    // A stream that ends immediately: this test is about what the engine
    // decides before llama.cpp sees a token.
    return (async function* generate(): AsyncGenerator<WllamaCompletionChunk> {
      return;
    })();
  }
}

const CONTEXT = { nCtx: 32_768, nParallel: 4 }; // 8192 tokens per slot

const UNIT_BUDGET = {
  maxPrefillTokens: 16,
  maxDecodeTokens: 8,
  wallTimeMs: 1_000,
};

function engineFor(runtime: WllamaRuntime): WllamaNativeEngine {
  return new WllamaNativeEngine({
    wasmUrl: 'about:blank#wasm',
    modelUrl: '/models/demo.gguf',
    ...CONTEXT,
    runtime,
  });
}

function words(count: number, prefix = 'word'): string {
  return Array.from({ length: count }, (_, index) => `${prefix}${index}`).join(
    ' '
  );
}

describe('WllamaNativeEngine context budget', () => {
  it('states the real per-slot budget and the generation reserve', () => {
    const engine = engineFor(new IdleRuntime());
    expect(engine.contextDiagnostics()).toMatchObject({
      nCtx: 32_768,
      nParallel: 4,
      perSlotTokens: 8_192,
      reserveTokens: 64,
      trims: 0,
      lastTrim: null,
    });
    expect(engine.contextDiagnostics().promptBudgetTokens).toBe(
      8_192 - 64
    );
  });

  it('defers a text prompt to the native tokenizer instead of rejecting an estimate', async () => {
    const runtime = new IdleRuntime();
    const engine = engineFor(runtime);

    await engine.submit({
      requestId: 'req-heuristic-overflow',
      promptTokens: [0],
      promptText: words(9_000),
      maxNewTokens: 64,
    });
    await engine.advance('req-heuristic-overflow', UNIT_BUDGET);

    // The heuristic can overcount punctuation-heavy application prompts;
    // the native tokenizer is the authority for text submissions.
    expect(runtime.seenOptions).toHaveLength(1);
    expect(await engine.slotStats()).toEqual([
      expect.objectContaining({ slotId: 0, inUse: true }),
      { slotId: 1, inUse: false, tokens: 0, state: 'idle' },
      { slotId: 2, inUse: false, tokens: 0, state: 'idle' },
      { slotId: 3, inUse: false, tokens: 0, state: 'idle' },
    ]);
  });

  it('refuses a token-only submission that cannot fit the slot', async () => {
    const runtime = new IdleRuntime();
    const engine = engineFor(runtime);

    await expect(
      engine.submit({
        requestId: 'req-token-overflow',
        promptTokens: Array.from({ length: 9_000 }, (_, index) => index),
        maxNewTokens: 64,
      })
    ).rejects.toMatchObject({
      code: 'context_overflow',
      message: expect.stringContaining('nCtx=32768, nParallel=4'),
    });
    expect(runtime.seenOptions).toHaveLength(0);
  });

  it('keeps the newest turns when a retained conversation outgrows the slot', async () => {
    const runtime = new IdleRuntime();
    const engine = engineFor(runtime);

    // turn one fills most of the slot
    const firstTurn = words(7_000, 'turn1');
    await engine.submit({
      requestId: 'req-turn-1',
      promptTokens: [0],
      promptText: firstTurn,
      maxNewTokens: 64,
    });
    await engine.advance('req-turn-1', UNIT_BUDGET);

    // turn two continues the same slot as a new user turn
    const secondTurn = words(2_000, 'turn2');
    await engine.submit({
      requestId: 'req-turn-2',
      promptTokens: [0],
      promptText: secondTurn,
      maxNewTokens: 64,
      resumeSlotId: 0,
      newTurn: true,
    });

    // the second turn's own unit, which is where the prefix is actually sent
    await engine.advance('req-turn-2', UNIT_BUDGET);

    const diagnostics = engine.contextDiagnostics();
    expect(diagnostics.trims).toBe(1);
    expect(diagnostics.lastTrim?.perSlotTokens).toBe(8_192);
    expect(diagnostics.lastTrim?.trimmedTokens).toBeGreaterThan(0);

    // the unit that carried the second turn is the second createCompletion
    const sentSecond = String(runtime.seenOptions[1]?.prompt ?? '');
    // the newest turn survives, the oldest content is dropped, and what is
    // sent still fits the slot
    expect(sentSecond).toContain('turn20');
    expect(sentSecond.length).toBeLessThan(
      firstTurn.length + secondTurn.length
    );
    const prefix = sentSecond.slice(0, sentSecond.indexOf('turn20'));
    expect(prefix).not.toContain('turn10');
  });

  it('does not touch a prefix that still fits', async () => {
    const runtime = new IdleRuntime();
    const engine = engineFor(runtime);
    await engine.submit({
      requestId: 'req-small-1',
      promptTokens: [0],
      promptText: words(100, 'small1'),
      maxNewTokens: 32,
    });
    await engine.advance('req-small-1', UNIT_BUDGET);
    await engine.submit({
      requestId: 'req-small-2',
      promptTokens: [0],
      promptText: words(100, 'small2'),
      maxNewTokens: 32,
      resumeSlotId: 0,
      newTurn: true,
    });
    await engine.advance('req-small-2', UNIT_BUDGET);

    // the prefix is intact: the second turn's unit carries both turns
    const sent = String(runtime.seenOptions[1]?.prompt ?? '');
    expect(sent).toContain('small10');
    expect(sent).toContain('small20');
    expect(engine.contextDiagnostics()).toMatchObject({
      trims: 0,
      lastTrim: null,
    });
  });

  it('attaches a chat-template turn delta without injecting whitespace', async () => {
    const runtime = new IdleRuntime();
    const engine = engineFor(runtime);
    const firstPrompt = 'alpha beta gamma';
    await engine.submit({
      requestId: 'chat-turn-1',
      promptTokens: [0],
      promptText: firstPrompt,
      maxNewTokens: 32,
    });
    await engine.advance('chat-turn-1', UNIT_BUDGET);

    const turnDelta =
      '<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nnext' +
      '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n';
    await engine.submit({
      requestId: 'chat-turn-2',
      promptTokens: [0],
      promptText: turnDelta,
      maxNewTokens: 32,
      resumeSlotId: 0,
      newTurn: true,
    });
    await engine.advance('chat-turn-2', UNIT_BUDGET);

    expect(runtime.seenOptions[1]?.prompt).toBe(firstPrompt + turnDelta);
  });
});
