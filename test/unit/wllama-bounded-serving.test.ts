import { describe, expect, it } from 'vitest';
import {
  WllamaNativeEngine,
  type WllamaRuntime,
} from '../../src/native/wllama-native-engine';
import type { WllamaCompletionChunk } from '../../src/native/wllama-native-engine';

interface FakeSlot {
  tokens: number[];
  kv: Uint8Array;
}

function tokenize(text: string): number[] {
  return text
    .trim()
    .split(/\s+/)
    .filter((word) => word.length > 0)
    .map((word) => (word.charCodeAt(0) + word.length * 31) % 100_000);
}

function commonPrefix(a: number[], b: number[]): number {
  let index = 0;
  while (index < a.length && index < b.length && a[index] === b[index]) {
    index += 1;
  }
  return index;
}

// Minimal stand-in for the patched wasm build: it keeps a per-slot KV state,
// reports how many prompt tokens came from that cache, and supports the
// export/import/drop API added by the bounded serving patch.
class FakeSlotRuntime implements WllamaRuntime {
  private readonly slots = new Map<number, FakeSlot>();
  private loaded = false;
  exportCalls = 0;
  importCalls = 0;
  dropCalls = 0;
  loadParams: Record<string, unknown> | null = null;
  cacheLoadCalls = 0;
  directLoadCalls = 0;
  readonly seenOptions: Record<string, unknown>[] = [];

  constructor(
    private readonly forcedBoundedStopReason?: NonNullable<
      NonNullable<WllamaCompletionChunk['timings']>['bounded_stop_reason']
    >,
    private readonly forceBoundedStop = false
  ) {}

  isModelLoaded(): boolean {
    return this.loaded;
  }

  async loadModelFromUrl(
    _source: string,
    params?: Record<string, unknown>
  ): Promise<void> {
    this.cacheLoadCalls += 1;
    this.loadParams = params === undefined ? null : { ...params };
    this.loaded = true;
  }

  async loadModelDirect(
    _source: string,
    params?: Record<string, unknown>
  ): Promise<void> {
    this.directLoadCalls += 1;
    this.loadParams = params === undefined ? null : { ...params };
    this.loaded = true;
  }

  async exit(): Promise<void> {
    this.loaded = false;
  }

  hasSlot(slotId: number): boolean {
    return this.slots.has(slotId);
  }

  async createCompletion(
    options: Record<string, unknown>
  ): Promise<AsyncIterable<WllamaCompletionChunk>> {
    const slotId = Number(options.id_slot ?? 0);
    const prompt = String(options.prompt ?? '');
    this.seenOptions.push(options);
    // max_tokens is the request-level cap; n_decode_max is the per-unit
    // token boundary the patched server enforces at a token boundary.
    const maxTokens = Number(options.max_tokens ?? 1);
    const unitBudget = Math.max(
      1,
      Math.min(maxTokens, Number(options.n_decode_max ?? maxTokens))
    );
    const boundedStop = unitBudget < maxTokens || this.forceBoundedStop;
    const promptTokens = tokenize(prompt);
    const slot = this.slots.get(slotId);
    const cached = slot ? commonPrefix(slot.tokens, promptTokens) : 0;

    const prefillMax = Number(options.n_prefill_max ?? 0);
    const uncached = promptTokens.length - cached;

    // prompt-chunk unit: llama.cpp evaluates only the configured prefix,
    // keeps it in the slot, and closes the task without decoding
    if (prefillMax > 0 && uncached > prefillMax) {
      const processed = cached + prefillMax;
      const prefixTokens = promptTokens.slice(0, processed);
      this.slots.set(slotId, {
        tokens: prefixTokens,
        kv: new Uint8Array([slotId & 0xff, processed & 0xff, 0x57, 0x4c]),
      });
      return (async function* generate(): AsyncGenerator<WllamaCompletionChunk> {
        yield {
          choices: [{ text: '', finish_reason: null }],
          prompt_progress: {
            total: promptTokens.length,
            cache: cached,
            processed,
            time_ms: 1,
          },
        };
        yield {
          choices: [{ text: '', finish_reason: 'length' }],
          timings: {
            predicted_n: 0,
            prompt_n: prefillMax,
            cache_n: cached,
            bounded_stop: true,
            bounded_stop_reason: 'prefill-budget',
          },
        };
      })();
    }

    const generated = Array.from(
      { length: unitBudget },
      (_, index) => `t${index}`
    );
    const pieces = generated.map((token) => ` ${token}`);
    const boundedStopReason =
      this.forcedBoundedStopReason ?? 'decode-budget';
    const fullTokens = [...promptTokens, ...tokenize(generated.join(' '))];
    this.slots.set(slotId, {
      tokens: fullTokens,
      kv: new Uint8Array([
        slotId & 0xff,
        fullTokens.length & 0xff,
        0x57,
        0x4c,
      ]),
    });

    return (async function* generate(): AsyncGenerator<WllamaCompletionChunk> {
      yield {
        choices: [{ text: '', finish_reason: null }],
        prompt_progress: {
          total: promptTokens.length,
          cache: cached,
          processed: promptTokens.length,
          time_ms: 1,
        },
      };
      for (let index = 0; index < pieces.length; index += 1) {
        const last = index === pieces.length - 1;
        yield {
          choices: [
            {
              text: pieces[index] ?? '',
              finish_reason: last ? 'length' : null,
            },
          ],
          timings: {
            predicted_n: 1,
            ...(boundedStop && last
              ? {
                  bounded_stop: true,
                  bounded_stop_reason:
                    boundedStopReason,
                }
              : {}),
          },
        };
      }
    })();
  }

  async exportSlotState(
    slotId: number,
    options: { drop?: boolean } = {}
  ): Promise<{ kv: Uint8Array; tokens: number[]; tokenCount: number }> {
    this.exportCalls += 1;
    const slot = this.slots.get(slotId);
    if (!slot) {
      throw new Error(`slot ${slotId} is empty`);
    }
    const snapshot = {
      kv: new Uint8Array(slot.kv),
      tokens: [...slot.tokens],
      tokenCount: slot.tokens.length,
    };
    if (options.drop ?? true) {
      this.slots.delete(slotId);
    }
    return snapshot;
  }

  async importSlotState(
    slotId: number,
    state: { kv: Uint8Array; tokens: number[] }
  ): Promise<number> {
    if (!this.loaded) {
      throw new Error('loadModel() is not yet called');
    }
    this.importCalls += 1;
    this.slots.set(slotId, {
      kv: new Uint8Array(state.kv),
      tokens: [...state.tokens],
    });
    return state.tokens.length;
  }

  async dropSlotState(slotId: number): Promise<number> {
    this.dropCalls += 1;
    this.slots.delete(slotId);
    return 0;
  }
}

describe('WllamaNativeEngine bounded serving', () => {
  it('uses only the direct loader when direct mode is selected', async () => {
    const runtime = new FakeSlotRuntime();
    const engine = new WllamaNativeEngine({
      wasmUrl: 'test.wasm',
      modelUrl: 'test-model.gguf',
      modelLoadMode: 'direct',
      runtime,
    });

    await engine.prewarm();

    expect(runtime.directLoadCalls).toBe(1);
    expect(runtime.cacheLoadCalls).toBe(0);
    expect(runtime.loadParams).not.toHaveProperty('useCache');
  });

  it('keeps the cache loader as the default path', async () => {
    const runtime = new FakeSlotRuntime();
    const engine = new WllamaNativeEngine({
      wasmUrl: 'test.wasm',
      modelUrl: 'test-model.gguf',
      runtime,
    });

    await engine.prewarm();

    expect(runtime.cacheLoadCalls).toBe(1);
    expect(runtime.directLoadCalls).toBe(0);
    expect(runtime.loadParams).toMatchObject({ useCache: true });
  });

  it('leaves cross-session batching to the coordinator', async () => {
    const runtime = new FakeSlotRuntime();
    const engine = new WllamaNativeEngine({
      wasmUrl: 'test.wasm',
      modelUrl: 'test-model.gguf',
      nParallel: 9,
      runtime,
    });

    await engine.prewarm();

    expect(runtime.loadParams).toMatchObject({
      n_parallel: 9,
      cont_batching: false,
      cache_idle_slots: false,
    });
  });

  it('loads the model before importing a checkpoint', async () => {
    const producerRuntime = new FakeSlotRuntime();
    const producer = new WllamaNativeEngine({
      wasmUrl: 'test.wasm',
      modelUrl: 'test-model.gguf',
      nParallel: 1,
      runtime: producerRuntime,
    });
    await producer.prewarm();
    const submitted = await producer.submit({
      requestId: 'source',
      promptText: 'alpha beta gamma',
      promptTokens: tokenize('alpha beta gamma'),
      maxNewTokens: 4,
    });
    await producer.advance('source', {
      maxPrefillTokens: 8,
      maxDecodeTokens: 2,
      wallTimeMs: 5_000,
    });
    const checkpoint = await producer.slotExport(submitted.slotId as number);

    const consumerRuntime = new FakeSlotRuntime();
    const consumer = new WllamaNativeEngine({
      wasmUrl: 'test.wasm',
      modelUrl: 'test-model.gguf',
      nParallel: 1,
      runtime: consumerRuntime,
    });
    expect(consumerRuntime.isModelLoaded()).toBe(false);
    const imported = await consumer.slotImport(checkpoint);
    expect(consumerRuntime.isModelLoaded()).toBe(true);
    expect(imported.tokens).toBe(checkpoint.tokens.length);
  });

  it('parks a slot into KV bytes and resumes it without re-prefilling', async () => {
    const runtime = new FakeSlotRuntime();
    const engine = new WllamaNativeEngine({
      wasmUrl: 'test.wasm',
      modelUrl: 'test-model.gguf',
      nParallel: 2,
      runtime,
    });
    await engine.prewarm();

    const promptText = 'alpha beta gamma';
    const submitted = await engine.submit({
      requestId: 'r1',
      promptText,
      promptTokens: tokenize(promptText),
      maxNewTokens: 4,
    });
    const slotId = submitted.slotId;
    expect(slotId).not.toBeNull();

    const first = await engine.advance('r1', {
      maxPrefillTokens: 8,
      maxDecodeTokens: 2,
      wallTimeMs: 5_000,
    });
    expect(first.decodeGenerated).toBe(2);
    expect(runtime.hasSlot(slotId as number)).toBe(true);

    const checkpoint = await engine.slotExport(slotId as number);
    expect(checkpoint.formatVersion).toBe(2);
    expect(checkpoint.kv?.byteLength ?? 0).toBeGreaterThan(0);
    expect(checkpoint.tokens.length).toBeGreaterThan(tokenize(promptText).length);
    // parking releases the live slot
    expect(runtime.hasSlot(slotId as number)).toBe(false);
    expect(
      (await engine.slotStats()).find((stats) => stats.slotId === slotId)
    ).toMatchObject({ inUse: false, tokens: 0, state: 'idle' });

    const stats = await engine.slotImport(checkpoint);
    expect(runtime.importCalls).toBe(1);
    expect(stats.tokens).toBe(checkpoint.tokens.length);

    await engine.submit({
      requestId: 'r2',
      promptText,
      promptTokens: tokenize(promptText),
      maxNewTokens: 4,
      resumeSlotId: stats.slotId,
    });
    const resumed = await engine.advance('r2', {
      maxPrefillTokens: 8,
      maxDecodeTokens: 2,
      wallTimeMs: 5_000,
    });

    // the resumed unit reuses the whole restored prefix instead of replaying it
    expect(resumed.prefillCached).toBe(checkpoint.tokens.length);
    expect(resumed.prefillProcessed).toBeGreaterThanOrEqual(
      checkpoint.tokens.length
    );
    expect(resumed.outputChunks.length).toBeGreaterThan(0);
  });

  it('starts a new turn from a restored KV prefix instead of resuming a finished budget', async () => {
    const runtime = new FakeSlotRuntime();
    const engine = new WllamaNativeEngine({
      wasmUrl: 'test.wasm',
      modelUrl: 'test-model.gguf',
      nParallel: 2,
      runtime,
    });
    await engine.prewarm();

    const first = await engine.submit({
      requestId: 'turn-1',
      promptText: 'alpha beta gamma',
      promptTokens: tokenize('alpha beta gamma'),
      maxNewTokens: 2,
    });
    await engine.advance('turn-1', {
      maxPrefillTokens: 8,
      maxDecodeTokens: 2,
      wallTimeMs: 5_000,
    });
    const checkpoint = await engine.slotExport(first.slotId as number);
    const restored = await engine.slotImport(checkpoint);

    await engine.submit({
      requestId: 'turn-2',
      promptText: 'delta',
      promptTokens: tokenize('delta'),
      maxNewTokens: 2,
      resumeSlotId: restored.slotId,
      newTurn: true,
    });
    const resumed = await engine.advance('turn-2', {
      maxPrefillTokens: 8,
      maxDecodeTokens: 2,
      wallTimeMs: 5_000,
    });

    expect(resumed.prefillCached).toBe(checkpoint.tokens.length);
    expect(resumed.decodeGenerated).toBe(2);
    expect(resumed.outputChunks).toHaveLength(2);
    expect(resumed.done).toBe(true);
  });

  it('drops the slot KV when a session is released', async () => {

    const runtime = new FakeSlotRuntime();
    const engine = new WllamaNativeEngine({
      wasmUrl: 'test.wasm',
      modelUrl: 'test-model.gguf',
      nParallel: 1,
      runtime,
    });
    await engine.prewarm();
    const submitted = await engine.submit({
      requestId: 'r1',
      promptText: 'one two',
      promptTokens: tokenize('one two'),
      maxNewTokens: 2,
    });
    await engine.advance('r1', {
      maxPrefillTokens: 4,
      maxDecodeTokens: 1,
      wallTimeMs: 5_000,
    });
    await engine.slotDrop(submitted.slotId as number);
    expect(runtime.dropCalls).toBe(1);
    expect(runtime.hasSlot(submitted.slotId as number)).toBe(false);
  });

  it('closes units on the native token and prompt-chunk budgets', async () => {
    const runtime = new FakeSlotRuntime();
    const engine = new WllamaNativeEngine({
      wasmUrl: 'test.wasm',
      modelUrl: 'test-model.gguf',
      nParallel: 2,
      runtime,
    });
    await engine.prewarm();

    const promptText = 'p1 p2 p3 p4 p5 p6';
    await engine.submit({
      requestId: 'r1',
      promptText,
      promptTokens: tokenize(promptText),
      maxNewTokens: 4,
    });

    // the prompt exceeds the per-unit budget, so this call only evaluates a
    // prompt chunk and yields the slot without decoding
    const prefillUnit = await engine.advance('r1', {
      maxPrefillTokens: 3,
      maxDecodeTokens: 4,
      nativeTimeBudgetMs: 17,
      wallTimeMs: 5_000,
    });
    expect(runtime.seenOptions[0]).toMatchObject({
      n_prefill_max: 3,
      n_decode_max: 4,
      t_unit_max_ms: 17,
      t_max_predict_ms: 17,
      max_tokens: 4,
    });
    expect(prefillUnit.unitStop).toBe('native-prefill-budget');
    expect(prefillUnit.boundedStops).toBe(1);
    expect(prefillUnit.decodeGenerated).toBe(0);
    expect(prefillUnit.done).toBe(false);

    // now the unit decodes, and the token budget closes it at two tokens
    const decodeUnit = await engine.advance('r1', {
      maxPrefillTokens: 64,
      maxDecodeTokens: 2,
      wallTimeMs: 5_000,
    });
    expect(runtime.seenOptions[1]).toMatchObject({
      n_decode_max: 2,
      max_tokens: 4,
    });
    expect(decodeUnit.unitStop).toBe('native-decode-budget');
    expect(decodeUnit.boundedStops).toBe(2);
    expect(decodeUnit.decodeGenerated).toBe(2);
    expect(decodeUnit.done).toBe(false);

    // the last unit reaches the request cap, so it is a normal completion
    const lastUnit = await engine.advance('r1', {
      maxPrefillTokens: 64,
      maxDecodeTokens: 4,
      wallTimeMs: 5_000,
    });
    expect(lastUnit.unitStop).toBe('complete');
    expect(lastUnit.done).toBe(true);
    expect(lastUnit.decodeGenerated).toBe(4);
  });

  it('uses the native stop reason instead of inferring it from counters', async () => {
    const runtime = new FakeSlotRuntime('unit-time-budget', true);
    const engine = new WllamaNativeEngine({
      wasmUrl: 'test.wasm',
      modelUrl: 'test-model.gguf',
      nParallel: 1,
      runtime,
    });
    await engine.prewarm();

    await engine.submit({
      requestId: 'time-stop',
      promptText: 'one two three',
      promptTokens: tokenize('one two three'),
      maxNewTokens: 8,
    });
    const unit = await engine.advance('time-stop', {
      maxPrefillTokens: 64,
      maxDecodeTokens: 4,
      nativeTimeBudgetMs: 1,
      wallTimeMs: 5_000,
    });

    expect(unit.unitStop).toBe('native-time-budget');
    expect(unit.done).toBe(false);
    expect(unit.decodeGenerated).toBe(4);
  });

  it('samples at the request temperature instead of the config default', async () => {
    const runtime = new FakeSlotRuntime();
    const engine = new WllamaNativeEngine({
      wasmUrl: 'test.wasm',
      modelUrl: 'test-model.gguf',
      nParallel: 2,
      // the deployment default only applies when a request is silent
      temperature: 0.8,
      runtime,
    });
    await engine.prewarm();

    await engine.submit({
      requestId: 'greedy',
      promptText: 'one two',
      promptTokens: tokenize('one two'),
      maxNewTokens: 2,
      temperature: 0,
    });
    await engine.advance('greedy', {
      maxPrefillTokens: 64,
      maxDecodeTokens: 2,
      wallTimeMs: 5_000,
    });
    expect(runtime.seenOptions[0]).toMatchObject({ temperature: 0 });

    await engine.submit({
      requestId: 'defaulted',
      promptText: 'three four',
      promptTokens: tokenize('three four'),
      maxNewTokens: 2,
    });
    await engine.advance('defaulted', {
      maxPrefillTokens: 64,
      maxDecodeTokens: 2,
      wallTimeMs: 5_000,
    });
    const defaulted = runtime.seenOptions.at(-1);
    expect(defaulted).toMatchObject({ temperature: 0.8 });
  });
});
