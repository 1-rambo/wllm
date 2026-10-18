import { describe, expect, it } from 'vitest';
import {
  WllamaNativeEngine,
  type WllamaRuntime,
} from '../../src/native/wllama-native-engine';
import type { WllamaCompletionChunk } from '../../src/native/wllama-native-engine';

interface ChunkScript {
  text: string;
  // cumulative llama.cpp counters for the unit, as reported by timings_per_token
  predictedN: number;
  promptN?: number;
  cacheN?: number;
  finish?: 'stop' | 'length';
}

// A runtime whose text chunks do not map one-to-one onto tokens: the first
// chunk carries two tokens at once. Only the native counters can tell us how
// many tokens were actually decoded.
class ScriptedRuntime implements WllamaRuntime {
  readonly seenOptions: Record<string, unknown>[] = [];
  private loaded = false;

  // one script per createCompletion call: llama.cpp resets its slot counters
  // per task, so a resumed unit reports only the tokens it decoded itself
  constructor(private readonly scripts: ChunkScript[][]) {}

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
    const script = this.scripts[this.seenOptions.length - 1] ?? [];
    return (async function* generate(): AsyncGenerator<WllamaCompletionChunk> {
      for (const chunk of script) {
        yield {
          choices: [
            { text: chunk.text, finish_reason: chunk.finish ?? null },
          ],
          timings: {
            predicted_n: chunk.predictedN,
            ...(chunk.promptN === undefined ? {} : { prompt_n: chunk.promptN }),
            ...(chunk.cacheN === undefined ? {} : { cache_n: chunk.cacheN }),
          },
        };
      }
    })();
  }
}

// A backend that never reports timings, exercising the honest fallback path.
class QuietRuntime implements WllamaRuntime {
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

  async createCompletion(): Promise<AsyncIterable<WllamaCompletionChunk>> {
    return (async function* generate(): AsyncGenerator<WllamaCompletionChunk> {
      yield { choices: [{ text: 'one two three', finish_reason: 'length' }] };
    })();
  }
}

async function readyEngine(runtime: WllamaRuntime): Promise<WllamaNativeEngine> {
  const engine = new WllamaNativeEngine({
    wasmUrl: 'test.wasm',
    modelUrl: 'test-model.gguf',
    nParallel: 2,
    runtime,
  });
  await engine.prewarm();
  return engine;
}

describe('WllamaNativeEngine token accounting', () => {
  it('counts decoded tokens from native counters, not streamed chunks', async () => {
    const runtime = new ScriptedRuntime([
      [
      { text: ' two tokens', predictedN: 2, promptN: 3, cacheN: 1 },
      { text: ' and', predictedN: 3, promptN: 3, cacheN: 1 },
      { text: ' four', predictedN: 4, promptN: 3, cacheN: 1, finish: 'length' },
      ],
    ]);
    const engine = await readyEngine(runtime);

    await engine.submit({
      requestId: 'r1',
      promptText: 'one two three',
      promptTokens: [1, 2, 3],
      maxNewTokens: 8,
    });
    const progress = await engine.advance('r1', {
      maxPrefillTokens: 8,
      maxDecodeTokens: 4,
      wallTimeMs: 5_000,
    });

    // three text chunks, four decoded tokens
    expect(progress.outputChunks).toHaveLength(3);
    expect(progress.decodeGenerated).toBe(4);
    expect(progress.tokenAccounting).toBe('native-token');
    expect(progress.counters).toMatchObject({
      promptN: 3,
      cacheN: 1,
      predictedN: 4,
    });
    // prompt accounting still comes from the native slot state
    expect(progress.prefillCached).toBe(1);
    expect(progress.prefillProcessed).toBe(4);
  });

  it('asks the backend for per-token timings', async () => {
    const runtime = new ScriptedRuntime([
      [
      { text: ' x', predictedN: 1, finish: 'length' },
      ],
    ]);
    const engine = await readyEngine(runtime);
    await engine.submit({
      requestId: 'r1',
      promptText: 'prompt',
      promptTokens: [1],
      maxNewTokens: 2,
    });
    await engine.advance('r1', {
      maxPrefillTokens: 8,
      maxDecodeTokens: 1,
      wallTimeMs: 5_000,
    });

    expect(runtime.seenOptions[0]?.timings_per_token).toBe(true);
    expect(runtime.seenOptions[0]?.id_slot).toBe(0);
    expect(runtime.seenOptions[0]?.n_prefill_max).toBe(8);
  });

  it('accumulates native counters across bounded units', async () => {
    const runtime = new ScriptedRuntime([
      [
      { text: ' a b', predictedN: 2 },
      ],
      [
      { text: ' c', predictedN: 1 },
      { text: ' d e', predictedN: 3, finish: 'length' },
      ],
    ]);
    const engine = await readyEngine(runtime);
    await engine.submit({
      requestId: 'r1',
      promptText: 'prompt',
      promptTokens: [1],
      maxNewTokens: 8,
    });

    const first = await engine.advance('r1', {
      maxPrefillTokens: 8,
      maxDecodeTokens: 2,
      wallTimeMs: 5_000,
    });
    const second = await engine.advance('r1', {
      maxPrefillTokens: 8,
      maxDecodeTokens: 3,
      wallTimeMs: 5_000,
    });

    expect(first.decodeGenerated).toBe(2);
    expect(second.decodeGenerated).toBe(5);
    // the second unit reports only its own counters
    expect(second.counters?.predictedN).toBe(3);
  });

  it('falls back to chunk counting and says so when counters are absent', async () => {
    const engine = await readyEngine(new QuietRuntime());
    await engine.submit({
      requestId: 'r1',
      promptText: 'prompt',
      promptTokens: [1],
      maxNewTokens: 8,
    });
    const progress = await engine.advance('r1', {
      maxPrefillTokens: 8,
      maxDecodeTokens: 8,
      wallTimeMs: 5_000,
    });

    expect(progress.decodeGenerated).toBe(1);
    expect(progress.tokenAccounting).toBe('stream-chunk');
    expect(progress.counters).toBeUndefined();
  });
});
