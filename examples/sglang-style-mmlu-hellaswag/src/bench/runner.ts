import { Wllama } from '@wllama/wllama';
import wllamaSingle from '@wllama/wllama/src/single-thread/wllama.wasm?url';
import wllamaMulti from '@wllama/wllama/src/multi-thread/wllama.wasm?url';
import type {
  BenchConfig,
  BenchLogEvent,
  BenchReport,
  BenchSummary,
  HellaSwagItem,
  MMLUItem,
  QAResult,
} from './types';

const WLLAMA_CONFIG_PATHS = {
  'single-thread/wllama.wasm': wllamaSingle,
  'multi-thread/wllama.wasm': wllamaMulti,
};

const CHOICE_LABELS = ['A', 'B', 'C', 'D'] as const;
const DECODE_CHUNK_SIZE = 128;
const DECODE_RETRY_CHUNK_SIZE = 32;
const DECODE_STEP_TIMEOUT_MS = 10000;
const EXIT_TIMEOUT_MS = 2000;

function assertNotAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw new DOMException('Aborted', 'AbortError');
  }
}

function avg(values: number[]): number {
  if (!values.length) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function safeText(text: string): string {
  return text.replace(/\s+/g, ' ').trim();
}

function isDecodeTimeoutError(err: unknown): boolean {
  const msg = err instanceof Error ? err.message : String(err);
  return msg.includes('decode chunk timeout');
}

function hardTerminateWllama(
  wllama: Wllama,
  onLog?: (e: BenchLogEvent) => void,
  label: string = 'wllama'
): boolean {
  try {
    const runtime = wllama as unknown as {
      proxy?: {
        worker?: Worker;
      };
    };
    const worker = runtime.proxy?.worker;
    if (!worker) {
      onLog?.({ text: `[${label}] hard terminate: worker not found` });
      return false;
    }

    worker.terminate();
    if (runtime.proxy) {
      runtime.proxy.worker = undefined;
    }
    (wllama as unknown as { proxy?: unknown }).proxy = null;
    onLog?.({ text: `[${label}] hard terminate done` });
    return true;
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    onLog?.({ text: `[${label}] hard terminate error: ${msg}` });
    return false;
  }
}

async function safeExitWllama(
  wllama: Wllama,
  onLog?: (e: BenchLogEvent) => void,
  label: string = 'wllama'
): Promise<boolean> {
  try {
    let timedOut = false;
    await Promise.race([
      wllama.exit(),
      new Promise<void>((resolve) => {
        setTimeout(() => {
          timedOut = true;
          resolve();
        }, EXIT_TIMEOUT_MS);
      }),
    ]);
    if (timedOut) {
      onLog?.({ text: `[${label}] exit timeout after ${EXIT_TIMEOUT_MS}ms` });
      onLog?.({ text: `[${label}] trying hard terminate...` });
      return hardTerminateWllama(wllama, onLog, label);
    }
    onLog?.({ text: `[${label}] exit done` });
    return true;
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    onLog?.({ text: `[${label}] exit error: ${msg}` });
    onLog?.({ text: `[${label}] trying hard terminate after exit error...` });
    return hardTerminateWllama(wllama, onLog, label);
  }
}

async function createLoadedWllama(
  config: BenchConfig,
  onLog?: (e: BenchLogEvent) => void
): Promise<Wllama> {
  const wllama = new Wllama(WLLAMA_CONFIG_PATHS, { preferWebGPU: true, noPerf: false });
  onLog?.({ text: '[Wllama] loading model...' });
  await wllama.loadModelFromUrl(config.modelUrl, {
    useCache: true,
    n_ctx: config.nCtx,
    n_batch: config.nBatch,
    // Keep full KV buffer semantics; avoids seq_cp assertion in tree save/restore.
    n_seq_max: 1,
    kv_unified: true,
  });
  onLog?.({ text: '[Wllama] model loaded.' });
  return wllama;
}

async function replaceLoadedWllama(
  oldWllama: Wllama | null,
  config: BenchConfig,
  onLog?: (e: BenchLogEvent) => void,
  reason: string = 'replace'
): Promise<Wllama> {
  if (oldWllama) {
    onLog?.({ text: `[Wllama] disposing old runtime (${reason})` });
    const exited = await safeExitWllama(oldWllama, onLog, `Wllama/${reason}`);
    if (!exited) {
      throw new Error(`[Wllama] old runtime did not exit cleanly (${reason}), refusing to create new model`);
    }
  }

  const fresh = await createLoadedWllama(config, onLog);
  return fresh;
}

function mmluQuestionBlock(item: MMLUItem, withAnswer = false): string {
  const lines = [
    `Question: ${safeText(item.question)}`,
    `A. ${safeText(item.choices[0])}`,
    `B. ${safeText(item.choices[1])}`,
    `C. ${safeText(item.choices[2])}`,
    `D. ${safeText(item.choices[3])}`,
    'Answer:',
  ];
  if (withAnswer) {
    lines.push(CHOICE_LABELS[item.answerIndex]);
  }
  return lines.join('\n');
}

function buildMmluSharedPrefix(shots: MMLUItem[]): string {
  const parts = [
    'You are taking a multiple-choice exam. Choose exactly one option: A, B, C, or D.',
    'Respond with only one letter.',
    '',
    ...shots.map((s) => mmluQuestionBlock(s, true)),
    '',
  ];
  return parts.join('\n');
}

function buildHellaSharedPrefix(shots: HellaSwagItem[]): string {
  const parts = [
    'Select the most likely continuation among options A/B/C/D.',
    'Choose based on commonsense plausibility.',
    '',
  ];

  for (const s of shots) {
    parts.push(`Context: ${safeText(s.ctx)}`);
    parts.push(`A. ${safeText(s.endings[0])}`);
    parts.push(`B. ${safeText(s.endings[1])}`);
    parts.push(`C. ${safeText(s.endings[2])}`);
    parts.push(`D. ${safeText(s.endings[3])}`);
    parts.push(`Answer: ${CHOICE_LABELS[s.label]}`);
    parts.push('');
  }

  return parts.join('\n');
}

async function getNextTokenDistribution(wllama: Wllama): Promise<Map<number, number>> {
  const logits = await wllama.getLogits(-1);
  const map = new Map<number, number>();
  for (const row of logits) {
    map.set(row.token, row.p);
  }
  return map;
}

async function scoreContinuation(
  wllama: Wllama,
  continuationTokens: number[],
  signal?: AbortSignal,
  onFirstToken?: () => void
): Promise<number> {
  let score = 0;
  let seenFirst = false;
  for (const tok of continuationTokens) {
    assertNotAborted(signal);
    const dist = await getNextTokenDistribution(wllama);
    if (!seenFirst) {
      seenFirst = true;
      onFirstToken?.();
    }
    const p = Math.max(dist.get(tok) ?? 1e-12, 1e-12);
    score += Math.log(p);
    await wllama.decode([tok], {});
  }
  return score;
}

async function decodeChunked(
  wllama: Wllama,
  tokens: number[],
  opts: {
    label: string;
    onLog?: (e: BenchLogEvent) => void;
    signal?: AbortSignal;
    chunkSize?: number;
    stepTimeoutMs?: number;
  }
): Promise<void> {
  const {
    label,
    onLog,
    signal,
    chunkSize = DECODE_CHUNK_SIZE,
    stepTimeoutMs = DECODE_STEP_TIMEOUT_MS,
  } = opts;
  const total = tokens.length;
  if (total === 0) {
    return;
  }

  const chunks = Math.ceil(total / chunkSize);

  for (let offset = 0; offset < total; offset += chunkSize) {
    assertNotAborted(signal);
    const end = Math.min(offset + chunkSize, total);
    const idx = Math.floor(offset / chunkSize) + 1;
    const chunkTokens = tokens.slice(offset, end);
    let timeoutId: ReturnType<typeof setTimeout> | null = null;
    try {
      await Promise.race([
        wllama.decode(chunkTokens, {}),
        new Promise<never>((_, reject) => {
          timeoutId = setTimeout(() => {
            reject(
              new Error(
                `[${label}] decode chunk timeout after ${stepTimeoutMs}ms at chunk ${idx}/${chunks} range=${offset}-${end - 1}`
              )
            );
          }, stepTimeoutMs);
        }),
      ]);
    } finally {
      if (timeoutId) clearTimeout(timeoutId);
    }
  }
}

async function runMmlu(
  wllama: Wllama,
  data: MMLUItem[],
  shots: number,
  evalCount: number,
  mode: 'flat' | 'tree',
  slotBase: number,
  onLog?: (e: BenchLogEvent) => void,
  signal?: AbortSignal,
  recoverWllama?: (oldRuntime: Wllama, reason: string) => Promise<Wllama>
): Promise<{ results: QAResult[]; latencyMs: number[]; ttftMs: number[]; tokensPerSecond: number[]; wllama: Wllama }> {
  let runtime = wllama;
  const rows: QAResult[] = [];
  const latencyMs: number[] = [];
  const ttftMs: number[] = [];
  const tokensPerSecond: number[] = [];

  const grouped = new Map<string, MMLUItem[]>();
  for (const item of data) {
    const key = item.subject || 'unknown';
    const bucket = grouped.get(key);
    if (bucket) {
      bucket.push(item);
    } else {
      grouped.set(key, [item]);
    }
  }

  const subjectEntries = [...grouped.entries()];
  for (let s = 0; s < subjectEntries.length; s += 1) {
    const [subject, items] = subjectEntries[s];
    const shotItems = items.slice(0, shots);
    const evalItems = items.slice(shots, shots + evalCount);
    if (!evalItems.length) {
      onLog?.({ text: `[MMLU/${mode}] subject=${subject} has no eval items, skipping.` });
      continue;
    }

    onLog?.({ text: `[MMLU/${mode}] subject=${subject} build sharedPrefix` });
    const sharedPrefix = buildMmluSharedPrefix(shotItems);
    onLog?.({ text: `[MMLU/${mode}] subject=${subject} tokenize sharedPrefix` });
    const sharedPrefixTokens = await runtime.tokenize(sharedPrefix, true);
    onLog?.({ text: `[MMLU/${mode}] subject=${subject} sharedPrefix tokens=${sharedPrefixTokens.length}` });
    let sharedSlotId = -1;
    let sharedNPast = 0;
    const prepareTreeSharedSlot = async (chunkSize: number = DECODE_CHUNK_SIZE): Promise<void> => {
      onLog?.({ text: `[MMLU/${mode}] subject=${subject} kvClear` });
      await runtime.kvClear();
      onLog?.({ text: `[MMLU/${mode}] subject=${subject} decode sharedPrefix` });
      await decodeChunked(runtime, sharedPrefixTokens, {
        label: `MMLU/${mode} subject=${subject} sharedPrefix`,
        onLog,
        signal,
        chunkSize,
        stepTimeoutMs: DECODE_STEP_TIMEOUT_MS,
      });
      sharedNPast = sharedPrefixTokens.length;
      sharedSlotId = Math.max(1, slotBase + s);
      onLog?.({ text: `[MMLU/${mode}] subject=${subject} kvSeqSave slot=${sharedSlotId}` });
      await runtime.kvSeqSave(sharedSlotId);
      onLog?.({
        text: `[MMLU/${mode}] subject=${subject} shared prefix cached in slot ${sharedSlotId} tokens=${sharedNPast}`,
      });
    };

    if (mode === 'tree') {
      try {
        await prepareTreeSharedSlot(DECODE_CHUNK_SIZE);
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        onLog?.({ text: `[MMLU/${mode}] subject=${subject} shared prefix build failed: ${msg}` });
        if (!isDecodeTimeoutError(err) || !recoverWllama) {
          throw err;
        }
        onLog?.({ text: `[MMLU/${mode}] subject=${subject} rebuilding runtime for shared prefix` });
        runtime = await recoverWllama(runtime, `MMLU/${mode} subject=${subject} shared-prefix-timeout`);
        try {
          await prepareTreeSharedSlot(DECODE_RETRY_CHUNK_SIZE);
        } catch (retryErr) {
          const retryMsg = retryErr instanceof Error ? retryErr.message : String(retryErr);
          onLog?.({ text: `[MMLU/${mode}] subject=${subject} shared prefix retry failed: ${retryMsg}` });
          onLog?.({ text: `[MMLU/${mode}] subject=${subject} skip whole subject` });
          runtime = await recoverWllama(runtime, `MMLU/${mode} subject=${subject} shared-prefix-retry-failed`);
          continue;
        }
      }
    }

    let skipRemainingSubject = false;
    for (let i = 0; i < evalItems.length; i += 1) {
      assertNotAborted(signal);
      const q = evalItems[i];
      const qPrompt = mmluQuestionBlock(q, false);
      const qTokens = await runtime.tokenize(qPrompt, true);

      const t0 = performance.now();
      if (mode === 'flat') {
        onLog?.({ text: `[MMLU/${mode}] ${q.id} kvClear` });
        await runtime.kvClear();
        onLog?.({ text: `[MMLU/${mode}] ${q.id} decode sharedPrefix+question` });
        const fullTokens = [...sharedPrefixTokens, ...qTokens];
        try {
          await decodeChunked(runtime, fullTokens, {
            label: `MMLU/${mode} ${q.id} sharedPrefix+question`,
            onLog,
            signal,
            chunkSize: DECODE_CHUNK_SIZE,
            stepTimeoutMs: DECODE_STEP_TIMEOUT_MS,
          });
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          onLog?.({ text: `[MMLU/${mode}] ${q.id} decode failed: ${msg}` });
          if (!isDecodeTimeoutError(err) || !recoverWllama) {
            throw err;
          }

          onLog?.({ text: `[MMLU/${mode}] ${q.id} rebuilding runtime due to decode timeout` });
          runtime = await recoverWllama(runtime, `MMLU/${mode} ${q.id} timeout`);

          try {
            onLog?.({ text: `[MMLU/${mode}] ${q.id} retry decode with smaller chunk=${DECODE_RETRY_CHUNK_SIZE}` });
            await runtime.kvClear();
            await decodeChunked(runtime, fullTokens, {
              label: `MMLU/${mode} ${q.id} sharedPrefix+question retry`,
              onLog,
              signal,
              chunkSize: DECODE_RETRY_CHUNK_SIZE,
              stepTimeoutMs: DECODE_STEP_TIMEOUT_MS,
            });
          } catch (retryErr) {
            const retryMsg = retryErr instanceof Error ? retryErr.message : String(retryErr);
            onLog?.({ text: `[MMLU/${mode}] ${q.id} retry failed: ${retryMsg}` });
            onLog?.({ text: `[MMLU/${mode}] ${q.id} skip this item and continue` });
            runtime = await recoverWllama(runtime, `MMLU/${mode} ${q.id} retry-failed`);
            rows.push({
              id: q.id,
              gtIndex: q.answerIndex,
              predIndexFlat: -1,
              predIndexTree: -1,
              correctFlat: false,
              correctTree: false,
              explainFlat: 'skipped: decode timeout',
              explainTree: '',
            });
            continue;
          }
        }
      } else {
        try {
          onLog?.({ text: `[MMLU/${mode}] ${q.id} kvSeqRestore slot=${sharedSlotId} nPast=${sharedNPast}` });
          await runtime.kvSeqRestore(sharedSlotId, sharedNPast);
          onLog?.({ text: `[MMLU/${mode}] ${q.id} decode question` });
          await decodeChunked(runtime, qTokens, {
            label: `MMLU/${mode} ${q.id} question`,
            onLog,
            signal,
            chunkSize: DECODE_CHUNK_SIZE,
            stepTimeoutMs: DECODE_STEP_TIMEOUT_MS,
          });
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          onLog?.({ text: `[MMLU/${mode}] ${q.id} decode failed: ${msg}` });
          if (!isDecodeTimeoutError(err) || !recoverWllama) {
            throw err;
          }
          onLog?.({ text: `[MMLU/${mode}] ${q.id} rebuilding runtime due to tree decode timeout` });
          runtime = await recoverWllama(runtime, `MMLU/${mode} ${q.id} tree-timeout`);

          try {
            await prepareTreeSharedSlot(DECODE_RETRY_CHUNK_SIZE);
            onLog?.({ text: `[MMLU/${mode}] ${q.id} retry kvSeqRestore slot=${sharedSlotId} nPast=${sharedNPast}` });
            await runtime.kvSeqRestore(sharedSlotId, sharedNPast);
            onLog?.({ text: `[MMLU/${mode}] ${q.id} retry decode question chunk=${DECODE_RETRY_CHUNK_SIZE}` });
            await decodeChunked(runtime, qTokens, {
              label: `MMLU/${mode} ${q.id} question retry`,
              onLog,
              signal,
              chunkSize: DECODE_RETRY_CHUNK_SIZE,
              stepTimeoutMs: DECODE_STEP_TIMEOUT_MS,
            });
          } catch (retryErr) {
            const retryMsg = retryErr instanceof Error ? retryErr.message : String(retryErr);
            onLog?.({ text: `[MMLU/${mode}] ${q.id} retry failed: ${retryMsg}` });
            onLog?.({ text: `[MMLU/${mode}] ${q.id} skip this item and continue` });
            runtime = await recoverWllama(runtime, `MMLU/${mode} ${q.id} tree-retry-failed`);

            try {
              onLog?.({ text: `[MMLU/${mode}] ${q.id} rebuilding subject shared slot after retry failure` });
              await prepareTreeSharedSlot(DECODE_RETRY_CHUNK_SIZE);
            } catch (reprepErr) {
              const reprepMsg = reprepErr instanceof Error ? reprepErr.message : String(reprepErr);
              onLog?.({ text: `[MMLU/${mode}] ${q.id} reprepare shared slot failed: ${reprepMsg}` });
              onLog?.({ text: `[MMLU/${mode}] subject=${subject} skip remaining eval items` });
              skipRemainingSubject = true;
            }

            rows.push({
              id: q.id,
              gtIndex: q.answerIndex,
              predIndexFlat: -1,
              predIndexTree: -1,
              correctFlat: false,
              correctTree: false,
              explainFlat: '',
              explainTree: 'skipped: decode timeout',
            });
            if (skipRemainingSubject) {
              break;
            }
            continue;
          }
        }
      }

      const dist = await getNextTokenDistribution(runtime);
      const firstTokenAt = performance.now();
      const optionTokenIds: number[] = [];
      for (const label of CHOICE_LABELS) {
        const toks = await runtime.tokenize(` ${label}`, true);
        optionTokenIds.push(toks[0]);
      }

      let bestIdx = 0;
      let bestP = -1;
      const probs: number[] = [];
      for (let j = 0; j < optionTokenIds.length; j += 1) {
        const p = dist.get(optionTokenIds[j]) ?? 0;
        probs.push(p);
        if (p > bestP) {
          bestP = p;
          bestIdx = j;
        }
      }

      const elapsed = performance.now() - t0;
      const ttft = Math.max(firstTokenAt - t0, 0);
      const scoredTokens = 1;
      const tps = (scoredTokens * 1000) / Math.max(elapsed, 1e-6);
      latencyMs.push(elapsed);
      ttftMs.push(ttft);
      tokensPerSecond.push(tps);

      onLog?.({
        text: `[MMLU/${mode}] ${q.id} pred=${CHOICE_LABELS[bestIdx]} gt=${CHOICE_LABELS[q.answerIndex]} probs=[${probs
          .map((p) => p.toFixed(4))
          .join(', ')}]`,
      });

      rows.push({
        id: q.id,
        gtIndex: q.answerIndex,
        predIndexFlat: mode === 'flat' ? bestIdx : -1,
        predIndexTree: mode === 'tree' ? bestIdx : -1,
        correctFlat: mode === 'flat' ? bestIdx === q.answerIndex : false,
        correctTree: mode === 'tree' ? bestIdx === q.answerIndex : false,
        explainFlat: mode === 'flat' ? `pred=${CHOICE_LABELS[bestIdx]}` : '',
        explainTree: mode === 'tree' ? `pred=${CHOICE_LABELS[bestIdx]}` : '',
      });
    }
    if (skipRemainingSubject) {
      continue;
    }
  }

  return { results: rows, latencyMs, ttftMs, tokensPerSecond, wllama: runtime };
}

async function runHella(
  wllama: Wllama,
  data: HellaSwagItem[],
  shots: number,
  evalCount: number,
  mode: 'flat' | 'tree',
  slotBase: number,
  onLog?: (e: BenchLogEvent) => void,
  signal?: AbortSignal,
  recoverWllama?: (oldRuntime: Wllama, reason: string) => Promise<Wllama>
): Promise<{ results: QAResult[]; latencyMs: number[]; ttftMs: number[]; tokensPerSecond: number[]; wllama: Wllama }> {
  let runtime = wllama;
  void slotBase;
  const shotItems = data.slice(0, shots);
  const evalItems = data.slice(shots, shots + evalCount);
  onLog?.({ text: `[Hella/${mode}] build sharedPrefix` });
  const sharedPrefix = buildHellaSharedPrefix(shotItems);
  onLog?.({ text: `[Hella/${mode}] tokenize sharedPrefix` });
  const sharedPrefixTokens = await runtime.tokenize(sharedPrefix, true);
  onLog?.({ text: `[Hella/${mode}] sharedPrefix tokens=${sharedPrefixTokens.length}` });

  let sharedSlotId = -1;
  let sharedNPast = 0;

  const prepareTreeSharedSlot = async (chunkSize: number = DECODE_CHUNK_SIZE): Promise<void> => {
    onLog?.({ text: `[Hella/${mode}] kvClear` });
    await runtime.kvClear();
    onLog?.({ text: `[Hella/${mode}] decode sharedPrefix` });
    await decodeChunked(runtime, sharedPrefixTokens, {
      label: `Hella/${mode} sharedPrefix`,
      onLog,
      signal,
      chunkSize,
      stepTimeoutMs: DECODE_STEP_TIMEOUT_MS,
    });
    sharedNPast = sharedPrefixTokens.length;
    sharedSlotId = Math.max(1, slotBase);
    onLog?.({ text: `[Hella/${mode}] kvSeqSave slot=${sharedSlotId}` });
    await runtime.kvSeqSave(sharedSlotId);
    onLog?.({ text: `[Hella/${mode}] shared prefix cached in slot ${sharedSlotId} tokens=${sharedNPast}` });
  };

  if (mode === 'tree') {
    try {
      await prepareTreeSharedSlot(DECODE_CHUNK_SIZE);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      onLog?.({ text: `[Hella/${mode}] shared prefix build failed: ${msg}` });
      if (!isDecodeTimeoutError(err) || !recoverWllama) {
        throw err;
      }
      runtime = await recoverWllama(runtime, `Hella/${mode} shared-prefix-timeout`);
      await prepareTreeSharedSlot(DECODE_RETRY_CHUNK_SIZE);
    }
  }

  const rows: QAResult[] = [];
  const latencyMs: number[] = [];
  const ttftMs: number[] = [];
  const tokensPerSecond: number[] = [];

  for (let i = 0; i < evalItems.length; i += 1) {
    assertNotAborted(signal);
    const q = evalItems[i];
    const qStem = [
      `Context: ${safeText(q.ctx)}`,
      'Choose the most plausible ending.',
      'Answer:',
    ].join('\n');
    const qStemTokens = await runtime.tokenize(qStem, true);

    const t0 = performance.now();
    let firstTokenAt = 0;
    let scoredTokens = 0;
    if (mode === 'flat') {
      onLog?.({ text: `[Hella/${mode}] ${q.id} kvClear` });
      await runtime.kvClear();
      onLog?.({ text: `[Hella/${mode}] ${q.id} decode sharedPrefix+qStem` });
      const fullStemTokens = [...sharedPrefixTokens, ...qStemTokens];
      try {
        await decodeChunked(runtime, fullStemTokens, {
          label: `Hella/${mode} ${q.id} sharedPrefix+qStem`,
          onLog,
          signal,
          chunkSize: DECODE_CHUNK_SIZE,
          stepTimeoutMs: DECODE_STEP_TIMEOUT_MS,
        });
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        onLog?.({ text: `[Hella/${mode}] ${q.id} decode failed: ${msg}` });
        if (!isDecodeTimeoutError(err) || !recoverWllama) {
          throw err;
        }
        onLog?.({ text: `[Hella/${mode}] ${q.id} rebuilding runtime due to flat decode timeout` });
        runtime = await recoverWllama(runtime, `Hella/${mode} ${q.id} flat-timeout`);
        try {
          onLog?.({ text: `[Hella/${mode}] ${q.id} retry decode with smaller chunk=${DECODE_RETRY_CHUNK_SIZE}` });
          await runtime.kvClear();
          await decodeChunked(runtime, fullStemTokens, {
            label: `Hella/${mode} ${q.id} sharedPrefix+qStem retry`,
            onLog,
            signal,
            chunkSize: DECODE_RETRY_CHUNK_SIZE,
            stepTimeoutMs: DECODE_STEP_TIMEOUT_MS,
          });
        } catch (retryErr) {
          const retryMsg = retryErr instanceof Error ? retryErr.message : String(retryErr);
          onLog?.({ text: `[Hella/${mode}] ${q.id} retry failed: ${retryMsg}` });
          onLog?.({ text: `[Hella/${mode}] ${q.id} skip this item and continue` });
          runtime = await recoverWllama(runtime, `Hella/${mode} ${q.id} flat-retry-failed`);
          rows.push({
            id: q.id,
            gtIndex: q.label,
            predIndexFlat: -1,
            predIndexTree: -1,
            correctFlat: false,
            correctTree: false,
            explainFlat: 'skipped: decode timeout',
            explainTree: '',
          });
          continue;
        }
      }
    }

    const scores: number[] = [];
    let skipItem = false;
    for (let j = 0; j < q.endings.length; j += 1) {
      assertNotAborted(signal);
      if (mode === 'tree') {
        try {
          onLog?.({ text: `[Hella/${mode}] ${q.id} ending=${j} kvSeqRestore slot=${sharedSlotId} nPast=${sharedNPast}` });
          // Only restore the shared slot to stay compatible with full-KV seq_cp constraints.
          await runtime.kvSeqRestore(sharedSlotId, sharedNPast);
          onLog?.({ text: `[Hella/${mode}] ${q.id} ending=${j} decode qStem` });
          await decodeChunked(runtime, qStemTokens, {
            label: `Hella/${mode} ${q.id} ending=${j} qStem`,
            onLog,
            signal,
            chunkSize: DECODE_CHUNK_SIZE,
            stepTimeoutMs: DECODE_STEP_TIMEOUT_MS,
          });
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          onLog?.({ text: `[Hella/${mode}] ${q.id} ending=${j} decode failed: ${msg}` });
          if (!isDecodeTimeoutError(err) || !recoverWllama) {
            throw err;
          }
          runtime = await recoverWllama(runtime, `Hella/${mode} ${q.id} ending=${j} tree-timeout`);
          try {
            await prepareTreeSharedSlot(DECODE_RETRY_CHUNK_SIZE);
            onLog?.({ text: `[Hella/${mode}] ${q.id} ending=${j} retry kvSeqRestore slot=${sharedSlotId} nPast=${sharedNPast}` });
            await runtime.kvSeqRestore(sharedSlotId, sharedNPast);
            await decodeChunked(runtime, qStemTokens, {
              label: `Hella/${mode} ${q.id} ending=${j} qStem retry`,
              onLog,
              signal,
              chunkSize: DECODE_RETRY_CHUNK_SIZE,
              stepTimeoutMs: DECODE_STEP_TIMEOUT_MS,
            });
          } catch (retryErr) {
            const retryMsg = retryErr instanceof Error ? retryErr.message : String(retryErr);
            onLog?.({ text: `[Hella/${mode}] ${q.id} ending=${j} retry failed: ${retryMsg}` });
            onLog?.({ text: `[Hella/${mode}] ${q.id} skip this item and continue` });
            runtime = await recoverWllama(runtime, `Hella/${mode} ${q.id} ending=${j} tree-retry-failed`);
            rows.push({
              id: q.id,
              gtIndex: q.label,
              predIndexFlat: -1,
              predIndexTree: -1,
              correctFlat: false,
              correctTree: false,
              explainFlat: '',
              explainTree: 'skipped: decode timeout',
            });
            skipItem = true;
            break;
          }
        }
      } else {
        try {
          onLog?.({ text: `[Hella/${mode}] ${q.id} ending=${j} kvClear` });
          await runtime.kvClear();
          onLog?.({ text: `[Hella/${mode}] ${q.id} ending=${j} decode sharedPrefix+qStem` });
          await decodeChunked(runtime, [...sharedPrefixTokens, ...qStemTokens], {
            label: `Hella/${mode} ${q.id} ending=${j} sharedPrefix+qStem`,
            onLog,
            signal,
            chunkSize: DECODE_CHUNK_SIZE,
            stepTimeoutMs: DECODE_STEP_TIMEOUT_MS,
          });
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          onLog?.({ text: `[Hella/${mode}] ${q.id} ending=${j} decode failed: ${msg}` });
          if (!isDecodeTimeoutError(err) || !recoverWllama) {
            throw err;
          }
          runtime = await recoverWllama(runtime, `Hella/${mode} ${q.id} ending=${j} flat-timeout`);
          try {
            await runtime.kvClear();
            await decodeChunked(runtime, [...sharedPrefixTokens, ...qStemTokens], {
              label: `Hella/${mode} ${q.id} ending=${j} sharedPrefix+qStem retry`,
              onLog,
              signal,
              chunkSize: DECODE_RETRY_CHUNK_SIZE,
              stepTimeoutMs: DECODE_STEP_TIMEOUT_MS,
            });
          } catch (retryErr) {
            const retryMsg = retryErr instanceof Error ? retryErr.message : String(retryErr);
            onLog?.({ text: `[Hella/${mode}] ${q.id} ending=${j} retry failed: ${retryMsg}` });
            onLog?.({ text: `[Hella/${mode}] ${q.id} skip this item and continue` });
            runtime = await recoverWllama(runtime, `Hella/${mode} ${q.id} ending=${j} flat-retry-failed`);
            rows.push({
              id: q.id,
              gtIndex: q.label,
              predIndexFlat: -1,
              predIndexTree: -1,
              correctFlat: false,
              correctTree: false,
              explainFlat: 'skipped: decode timeout',
              explainTree: '',
            });
            skipItem = true;
            break;
          }
        }
      }
      const endingTokens = await runtime.tokenize(` ${safeText(q.endings[j])}`, true);
      scoredTokens += endingTokens.length;
      const raw = await scoreContinuation(
        runtime,
        endingTokens,
        signal,
        () => {
          if (!firstTokenAt) firstTokenAt = performance.now();
        }
      );
      const norm = raw / Math.max(endingTokens.length, 1);
      scores.push(norm);
    }

    if (skipItem) {
      continue;
    }

    let bestIdx = 0;
    for (let j = 1; j < scores.length; j += 1) {
      if (scores[j] > scores[bestIdx]) bestIdx = j;
    }

    const elapsed = performance.now() - t0;
    const ttft = Math.max((firstTokenAt || t0) - t0, 0);
    const tps = (scoredTokens * 1000) / Math.max(elapsed, 1e-6);
    latencyMs.push(elapsed);
    ttftMs.push(ttft);
    tokensPerSecond.push(tps);

    onLog?.({
      text: `[Hella/${mode}] ${q.id} pred=${CHOICE_LABELS[bestIdx]} gt=${CHOICE_LABELS[q.label]} scores=[${scores
        .map((s) => s.toFixed(3))
        .join(', ')}]`,
    });

    rows.push({
      id: q.id,
      gtIndex: q.label,
      predIndexFlat: mode === 'flat' ? bestIdx : -1,
      predIndexTree: mode === 'tree' ? bestIdx : -1,
      correctFlat: mode === 'flat' ? bestIdx === q.label : false,
      correctTree: mode === 'tree' ? bestIdx === q.label : false,
      explainFlat: mode === 'flat' ? `score=${scores[bestIdx].toFixed(3)}` : '',
      explainTree: mode === 'tree' ? `score=${scores[bestIdx].toFixed(3)}` : '',
    });
  }

  return { results: rows, latencyMs, ttftMs, tokensPerSecond, wllama: runtime };
}

function mergeRows(flatRows: QAResult[], treeRows: QAResult[]): QAResult[] {
  const treeMap = new Map(treeRows.map((r) => [r.id, r]));
  return flatRows.map((f) => {
    const t = treeMap.get(f.id);
    if (!t) return f;
    return {
      id: f.id,
      gtIndex: f.gtIndex,
      predIndexFlat: f.predIndexFlat,
      predIndexTree: t.predIndexTree,
      correctFlat: f.correctFlat,
      correctTree: t.correctTree,
      explainFlat: f.explainFlat,
      explainTree: t.explainTree,
    };
  });
}

function summarize(
  benchmark: 'MMLU' | 'HellaSwag',
  shots: number,
  evalCount: number,
  rows: QAResult[],
  latencyFlat: number[],
  latencyTree: number[],
  ttftFlat: number[],
  ttftTree: number[],
  tpsFlat: number[],
  tpsTree: number[]
): BenchSummary {
  const accFlat = rows.filter((r) => r.correctFlat).length / Math.max(rows.length, 1);
  const accTree = rows.filter((r) => r.correctTree).length / Math.max(rows.length, 1);
  const avgFlat = avg(latencyFlat);
  const avgTree = avg(latencyTree);
  const avgTtftFlat = avg(ttftFlat);
  const avgTtftTree = avg(ttftTree);
  const avgTpsFlat = avg(tpsFlat);
  const avgTpsTree = avg(tpsTree);
  const speedupPct = avgFlat > 0 ? ((avgFlat - avgTree) / avgFlat) * 100 : 0;
  const ttftSpeedupPct = avgTtftFlat > 0 ? ((avgTtftFlat - avgTtftTree) / avgTtftFlat) * 100 : 0;
  const tpsGainPct = avgTpsFlat > 0 ? ((avgTpsTree - avgTpsFlat) / avgTpsFlat) * 100 : 0;
  return {
    benchmark,
    shots,
    evalCount,
    accFlat,
    accTree,
    avgTtftMsFlat: avgTtftFlat,
    avgTtftMsTree: avgTtftTree,
    ttftSpeedupPct,
    avgTokensPerSecondFlat: avgTpsFlat,
    avgTokensPerSecondTree: avgTpsTree,
    tpsGainPct,
    avgLatencyMsFlat: avgFlat,
    avgLatencyMsTree: avgTree,
    speedupPct,
    results: rows,
  };
}

function emptySummary(benchmark: 'MMLU' | 'HellaSwag', shots: number, evalCount: number): BenchSummary {
  return {
    benchmark,
    shots,
    evalCount,
    accFlat: 0,
    accTree: 0,
    avgTtftMsFlat: 0,
    avgTtftMsTree: 0,
    ttftSpeedupPct: 0,
    avgTokensPerSecondFlat: 0,
    avgTokensPerSecondTree: 0,
    tpsGainPct: 0,
    avgLatencyMsFlat: 0,
    avgLatencyMsTree: 0,
    speedupPct: 0,
    results: [],
  };
}

export async function runSglangStyleBench(
  config: BenchConfig,
  mmluData: MMLUItem[],
  hellaData: HellaSwagItem[],
  onLog?: (e: BenchLogEvent) => void,
  signal?: AbortSignal
): Promise<BenchReport> {
  let wllama: Wllama | null = await createLoadedWllama(config, onLog);

  try {
    assertNotAborted(signal);

    let mmluSummary: BenchSummary;
    if (config.mmluEvalCount > 0) {
      onLog?.({ text: 'Running MMLU (flat)...' });
      const mmluFlat = await runMmlu(
        wllama,
        mmluData,
        config.mmluShots,
        config.mmluEvalCount,
        'flat',
        1,
        onLog,
        signal,
        async (oldRuntime, reason) => {
          wllama = await replaceLoadedWllama(oldRuntime, config, onLog, reason);
          return wllama;
        }
      );
      wllama = mmluFlat.wllama;

      onLog?.({ text: 'Running MMLU (tree)...' });
      const mmluTree = await runMmlu(
        wllama,
        mmluData,
        config.mmluShots,
        config.mmluEvalCount,
        'tree',
        1,
        onLog,
        signal,
        async (oldRuntime, reason) => {
          wllama = await replaceLoadedWllama(oldRuntime, config, onLog, reason);
          return wllama;
        }
      );
      wllama = mmluTree.wllama;

      const mmluRows = mergeRows(mmluFlat.results, mmluTree.results);
      mmluSummary = summarize(
        'MMLU',
        config.mmluShots,
        mmluRows.length,
        mmluRows,
        mmluFlat.latencyMs,
        mmluTree.latencyMs,
        mmluFlat.ttftMs,
        mmluTree.ttftMs,
        mmluFlat.tokensPerSecond,
        mmluTree.tokensPerSecond
      );
    } else {
      onLog?.({ text: 'Skipping MMLU (mmluEvalCount=0).' });
      mmluSummary = emptySummary('MMLU', config.mmluShots, config.mmluEvalCount);
    }

    let hellaSummary: BenchSummary;
    if (config.hellaEvalCount > 0) {
      onLog?.({ text: 'Running HellaSwag (flat)...' });
      const hellaFlat = await runHella(
        wllama,
        hellaData,
        config.hellaShots,
        config.hellaEvalCount,
        'flat',
        1,
        onLog,
        signal,
        async (oldRuntime, reason) => {
          wllama = await replaceLoadedWllama(oldRuntime, config, onLog, reason);
          return wllama;
        }
      );
      wllama = hellaFlat.wllama;

      onLog?.({ text: 'Running HellaSwag (tree)...' });
      const hellaTree = await runHella(
        wllama,
        hellaData,
        config.hellaShots,
        config.hellaEvalCount,
        'tree',
        1,
        onLog,
        signal,
        async (oldRuntime, reason) => {
          wllama = await replaceLoadedWllama(oldRuntime, config, onLog, reason);
          return wllama;
        }
      );
      wllama = hellaTree.wllama;

      const hellaRows = mergeRows(hellaFlat.results, hellaTree.results);
      hellaSummary = summarize(
        'HellaSwag',
        config.hellaShots,
        config.hellaEvalCount,
        hellaRows,
        hellaFlat.latencyMs,
        hellaTree.latencyMs,
        hellaFlat.ttftMs,
        hellaTree.ttftMs,
        hellaFlat.tokensPerSecond,
        hellaTree.tokensPerSecond
      );
    } else {
      onLog?.({ text: 'Skipping HellaSwag (hellaEvalCount=0).' });
      hellaSummary = emptySummary('HellaSwag', config.hellaShots, config.hellaEvalCount);
    }

    return {
      modelUrl: config.modelUrl,
      config,
      mmlu: mmluSummary,
      hella: hellaSummary,
    };
  } finally {
    if (wllama) {
      await safeExitWllama(wllama, onLog, 'runSglangStyleBench/finally');
      wllama = null;
    }
  }
}
