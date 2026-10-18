import type {
  GpuBurstPolicy,
  NativeSlotCheckpoint,
  NativeSlotStats,
  NativeSubmitRequest,
  NativeUnitBudget,
  NativeUnitProgress,
} from '../contracts';
import { WllmError } from '../contracts';
import type { NativeEngine } from './native-engine';
import {
  checksumBytes,
  concatBytes,
  decodeInt32Array,
  encodeInt32Array,
} from '../utils/bytes';

interface FakeRequestState {
  requestId: string;
  slotId: number;
  promptText: string;
  promptTokens: number[];
  prefillProcessed: number;
  decodeGenerated: number;
  maxNewTokens: number;
  stage: 'prefill' | 'decode' | 'done';
  cancelled: boolean;
  failure: boolean;
  prefillCached: number;
  outputChunks: string[];
}

export interface FakeNativeEngineOptions {
  slotCount?: number;
  prefillMsPerToken?: number;
  decodeMsPerToken?: number;
}

export class FakeNativeEngine implements NativeEngine {
  private readonly slotCount: number;
  private readonly prefillMsPerToken: number;
  private readonly decodeMsPerToken: number;
  private readonly requests = new Map<string, FakeRequestState>();
  private readonly slots = new Map<number, string | null>();
  // tier two stand-in: parked checkpoints kept by the engine, not the coordinator
  private readonly parked = new Map<number, NativeSlotCheckpoint>();
  private nextParkHandle = 1;
  private nextSlot = 0;

  advanceCalls = 0;
  cancelCalls = 0;
  readonly gpuPolicies: GpuBurstPolicy[] = [];

  constructor(options: FakeNativeEngineOptions = {}) {
    this.slotCount = options.slotCount ?? 8;
    this.prefillMsPerToken = options.prefillMsPerToken ?? 1;
    this.decodeMsPerToken = options.decodeMsPerToken ?? 2;
    for (let slot = 0; slot < this.slotCount; slot += 1) {
      this.slots.set(slot, null);
    }
  }

  async prewarm(): Promise<void> {
    return;
  }

  async setGpuBurstPolicy(policy: GpuBurstPolicy): Promise<void> {
    this.gpuPolicies.push({ ...policy });
  }

  async submit(request: NativeSubmitRequest): Promise<NativeUnitProgress> {
    if (this.requests.has(request.requestId)) {
      throw new WllmError(
        `native request already exists: ${request.requestId}`,
        'duplicate_native_request'
      );
    }
    let slotId: number | null;
    let previous: FakeRequestState | undefined;
    if (request.resumeSlotId !== undefined && request.resumeSlotId !== null) {
      slotId = request.resumeSlotId;
      const previousRequestId = this.slots.get(slotId);
      if (previousRequestId) {
        previous = this.requests.get(previousRequestId);
        this.requests.delete(previousRequestId);
      }
      this.slots.set(slotId, request.requestId);
    } else {
      slotId = this.allocateSlot(request.requestId);
    }
    if (slotId === null) {
      throw new WllmError('no free native slot', 'capacity');
    }
    const state: FakeRequestState = {
      requestId: request.requestId,
      slotId,
      promptText:
        request.newTurn && previous && request.replaceContext !== true
          ? appendText(
              previous.promptText,
              request.promptText ?? request.promptTokens.join(' ')
            )
          : request.promptText ?? request.promptTokens.join(' '),
      promptTokens:
        request.newTurn && previous && request.replaceContext !== true
          ? [...previous.promptTokens, ...request.promptTokens]
          : [...request.promptTokens],
      prefillProcessed: previous?.prefillProcessed ?? 0,
      prefillCached: previous?.prefillProcessed ?? 0,
      decodeGenerated: request.newTurn ? 0 : previous?.decodeGenerated ?? 0,
      maxNewTokens: request.maxNewTokens,
      stage:
        request.promptTokens.length === 0 && !request.newTurn
          ? 'decode'
          : 'prefill',
      cancelled: false,
      failure: false,
      outputChunks: [],
    };
    this.requests.set(request.requestId, state);
    return this.progress(state, [], 0);
  }

  async advance(
    nativeRequestId: string,
    budget: NativeUnitBudget
  ): Promise<NativeUnitProgress> {
    this.advanceCalls += 1;
    const state = this.requireRequest(nativeRequestId);
    if (state.cancelled) {
      return this.progress(state, [], 0);
    }
    if (state.stage === 'done') {
      return this.progress(state, [], 0);
    }
    if (state.failure) {
      state.failure = false;
      return {
        ...this.progress(state, [], 0),
        isError: true,
      };
    }

    const start = performance.now();
    let elapsedMs = 0;
    const chunks: string[] = [];

    // An absent native quantum models the unsliced control: the host
    // watchdog still exists, but it is not a cooperative token-boundary
    // budget for llama.cpp.
    const nativeTimeBudgetMs =
      budget.nativeTimeBudgetMs ?? Number.POSITIVE_INFINITY;

    const prefillByTime = Math.floor(
      nativeTimeBudgetMs / this.prefillMsPerToken
    );
    const prefillBudget = Math.max(
      0,
      Math.min(budget.maxPrefillTokens, prefillByTime)
    );

    if (state.stage === 'prefill') {
      const remaining = state.promptTokens.length - state.prefillProcessed;
      const accepted = Math.max(
        0,
        Math.min(remaining, Math.floor(prefillBudget))
      );
      state.prefillProcessed += accepted;
      elapsedMs += accepted * this.prefillMsPerToken;
      if (state.prefillProcessed < state.promptTokens.length) {
        return this.progress(state, chunks, this.measure(start, elapsedMs));
      }
      state.stage = 'decode';
    }

    const decodeTimeRemaining = Math.max(
      0,
      nativeTimeBudgetMs - elapsedMs
    );
    const decodeByTime = Math.floor(
      decodeTimeRemaining / this.decodeMsPerToken
    );
    const decodeBudget = Math.max(
      0,
      Math.min(budget.maxDecodeTokens, decodeByTime)
    );
    const remainingDecode = state.maxNewTokens - state.decodeGenerated;
    const acceptedDecode = Math.max(
      0,
      Math.min(remainingDecode, Math.floor(decodeBudget))
    );

    for (let index = 0; index < acceptedDecode; index += 1) {
      state.decodeGenerated += 1;
      const generatedTokenNumber = state.decodeGenerated;
      const piece =
        generatedTokenNumber % 7 === 0
          ? `\n`
          : `tok${generatedTokenNumber < 10 ? '0' : ''}${generatedTokenNumber} `;
      state.outputChunks.push(piece);
      chunks.push(piece);
    }
    elapsedMs += acceptedDecode * this.decodeMsPerToken;

    if (state.decodeGenerated >= state.maxNewTokens) {
      state.stage = 'done';
    }

    return this.progress(state, chunks, this.measure(start, elapsedMs));
  }

  async status(nativeRequestId: string): Promise<NativeUnitProgress> {
    return this.progress(this.requireRequest(nativeRequestId), [], 0);
  }

  async cancel(nativeRequestId: string): Promise<void> {
    this.cancelCalls += 1;
    const state = this.requireRequest(nativeRequestId);
    state.cancelled = true;
    state.stage = 'done';
    this.releaseRequestSlot(state);
  }

  async interrupt(_nativeRequestId: string): Promise<void> {
    // The fake advance completes synchronously, so there is no in-flight
    // stream to cut short; tests that need preemption assert on the
    // scheduler trace instead.
  }

  async slotExport(slotId: number): Promise<NativeSlotCheckpoint> {
    const requestId = this.slots.get(slotId);
    if (!requestId) {
      throw new WllmError(`slot is empty: ${slotId}`, 'empty_slot');
    }
    const state = this.requireRequest(requestId);
    const tokens = [
      ...state.promptTokens.slice(0, state.prefillProcessed),
    ];
    const payload = new TextEncoder().encode(
      JSON.stringify({
        promptText: state.promptText,
        tokens,
        stage: state.stage === 'prefill' ? 'prefill' : 'decode',
        decodeGenerated: state.decodeGenerated,
        prefillProcessed: state.prefillProcessed,
      })
    );
    const bytes = payload;
    const kv = encodeInt32Array(tokens);
    return {
      bytes,
      kv,
      tokens,
      stage: state.stage === 'prefill' ? 'prefill' : 'decode',
      decodeGenerated: state.decodeGenerated,
      formatVersion: 1,
      checksum: checksumBytes(concatBytes([bytes, kv])),
    };
  }

  async slotSnapshot(slotId: number): Promise<NativeSlotCheckpoint> {
    // The fake export is already non-destructive, so the same snapshot
    // implementation is valid for both export and fork.
    return this.slotExport(slotId);
  }

  async slotImport(
    checkpoint: NativeSlotCheckpoint,
    targetSlotId?: number
  ): Promise<NativeSlotStats> {
    if (checkpoint.parkedHandle !== undefined) {
      const parked = this.parked.get(checkpoint.parkedHandle);
      if (!parked) {
        throw new WllmError(
          `parked state not found: ${checkpoint.parkedHandle}`,
          'checkpoint_mismatch'
        );
      }
      this.consumeParked(checkpoint.parkedHandle);
      checkpoint = parked;
    }
    const tokensFromPayload = decodeInt32Array(
      checkpoint.kv ?? new Uint8Array()
    );
    if (
      tokensFromPayload.length !== checkpoint.tokens.length ||
      tokensFromPayload.some((token, index) => token !== checkpoint.tokens[index])
    ) {
      throw new WllmError('checkpoint token mismatch', 'checkpoint_mismatch');
    }

    const slotId = targetSlotId ?? this.allocateSlot('imported');
    if (slotId === null) {
      throw new WllmError('no free native slot', 'capacity');
    }
    const requestId = `import:${slotId}:${checkpoint.checksum}`;
    const continuation = decodeFakeContinuation(
      checkpoint.bytes,
      checkpoint.tokens
    );
    this.requests.set(requestId, {
      requestId,
      slotId,
      promptText: continuation.promptText,
      promptTokens: [...checkpoint.tokens],
      prefillProcessed: checkpoint.tokens.length,
      prefillCached: continuation.prefillProcessed ?? checkpoint.tokens.length,
      decodeGenerated: checkpoint.decodeGenerated,
      maxNewTokens: checkpoint.decodeGenerated,
      stage: checkpoint.stage,
      cancelled: false,
      failure: false,
      outputChunks: [],
    });
    this.slots.set(slotId, requestId);
    return {
      slotId,
      inUse: true,
      tokens: checkpoint.tokens.length,
      state: checkpoint.stage,
    };
  }

  // a resume moves the state into the live slot, so the parked copy goes away,
  // matching the wllama engine which releases the native park handle
  private consumeParked(handle: number | undefined): void {
    if (handle !== undefined) {
      this.parked.delete(handle);
    }
  }

  // tier two in the fake backend: parked states live in an engine-side map, so
  // the coordinator holds a handle rather than the checkpoint bytes
  async slotPark(slotId: number): Promise<NativeSlotCheckpoint> {
    const checkpoint = await this.slotExport(slotId);
    const parked = this.installParked(checkpoint);
    await this.slotDrop(slotId);
    return parked;
  }

  async slotParkCheckpoint(
    checkpoint: NativeSlotCheckpoint
  ): Promise<NativeSlotCheckpoint> {
    if (checkpoint.parkedHandle !== undefined) {
      return checkpoint;
    }
    return this.installParked(checkpoint);
  }

  async slotReadParked(
    checkpoint: NativeSlotCheckpoint
  ): Promise<NativeSlotCheckpoint> {
    const handle = checkpoint.parkedHandle;
    if (handle === undefined) {
      return checkpoint;
    }
    const parked = this.parked.get(handle);
    if (!parked) {
      throw new WllmError(`parked state not found: ${handle}`, 'checkpoint_mismatch');
    }
    return parked;
  }

  async slotReleaseParked(checkpoint: NativeSlotCheckpoint): Promise<void> {
    if (checkpoint.parkedHandle !== undefined) {
      this.parked.delete(checkpoint.parkedHandle);
    }
  }

  private installParked(
    checkpoint: NativeSlotCheckpoint
  ): NativeSlotCheckpoint {
    const handle = this.nextParkHandle++;
    const parked: NativeSlotCheckpoint = {
      ...checkpoint,
      parkedHandle: handle,
      parkedByteLength: checkpoint.bytes.byteLength,
    };
    this.parked.set(handle, parked);
    const { kv: _kv, ...hostHandle } = parked;
    void _kv;
    return hostHandle;
  }

  async slotDrop(slotId: number): Promise<void> {
    const requestId = this.slots.get(slotId);
    if (requestId) {
      this.slots.set(slotId, null);
      const state = this.requests.get(requestId);
      if (state) {
        state.stage = 'done';
        state.cancelled = true;
      }
      // the parked state now lives in the checkpoint, so the live one is gone
      this.requests.delete(requestId);
    }
  }

  async slotStats(): Promise<NativeSlotStats[]> {
    return [...this.slots.entries()].map(([slotId, requestId]) => {
      if (!requestId) {
        return { slotId, inUse: false, tokens: 0, state: 'idle' };
      }
      const state = this.requireRequest(requestId);
      return {
        slotId,
        inUse: true,
        tokens: state.prefillProcessed + state.decodeGenerated,
        state: state.stage,
      };
    });
  }

  failNextAdvance(nativeRequestId: string): void {
    this.requireRequest(nativeRequestId).failure = true;
  }

  // diagnostics for tests and the demo: native parked states, not host copies
  get parkedStateCount(): number {
    return this.parked.size;
  }

  get parkedByteLength(): number {
    let total = 0;
    for (const state of this.parked.values()) {
      total += state.bytes.byteLength;
    }
    return total;
  }

  private requireRequest(nativeRequestId: string): FakeRequestState {
    const state = this.requests.get(nativeRequestId);
    if (!state) {
      throw new WllmError(
        `native request not found: ${nativeRequestId}`,
        'unknown_request'
      );
    }
    return state;
  }

  private allocateSlot(owner: string): number | null {
    for (let offset = 0; offset < this.slotCount; offset += 1) {
      const slotId = (this.nextSlot + offset) % this.slotCount;
      if (this.slots.get(slotId) === null) {
        this.slots.set(slotId, owner);
        this.nextSlot = (slotId + 1) % this.slotCount;
        return slotId;
      }
    }
    return null;
  }

  private releaseRequestSlot(state: FakeRequestState): void {
    if (this.slots.get(state.slotId) === state.requestId) {
      this.slots.set(state.slotId, null);
    }
  }

  private progress(
    state: FakeRequestState,
    outputChunks: string[],
    elapsedMs: number
  ): NativeUnitProgress {
    return {
      requestId: state.requestId,
      stage: state.stage,
      prefillProcessed: state.prefillProcessed,
      prefillTotal: state.promptTokens.length,
      decodeGenerated: state.decodeGenerated,
      decodeTotal: state.maxNewTokens,
      done: state.stage === 'done',
      cancelled: state.cancelled,
      isError: state.failure,
      outputChunks: [...outputChunks],
      elapsedMs,
      slotId: state.cancelled ? null : state.slotId,
      // the fake engine advances one token per step, so its count is exact
      tokenAccounting: 'native-token',
      prefillCached: state.prefillCached,
      residentTokens: state.prefillProcessed + state.decodeGenerated,
    };
  }

  private measure(start: number, simulatedMs: number): number {
    const realMs = performance.now() - start;
    return Math.max(simulatedMs, realMs);
  }
}

function appendText(prefix: string, next: string): string {
  if (!prefix) {
    return next;
  }
  if (!next) {
    return prefix;
  }
  return `${prefix}${next.startsWith(' ') ? '' : ' '}${next}`;
}

function decodeFakeContinuation(
  bytes: Uint8Array,
  tokens: number[]
): { promptText: string; prefillProcessed?: number } {
  try {
    const parsed = JSON.parse(
      new TextDecoder().decode(bytes)
    ) as { promptText?: unknown; prefillProcessed?: unknown };
    return {
      promptText:
        typeof parsed.promptText === 'string' ? parsed.promptText : '',
      ...(typeof parsed.prefillProcessed === 'number' &&
      Number.isFinite(parsed.prefillProcessed)
        ? { prefillProcessed: parsed.prefillProcessed }
        : {}),
    };
  } catch {
    return { promptText: '' };
  }
}
