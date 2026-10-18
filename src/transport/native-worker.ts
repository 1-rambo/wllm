import type {
  MessagePortLike,
  NativeWorkerEvent,
  NativeWorkerRequest,
  NativeWorkerResponse,
} from './protocol';
import type { NativeEngine } from '../native/native-engine';

export class MessagePortRpcClient {
  private nextCallback = 1;
  private readonly pending = new Map<
    string,
    { resolve: (value: unknown) => void; reject: (error: Error) => void }
  >();

  constructor(
    private readonly port: MessagePortLike,
    private readonly onEvent?: (event: NativeWorkerEvent) => void
  ) {
    port.addEventListener('message', (event) => this.onMessage(event.data));
    port.start?.();
  }

  call<T>(method: NativeWorkerRequest['method'], args: unknown[]): Promise<T> {
    const callbackId = `${this.nextCallback++}`;
    const message: NativeWorkerRequest = {
      type: 'native.call',
      callbackId,
      method,
      args,
    };
    return new Promise<T>((resolve, reject) => {
      this.pending.set(callbackId, {
        resolve: resolve as (value: unknown) => void,
        reject,
      });
      this.port.postMessage(message);
    });
  }

  dispose(error = new Error('native worker disconnected')): void {
    for (const pending of this.pending.values()) {
      pending.reject(error);
    }
    this.pending.clear();
  }

  private onMessage(data: unknown): void {
    if (isNativeWorkerEvent(data)) {
      if (data.type === 'native.chunk') {
        this.onEvent?.({ type: 'native.chunk', event: data.event });
      } else {
        this.onEvent?.({ type: 'native.event', event: data.event });
      }
      return;
    }
    if (!isNativeWorkerResponse(data)) {
      return;
    }
    const pending = this.pending.get(data.callbackId);
    if (!pending) {
      return;
    }
    this.pending.delete(data.callbackId);
    if (data.ok) {
      pending.resolve(data.result);
    } else {
      pending.reject(new Error(data.error ?? 'native worker error'));
    }
  }
}

export class NativeEngineWorkerServer {
  constructor(
    private readonly port: MessagePortLike,
    private readonly engine: NativeEngine
  ) {
    port.addEventListener('message', (event) => {
      void this.handle(event.data);
    });
    port.start?.();
  }

  private async handle(data: unknown): Promise<void> {
    if (!isNativeWorkerRequest(data)) {
      return;
    }
    try {
      const result = await this.call(data.method, data.args);
      this.port.postMessage({
        type: 'native.result',
        callbackId: data.callbackId,
        ok: true,
        result,
      } satisfies NativeWorkerResponse);
    } catch (error) {
      this.port.postMessage({
        type: 'native.result',
        callbackId: data.callbackId,
        ok: false,
        error: error instanceof Error ? error.message : String(error),
      } satisfies NativeWorkerResponse);
    }
  }

  private call(method: NativeWorkerRequest['method'], args: unknown[]): Promise<unknown> {
    const engine = this.engine as unknown as Record<
      string,
      (...callArgs: unknown[]) => Promise<unknown>
    >;
    const fn = engine[method];
    if (!fn) {
      throw new Error(`unsupported native method: ${method}`);
    }
    return fn.apply(this.engine, args);
  }
}

export class NativeEngineProxy implements NativeEngine {
  constructor(private readonly client: MessagePortRpcClient) {}

  dispose(error?: Error): void {
    this.client.dispose(error);
  }

  async prewarm(): Promise<void> {
    await this.client.call('prewarm', []);
  }

  async setGpuBurstPolicy(
    policy: Parameters<NativeEngine['setGpuBurstPolicy']>[0]
  ): Promise<void> {
    await this.client.call('setGpuBurstPolicy', [policy]);
  }

  submit(request: Parameters<NativeEngine['submit']>[0]): ReturnType<NativeEngine['submit']> {
    return this.client.call('submit', [request]);
  }

  advance(
    requestId: string,
    budget: Parameters<NativeEngine['advance']>[1]
  ): ReturnType<NativeEngine['advance']> {
    return this.client.call('advance', [requestId, budget]);
  }

  status(requestId: string): ReturnType<NativeEngine['status']> {
    return this.client.call('status', [requestId]);
  }

  async cancel(requestId: string): Promise<void> {
    await this.client.call('cancel', [requestId]);
  }

  async interrupt(requestId: string): Promise<void> {
    await this.client.call('interrupt', [requestId]);
  }

  slotExport(slotId: number): ReturnType<NativeEngine['slotExport']> {
    return this.client.call('slotExport', [slotId]);
  }

  slotSnapshot(slotId: number): ReturnType<NativeEngine['slotSnapshot']> {
    return this.client.call('slotSnapshot', [slotId]);
  }

  slotImport(
    checkpoint: Parameters<NativeEngine['slotImport']>[0],
    targetSlotId?: number
  ): ReturnType<NativeEngine['slotImport']> {
    return this.client.call('slotImport', [checkpoint, targetSlotId]);
  }

  async slotDrop(slotId: number): Promise<void> {
    await this.client.call('slotDrop', [slotId]);
  }

  slotPark(slotId: number): ReturnType<NativeEngine['slotPark']> {
    return this.client.call('slotPark', [slotId]);
  }

  slotParkCheckpoint(
    checkpoint: Parameters<NativeEngine['slotParkCheckpoint']>[0]
  ): ReturnType<NativeEngine['slotParkCheckpoint']> {
    return this.client.call('slotParkCheckpoint', [checkpoint]);
  }

  slotReadParked(
    checkpoint: Parameters<NativeEngine['slotReadParked']>[0]
  ): ReturnType<NativeEngine['slotReadParked']> {
    return this.client.call('slotReadParked', [checkpoint]);
  }

  async slotReleaseParked(
    checkpoint: Parameters<NativeEngine['slotReleaseParked']>[0]
  ): Promise<void> {
    await this.client.call('slotReleaseParked', [checkpoint]);
  }

  slotStats(): ReturnType<NativeEngine['slotStats']> {
    return this.client.call('slotStats', []);
  }
}

function isNativeWorkerRequest(value: unknown): value is NativeWorkerRequest {
  return (
    typeof value === 'object' &&
    value !== null &&
    (value as { type?: unknown }).type === 'native.call' &&
    typeof (value as { callbackId?: unknown }).callbackId === 'string'
  );
}

function isNativeWorkerResponse(
  value: unknown
): value is NativeWorkerResponse {
  return (
    typeof value === 'object' &&
    value !== null &&
    (value as { type?: unknown }).type === 'native.result' &&
    typeof (value as { callbackId?: unknown }).callbackId === 'string'
  );
}

function isNativeWorkerEvent(value: unknown): value is NativeWorkerEvent {
  if (typeof value !== 'object' || value === null) {
    return false;
  }
  const candidate = value as { type?: unknown; event?: unknown };
  return (
    (candidate.type === 'native.event' || candidate.type === 'native.chunk') &&
    typeof candidate.event === 'object' &&
    candidate.event !== null
  );
}
