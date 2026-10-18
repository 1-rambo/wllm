import type {
  GpuBurstPolicy,
  LifecycleRecord,
  NativeRuntimeEvent,
  RenderPressureSignal,
  RequestId,
  RequestRecord,
} from '../contracts';
import type {
  CoordinatorRequest,
  CoordinatorResponse,
  CoordinatorEvent,
  MessagePortLike,
  CoordinatorDiagnostics,
} from '../transport/protocol';
export interface BrowserLLMClientEvents {
  onUnit?: (event: CoordinatorEvent) => void;
  onFenced?: (requestId: RequestId) => void;
  onRuntime?: (event: NativeRuntimeEvent) => void;
}

export class BrowserLLMClient {
  private callbackCounter = 0;
  private lifecycle: LifecycleRecord | null = null;
  private readonly pending = new Map<
    string,
    { resolve: (value: unknown) => void; reject: (error: Error) => void }
  >();

  constructor(
    private readonly port: MessagePortLike,
    private readonly events: BrowserLLMClientEvents = {}
  ) {
    port.addEventListener('message', (event) => this.onMessage(event.data));
    port.start?.();
  }

  async attach(clientId: string, lifecycle: LifecycleRecord): Promise<void> {
    this.lifecycle = { ...lifecycle };
    await this.call('attach', (callbackId) => ({
      type: 'attach',
      callbackId,
      clientId,
      lifecycle,
    }));
  }

  async detach(clientId: string): Promise<void> {
    await this.call('detach', (callbackId) => ({
      type: 'detach',
      callbackId,
      clientId,
    }));
  }

  async setLifecycle(lifecycle: LifecycleRecord): Promise<boolean> {
    this.lifecycle = { ...lifecycle };
    return this.call<boolean>('lifecycle', (callbackId) => ({
      type: 'lifecycle',
      callbackId,
      record: lifecycle,
    }));
  }

  async reportRenderPressure(
    signal: RenderPressureSignal
  ): Promise<GpuBurstPolicy> {
    return this.call<GpuBurstPolicy>('render.signal', (callbackId) => ({
      type: 'render.signal',
      callbackId,
      signal,
    }));
  }

  async submit(
    request: Omit<RequestRecord, 'lifecycle'>,
    lifecycleOverride?: LifecycleRecord
  ): Promise<{ accepted: boolean; reason?: string }> {
    if (!this.lifecycle) {
      throw new Error('client is not attached');
    }
    return this.call('submit', (callbackId) => ({
      type: 'submit',
      callbackId,
      request: {
        ...request,
        lifecycle: { ...(lifecycleOverride ?? this.lifecycle!) },
      },
    }));
  }

  async cancel(requestId: RequestId): Promise<boolean> {
    return this.call('cancel', (callbackId) => ({
      type: 'cancel',
      callbackId,
      requestId,
    }));
  }

  async status(requestId: RequestId): Promise<RequestRecord | null> {
    return this.call('status', (callbackId) => ({
      type: 'status',
      callbackId,
      requestId,
    }));
  }

  async pump(maxSteps = 1): Promise<number> {
    return this.call('pump', (callbackId) => ({
      type: 'pump',
      callbackId,
      maxSteps,
    }));
  }

  async diagnostics(): Promise<CoordinatorDiagnostics> {
    return this.call('diagnostics', (callbackId) => ({
      type: 'diagnostics',
      callbackId,
    }));
  }

  private call<T>(
    label: string,
    build: (callbackId: string) => CoordinatorRequest
  ): Promise<T> {
    const callbackId = `${label}:${++this.callbackCounter}`;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(callbackId, {
        resolve: resolve as (value: unknown) => void,
        reject,
      });
      this.port.postMessage(build(callbackId));
    });
  }

  private onMessage(data: unknown): void {
    if (!isCoordinatorResponse(data)) {
      return;
    }
    if (data.type === 'event') {
      this.events.onUnit?.(data.event);
      if (data.event.type === 'runtime') {
        this.events.onRuntime?.(data.event);
      }
      if (data.event.type === 'fenced') {
        this.events.onFenced?.(data.event.requestId);
      }
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
      pending.reject(new Error(data.error));
    }
  }
}

function isCoordinatorResponse(value: unknown): value is CoordinatorResponse {
  return (
    typeof value === 'object' &&
    value !== null &&
    ((value as { type?: unknown }).type === 'response' ||
      (value as { type?: unknown }).type === 'event')
  );
}
