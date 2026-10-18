import type { Clock } from './contracts';
import { systemClock } from './contracts';

export type TelemetryEventName =
  | 'lifecycle'
  | 'request_enqueue'
  | 'request_select'
  | 'unit_start'
  | 'unit_end'
  | 'request_requeue'
  | 'request_complete'
  | 'request_cancel'
  | 'state_residency'
  | 'state_victim'
  | 'state_restore'
  | 'state_restore_failed'
  | 'native_submit'
  | 'native_cancel'
  | 'native_worker_loss'
  | 'gpu_burst_policy'
  | 'render_yield'
  | 'render_gate'
  | 'lease_unit'
  | 'gpu_burst_policy_error';

export interface TelemetryEvent {
  engineSeq: number;
  timestamp: number;
  name: TelemetryEventName;
  requestId?: string;
  clientId?: string;
  sessionId?: string;
  data: Readonly<Record<string, unknown>>;
}

export interface TelemetrySink {
  emit(
    name: TelemetryEventName,
    data?: Readonly<Record<string, unknown>>,
    context?: {
      requestId?: string;
      clientId?: string;
      sessionId?: string;
    }
  ): TelemetryEvent;
}

export class InMemoryTelemetry implements TelemetrySink {
  readonly events: TelemetryEvent[] = [];
  private sequence = 0;

  constructor(private readonly clock: Clock = systemClock) {}

  emit(
    name: TelemetryEventName,
    data: Readonly<Record<string, unknown>> = {},
    context: {
      requestId?: string;
      clientId?: string;
      sessionId?: string;
    } = {}
  ): TelemetryEvent {
    const event: TelemetryEvent = {
      engineSeq: ++this.sequence,
      timestamp: this.clock.now(),
      name,
      data: { ...data },
      ...(context.requestId !== undefined
        ? { requestId: context.requestId }
        : {}),
      ...(context.clientId !== undefined ? { clientId: context.clientId } : {}),
      ...(context.sessionId !== undefined
        ? { sessionId: context.sessionId }
        : {}),
    };
    this.events.push(event);
    return event;
  }
}
