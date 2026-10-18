import type {
  CoordinatorRequest,
  CoordinatorResponse,
  MessagePortLike,
} from '../transport/protocol';
import type { SharedWorkerCoordinator } from './shared-worker-coordinator';

export interface CoordinatorPortHooks {
  // Called after every coordinator request settles. The host uses this to
  // keep an event-driven pump alive without polling on a fixed interval.
  onActivity?: () => void;
}

export function startCoordinatorPort(
  port: MessagePortLike,
  coordinator: SharedWorkerCoordinator,
  hooks: CoordinatorPortHooks = {}
): void {
  const clientIds = new Set<string>();
  coordinator.subscribe((event) => {
    if (event.type !== 'runtime' && !clientIds.has(event.clientId)) {
      return;
    }
    port.postMessage({ type: 'event', event } satisfies CoordinatorResponse);
  });
  port.addEventListener('message', (event) => {
    if (isCoordinatorRequest(event.data)) {
      if (event.data.type === 'attach') {
        clientIds.add(event.data.clientId);
      } else if (event.data.type === 'detach') {
        clientIds.delete(event.data.clientId);
      }
    }
    void handleMessage(port, coordinator, event.data, hooks);
  });
  port.start?.();
}

async function handleMessage(
  port: MessagePortLike,
  coordinator: SharedWorkerCoordinator,
  data: unknown,
  hooks: CoordinatorPortHooks
): Promise<void> {
  if (!isCoordinatorRequest(data)) {
    return;
  }
  try {
    const result = await dispatch(coordinator, data);
    port.postMessage({
      type: 'response',
      callbackId: data.callbackId,
      ok: true,
      result,
    } satisfies CoordinatorResponse);
  } catch (error) {
    port.postMessage({
      type: 'response',
      callbackId: data.callbackId,
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    } satisfies CoordinatorResponse);
  } finally {
    hooks.onActivity?.();
  }
}

async function dispatch(
  coordinator: SharedWorkerCoordinator,
  request: CoordinatorRequest
): Promise<unknown> {
  switch (request.type) {
    case 'attach':
      await coordinator.attach(request.clientId, request.lifecycle);
      return null;
    case 'detach':
      await coordinator.detach(request.clientId);
      return null;
    case 'lifecycle':
      return coordinator.updateLifecycle(request.record);
    case 'render.signal':
      return coordinator.updateRenderPressure(request.signal);
    case 'submit':
      return coordinator.submit(request.request);
    case 'cancel':
      return coordinator.cancel(request.requestId);
    case 'status':
      return coordinator.status(request.requestId);
    case 'pump':
      return coordinator.pump(request.maxSteps ?? 1);
    case 'diagnostics':
      return coordinator.diagnostics();
  }
}

function isCoordinatorRequest(value: unknown): value is CoordinatorRequest {
  if (typeof value !== 'object' || value === null) {
    return false;
  }
  const candidate = value as { type?: unknown; callbackId?: unknown };
  return (
    typeof candidate.callbackId === 'string' &&
    (candidate.type === 'attach' ||
      candidate.type === 'detach' ||
      candidate.type === 'lifecycle' ||
      candidate.type === 'render.signal' ||
      candidate.type === 'submit' ||
      candidate.type === 'cancel' ||
      candidate.type === 'status' ||
      candidate.type === 'pump' ||
      candidate.type === 'diagnostics')
  );
}
