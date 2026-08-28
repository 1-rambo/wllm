import { Wllama, type WllamaChatFromNodeOptions } from '../wllama';
import type {
  BrowserClientLifecycle,
  BrowserServiceEvent,
  BrowserServiceInitPayload,
} from '../browser-service-protocol';

declare const self: {
  onconnect: ((event: MessageEvent) => void) | null;
  onmessage: ((event: MessageEvent) => void) | null;
  postMessage: (message: unknown, transfer?: Transferable[]) => void;
  close: () => void;
};
declare function importScripts(...urls: string[]): void;

const HEARTBEAT_TIMEOUT_MS = 6000;
const BACKGROUND_HEARTBEAT_TIMEOUT_MS = 2 * 60 * 1000;
const FROZEN_HEARTBEAT_TIMEOUT_MS = 5 * 60 * 1000;
const SWEEP_INTERVAL_MS = 2000;
const TELEMETRY_LIMIT = 10000;

interface ClientRecord {
  id: string;
  port: MessagePort;
  lastSeenAt: number;
  lifecycle: BrowserClientLifecycle;
  requests: Map<number, AbortController>;
}

let runtime: Wllama | null = null;
let status: 'uninitialized' | 'initializing' | 'ready' | 'error' =
  'uninitialized';
let modelKey: string | null = null;
let lastError: string | null = null;
let initializing: Promise<void> | null = null;
let sessionConfigKey: string | null = null;
let sessionInitializing: Promise<unknown> | null = null;
const clients = new Map<string, ClientRecord>();
const telemetry: BrowserServiceEvent[] = [];
let eventSequence = 0;

const now = (): number =>
  typeof performance !== 'undefined' ? performance.now() : Date.now();
const wallNow = (): number =>
  typeof performance !== 'undefined'
    ? performance.timeOrigin + performance.now()
    : Date.now();

const emit = (
  type: string,
  clientId?: string,
  requestId?: number,
  data?: Record<string, unknown>
) => {
  const record: BrowserServiceEvent = {
    schemaVersion: 'wllm.browser-service.v2',
    sequence: ++eventSequence,
    type,
    timestampMs: now(),
    wallTimeMs: wallNow(),
    ...(clientId ? { clientId } : {}),
    ...(requestId !== undefined ? { requestId } : {}),
    ...(data ? { data } : {}),
  };
  telemetry.push(record);
  if (telemetry.length > TELEMETRY_LIMIT) telemetry.shift();
  for (const client of clients.values()) {
    client.port.postMessage({ event: 'telemetry', record });
  }
};

const makeInScopeWorker = (code: string | Blob): Worker => {
  let workerHandler: ((event: MessageEvent) => void) | null = null;
  const fakeWorker = {
    onmessage: null as ((event: MessageEvent) => void) | null,
    onerror: null as ((event: ErrorEvent) => void) | null,
    postMessage: (message: unknown) => {
      queueMicrotask(() => workerHandler?.({ data: message } as MessageEvent));
    },
    terminate: () => {},
  };
  self.postMessage = (message: unknown) => {
    queueMicrotask(() =>
      fakeWorker.onmessage?.({ data: message } as MessageEvent)
    );
  };
  const url = URL.createObjectURL(
    typeof code === 'string'
      ? new Blob([code], { type: 'text/javascript' })
      : code
  );
  importScripts(url);
  workerHandler = self.onmessage;
  self.onmessage = null;
  return fakeWorker as unknown as Worker;
};

const requireRuntime = (): Wllama => {
  if (!runtime || status !== 'ready') {
    throw new Error(`browser LLM service is not ready (${status})`);
  }
  return runtime;
};

const initialize = async (
  payload: BrowserServiceInitPayload
): Promise<void> => {
  if (status === 'ready') {
    if (modelKey !== payload.modelKey) {
      throw new Error(`another model is already loaded: ${modelKey}`);
    }
    return;
  }
  if (initializing) {
    await initializing;
    if (modelKey !== payload.modelKey) {
      throw new Error(`another model is already loaded: ${modelKey}`);
    }
    return;
  }

  status = 'initializing';
  modelKey = payload.modelKey;
  initializing = (async () => {
    runtime = new Wllama(
      payload.pathConfig,
      {
        ...payload.config,
        preferWebGPU: payload.config.preferWebGPU ?? true,
        engineChatSchedulingPolicy:
          payload.config.engineChatSchedulingPolicy ?? 'foreground-aware',
        engineChatEventListener: (event) =>
          emit(event.type, event.clientId, event.requestId, event.data),
      },
      {
        moduleCode: payload.moduleCode,
        llamaCppCode: payload.llamaCppCode,
        workerFactory: makeInScopeWorker,
      }
    );
    await runtime.loadModel(
      payload.modelFiles.map((file) => file.blob),
      { ...payload.loadConfig, n_threads: 1 }
    );
    for (const client of clients.values()) {
      await runtime.updateEngineChatClientState(client.id, client.lifecycle);
    }
    status = 'ready';
    emit('service_ready', undefined, undefined, { modelKey: payload.modelKey });
    for (const client of clients.values()) {
      client.port.postMessage({ event: 'ready' });
    }
  })();

  try {
    await initializing;
  } catch (error) {
    status = 'error';
    lastError = error instanceof Error ? error.message : String(error);
    runtime = null;
    throw error;
  } finally {
    initializing = null;
  }
};

const cancelClientRequests = (client: ClientRecord) => {
  for (const controller of client.requests.values()) controller.abort();
  client.requests.clear();
};

const removeClient = (clientId: string) => {
  const client = clients.get(clientId);
  if (!client) return;
  cancelClientRequests(client);
  runtime?.removeEngineChatClient(clientId);
  clients.delete(clientId);
  emit('client_disconnected', clientId);
};

setInterval(() => {
  const current = Date.now();
  for (const client of clients.values()) {
    if (
      current - client.lastSeenAt >
      (client.lifecycle.lifecycleState === 'frozen'
        ? FROZEN_HEARTBEAT_TIMEOUT_MS
        : client.lifecycle.lifecycleState === 'active'
          ? HEARTBEAT_TIMEOUT_MS
          : BACKGROUND_HEARTBEAT_TIMEOUT_MS)
    ) {
      removeClient(client.id);
    }
  }
}, SWEEP_INTERVAL_MS);

const handleRequest = async (
  client: ClientRecord,
  verb: string,
  payload: any
): Promise<unknown> => {
  client.lastSeenAt = Date.now();
  switch (verb) {
    case 'heartbeat':
      return null;
    case 'status':
      return {
        status,
        modelKey,
        clientCount: clients.size,
        lastError,
        runtimeConfig:
          runtime && status === 'ready'
            ? {
                usingWebGPU: runtime.usingWebGPU(),
                multithread: runtime.isMultithread(),
                numThreads: runtime.getNumThreads(),
              }
            : null,
      };
    case 'init':
      await initialize(payload);
      return null;
    case 'session-init': {
      const key = JSON.stringify({
        memoryCapBytes: payload.memoryCapBytes,
        tieredCache: payload.tieredCache,
      });
      if (sessionConfigKey !== null) {
        if (sessionConfigKey !== key) {
          throw new Error(
            'browser LLM service session already uses another configuration'
          );
        }
        return await requireRuntime().chatGetState();
      }
      if (sessionInitializing) {
        await sessionInitializing;
        if (sessionConfigKey !== key) {
          throw new Error(
            'browser LLM service session initialized with another configuration'
          );
        }
        return await requireRuntime().chatGetState();
      }
      sessionInitializing = requireRuntime().chatSessionInit(
        payload.memoryCapBytes,
        payload.tieredCache
      );
      try {
        const state = await sessionInitializing;
        sessionConfigKey = key;
        return state;
      } finally {
        sessionInitializing = null;
      }
    }
    case 'state':
      return await requireRuntime().chatGetState();
    case 'session-reset':
      return await requireRuntime().chatReset();
    case 'session-delete':
      return await requireRuntime().chatDeleteNode(payload.nodeId);
    case 'session-release':
      await requireRuntime().releaseEngineChatStates(
        payload.nodeIds,
        payload.reason
      );
      return null;
    case 'lifecycle': {
      const update = payload as BrowserClientLifecycle;
      if (update.lifecycleEpoch >= client.lifecycle.lifecycleEpoch) {
        client.lifecycle = update;
        if (runtime && status === 'ready') {
          await runtime.updateEngineChatClientState(client.id, update);
        } else {
          emit('foreground_change', client.id, undefined, {
            foreground: update.foreground,
            lifecycleState: update.lifecycleState,
            lifecycleEpoch: update.lifecycleEpoch,
          });
        }
      }
      return null;
    }
    case 'flush-lifecycle':
      await requireRuntime().flushEngineChatRetentionPolicies();
      return null;
    case 'chat': {
      const controller = new AbortController();
      client.requests.set(payload.requestId, controller);
      const options: WllamaChatFromNodeOptions = {
        ...payload.options,
        clientId: client.id,
        foreground: client.lifecycle.foreground,
        lifecycleState: client.lifecycle.lifecycleState,
        lifecycleEpoch: client.lifecycle.lifecycleEpoch,
        abortSignal: controller.signal,
        onChunk: (piece, fullText) => {
          client.port.postMessage({
            event: 'chunk',
            clientId: client.id,
            requestId: payload.requestId,
            piece,
            fullText,
          });
        },
      };
      try {
        return await requireRuntime().chatFromNode(
          payload.parentId,
          payload.userText,
          options
        );
      } finally {
        client.requests.delete(payload.requestId);
      }
    }
    case 'cancel':
      client.requests.get(payload.requestId)?.abort();
      return null;
    case 'debug':
      return {
        service: {
          status,
          modelKey,
          clients: [...clients.values()].map((item) => ({
            id: item.id,
            lifecycle: item.lifecycle,
            pendingRequests: item.requests.size,
          })),
          telemetry: telemetry.slice(-500),
        },
        runtimeConfig: {
          usingWebGPU: requireRuntime().usingWebGPU(),
          multithread: requireRuntime().isMultithread(),
          numThreads: requireRuntime().getNumThreads(),
        },
        runtime: await requireRuntime()._getDebugInfo(),
      };
    case 'telemetry': {
      const snapshot = [...telemetry];
      if (payload?.clear) telemetry.length = 0;
      return snapshot;
    }
    case 'record-event':
      emit(payload.type, client.id, undefined, payload.data);
      return null;
    case 'detach':
      removeClient(client.id);
      return null;
    case 'destroy':
      if (runtime) await runtime.exit();
      for (const item of clients.values()) {
        item.port.postMessage({ event: 'dying', reason: 'service destroyed' });
      }
      setTimeout(() => self.close(), 0);
      return null;
    default:
      throw new Error(`unknown browser LLM service verb: ${verb}`);
  }
};

self.onconnect = (event: MessageEvent) => {
  const port = event.ports[0];
  let portClientId: string | null = null;
  port.onmessage = async (message: MessageEvent) => {
    const { id, verb, payload, clientId } = message.data ?? {};
    if (!id || !clientId) return;
    let client = clients.get(clientId);
    if (!client) {
      portClientId = clientId;
      client = {
        id: clientId,
        port,
        lastSeenAt: Date.now(),
        lifecycle: {
          foreground: false,
          lifecycleState: 'hidden',
          lifecycleEpoch: 0,
          changedAt: now(),
        },
        requests: new Map(),
      };
      clients.set(clientId, client);
      emit('client_connected', clientId);
    }
    try {
      const result = await handleRequest(client, verb, payload);
      port.postMessage({ id, result });
    } catch (error) {
      port.postMessage({
        id,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  };
  port.onmessageerror = () => {
    if (portClientId) removeClient(portClientId);
  };
  port.start();
  port.postMessage({ event: 'hello', clientId: '' });
};
