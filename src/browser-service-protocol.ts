import type {
  LoadModelConfig,
  WllamaChatFromNodeOptions,
  WllamaConfig,
  WllamaTieredCacheOptions,
} from './wllama';

export type BrowserLifecycleState = 'active' | 'passive' | 'hidden' | 'frozen';

export interface BrowserClientLifecycle {
  foreground: boolean;
  lifecycleState: BrowserLifecycleState;
  lifecycleEpoch: number;
  changedAt: number;
}

export interface BrowserServiceEvent {
  schemaVersion: 'wllm.browser-service.v1';
  sequence: number;
  type: string;
  timestampMs: number;
  clientId?: string;
  requestId?: number;
  data?: Record<string, unknown>;
}

export interface BrowserServiceStatus {
  status: 'uninitialized' | 'initializing' | 'ready' | 'error';
  modelKey: string | null;
  clientCount: number;
  lastError: string | null;
}

export interface BrowserServiceInitPayload {
  modelKey: string;
  pathConfig: {
    'single-thread/wllama.wasm': string;
    'multi-thread/wllama.wasm'?: string;
  };
  config: WllamaConfig;
  loadConfig: LoadModelConfig;
  modelFiles: Array<{ name: string; blob: Blob }>;
  moduleCode: string;
  llamaCppCode: string;
}

export type BrowserServiceChatOptions = Omit<
  WllamaChatFromNodeOptions,
  'onChunk' | 'abortSignal'
>;

export type BrowserServiceRequest =
  | { id: number; verb: 'heartbeat'; payload?: undefined }
  | { id: number; verb: 'status'; payload?: undefined }
  | { id: number; verb: 'init'; payload: BrowserServiceInitPayload }
  | {
      id: number;
      verb: 'session-init';
      payload: {
        memoryCapBytes: number;
        tieredCache: WllamaTieredCacheOptions;
      };
    }
  | { id: number; verb: 'session-reset'; payload?: undefined }
  | { id: number; verb: 'session-delete'; payload: { nodeId: number } }
  | {
      id: number;
      verb: 'chat';
      payload: {
        requestId: number;
        parentId: number;
        userText: string;
        options: BrowserServiceChatOptions;
      };
    }
  | { id: number; verb: 'state'; payload?: undefined }
  | {
      id: number;
      verb: 'lifecycle';
      payload: BrowserClientLifecycle;
    }
  | { id: number; verb: 'flush-lifecycle'; payload?: undefined }
  | { id: number; verb: 'cancel'; payload: { requestId: number } }
  | { id: number; verb: 'debug'; payload?: undefined }
  | { id: number; verb: 'telemetry'; payload: { clear?: boolean } }
  | {
      id: number;
      verb: 'record-event';
      payload: { type: string; data?: Record<string, unknown> };
    }
  | { id: number; verb: 'detach'; payload?: undefined }
  | { id: number; verb: 'destroy'; payload?: undefined };

export type BrowserServiceMessage =
  | { id: number; result: unknown }
  | { id: number; error: string }
  | { event: 'hello'; clientId: string }
  | { event: 'ready' }
  | { event: 'dying'; reason: string }
  | {
      event: 'chunk';
      clientId: string;
      requestId: number;
      piece: string;
      fullText: string;
    }
  | { event: 'telemetry'; record: BrowserServiceEvent };
