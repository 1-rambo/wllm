import {
  Activity,
  Cpu,
  Database,
  Eraser,
  Inbox,
  Layers3,
  ListPlus,
  Moon,
  PanelTopOpen,
  Play,
  Radio,
  Zap,
  createIcons,
} from 'lucide';
import type {
  CoordinatorEvent,
  ExecutionConfig,
  GpuBurstPolicy,
  LifecycleRecord,
  NativeRuntimeEvent,
  NativeTokenCounters,
  RequestRecord,
  TokenAccounting,
} from '../index';
import {
  InferenceBridge,
  type InferenceBridgeOptions,
} from '../client/inference-bridge';
import { BrowserLLMClient } from '../client/browser-llm-client';
import {
  observeVisibilityPriority,
  type VisibilityObservation,
} from './visibility-priority';
import { observeFramePressure } from '../client/frame-pressure-monitor';
import coordinatorWorkerAssetUrl from '../workers/coordinator.shared-worker.ts?sharedworker&url';
import { coordinatorWorkerName } from '../coordinator/worker-name';
import './styles.css';

type RequestStatus =
  | 'queued'
  | 'running'
  | 'complete'
  | 'cancelled'
  | 'failed';

interface RequestView {
  requestId: string;
  sessionId: string;
  prompt: string;
  maxNewTokens: number;
  foreground: boolean;
  status: RequestStatus;
  output: string;
  units: number;
  decodeGenerated: number;
  tokenAccounting: TokenAccounting;
  counters?: NativeTokenCounters;
  prefillProcessed: number;
  prefillTotal: number;
  prefillCached: number;
  initialPrefillCached: number | null;
  slotId: number | null;
  submittedAt: number;
  startedAt: number | null;
  firstTokenAt: number | null;
  finishedAt: number | null;
  item: HTMLElement;
  state: HTMLElement;
  outputElement: HTMLElement;
  metrics: HTMLElement;
  cancelButton: HTMLButtonElement;
}

const iconSet = {
  Activity,
  Cpu,
  Database,
  Eraser,
  Inbox,
  Layers3,
  ListPlus,
  Moon,
  PanelTopOpen,
  Play,
  Radio,
  Zap,
};

createIcons({ icons: iconSet });

// Each tab is a distinct client: a new tab must not fence the tab it was opened from.
const pageParams = new URLSearchParams(window.location.search);
const primaryClientId = pageParams.get('clientId') ?? makeId();
const sessionId = pageParams.get('sessionId') ?? makeId();
const parentNodeId =
  pageParams.get('parentNodeId') ?? pageParams.get('nodeId');
const backgroundClientIds = Array.from({ length: 4 }, () => makeId());
// measurement control: a fresh session per request removes the coordinator's
// continuation replay, so every request in one sweep has the same prompt
// cost instead of a prompt that grows with each turn.
const freshSessionPerRequest = readBooleanParam(
  pageParams.get('freshSession'),
  false
);
// measurement control: a keeper tab hosts the inference worker but must not
// report frame pressure for the tab that is actually being measured.
const reportRenderPressure = readBooleanParam(
  pageParams.get('renderPressure'),
  true
);
let freshSessionSequence = 0;

const promptElement = requiredElement<HTMLTextAreaElement>('prompt');
const maxTokensElement = requiredElement<HTMLInputElement>('max-tokens');
const seedElement = requiredElement<HTMLInputElement>('seed');
const submitButton = requiredElement<HTMLButtonElement>('submit');
const burstButton = requiredElement<HTMLButtonElement>('burst');
const newTabButton = requiredElement<HTMLButtonElement>('new-tab');
const clearFinishedButton =
  requiredElement<HTMLButtonElement>('clear-finished');
const sessionLabel = requiredElement<HTMLElement>('session-label');
const runtimeStatus = requiredElement<HTMLElement>('runtime-status');
const agentStatus = requiredElement<HTMLElement>('agent-status');
const queueCount = requiredElement<HTMLElement>('queue-count');
const stateTiers = requiredElement<HTMLElement>('state-tiers');
const modelProgress = requiredElement<HTMLElement>('model-progress');
const modelProgressText = requiredElement<HTMLElement>('model-progress-text');
const trace = requiredElement<HTMLOListElement>('trace');
const traceCounter = requiredElement<HTMLElement>('trace-counter');
const requestsElement = requiredElement<HTMLElement>('requests');
const frameStatus = requiredElement<HTMLElement>('frame-status');
const frameMetrics = requiredElement<HTMLElement>('frame-metrics');
const frameSweep = requiredElement<HTMLElement>('frame-sweep');
const frameMarkButton = requiredElement<HTMLButtonElement>('frame-mark');
const frameResetButton = requiredElement<HTMLButtonElement>('frame-reset');
const emptyState = requiredElement<HTMLElement>('empty-state');
const modeButtons = Array.from(
  document.querySelectorAll<HTMLButtonElement>('.mode-button')
);

const inferenceOptions = readInferenceOptions();
const runtimeGlobals = window as unknown as Record<string, unknown>;
const burstPolicyHistory: Array<
  GpuBurstPolicy & { reason: string; clientId: string | null; recordedAt: number }
> = [];
runtimeGlobals.__wllmRenderAware = inferenceOptions.renderAware ?? true;
runtimeGlobals.__wllmBurstPolicyHistory = burstPolicyHistory;

function recordBurstPolicy(
  policy: GpuBurstPolicy,
  reason: string,
  clientId: string | null
): void {
  const record = {
    ...policy,
    reason,
    clientId,
    recordedAt: Date.now(),
  };
  runtimeGlobals.__wllmBurstPolicy = record;
  burstPolicyHistory.push(record);
}

const coordinatorUrl = new URL(coordinatorWorkerAssetUrl, import.meta.url);
applyCoordinatorWorkerOptions(coordinatorUrl, inferenceOptions);

const coordinatorWorkerNameValue = await coordinatorWorkerName(coordinatorUrl);
const worker = new SharedWorker(coordinatorUrl, {
  name: coordinatorWorkerNameValue,
  type: 'module',
});

let epoch = 1;
const inferenceBridge = new InferenceBridge(
  worker.port as unknown as MessagePort,
  readInferenceOptions(),
  {
    onState: (state, detail) => {
      if (state === 'offering' || state === 'idle') {
        runtimeStatus.textContent = 'Starting inference engine';
      } else if (state === 'released') {
        runtimeStatus.textContent = 'Inference engine released';
      }
      if (detail) {
        agentStatus.textContent = `Connected / ${shortId(primaryClientId)} / ${detail}`;
      }
    },
  }
);
let manualBackground = false;

const lifecycleObserver = observeVisibilityPriority({
  forcedBackground: () => manualBackground,
  onChange: (observation) => {
    void applyLifecycle(observation);
  },
});
let traceUnits = 0;
const activeViews = new Map<string, RequestView>();
const residencyByNode = new Map<
  string,
  { level: string; slotId: number | null }
>();

const client = new BrowserLLMClient(
  worker.port as unknown as ConstructorParameters<typeof BrowserLLMClient>[0],
  {
    onUnit: (event) => handleCoordinatorEvent(event),
    onRuntime: (event) => handleRuntimeEvent(event),
    onFenced: (requestId) => {
      const view = activeViews.get(requestId);
      if (view) {
        updateStatus(view, 'cancelled');
      }
    },
  }
);

const framePressureMonitor = observeFramePressure({
  canvas: () =>
    document.querySelector<HTMLCanvasElement>(
      '#gpu-render-probe, canvas[data-wllm-presentation]'
    ),
  onSample: (sample) => {
    const current = lifecycleObserver.current('render-pressure');
    if (!current.foreground) {
      return;
    }
    (window as unknown as Record<string, unknown>).__wllmRenderPressure = {
      ...sample,
      epoch,
      sampledAt: Date.now(),
    };
    void client
      .reportRenderPressure({
        clientId: primaryClientId,
        epoch,
        sampledAt: Date.now(),
        ...sample,
      })
      .catch(() => undefined);
  },
});
if (reportRenderPressure) {
  framePressureMonitor.start();
}

sessionLabel.textContent = `Client ${shortId(primaryClientId)} / session ${shortId(sessionId)}`;
agentStatus.textContent = 'Connecting';

void initialize().catch((error: unknown) => reportFailure('Initialize', error));

async function initialize(): Promise<void> {
  const startupLifecycle = lifecycleObserver.current('startup');
  for (const clientId of backgroundClientIds) {
    await client.attach(
      clientId,
      lifecycleFor(clientId, {
        foreground: false,
        state: 'hidden',
        epoch: 1,
      })
    );
  }
  await client.attach(
    primaryClientId,
    lifecycleFor(primaryClientId, {
      foreground: startupLifecycle.foreground,
      state: startupLifecycle.state,
      epoch,
    })
  );
  agentStatus.textContent = `Connected / ${shortId(primaryClientId)}`;
  const diagnostics = await client.diagnostics();
  runtimeGlobals.__wllmCoordinatorDiagnostics = diagnostics;
  runtimeGlobals.__wllmCoordinatorDiagnosticsProvider = () =>
    client.diagnostics();
  recordBurstPolicy(
    diagnostics.runtime.gpuBurstPolicy,
    'initial',
    primaryClientId
  );
  addTrace(
    'dispatch',
    'Coordinator ready',
    diagnostics.recovery.recovered > 0
      ? `recovered ${diagnostics.recovery.recovered} L3 node(s)`
      : `L3 scan: ${diagnostics.recovery.scanned}, no recoverable node`
  );
  addTrace('dispatch', 'Coordinator worker', coordinatorWorkerNameValue);
  void inferenceBridge
    .ensure()
    .catch((error: unknown) => {
      console.error('[wllm] inference bridge failed', error);
    });
  submitButton.addEventListener('click', () => {
    void submitCurrentPrompt().catch((error: unknown) =>
      reportFailure('Submit', error)
    );
  });
  burstButton.addEventListener('click', () => {
    void submitBurst().catch((error: unknown) => reportFailure('Burst', error));
  });
  newTabButton.addEventListener('click', () => {
    window.open(newAgentTabUrl(), '_blank', 'noopener,noreferrer');
  });
  clearFinishedButton.addEventListener('click', clearFinishedRequests);
  frameMarkButton.addEventListener('click', markFrame);
  frameResetButton.addEventListener('click', resetFrameStats);
  requestAnimationFrame(animationLoop);
  for (const button of modeButtons) {
    button.addEventListener('click', () => {
      manualBackground = button.dataset.mode === 'background';
      renderLifecycleButtons();
      void applyLifecycle();
    });
  }
  window.addEventListener('beforeunload', () => {
    framePressureMonitor.stop();
    for (const clientId of [primaryClientId, ...backgroundClientIds]) {
      void client.detach(clientId);
    }
    inferenceBridge.release();
  });
}

async function submitCurrentPrompt(): Promise<void> {
  const prompt = promptElement.value.trim();
  if (prompt.length === 0) {
    promptElement.focus();
    return;
  }
  await submitRequest({
    prompt,
    maxNewTokens: clamp(
      Number(maxTokensElement.value) || 64,
      4,
      512
    ),
    seed: readSeed(0),
    foreground: !manualBackground && document.visibilityState === 'visible',
    clientId: primaryClientId,
    sessionId: freshSessionPerRequest
      ? `${sessionId}-r${(freshSessionSequence += 1)}`
      : sessionId,
    parentNodeId: freshSessionPerRequest ? null : parentNodeId,
    epoch,
  });
}

async function submitBurst(): Promise<void> {
  const seeds = [
    'The lantern went dark and',
    'At the edge of the map,',
    'The small robot counted every',
    'Nobody expected the river to',
  ];
  for (let index = 0; index < seeds.length; index += 1) {
    const clientId = backgroundClientIds[index];
    if (!clientId) {
      continue;
    }
    await submitRequest({
      prompt: `${seeds[index]} `,
      maxNewTokens: 32 + index * 8,
      seed: readSeed(index + 1),
      foreground: false,
      clientId,
      sessionId: `burst-${index + 1}`,
      parentNodeId: null,
      epoch: 1,
    });
  }
}

async function submitRequest(input: {
  prompt: string;
  maxNewTokens: number;
  seed: number;
  foreground: boolean;
  clientId: string;
  sessionId: string;
  parentNodeId: string | null;
  epoch: number;
}): Promise<void> {
  const requestId = makeId();
  const promptTokens = estimateTokens(input.prompt);
  const payload: Omit<RequestRecord, 'lifecycle'> = {
    requestId,
    session: {
      clientId: input.clientId,
      sessionId: input.sessionId,
    },
    parentNodeId: input.parentNodeId,
    dependencyRequestIds: [],
    promptText: input.prompt,
    promptTokens,
    maxNewTokens: input.maxNewTokens,
    seed: input.seed,
    createdAt: Date.now(),
    enqueuedAt: Date.now(),
    waitBudgetMs: 0,
    serviceEstimateMs: 0,
    serviceMs: 0,
    stage: 'prefill',
    prefillProcessed: 0,
    decodeGenerated: 0,
    tokenAccounting: 'native-token',
    nativeRequestId: null,
    slotId: null,
    outputChunks: [],
  };
  const view = createRequestView({
    requestId,
    sessionId: input.sessionId,
    prompt: input.prompt,
    maxNewTokens: input.maxNewTokens,
    foreground: input.foreground,
  });
  activeViews.set(requestId, view);
  addTrace('dispatch', `Queued ${shortId(requestId)}`, input.foreground ? 'foreground' : 'background');

  const result = await client.submit(
    payload,
    lifecycleFor(input.clientId, {
      foreground: input.foreground,
      state: input.foreground ? 'active' : 'hidden',
      epoch: input.epoch,
    })
  );
  if (!result.accepted) {
    updateStatus(view, 'failed');
    addTrace('complete', `Rejected ${shortId(requestId)}`, result.reason);
    return;
  }

  try {
    await client.pump(1);
  } catch (error) {
    updateStatus(view, 'failed');
    addTrace(
      'complete',
      `Failed ${shortId(requestId)}`,
      error instanceof Error ? error.message : String(error)
    );
  }
}

async function applyLifecycle(
  observation?: VisibilityObservation
): Promise<void> {
  epoch += 1;
  const current = observation ?? lifecycleObserver.current('manual');
  const lifecycle = lifecycleFor(primaryClientId, {
    foreground: current.foreground,
    state: current.state,
    epoch,
  });
  agentStatus.textContent = current.foreground
    ? `Foreground / ${shortId(primaryClientId)}`
    : `Background ${current.state} / ${shortId(primaryClientId)}`;
  addTrace(
    'dispatch',
    current.foreground ? 'Attention foreground' : 'Attention background',
    `epoch ${epoch} / ${current.state} / ${current.reason}`
  );
  await client.setLifecycle(lifecycle);
}

function handleRuntimeEvent(event: NativeRuntimeEvent): void {
  switch (event.phase) {
    case 'download': {
      const total = event.total ?? 0;
      const loaded = event.loaded ?? 0;
      const percent = total > 0 ? Math.round((loaded / total) * 100) : 0;
      runtimeStatus.textContent = `Downloading ${percent}%`;
      setProgress(percent);
      break;
    }
    case 'load':
      runtimeStatus.textContent = 'Loading model';
      setProgress(8);
      break;
    case 'ready':
      runtimeStatus.textContent = 'Ready';
      setProgress(100);
      break;
    case 'error':
      runtimeStatus.textContent = event.message ?? 'Runtime error';
      setProgress(0);
      break;
  }
}

function applyResidencyEvent(
  event: Extract<CoordinatorEvent, { type: 'residency' }>
): void {
  const { totals } = event;
  stateTiers.textContent = `${totals.l1Tokens} tok / ${totals.l2Tokens} tok / ${formatBytes(
    totals.l3Bytes
  )}`;
  const previous = residencyByNode.get(event.nodeId);
  residencyByNode.set(event.nodeId, {
    level: event.level,
    slotId: event.slotId,
  });
  if (previous?.level === event.level && previous.slotId === event.slotId) {
    return;
  }
  const parked =
    event.bufferBytes > 0
      ? ` / ${formatBytes(event.bufferBytes)} in wasm` +
        (event.hostBytes > 0
          ? `, ${formatBytes(event.hostBytes)} host copy`
          : ', no host copy')
      : '';
  addTrace(
    'dispatch',
    `${shortId(event.nodeId)} -> ${event.level}`,
    `slot ${event.slotId ?? '-'} / ${event.tokenCount} tok${parked}`
  );
}

function formatBytes(bytes: number): string {
  if (bytes <= 0) {
    return '0 B';
  }
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KiB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
}

function handleCoordinatorEvent(event: CoordinatorEvent): void {
  if (event.type === 'runtime') {
    handleRuntimeEvent(event);
    return;
  }
  if (event.type === 'burst') {
    recordBurstPolicy(event.policy, event.reason, event.clientId);
    return;
  }
  if (event.type === 'residency') {
    applyResidencyEvent(event);
    return;
  }
  if (event.type === 'fenced') {
    const view = activeViews.get(event.requestId);
    if (view) {
      updateStatus(view, 'cancelled');
    }
    return;
  }
  const view = activeViews.get(event.requestId);
  if (!view) {
    return;
  }

  if (event.type === 'token') {
    if (view.startedAt === null) {
      view.startedAt = performance.now();
    }
    view.firstTokenAt ??= performance.now();
    view.decodeGenerated = Math.max(
      view.decodeGenerated,
      event.decodeGenerated
    );
    view.output += event.text;
    view.outputElement.textContent = view.output;
    updateStatus(view, 'running');
    return;
  }

  if (event.type === 'unit') {
    const progress = event.progress;
    const budget = event.budget;
    if (view.startedAt === null) {
      view.startedAt = performance.now();
    }
    view.units += 1;
    traceUnits += 1;
    traceCounter.textContent = `${traceUnits} units`;
    view.decodeGenerated = progress.decodeGenerated;
    view.tokenAccounting = progress.tokenAccounting;
    if (progress.counters) {
      view.counters = { ...progress.counters };
    }
    view.prefillProcessed = progress.prefillProcessed;
    view.prefillTotal = progress.prefillTotal;
    view.prefillCached = progress.prefillCached ?? 0;
    if (view.initialPrefillCached === null) {
      view.initialPrefillCached = progress.prefillCached ?? 0;
      view.item.dataset.initialCached = String(view.initialPrefillCached);
    }
    view.slotId = progress.slotId;
    if (progress.boundedStops !== undefined) {
      view.item.dataset.boundedStops = String(progress.boundedStops);
    }
    view.item.dataset.decodeBudget = String(budget.maxDecodeTokens);
    view.item.dataset.prefillBudget = String(budget.maxPrefillTokens);
    if (progress.unitStop !== undefined) {
      view.item.dataset.unitStop = progress.unitStop;
      // the final unit of a request completes; keep the last real boundary
      // so the UI and the smoke harness can tell which budget closed units
      if (progress.unitStop !== 'complete') {
        view.item.dataset.boundaryStop = progress.unitStop;
      }
    }
    if (progress.outputChunks.length > 0) {
      view.firstTokenAt ??= performance.now();
      view.output += progress.outputChunks.join('');
      view.outputElement.textContent = view.output;
    }
    updateStatus(view, progress.done ? 'complete' : 'running');
    addTrace(
      'unit',
      `${shortId(event.requestId)} unit ${view.units}`,
      `${progress.stage} / slot ${progress.slotId ?? '-'} / ${Math.round(
        progress.elapsedMs
      )} ms${
        progress.prefillCached
          ? ` / ${progress.prefillCached} tok from cache`
          : ''
      }`
      + (progress.counters ? ` / +${progress.counters.predictedN} tok` : '')
      + ` / <=${budget.maxDecodeTokens} tok`
      + (progress.unitStop ? ` / ${progress.unitStop}` : '')
    );
    return;
  }

  if (event.type === 'complete') {
    view.output = event.output;
    view.outputElement.textContent = event.output;
    view.finishedAt = performance.now();
    updateStatus(view, 'complete');
    addTrace('complete', `Completed ${shortId(event.requestId)}`, `${view.units} units`);
  }
  if (event.type === 'failed') {
    view.finishedAt = performance.now();
    view.item.dataset.error = event.error;
    view.outputElement.textContent = view.output
      ? `${view.output}\n\n[error] ${event.error}`
      : `[error] ${event.error}`;
    updateStatus(view, 'failed');
    addTrace('error', `Failed ${shortId(event.requestId)}`, event.error);
  }
}

function createRequestView(input: {
  requestId: string;
  sessionId: string;
  prompt: string;
  maxNewTokens: number;
  foreground: boolean;
}): RequestView {
  const item = document.createElement('article');
  item.className = 'request-item';
  item.dataset.state = 'queued';
  item.dataset.requestId = input.requestId;
  item.innerHTML = `
    <div class="request-header">
      <div>
        <span class="request-id"></span>
        <span class="request-sub"></span>
      </div>
      <span class="request-state">queued</span>
    </div>
    <p class="request-prompt"></p>
    <pre class="request-output"></pre>
    <div class="request-footer">
      <span class="request-metrics">waiting for dispatch</span>
      <button class="text-button" type="button">Cancel</button>
    </div>
  `;
  const idElement = item.querySelector<HTMLElement>('.request-id');
  const subElement = item.querySelector<HTMLElement>('.request-sub');
  const promptText = item.querySelector<HTMLElement>('.request-prompt');
  const outputElement = item.querySelector<HTMLElement>('.request-output');
  const metrics = item.querySelector<HTMLElement>('.request-metrics');
  const state = item.querySelector<HTMLElement>('.request-state');
  const cancelButton = item.querySelector<HTMLButtonElement>('.text-button');
  if (
    !idElement ||
    !subElement ||
    !promptText ||
    !outputElement ||
    !metrics ||
    !state ||
    !cancelButton
  ) {
    throw new Error('failed to construct request view');
  }
  idElement.textContent = `${shortId(input.requestId)} / ${input.foreground ? 'FG' : 'BG'}`;
  subElement.textContent = `session ${shortId(input.sessionId)} / max ${input.maxNewTokens}`;
  promptText.textContent = input.prompt;
  outputElement.textContent = '';
  requestsElement.prepend(item);
  emptyState.classList.remove('visible');

  const view: RequestView = {
    requestId: input.requestId,
    sessionId: input.sessionId,
    prompt: input.prompt,
    maxNewTokens: input.maxNewTokens,
    foreground: input.foreground,
    status: 'queued',
    output: '',
    units: 0,
    decodeGenerated: 0,
    tokenAccounting: 'native-token',
    prefillProcessed: 0,
    prefillTotal: 0,
    prefillCached: 0,
    initialPrefillCached: null,
    slotId: null,
    submittedAt: performance.now(),
    startedAt: null,
    firstTokenAt: null,
    finishedAt: null,
    item,
    state,
    outputElement,
    metrics,
    cancelButton,
  };
  cancelButton.addEventListener('click', () => {
    void client.cancel(input.requestId).then((cancelled) => {
      if (cancelled) {
        updateStatus(view, 'cancelled');
      }
    });
  });
  updateQueueCount();
  return view;
}

function updateStatus(view: RequestView, status: RequestStatus): void {
  if (view.status === status && view.status !== 'running') {
    return;
  }
  view.status = status;
  view.item.dataset.state = status;
  view.state.textContent = status;
  const terminal =
    status === 'complete' || status === 'cancelled' || status === 'failed';
  view.cancelButton.disabled = terminal;
  view.cancelButton.hidden = terminal;
  const metrics = formatMetrics(view);
  view.metrics.textContent = metrics;
  updateQueueCount();
}

function formatMetrics(view: RequestView): string {
  const elapsed =
    (view.finishedAt ?? performance.now()) - view.submittedAt;
  const ttft =
    view.firstTokenAt === null ? null : view.firstTokenAt - view.submittedAt;
  const tokensPerSecond =
    elapsed > 0 && view.decodeGenerated > 0
      ? (view.decodeGenerated / elapsed) * 1_000
      : 0;
  return [
    `prefill ${view.prefillProcessed}/${view.prefillTotal || '?'}`,
    view.prefillCached > 0 ? `cached ${view.prefillCached}` : null,
    `decode ${view.decodeGenerated}/${view.maxNewTokens}`,
    view.tokenAccounting === 'native-token'
      ? 'tokens llama.cpp'
      : 'tokens est. per chunk',
    `slot ${view.slotId ?? '-'}`,
    `units ${view.units}`,
    ttft === null ? null : `TTFT ${Math.round(ttft)} ms`,
    `e2e ${tokensPerSecond.toFixed(1)} tok/s`,
  ]
    .filter((value): value is string => value !== null)
    .join(' / ');
}

function reportFailure(label: string, error: unknown): void {
  const message = error instanceof Error ? error.message : String(error);
  console.error(`[wllm] ${label} failed`, error);
  addTrace('complete', `${label} failed`, message);
}

function addTrace(kind: string, title: string, meta: string | undefined): void {
  const item = document.createElement('li');
  item.className = 'trace-item';
  item.dataset.kind = kind;
  item.innerHTML = `
    <span class="trace-dot"></span>
    <span>
      <span class="trace-title"></span>
      <span class="trace-meta"></span>
    </span>
    <time class="trace-time"></time>
  `;
  const titleElement = item.querySelector<HTMLElement>('.trace-title');
  const metaElement = item.querySelector<HTMLElement>('.trace-meta');
  const timeElement = item.querySelector<HTMLElement>('.trace-time');
  if (titleElement) {
    titleElement.textContent = title;
  }
  if (metaElement) {
    metaElement.textContent = meta ?? '';
  }
  if (timeElement) {
    timeElement.textContent = new Date().toLocaleTimeString([], {
      hour12: false,
      minute: '2-digit',
      second: '2-digit',
    });
  }
  trace.prepend(item);
  while (trace.children.length > 80) {
    trace.lastElementChild?.remove();
  }
}

function clearFinishedRequests(): void {
  for (const [requestId, view] of activeViews) {
    if (
      view.status === 'complete' ||
      view.status === 'cancelled' ||
      view.status === 'failed'
    ) {
      view.item.remove();
      activeViews.delete(requestId);
    }
  }
  emptyState.classList.toggle('visible', activeViews.size === 0);
  updateQueueCount();
}

function updateQueueCount(): void {
  let count = 0;
  for (const view of activeViews.values()) {
    if (view.status === 'queued' || view.status === 'running') {
      count += 1;
    }
  }
  queueCount.textContent = String(count);
}

function renderLifecycleButtons(): void {
  for (const button of modeButtons) {
    const active =
      button.dataset.mode === (manualBackground ? 'background' : 'foreground');
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', String(active));
  }
}

function setProgress(percent: number): void {
  const bounded = clamp(percent, 0, 100);
  const bar = modelProgress.querySelector<HTMLElement>('span');
  if (bar) {
    bar.style.width = `${bounded}%`;
  }
  modelProgress.setAttribute('aria-valuenow', String(bounded));
  modelProgressText.textContent = `${bounded}%`;
}

function lifecycleFor(
  clientId: string,
  input: Pick<LifecycleRecord, 'foreground' | 'state' | 'epoch'>
): LifecycleRecord {
  return {
    clientId,
    epoch: input.epoch,
    state: input.state,
    foreground: input.foreground,
    changedAt: Date.now(),
  };
}

function estimateTokens(value: string): number[] {
  const estimated = clamp(
    Math.ceil(new TextEncoder().encode(value).byteLength / 4),
    1,
    4_096
  );
  return Array.from({ length: estimated }, (_, index) => index % 32_000);
}

function requiredElement<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) {
    throw new Error(`missing element #${id}`);
  }
  return element as T;
}

// ---------------------------------------------------------------------------
// Foreground rendering monitor
//
// The tab keeps an animation loop running while requests are served, which is
// the condition the execution controller is built for. Frame intervals and a
// marker-to-second-frame interval are collected here so a run shows whether
// bounded units leave the frame loop alone.
// ---------------------------------------------------------------------------

const FRAME_SAMPLE_CAP = 900;
const frameIntervals: number[] = [];
let lastAnimationAt = performance.now();
let lastFramePaintAt = 0;
let sweepPhase = 0;
let markerAt: number | null = null;
let markerFrames = 0;
let markerFrameMs: number | null = null;

function animationLoop(now: number): void {
  const delta = now - lastAnimationAt;
  lastAnimationAt = now;
  if (delta > 0 && delta < 2_000) {
    frameIntervals.push(delta);
    if (frameIntervals.length > FRAME_SAMPLE_CAP) {
      frameIntervals.splice(0, frameIntervals.length - FRAME_SAMPLE_CAP);
    }
  }

  // a real per-frame paint, so a blocked frame loop is observable
  sweepPhase = (sweepPhase + delta / 1_400) % 1;
  frameSweep.style.transform = `translateX(${(sweepPhase * 460).toFixed(1)}%)`;

  if (markerAt !== null) {
    markerFrames += 1;
    if (markerFrames >= 2) {
      markerFrameMs = now - markerAt;
      markerAt = null;
    }
  }

  if (now - lastFramePaintAt >= 250) {
    lastFramePaintAt = now;
    renderFrameMetrics();
  }
  requestAnimationFrame(animationLoop);
}

interface FrameStats {
  frames: number;
  p95Ms: number;
  meanMs: number;
  maxMs: number;
  over50MsShare: number;
  markerFrameMs: number | null;
}

function frameStats(): FrameStats {
  if (frameIntervals.length === 0) {
      return {
        frames: 0,
        p95Ms: 0,
        meanMs: 0,
        maxMs: 0,
        over50MsShare: 0,
        markerFrameMs,
      };
  }
  const sorted = [...frameIntervals].sort((a, b) => a - b);
  const p95Index = Math.min(
    sorted.length - 1,
    Math.floor(sorted.length * 0.95)
  );
  const p95Ms = sorted[p95Index] ?? 0;
  const total = frameIntervals.reduce((sum, value) => sum + value, 0);
  const over = frameIntervals.filter((value) => value > 50).length;
  return {
    frames: frameIntervals.length,
    p95Ms,
    meanMs: total / frameIntervals.length,
    maxMs: sorted[sorted.length - 1] ?? 0,
    over50MsShare: (over / frameIntervals.length) * 100,
    markerFrameMs,
  };
}

function renderFrameMetrics(): void {
  const stats = frameStats();
  if (stats.frames === 0) {
    return;
  }
  const marker =
    stats.markerFrameMs === null
      ? 'n/a'
      : `${stats.markerFrameMs.toFixed(1)} ms`;
  frameMetrics.textContent = `frame P95 ${stats.p95Ms.toFixed(1)} ms / max ${stats.maxMs.toFixed(1)} ms / ${stats.over50MsShare.toFixed(1)}% over 50 ms / marker-frame ${marker}`;
  frameStatus.textContent = `P95 ${stats.p95Ms.toFixed(0)} ms`;
}

// a marker is scripted, not a real input event: it is a responsiveness proxy
function markFrame(): void {
  markerAt = performance.now();
  markerFrames = 0;
  markerFrameMs = null;
}

function resetFrameStats(): void {
  frameIntervals.length = 0;
  markerAt = null;
  markerFrames = 0;
  markerFrameMs = null;
  frameMetrics.textContent = 'Waiting for animation frames';
}

// automation hook: the smoke harness reads the same numbers the panel shows
(window as unknown as Record<string, unknown>).__wllmFrameStats = frameStats;

// automation hook: a harness that reuses one browser profile across tabs
// detaches the client explicitly instead of relying on `beforeunload`, which
// `page.close()` does not run by default.
(window as unknown as Record<string, unknown>).__wllmDetachClient =
  async (): Promise<void> => {
    framePressureMonitor.stop();
    await client.detach(primaryClientId);
  };

function shortId(value: string): string {
  return value.replaceAll('-', '').slice(0, 8);
}

function readInferenceOptions(): InferenceBridgeOptions {
  const params = new URLSearchParams(window.location.search);
  const options: InferenceBridgeOptions = {};
  const modelUrl = params.get('modelUrl');
  if (modelUrl) {
    options.modelUrl = modelUrl;
  }
  for (const key of [
    'nCtx',
    'nBatch',
    'nParallel',
    'nThreads',
    'l1TokenCap',
    'l2TokenCap',
    'l3ByteCap',
    'repairBudget',
    'webgpuCommandSubmitBatchSize',
    'webgpuMaxInflightBatches',
  ] as const) {
    const parsed = Number(params.get(key));
    if (Number.isFinite(parsed) && parsed > 0) {
      options[key] = Math.floor(parsed);
    }
  }
  // 0 is a valid value here: it forces the CPU backend for the control run.
  const gpuLayersRaw = params.get('nGpuLayers');
  const gpuLayers = Number(gpuLayersRaw);
  if (
    gpuLayersRaw !== null &&
    Number.isFinite(gpuLayers) &&
    gpuLayers >= 0
  ) {
    options.nGpuLayers = Math.floor(gpuLayers);
  }
  const execution: Partial<ExecutionConfig> = {};
  const unsliced =
    params.get('unsliced') === '1' || params.get('slice') === '0';
  if (unsliced) {
    Object.assign(execution, {
      maxPrefillTokens: 8_192,
      baseDecodeTokens: 2_048,
      foregroundDecodeTokens: 2_048,
      backgroundDecodeTokens: 2_048,
      prefillMsPerToken: 0.1,
      maxUnitWallTimeMs: 30_000,
      adaptiveUnits: false,
      nativeTimeBudgetEnabled: false,
    } satisfies Partial<ExecutionConfig>);
  }
  for (const key of [
    'maxPrefillTokens',
    'baseDecodeTokens',
    'foregroundDecodeTokens',
    'backgroundDecodeTokens',
    'prefillMsPerToken',
    'maxUnitWallTimeMs',
    'unitWallTimeGraceMs',
    'targetUnitWallTimeMs',
    'minDecodeTokens',
    'maxDecodeTokens',
  ] as const) {
    const parsed = Number(params.get(key));
    if (Number.isFinite(parsed) && parsed > 0) {
      execution[key] = parsed;
    }
  }
  const adaptive = params.get('adaptiveUnits');
  if (adaptive !== null) {
    execution.adaptiveUnits = !['0', 'false', 'off', 'no'].includes(
      adaptive.toLowerCase()
    );
  }
  // explicit switch so a harness that reuses one coordinator across shapes can
  // turn the cooperative time budget back on after an unsliced page turned it
  // off; without it the unsliced setting would leak into the next shape
  const nativeTimeBudget = params.get('nativeTimeBudget');
  if (nativeTimeBudget !== null) {
    execution.nativeTimeBudgetEnabled = ![
      '0',
      'false',
      'off',
      'no',
    ].includes(nativeTimeBudget.toLowerCase());
  }
  if (Object.keys(execution).length > 0) {
    options.execution = execution;
  }
  const parkCompleted = params.get('parkCompleted');
  if (parkCompleted !== null) {
    options.parkCompleted = !['0', 'false', 'off', 'no'].includes(
      parkCompleted.toLowerCase()
    );
  }
  const completedTier = readCompletedTier(params.get('completedTier'));
  if (completedTier !== undefined) {
    options.completedTier = completedTier;
  }
  const renderAware = params.get('renderAware');
  if (renderAware !== null) {
    options.renderAware = !['0', 'false', 'off', 'no'].includes(
      renderAware.toLowerCase()
    );
  }
  const renderGate = params.get('renderGate');
  if (renderGate !== null) {
    options.renderGate = !['0', 'false', 'off', 'no'].includes(
      renderGate.toLowerCase()
    );
  }
  const executionMode = readExecutionMode(params.get('executionMode'));
  if (executionMode !== undefined) {
    options.executionMode = executionMode;
  }
  for (const key of [
    'leaseMaxBudgetMs',
    'leaseWorkConservingQuantumMs',
    'leaseMaxPauseMs',
    'leasePausePerDebtMs',
    'leaseMinPressureDecodeTokens',
    'leaseMaxCommandBatchSize',
    'leaseInflightCap',
    'leaseMildPressureFramePeriods',
    'leaseSeverePressureFramePeriods',
    'leaseQueueWaitToleranceMs',
  ] as const) {
    const raw = params.get(key);
    if (raw === null || raw.trim() === '') {
      continue;
    }
    const parsed = Number(raw);
    if (Number.isFinite(parsed) && parsed >= 0) {
      options[key] = parsed;
    }
  }
  return options;
}

function readExecutionMode(
  value: string | null
): InferenceBridgeOptions['executionMode'] | undefined {
  switch (value?.trim().toLowerCase()) {
    case 'bounded':
      return 'bounded';
    case 'unsliced':
      return 'unsliced';
    case 'deadline':
    case 'lease':
      return 'deadline';
    default:
      return undefined;
  }
}

function readCompletedTier(
  value: string | null
): InferenceBridgeOptions['completedTier'] | undefined {
  switch (value?.trim().toLowerCase()) {
    case '1':
    case 'l1':
      return 'L1';
    case '2':
    case 'l2':
      return 'L2';
    case '3':
    case 'l3':
      return 'L3';
    case 'rebuild':
    case 'cold':
    case 'none':
      return 'rebuild';
    default:
      return undefined;
  }
}

function applyCoordinatorWorkerOptions(
  url: URL,
  options: InferenceBridgeOptions
): void {
  if (options.modelUrl) {
    url.searchParams.set(
      'modelUrl',
      new URL(options.modelUrl, window.location.href).href
    );
  }
  for (const key of [
    'nCtx',
    'nBatch',
    'nParallel',
    'nThreads',
    'nGpuLayers',
    'l1TokenCap',
    'l2TokenCap',
    'l3ByteCap',
    'repairBudget',
    'webgpuCommandSubmitBatchSize',
    'webgpuMaxInflightBatches',
  ] as const) {
    const value = options[key];
    if (value !== undefined) {
      url.searchParams.set(key, String(value));
    }
  }
  if (options.parkCompleted !== undefined) {
    url.searchParams.set('parkCompleted', options.parkCompleted ? '1' : '0');
  }
  if (options.completedTier !== undefined) {
    url.searchParams.set('completedTier', options.completedTier);
  }
  if (options.renderAware !== undefined) {
    url.searchParams.set('renderAware', options.renderAware ? '1' : '0');
  }
  if (options.renderGate !== undefined) {
    url.searchParams.set('renderGate', options.renderGate ? '1' : '0');
  }
  if (options.executionMode !== undefined) {
    url.searchParams.set('executionMode', options.executionMode);
  }
  for (const key of [
    'leaseMaxBudgetMs',
    'leaseWorkConservingQuantumMs',
    'leaseMaxPauseMs',
    'leasePausePerDebtMs',
    'leaseMinPressureDecodeTokens',
    'leaseMaxCommandBatchSize',
    'leaseInflightCap',
    'leaseMildPressureFramePeriods',
    'leaseSeverePressureFramePeriods',
    'leaseQueueWaitToleranceMs',
  ] as const) {
    const value = options[key];
    if (value !== undefined) {
      url.searchParams.set(key, String(value));
    }
  }

  for (const key of [
    'maxPrefillTokens',
    'baseDecodeTokens',
    'foregroundDecodeTokens',
    'backgroundDecodeTokens',
    'prefillMsPerToken',
    'maxUnitWallTimeMs',
    'unitWallTimeGraceMs',
    'targetUnitWallTimeMs',
    'minDecodeTokens',
    'maxDecodeTokens',
  ] as const) {
    const value = options.execution?.[key];
    if (value !== undefined) {
      url.searchParams.set(key, String(value));
    }
  }
  if (options.execution?.adaptiveUnits !== undefined) {
    url.searchParams.set(
      'adaptiveUnits',
      options.execution.adaptiveUnits ? '1' : '0'
    );
  }
}

function newAgentTabUrl(): string {
  const url = new URL(window.location.href);
  for (const key of ['clientId', 'sessionId', 'nodeId', 'parentNodeId']) {
    url.searchParams.delete(key);
  }
  return url.href;
}

function makeId(): string {
  return globalThis.crypto?.randomUUID?.() ?? Math.random().toString(36).slice(2);
}

function readBooleanParam(value: string | null, fallback: boolean): boolean {
  if (value === null) {
    return fallback;
  }
  return !['0', 'false', 'off', 'no'].includes(value.trim().toLowerCase());
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function readSeed(offset: number): number {
  const parsed = Number(seedElement.value);
  const base = Number.isFinite(parsed) && parsed >= 0 ? Math.floor(parsed) : 0;
  return (base + offset) % 0x7fffffff;
}
