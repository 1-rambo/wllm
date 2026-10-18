/*
 * Evaluation-only browser adapter.
 *
 * The HTTP server injects this file into the real ArtifexWebGame and
 * Holtwick Tavern pages. It never imports impl/src. Generation goes through
 * the public engine-page global over a same-origin iframe, while all tasks
 * are submitted through the application UI that the experiment is measuring.
 */

const params = new URLSearchParams(location.search);
const ENGINE_GLOBAL = "__wllmEnginePage";
const MAX_PROGRESS_EVENTS = 4096;
const DEFAULT_TIMEOUT_MS = 10 * 60_000;
const LIFECYCLE_PREFIX = "wllm-e2e-lifecycle:";

const config = {
  app: params.get("app") ?? "unknown",
  arm: params.get("arm") ?? "ours",
  engine:
    params.get("engine") ??
    (params.get("arm") === "wllama-per-tab" ? "wllama" : "wllm"),
  enginePage:
    params.get("enginePage") ?? "/engines/wllm.html",
  coordinatorWorkerUrl: params.get("coordinatorWorkerUrl"),
  sessionId:
    params.get("sessionId") ??
    `session-${Date.now()}-${Math.random().toString(36).slice(2)}`,
  modelUrl:
    params.get("modelUrl") ??
    "/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
  modelLoadMode: params.get("modelLoadMode") === "direct" ? "direct" : "cache",
  promptFormat: params.get("promptFormat") ?? "llama3-chat",
  priorityMode: params.get("priorityMode") ?? "enabled",
  initialPriority: params.get("initialPriority") ?? "normal",
  executionMode: params.get("executionMode") ?? "deadline",
  completedTier: params.get("completedTier") ?? "L2",
  parkCompleted: params.get("parkCompleted") ?? "1",
  preemptOnActivation: params.get("preemptOnActivation") === "1",
  queueWaitTimeoutMs: numberParam("queueWaitTimeoutMs", 300_000),
  webgpuCommandSubmitBatchSize: numberParam(
    "webgpuCommandSubmitBatchSize",
    32
  ),
  webgpuMaxInflightBatches: numberParam("webgpuMaxInflightBatches", 4),
  leaseWorkConservingBarrier:
    params.get("leaseWorkConservingBarrier") === "1",
  leaseBackgroundBarrier: params.get("leaseBackgroundBarrier") === "1",
  leaseStaleSignalShape: params.get("leaseStaleSignalShape"),
  leaseBatchSize: numberParam("leaseBatchSize", 2),
  leaseWorkConservingQuantumMs: numberParam(
    "leaseWorkConservingQuantumMs",
    1000
  ),
  schedulerForegroundBurstUnits: numberParam(
    "schedulerForegroundBurstUnits",
    2
  ),
  schedulerResidentBurstUnits: numberParam(
    "schedulerResidentBurstUnits",
    1
  ),
  schedulerRetentionWindowMs: numberParam(
    "schedulerRetentionWindowMs",
    30_000
  ),
  batchHandoff: params.get("batchHandoff") === "1",
  handoffBatchSize: numberParam("handoffBatchSize", 2),
  expressHandoff: params.get("expressHandoff") === "1",
  handoffPrefillTokens: numberParam("handoffPrefillTokens", 64),
  nCtx: numberParam("nCtx", 2048),
  nBatch: numberParam("nBatch", 128),
  nParallel: numberParam("nParallel", 4),
  nThreads: numberParam("nThreads", 4),
  l1NodeCap: numberParam("l1NodeCap", 9),
  l1TokenCap: numberParam("l1TokenCap", 16384),
  l2TokenCap: numberParam("l2TokenCap", 16384),
  l3ByteCap: numberParam("l3ByteCap", 67108864),
  repairBudget: numberParam("repairBudget", 4),
  maxNewTokens: numberParam("maxNewTokens", 64),
  temperature: numberParam("temperature", 0),
  seed: numberParam("seed", 20261015),
  timeoutMs: numberParam("timeoutMs", DEFAULT_TIMEOUT_MS)
};

const progressEvents = [];
const priorityMarkers = [];
const lifecycleLog = [];
const errors = [];
let engineFrame = null;
let engineApi = null;
let unsubscribeProgress = null;
let readyPromise = null;
let ready = false;
let firstError = null;
let activeTask = null;

function numberParam(name, fallback) {
  const raw = params.get(name);
  if (raw === null || raw === "") return fallback;
  const value = Number(raw);
  return Number.isFinite(value) ? value : fallback;
}

function nowMs() {
  return performance.now();
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function describeError(error) {
  if (error instanceof Error) return error.message;
  return String(error);
}

function recordError(error) {
  const message = describeError(error);
  firstError = firstError ?? message;
  errors.push({ message, atMs: Date.now() });
  if (errors.length > 128) errors.shift();
  return new Error(message);
}

async function waitForConditionQuiet(predicate, timeoutMs) {
  const started = nowMs();
  while (nowMs() - started < timeoutMs) {
    try {
      const value = await predicate();
      if (value) return value;
    } catch {
      // Setup retries must not turn an intermediate miss into a task error.
    }
    await delay(50);
  }
  throw new Error("condition did not become true before timeout");
}

async function waitForCondition(predicate, timeoutMs, description) {
  const started = nowMs();
  while (nowMs() - started < timeoutMs) {
    try {
      const value = await predicate();
      if (value) return value;
    } catch {
      // The document may still be navigating or the application may still
      // be wiring its event handlers. Retry until the timeout.
    }
    await delay(50);
  }
  throw recordError(`${description} did not become true within ${timeoutMs} ms`);
}

function lifecycleStorageKey() {
  return `${LIFECYCLE_PREFIX}${config.sessionId}`;
}

function readLifecycleLog() {
  try {
    const raw = sessionStorage.getItem(lifecycleStorageKey());
    if (raw === null) return [];
    const value = JSON.parse(raw);
    return Array.isArray(value) ? value : [];
  } catch {
    return [];
  }
}

function persistLifecycleLog() {
  try {
    sessionStorage.setItem(lifecycleStorageKey(), JSON.stringify(lifecycleLog));
  } catch {
    // Lifecycle telemetry is best effort. A blocked storage API must not
    // stop the application task.
  }
}

function appendLifecycle(type, detail = {}) {
  const entry = {
    type,
    atMs: Date.now(),
    visibilityState: document.visibilityState,
    ...detail
  };
  lifecycleLog.push(entry);
  if (lifecycleLog.length > 256) lifecycleLog.shift();
  persistLifecycleLog();
  return entry;
}

function initializeLifecycleTracking() {
  try {
    const stored = readLifecycleLog();
    lifecycleLog.push(...stored);
  } catch {
    // Ignore storage failures.
  }
  const loadCount =
    lifecycleLog.filter((entry) => entry.type === "load").length + 1;
  appendLifecycle("load", { loadCount });
  appendLifecycle("visibility-initial", {
    visibilityState: document.visibilityState
  });
  document.addEventListener("visibilitychange", () => {
    appendLifecycle(
      document.visibilityState === "hidden" ? "hidden" : "visible"
    );
  });
  window.addEventListener("focus", () => appendLifecycle("focus"));
  window.addEventListener("blur", () => appendLifecycle("blur"));
  window.addEventListener("pagehide", (event) =>
    appendLifecycle("pagehide", { persisted: event.persisted === true })
  );
  window.addEventListener("pageshow", (event) =>
    appendLifecycle("pageshow", { persisted: event.persisted === true })
  );
  window.addEventListener("freeze", () => appendLifecycle("freeze"));
  window.addEventListener("resume", () => appendLifecycle("resume"));
}

function engineUrl() {
  const page = new URL(config.enginePage, location.origin);
  const query = new URLSearchParams();
  query.set("modelUrl", new URL(config.modelUrl, location.origin).toString());
  query.set("modelLoadMode", config.modelLoadMode);
  query.set("nCtx", String(config.nCtx));
  query.set("nBatch", String(config.nBatch));
  query.set("nParallel", String(config.nParallel));
  query.set("nThreads", String(config.nThreads));
  query.set("temperature", String(config.temperature));
  query.set("seed", String(config.seed));
  query.set("frameMonitor", "0");
  query.set("useCache", "1");
  query.set("cachePrompt", "1");
  query.set("priorityMode", config.priorityMode);
  query.set("initialPriority", config.initialPriority);
  query.set("executionMode", config.executionMode);
  query.set("completedTier", config.completedTier);
  query.set("parkCompleted", config.parkCompleted);
  query.set("preemptOnActivation", config.preemptOnActivation ? "1" : "0");
  query.set("queueWaitTimeoutMs", String(config.queueWaitTimeoutMs));
  query.set(
    "webgpuCommandSubmitBatchSize",
    String(config.webgpuCommandSubmitBatchSize)
  );
  query.set(
    "webgpuMaxInflightBatches",
    String(config.webgpuMaxInflightBatches)
  );
  query.set(
    "leaseWorkConservingBarrier",
    config.leaseWorkConservingBarrier ? "1" : "0"
  );
  query.set(
    "leaseBackgroundBarrier",
    config.leaseBackgroundBarrier ? "1" : "0"
  );
  if (
    config.leaseStaleSignalShape === "barrier" ||
    config.leaseStaleSignalShape === "bounded"
  ) {
    query.set("leaseStaleSignalShape", config.leaseStaleSignalShape);
  }
  query.set("leaseBatchSize", String(config.leaseBatchSize));
  query.set(
    "leaseWorkConservingQuantumMs",
    String(config.leaseWorkConservingQuantumMs)
  );
  query.set(
    "schedulerForegroundBurstUnits",
    String(config.schedulerForegroundBurstUnits)
  );
  query.set(
    "schedulerResidentBurstUnits",
    String(config.schedulerResidentBurstUnits)
  );
  query.set(
    "schedulerRetentionWindowMs",
    String(config.schedulerRetentionWindowMs)
  );
  query.set("batchHandoff", config.batchHandoff ? "1" : "0");
  query.set("leaseHandoffBatchSize", String(config.handoffBatchSize));
  query.set("expressHandoff", config.expressHandoff ? "1" : "0");
  query.set("handoffPrefillTokens", String(config.handoffPrefillTokens));
  query.set("l1NodeCap", String(config.l1NodeCap));
  query.set("l1TokenCap", String(config.l1TokenCap));
  query.set("l2TokenCap", String(config.l2TokenCap));
  query.set("l3ByteCap", String(config.l3ByteCap));
  query.set("repairBudget", String(config.repairBudget));
  if (config.coordinatorWorkerUrl) {
    query.set("coordinatorWorkerUrl", config.coordinatorWorkerUrl);
  }
  page.search = query.toString();
  return page.toString();
}

function createEngineFrame() {
  if (engineFrame !== null) return engineFrame;
  engineFrame = document.createElement("iframe");
  engineFrame.id = "wllm-e2e-engine";
  engineFrame.title = "wllm evaluation engine";
  engineFrame.setAttribute("aria-hidden", "true");
  engineFrame.style.width = "1px";
  engineFrame.style.height = "1px";
  engineFrame.style.opacity = "0";
  engineFrame.style.pointerEvents = "none";
  engineFrame.style.position = "fixed";
  engineFrame.style.left = "-10000px";
  engineFrame.style.top = "0";
  engineFrame.src = engineUrl();
  document.documentElement.appendChild(engineFrame);
  return engineFrame;
}

async function waitForEngine(timeoutMs = config.timeoutMs) {
  const frame = createEngineFrame();
  return waitForCondition(
    () => {
      const candidate = frame.contentWindow?.[ENGINE_GLOBAL];
      return candidate && typeof candidate.prewarm === "function"
        ? candidate
        : null;
    },
    timeoutMs,
    `engine page ${ENGINE_GLOBAL}`
  );
}

function subscribeToProgress(api) {
  if (unsubscribeProgress !== null || typeof api.subscribeProgress !== "function") {
    return;
  }
  unsubscribeProgress = api.subscribeProgress((event) => {
    const record = {
      sessionId: event.sessionId,
      requestId: event.requestId,
      type: event.type,
      state: event.state ?? null,
      text: event.text ?? "",
      decodeGenerated: event.decodeGenerated ?? 0,
      timestampMs: event.timestampMs ?? Date.now(),
      receivedAtMs: nowMs(),
      error: event.error ?? null
    };
    progressEvents.push(record);
    if (progressEvents.length > MAX_PROGRESS_EVENTS) {
      progressEvents.splice(0, progressEvents.length - MAX_PROGRESS_EVENTS);
    }
  });
}

async function initialize() {
  if (readyPromise !== null) return readyPromise;
  readyPromise = (async () => {
    initializeLifecycleTracking();
    await waitForCondition(
      () => document.querySelector("canvas") !== null,
      Math.min(config.timeoutMs, 30_000),
      `${config.app} visible canvas`
    );
    if (config.app === "tavern") {
      try {
        if (localStorage.getItem("holtwick-tavern:tts:v1") === null) {
          localStorage.setItem("holtwick-tavern:tts:v1", "0");
        }
      } catch {
        // TTS is optional and must not affect the measured task.
      }
    }
    const api = await waitForEngine();
    engineApi = api;
    subscribeToProgress(api);
    await api.prewarm();
    await api.createSession({
      sessionId: config.sessionId,
      pageId: `${config.app}-${config.sessionId}`
    });
    ready = true;
    window.__wllmE2EReady = true;
    return api;
  })().catch((error) => {
    firstError = describeError(error);
    ready = false;
    window.__wllmE2EReady = false;
    throw error;
  });
  return readyPromise;
}

async function requireEngine() {
  if (!ready) await initialize();
  if (engineApi === null) throw recordError("engine API unavailable");
  return engineApi;
}

function summarizeProgress(requestId, startedAtMs, finishedAtMs) {
  return progressEvents
    .filter((event) => event.requestId === requestId)
    .filter(
      (event) =>
        event.receivedAtMs >= startedAtMs &&
        event.receivedAtMs <= finishedAtMs + 100
    )
    .map((event) => ({ ...event }));
}

function numberOrNull(value) {
  return Number.isFinite(Number(value)) ? Number(value) : null;
}

function measuredPrefillMsPerToken(result) {
  const prefillMs = numberOrNull(result?.prefillMs);
  const prefillTokens = numberOrNull(result?.prefillTokens);
  if (prefillMs === null || prefillTokens === null || prefillTokens <= 0) {
    return null;
  }
  return prefillMs / prefillTokens;
}

async function generateInSession(sessionId, request = {}) {
  const api = await requireEngine();
  const requestId =
    request.requestId ??
    `req-${Date.now()}-${Math.random().toString(36).slice(2)}`;
  const promptText = String(request.promptText ?? "");
  const startedAtMs = nowMs();
  const startedWallMs = Date.now();
  const priorityMarker =
    priorityMarkers.length === 0
      ? null
      : { ...priorityMarkers[priorityMarkers.length - 1] };
  try {
    const result = await api.generate(sessionId, {
      requestId,
      promptText,
      maxNewTokens: request.maxNewTokens ?? config.maxNewTokens,
      temperature: request.temperature ?? config.temperature,
      seed: request.seed ?? config.seed
    });
    const finishedAtMs = nowMs();
    return {
      ...result,
      requestId,
      sessionId,
      startedAtMs,
      startedWallMs,
      finishedAtMs,
      finishedWallMs: Date.now(),
      elapsedMs: finishedAtMs - startedAtMs,
      measuredPrefillMsPerToken: measuredPrefillMsPerToken(result),
      priorityMarker,
      progress: summarizeProgress(requestId, startedAtMs, finishedAtMs),
      error: null
    };
  } catch (error) {
    const finishedAtMs = nowMs();
    return {
      requestId,
      sessionId,
      text: "",
      startedAtMs,
      startedWallMs,
      finishedAtMs,
      finishedWallMs: Date.now(),
      elapsedMs: finishedAtMs - startedAtMs,
      ttftMs: null,
      decodeMs: 0,
      decodeTokens: 0,
      prefillTokens: null,
      cacheTokens: null,
      prefillProcessedTokens: null,
      prefillMs: null,
      measuredPrefillMsPerToken: null,
      units: null,
      stopReason: "failed",
      priorityMarker,
      progress: summarizeProgress(requestId, startedAtMs, finishedAtMs),
      error: describeError(error)
    };
  }
}

async function generate(request = {}) {
  return generateInSession(config.sessionId, request);
}

function normalizeMessages(messages) {
  if (!Array.isArray(messages)) return [];
  return messages
    .map((message) => ({
      role: String(message?.role ?? "user"),
      content: String(message?.content ?? "")
    }))
    .filter((message) => message.content.trim() !== "");
}

function formatMessages(messages, format) {
  const normalized = normalizeMessages(messages);
  if (format === "qwen-chatml") {
    const parts = [];
    for (const message of normalized) {
      parts.push(
        `<|im_start|>${message.role}\n${message.content}\n<|im_end|>\n`
      );
    }
    parts.push("<|im_start|>assistant\n");
    return parts.join("");
  }
  const parts = ["<|begin_of_text|>"];
  for (const message of normalized) {
    parts.push(
      `<|start_header_id|>${message.role}<|end_header_id|>\n\n${message.content}<|eot_id|>`
    );
  }
  parts.push("<|start_header_id|>assistant<|end_header_id|>\n\n");
  return parts.join("");
}

function formatTurnDelta(prompt, format) {
  const content = String(prompt ?? "");
  if (format === "qwen-chatml") {
    return `<|im_end|>\n<|im_start|>user\n${content}\n<|im_end|>\n<|im_start|>assistant\n`;
  }
  return `<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n${content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n`;
}

async function generateFromMessages(messages, request = {}) {
  const effectiveMessages =
    activeTask?.continuationMessages?.length > 0
      ? activeTask.continuationMessages
      : normalizeMessages(messages);
  const promptText = formatMessages(
    effectiveMessages,
    request.promptFormat ?? config.promptFormat
  );
  const turnDelta = activeTask?.turnDeltaPrompt;
  const effectivePromptText =
    typeof turnDelta === "string" && turnDelta.length > 0
      ? formatTurnDelta(turnDelta, request.promptFormat ?? config.promptFormat)
      : promptText;
  // The app's native request surface must not override the frozen
  // experiment generation settings. The recovery control probe uses the
  // same values, so this also keeps state-equivalence checks comparable.
  const taskSettings = activeTask?.task ?? {};
  const maxNewTokens = Number.isFinite(Number(taskSettings.maxNewTokens))
    ? Number(taskSettings.maxNewTokens)
    : (request.maxNewTokens ?? config.maxNewTokens);
  const temperature = Number.isFinite(Number(taskSettings.temperature))
    ? Number(taskSettings.temperature)
    : (request.temperature ?? config.temperature);
  const seed = Number.isFinite(Number(taskSettings.seed))
    ? Number(taskSettings.seed)
    : (request.seed ?? config.seed);
  const result = await generate({
    requestId: activeTask?.requestId ?? request.requestId,
    promptText: effectivePromptText,
    maxNewTokens,
    temperature,
    seed
  });
  if (activeTask !== null) {
    activeTask.messages = effectiveMessages;
    activeTask.promptText = effectivePromptText;
    activeTask.generation = result;
  }
  return result;
}

async function runProbe(probe = {}) {
  const api = await requireEngine();
  const sessionId = String(
    probe.sessionId ??
      `${config.sessionId}-probe-${Date.now()}-${Math.random().toString(36).slice(2)}`
  );
  const messages = normalizeMessages(probe.messages);
  const promptText = formatMessages(
    messages,
    probe.promptFormat ?? config.promptFormat
  );
  await api.createSession({
    sessionId,
    pageId: `${config.app}-${sessionId}`
  });
  try {
    return {
      ...(await generateInSession(sessionId, {
        promptText,
        maxNewTokens: probe.maxNewTokens ?? config.maxNewTokens,
        temperature: probe.temperature ?? config.temperature,
        seed: probe.seed ?? config.seed
      })),
      sessionId,
      messages,
      promptText
    };
  } finally {
    await api.disposeSession(sessionId).catch(() => undefined);
  }
}

function setInputValue(input, value) {
  input.focus();
  input.value = value;
  input.dispatchEvent(new Event("input", { bubbles: true }));
}

function lastText(selector) {
  const nodes = [...document.querySelectorAll(selector)];
  return nodes.length === 0
    ? ""
    : String(nodes[nodes.length - 1]?.textContent ?? "");
}

function artifexPanelReady() {
  return (
    window.__gameDebug?.chatOpen?.() === true &&
    document.querySelector("#npc-chat-input") !== null &&
    document.querySelector("#npc-chat-panel") !== null
  );
}

async function ensureArtifexPanel(timeoutMs = config.timeoutMs) {
  await waitForCondition(
    () => window.__gameReady === true && window.__gameDebug,
    timeoutMs,
    "Artifex game readiness"
  );
  if (artifexPanelReady()) return;

  const deadline = nowMs() + timeoutMs;
  const teleported = window.__gameDebug.teleportToNearestSettlementSign?.();
  if (teleported !== true) {
    throw recordError("Artifex could not find a settlement");
  }
  await delay(2500);
  try {
    await waitForConditionQuiet(
      () => (window.__gameDebug?.npcs?.()?.length ?? 0) > 0,
      Math.min(30_000, Math.max(1_000, deadline - nowMs()))
    );
  } catch (error) {
    throw recordError(describeError(error));
  }

  let lastError = null;
  for (let attempt = 0; attempt < 3 && nowMs() < deadline; attempt += 1) {
    if (artifexPanelReady()) return;
    if (window.__gameDebug.teleportToNearestNpc?.() !== true) {
      lastError = new Error("Artifex could not reach an NPC");
      break;
    }
    await delay(350 + attempt * 250);
    try {
      await waitForConditionQuiet(
        () =>
          artifexPanelReady() ||
          window.__gameStats?.interactPrompt?.includes("talk") === true,
        Math.min(5_000, Math.max(1_000, deadline - nowMs()))
      );
    } catch (error) {
      lastError = error;
    }
    window.dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "e",
        code: "KeyE",
        bubbles: true
      })
    );
    try {
      await waitForConditionQuiet(
        () => artifexPanelReady(),
        Math.min(5_000, Math.max(1_000, deadline - nowMs()))
      );
      return;
    } catch (error) {
      lastError = error;
    }
  }

  throw recordError(
    lastError instanceof Error
      ? lastError.message
      : "Artifex NPC chat panel did not become ready"
  );
}

async function waitForApplicationReply(selector, previousCount, description) {
  const startedAtMs = nowMs();
  while (nowMs() - startedAtMs < config.timeoutMs) {
    if (document.querySelectorAll(selector).length > previousCount) return;
    const generation = activeTask?.generation;
    if (generation?.error || generation?.stopReason === "failed") {
      throw recordError(
        `${description} failed: ${generation.error ?? "generation stopped"}`
      );
    }
    await delay(50);
  }
  throw recordError(`${description} did not appear within ${config.timeoutMs} ms`);
}

async function runArtifexTask(task) {
  await ensureArtifexPanel();
  const input = document.querySelector("#npc-chat-input");
  const send = document.querySelector("#npc-chat-send");
  if (!input || !send) throw recordError("Artifex chat controls are unavailable");
  const before = document.querySelectorAll(
    "#npc-chat-history .msg.assistant:not(.thinking)"
  ).length;
  setInputValue(input, String(task.prompt ?? ""));
  send.click();
  await waitForApplicationReply(
    "#npc-chat-history .msg.assistant:not(.thinking)",
    before,
    "Artifex assistant reply"
  );
  return {
    uiReplyText: lastText(
      "#npc-chat-history .msg.assistant:not(.thinking)"
    )
  };
}

async function runTavernTask(task) {
  await waitForCondition(
    () => window.npcChat && typeof window.npcChat.onOpen === "function",
    config.timeoutMs,
    "Tavern chat bridge"
  );
  await window.npcChat.onOpen(task.npcId ?? "edda");
  const input = document.querySelector("#chat-input");
  const send = document.querySelector("#chat-send");
  if (!input || !send) throw recordError("Tavern chat controls are unavailable");
  await waitForCondition(
    () => input.disabled === false && send.disabled === false,
    config.timeoutMs,
    "Tavern chat readiness"
  );
  const before = document.querySelectorAll(
    "#chat-messages .msg.npc:not(.streaming)"
  ).length;
  setInputValue(input, String(task.prompt ?? ""));
  send.click();
  await waitForApplicationReply(
    "#chat-messages .msg.npc:not(.streaming)",
    before,
    "Tavern assistant reply"
  );
  return {
    uiReplyText: lastText("#chat-messages .msg.npc:not(.streaming)")
  };
}

async function runApplicationTask(task) {
  if (config.app === "artifex") return runArtifexTask(task);
  if (config.app === "tavern") return runTavernTask(task);
  throw recordError(`unsupported application ${config.app}`);
}

async function runTask(task = {}) {
  if (activeTask !== null) {
    throw recordError("an application task is already running in this tab");
  }
  activeTask = {
    taskId: task.id ?? null,
    task,
    requestId:
      task.requestId ??
      `req-${Date.now()}-${Math.random().toString(36).slice(2)}`,
    phase: task.phase ?? null,
    continuationMessages: normalizeMessages(task.continuationMessages),
    turnDeltaPrompt:
      typeof task.turnDeltaPrompt === "string" ? task.turnDeltaPrompt : null,
    messages: null,
    promptText: null,
    generation: null,
    startedAtMs: nowMs()
  };
  if (activeTask.phase === "recovery") {
    appendLifecycle("restore");
  }
  try {
    const ui = await runApplicationTask(task);
    const generation = activeTask.generation;
    if (generation === null) {
      throw recordError("application did not submit a generation");
    }
    const result = {
      ...generation,
      taskId: activeTask.taskId,
      phase: activeTask.phase,
      promptText: activeTask.promptText,
      applicationMessages: activeTask.messages ?? [],
      uiReplyText: ui.uiReplyText ?? "",
      uiReplyHash: hashText(ui.uiReplyText ?? ""),
      startedAtMs: activeTask.startedAtMs,
      finishedAtMs: nowMs()
    };
    return result;
  } finally {
    activeTask = null;
  }
}

async function setPriority(level) {
  const api = await requireEngine();
  const requestedAtMs = nowMs();
  if (typeof api.setSessionPriority !== "function") {
    const marker = {
      level,
      requestedAtMs,
      resolvedAtMs: nowMs(),
      accepted: false,
      applied: false,
      error: "setSessionPriority is not exposed"
    };
    priorityMarkers.push(marker);
    return marker;
  }
  try {
    const update = await api.setSessionPriority(config.sessionId, level);
    const marker = {
      level,
      requestedAtMs,
      resolvedAtMs: nowMs(),
      accepted: update.accepted === true,
      applied: update.applied !== false && update.accepted === true,
      epoch: update.epoch ?? null,
      atMs: update.atMs ?? null,
      error: null
    };
    priorityMarkers.push(marker);
    return marker;
  } catch (error) {
    const marker = {
      level,
      requestedAtMs,
      resolvedAtMs: nowMs(),
      accepted: false,
      applied: false,
      error: describeError(error)
    };
    priorityMarkers.push(marker);
    return marker;
  }
}

async function reportRenderPressure(sample) {
  const api = await requireEngine();
  const payload = {
    source: sample.source ?? "presented-frame",
    frameCount: numberOr(sample.frameCount, 0),
    p50Ms: numberOr(sample.p50Ms, 0),
    p95Ms: numberOr(sample.p95Ms, 0),
    p99Ms: numberOr(sample.p99Ms, 0),
    maxMs: numberOr(sample.maxMs, 0),
    overBudgetRatio: numberOr(sample.overBudgetRatio, 0),
    over33Ratio: numberOr(sample.over33Ratio, 0),
    over50Ratio: numberOr(sample.over50Ratio, 0),
    longestStallMs: numberOr(sample.longestStallMs, 0),
    ...(sample.refreshHz === undefined
      ? {}
      : { refreshHz: numberOr(sample.refreshHz, 60) }),
    pageOwned: sample.pageOwned !== false
  };
  if (typeof api.reportRenderPressure !== "function") {
    return { applied: false, error: "reportRenderPressure is not exposed" };
  }
  try {
    await api.reportRenderPressure(payload);
    return { applied: true, payload };
  } catch (error) {
    return { applied: false, error: describeError(error), payload };
  }
}

function numberOr(value, fallback) {
  return Number.isFinite(Number(value)) ? Number(value) : fallback;
}

async function diagnostics() {
  const api = await requireEngine();
  const status = api.status?.() ?? null;
  const result = {
    engine: config.engine,
    arm: config.arm,
    sessionId: config.sessionId,
    engineUrl: engineFrame?.src ?? null,
    status,
    metrics: null,
    coordinator: null,
    priorityLog:
      typeof api.priorityLog === "function" ? api.priorityLog() : [],
    pressure: null,
    lifecycleLog: lifecycleLog.map((entry) => ({ ...entry })),
    ui: {
      app: config.app,
      visibilityState: document.visibilityState,
      panel: document.querySelector("#npc-chat-panel") !== null,
      tavern:
        document.querySelector("#chat-input") !== null &&
        document.querySelector("#chat-send") !== null
    }
  };
  try {
    result.metrics = await api.metrics();
  } catch (error) {
    result.metricsError = describeError(error);
  }
  try {
    const provider = engineFrame?.contentWindow?.__wllmCoordinatorDiagnosticsProvider;
    if (typeof provider === "function") {
      result.coordinator = await provider();
    }
  } catch (error) {
    result.coordinatorError = describeError(error);
  }
  try {
    if (typeof api.reportRenderPressure === "function") {
      result.pressure =
        engineFrame?.contentWindow?.__wllmRenderPressure ?? null;
    }
  } catch {
    result.pressure = null;
  }
  return result;
}

function hashText(value) {
  let hash = 2166136261;
  const text = String(value);
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

function snapshot() {
  return {
    app: config.app,
    arm: config.arm,
    engine: config.engine,
    sessionId: config.sessionId,
    ready,
    error: firstError,
    errors: errors.map((entry) => ({ ...entry })),
    priorityMarkers: priorityMarkers.map((entry) => ({ ...entry })),
    progress: progressEvents.map((entry) => ({ ...entry })),
    lifecycleLog: lifecycleLog.map((entry) => ({ ...entry })),
    ui: {
      visibilityState: document.visibilityState,
      canvas: document.querySelector("canvas") !== null,
      artifexPanel: document.querySelector("#npc-chat-panel") !== null,
      tavernPanel: document.querySelector("#chat-input") !== null
    }
  };
}

function installArtifexMock() {
  window.__NPC_CHAT_MOCK__ = async (messages) => {
    const result = await generateFromMessages(messages, {
      promptFormat: config.promptFormat,
      maxNewTokens:
        activeTask?.task?.maxNewTokens ?? config.maxNewTokens,
      temperature: config.temperature,
      seed: config.seed
    });
    if (result.error !== null) throw new Error(result.error);
    return result.text;
  };
}

window.__wllmE2E = {
  config: { ...config },
  initialize,
  ready: () => ready,
  readyPromise: () => readyPromise,
  runTask,
  generate,
  generateFromMessages,
  runProbe,
  setPriority,
  reportRenderPressure,
  diagnostics,
  snapshot,
  getProgress: () => progressEvents.map((entry) => ({ ...entry })),
  getPriorityMarkers: () =>
    priorityMarkers.map((entry) => ({ ...entry })),
  getErrors: () => errors.map((entry) => ({ ...entry })),
  getLifecycleLog: () => lifecycleLog.map((entry) => ({ ...entry })),
  getLoadCount: () =>
    lifecycleLog.filter((entry) => entry.type === "load").length,
  markLifecycle: (request) =>
    appendLifecycle(
      typeof request === "string" ? request : String(request?.type ?? "unknown"),
      typeof request === "string" ? {} : request?.detail ?? {}
    )
};

installArtifexMock();
void initialize().catch((error) => {
  recordError(error);
  console.error("[wllm-e2e] adapter initialization failed", error);
});
