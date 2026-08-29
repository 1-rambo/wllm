import { spawn } from 'node:child_process';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import {
  copyFile,
  mkdir,
  readFile,
  rm,
  stat,
  writeFile,
} from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import { createServer } from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { chromium } from 'playwright';

const PROTOCOL_VERSION = 'wllm.browser-service.experiment.v2';
const args = Object.fromEntries(
  process.argv.slice(2).map((arg) => {
    const [key, ...rest] = arg.replace(/^--/, '').split('=');
    return [key, rest.join('=') || 'true'];
  })
);
const scenario = args.scenario ?? 'activate';
const wasmBuild = args['wasm-build'] ?? process.env.WLLM_WASM_BUILD_DIR;
const outputRoot = args.output ?? 'artifacts/runs';
const modelFile = args['model-file'] ?? process.env.WLLM_MODEL_FILE;
const runId =
  args['run-id'] ??
  `${new Date().toISOString().replace(/[:.]/g, '-')}-${scenario}`;
const reservePort = () =>
  new Promise((resolve, reject) => {
    const probe = createServer();
    probe.once('error', reject);
    probe.listen(0, '127.0.0.1', () => {
      const address = probe.address();
      if (!address || typeof address === 'string') {
        probe.close();
        reject(new Error('cannot reserve experiment server port'));
        return;
      }
      probe.close((error) =>
        error ? reject(error) : resolve(Number(address.port))
      );
    });
  });
const port = args.port ? Number(args.port) : await reservePort();
const scratchName = runId.replace(/[^A-Za-z0-9_.-]/g, '_');
const scratchDir = path.join('_tmp', 'browser-service-experiment', scratchName);
const scratchUrl = `/_tmp/browser-service-experiment/${scratchName}`;
const modelUrl = modelFile
  ? `http://127.0.0.1:${port}/__wllm_model.gguf`
  : (args.model ??
    process.env.WLLM_MODEL_URL ??
    'https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf');
const startedAt = new Date().toISOString();
const backgroundPredict = Number(args['background-predict'] ?? 64);
const foregroundPredict = Number(args['foreground-predict'] ?? 8);
const sessionPredict = Number(args['session-predict'] ?? 8);
const pressureSessions = Number(args['pressure-sessions'] ?? 0);
const retentionDecoys = Math.max(0, Number(args['retention-decoys'] ?? 0));
const pressurePredict = Number(args['pressure-predict'] ?? sessionPredict);
const targetPromptRepeats = Math.max(
  1,
  Number(args['target-prompt-repeats'] ?? 1)
);
const pressurePromptRepeats = Math.max(
  1,
  Number(args['pressure-prompt-repeats'] ?? 1)
);
const backgroundPromptRepeats = Math.max(
  1,
  Number(args['background-prompt-repeats'] ?? 1)
);
const foregroundPromptRepeats = Math.max(
  1,
  Number(args['foreground-prompt-repeats'] ?? 1)
);
const requestCount = Math.max(1, Number(args.requests ?? 16));
const requestPredict =
  args['request-predict'] === undefined
    ? null
    : Math.max(1, Number(args['request-predict']));
const foregroundEvery = Math.max(1, Number(args['foreground-every'] ?? 4));
const arrivalGapMs = Math.max(0, Number(args['arrival-gap-ms'] ?? 0));
const seed = Number(args.seed ?? 42);
const memoryCapMb = Number(args['memory-cap-mb'] ?? 128);
const warmupRuns = Number(args['warmup-runs'] ?? 1);
const warmupPredict = Number(args['warmup-predict'] ?? 4);
const interactionSamples = Number(args['interaction-samples'] ?? 1);
const interactionIntervalMs = Number(args['interaction-interval-ms'] ?? 0);
const uiWorkItems = Math.max(0, Number(args['ui-work-items'] ?? 0));
const traceFile = args['trace-file'] ?? process.env.WLLM_TRACE_FILE;
const trace = traceFile
  ? JSON.parse(await readFile(path.resolve(traceFile), 'utf8'))
  : null;
if (trace && trace.schemaVersion !== 'wllm.browser-service.trace.v1') {
  throw new Error(
    `Unsupported trace schema: ${trace.schemaVersion ?? 'missing'}`
  );
}
const traceClientCount = Array.isArray(trace?.clients)
  ? trace.clients.length
  : 0;
const clientCount = Math.max(
  2,
  Number(args['client-count'] ?? traceClientCount ?? 2)
);
const deviceId =
  args['device-id'] ?? process.env.WLLM_DEVICE_ID ?? 'unassigned';
const operator = args.operator ?? process.env.WLLM_OPERATOR ?? 'unassigned';
const trial = Number(args.trial ?? 1);
const deviceFile = args['device-file'] ?? process.env.WLLM_DEVICE_FILE;
const gitRevision = execFileSync('git', ['rev-parse', 'HEAD'], {
  encoding: 'utf8',
}).trim();
const gitDirty =
  execFileSync('git', ['status', '--porcelain'], {
    encoding: 'utf8',
  }).trim().length > 0;
const llamaCppRevision = execFileSync(
  'git',
  ['-C', 'llama.cpp', 'rev-parse', 'HEAD'],
  { encoding: 'utf8' }
).trim();
const sha256File = (filePath) =>
  new Promise((resolve, reject) => {
    const hash = createHash('sha256');
    const input = createReadStream(filePath);
    input.on('error', reject);
    input.on('data', (chunk) => hash.update(chunk));
    input.on('end', () => resolve(hash.digest('hex')));
  });

if (!wasmBuild) {
  throw new Error('Set --wasm-build=<source-built Wasm directory>');
}
if (gitDirty && args['allow-dirty'] !== 'true') {
  throw new Error(
    'Refusing to record an experiment from a dirty worktree. Commit/stash changes or pass --allow-dirty=true for development-only runs.'
  );
}

const browserCandidates = [
  args.browser,
  process.env.CHROMIUM_EXECUTABLE_PATH,
  chromium.executablePath(),
  `${process.env.HOME}/.cache/ms-playwright/chromium-1148/chrome-linux/chrome`,
  `${process.env.HOME}/.cache/ms-playwright/chromium-1217/chrome-linux64/chrome`,
  `${process.env.HOME}/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome`,
].filter(Boolean);
const executablePath = browserCandidates.find((candidate) =>
  existsSync(candidate)
);
if (!executablePath) throw new Error('Chromium executable is unavailable');

const modelFileStat = modelFile ? await stat(modelFile) : null;
const deviceMetadata = deviceFile
  ? JSON.parse(await readFile(deviceFile, 'utf8'))
  : null;
if (
  deviceMetadata?.deviceId &&
  String(deviceMetadata.deviceId) !== String(deviceId)
) {
  throw new Error(
    `device.json deviceId=${deviceMetadata.deviceId} does not match --device-id=${deviceId}`
  );
}
const [modelSha256, wasmJsSha256, wasmBinarySha256] = await Promise.all([
  args['model-sha256']
    ? Promise.resolve(args['model-sha256'])
    : modelFile
      ? sha256File(modelFile)
      : Promise.resolve(null),
  args['wasm-js-sha256']
    ? Promise.resolve(args['wasm-js-sha256'])
    : sha256File(path.join(wasmBuild, 'wllama.js')),
  args['wasm-sha256']
    ? Promise.resolve(args['wasm-sha256'])
    : sha256File(path.join(wasmBuild, 'wllama.wasm')),
]);
const traceSha256 = traceFile
  ? await sha256File(path.resolve(traceFile))
  : null;

await mkdir(scratchDir, { recursive: true });
await copyFile(
  path.join(wasmBuild, 'wllama.js'),
  path.join(scratchDir, 'wllama.js')
);
await copyFile(
  path.join(wasmBuild, 'wllama.wasm'),
  path.join(scratchDir, 'wllama.wasm')
);
const server = spawn(process.execPath, ['scripts/http_server.js'], {
  cwd: new URL('../..', import.meta.url),
  env: {
    ...process.env,
    PORT: String(port),
    ...(modelFile ? { WLLM_MODEL_FILE: path.resolve(modelFile) } : {}),
  },
  stdio: ['ignore', 'pipe', 'inherit'],
});
let browser;

const waitServer = () =>
  new Promise((resolve, reject) => {
    const timer = setTimeout(
      () => reject(new Error('server startup timeout')),
      10000
    );
    server.stdout.on('data', (chunk) => {
      if (String(chunk).includes('Server running')) {
        clearTimeout(timer);
        resolve();
      }
    });
    server.once('exit', (code) =>
      reject(new Error(`server exited with ${code}`))
    );
  });

const waitFor = async (read, predicate, timeoutMs = 30000) => {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const value = await read();
    if (predicate(value)) return value;
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
  throw new Error('condition timeout');
};

const focusClient = async (page, observer, clientId) => {
  const before = await observer.evaluate(() =>
    window.wllmBrowserClient.getTelemetry()
  );
  const beforeSequence = before.at(-1)?.sequence ?? 0;
  await page.bringToFront();
  await page.evaluate(() => window.wllmBrowserClient.updateLifecycle());
  await waitFor(
    () => observer.evaluate(() => window.wllmBrowserClient.getTelemetry()),
    (events) =>
      events.some(
        (event) =>
          event.sequence > beforeSequence &&
          event.type === 'foreground_change' &&
          event.clientId === clientId &&
          event.data?.foreground === true
      )
  );
};

const demoteClient = async (
  page,
  observer,
  clientId,
  lifecycleState = 'hidden'
) => {
  const before = await observer.evaluate(() =>
    window.wllmBrowserClient.getTelemetry()
  );
  const beforeSequence = before.at(-1)?.sequence ?? 0;
  await page.evaluate(
    (state) =>
      window.wllmBrowserClient.updateLifecycle({
        foreground: false,
        lifecycleState: state,
      }),
    lifecycleState
  );
  await waitFor(
    () => observer.evaluate(() => window.wllmBrowserClient.getTelemetry()),
    (events) =>
      events.some(
        (event) =>
          event.sequence > beforeSequence &&
          event.type === 'foreground_change' &&
          event.clientId === clientId &&
          event.data?.foreground === false
      )
  );
};

const focusClientWithoutAck = async (page) => {
  await page.bringToFront();
  await page.evaluate(() => {
    void window.wllmBrowserClient.updateLifecycle();
  });
};

const demoteClientWithoutAck = async (page) => {
  await page.evaluate(() => {
    void window.wllmBrowserClient.updateLifecycle({
      foreground: false,
      lifecycleState: 'hidden',
    });
  });
};

const markMeasurement = async (page, type, data = {}) => {
  await page.evaluate(
    ({ type, data }) => window.wllmBrowserClient.recordEvent(type, data),
    { type, data }
  );
};

const makePrompt = (label, repeats) =>
  Array.from(
    { length: Math.max(1, Math.floor(repeats)) },
    (_, index) =>
      `${label} segment ${index}: preserve concrete page facts, compare alternatives, and keep the reasoning state available for a later follow-up.`
  ).join(' ');

const recordStateSnapshot = async (page, label, caps) => {
  await page.evaluate(
    async ({ label, caps }) => {
      const state = await window.wllmBrowserClient.chatGetState();
      const usage = [
        state.tierStats.l1Tokens,
        state.tierStats.l2Tokens,
        state.tierStats.l3Tokens,
      ];
      await window.wllmBrowserClient.recordEvent('state_snapshot', {
        label,
        totalSnapshotTokenBytes: state.totalSnapshotTokenBytes,
        contextMemoryBytes: state.contextMemoryBytes,
        memoryCapBytes: state.memoryCapBytes,
        nodeCount: state.nodes.size,
        capacityViolationCount: usage.filter(
          (value, index) => caps[index] > 0 && value > caps[index]
        ).length,
        ...state.tierStats,
      });
    },
    { label, caps }
  );
};

const quantile = (values, q) => {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const position = Math.min(
    sorted.length - 1,
    Math.max(0, (sorted.length - 1) * q)
  );
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sorted[lower];
  const weight = position - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
};

const summarize = (events, clientIds, scenarioName) => {
  const measurementStart = events.find(
    (event) => event.type === 'measurement_start'
  );
  const measurementEnd = [...events]
    .reverse()
    .find((event) => event.type === 'measurement_end');
  const measuredEvents = measurementStart
    ? events.filter(
        (event) =>
          event.sequence >= measurementStart.sequence &&
          (!measurementEnd || event.sequence <= measurementEnd.sequence)
      )
    : events;
  const measurementStartSequence = measurementStart?.sequence ?? 0;
  const lastForegroundChangeBeforeMeasurement = [...events]
    .reverse()
    .find(
      (event) =>
        event.sequence < measurementStartSequence &&
        event.type === 'foreground_change' &&
        event.clientId === clientIds.foreground &&
        event.data?.foreground === true
    );
  const foregroundChanges = measuredEvents.filter(
    (event) =>
      event.type === 'foreground_change' &&
      event.data?.foreground === true &&
      (scenarioName === 'trace' || event.clientId === clientIds.foreground)
  );
  const foregroundChange = [...measuredEvents]
    .reverse()
    .find(
      (event) =>
        event.type === 'foreground_change' &&
        event.clientId === clientIds.foreground &&
        event.data?.foreground === true
    );
  const foregroundReference =
    foregroundChange ??
    (scenarioName === 'load'
      ? lastForegroundChangeBeforeMeasurement
      : undefined);
  const foregroundChangedAt = Number(
    foregroundReference?.data?.changedAt ?? foregroundReference?.wallTimeMs ?? 0
  );
  const firstDispatch = measuredEvents.find(
    (event) =>
      event.type === 'slice_dispatch' &&
      event.clientId === clientIds.foreground &&
      (!foregroundReference ||
        event.timestampMs >= foregroundReference.timestampMs)
  );
  const firstToken = measuredEvents.find(
    (event) =>
      event.type === 'first_token' &&
      event.clientId === clientIds.foreground &&
      (!foregroundReference ||
        event.timestampMs >= foregroundReference.timestampMs)
  );
  const admissions = measuredEvents.filter((event) =>
    ['admission_commit', 'admission_fallback', 'admission_reject'].includes(
      event.type
    )
  );
  let latestHotNodeIds = new Set();
  let protectedVictimViolationCount = 0;
  let cacheHintFailureCount = 0;
  for (const event of measuredEvents) {
    if (event.type === 'cache_hint') {
      if (event.data?.applied === false) {
        cacheHintFailureCount += 1;
      } else {
        latestHotNodeIds = new Set(
          Array.isArray(event.data?.hotNodeIds)
            ? event.data.hotNodeIds.map(Number)
            : []
        );
      }
      continue;
    }
    if (
      ['admission_commit', 'admission_fallback', 'admission_reject'].includes(
        event.type
      )
    ) {
      const victims = Array.isArray(event.data?.victimIds)
        ? event.data.victimIds.map(Number)
        : [];
      protectedVictimViolationCount += victims.filter((nodeId) =>
        latestHotNodeIds.has(nodeId)
      ).length;
    }
  }
  const count = (type) =>
    measuredEvents.filter((event) => event.type === type).length;
  const released = measuredEvents.filter(
    (event) =>
      event.type === 'state_release' &&
      event.data?.reason !== 'experiment_cleanup'
  );
  const cleanupReleased = measuredEvents.filter(
    (event) =>
      event.type === 'state_release' &&
      event.data?.reason === 'experiment_cleanup'
  );
  const resumedClientId =
    scenarioName === 'retain'
      ? clientIds.background
      : scenarioName === 'handoff'
        ? clientIds.foreground
        : undefined;
  const sessionResume =
    scenarioName === 'retain' || scenarioName === 'handoff'
      ? [...measuredEvents]
          .reverse()
          .find(
            (event) =>
              event.type === 'session_resume' &&
              event.clientId === resumedClientId
          )
      : undefined;
  const resumedSession = measuredEvents.find(
    (event) =>
      (event.type === 'state_restore' || event.type === 'state_rebuild') &&
      (resumedClientId === undefined || event.clientId === resumedClientId) &&
      (!sessionResume || event.sequence >= sessionResume.sequence) &&
      typeof event.data?.restoreSource === 'string'
  );
  const paintLatencies = measuredEvents
    .filter(
      (event) =>
        event.type === 'next_paint' && event.clientId === clientIds.foreground
    )
    .map((event) => Number(event.data?.latencyMs ?? 0));
  const sliceDurationsMs = measuredEvents
    .filter((event) => event.type === 'slice_complete')
    .map((event) => Number(event.data?.durationMs ?? 0));
  const foregroundComplete = [...measuredEvents]
    .reverse()
    .find(
      (event) =>
        event.type === 'request_complete' &&
        event.clientId === clientIds.foreground
    );
  const backgroundComplete = [...measuredEvents]
    .reverse()
    .find(
      (event) =>
        event.type === 'request_complete' &&
        event.clientId === clientIds.background
    );
  const sessionResumedAt = Number(
    sessionResume?.data?.changedAt ?? sessionResume?.wallTimeMs ?? 0
  );
  const resumeDispatch = sessionResume
    ? measuredEvents.find(
        (event) =>
          event.type === 'slice_dispatch' &&
          event.clientId === clientIds.background &&
          event.timestampMs >= sessionResume.timestampMs
      )
    : undefined;
  const resumeFirstToken = sessionResume
    ? measuredEvents.find(
        (event) =>
          event.type === 'first_token' &&
          event.clientId === clientIds.background &&
          event.timestampMs >= sessionResume.timestampMs
      )
    : undefined;
  const completedRequests = count('request_complete');
  const failedRequests = count('request_fail');
  const measuredCompleted = measuredEvents.filter(
    (event) => event.type === 'request_complete'
  );
  const measuredFailed = measuredEvents.filter(
    (event) => event.type === 'request_fail'
  );
  const requestArrivals = measuredEvents.filter(
    (event) => event.type === 'request_arrive'
  );
  const workloadStartedAt = measurementStart
    ? Number(measurementStart.wallTimeMs)
    : requestArrivals.length
      ? Math.min(...requestArrivals.map((event) => Number(event.wallTimeMs)))
      : 0;
  const terminalEvents = [...measuredCompleted, ...measuredFailed];
  const workloadCompletedAt = measurementEnd
    ? Number(measurementEnd.wallTimeMs)
    : terminalEvents.length
      ? Math.max(...terminalEvents.map((event) => Number(event.wallTimeMs)))
      : workloadStartedAt;
  const completedTokens = measuredCompleted.reduce(
    (sum, event) => sum + Number(event.data?.generatedTokens ?? 0),
    0
  );
  const schedulerDecisions = measuredEvents
    .filter((event) => event.type === 'scheduler_decision')
    .map((event) => Number(event.data?.durationMs ?? 0));
  const cacheHints = measuredEvents
    .filter((event) => event.type === 'cache_hint')
    .map((event) => Number(event.data?.durationMs ?? 0));
  const requestWallTimes = measuredCompleted.map((event) =>
    Number(event.data?.totalWallMs ?? 0)
  );
  const requestTtfts = measuredCompleted.map((event) =>
    Number(event.data?.ttftMs ?? 0)
  );
  const requestQueueWaits = measuredCompleted.map((event) =>
    Number(event.data?.queueWaitMs ?? 0)
  );
  const foregroundRequestWallTimes = measuredCompleted
    .filter((event) => event.data?.foreground === true)
    .map((event) => Number(event.data?.totalWallMs ?? 0));
  const backgroundRequestWallTimes = measuredCompleted
    .filter((event) => event.data?.foreground !== true)
    .map((event) => Number(event.data?.totalWallMs ?? 0));
  const phaseTotal = (name) =>
    measuredCompleted.reduce(
      (sum, event) => sum + Number(event.data?.phases?.[name] ?? 0),
      0
    );
  const totalControlPlaneMs =
    schedulerDecisions.reduce((sum, value) => sum + value, 0) +
    cacheHints.reduce((sum, value) => sum + value, 0);
  const admissionTotalMs = admissions.reduce(
    (sum, event) => sum + Number(event.data?.latencyMs ?? 0),
    0
  );
  const workloadMakespanMs = Math.max(
    0,
    workloadCompletedAt - workloadStartedAt
  );
  const stateSnapshots = events.filter(
    (event) => event.type === 'state_snapshot'
  );
  const peakSnapshotTokenBytes = stateSnapshots.length
    ? Math.max(
        ...stateSnapshots.map((event) =>
          Number(event.data?.totalSnapshotTokenBytes ?? 0)
        )
      )
    : 0;
  const peakL1Tokens = stateSnapshots.length
    ? Math.max(
        ...stateSnapshots.map((event) => Number(event.data?.l1Tokens ?? 0))
      )
    : 0;
  const peakL2Tokens = stateSnapshots.length
    ? Math.max(
        ...stateSnapshots.map((event) => Number(event.data?.l2Tokens ?? 0))
      )
    : 0;
  const peakL3Tokens = stateSnapshots.length
    ? Math.max(
        ...stateSnapshots.map((event) => Number(event.data?.l3Tokens ?? 0))
      )
    : 0;
  const overduePromotionCount = measuredCompleted.reduce(
    (sum, event) => sum + Number(event.data?.promotedToOverdueCount ?? 0),
    0
  );
  const configuredCaps = [
    Number(args['l1-cap'] ?? 512),
    Number(args['l2-cap'] ?? 2048),
    Number(args['l3-cap'] ?? 8192),
  ];
  const capacityViolationCount = stateSnapshots.filter((event) => {
    const usage = [
      Number(event.data?.l1Tokens ?? 0),
      Number(event.data?.l2Tokens ?? 0),
      Number(event.data?.l3Tokens ?? 0),
    ];
    return usage.some(
      (value, index) =>
        configuredCaps[index] > 0 && value > configuredCaps[index]
    );
  }).length;
  const admissionRepairSteps = admissions.map((event) =>
    Number(event.data?.repairSteps ?? 0)
  );
  const configuredRepairBudget = Number(args['repair-budget'] ?? 32);
  const admissionBudgetViolationCount =
    args.admission === 'monotonic'
      ? admissionRepairSteps.filter((steps) => steps > configuredRepairBudget)
          .length
      : 0;
  const activationLatenciesMs = foregroundChanges
    .map((change) => {
      const dispatch = measuredEvents.find(
        (event) =>
          event.type === 'slice_dispatch' &&
          event.clientId === change.clientId &&
          event.timestampMs >= change.timestampMs
      );
      const changedAt = Number(
        change.data?.changedAt ?? change.wallTimeMs ?? 0
      );
      return dispatch ? Number(dispatch.wallTimeMs) - changedAt : null;
    })
    .filter((value) => value !== null);
  const foregroundResumeTtftsMs = foregroundChanges
    .map((change) => {
      const token = measuredEvents.find(
        (event) =>
          event.type === 'first_token' &&
          event.clientId === change.clientId &&
          event.timestampMs >= change.timestampMs
      );
      const changedAt = Number(
        change.data?.changedAt ?? change.wallTimeMs ?? 0
      );
      return token ? Number(token.wallTimeMs) - changedAt : null;
    })
    .filter((value) => value !== null);
  const restoreSourceCounts = measuredEvents
    .filter(
      (event) =>
        (event.type === 'state_restore' || event.type === 'state_rebuild') &&
        typeof event.data?.restoreSource === 'string'
    )
    .reduce((counts, event) => {
      const source = String(event.data.restoreSource);
      counts[source] = (counts[source] ?? 0) + 1;
      return counts;
    }, {});
  const uiMeasurement = [...events]
    .reverse()
    .find((event) => event.type === 'ui_measurement')?.data;
  return {
    eventCount: events.length,
    foregroundActivationMs:
      foregroundReference && firstDispatch
        ? Number(firstDispatch.wallTimeMs) - foregroundChangedAt
        : null,
    foregroundFirstTokenMs:
      foregroundReference && firstToken
        ? Number(firstToken.wallTimeMs) - foregroundChangedAt
        : null,
    foregroundTotalWallMs:
      foregroundComplete?.data?.totalWallMs === undefined
        ? null
        : Number(foregroundComplete.data.totalWallMs),
    backgroundTotalWallMs:
      backgroundComplete?.data?.totalWallMs === undefined
        ? null
        : Number(backgroundComplete.data.totalWallMs),
    sessionResumeToDispatchMs:
      sessionResume && resumeDispatch
        ? Number(resumeDispatch.wallTimeMs) - sessionResumedAt
        : null,
    sessionResumeToFirstTokenMs:
      sessionResume && resumeFirstToken
        ? Number(resumeFirstToken.wallTimeMs) - sessionResumedAt
        : null,
    resumeRestoreMs: Number(resumedSession?.data?.restoreMs ?? 0),
    resumeRebuildMs: Number(resumedSession?.data?.rebuildMs ?? 0),
    sliceDurationsMs,
    sliceDurationP95Ms: quantile(sliceDurationsMs, 0.95),
    sliceDurationMaxMs: sliceDurationsMs.length
      ? Math.max(...sliceDurationsMs)
      : 0,
    admissionCount: admissions.length,
    admissionCommitCount: count('admission_commit'),
    admissionFallbackCount: count('admission_fallback'),
    admissionRejectCount: count('admission_reject'),
    admissionRepairStepsP95: quantile(admissionRepairSteps, 0.95),
    admissionRepairStepsMax: admissionRepairSteps.length
      ? Math.max(...admissionRepairSteps)
      : 0,
    admissionBudgetViolationCount,
    protectedVictimViolationCount,
    cacheHintFailureCount,
    admissionLatencyP95Ms: quantile(
      admissions.map((event) => Number(event.data?.latencyMs ?? 0)),
      0.95
    ),
    admissionInitialTokenDeficitP95: quantile(
      admissions.map((event) => Number(event.data?.initialTokenDeficit ?? 0)),
      0.95
    ),
    admissionInitialSlotDeficitP95: quantile(
      admissions.map((event) => Number(event.data?.initialSlotDeficit ?? 0)),
      0.95
    ),
    admissionDemotedTokens: admissions.reduce(
      (sum, event) => sum + Number(event.data?.demotedTokens ?? 0),
      0
    ),
    admissionCheckpointedTokens: admissions.reduce(
      (sum, event) => sum + Number(event.data?.checkpointedTokens ?? 0),
      0
    ),
    admissionRemovedStates: admissions.reduce(
      (sum, event) => sum + Number(event.data?.removedStates ?? 0),
      0
    ),
    admissionVictimCount: admissions.reduce(
      (sum, event) =>
        sum +
        (Array.isArray(event.data?.victimIds)
          ? event.data.victimIds.length
          : 0),
      0
    ),
    stateReleaseCount: released.length,
    releasedTokens: released.reduce(
      (sum, event) => sum + Number(event.data?.releasedTokens ?? 0),
      0
    ),
    cleanupReleaseCount: cleanupReleased.length,
    cleanupReleasedTokens: cleanupReleased.reduce(
      (sum, event) => sum + Number(event.data?.releasedTokens ?? 0),
      0
    ),
    stateRebuildCount: count('state_rebuild'),
    resumedSessionSource: resumedSession?.data?.restoreSource ?? null,
    completedRequestCount: completedRequests,
    requestFailureCount: failedRequests,
    requestSuccessRate:
      completedRequests + failedRequests > 0
        ? completedRequests / (completedRequests + failedRequests)
        : 0,
    measuredCompletedRequestCount: measuredCompleted.length,
    measuredRequestFailureCount: measuredFailed.length,
    measuredRequestSuccessRate:
      measuredCompleted.length + measuredFailed.length > 0
        ? measuredCompleted.length /
          (measuredCompleted.length + measuredFailed.length)
        : 0,
    foregroundSwitchCount: foregroundChanges.length,
    foregroundActivationP50Ms: quantile(activationLatenciesMs, 0.5),
    foregroundActivationP95Ms: quantile(activationLatenciesMs, 0.95),
    foregroundResumeTtftP50Ms: quantile(foregroundResumeTtftsMs, 0.5),
    foregroundResumeTtftP95Ms: quantile(foregroundResumeTtftsMs, 0.95),
    restoreSourceCounts,
    workloadMakespanMs,
    aggregateThroughputTokensPerSecond:
      workloadMakespanMs > 0
        ? (completedTokens * 1000) / workloadMakespanMs
        : 0,
    requestWallTimeP50Ms: quantile(requestWallTimes, 0.5),
    requestWallTimeP95Ms: quantile(requestWallTimes, 0.95),
    requestTtftP50Ms: quantile(requestTtfts, 0.5),
    requestTtftP95Ms: quantile(requestTtfts, 0.95),
    requestQueueWaitP95Ms: quantile(requestQueueWaits, 0.95),
    foregroundRequestWallTimeP95Ms: quantile(foregroundRequestWallTimes, 0.95),
    backgroundRequestWallTimeP95Ms: quantile(backgroundRequestWallTimes, 0.95),
    overduePromotionCount,
    totalPrefillMs: phaseTotal('prefillMs'),
    totalDecodeMs: phaseTotal('decodeMs'),
    totalCacheLoadMs: phaseTotal('cacheLoadMs'),
    totalRebuildMs: phaseTotal('rebuildMs'),
    totalParentRecoverMs: phaseTotal('parentRecoverMs'),
    totalPostMs: phaseTotal('postMs'),
    schedulerDecisionCount: schedulerDecisions.length,
    schedulerDecisionTotalMs: schedulerDecisions.reduce(
      (sum, value) => sum + value,
      0
    ),
    schedulerDecisionP95Ms: quantile(schedulerDecisions, 0.95),
    cacheHintCount: cacheHints.length,
    cacheHintTotalMs: cacheHints.reduce((sum, value) => sum + value, 0),
    cacheHintP95Ms: quantile(cacheHints, 0.95),
    admissionTotalMs,
    controlPlaneTotalMs: totalControlPlaneMs,
    controlPlanePct:
      workloadMakespanMs > 0
        ? (totalControlPlaneMs / workloadMakespanMs) * 100
        : 0,
    stateSnapshotCount: stateSnapshots.length,
    peakSnapshotTokenBytes,
    peakL1Tokens,
    peakL2Tokens,
    peakL3Tokens,
    capacityViolationCount,
    uiFrameSampleCount: Number(uiMeasurement?.sampleCount ?? 0),
    uiWorkloadBackend: String(uiMeasurement?.backend ?? 'disabled'),
    uiAverageFps: Number(uiMeasurement?.averageFps ?? 0),
    uiFrameTimeP50Ms: Number(uiMeasurement?.frameTimeP50Ms ?? 0),
    uiFrameTimeP95Ms: Number(uiMeasurement?.frameTimeP95Ms ?? 0),
    uiFrameTimeP99Ms: Number(uiMeasurement?.frameTimeP99Ms ?? 0),
    uiMaxFrameTimeMs: Number(uiMeasurement?.maxFrameTimeMs ?? 0),
    uiLongFrameRatio: Number(uiMeasurement?.longFrameRatio ?? 0),
    inputToNextPaintP95Ms: quantile(paintLatencies, 0.95),
  };
};

try {
  await waitServer();
  browser = await chromium.launch({
    executablePath,
    headless: args.headless !== 'false',
    args: os.platform() === 'linux' ? ['--no-sandbox'] : [],
  });
  const context = await browser.newContext();
  const background = await context.newPage();
  const foreground = await context.newPage();
  const extraPages = await Promise.all(
    Array.from({ length: Math.max(0, clientCount - 2) }, () =>
      context.newPage()
    )
  );
  const pages = [background, foreground, ...extraPages];
  const base = new URL(
    `http://127.0.0.1:${port}/examples/browser-service-smoke/`
  );
  base.searchParams.set('module', `${scratchUrl}/wllama.js`);
  base.searchParams.set('wasm', `${scratchUrl}/wllama.wasm`);
  base.searchParams.set('gpu', args.gpu === 'true' ? '1' : '0');
  base.searchParams.set('scheduler', args.scheduler ?? 'foreground-aware');
  base.searchParams.set('slice', args.slice ?? '32');
  base.searchParams.set(
    'foregroundSlice',
    args['foreground-slice'] ?? args.slice ?? '32'
  );
  base.searchParams.set('backgroundSlice', args['background-slice'] ?? '4');
  base.searchParams.set('handoffGrace', args['handoff-grace'] ?? '16');
  base.searchParams.set('yield', args.yield ?? 'true');
  base.searchParams.set('queueMaxPending', args['queue-max-pending'] ?? '128');
  base.searchParams.set(
    'prefillSliceMaxMs',
    args['prefill-slice-max-ms'] ?? '5000'
  );
  base.searchParams.set('waitBudgetScale', args['wait-budget-scale'] ?? '1');
  if (args['wait-budget-cap-ms'] !== undefined) {
    base.searchParams.set('waitBudgetCapMs', args['wait-budget-cap-ms']);
  }
  base.searchParams.set(
    'costWarmupRequests',
    args['cost-warmup-requests'] ?? '20'
  );
  base.searchParams.set(
    'costSampleWindow',
    args['cost-sample-window'] ?? '256'
  );
  base.searchParams.set(
    'evictionHintTopK',
    args['eviction-hint-top-k'] ?? '64'
  );
  base.searchParams.set('retention', args.retention ?? 'lifecycle-aware');
  base.searchParams.set('retentionTtl', args['retention-ttl'] ?? '30000');
  // Establish the first SharedWorker before attaching the remaining tabs.
  // Starting several blob-backed SharedWorkers concurrently can race through
  // the localStorage registry on slower browsers.
  await background.goto(base.href);
  await background.evaluate(() => window.wllmBrowserReady);
  for (const page of pages.slice(1)) {
    await page.goto(base.href);
    await page.evaluate(() => window.wllmBrowserReady);
  }
  const browserRuntime = await foreground.evaluate(async () => {
    const runtime = {
      userAgent: navigator.userAgent,
      hardwareConcurrency: navigator.hardwareConcurrency,
      deviceMemoryGiB:
        'deviceMemory' in navigator ? Number(navigator.deviceMemory) : null,
      webgpu: {
        supported: Boolean(navigator.gpu),
        adapterAvailable: false,
        vendor: null,
        architecture: null,
        device: null,
        description: null,
      },
    };
    if (!navigator.gpu) return runtime;
    try {
      const adapter = await navigator.gpu.requestAdapter();
      runtime.webgpu.adapterAvailable = Boolean(adapter);
      if (adapter) {
        const info = adapter.info ?? {};
        runtime.webgpu.vendor = info.vendor || null;
        runtime.webgpu.architecture = info.architecture || null;
        runtime.webgpu.device = info.device || null;
        runtime.webgpu.description = info.description || null;
      }
    } catch {
      // Keep the capability record even when adapter probing is unavailable.
    }
    return runtime;
  });

  const modelLoadStartedAt = performance.now();
  await background.evaluate(
    ({ modelUrl, useWebGPU, nCtx, nBatch, nSeqMax, seed }) =>
      window.wllmBrowserClient.loadModelFromUrl(modelUrl, {
        seed,
        n_ctx: nCtx,
        n_batch: nBatch,
        n_seq_max: nSeqMax,
        offload_kqv: useWebGPU,
      }),
    {
      modelUrl,
      useWebGPU: args.gpu === 'true',
      nCtx: Number(args['n-ctx'] ?? 8192),
      nBatch: Number(args['n-batch'] ?? 128),
      nSeqMax: Number(args['n-seq-max'] ?? 8),
      seed,
    }
  );
  const modelLoadMs = performance.now() - modelLoadStartedAt;
  const clientAttachStartedAt = performance.now();
  await Promise.all(
    pages.slice(1).map((page) =>
      page.evaluate(
        ({ modelUrl, useWebGPU, nCtx, nBatch, nSeqMax, seed }) =>
          window.wllmBrowserClient.loadModelFromUrl(modelUrl, {
            seed,
            n_ctx: nCtx,
            n_batch: nBatch,
            n_seq_max: nSeqMax,
            offload_kqv: useWebGPU,
          }),
        {
          modelUrl,
          useWebGPU: args.gpu === 'true',
          nCtx: Number(args['n-ctx'] ?? 8192),
          nBatch: Number(args['n-batch'] ?? 128),
          nSeqMax: Number(args['n-seq-max'] ?? 8),
          seed,
        }
      )
    )
  );
  const additionalClientAttachMs = performance.now() - clientAttachStartedAt;
  if (modelFile && args['keep-model-cache'] !== 'true') {
    await background.evaluate(() =>
      window.wllmBrowserClient.cacheManager.clear()
    );
  }
  const runtimeConfig = (
    await foreground.evaluate(() => window.wllmBrowserClient.status())
  ).runtimeConfig;
  if (
    args['require-webgpu'] === 'true' &&
    runtimeConfig?.usingWebGPU !== true
  ) {
    throw new Error(
      'WebGPU was required but the runtime did not activate the WebGPU backend'
    );
  }
  const tieredCache = {
    enabled: args['tiered-cache'] !== 'false',
    l1TokenCap: Number(args['l1-cap'] ?? 512),
    l2TokenCap: Number(args['l2-cap'] ?? 2048),
    l3TokenCap: Number(args['l3-cap'] ?? 8192),
    replacementPolicy: args.replacement ?? 'hybrid',
    admissionPolicy: args.admission ?? 'monotonic',
    admissionMaxRepairSteps: Number(args['repair-budget'] ?? 32),
  };
  const tierCaps = [
    tieredCache.l1TokenCap,
    tieredCache.l2TokenCap,
    tieredCache.l3TokenCap,
  ];
  await background.evaluate(
    ({ tieredCache, memoryCapBytes }) =>
      window.wllmBrowserClient.chatSessionInit(memoryCapBytes, tieredCache),
    { tieredCache, memoryCapBytes: memoryCapMb * 1024 * 1024 }
  );
  await foreground.evaluate(
    ({ tieredCache, memoryCapBytes }) =>
      window.wllmBrowserClient.chatSessionInit(memoryCapBytes, tieredCache),
    { tieredCache, memoryCapBytes: memoryCapMb * 1024 * 1024 }
  );
  for (let i = 0; i < warmupRuns; i++) {
    await foreground.evaluate(
      ({ index, nPredict }) =>
        window.wllmBrowserClient.chatFromNode(
          0,
          `Warm up the runtime (${index}).`,
          {
            nPredict,
            logicalSessionId: 'warmup',
          }
        ),
      { index: i, nPredict: warmupPredict }
    );
  }
  if (warmupRuns > 0) {
    await foreground.evaluate(() => window.wllmBrowserClient.chatReset());
  }
  await foreground.evaluate(() => window.wllmBrowserClient.getTelemetry(true));
  await foreground.evaluate(() => window.wllmClearExperimentEvents());

  const clientIds = {
    background: await background.evaluate(
      () => window.wllmBrowserClient.clientId
    ),
    foreground: await foreground.evaluate(
      () => window.wllmBrowserClient.clientId
    ),
    competitors: await Promise.all(
      extraPages.map((page) =>
        page.evaluate(() => window.wllmBrowserClient.clientId)
      )
    ),
  };
  const operationErrors = [];
  const traceResponses = [];
  const pressureNodeIds = [];
  const startUiMeasurement = async () => {
    if (uiWorkItems > 0) {
      await foreground.evaluate(
        (workItems) => window.wllmStartUiMeasurement(workItems),
        uiWorkItems
      );
    }
  };
  const stopUiMeasurement = async () => {
    if (uiWorkItems > 0) {
      await foreground.evaluate(() => window.wllmStopUiMeasurement());
    }
  };

  if (scenario === 'activate') {
    await startUiMeasurement();
    await markMeasurement(foreground, 'measurement_start', { scenario });
    await focusClient(background, foreground, clientIds.background);
    await demoteClient(foreground, foreground, clientIds.foreground);
    const backgroundRequest = background.evaluate(
      ({ nPredict, prompt }) => (
        (window.wllmBackgroundChunkCount = 0),
        window.wllmBrowserClient.chatFromNode(0, prompt, {
          nPredict,
          logicalSessionId: 'background',
          onChunk: () => {
            window.wllmBackgroundChunkCount += 1;
          },
        })
      ),
      {
        nPredict: backgroundPredict,
        prompt: makePrompt(
          'Continue this background analysis in detail',
          backgroundPromptRepeats
        ),
      }
    );
    await waitFor(
      () => background.evaluate(() => window.wllmBackgroundChunkCount ?? 0),
      (chunkCount) => chunkCount > 0
    );
    await demoteClientWithoutAck(background);
    await focusClientWithoutAck(foreground);
    const interaction = foreground.evaluate(
      async ({ samples, intervalMs }) => {
        const latencies = [];
        for (let i = 0; i < samples; i++) {
          latencies.push(await window.wllmMeasureInteraction());
          if (intervalMs > 0 && i + 1 < samples) {
            await new Promise((resolve) => setTimeout(resolve, intervalMs));
          }
        }
        return latencies;
      },
      { samples: interactionSamples, intervalMs: interactionIntervalMs }
    );
    const [foregroundRequest] = await Promise.all([
      foreground.evaluate(
        ({ nPredict, prompt }) =>
          window.wllmBrowserClient.chatFromNode(0, prompt, {
            nPredict,
            logicalSessionId: 'foreground',
          }),
        {
          nPredict: foregroundPredict,
          prompt: makePrompt(
            'Answer the newly foreground browser request now',
            foregroundPromptRepeats
          ),
        }
      ),
      interaction,
    ]);
    await Promise.all([backgroundRequest, foregroundRequest]);
    await stopUiMeasurement();
    await markMeasurement(foreground, 'measurement_end', { scenario });
  } else if (scenario === 'retain') {
    await markMeasurement(foreground, 'measurement_start', { scenario });
    const pressurePages = extraPages.length > 0 ? extraPages : [foreground];
    for (let i = 0; i < retentionDecoys; i++) {
      const decoyPage = pressurePages[i % pressurePages.length];
      const decoyClientId =
        extraPages.length > 0
          ? clientIds.competitors[i % pressurePages.length]
          : clientIds.foreground;
      await focusClient(decoyPage, foreground, decoyClientId);
      const decoyState = await decoyPage.evaluate(
        ({ index, nPredict, prompt }) =>
          window.wllmBrowserClient.chatFromNode(
            0,
            `${prompt} Initial decoy ${index}.`,
            {
              nPredict,
              logicalSessionId: `retention-decoy-${index}`,
            }
          ),
        {
          index: i,
          nPredict: pressurePredict,
          prompt: makePrompt(
            'Create an initially resident competing session',
            pressurePromptRepeats
          ),
        }
      );
      pressureNodeIds.push(decoyState.nodeId);
      await demoteClient(decoyPage, foreground, decoyClientId, 'frozen');
    }
    await focusClient(background, foreground, clientIds.background);
    await demoteClient(foreground, foreground, clientIds.foreground);
    const state = await background.evaluate(
      ({ nPredict, prompt }) =>
        window.wllmBrowserClient.chatFromNode(0, prompt, {
          nPredict,
          logicalSessionId: 'resume-target',
        }),
      {
        nPredict: sessionPredict,
        prompt: makePrompt(
          'Remember the facts and intermediate reasoning for this browser session',
          targetPromptRepeats
        ),
      }
    );
    await focusClient(foreground, foreground, clientIds.foreground);
    await demoteClient(
      background,
      foreground,
      clientIds.background,
      args['pause-state'] ?? 'hidden'
    );
    for (let i = 0; i < pressurePages.length; i++) {
      const pressureClientId =
        extraPages.length > 0 ? clientIds.competitors[i] : clientIds.foreground;
      await demoteClient(
        pressurePages[i],
        foreground,
        pressureClientId,
        'hidden'
      );
    }
    await new Promise((resolve) =>
      setTimeout(resolve, Number(args['pause-ms'] ?? 1000))
    );
    for (let i = 0; i < pressureSessions; i++) {
      const pressurePage = pressurePages[i % pressurePages.length];
      const pressureState = await pressurePage.evaluate(
        ({ index, nPredict, prompt }) =>
          window.wllmBrowserClient.chatFromNode(
            0,
            `${prompt} Session ${index}.`,
            {
              nPredict,
              logicalSessionId: 'pressure',
            }
          ),
        {
          index: i,
          nPredict: pressurePredict,
          prompt: makePrompt(
            'Create a competing browser-session state',
            pressurePromptRepeats
          ),
        }
      );
      pressureNodeIds.push(pressureState.nodeId);
      await demoteClient(
        pressurePage,
        foreground,
        extraPages.length > 0
          ? clientIds.competitors[i % pressurePages.length]
          : clientIds.foreground,
        'frozen'
      );
    }
    await demoteClient(foreground, foreground, clientIds.foreground);
    await focusClient(background, foreground, clientIds.background);
    await background.evaluate(
      ({ parentId, nPredict, prompt }) =>
        window.wllmBrowserClient.chatFromNode(parentId, prompt, {
          nPredict,
          logicalSessionId: 'resume-target',
        }),
      {
        parentId: state.nodeId,
        nPredict: sessionPredict,
        prompt: makePrompt(
          'Resume the task using the retained context',
          foregroundPromptRepeats
        ),
      }
    );
    await recordStateSnapshot(foreground, 'after-resume', tierCaps);
    await markMeasurement(foreground, 'measurement_end', { scenario });
    if (args['release-pressure'] !== 'false' && pressureNodeIds.length > 0) {
      await foreground.evaluate(
        (nodeIds) =>
          window.wllmBrowserClient.releaseSessionStates(
            nodeIds,
            'experiment_cleanup'
          ),
        pressureNodeIds
      );
    }
  } else if (scenario === 'admit') {
    await markMeasurement(foreground, 'measurement_start', { scenario });
    await focusClient(background, foreground, clientIds.background);
    await demoteClient(foreground, foreground, clientIds.foreground);
    const nodes = [];
    for (let i = 0; i < Number(args.sessions ?? 8); i++) {
      try {
        nodes.push(
          await background.evaluate(
            ({ index, nPredict, prompt }) =>
              window.wllmBrowserClient.chatFromNode(
                0,
                `${prompt} Session ${index}.`,
                {
                  nPredict,
                  logicalSessionId: `state-${index}`,
                }
              ),
            {
              index: i,
              nPredict: sessionPredict,
              prompt: makePrompt(
                'Create a resumable state under finite capacity',
                pressurePromptRepeats
              ),
            }
          )
        );
      } catch (error) {
        operationErrors.push(
          error instanceof Error ? error.message : String(error)
        );
        break;
      }
    }
    if (nodes.length > 0) {
      await demoteClient(background, foreground, clientIds.background);
      await focusClient(foreground, foreground, clientIds.foreground);
      try {
        await foreground.evaluate(
          ({ parentId, nPredict, prompt }) =>
            window.wllmBrowserClient.chatFromNode(parentId, prompt, {
              nPredict,
              logicalSessionId: 'admission-target',
            }),
          {
            parentId: nodes[0].nodeId,
            nPredict: foregroundPredict,
            prompt: makePrompt(
              'Resume the selected state under capacity pressure',
              foregroundPromptRepeats
            ),
          }
        );
      } catch (error) {
        operationErrors.push(
          error instanceof Error ? error.message : String(error)
        );
      }
    } else {
      operationErrors.push('no resumable state was admitted');
    }
    await recordStateSnapshot(foreground, 'after-admission', tierCaps);
    await markMeasurement(foreground, 'measurement_end', { scenario });
  } else if (scenario === 'handoff') {
    const competitors = [background, ...extraPages];
    await focusClient(foreground, foreground, clientIds.foreground);
    for (let i = 0; i < competitors.length; i++) {
      const clientId =
        i === 0 ? clientIds.background : clientIds.competitors[i - 1];
      await demoteClient(competitors[i], foreground, clientId);
    }
    const targetState = await foreground.evaluate(
      ({ nPredict, prompt }) =>
        window.wllmBrowserClient.chatFromNode(0, prompt, {
          nPredict,
          logicalSessionId: 'handoff-target',
        }),
      {
        nPredict: sessionPredict,
        prompt: makePrompt(
          'Build and retain the context for this interactive browser task',
          targetPromptRepeats
        ),
      }
    );
    await demoteClient(foreground, foreground, clientIds.foreground);
    await focusClient(background, foreground, clientIds.background);

    const competingStates = [];
    for (let i = 0; i < pressureSessions; i++) {
      const page = competitors[i % competitors.length];
      try {
        const state = await page.evaluate(
          ({ index, nPredict, prompt }) =>
            window.wllmBrowserClient.chatFromNode(
              0,
              `${prompt} Task ${index}.`,
              {
                nPredict,
                logicalSessionId: `handoff-pressure-${index}`,
              }
            ),
          {
            index: i,
            nPredict: pressurePredict,
            prompt: makePrompt(
              'Prepare a competing background browser task',
              pressurePromptRepeats
            ),
          }
        );
        competingStates.push({ pageIndex: i % competitors.length, ...state });
        pressureNodeIds.push(state.nodeId);
      } catch (error) {
        operationErrors.push(
          error instanceof Error ? error.message : String(error)
        );
      }
    }
    for (let i = 0; i < competitors.length; i++) {
      const clientId =
        i === 0 ? clientIds.background : clientIds.competitors[i - 1];
      await demoteClient(competitors[i], foreground, clientId, 'frozen');
    }

    await new Promise((resolve) =>
      setTimeout(resolve, Number(args['pause-ms'] ?? 5000))
    );
    await recordStateSnapshot(foreground, 'before-handoff', tierCaps);
    await startUiMeasurement();
    await markMeasurement(foreground, 'measurement_start', { scenario });
    const backgroundRequests = competitors.map((page, index) => {
      const parent =
        [...competingStates]
          .reverse()
          .find((state) => state.pageIndex === index)?.nodeId ?? 0;
      return page
        .evaluate(
          ({ parentId, nPredict, index, prompt }) => (
            (window.wllmBackgroundChunkCount = 0),
            window.wllmBrowserClient.chatFromNode(
              parentId,
              `${prompt} Task ${index}.`,
              {
                nPredict,
                logicalSessionId: `handoff-background-${index}`,
                onChunk: () => {
                  window.wllmBackgroundChunkCount += 1;
                },
              }
            )
          ),
          {
            parentId: parent,
            nPredict: backgroundPredict,
            index,
            prompt: makePrompt(
              'Continue this background browser task in detail',
              backgroundPromptRepeats
            ),
          }
        )
        .catch((error) => {
          operationErrors.push(
            error instanceof Error ? error.message : String(error)
          );
          return null;
        });
    });
    await waitFor(
      () => background.evaluate(() => window.wllmBackgroundChunkCount ?? 0),
      (chunkCount) => chunkCount > 0
    );
    await demoteClientWithoutAck(background);
    await focusClientWithoutAck(foreground);
    const interaction = foreground.evaluate(
      async ({ samples, intervalMs }) => {
        const latencies = [];
        for (let i = 0; i < samples; i++) {
          latencies.push(await window.wllmMeasureInteraction());
          if (intervalMs > 0 && i + 1 < samples) {
            await new Promise((resolve) => setTimeout(resolve, intervalMs));
          }
        }
        return latencies;
      },
      { samples: interactionSamples, intervalMs: interactionIntervalMs }
    );
    try {
      await Promise.all([
        foreground.evaluate(
          ({ parentId, nPredict, prompt }) =>
            window.wllmBrowserClient.chatFromNode(parentId, prompt, {
              nPredict,
              logicalSessionId: 'handoff-target',
            }),
          {
            parentId: targetState.nodeId,
            nPredict: foregroundPredict,
            prompt: makePrompt(
              'Resume this browser task and answer immediately',
              foregroundPromptRepeats
            ),
          }
        ),
        interaction,
      ]);
    } catch (error) {
      operationErrors.push(
        error instanceof Error ? error.message : String(error)
      );
    }
    await Promise.all(backgroundRequests);
    await recordStateSnapshot(foreground, 'after-handoff', tierCaps);
    await stopUiMeasurement();
    await markMeasurement(foreground, 'measurement_end', { scenario });
    if (args['release-pressure'] !== 'false' && pressureNodeIds.length > 0) {
      await background.evaluate(
        (nodeIds) =>
          window.wllmBrowserClient.releaseSessionStates(
            nodeIds,
            'experiment_cleanup'
          ),
        pressureNodeIds
      );
    }
  } else if (scenario === 'load') {
    const allPages = [foreground, background, ...extraPages];
    await focusClient(foreground, foreground, clientIds.foreground);
    for (let i = 1; i < allPages.length; i++) {
      const clientId =
        i === 1 ? clientIds.background : clientIds.competitors[i - 2];
      await demoteClient(allPages[i], foreground, clientId);
    }
    await startUiMeasurement();
    await markMeasurement(foreground, 'measurement_start', { scenario });
    const requests = [];
    for (let i = 0; i < requestCount; i++) {
      const isForeground = i % foregroundEvery === 0;
      const page = isForeground
        ? foreground
        : allPages[1 + ((i - 1) % Math.max(1, allPages.length - 1))];
      const promise = page
        .evaluate(
          ({ index, nPredict, prompt, isForeground }) =>
            window.wllmBrowserClient.chatFromNode(
              0,
              `${prompt} Request ${index}.`,
              {
                nPredict,
                logicalSessionId: `${isForeground ? 'foreground' : 'background'}-load-${index}`,
              }
            ),
          {
            index: i,
            nPredict:
              requestPredict ??
              (isForeground ? foregroundPredict : backgroundPredict),
            prompt: makePrompt(
              isForeground
                ? 'Serve this short foreground interaction'
                : 'Process this longer background browser task',
              isForeground ? foregroundPromptRepeats : backgroundPromptRepeats
            ),
            isForeground,
          }
        )
        .catch((error) => {
          operationErrors.push(
            error instanceof Error ? error.message : String(error)
          );
          return null;
        });
      requests.push(promise);
      if (arrivalGapMs > 0 && i + 1 < requestCount) {
        await new Promise((resolve) => setTimeout(resolve, arrivalGapMs));
      }
    }
    const interaction = foreground.evaluate(
      async ({ samples, intervalMs }) => {
        const latencies = [];
        for (let i = 0; i < samples; i++) {
          latencies.push(await window.wllmMeasureInteraction());
          if (intervalMs > 0 && i + 1 < samples) {
            await new Promise((resolve) => setTimeout(resolve, intervalMs));
          }
        }
        return latencies;
      },
      { samples: interactionSamples, intervalMs: interactionIntervalMs }
    );
    await Promise.all([...requests, interaction]);
    await recordStateSnapshot(foreground, 'after-load', tierCaps);
    await stopUiMeasurement();
    await markMeasurement(foreground, 'measurement_end', { scenario });
  } else if (scenario === 'trace') {
    if (!trace || !Array.isArray(trace.events)) {
      throw new Error('--trace-file is required for scenario=trace');
    }
    const tracePages = pages.slice(0, traceClientCount || pages.length);
    const sessionChains = new Map();
    const orderedTraceEvents = [...trace.events].sort(
      (left, right) => Number(left.atMs ?? 0) - Number(right.atMs ?? 0)
    );
    const initialLifecycleEvents = orderedTraceEvents.filter(
      (event) =>
        event.action === 'lifecycle' &&
        Number(event.atMs ?? 0) === 0 &&
        event.state !== 'active'
    );
    for (const event of initialLifecycleEvents) {
      const page = tracePages[Number(event.client ?? 0)];
      if (!page) {
        throw new Error(
          `trace references missing client index ${event.client}`
        );
      }
      await page.evaluate(
        ({ foreground, lifecycleState }) =>
          window.wllmBrowserClient.updateLifecycle({
            foreground,
            lifecycleState,
          }),
        { foreground: false, lifecycleState: event.state ?? 'hidden' }
      );
    }
    const preloadEvents = orderedTraceEvents.filter(
      (event) =>
        event.action === 'request' && event.phase === 'background-preload'
    );
    for (const event of preloadEvents) {
      const page = tracePages[Number(event.client ?? 0)];
      if (!page) {
        throw new Error(
          `trace references missing client index ${event.client}`
        );
      }
      const sessionId = String(event.session ?? `trace-${event.client}`);
      const key = `${event.client}\u0000${sessionId}`;
      try {
        const state = await page.evaluate(
          ({ prompt, nPredict, sessionId }) =>
            window.wllmBrowserClient.chatFromNode(0, prompt, {
              nPredict,
              logicalSessionId: sessionId,
            }),
          {
            prompt: String(event.prompt ?? ''),
            nPredict: Number(event.nPredict ?? 0),
            sessionId,
          }
        );
        sessionChains.set(key, Promise.resolve(state));
      } catch (error) {
        operationErrors.push(
          error instanceof Error ? error.message : String(error)
        );
        sessionChains.set(key, Promise.resolve(null));
      }
    }
    await recordStateSnapshot(foreground, 'before-trace', tierCaps);
    const pending = [];
    await startUiMeasurement();
    const traceStartedAt = performance.now();
    await markMeasurement(foreground, 'measurement_start', {
      scenario,
      traceId: trace.traceId ?? path.basename(traceFile),
    });
    const replayEvents = orderedTraceEvents.filter(
      (event) =>
        event.phase !== 'background-preload' &&
        !initialLifecycleEvents.includes(event)
    );
    const replayOriginMs = replayEvents.length
      ? Number(replayEvents[0].atMs ?? 0)
      : 0;
    for (const event of replayEvents) {
      const targetAt = Math.max(0, Number(event.atMs ?? 0) - replayOriginMs);
      const remainingMs = targetAt - (performance.now() - traceStartedAt);
      if (remainingMs > 0) {
        await new Promise((resolve) => setTimeout(resolve, remainingMs));
      }
      const page = tracePages[Number(event.client ?? 0)];
      if (!page) {
        throw new Error(
          `trace references missing client index ${event.client}`
        );
      }
      if (event.action === 'lifecycle') {
        const lifecycleState = event.state ?? 'hidden';
        const foregroundState = lifecycleState === 'active';
        if (foregroundState) {
          await page.bringToFront();
        }
        await page.evaluate(
          ({ foreground, lifecycleState }) =>
            window.wllmBrowserClient.updateLifecycle({
              foreground,
              lifecycleState,
            }),
          { foreground: foregroundState, lifecycleState }
        );
        continue;
      }
      if (event.action === 'interaction') {
        pending.push(
          page
            .evaluate(
              async ({ samples, intervalMs }) => {
                for (let i = 0; i < samples; i++) {
                  await window.wllmMeasureInteraction();
                  if (intervalMs > 0 && i + 1 < samples) {
                    await new Promise((resolve) =>
                      setTimeout(resolve, intervalMs)
                    );
                  }
                }
              },
              {
                samples: Number(event.samples ?? 1),
                intervalMs: Number(event.intervalMs ?? 0),
              }
            )
            .catch((error) => {
              operationErrors.push(
                error instanceof Error ? error.message : String(error)
              );
            })
        );
        continue;
      }
      if (event.action !== 'request') {
        throw new Error(`unknown trace action: ${event.action}`);
      }
      const sessionId = String(event.session ?? `trace-${event.client}`);
      const key = `${event.client}\u0000${sessionId}`;
      const predecessor =
        event.parent === 'root'
          ? Promise.resolve({ nodeId: 0 })
          : (sessionChains.get(key) ?? Promise.resolve({ nodeId: 0 }));
      const request = predecessor
        .then((parentState) =>
          page.evaluate(
            ({ parentId, prompt, nPredict, sessionId, telemetryTags }) =>
              window.wllmBrowserClient.chatFromNode(parentId, prompt, {
                nPredict,
                logicalSessionId: sessionId,
                telemetryTags,
              }),
            {
              parentId: parentState?.nodeId ?? 0,
              prompt: String(event.prompt ?? ''),
              nPredict: Number(event.nPredict ?? foregroundPredict),
              sessionId,
              telemetryTags: {
                traceId: trace.traceId ?? path.basename(traceFile),
                phase: event.phase ?? 'request',
                sourceTaskId: event.sourceTaskId ?? '',
                site: event.site ?? '',
                traceClient: Number(event.client ?? 0),
              },
            }
          )
        )
        .then((result) => {
          if (result && event.phase !== 'background-preload') {
            traceResponses.push({
              sourceTaskId: event.sourceTaskId ?? null,
              site: event.site ?? null,
              client: Number(event.client ?? 0),
              session: sessionId,
              phase: event.phase ?? 'request',
              assistantText: result.assistantText ?? '',
              evaluation: event.evaluation ?? null,
            });
          }
          return result;
        })
        .catch((error) => {
          operationErrors.push(
            error instanceof Error ? error.message : String(error)
          );
          return null;
        });
      sessionChains.set(key, request);
      pending.push(request);
    }
    await Promise.all(pending);
    await recordStateSnapshot(foreground, 'after-trace', tierCaps);
    await stopUiMeasurement();
    await markMeasurement(foreground, 'measurement_end', {
      scenario,
      traceId: trace.traceId ?? path.basename(traceFile),
    });
  } else {
    throw new Error(`unknown scenario ${scenario}`);
  }

  const finalTreeState = await foreground
    .evaluate(() => window.wllmBrowserClient.chatGetState())
    .catch(() => null);
  const rawEvents = await foreground.evaluate(() => [
    ...window.wllmExperimentEvents,
  ]);
  const events = rawEvents.map((event) => ({ runId, ...event }));
  const normalizeAnswer = (value) =>
    String(value ?? '')
      .normalize('NFKC')
      .toLowerCase()
      .replace(/[^\p{L}\p{N}.+-]+/gu, ' ')
      .trim();
  const scoreTraceResponse = (response) => {
    const answers =
      response.evaluation?.reference_answers ?? response.evaluation ?? {};
    const outputText = response.assistantText ?? '';
    if (answers.exact_match !== undefined) {
      return {
        supported: true,
        correct:
          normalizeAnswer(outputText) === normalizeAnswer(answers.exact_match),
        criterion: 'exact_match',
      };
    }
    if (Array.isArray(answers.must_include)) {
      return {
        supported: true,
        correct: answers.must_include.every((answer) =>
          normalizeAnswer(outputText).includes(normalizeAnswer(answer))
        ),
        criterion: 'must_include',
      };
    }
    return {
      supported: false,
      correct: false,
      criterion: 'official-evaluator-required',
    };
  };
  const scoredTraceResponses = traceResponses.map((response) => ({
    ...response,
    ...scoreTraceResponse(response),
  }));
  const supportedTraceResponses = scoredTraceResponses.filter(
    (response) => response.supported
  );
  const correctTraceResponses = supportedTraceResponses.filter(
    (response) => response.correct
  );
  const summary = {
    protocolVersion: PROTOCOL_VERSION,
    runId,
    scenario,
    gitRevision,
    llamaCppRevision,
    gitDirty,
    startedAt,
    deviceId,
    operator,
    trial,
    modelUrl,
    model: {
      source: modelFile ? path.basename(modelFile) : modelUrl,
      sizeBytes: modelFileStat?.size ?? null,
      sha256: modelSha256,
    },
    trace: trace
      ? {
          traceId: trace.traceId ?? path.basename(traceFile),
          source: path.basename(traceFile),
          sha256: traceSha256,
          eventCount: Array.isArray(trace.events) ? trace.events.length : 0,
        }
      : null,
    buildArtifacts: {
      wllamaJsSha256: wasmJsSha256,
      wllamaWasmSha256: wasmBinarySha256,
    },
    requestedExecutionMode: args.gpu === 'true' ? 'webgpu' : 'cpu',
    actualExecutionMode: runtimeConfig?.usingWebGPU === true ? 'webgpu' : 'cpu',
    browser: await browser.version(),
    environment: {
      operatingSystem: `${os.platform()} ${os.release()} ${os.arch()}`,
      nodeVersion: process.version,
      cpuModel: os.cpus()[0]?.model ?? 'unknown',
      logicalCpuCount: os.cpus().length,
      totalMemoryBytes: os.totalmem(),
      browserRuntime,
    },
    deviceMetadata,
    runtimeConfig,
    modelLoadMs,
    additionalClientAttachMs,
    clientCount,
    finalState: finalTreeState
      ? {
          nodeCount: finalTreeState.nodes.size,
          contextMemoryBytes: finalTreeState.contextMemoryBytes,
          memoryCapBytes: finalTreeState.memoryCapBytes,
          totalSnapshotTokenBytes: finalTreeState.totalSnapshotTokenBytes,
          tierStats: finalTreeState.tierStats,
        }
      : null,
    arguments: args,
    clientIds,
    tieredCache,
    operationErrors,
    traceScoredCount: supportedTraceResponses.length,
    traceCorrectCount: correctTraceResponses.length,
    traceAccuracy:
      supportedTraceResponses.length > 0
        ? correctTraceResponses.length / supportedTraceResponses.length
        : null,
    traceUnsupportedCount:
      scoredTraceResponses.length - supportedTraceResponses.length,
    ...summarize(events, clientIds, scenario),
  };
  const outputDir = path.join(outputRoot, runId);
  await mkdir(outputDir, { recursive: true });
  await writeFile(
    path.join(outputDir, 'events.jsonl'),
    `${events.map((event) => JSON.stringify(event)).join('\n')}\n`
  );
  await writeFile(
    path.join(outputDir, 'summary.json'),
    `${JSON.stringify(summary, null, 2)}\n`
  );
  if (scenario === 'trace') {
    await writeFile(
      path.join(outputDir, 'trace-responses.json'),
      `${JSON.stringify(scoredTraceResponses, null, 2)}\n`
    );
  }
  console.log(JSON.stringify(summary));

  await background.evaluate(() => window.wllmBrowserClient.exit());
  await foreground.evaluate(() => window.wllmBrowserClient.destroyService());
  const controlledAdmissionFailure =
    summary.requestFailureCount > 0 &&
    summary.admissionRejectCount >= summary.requestFailureCount &&
    summary.operationErrors.length === summary.requestFailureCount;
  if (summary.operationErrors.length > 0 && !controlledAdmissionFailure) {
    process.exitCode = 1;
  }
} finally {
  await browser?.close().catch(() => {});
  if (server.exitCode === null && server.signalCode === null) {
    server.kill('SIGTERM');
    await Promise.race([
      new Promise((resolve) => server.once('exit', resolve)),
      new Promise((resolve) => setTimeout(resolve, 2000)),
    ]);
  }
  await rm(scratchDir, { recursive: true, force: true });
}
