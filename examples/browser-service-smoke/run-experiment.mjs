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

const PROTOCOL_VERSION = 'wllm.browser-service.experiment.v1';
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
const pressurePredict = Number(args['pressure-predict'] ?? sessionPredict);
const seed = Number(args.seed ?? 42);
const memoryCapMb = Number(args['memory-cap-mb'] ?? 128);
const warmupRuns = Number(args['warmup-runs'] ?? 1);
const warmupPredict = Number(args['warmup-predict'] ?? 4);
const interactionSamples = Number(args['interaction-samples'] ?? 1);
const interactionIntervalMs = Number(args['interaction-interval-ms'] ?? 0);
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

const demoteClient = async (page, observer, clientId) => {
  const before = await observer.evaluate(() =>
    window.wllmBrowserClient.getTelemetry()
  );
  const beforeSequence = before.at(-1)?.sequence ?? 0;
  await page.evaluate(() =>
    window.wllmBrowserClient.updateLifecycle({
      foreground: false,
      lifecycleState: 'hidden',
    })
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

const quantile = (values, q) => {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[
    Math.min(sorted.length - 1, Math.floor(q * (sorted.length - 1)))
  ];
};

const summarize = (events, clientIds, scenarioName) => {
  const foregroundChange = [...events]
    .reverse()
    .find(
      (event) =>
        event.type === 'foreground_change' &&
        event.clientId === clientIds.foreground &&
        event.data?.foreground === true
    );
  const foregroundChangedAt = Number(
    foregroundChange?.data?.changedAt ?? foregroundChange?.wallTimeMs ?? 0
  );
  const firstDispatch = events.find(
    (event) =>
      event.type === 'slice_dispatch' &&
      event.clientId === clientIds.foreground &&
      (!foregroundChange || event.timestampMs >= foregroundChange.timestampMs)
  );
  const firstToken = events.find(
    (event) =>
      event.type === 'first_token' &&
      event.clientId === clientIds.foreground &&
      (!foregroundChange || event.timestampMs >= foregroundChange.timestampMs)
  );
  const admissions = events.filter((event) =>
    ['admission_commit', 'admission_fallback', 'admission_reject'].includes(
      event.type
    )
  );
  const count = (type) => events.filter((event) => event.type === type).length;
  const released = events.filter(
    (event) =>
      event.type === 'state_release' &&
      event.data?.reason !== 'experiment_cleanup'
  );
  const cleanupReleased = events.filter(
    (event) =>
      event.type === 'state_release' &&
      event.data?.reason === 'experiment_cleanup'
  );
  const resumedClientId =
    scenarioName === 'retain' ? clientIds.background : clientIds.foreground;
  const resumedSession = [...events]
    .reverse()
    .find(
      (event) =>
        (event.type === 'state_restore' || event.type === 'state_rebuild') &&
        event.clientId === resumedClientId &&
        typeof event.data?.restoreSource === 'string'
    );
  const paintLatencies = events
    .filter(
      (event) =>
        event.type === 'next_paint' && event.clientId === clientIds.foreground
    )
    .map((event) => Number(event.data?.latencyMs ?? 0));
  const sliceDurationsMs = events
    .filter((event) => event.type === 'slice_complete')
    .map((event) => Number(event.data?.durationMs ?? 0));
  const foregroundComplete = [...events]
    .reverse()
    .find(
      (event) =>
        event.type === 'request_complete' &&
        event.clientId === clientIds.foreground
    );
  const backgroundComplete = [...events]
    .reverse()
    .find(
      (event) =>
        event.type === 'request_complete' &&
        event.clientId === clientIds.background
    );
  const sessionResume =
    scenarioName === 'retain'
      ? [...events]
          .reverse()
          .find(
            (event) =>
              event.type === 'session_resume' &&
              event.clientId === clientIds.background
          )
      : undefined;
  const sessionResumedAt = Number(
    sessionResume?.data?.changedAt ?? sessionResume?.wallTimeMs ?? 0
  );
  const resumeDispatch = sessionResume
    ? events.find(
        (event) =>
          event.type === 'slice_dispatch' &&
          event.clientId === clientIds.background &&
          event.timestampMs >= sessionResume.timestampMs
      )
    : undefined;
  const resumeFirstToken = sessionResume
    ? events.find(
        (event) =>
          event.type === 'first_token' &&
          event.clientId === clientIds.background &&
          event.timestampMs >= sessionResume.timestampMs
      )
    : undefined;
  const completedRequests = count('request_complete');
  const failedRequests = count('request_fail');
  return {
    eventCount: events.length,
    foregroundActivationMs:
      foregroundChange && firstDispatch
        ? Number(firstDispatch.wallTimeMs) - foregroundChangedAt
        : null,
    foregroundFirstTokenMs:
      foregroundChange && firstToken
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
    admissionRepairStepsP95: quantile(
      admissions.map((event) => Number(event.data?.repairSteps ?? 0)),
      0.95
    ),
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
  const base = new URL(
    `http://127.0.0.1:${port}/examples/browser-service-smoke/`
  );
  base.searchParams.set('module', `${scratchUrl}/wllama.js`);
  base.searchParams.set('wasm', `${scratchUrl}/wllama.wasm`);
  base.searchParams.set('gpu', args.gpu === 'true' ? '1' : '0');
  base.searchParams.set('scheduler', args.scheduler ?? 'foreground-aware');
  base.searchParams.set('slice', args.slice ?? '32');
  base.searchParams.set('backgroundSlice', args['background-slice'] ?? '4');
  base.searchParams.set('handoffGrace', args['handoff-grace'] ?? '16');
  base.searchParams.set('yield', args.yield ?? 'true');
  base.searchParams.set('retention', args.retention ?? 'lifecycle-aware');
  base.searchParams.set('retentionTtl', args['retention-ttl'] ?? '30000');
  await background.goto(base.href);
  await foreground.goto(base.href);
  await Promise.all([
    background.evaluate(() => window.wllmBrowserReady),
    foreground.evaluate(() => window.wllmBrowserReady),
  ]);
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
  await foreground.evaluate(
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
    enabled: true,
    l1TokenCap: Number(args['l1-cap'] ?? 512),
    l2TokenCap: Number(args['l2-cap'] ?? 2048),
    l3TokenCap: Number(args['l3-cap'] ?? 8192),
    replacementPolicy: args.replacement ?? 'hybrid',
    admissionPolicy: args.admission ?? 'monotonic',
    admissionMaxRepairSteps: Number(args['repair-budget'] ?? 32),
  };
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

  const clientIds = {
    background: await background.evaluate(
      () => window.wllmBrowserClient.clientId
    ),
    foreground: await foreground.evaluate(
      () => window.wllmBrowserClient.clientId
    ),
  };
  const operationErrors = [];
  const pressureNodeIds = [];

  if (scenario === 'activate') {
    await focusClient(background, foreground, clientIds.background);
    await demoteClient(foreground, foreground, clientIds.foreground);
    const backgroundRequest = background.evaluate(
      (nPredict) => (
        (window.wllmBackgroundChunkCount = 0),
        window.wllmBrowserClient.chatFromNode(
          0,
          'Continue this background analysis in detail.',
          {
            nPredict,
            logicalSessionId: 'background',
            onChunk: () => {
              window.wllmBackgroundChunkCount += 1;
            },
          }
        )
      ),
      backgroundPredict
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
        (nPredict) =>
          window.wllmBrowserClient.chatFromNode(0, 'Respond now.', {
            nPredict,
            logicalSessionId: 'foreground',
          }),
        foregroundPredict
      ),
      interaction,
    ]);
    await Promise.all([backgroundRequest, foregroundRequest]);
  } else if (scenario === 'retain') {
    await focusClient(background, foreground, clientIds.background);
    await demoteClient(foreground, foreground, clientIds.foreground);
    const state = await background.evaluate(
      (nPredict) =>
        window.wllmBrowserClient.chatFromNode(0, 'Remember this session.', {
          nPredict,
          logicalSessionId: 'resume-target',
        }),
      sessionPredict
    );
    await demoteClient(background, foreground, clientIds.background);
    await focusClient(foreground, foreground, clientIds.foreground);
    for (let i = 0; i < pressureSessions; i++) {
      const pressureState = await foreground.evaluate(
        ({ index, nPredict }) =>
          window.wllmBrowserClient.chatFromNode(
            0,
            `Create competing foreground state ${index}.`,
            {
              nPredict,
              logicalSessionId: 'pressure',
            }
          ),
        { index: i, nPredict: pressurePredict }
      );
      pressureNodeIds.push(pressureState.nodeId);
    }
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
    await new Promise((resolve) =>
      setTimeout(resolve, Number(args['pause-ms'] ?? 1000))
    );
    await demoteClient(foreground, foreground, clientIds.foreground);
    await focusClient(background, foreground, clientIds.background);
    await background.evaluate(
      ({ parentId, nPredict }) =>
        window.wllmBrowserClient.chatFromNode(parentId, 'Resume it.', {
          nPredict,
          logicalSessionId: 'resume-target',
        }),
      { parentId: state.nodeId, nPredict: sessionPredict }
    );
  } else if (scenario === 'admit') {
    await focusClient(background, foreground, clientIds.background);
    await demoteClient(foreground, foreground, clientIds.foreground);
    const nodes = [];
    for (let i = 0; i < Number(args.sessions ?? 8); i++) {
      try {
        nodes.push(
          await background.evaluate(
            ({ index, nPredict }) =>
              window.wllmBrowserClient.chatFromNode(
                0,
                `Create resumable state ${index}.`,
                {
                  nPredict,
                  logicalSessionId: `state-${index}`,
                }
              ),
            { index: i, nPredict: sessionPredict }
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
          ({ parentId, nPredict }) =>
            window.wllmBrowserClient.chatFromNode(
              parentId,
              'Resume under pressure.',
              {
                nPredict,
                logicalSessionId: 'admission-target',
              }
            ),
          { parentId: nodes[0].nodeId, nPredict: foregroundPredict }
        );
      } catch (error) {
        operationErrors.push(
          error instanceof Error ? error.message : String(error)
        );
      }
    } else {
      operationErrors.push('no resumable state was admitted');
    }
  } else {
    throw new Error(`unknown scenario ${scenario}`);
  }

  const rawEvents = await foreground.evaluate(() =>
    window.wllmBrowserClient.getTelemetry()
  );
  const events = rawEvents.map((event) => ({ runId, ...event }));
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
    arguments: args,
    clientIds,
    tieredCache,
    operationErrors,
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
  console.log(JSON.stringify(summary));

  await background.evaluate(() => window.wllmBrowserClient.exit());
  await foreground.evaluate(() => window.wllmBrowserClient.destroyService());
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
