import { spawn } from 'node:child_process';
import { execFileSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { copyFile, mkdir, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { chromium } from 'playwright';

const args = Object.fromEntries(
  process.argv.slice(2).map((arg) => {
    const [key, ...rest] = arg.replace(/^--/, '').split('=');
    return [key, rest.join('=') || 'true'];
  })
);
const scenario = args.scenario ?? 'activate';
const wasmBuild = args['wasm-build'] ?? process.env.WLLM_WASM_BUILD_DIR;
const outputRoot = args.output ?? 'artifacts/runs';
const port = Number(args.port ?? 18080 + Math.floor(Math.random() * 1000));
const modelFile = args['model-file'] ?? process.env.WLLM_MODEL_FILE;
const modelUrl = modelFile
  ? `http://127.0.0.1:${port}/_tmp/browser-service-experiment/model.gguf`
  : (args.model ??
    process.env.WLLM_MODEL_URL ??
    'https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf');
const runId =
  args['run-id'] ??
  `${new Date().toISOString().replace(/[:.]/g, '-')}-${scenario}`;
const startedAt = new Date().toISOString();
const backgroundPredict = Number(args['background-predict'] ?? 64);
const foregroundPredict = Number(args['foreground-predict'] ?? 8);
const sessionPredict = Number(args['session-predict'] ?? 8);
const gitRevision = execFileSync('git', ['rev-parse', 'HEAD'], {
  encoding: 'utf8',
}).trim();

if (!wasmBuild) {
  throw new Error('Set --wasm-build=<source-built Wasm directory>');
}

const browserCandidates = [
  args.browser,
  process.env.CHROMIUM_EXECUTABLE_PATH,
  `${process.env.HOME}/.cache/ms-playwright/chromium-1148/chrome-linux/chrome`,
  `${process.env.HOME}/.cache/ms-playwright/chromium-1217/chrome-linux64/chrome`,
  `${process.env.HOME}/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome`,
].filter(Boolean);
const executablePath = browserCandidates.find((candidate) =>
  existsSync(candidate)
);
if (!executablePath) throw new Error('Chromium executable is unavailable');

await mkdir('_tmp/browser-service-experiment', { recursive: true });
await copyFile(
  path.join(wasmBuild, 'wllama.js'),
  '_tmp/browser-service-experiment/wllama.js'
);
await copyFile(
  path.join(wasmBuild, 'wllama.wasm'),
  '_tmp/browser-service-experiment/wllama.wasm'
);
if (modelFile) {
  await copyFile(modelFile, '_tmp/browser-service-experiment/model.gguf');
}

const server = spawn(process.execPath, ['scripts/http_server.js'], {
  cwd: new URL('../..', import.meta.url),
  env: { ...process.env, PORT: String(port) },
  stdio: ['ignore', 'pipe', 'inherit'],
});

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
  const released = events.filter((event) => event.type === 'state_release');
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
  return {
    eventCount: events.length,
    foregroundActivationMs:
      foregroundChange && firstDispatch
        ? firstDispatch.timestampMs - foregroundChange.timestampMs
        : null,
    foregroundFirstTokenMs:
      foregroundChange && firstToken
        ? firstToken.timestampMs - foregroundChange.timestampMs
        : null,
    sliceDurationsMs: events
      .filter((event) => event.type === 'slice_complete')
      .map((event) => Number(event.data?.durationMs ?? 0)),
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
    stateReleaseCount: released.length,
    releasedTokens: released.reduce(
      (sum, event) => sum + Number(event.data?.releasedTokens ?? 0),
      0
    ),
    stateRebuildCount: count('state_rebuild'),
    resumedSessionSource: resumedSession?.data?.restoreSource ?? null,
    requestFailureCount: count('request_fail'),
    inputToNextPaintP95Ms: quantile(paintLatencies, 0.95),
  };
};

try {
  await waitServer();
  const browser = await chromium.launch({
    executablePath,
    headless: args.headless !== 'false',
    args: ['--no-sandbox'],
  });
  const context = await browser.newContext();
  const background = await context.newPage();
  const foreground = await context.newPage();
  const base = new URL(
    `http://127.0.0.1:${port}/examples/browser-service-smoke/`
  );
  base.searchParams.set('module', '/_tmp/browser-service-experiment/wllama.js');
  base.searchParams.set('wasm', '/_tmp/browser-service-experiment/wllama.wasm');
  base.searchParams.set('scheduler', args.scheduler ?? 'foreground-aware');
  base.searchParams.set('slice', args.slice ?? '32');
  base.searchParams.set('backgroundSlice', args['background-slice'] ?? '4');
  base.searchParams.set('yield', args.yield ?? 'true');
  base.searchParams.set('retention', args.retention ?? 'lifecycle-aware');
  base.searchParams.set('retentionTtl', args['retention-ttl'] ?? '30000');
  await background.goto(base.href);
  await foreground.goto(base.href);
  await Promise.all([
    background.evaluate(() => window.wllmBrowserReady),
    foreground.evaluate(() => window.wllmBrowserReady),
  ]);

  await background.evaluate(
    ({ modelUrl, useWebGPU, nCtx, nBatch, nSeqMax }) =>
      window.wllmBrowserClient.loadModelFromUrl(modelUrl, {
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
    }
  );
  await foreground.evaluate(
    ({ modelUrl, useWebGPU, nCtx, nBatch, nSeqMax }) =>
      window.wllmBrowserClient.loadModelFromUrl(modelUrl, {
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
    }
  );
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
    (tieredCache) =>
      window.wllmBrowserClient.chatSessionInit(128 * 1024 * 1024, tieredCache),
    tieredCache
  );
  await foreground.evaluate(
    (tieredCache) =>
      window.wllmBrowserClient.chatSessionInit(128 * 1024 * 1024, tieredCache),
    tieredCache
  );
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

  if (scenario === 'activate') {
    await focusClient(background, foreground, clientIds.background);
    await demoteClient(foreground, foreground, clientIds.foreground);
    const backgroundRequest = background.evaluate(
      (nPredict) =>
        window.wllmBrowserClient.chatFromNode(
          0,
          'Continue this background analysis in detail.',
          {
            nPredict,
            logicalSessionId: 'background',
          }
        ),
      backgroundPredict
    );
    await waitFor(
      () => foreground.evaluate(() => window.wllmBrowserClient.getTelemetry()),
      (events) =>
        events.some(
          (event) =>
            event.type === 'slice_dispatch' &&
            event.clientId === clientIds.background
        )
    );
    await demoteClient(background, foreground, clientIds.background);
    await focusClient(foreground, foreground, clientIds.foreground);
    const [foregroundRequest] = await Promise.all([
      foreground.evaluate(
        (nPredict) =>
          window.wllmBrowserClient.chatFromNode(0, 'Respond now.', {
            nPredict,
            logicalSessionId: 'foreground',
          }),
        foregroundPredict
      ),
      foreground.evaluate(() => window.wllmMeasureInteraction()),
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
    runId,
    scenario,
    gitRevision,
    startedAt,
    modelUrl,
    browser: await browser.version(),
    operatingSystem: `${os.platform()} ${os.release()} ${os.arch()}`,
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
  await browser.close();
} finally {
  server.kill('SIGTERM');
}
