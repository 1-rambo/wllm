import { chromium } from 'playwright-core';
import { access, mkdir, writeFile } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { smokeModelUrl } from './model-config.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '..');
const timeoutMs = Number(process.env.SMOKE_TIMEOUT_MS ?? 240_000);
const demoUrl = process.env.DEMO_URL ?? 'http://localhost:5178/';
// `SMOKE_EXECUTION_MODE=deadline` runs the shipped lease instead of the
// bounded/unsliced controls, so the production path gets the same real-model,
// real-browser coverage the fixed shapes already have.
const executionMode = process.env.SMOKE_EXECUTION_MODE ?? null;
const outputRoot =
  process.env.SMOKE_OUT ?? path.join(root, 'artifacts', 'smoke');
const stamp = new Date().toISOString().replaceAll(':', '-');
const outputDir = path.join(outputRoot, stamp);

const chromePath = findChrome();
const summary = {
  kind: 'smoke-evidence',
  note: 'Smoke evidence only; not a controlled performance measurement.',
  demoUrl,
  chromePath,
  startedAt: new Date().toISOString(),
  scenarios: {},
  errors: [],
};

await assertServer(demoUrl);
await mkdir(outputDir, { recursive: true });

const browser = await chromium.launch({
  executablePath: chromePath,
  headless: process.env.HEADFUL !== '1',
  args: ['--no-sandbox', '--disable-dev-shm-usage'],
});

try {
  summary.scenarios.bounded = await runScenario(browser, {
    name: 'bounded',
    turns: 2,
    expectParking: true,
  });
  summary.scenarios.unsliced = await runScenario(browser, {
    name: 'unsliced',
    turns: 1,
    expectParking: false,
    unsliced: true,
  });
  summary.scenarios.priority = await runPriorityScenario(browser);
} finally {
  await browser.close();
}

const boundedUnits = summary.scenarios.bounded.lastMetrics.units;
const unslicedUnits = summary.scenarios.unsliced.lastMetrics.units;
const parked = summary.scenarios.bounded.trace.some(
  (line) => line.includes('-> L2') && line.includes('no host copy')
);
const nativeTokens = [
  summary.scenarios.bounded,
  summary.scenarios.unsliced,
].every((scenario) =>
  scenario.trace.some((line) => line.includes('native-token')) ||
  scenario.metricsHistory.every((line) => line.includes('tokens llama.cpp'))
);

const nativeBoundedStops = summary.scenarios.bounded.requests.filter(
  (entry) =>
    entry.boundedStops > 0 && (entry.boundaryStop ?? '').startsWith('native-')
).length;
const nativeTimeStops = summary.scenarios.bounded.requests.filter(
  (entry) => entry.boundaryStop === 'native-time-budget'
).length;
if (executionMode === null) {
  assert(
    boundedUnits > 1,
    `expected bounded execution to use >1 unit, got ${boundedUnits}`
  );
  assert(
    unslicedUnits === 1,
    `expected unsliced execution to use 1 unit, got ${unslicedUnits}`
  );
  assert(
    nativeBoundedStops > 0,
    'expected the bounded run to close at least one unit on a native budget'
  );
  assert(parked, 'expected bounded run to park a completed node in Wasm L2');
} else {
  // An engine mode owns the unit shape, so the fixed-shape expectations do
  // not apply. What has to hold is that the mode reached the coordinator and
  // that the lease, when the mode has one, shaped at least one unit.
  const leased = [
    summary.scenarios.bounded,
    summary.scenarios.unsliced,
  ];
  assert(
    leased.every((scenario) => scenario.runtime?.executionMode === executionMode),
    `expected executionMode=${executionMode} in both scenarios, got ${JSON.stringify(
      leased.map((scenario) => scenario.runtime?.executionMode ?? null)
    )}`
  );
  if (executionMode === 'deadline') {
    assert(
      leased.every((scenario) => scenario.runtime?.leaseSource !== null),
      'expected a live lease in both deadline scenarios'
    );
    assert(
      leased.some((scenario) => scenario.runtime?.leaseLastUnit !== null),
      'expected the lease to report the unit it shaped'
    );
  }
}
assert(nativeTokens, 'expected llama.cpp token accounting in both scenarios');
assert(
  summary.scenarios.priority.backgroundIncompleteAtForeground > 0,
  'expected foreground priority to leave background work pending'
);
assert(summary.errors.length === 0, `browser errors: ${summary.errors.join('; ')}`);

summary.finishedAt = new Date().toISOString();
summary.nativeBoundedStops = nativeBoundedStops;
summary.nativeTimeStops = nativeTimeStops;
summary.verdict = 'pass';
await writeFile(
  path.join(outputDir, 'summary.json'),
  `${JSON.stringify(summary, null, 2)}\n`
);

console.log(`SMOKE PASS ${outputDir}`);
console.log(
  `bounded units=${boundedUnits}, unsliced units=${unslicedUnits}, wasm-L2=${parked ? 'yes' : 'no'}`
);
console.log(
  `native bounded stops=${nativeBoundedStops} of ${summary.scenarios.bounded.requests.length} requests`
);
console.log(`native time-budget stops=${nativeTimeStops}`);
for (const scenario of Object.values(summary.scenarios)) {
  if (scenario.metricsHistory) {
    console.log(
      `${scenario.name}: ${scenario.metricsHistory.at(-1)} | frame P95=${scenario.frame.p95Ms.toFixed(1)} ms | marker=${scenario.frame.markerFrameMs?.toFixed(1) ?? 'n/a'} ms`
    );
  } else {
    console.log(
      `priority: foreground ${scenario.foregroundDurationMs} ms, background still pending at foreground completion=${scenario.backgroundIncompleteAtForeground}`
    );
  }
}

async function runScenario(browser, options) {
  const context = await browser.newContext({
    viewport: { width: 1440, height: 980 },
  });
  const page = await context.newPage();
  const errors = [];
  page.on('console', (message) => {
    if (message.type() === 'error') {
      errors.push(message.text());
    }
  });
  page.on('pageerror', (error) => errors.push(error.message));
  let closedByScript = false;
  page.on('crash', () => errors.push('page crashed'));
  page.on('close', () => {
    if (!closedByScript) {
      errors.push('page closed before the scenario read its results');
    }
  });
  page.on('requestfailed', (request) => {
    const failure = request.failure()?.errorText ?? 'request failed';
    const expectedAbort =
      failure.includes('ERR_ABORTED') || request.url().includes('favicon');
    if (!expectedAbort) {
      errors.push(`${request.url()} ${failure}`);
    }
  });

  await page.goto(scenarioUrl(options), { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(
    () => document.querySelector('#runtime-status')?.textContent === 'Ready',
    null,
    { timeout: timeoutMs }
  );

  const metricsHistory = [];
  let trace = [];
  for (let turn = 0; turn < options.turns; turn += 1) {
    const prompt =
      turn === 0
        ? 'Once upon a time there was a little girl named Lily'
        : 'She lived in a small village near the sea';
    await page.fill('#prompt', prompt);
    await page.fill('#max-tokens', '32');
    await page.click('#frame-reset');
    const before = await page.$$eval('.request-item', (items) => items.length);
    await page.click('#submit');
    await page.waitForFunction(
      (count) => document.querySelectorAll('.request-item').length > count,
      before
    );
    await page.waitForTimeout(40);
    await page.click('#frame-mark');
    await page.waitForFunction(
      (count) => {
        const items = [...document.querySelectorAll('.request-item')];
        return (
          items.length > count &&
          items.every((item) =>
            ['complete', 'cancelled', 'failed'].includes(
              item.dataset.state ?? ''
            )
          )
        );
      },
      before,
      { timeout: timeoutMs }
    );
    const metrics = await pageText(page, '.request-item .request-metrics');
    metricsHistory.push(metrics);
  }
  await page.waitForTimeout(700);
  trace = await page.$$eval('#trace li', (items) =>
    items.map((item) => item.textContent?.replace(/\s+/g, ' ').trim() ?? '')
  );
  const frame = await page.evaluate(() => window.__wllmFrameStats?.() ?? null);
  assert(frame && frame.frames > 0, `${options.name}: no animation samples`);
  const requests = await page.$$eval('.request-item', (items) =>
    items.map((item) => ({
      state: item.dataset.state,
      unitStop: item.dataset.unitStop ?? null,
      boundaryStop: item.dataset.boundaryStop ?? null,
      boundedStops: Number(item.dataset.boundedStops ?? 0),
      decodeBudget: Number(item.dataset.decodeBudget ?? 0),
      output: item.querySelector('.request-output')?.textContent ?? '',
      metrics: item.querySelector('.request-metrics')?.textContent ?? '',
    }))
  );
  const pageUrl = page.url();
  await page.screenshot({
    path: path.join(outputDir, `${options.name}.png`),
    fullPage: true,
  });

  // Read the coordinator diagnostics before the context closes. Reading them
  // afterwards throws "Target page, context or browser has been closed" and
  // hides the lease that shaped the units.
  const runtime = await page.evaluate(async () => {
    const diagnostics = await window.__wllmCoordinatorDiagnosticsProvider?.();
    const state = diagnostics?.runtime;
    if (!state) return null;
    return {
      executionMode: state.executionMode ?? null,
      leaseSource: state.lease?.source ?? null,
      leaseReason: state.lease?.reason ?? null,
      leaseBudgetMs: state.lease?.budgetMs ?? null,
      leaseLastUnit: state.lease?.lastUnit ?? null,
    };
  });
  closedByScript = true;
  await context.close();
  const lastMetrics = parseMetrics(metricsHistory.at(-1) ?? '');
  if (executionMode !== null) {
    // An engine mode owns the unit shape, so the fixed-shape expectations do
    // not apply. What has to hold here is that no request was stranded.
    const unfinished = requests.filter(
      (entry) =>
        !['complete', 'cancelled', 'failed'].includes(entry.state ?? '')
    );
    assert(
      unfinished.length === 0,
      `${options.name}: ${unfinished.length} request(s) never reached a terminal state`
    );
  } else if (options.expectParking) {
    assert(
      trace.some(
        (line) => line.includes('-> L2') && line.includes('no host copy')
      ),
      'bounded: missing native Wasm L2 park line'
    );
  }
  // An engine mode owns the unit shape, so the fixed-shape checks below are
  // controls-only. They used to run under deadline mode as well, which asked
  // the lease for an unsliced single unit it never promised.
  if (executionMode === null && options.unsliced) {
    assert(
      lastMetrics.units === 1,
      `unsliced: expected one unit, got ${lastMetrics.units}`
    );
    assert(
      requests.every((entry) => entry.unitStop === 'complete'),
      `unsliced: expected a clean completion, got ${JSON.stringify(
        requests.map((entry) => entry.unitStop)
      )}`
    );
    assert(
      requests.every((entry) => entry.boundaryStop === null),
      `unsliced: expected one unsliced unit, got boundaries ${JSON.stringify(
        requests.map((entry) => entry.boundaryStop)
      )}`
    );
    assert(
      requests.every((entry) => entry.boundedStops === 0),
      'unsliced: the native budget closed a unit in the unsliced control'
    );
  } else if (executionMode === null) {
    assert(
      lastMetrics.units > 1,
      `bounded: expected multiple units, got ${lastMetrics.units}`
    );
    const nativeStops = requests.filter((entry) =>
      [
        'native-decode-budget',
        'native-prefill-budget',
        'native-time-budget',
      ].includes(
        entry.boundaryStop ?? ''
      )
    );
    assert(
      nativeStops.length > 0,
      `bounded: no unit ended on a native bounded budget: ${JSON.stringify(
        requests.map((entry) => [entry.boundaryStop, entry.boundedStops])
      )}`
    );
    assert(
      requests.every((entry) => entry.boundedStops > 0),
      `bounded: requests without a native bounded stop: ${JSON.stringify(
        requests.map((entry) => entry.boundedStops)
      )}`
    );
    assert(
      requests.every((entry) => entry.boundaryStop !== 'host-abort'),
      'bounded: a unit was aborted by the host wall-clock guard instead of the native budget'
    );
  }
  assert(
    metricsHistory.every((line) => line.includes('tokens llama.cpp')),
    `${options.name}: token accounting fell back to stream chunks`
  );
  summary.errors.push(...errors.map((error) => `${options.name}: ${error}`));

  return {
    name: options.name,
    url: pageUrl,
    metricsHistory,
    lastMetrics,
    frame,
    requests,
    trace,
    runtime,
  };
}

function scenarioUrl(options) {
  const url = new URL(demoUrl);
  setDefault(url, 'modelUrl', smokeModelUrl);
  setDefault(url, 'nCtx', '2048');
  setDefault(url, 'nParallel', '4');
  setDefault(url, 'l1TokenCap', '192');
  setDefault(url, 'l2TokenCap', '512');
  setDefault(url, 'parkCompleted', '1');
  // These scenarios assert the slicing mechanics. The render-aware policy would
  // shrink even an unsliced run into several units under render pressure, so it
  // stays out of the way here and is exercised by the contention probe.
  setDefault(url, 'renderAware', '0');
  if (options.unsliced) {
    url.searchParams.set('unsliced', '1');
  } else {
    // Keep prompts larger than the native prefill chunk so the smoke test
    // exercises both prompt and decode unit boundaries.
    url.searchParams.set('maxPrefillTokens', '4');
  }
  if (options.priority) {
    // The priority scenario needs scheduling evidence, not the tiny chunks
    // used to prove that a native prefill boundary is reachable.
    url.searchParams.set('maxPrefillTokens', '32');
    url.searchParams.set('backgroundDecodeTokens', '4');
  }
  if (executionMode !== null) {
    // Applied last so the mode owns the shape: the deadline case runs the
    // production configuration instead of the fixed-shape assertions above.
    url.searchParams.set('executionMode', executionMode);
    if (executionMode === 'deadline') {
      url.searchParams.set('renderAware', '1');
      url.searchParams.set('unsliced', '0');
      url.searchParams.set('maxPrefillTokens', '256');
      url.searchParams.set('targetUnitWallTimeMs', '400');
      url.searchParams.set('maxUnitWallTimeMs', '400');
      url.searchParams.set('unitWallTimeGraceMs', '400');
      url.searchParams.set('adaptiveUnits', '0');
      url.searchParams.set('nativeTimeBudget', '1');
    }
  }
  return url.href;
}

function setDefault(url, key, value) {
  if (!url.searchParams.has(key)) {
    url.searchParams.set(key, value);
  }
}

function parseMetrics(value) {
  const units = Number(/units (\d+)/.exec(value)?.[1] ?? 0);
  const decode = /decode (\d+)\/(\d+)/.exec(value);
  const cached = /cached (\d+)/.exec(value);
  return {
    raw: value,
    units,
    decode: Number(decode?.[1] ?? 0),
    maxDecode: Number(decode?.[2] ?? 0),
    cached: Number(cached?.[1] ?? 0),
  };
}

async function pageText(page, selector) {
  return (await page.textContent(selector)) ?? '';
}

async function assertServer(url) {
  try {
    const response = await fetch(url, { method: 'GET' });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
  } catch (error) {
    throw new Error(
      `demo server is not reachable at ${url}; start it with "npm run demo". ${String(error)}`
    );
  }
}

function findChrome() {
  if (process.env.CHROME_PATH) {
    return process.env.CHROME_PATH;
  }
  const candidates = [
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Chromium.app/Contents/MacOS/Chromium',
    '/usr/bin/google-chrome',
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser',
  ];
  for (const candidate of candidates) {
    try {
      execFileSync(candidate, ['--version'], { stdio: 'ignore' });
      return candidate;
    } catch {
      // try the next known location
    }
  }
  throw new Error('Chrome not found; set CHROME_PATH to a Chromium executable');
}

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

async function runPriorityScenario(browser) {
  const context = await browser.newContext({
    viewport: { width: 1440, height: 980 },
  });
  const page = await context.newPage();
  const errors = [];
  page.on('console', (message) => {
    if (message.type() === 'error') {
      errors.push(message.text());
    }
  });
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto(scenarioUrl({ name: 'priority', priority: true }), {
    waitUntil: 'domcontentloaded',
  });
  await page.waitForFunction(
    () => document.querySelector('#runtime-status')?.textContent === 'Ready',
    null,
    { timeout: timeoutMs }
  );

  await page.click('#burst');
  await page.waitForFunction(
    () => document.querySelectorAll('.request-item').length >= 4,
    null,
    { timeout: timeoutMs }
  );
  await page.fill('#prompt', 'A foreground turn during background work');
  await page.fill('#max-tokens', '48');
  const sentAt = Date.now();
  await page.click('#submit');
  await page.waitForFunction(
    () => document.querySelectorAll('.request-item').length >= 5,
    null,
    { timeout: timeoutMs }
  );

  const foregroundPrompt = 'A foreground turn during background work';
  let foregroundAt = null;
  let backgroundStatesAtForeground = [];
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const items = await page.$$eval('.request-item', (nodes) =>
      nodes.map((node) => ({
        prompt: node.querySelector('.request-prompt')?.textContent ?? '',
        state: node.dataset.state ?? '',
      }))
    );
    const foreground = items.find((item) => item.prompt === foregroundPrompt);
    if (foreground?.state === 'complete') {
      foregroundAt = Date.now();
      backgroundStatesAtForeground = items
        .filter((item) => item.prompt !== foregroundPrompt)
        .map((item) => item.state);
      break;
    }
    await page.waitForTimeout(25);
  }

  assert(foregroundAt, 'priority: foreground request did not complete');
  assert(
    backgroundStatesAtForeground.length >= 4,
    'priority: background burst was not observed at foreground completion'
  );
  const backgroundIncompleteAtForeground =
    backgroundStatesAtForeground.filter(
      (state) => !['complete', 'cancelled', 'failed'].includes(state)
    ).length;
  const trace = await page.$$eval('#trace li', (items) =>
    items.map((item) => item.textContent?.replace(/\s+/g, ' ').trim() ?? '')
  );
  const pageUrl = page.url();
  await page.screenshot({
    path: path.join(outputDir, 'priority.png'),
    fullPage: true,
  });
  await context.close();
  summary.errors.push(...errors.map((error) => `priority: ${error}`));
  assert(
    backgroundIncompleteAtForeground > 0,
    'priority: background work finished before foreground'
  );
  return {
    name: 'priority',
    url: pageUrl,
    foregroundDurationMs: foregroundAt - sentAt,
    backgroundIncompleteAtForeground,
    backgroundStatesAtForeground,
    trace,
  };
}
