import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createReadStream, existsSync } from 'node:fs';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';

const PROTOCOL_VERSION = 'wllm.browser-service.experiment.v1';
const args = Object.fromEntries(
  process.argv.slice(2).map((arg) => {
    const [key, ...rest] = arg.replace(/^--/, '').split('=');
    return [key, rest.join('=') || 'true'];
  })
);

const required = (name) => {
  const value =
    args[name] ??
    process.env[`WLLM_${name.toUpperCase().replaceAll('-', '_')}`];
  if (!value) throw new Error(`Missing --${name}=...`);
  return value;
};

const suite = args.suite ?? 'smoke';
const group = args.group ?? 'all';
const mode = args.mode ?? 'webgpu';
const trials = Number(args.trials ?? (suite === 'smoke' ? 1 : 5));
const order = args.order ?? (suite === 'smoke' ? 'fixed' : 'rotate');
const cooldownMs = Number(
  args['cooldown-ms'] ?? (suite === 'smoke' ? 0 : 15000)
);
const deviceId = required('device-id');
const modelFile = required('model-file');
const wasmBuild = required('wasm-build');
const deviceFile = args['device-file'] ?? process.env.WLLM_DEVICE_FILE;
const operator = args.operator ?? process.env.WLLM_OPERATOR ?? 'unassigned';
const output =
  args.output ??
  path.join(
    'artifacts',
    'runs',
    new Date().toISOString().slice(0, 10),
    deviceId,
    suite
  );

if (!['smoke', 'core', 'sensitivity'].includes(suite)) {
  throw new Error(`Unknown suite: ${suite}`);
}
if (!['all', 'activate', 'retain', 'admit'].includes(group)) {
  throw new Error(`Unknown group: ${group}`);
}
if (!['cpu', 'webgpu'].includes(mode)) {
  throw new Error(`Unknown mode: ${mode}`);
}
if (suite !== 'smoke' && !deviceFile) {
  throw new Error(
    `--device-file is required for the ${suite} suite so hardware metadata is auditable`
  );
}
if (!Number.isInteger(trials) || trials < 1) {
  throw new Error('--trials must be a positive integer');
}
if (!['fixed', 'rotate'].includes(order)) {
  throw new Error('--order must be fixed or rotate');
}
if (!Number.isFinite(cooldownMs) || cooldownMs < 0) {
  throw new Error('--cooldown-ms must be a non-negative number');
}

const common = {
  seed: 42,
  'n-batch': 128,
  'memory-cap-mb': suite === 'smoke' ? 128 : 1024,
  'warmup-runs': 1,
  'warmup-predict': 4,
  'interaction-samples': suite === 'smoke' ? 1 : 20,
  'interaction-interval-ms': suite === 'smoke' ? 0 : 20,
};

const smokeCases = [
  {
    group: 'activate',
    id: 'activate-foreground-aware',
    args: {
      scenario: 'activate',
      scheduler: 'foreground-aware',
      slice: 8,
      'background-slice': 2,
      'background-predict': 64,
      'foreground-predict': 8,
      'n-ctx': 8192,
      'n-seq-max': 8,
    },
  },
  {
    group: 'activate',
    id: 'activate-dual-queue',
    args: {
      scenario: 'activate',
      scheduler: 'dual-queue',
      slice: 8,
      'background-slice': 2,
      'background-predict': 64,
      'foreground-predict': 8,
      'n-ctx': 8192,
      'n-seq-max': 8,
    },
  },
  {
    group: 'retain',
    id: 'retain-lifecycle-aware',
    args: {
      scenario: 'retain',
      retention: 'lifecycle-aware',
      slice: 8,
      'pause-ms': 1000,
      'session-predict': 8,
      'n-ctx': 8192,
      'n-seq-max': 8,
    },
  },
  {
    group: 'retain',
    id: 'retain-evict-on-pause',
    args: {
      scenario: 'retain',
      retention: 'evict-on-pause',
      slice: 8,
      'pause-ms': 1000,
      'session-predict': 8,
      'n-ctx': 8192,
      'n-seq-max': 8,
    },
  },
  {
    group: 'admit',
    id: 'admit-monotonic-tight',
    args: {
      scenario: 'admit',
      admission: 'monotonic',
      'repair-budget': 8,
      sessions: 4,
      'session-predict': 4,
      'foreground-predict': 4,
      'n-ctx': 4096,
      'n-seq-max': 4,
      'l1-cap': 40,
      'l2-cap': 50,
      'l3-cap': 60,
      slice: 4,
    },
  },
  {
    group: 'admit',
    id: 'admit-reject-tight',
    args: {
      scenario: 'admit',
      admission: 'reject',
      'repair-budget': 8,
      sessions: 4,
      'session-predict': 4,
      'foreground-predict': 4,
      'n-ctx': 4096,
      'n-seq-max': 4,
      'l1-cap': 40,
      'l2-cap': 50,
      'l3-cap': 60,
      slice: 4,
    },
  },
];

const coreCases = [
  {
    group: 'activate',
    id: 'activate-full',
    args: {
      scenario: 'activate',
      scheduler: 'foreground-aware',
      slice: 32,
      'background-slice': 4,
      'handoff-grace': 16,
      yield: true,
      'background-predict': 256,
      'foreground-predict': 32,
      'n-ctx': 8192,
      'n-seq-max': 8,
    },
  },
  {
    group: 'activate',
    id: 'activate-dual-queue',
    args: {
      scenario: 'activate',
      scheduler: 'dual-queue',
      slice: 32,
      'background-slice': 4,
      yield: true,
      'background-predict': 256,
      'foreground-predict': 32,
      'n-ctx': 8192,
      'n-seq-max': 8,
    },
  },
  {
    group: 'activate',
    id: 'activate-fcfs-no-slice',
    args: {
      scenario: 'activate',
      scheduler: 'fcfs-no-slice',
      yield: false,
      'background-predict': 256,
      'foreground-predict': 32,
      'n-ctx': 8192,
      'n-seq-max': 8,
    },
  },
  {
    group: 'activate',
    id: 'activate-no-yield',
    args: {
      scenario: 'activate',
      scheduler: 'foreground-aware',
      slice: 32,
      'background-slice': 4,
      yield: false,
      'background-predict': 256,
      'foreground-predict': 32,
      'n-ctx': 8192,
      'n-seq-max': 8,
    },
  },
  {
    group: 'retain',
    id: 'retain-lifecycle-aware',
    args: {
      scenario: 'retain',
      retention: 'lifecycle-aware',
      'retention-ttl': 30000,
      'pause-ms': 5000,
      'pressure-sessions': 12,
      'pressure-predict': 16,
      'release-pressure': true,
      'session-predict': 16,
      'n-ctx': 8192,
      'n-seq-max': 8,
      'l1-cap': 512,
      'l2-cap': 2048,
      'l3-cap': 8192,
      slice: 16,
    },
  },
  {
    group: 'retain',
    id: 'retain-lru',
    args: {
      scenario: 'retain',
      retention: 'lru',
      'pause-ms': 5000,
      'pressure-sessions': 12,
      'pressure-predict': 16,
      'release-pressure': true,
      'session-predict': 16,
      'n-ctx': 8192,
      'n-seq-max': 8,
      'l1-cap': 512,
      'l2-cap': 2048,
      'l3-cap': 8192,
      slice: 16,
    },
  },
  {
    group: 'retain',
    id: 'retain-ttl',
    args: {
      scenario: 'retain',
      retention: 'ttl',
      'retention-ttl': 30000,
      'pause-ms': 5000,
      'pressure-sessions': 12,
      'pressure-predict': 16,
      'release-pressure': true,
      'session-predict': 16,
      'n-ctx': 8192,
      'n-seq-max': 8,
      'l1-cap': 512,
      'l2-cap': 2048,
      'l3-cap': 8192,
      slice: 16,
    },
  },
  {
    group: 'retain',
    id: 'retain-evict-on-pause',
    args: {
      scenario: 'retain',
      retention: 'evict-on-pause',
      'pause-ms': 5000,
      'pressure-sessions': 12,
      'pressure-predict': 16,
      'release-pressure': true,
      'session-predict': 16,
      'n-ctx': 8192,
      'n-seq-max': 8,
      'l1-cap': 512,
      'l2-cap': 2048,
      'l3-cap': 8192,
      slice: 16,
    },
  },
  {
    group: 'retain',
    id: 'retain-until-full',
    args: {
      scenario: 'retain',
      retention: 'retain-until-full',
      'pause-ms': 5000,
      'pressure-sessions': 12,
      'pressure-predict': 16,
      'release-pressure': true,
      'session-predict': 16,
      'n-ctx': 8192,
      'n-seq-max': 8,
      'l1-cap': 512,
      'l2-cap': 2048,
      'l3-cap': 8192,
      slice: 16,
    },
  },
  {
    group: 'admit',
    id: 'admit-monotonic',
    args: {
      scenario: 'admit',
      admission: 'monotonic',
      'repair-budget': 2,
      sessions: 12,
      'session-predict': 16,
      'foreground-predict': 16,
      'n-ctx': 8192,
      'n-seq-max': 8,
      'l1-cap': 128,
      'l2-cap': 128,
      'l3-cap': 128,
      slice: 16,
    },
  },
  {
    group: 'admit',
    id: 'admit-reactive',
    args: {
      scenario: 'admit',
      admission: 'reactive',
      'repair-budget': 2,
      sessions: 12,
      'session-predict': 16,
      'foreground-predict': 16,
      'n-ctx': 8192,
      'n-seq-max': 8,
      'l1-cap': 128,
      'l2-cap': 128,
      'l3-cap': 128,
      slice: 16,
    },
  },
  {
    group: 'admit',
    id: 'admit-reject',
    args: {
      scenario: 'admit',
      admission: 'reject',
      'repair-budget': 2,
      sessions: 12,
      'session-predict': 16,
      'foreground-predict': 16,
      'n-ctx': 8192,
      'n-seq-max': 8,
      'l1-cap': 128,
      'l2-cap': 128,
      'l3-cap': 128,
      slice: 16,
    },
  },
];

const sensitivityCases = [
  ...[4, 8, 16, 32, 64].map((slice) => ({
    group: 'activate',
    id: `activate-slice-${slice}`,
    args: {
      scenario: 'activate',
      scheduler: 'foreground-aware',
      slice,
      'background-slice': Math.max(1, Math.floor(slice / 8)),
      'handoff-grace': 16,
      yield: true,
      'background-predict': 256,
      'foreground-predict': 32,
      'n-ctx': 8192,
      'n-seq-max': 8,
    },
  })),
  ...[0, 1000, 5000, 30000, 60000].map((pauseMs) => ({
    group: 'retain',
    id: `retain-pause-${pauseMs}`,
    args: {
      scenario: 'retain',
      retention: 'lifecycle-aware',
      'retention-ttl': 30000,
      'pause-ms': pauseMs,
      'pressure-sessions': 12,
      'pressure-predict': 16,
      'release-pressure': true,
      'session-predict': 16,
      'n-ctx': 8192,
      'n-seq-max': 8,
      'l1-cap': 512,
      'l2-cap': 2048,
      'l3-cap': 8192,
      slice: 16,
    },
  })),
  ...[1, 2, 4, 8, 16, 32].map((budget) => ({
    group: 'admit',
    id: `admit-repair-${budget}`,
    args: {
      scenario: 'admit',
      admission: 'monotonic',
      'repair-budget': budget,
      sessions: 12,
      'session-predict': 16,
      'foreground-predict': 16,
      'n-ctx': 8192,
      'n-seq-max': 8,
      'l1-cap': 128,
      'l2-cap': 128,
      'l3-cap': 128,
      slice: 16,
    },
  })),
];

const suites = {
  smoke: smokeCases,
  core: coreCases,
  sensitivity: sensitivityCases,
};
const selectedCases = suites[suite].filter(
  (item) => group === 'all' || item.group === group
);

if (selectedCases.length === 0) {
  throw new Error(`No cases selected for suite=${suite}, group=${group}`);
}

const stringifyArgs = (values) =>
  Object.entries(values).map(([key, value]) => `--${key}=${value}`);
const sha256File = (filePath) =>
  new Promise((resolve, reject) => {
    const hash = createHash('sha256');
    const input = createReadStream(filePath);
    input.on('error', reject);
    input.on('data', (chunk) => hash.update(chunk));
    input.on('end', () => resolve(hash.digest('hex')));
  });
const [modelSha256, wasmJsSha256, wasmSha256] = await Promise.all([
  sha256File(path.resolve(modelFile)),
  sha256File(path.resolve(wasmBuild, 'wllama.js')),
  sha256File(path.resolve(wasmBuild, 'wllama.wasm')),
]);
await mkdir(output, { recursive: true });
const manifest = {
  generatedAt: new Date().toISOString(),
  protocolVersion: PROTOCOL_VERSION,
  suite,
  group,
  mode,
  trials,
  order,
  cooldownMs,
  deviceId,
  operator,
  modelFile: path.resolve(modelFile),
  wasmBuild: path.resolve(wasmBuild),
  deviceFile: deviceFile ? path.resolve(deviceFile) : null,
  modelSha256,
  wasmJsSha256,
  wasmSha256,
  cases: selectedCases,
  runs: [],
};
const manifestPath = path.join(output, 'matrix.json');
await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);

let failed = 0;
let completed = 0;
const totalRuns = selectedCases.length * trials;
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
for (let trial = 1; trial <= trials; trial++) {
  const offset = order === 'rotate' ? (trial - 1) % selectedCases.length : 0;
  const orderedCases = [
    ...selectedCases.slice(offset),
    ...selectedCases.slice(0, offset),
  ];
  for (const item of orderedCases) {
    const runId = `${deviceId}__${mode}__${suite}__${item.id}__trial-${String(
      trial
    ).padStart(2, '0')}`;
    const runArgs = {
      ...common,
      ...item.args,
      suite,
      'case-id': item.id,
      'run-id': runId,
      'device-id': deviceId,
      operator,
      trial,
      'model-file': path.resolve(modelFile),
      'wasm-build': path.resolve(wasmBuild),
      'model-sha256': modelSha256,
      'wasm-js-sha256': wasmJsSha256,
      'wasm-sha256': wasmSha256,
      output: path.resolve(output),
      gpu: mode === 'webgpu',
      'require-webgpu': mode === 'webgpu',
      headless: mode === 'webgpu' ? false : (args.headless ?? true),
      'allow-dirty': args['allow-dirty'] ?? false,
      ...(deviceFile ? { 'device-file': path.resolve(deviceFile) } : {}),
    };
    const existingSummary = path.join(output, runId, 'summary.json');
    if (existsSync(existingSummary) && args.overwrite !== 'true') {
      throw new Error(
        `Run already exists: ${existingSummary}. Use a new output directory or pass --overwrite=true.`
      );
    }
    console.log(`\n===== ${runId} =====`);
    const startedAt = new Date().toISOString();
    const result = spawnSync(
      process.execPath,
      [
        path.resolve('examples', 'browser-service-smoke', 'run-experiment.mjs'),
        ...stringifyArgs(runArgs),
      ],
      {
        cwd: process.cwd(),
        stdio: 'inherit',
        env: process.env,
      }
    );
    const record = {
      runId,
      caseId: item.id,
      trial,
      startedAt,
      finishedAt: new Date().toISOString(),
      exitCode: result.status,
      signal: result.signal,
    };
    manifest.runs.push(record);
    if (result.status !== 0) failed++;
    completed++;
    await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
    if (cooldownMs > 0 && completed < totalRuns) {
      console.log(`Cooling down for ${cooldownMs} ms...`);
      await sleep(cooldownMs);
    }
  }
}

console.log(
  JSON.stringify({
    output: path.resolve(output),
    totalRuns: manifest.runs.length,
    failedRuns: failed,
  })
);
process.exitCode = failed > 0 ? 1 : 0;
