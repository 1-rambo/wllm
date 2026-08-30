import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createReadStream, existsSync } from 'node:fs';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';

const PROTOCOL_VERSION = 'wllm.browser-service.experiment.v3';
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
const requestedCases = new Set(
  String(args.case ?? '')
    .split(',')
    .map((value) => value.trim())
    .filter(Boolean)
);
const mode = args.mode ?? 'webgpu';
const defaultTrials = ['sensitivity', 'scaling', 'trace'].includes(suite)
  ? 5
  : suite === 'smoke'
    ? 1
    : 10;
const trials = Number(args.trials ?? defaultTrials);
const order = args.order ?? (suite === 'smoke' ? 'fixed' : 'rotate');
const cooldownMs = Number(
  args['cooldown-ms'] ?? (suite === 'smoke' ? 0 : 15000)
);
const tunedRetentionTtlMs = Number(args['tuned-retention-ttl-ms'] ?? 5000);
const deviceId = required('device-id');
const modelFile = required('model-file');
const wasmBuild = required('wasm-build');
const deviceFile = args['device-file'] ?? process.env.WLLM_DEVICE_FILE;
const traceFile = args['trace-file'] ?? process.env.WLLM_TRACE_FILE;
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

if (
  ![
    'smoke',
    'e2e',
    'core',
    'sensitivity',
    'scaling',
    'overhead',
    'trace',
  ].includes(suite)
) {
  throw new Error(`Unknown suite: ${suite}`);
}
if (!['all', 'e2e', 'activate', 'retain', 'admit', 'system'].includes(group)) {
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
if (suite === 'trace' && !traceFile) {
  throw new Error('--trace-file is required for the trace suite');
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
  'event-loop-yield': true,
  'foreground-interrupt': true,
  'warmup-runs': 1,
  'warmup-predict': 4,
  'interaction-samples': suite === 'smoke' ? 1 : 20,
  'interaction-interval-ms': suite === 'smoke' ? 0 : 20,
};

const fullHandoffArgs = {
  scenario: 'handoff',
  scheduler: 'foreground-aware',
  retention: 'lifecycle-aware',
  admission: 'monotonic',
  replacement: 'hybrid',
  slice: 32,
  'background-slice': 4,
  'handoff-grace': 16,
  yield: true,
  'repair-budget': 8,
  'client-count': 4,
  'pause-ms': 5000,
  'pressure-sessions': 8,
  'session-predict': 32,
  'pressure-predict': 16,
  'background-predict': 128,
  'foreground-predict': 32,
  'ui-work-items': 20000,
  'target-prompt-repeats': 16,
  'pressure-prompt-repeats': 12,
  'background-prompt-repeats': 8,
  'foreground-prompt-repeats': 2,
  'eviction-hint-top-k': 4,
  'require-protected-target': true,
  'n-ctx': 8192,
  'n-seq-max': 8,
  'l1-cap': 512,
  'l2-cap': 2048,
  'l3-cap': 8192,
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
      'eviction-hint-top-k': 4,
      'require-protected-target': true,
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
      retention: 'lru',
      admission: 'monotonic',
      'repair-budget': 8,
      sessions: 4,
      'session-predict': 4,
      'foreground-predict': 4,
      'n-ctx': 4096,
      'n-seq-max': 4,
      'l1-cap': 80,
      'l2-cap': 100,
      'l3-cap': 120,
      slice: 4,
    },
  },
  {
    group: 'admit',
    id: 'admit-reject-tight',
    args: {
      scenario: 'admit',
      retention: 'lru',
      admission: 'reject',
      'repair-budget': 8,
      sessions: 4,
      'session-predict': 4,
      'foreground-predict': 4,
      'n-ctx': 4096,
      'n-seq-max': 4,
      'l1-cap': 80,
      'l2-cap': 100,
      'l3-cap': 120,
      slice: 4,
    },
  },
];

const e2eCases = [
  {
    group: 'e2e',
    id: 'handoff-full',
    args: { ...fullHandoffArgs },
  },
  {
    group: 'e2e',
    id: 'handoff-shared-default',
    args: {
      ...fullHandoffArgs,
      scheduler: 'dual-queue',
      retention: 'lru',
      admission: 'reactive',
      replacement: 'lru',
      'require-protected-target': false,
    },
  },
  {
    group: 'e2e',
    id: 'handoff-cold-fcfs',
    args: {
      ...fullHandoffArgs,
      scheduler: 'fcfs-no-slice',
      retention: 'evict-on-pause',
      admission: 'reactive',
      replacement: 'lru',
      yield: false,
      'require-protected-target': false,
    },
  },
  {
    group: 'e2e',
    id: 'handoff-no-retain',
    args: {
      ...fullHandoffArgs,
      retention: 'lru',
      replacement: 'hybrid',
      'require-protected-target': false,
    },
  },
  {
    group: 'e2e',
    id: 'handoff-no-bounded-admit',
    args: {
      ...fullHandoffArgs,
      admission: 'reactive',
    },
  },
  {
    group: 'e2e',
    id: 'handoff-no-foreground-activate',
    args: {
      ...fullHandoffArgs,
      scheduler: 'dual-queue',
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
    id: 'activate-fixed-foreground-slice',
    args: {
      scenario: 'activate',
      scheduler: 'foreground-aware',
      slice: 32,
      'background-slice': 32,
      'handoff-grace': 16,
      yield: true,
      'event-loop-yield': true,
      'foreground-interrupt': false,
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
      'foreground-interrupt': false,
      'background-predict': 256,
      'foreground-predict': 32,
      'n-ctx': 8192,
      'n-seq-max': 8,
    },
  },
  {
    group: 'activate',
    id: 'activate-size-aware',
    args: {
      scenario: 'activate',
      scheduler: 'single-size-aware',
      slice: 32,
      'background-slice': 4,
      yield: true,
      'foreground-interrupt': false,
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
      'foreground-interrupt': false,
      'background-predict': 256,
      'foreground-predict': 32,
      'n-ctx': 8192,
      'n-seq-max': 8,
    },
  },
  {
    group: 'activate',
    id: 'activate-no-event-loop-yield',
    args: {
      scenario: 'activate',
      scheduler: 'foreground-aware',
      slice: 32,
      'background-slice': 4,
      yield: true,
      'event-loop-yield': false,
      'background-predict': 256,
      'foreground-predict': 32,
      'n-ctx': 8192,
      'n-seq-max': 8,
    },
  },
  {
    group: 'activate',
    id: 'activate-no-cooperative-slicing',
    args: {
      scenario: 'activate',
      scheduler: 'foreground-aware',
      slice: 32,
      'background-slice': 4,
      'handoff-grace': 16,
      yield: false,
      'event-loop-yield': false,
      'background-predict': 256,
      'foreground-predict': 32,
      'n-ctx': 8192,
      'n-seq-max': 8,
    },
  },
  {
    group: 'activate',
    id: 'activate-no-grace',
    args: {
      scenario: 'activate',
      scheduler: 'foreground-aware',
      slice: 32,
      'background-slice': 4,
      'handoff-grace': 0,
      yield: true,
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
      'pause-ms': 2000,
      'pause-state': 'passive',
      'client-count': 4,
      'retention-decoys': 1,
      'pressure-sessions': 12,
      'pressure-predict': 16,
      'release-pressure': true,
      'session-predict': 16,
      'target-prompt-repeats': 8,
      'pressure-prompt-repeats': 8,
      'eviction-hint-top-k': 4,
      'require-protected-target': true,
      'n-ctx': 8192,
      'n-seq-max': 8,
      'l1-cap': 512,
      'l2-cap': 2048,
      'l3-cap': 8192,
      replacement: 'hybrid',
      slice: 16,
    },
  },
  {
    group: 'retain',
    id: 'retain-hybrid-no-lifecycle',
    args: {
      scenario: 'retain',
      retention: 'lru',
      replacement: 'hybrid',
      'pause-ms': 2000,
      'pause-state': 'passive',
      'client-count': 4,
      'retention-decoys': 1,
      'pressure-sessions': 12,
      'pressure-predict': 16,
      'release-pressure': true,
      'session-predict': 16,
      'target-prompt-repeats': 8,
      'pressure-prompt-repeats': 8,
      'eviction-hint-top-k': 4,
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
      'pause-ms': 2000,
      'pause-state': 'passive',
      'client-count': 4,
      'retention-decoys': 1,
      'pressure-sessions': 12,
      'pressure-predict': 16,
      'release-pressure': true,
      'session-predict': 16,
      'target-prompt-repeats': 8,
      'pressure-prompt-repeats': 8,
      'eviction-hint-top-k': 4,
      'n-ctx': 8192,
      'n-seq-max': 8,
      'l1-cap': 512,
      'l2-cap': 2048,
      'l3-cap': 8192,
      replacement: 'lru',
      slice: 16,
    },
  },
  {
    group: 'retain',
    id: 'retain-ttl',
    args: {
      scenario: 'retain',
      retention: 'ttl',
      'retention-ttl': tunedRetentionTtlMs,
      'pause-ms': 2000,
      'pause-state': 'passive',
      'client-count': 4,
      'retention-decoys': 1,
      'pressure-sessions': 12,
      'pressure-predict': 16,
      'release-pressure': true,
      'session-predict': 16,
      'target-prompt-repeats': 8,
      'pressure-prompt-repeats': 8,
      'eviction-hint-top-k': 4,
      'n-ctx': 8192,
      'n-seq-max': 8,
      'l1-cap': 512,
      'l2-cap': 2048,
      'l3-cap': 8192,
      replacement: 'hybrid',
      slice: 16,
    },
  },
  {
    group: 'retain',
    id: 'retain-evict-on-pause',
    args: {
      scenario: 'retain',
      retention: 'evict-on-pause',
      'pause-ms': 2000,
      'pause-state': 'passive',
      'client-count': 4,
      'retention-decoys': 1,
      'pressure-sessions': 12,
      'pressure-predict': 16,
      'release-pressure': true,
      'session-predict': 16,
      'target-prompt-repeats': 8,
      'pressure-prompt-repeats': 8,
      'eviction-hint-top-k': 4,
      'n-ctx': 8192,
      'n-seq-max': 8,
      'l1-cap': 512,
      'l2-cap': 2048,
      'l3-cap': 8192,
      replacement: 'hybrid',
      slice: 16,
    },
  },
  {
    group: 'retain',
    id: 'retain-until-full',
    args: {
      scenario: 'retain',
      retention: 'retain-until-full',
      'pause-ms': 2000,
      'pause-state': 'passive',
      'client-count': 4,
      'retention-decoys': 1,
      'pressure-sessions': 12,
      'pressure-predict': 16,
      'release-pressure': true,
      'session-predict': 16,
      'target-prompt-repeats': 8,
      'pressure-prompt-repeats': 8,
      'eviction-hint-top-k': 4,
      'n-ctx': 8192,
      'n-seq-max': 8,
      'l1-cap': 512,
      'l2-cap': 2048,
      'l3-cap': 8192,
      replacement: 'hybrid',
      slice: 16,
    },
  },
  {
    group: 'admit',
    id: 'admit-monotonic',
    args: {
      scenario: 'admit',
      retention: 'lru',
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
      replacement: 'hybrid',
      slice: 16,
    },
  },
  {
    group: 'admit',
    id: 'admit-reactive',
    args: {
      scenario: 'admit',
      retention: 'lru',
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
      replacement: 'hybrid',
      slice: 16,
    },
  },
  {
    group: 'admit',
    id: 'admit-reactive-lru',
    args: {
      scenario: 'admit',
      retention: 'lru',
      admission: 'reactive',
      replacement: 'lru',
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
      retention: 'lru',
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
      replacement: 'hybrid',
      slice: 16,
    },
  },
];

const sensitivityCases = [
  ...[1, 2, 4].flatMap((protectionBudget) =>
    [1, 4, 8].map((repairBudget) => ({
      group: 'e2e',
      id: `handoff-contract-k-${protectionBudget}-b-${repairBudget}`,
      args: {
        ...fullHandoffArgs,
        'eviction-hint-top-k': protectionBudget,
        'repair-budget': repairBudget,
        'require-protected-target': false,
      },
    }))
  ),
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
      'pause-state': 'hidden',
      'client-count': 4,
      'retention-decoys': 1,
      'pressure-sessions': 12,
      'pressure-predict': 16,
      'release-pressure': true,
      'session-predict': 16,
      'target-prompt-repeats': 8,
      'pressure-prompt-repeats': 8,
      'eviction-hint-top-k': 4,
      'require-protected-target': pauseMs < 30000,
      'n-ctx': 8192,
      'n-seq-max': 8,
      'l1-cap': 512,
      'l2-cap': 2048,
      'l3-cap': 8192,
      replacement: 'hybrid',
      slice: 16,
    },
  })),
  ...[1000, 5000, 30000, 60000].map((ttlMs) => ({
    group: 'retain',
    id: `retain-ttl-${ttlMs}`,
    args: {
      scenario: 'retain',
      retention: 'ttl',
      'retention-ttl': ttlMs,
      'pause-ms': 2000,
      'pause-state': 'passive',
      'client-count': 4,
      'retention-decoys': 1,
      'pressure-sessions': 12,
      'pressure-predict': 16,
      'release-pressure': true,
      'session-predict': 16,
      'target-prompt-repeats': 8,
      'pressure-prompt-repeats': 8,
      'eviction-hint-top-k': 4,
      'require-protected-target': false,
      'n-ctx': 8192,
      'n-seq-max': 8,
      'l1-cap': 512,
      'l2-cap': 2048,
      'l3-cap': 8192,
      replacement: 'hybrid',
      slice: 16,
    },
  })),
  ...[1, 2, 4, 8, 16, 32].map((budget) => ({
    group: 'admit',
    id: `admit-repair-${budget}`,
    args: {
      scenario: 'admit',
      retention: 'lru',
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
      replacement: 'hybrid',
      slice: 16,
    },
  })),
  ...[4, 8, 12, 16].map((sessions) => ({
    group: 'admit',
    id: `admit-sessions-${sessions}`,
    args: {
      scenario: 'admit',
      retention: 'lru',
      admission: 'monotonic',
      replacement: 'hybrid',
      'repair-budget': 2,
      sessions,
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
  ...[128, 256, 512].map((capacity) => ({
    group: 'admit',
    id: `admit-capacity-${capacity}`,
    args: {
      scenario: 'admit',
      retention: 'lru',
      admission: 'monotonic',
      replacement: 'hybrid',
      'repair-budget': 2,
      sessions: 12,
      'session-predict': 16,
      'foreground-predict': 16,
      'n-ctx': 8192,
      'n-seq-max': 8,
      'l1-cap': capacity,
      'l2-cap': capacity,
      'l3-cap': capacity,
      slice: 16,
    },
  })),
  ...[4, 8, 16].map((targetPromptRepeats) => ({
    group: 'retain',
    id: `retain-prefix-${targetPromptRepeats}`,
    args: {
      scenario: 'retain',
      retention: 'lifecycle-aware',
      'retention-ttl': 30000,
      'pause-ms': 2000,
      'pause-state': 'passive',
      'client-count': 4,
      'retention-decoys': 1,
      'pressure-sessions': 12,
      'pressure-predict': 16,
      'release-pressure': true,
      'session-predict': 16,
      'target-prompt-repeats': targetPromptRepeats,
      'pressure-prompt-repeats': 8,
      'eviction-hint-top-k': 4,
      'require-protected-target': true,
      'n-ctx': 8192,
      'n-seq-max': 8,
      'l1-cap': 512,
      'l2-cap': 2048,
      'l3-cap': 8192,
      replacement: 'hybrid',
      slice: 16,
    },
  })),
  ...[64, 128, 256, 512].map((backgroundPredict) => ({
    group: 'activate',
    id: `activate-background-${backgroundPredict}`,
    args: {
      scenario: 'activate',
      scheduler: 'foreground-aware',
      slice: 32,
      'background-slice': 4,
      'handoff-grace': 16,
      yield: true,
      'background-predict': backgroundPredict,
      'foreground-predict': 32,
      'n-ctx': 8192,
      'n-seq-max': 8,
    },
  })),
];

const scalingCases = [2, 4, 8].flatMap((clients) => [
  {
    group: 'system',
    id: `load-full-clients-${clients}`,
    args: {
      scenario: 'load',
      scheduler: 'foreground-aware',
      retention: 'lifecycle-aware',
      admission: 'monotonic',
      'repair-budget': 8,
      'client-count': clients,
      requests: clients * 4,
      'foreground-every': 4,
      'arrival-gap-ms': 0,
      'background-predict': 128,
      'foreground-predict': 32,
      'background-prompt-repeats': 8,
      'foreground-prompt-repeats': 2,
      'n-ctx': 8192,
      'n-seq-max': 8,
      'l1-cap': 512,
      'l2-cap': 2048,
      'l3-cap': 8192,
      slice: 32,
      'background-slice': 4,
      'handoff-grace': 16,
      yield: true,
    },
  },
  {
    group: 'system',
    id: `load-fcfs-clients-${clients}`,
    args: {
      scenario: 'load',
      scheduler: 'fcfs-no-slice',
      retention: 'lru',
      admission: 'reactive',
      'repair-budget': 8,
      'client-count': clients,
      requests: clients * 4,
      'foreground-every': 4,
      'arrival-gap-ms': 0,
      'background-predict': 128,
      'foreground-predict': 32,
      'background-prompt-repeats': 8,
      'foreground-prompt-repeats': 2,
      'n-ctx': 8192,
      'n-seq-max': 8,
      'l1-cap': 512,
      'l2-cap': 2048,
      'l3-cap': 8192,
      slice: 32,
      yield: false,
    },
  },
]);

const overheadCases = [2, 4, 8].map((clients) => ({
  group: 'system',
  id: `attach-clients-${clients}`,
  args: {
    scenario: 'load',
    scheduler: 'foreground-aware',
    retention: 'lifecycle-aware',
    admission: 'monotonic',
    'repair-budget': 8,
    'client-count': clients,
    requests: 1,
    'foreground-every': 1,
    'background-predict': 1,
    'foreground-predict': 1,
    'background-prompt-repeats': 1,
    'foreground-prompt-repeats': 1,
    'interaction-samples': 1,
    'n-ctx': 8192,
    'n-seq-max': 8,
    'l1-cap': 512,
    'l2-cap': 2048,
    'l3-cap': 8192,
    slice: 32,
    'background-slice': 4,
    'handoff-grace': 16,
    yield: true,
  },
}));

const traceCases = [
  {
    group: 'e2e',
    id: 'trace-full',
    args: {
      scenario: 'trace',
      scheduler: 'foreground-aware',
      retention: 'lifecycle-aware',
      admission: 'monotonic',
      replacement: 'hybrid',
      'repair-budget': 8,
      slice: 32,
      'background-slice': 4,
      'handoff-grace': 16,
      yield: true,
      'n-ctx': 8192,
      'n-seq-max': 8,
      'l1-cap': 512,
      'l2-cap': 2048,
      'l3-cap': 8192,
    },
  },
  {
    group: 'e2e',
    id: 'trace-shared-default',
    args: {
      scenario: 'trace',
      scheduler: 'dual-queue',
      retention: 'lru',
      admission: 'reactive',
      replacement: 'lru',
      'repair-budget': 8,
      slice: 32,
      'background-slice': 4,
      yield: true,
      'n-ctx': 8192,
      'n-seq-max': 8,
      'l1-cap': 512,
      'l2-cap': 2048,
      'l3-cap': 8192,
    },
  },
  {
    group: 'e2e',
    id: 'trace-cold-fcfs',
    args: {
      scenario: 'trace',
      scheduler: 'fcfs-no-slice',
      retention: 'evict-on-pause',
      admission: 'reactive',
      replacement: 'lru',
      'repair-budget': 8,
      slice: 32,
      yield: false,
      'n-ctx': 8192,
      'n-seq-max': 8,
      'l1-cap': 512,
      'l2-cap': 2048,
      'l3-cap': 8192,
    },
  },
];

for (const item of [
  ...coreCases.filter((item) => item.group === 'activate'),
  ...sensitivityCases.filter((item) => item.group === 'activate'),
  ...scalingCases,
  ...traceCases,
]) {
  item.args['ui-work-items'] ??= 20000;
}

const suites = {
  smoke: smokeCases,
  e2e: e2eCases,
  core: coreCases,
  sensitivity: sensitivityCases,
  scaling: scalingCases,
  overhead: overheadCases,
  trace: traceCases,
};
const selectedCases = suites[suite].filter(
  (item) =>
    (group === 'all' || item.group === group) &&
    (requestedCases.size === 0 || requestedCases.has(item.id))
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
  tunedRetentionTtlMs,
  deviceId,
  operator,
  modelFile: path.resolve(modelFile),
  wasmBuild: path.resolve(wasmBuild),
  deviceFile: deviceFile ? path.resolve(deviceFile) : null,
  traceFile: traceFile ? path.resolve(traceFile) : null,
  modelSha256,
  wasmJsSha256,
  wasmSha256,
  cases: selectedCases,
  runs: [],
};
const manifestPath = path.join(output, 'matrix.json');
await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
if (args['dry-run'] === 'true') {
  console.log(
    JSON.stringify({
      output: path.resolve(output),
      suite,
      group,
      caseCount: selectedCases.length,
      totalRuns: selectedCases.length * trials,
      cases: selectedCases.map((item) => item.id),
    })
  );
  process.exit(0);
}

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
      ...(mode === 'cpu' ? { 'ui-work-items': 0 } : {}),
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
      ...(traceFile ? { 'trace-file': path.resolve(traceFile) } : {}),
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
