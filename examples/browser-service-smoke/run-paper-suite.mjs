import { spawnSync } from 'node:child_process';
import path from 'node:path';

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
const profile = args.profile ?? 'primary';
const primarySuites = [
  'e2e',
  'core',
  'sensitivity',
  'scaling',
  'overhead',
  'trace',
];
const schedules = {
  primary: primarySuites.map((suite) => ({ suite, cases: null })),
  replication: [
    {
      suite: 'e2e',
      cases: ['handoff-full', 'handoff-shared-default', 'handoff-cold-fcfs'],
    },
    {
      suite: 'core',
      cases: [
        'retain-lifecycle-aware',
        'retain-lru',
        'admit-monotonic',
        'admit-reactive',
        'activate-full',
        'activate-fixed-foreground-slice',
        'activate-dual-queue',
        'activate-no-event-loop-yield',
      ],
    },
  ],
};
if (!Object.hasOwn(schedules, profile)) {
  throw new Error('--profile must be primary or replication');
}
const requestedSuites = args.suites
  ? String(args.suites)
      .split(',')
      .map((value) => value.trim())
      .filter(Boolean)
  : null;
const schedule = requestedSuites
  ? requestedSuites.map((suite) => ({ suite, cases: null }))
  : schedules[profile];
const suites = schedule.map((item) => item.suite);
const allowed = new Set([
  'smoke',
  'e2e',
  'core',
  'sensitivity',
  'scaling',
  'overhead',
  'trace',
]);
for (const { suite, cases } of schedule) {
  if (!allowed.has(suite)) throw new Error(`Unknown suite: ${suite}`);
}
if (schedule.some(({ suite }) => suite === 'trace') && !args['trace-file']) {
  throw new Error(
    'The selected paper profile includes trace replay; pass --trace-file=...'
  );
}
const deviceId = required('device-id');
const outputRoot = path.resolve(
  args.output ??
    path.join(
      'artifacts',
      'runs',
      new Date().toISOString().slice(0, 10),
      deviceId
    )
);
const forward = [
  'mode',
  'device-id',
  'operator',
  'device-file',
  'model-file',
  'wasm-build',
  'order',
  'cooldown-ms',
  'tuned-retention-ttl-ms',
  'event-loop-yield',
  'foreground-interrupt',
  'allow-dirty',
  'headless',
  'overwrite',
  'trace-file',
  'dry-run',
];
let failed = 0;
for (const { suite, cases } of schedule) {
  const commandArgs = [
    path.resolve('examples/browser-service-smoke/run-matrix.mjs'),
    `--suite=${suite}`,
    '--group=all',
    `--output=${path.join(outputRoot, suite)}`,
    ...(cases ? [`--case=${cases.join(',')}`] : []),
    ...forward
      .filter((key) => args[key] !== undefined)
      .map((key) => `--${key}=${args[key]}`),
  ];
  if (args.trials !== undefined) {
    commandArgs.push(`--trials=${args.trials}`);
  } else if (profile === 'replication') {
    commandArgs.push('--trials=5');
  }
  console.log(`\n######## PAPER SUITE: ${suite} ########`);
  const result = spawnSync(process.execPath, commandArgs, {
    cwd: process.cwd(),
    stdio: 'inherit',
    env: process.env,
  });
  if (result.status !== 0) {
    failed += 1;
    if (args['continue-on-error'] !== 'true') break;
  }
}
console.log(
  JSON.stringify({
    output: outputRoot,
    profile,
    requestedSuites: suites,
    failedSuites: failed,
  })
);
process.exitCode = failed > 0 ? 1 : 0;
