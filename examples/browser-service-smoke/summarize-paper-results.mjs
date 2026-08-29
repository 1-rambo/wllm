import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';

const args = Object.fromEntries(
  process.argv.slice(2).map((arg) => {
    const [key, ...rest] = arg.replace(/^--/, '').split('=');
    return [key, rest.join('=') || 'true'];
  })
);
const input = path.resolve(args.input ?? 'artifacts/runs/aggregate.csv');
const outputDir = path.resolve(
  args.output ?? path.join(path.dirname(input), 'paper-summary')
);

const parseCsv = (text) => {
  const rows = [];
  let row = [];
  let cell = '';
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (quoted) {
      if (char === '"' && text[i + 1] === '"') {
        cell += '"';
        i++;
      } else if (char === '"') {
        quoted = false;
      } else {
        cell += char;
      }
    } else if (char === '"') {
      quoted = true;
    } else if (char === ',') {
      row.push(cell);
      cell = '';
    } else if (char === '\n') {
      row.push(cell);
      if (row.some((value) => value.length > 0)) rows.push(row);
      row = [];
      cell = '';
    } else if (char !== '\r') {
      cell += char;
    }
  }
  if (cell.length > 0 || row.length > 0) {
    row.push(cell);
    rows.push(row);
  }
  const [header, ...data] = rows;
  return data.map((values) =>
    Object.fromEntries(header.map((column, index) => [column, values[index]]))
  );
};

const csv = (value) => {
  if (value === null || value === undefined) return '';
  const text = String(value);
  return `"${text.replaceAll('"', '""')}"`;
};

const quantile = (values, q) => {
  if (!values.length) return null;
  const sorted = [...values].sort((left, right) => left - right);
  const position = (sorted.length - 1) * q;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sorted[lower];
  const weight = position - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
};
let randomState = 42;
const random = () => {
  randomState = (1664525 * randomState + 1013904223) >>> 0;
  return randomState / 2 ** 32;
};
const bootstrapMedianCi = (values, iterations = 10000) => {
  if (!values.length) return { low: null, high: null };
  if (values.length === 1) return { low: values[0], high: values[0] };
  const medians = [];
  for (let iteration = 0; iteration < iterations; iteration++) {
    const sample = Array.from(
      { length: values.length },
      () => values[Math.floor(random() * values.length)]
    );
    medians.push(quantile(sample, 0.5));
  }
  return {
    low: quantile(medians, 0.025),
    high: quantile(medians, 0.975),
  };
};
const numberOrNull = (value) => {
  if (value === '' || value === null || value === undefined) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
};

const metricDirections = {
  foregroundActivationMs: 'lower',
  foregroundActivationP50Ms: 'lower',
  foregroundActivationP95Ms: 'lower',
  foregroundFirstTokenMs: 'lower',
  foregroundResumeTtftP50Ms: 'lower',
  foregroundResumeTtftP95Ms: 'lower',
  traceAccuracy: 'higher',
  foregroundTotalWallMs: 'lower',
  backgroundTotalWallMs: 'lower',
  inputToNextPaintP95Ms: 'lower',
  sessionResumeToDispatchMs: 'lower',
  sessionResumeToFirstTokenMs: 'lower',
  resumeRestoreMs: 'lower',
  resumeRebuildMs: 'lower',
  sliceDurationP95Ms: 'lower',
  sliceDurationMaxMs: 'lower',
  requestSuccessRate: 'higher',
  measuredRequestSuccessRate: 'higher',
  workloadMakespanMs: 'lower',
  aggregateThroughputTokensPerSecond: 'higher',
  requestWallTimeP50Ms: 'lower',
  requestWallTimeP95Ms: 'lower',
  requestTtftP50Ms: 'lower',
  requestTtftP95Ms: 'lower',
  requestQueueWaitP95Ms: 'lower',
  foregroundRequestWallTimeP95Ms: 'lower',
  backgroundRequestWallTimeP95Ms: 'lower',
  admissionRepairStepsP95: 'lower',
  admissionRepairStepsMax: 'lower',
  admissionBudgetViolationCount: 'lower',
  protectedVictimViolationCount: 'lower',
  cacheHintFailureCount: 'lower',
  admissionLatencyP95Ms: 'lower',
  admissionDemotedTokens: 'lower',
  admissionCheckpointedTokens: 'lower',
  admissionRemovedStates: 'lower',
  admissionRejectCount: 'lower',
  stateReleaseCount: 'lower',
  releasedTokens: 'lower',
  controlPlanePct: 'lower',
  schedulerDecisionP95Ms: 'lower',
  cacheHintP95Ms: 'lower',
  modelLoadMs: 'lower',
  additionalClientAttachMs: 'lower',
  peakSnapshotTokenBytes: 'lower',
  peakL1Tokens: 'lower',
  peakL2Tokens: 'lower',
  peakL3Tokens: 'lower',
  capacityViolationCount: 'lower',
  uiAverageFps: 'higher',
  uiFrameTimeP50Ms: 'lower',
  uiFrameTimeP95Ms: 'lower',
  uiFrameTimeP99Ms: 'lower',
  uiMaxFrameTimeMs: 'lower',
  uiLongFrameRatio: 'lower',
  finalDiskReads: 'lower',
  finalDiskWrites: 'lower',
  finalRestoreAttempts: 'lower',
  finalRestoreHitsL1: 'higher',
  finalRestoreHitsL2: 'higher',
  finalRestoreHitsL3: 'higher',
  finalRestoreMisses: 'lower',
  finalRestoreRebuilds: 'lower',
  overduePromotionCount: 'lower',
};
const metrics = Object.keys(metricDirections);

const rows = parseCsv(await readFile(input, 'utf8')).filter(
  (row) => row.valid === 'true'
);
const groups = new Map();
for (const row of rows) {
  const key = [row.deviceId, row.suite, row.caseId].join('\u0000');
  const bucket = groups.get(key) ?? [];
  bucket.push(row);
  groups.set(key, bucket);
}

const summaries = [];
for (const [key, groupRows] of groups) {
  const [deviceId, suite, caseId] = key.split('\u0000');
  for (const metric of metrics) {
    const values = groupRows
      .map((row) => numberOrNull(row[metric]))
      .filter((value) => value !== null);
    if (values.length === 0) continue;
    const confidence = bootstrapMedianCi(values);
    summaries.push({
      deviceId,
      suite,
      caseId,
      metric,
      direction: metricDirections[metric],
      n: values.length,
      median: quantile(values, 0.5),
      p25: quantile(values, 0.25),
      p75: quantile(values, 0.75),
      min: Math.min(...values),
      max: Math.max(...values),
      medianCi95Low: confidence.low,
      medianCi95High: confidence.high,
    });
  }
}

const summaryIndex = new Map(
  summaries.map((row) => [
    [row.deviceId, row.suite, row.caseId, row.metric].join('\u0000'),
    row,
  ])
);
const comparisonPairs = [
  ['e2e', 'handoff-full', 'handoff-shared-default'],
  ['e2e', 'handoff-full', 'handoff-cold-fcfs'],
  ['e2e', 'handoff-full', 'handoff-no-retain'],
  ['e2e', 'handoff-full', 'handoff-no-bounded-admit'],
  ['e2e', 'handoff-full', 'handoff-no-foreground-activate'],
  ['trace', 'trace-full', 'trace-shared-default'],
  ['trace', 'trace-full', 'trace-cold-fcfs'],
  ['core', 'activate-full', 'activate-dual-queue'],
  ['core', 'activate-full', 'activate-size-aware'],
  ['core', 'activate-full', 'activate-fcfs-no-slice'],
  ['core', 'activate-full', 'activate-no-yield'],
  ['core', 'activate-full', 'activate-no-grace'],
  ['core', 'retain-lifecycle-aware', 'retain-hybrid-no-lifecycle'],
  ['core', 'retain-lifecycle-aware', 'retain-lru'],
  ['core', 'retain-lifecycle-aware', 'retain-ttl'],
  ['core', 'retain-lifecycle-aware', 'retain-evict-on-pause'],
  ['core', 'retain-lifecycle-aware', 'retain-until-full'],
  ['core', 'admit-monotonic', 'admit-reactive'],
  ['core', 'admit-monotonic', 'admit-reactive-lru'],
  ['core', 'admit-monotonic', 'admit-reject'],
];
const devices = [...new Set(rows.map((row) => row.deviceId))];
const comparisons = [];
for (const deviceId of devices) {
  for (const [suite, method, baseline] of comparisonPairs) {
    for (const metric of metrics) {
      const left = summaryIndex.get(
        [deviceId, suite, method, metric].join('\u0000')
      );
      const right = summaryIndex.get(
        [deviceId, suite, baseline, metric].join('\u0000')
      );
      if (!left || !right) continue;
      const methodByTrial = new Map(
        groups
          .get([deviceId, suite, method].join('\u0000'))
          .map((row) => [row.trial, numberOrNull(row[metric])])
          .filter(([, value]) => value !== null)
      );
      const pairedDeltas = groups
        .get([deviceId, suite, baseline].join('\u0000'))
        .map((row) => {
          const methodValue = methodByTrial.get(row.trial);
          const baselineValue = numberOrNull(row[metric]);
          if (methodValue === undefined || baselineValue === null) return null;
          return methodValue - baselineValue;
        })
        .filter((value) => value !== null);
      const delta = pairedDeltas.length
        ? quantile(pairedDeltas, 0.5)
        : left.median - right.median;
      const pairedCi = bootstrapMedianCi(pairedDeltas);
      const improvement =
        right.median === 0 || left.median === null
          ? null
          : metricDirections[metric] === 'lower'
            ? ((right.median - left.median) / Math.abs(right.median)) * 100
            : ((left.median - right.median) / Math.abs(right.median)) * 100;
      comparisons.push({
        deviceId,
        suite,
        method,
        baseline,
        metric,
        direction: metricDirections[metric],
        methodMedian: left.median,
        baselineMedian: right.median,
        delta,
        improvementPct: improvement,
        pairedN: pairedDeltas.length,
        deltaCi95Low: pairedCi.low,
        deltaCi95High: pairedCi.high,
      });
    }
  }
}

const writeCsv = async (filePath, records) => {
  const columns = records.length ? Object.keys(records[0]) : [];
  const content = [
    columns.join(','),
    ...records.map((record) =>
      columns.map((column) => csv(record[column])).join(',')
    ),
  ].join('\n');
  await writeFile(filePath, `${content}\n`);
};

await import('node:fs/promises').then(({ mkdir }) =>
  mkdir(outputDir, { recursive: true })
);
await writeCsv(path.join(outputDir, 'paper-summary.csv'), summaries);
await writeCsv(path.join(outputDir, 'paired-comparisons.csv'), comparisons);
const tableSpecs = [
  {
    title: 'End-to-end handoff',
    suite: 'e2e',
    cases: ['handoff-cold-fcfs', 'handoff-shared-default', 'handoff-full'],
    metrics: [
      'foregroundResumeTtftP95Ms',
      'foregroundRequestWallTimeP95Ms',
      'inputToNextPaintP95Ms',
      'measuredRequestSuccessRate',
      'traceAccuracy',
      'aggregateThroughputTokensPerSecond',
    ],
  },
  {
    title: 'Retain',
    suite: 'core',
    cases: [
      'retain-lifecycle-aware',
      'retain-hybrid-no-lifecycle',
      'retain-lru',
      'retain-ttl',
      'retain-evict-on-pause',
      'retain-until-full',
    ],
    metrics: [
      'sessionResumeToFirstTokenMs',
      'resumeRestoreMs',
      'resumeRebuildMs',
      'peakSnapshotTokenBytes',
      'protectedVictimViolationCount',
    ],
  },
  {
    title: 'Admit',
    suite: 'core',
    cases: [
      'admit-monotonic',
      'admit-reactive',
      'admit-reactive-lru',
      'admit-reject',
    ],
    metrics: [
      'measuredRequestSuccessRate',
      'admissionLatencyP95Ms',
      'admissionRepairStepsP95',
      'admissionRepairStepsMax',
      'admissionRejectCount',
      'capacityViolationCount',
      'protectedVictimViolationCount',
    ],
  },
  {
    title: 'Activate',
    suite: 'core',
    cases: [
      'activate-full',
      'activate-dual-queue',
      'activate-size-aware',
      'activate-fcfs-no-slice',
      'activate-no-yield',
      'activate-no-grace',
    ],
    metrics: [
      'foregroundActivationMs',
      'foregroundFirstTokenMs',
      'uiFrameTimeP95Ms',
      'backgroundTotalWallMs',
      'aggregateThroughputTokensPerSecond',
    ],
  },
  {
    title: 'WebArena-derived trace replay',
    suite: 'trace',
    cases: ['trace-cold-fcfs', 'trace-shared-default', 'trace-full'],
    metrics: [
      'foregroundActivationP95Ms',
      'foregroundResumeTtftP95Ms',
      'requestWallTimeP95Ms',
      'measuredRequestSuccessRate',
      'aggregateThroughputTokensPerSecond',
    ],
  },
  {
    title: 'Scaling',
    suite: 'scaling',
    cases: [
      'load-full-clients-2',
      'load-fcfs-clients-2',
      'load-full-clients-4',
      'load-fcfs-clients-4',
      'load-full-clients-8',
      'load-fcfs-clients-8',
    ],
    metrics: [
      'workloadMakespanMs',
      'aggregateThroughputTokensPerSecond',
      'requestQueueWaitP95Ms',
      'foregroundRequestWallTimeP95Ms',
      'measuredRequestSuccessRate',
    ],
  },
  {
    title: 'Startup and control-plane overhead',
    suite: 'overhead',
    cases: ['attach-clients-2', 'attach-clients-4', 'attach-clients-8'],
    metrics: [
      'modelLoadMs',
      'additionalClientAttachMs',
      'controlPlanePct',
      'schedulerDecisionP95Ms',
      'cacheHintP95Ms',
    ],
  },
];
const tableLines = [
  '# Paper result tables',
  '',
  'Values are per-device medians. The full distribution and bootstrap intervals are in `paper-summary.csv`.',
  '',
];
for (const spec of tableSpecs) {
  tableLines.push(`## ${spec.title}`, '');
  tableLines.push(
    `| Device | Case | ${spec.metrics.join(' | ')} |`,
    `| --- | --- | ${spec.metrics.map(() => '---:').join(' | ')} |`
  );
  for (const deviceId of devices) {
    for (const caseId of spec.cases) {
      const values = spec.metrics.map((metric) => {
        const row = summaryIndex.get(
          [deviceId, spec.suite, caseId, metric].join('\u0000')
        );
        return row ? Number(row.median).toFixed(3) : 'TBD';
      });
      tableLines.push(`| ${deviceId} | ${caseId} | ${values.join(' | ')} |`);
    }
  }
  tableLines.push('');
}
await writeFile(
  path.join(outputDir, 'paper-tables.md'),
  `${tableLines.join('\n')}\n`
);
await writeFile(
  path.join(outputDir, 'paper-results.json'),
  `${JSON.stringify(
    {
      generatedAt: new Date().toISOString(),
      source: input,
      validRunCount: rows.length,
      summaries,
      comparisons,
    },
    null,
    2
  )}\n`
);
console.log(
  JSON.stringify({
    output: outputDir,
    validRunCount: rows.length,
    summaryRows: summaries.length,
    comparisonRows: comparisons.length,
    tables: path.join(outputDir, 'paper-tables.md'),
  })
);
