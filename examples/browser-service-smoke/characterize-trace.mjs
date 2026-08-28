import { readdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';

const args = Object.fromEntries(
  process.argv.slice(2).map((arg) => {
    const [key, ...rest] = arg.replace(/^--/, '').split('=');
    return [key, rest.join('=') || 'true'];
  })
);
const required = (name) => {
  if (!args[name]) throw new Error(`Missing --${name}=...`);
  return path.resolve(args[name]);
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
const describe = (values) => ({
  count: values.length,
  p50: quantile(values, 0.5),
  p95: quantile(values, 0.95),
  p99: quantile(values, 0.99),
  max: values.length ? Math.max(...values) : null,
});
const findEventFiles = async (directory) => {
  const files = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const target = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      files.push(...(await findEventFiles(target)));
    } else if (entry.name === 'events.jsonl') {
      files.push(target);
    }
  }
  return files;
};

const traceFile = required('trace');
const output = path.resolve(
  args.output ??
    path.join(path.dirname(traceFile), 'trace-characterization.json')
);
const trace = JSON.parse(await readFile(traceFile, 'utf8'));
const requests = trace.events.filter((event) => event.action === 'request');
const lifecycle = trace.events.filter((event) => event.action === 'lifecycle');
const interactions = trace.events.filter(
  (event) => event.action === 'interaction'
);
const sessionCounts = new Map();
const lastSessionIndex = new Map();
const resumedRequests = [];
const reuseDistanceRequests = [];
for (let index = 0; index < requests.length; index++) {
  const request = requests[index];
  const key = `${request.client}\u0000${request.session ?? 'default'}`;
  const seen = sessionCounts.get(key) ?? 0;
  if (seen > 0 && request.parent !== 'root') {
    resumedRequests.push(request);
    reuseDistanceRequests.push(index - lastSessionIndex.get(key) - 1);
  }
  sessionCounts.set(key, seen + 1);
  lastSessionIndex.set(key, index);
}
const pauseStartedAt = new Map();
const pauseDurationsMs = [];
for (const event of [...lifecycle].sort(
  (left, right) => Number(left.atMs) - Number(right.atMs)
)) {
  if (event.state === 'hidden' || event.state === 'frozen') {
    pauseStartedAt.set(event.client, Number(event.atMs));
  } else if (event.state === 'active' && pauseStartedAt.has(event.client)) {
    pauseDurationsMs.push(
      Math.max(0, Number(event.atMs) - pauseStartedAt.get(event.client))
    );
    pauseStartedAt.delete(event.client);
  }
}
const arrivals = requests
  .map((event) => Number(event.atMs))
  .sort((left, right) => left - right);
const interarrivalMs = arrivals
  .slice(1)
  .map((value, index) => value - arrivals[index]);

const runtime = {
  runCount: 0,
  foregroundSwitchCount: 0,
  foregroundSwitchWithBackgroundActiveCount: 0,
  restoreSources: {},
  admissionPressureCount: 0,
  admissionRejectCount: 0,
  capacityViolationCount: 0,
  peakTierUtilization: 0,
  foregroundResidualSliceMs: [],
  perSite: {},
};
if (args.events) {
  for (const file of await findEventFiles(path.resolve(args.events))) {
    const summary = JSON.parse(
      await readFile(path.join(path.dirname(file), 'summary.json'), 'utf8')
    );
    const events = (await readFile(file, 'utf8'))
      .trim()
      .split('\n')
      .filter(Boolean)
      .map((line) => JSON.parse(line));
    runtime.runCount += 1;
    const caps = [
      Number(summary.arguments?.['l1-cap'] ?? 0),
      Number(summary.arguments?.['l2-cap'] ?? 0),
      Number(summary.arguments?.['l3-cap'] ?? 0),
    ];
    const sliceIntervals = [];
    const openSlices = new Map();
    for (const event of events) {
      if (event.type === 'slice_dispatch') {
        openSlices.set(event.requestId, event);
      } else if (event.type === 'slice_complete') {
        const dispatch = openSlices.get(event.requestId);
        if (dispatch) {
          sliceIntervals.push({
            requestId: event.requestId,
            clientId: event.clientId,
            startSequence: dispatch.sequence,
            endSequence: event.sequence,
            endWallTimeMs: Number(event.wallTimeMs),
          });
          openSlices.delete(event.requestId);
        }
      }
    }
    const activeRequests = new Map();
    for (const event of events) {
      if (event.type === 'request_arrive') {
        activeRequests.set(event.requestId, {
          clientId: event.clientId,
          telemetryTags: event.data?.telemetryTags ?? {},
        });
      } else if (
        event.type === 'request_complete' ||
        event.type === 'request_fail'
      ) {
        const request = activeRequests.get(event.requestId);
        const site = request?.telemetryTags?.site;
        if (site) {
          const siteStats = runtime.perSite[site] ?? {
            completed: 0,
            failed: 0,
            totalWallMs: [],
            ttftMs: [],
          };
          if (event.type === 'request_complete') {
            siteStats.completed += 1;
            siteStats.totalWallMs.push(Number(event.data?.totalWallMs ?? 0));
            siteStats.ttftMs.push(Number(event.data?.ttftMs ?? 0));
          } else {
            siteStats.failed += 1;
          }
          runtime.perSite[site] = siteStats;
        }
        activeRequests.delete(event.requestId);
      } else if (
        event.type === 'foreground_change' &&
        event.data?.foreground === true
      ) {
        runtime.foregroundSwitchCount += 1;
        if (
          [...activeRequests.values()].some(
            (request) => request.clientId !== event.clientId
          )
        ) {
          runtime.foregroundSwitchWithBackgroundActiveCount += 1;
        }
        const activeSlice = sliceIntervals.find(
          (interval) =>
            interval.clientId !== event.clientId &&
            interval.startSequence < event.sequence &&
            interval.endSequence > event.sequence
        );
        if (activeSlice) {
          runtime.foregroundResidualSliceMs.push(
            Math.max(0, activeSlice.endWallTimeMs - Number(event.wallTimeMs))
          );
        }
      } else if (
        event.type === 'state_restore' ||
        event.type === 'state_rebuild'
      ) {
        if (typeof event.data?.restoreSource === 'string') {
          const source = event.data.restoreSource;
          runtime.restoreSources[source] =
            (runtime.restoreSources[source] ?? 0) + 1;
        }
      } else if (
        event.type === 'admission_fallback' ||
        event.type === 'admission_reject'
      ) {
        runtime.admissionPressureCount += 1;
        if (event.type === 'admission_reject') {
          runtime.admissionRejectCount += 1;
        }
      } else if (
        event.type === 'state_snapshot' &&
        Number(event.data?.capacityViolationCount ?? 0) > 0
      ) {
        runtime.capacityViolationCount += 1;
      }
      if (event.type === 'state_snapshot') {
        const usage = [
          Number(event.data?.l1Tokens ?? 0),
          Number(event.data?.l2Tokens ?? 0),
          Number(event.data?.l3Tokens ?? 0),
        ];
        for (let index = 0; index < usage.length; index++) {
          if (caps[index] > 0) {
            runtime.peakTierUtilization = Math.max(
              runtime.peakTierUtilization,
              usage[index] / caps[index]
            );
          }
        }
      }
    }
  }
}
for (const [site, stats] of Object.entries(runtime.perSite)) {
  runtime.perSite[site] = {
    completed: stats.completed,
    failed: stats.failed,
    latencyMs: describe(stats.totalWallMs),
    ttftMs: describe(stats.ttftMs),
  };
}

const report = {
  generatedAt: new Date().toISOString(),
  traceId: trace.traceId,
  source: trace.source,
  static: {
    clientCount: trace.clients.length,
    requestCount: requests.length,
    sessionCount: sessionCounts.size,
    lifecycleEventCount: lifecycle.length,
    interactionEventCount: interactions.length,
    resumeRequestCount: resumedRequests.length,
    resumeFraction:
      requests.length > 0 ? resumedRequests.length / requests.length : 0,
    reuseDistanceRequests: describe(reuseDistanceRequests),
    pauseDurationMs: describe(pauseDurationsMs),
    interarrivalMs: describe(interarrivalMs),
    durationMs: trace.events.length
      ? Math.max(...trace.events.map((event) => Number(event.atMs ?? 0)))
      : 0,
  },
  runtime: {
    ...runtime,
    foregroundOverlapFraction:
      runtime.foregroundSwitchCount > 0
        ? runtime.foregroundSwitchWithBackgroundActiveCount /
          runtime.foregroundSwitchCount
        : null,
    foregroundResidualSliceMs: describe(runtime.foregroundResidualSliceMs),
  },
};
await writeFile(output, `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify({ output, traceId: trace.traceId }));
