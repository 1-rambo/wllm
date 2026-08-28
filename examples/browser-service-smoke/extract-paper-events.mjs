import { mkdir, readdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';

const args = Object.fromEntries(
  process.argv.slice(2).map((arg) => {
    const [key, ...rest] = arg.replace(/^--/, '').split('=');
    return [key, rest.join('=') || 'true'];
  })
);
const input = path.resolve(args.input ?? 'artifacts/runs');
const outputDir = path.resolve(args.output ?? path.join(input, 'paper-events'));

const findEventFiles = async (directory) => {
  const found = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const target = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      found.push(...(await findEventFiles(target)));
    } else if (entry.name === 'events.jsonl') {
      found.push(target);
    }
  }
  return found;
};

const csv = (value) => {
  if (value === null || value === undefined) return '';
  const text =
    typeof value === 'object' ? JSON.stringify(value) : String(value);
  return `"${text.replaceAll('"', '""')}"`;
};

const writeCsv = async (fileName, columns, rows) => {
  const content = [
    columns.join(','),
    ...rows.map((row) => columns.map((column) => csv(row[column])).join(',')),
  ].join('\n');
  await writeFile(path.join(outputDir, fileName), `${content}\n`);
};

const requestRows = [];
const admissionRows = [];
const stateRows = [];
const uiRows = [];
const timelineRows = [];
for (const eventPath of (await findEventFiles(input)).sort()) {
  const summary = JSON.parse(
    await readFile(path.join(path.dirname(eventPath), 'summary.json'), 'utf8')
  );
  const events = (await readFile(eventPath, 'utf8'))
    .trim()
    .split('\n')
    .filter(Boolean)
    .map((line) => JSON.parse(line));
  const arrivals = new Map(
    events
      .filter((event) => event.type === 'request_arrive')
      .map((event) => [event.requestId, event])
  );
  for (const event of events) {
    const common = {
      runId: summary.runId,
      deviceId: summary.deviceId,
      suite: summary.arguments?.suite,
      caseId: summary.arguments?.['case-id'],
      trial: summary.trial,
      sequence: event.sequence,
      wallTimeMs: event.wallTimeMs,
      clientId: event.clientId,
      requestId: event.requestId,
    };
    if (event.type === 'request_complete' || event.type === 'request_fail') {
      const arrival = arrivals.get(event.requestId);
      requestRows.push({
        ...common,
        outcome: event.type === 'request_complete' ? 'complete' : 'fail',
        foreground: event.data?.foreground ?? arrival?.data?.foreground,
        lifecycleState:
          event.data?.lifecycleState ?? arrival?.data?.lifecycleState,
        site:
          event.data?.telemetryTags?.site ?? arrival?.data?.telemetryTags?.site,
        sourceTaskId:
          event.data?.telemetryTags?.sourceTaskId ??
          arrival?.data?.telemetryTags?.sourceTaskId,
        totalWallMs: event.data?.totalWallMs,
        ttftMs: event.data?.ttftMs,
        queueWaitMs: event.data?.queueWaitMs,
        serviceMs: event.data?.serviceMs,
        generatedTokens: event.data?.generatedTokens,
        sliceCount: event.data?.sliceCount,
        prefillMs: event.data?.phases?.prefillMs,
        decodeMs: event.data?.phases?.decodeMs,
        cacheLoadMs: event.data?.phases?.cacheLoadMs,
        rebuildMs: event.data?.phases?.rebuildMs,
        error: event.data?.message,
      });
    }
    if (
      event.type === 'admission_commit' ||
      event.type === 'admission_fallback' ||
      event.type === 'admission_reject'
    ) {
      admissionRows.push({
        ...common,
        outcome: event.type,
        admissionId: event.data?.id,
        targetLevel: event.data?.targetLevel,
        incomingTokens: event.data?.incomingTokens,
        initialTokenDeficit: event.data?.initialTokenDeficit,
        initialSlotDeficit: event.data?.initialSlotDeficit,
        repairSteps: event.data?.repairSteps,
        demotedTokens: event.data?.demotedTokens,
        checkpointedTokens: event.data?.checkpointedTokens,
        removedStates: event.data?.removedStates,
        latencyMs: event.data?.latencyMs,
        victimIds: event.data?.victimIds,
      });
    }
    if (event.type === 'state_snapshot') {
      stateRows.push({
        ...common,
        label: event.data?.label,
        totalSnapshotTokenBytes: event.data?.totalSnapshotTokenBytes,
        contextMemoryBytes: event.data?.contextMemoryBytes,
        memoryCapBytes: event.data?.memoryCapBytes,
        nodeCount: event.data?.nodeCount,
        l1Tokens: event.data?.l1Tokens,
        l2Tokens: event.data?.l2Tokens,
        l3Tokens: event.data?.l3Tokens,
        l1Slots: event.data?.l1Slots,
        l2Slots: event.data?.l2Slots,
        l3Slots: event.data?.l3Slots,
        diskReads: event.data?.diskReads,
        diskWrites: event.data?.diskWrites,
        capacityViolationCount: event.data?.capacityViolationCount,
      });
    }
    if (event.type === 'ui_measurement') {
      uiRows.push({ ...common, ...event.data });
    }
    if (
      [
        'measurement_start',
        'foreground_change',
        'session_pause',
        'session_resume',
        'request_arrive',
        'admission_commit',
        'admission_fallback',
        'admission_reject',
        'state_restore',
        'state_rebuild',
        'state_release',
        'state_snapshot',
        'slice_dispatch',
        'slice_complete',
        'first_token',
        'ui_measurement',
        'request_complete',
        'request_fail',
        'measurement_end',
      ].includes(event.type)
    ) {
      timelineRows.push({
        ...common,
        eventType: event.type,
        data: event.data,
      });
    }
  }
}

await mkdir(outputDir, { recursive: true });
await writeCsv(
  'request-level.csv',
  [
    'runId',
    'deviceId',
    'suite',
    'caseId',
    'trial',
    'sequence',
    'wallTimeMs',
    'clientId',
    'requestId',
    'outcome',
    'foreground',
    'lifecycleState',
    'site',
    'sourceTaskId',
    'totalWallMs',
    'ttftMs',
    'queueWaitMs',
    'serviceMs',
    'generatedTokens',
    'sliceCount',
    'prefillMs',
    'decodeMs',
    'cacheLoadMs',
    'rebuildMs',
    'error',
  ],
  requestRows
);
await writeCsv(
  'admission-level.csv',
  [
    'runId',
    'deviceId',
    'suite',
    'caseId',
    'trial',
    'sequence',
    'wallTimeMs',
    'clientId',
    'requestId',
    'outcome',
    'admissionId',
    'targetLevel',
    'incomingTokens',
    'initialTokenDeficit',
    'initialSlotDeficit',
    'repairSteps',
    'demotedTokens',
    'checkpointedTokens',
    'removedStates',
    'latencyMs',
    'victimIds',
  ],
  admissionRows
);
await writeCsv(
  'state-snapshots.csv',
  [
    'runId',
    'deviceId',
    'suite',
    'caseId',
    'trial',
    'sequence',
    'wallTimeMs',
    'label',
    'totalSnapshotTokenBytes',
    'contextMemoryBytes',
    'memoryCapBytes',
    'nodeCount',
    'l1Tokens',
    'l2Tokens',
    'l3Tokens',
    'l1Slots',
    'l2Slots',
    'l3Slots',
    'diskReads',
    'diskWrites',
    'capacityViolationCount',
  ],
  stateRows
);
await writeCsv(
  'ui-measurements.csv',
  [
    'runId',
    'deviceId',
    'suite',
    'caseId',
    'trial',
    'backend',
    'sampleCount',
    'averageFps',
    'frameTimeP50Ms',
    'frameTimeP95Ms',
    'frameTimeP99Ms',
    'maxFrameTimeMs',
    'longFrameRatio',
    'workItems',
  ],
  uiRows
);
await writeCsv(
  'timelines.csv',
  [
    'runId',
    'deviceId',
    'suite',
    'caseId',
    'trial',
    'sequence',
    'wallTimeMs',
    'clientId',
    'requestId',
    'eventType',
    'data',
  ],
  timelineRows
);
console.log(
  JSON.stringify({
    output: outputDir,
    requestRows: requestRows.length,
    admissionRows: admissionRows.length,
    stateRows: stateRows.length,
    uiRows: uiRows.length,
    timelineRows: timelineRows.length,
  })
);
