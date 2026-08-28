import { readdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';

const args = Object.fromEntries(
  process.argv.slice(2).map((arg) => {
    const [key, ...rest] = arg.replace(/^--/, '').split('=');
    return [key, rest.join('=') || 'true'];
  })
);
const input = path.resolve(args.input ?? 'artifacts/runs');
const output = path.resolve(args.output ?? path.join(input, 'aggregate.csv'));
const expected = {
  protocolVersion:
    args['expected-protocol'] ?? 'wllm.browser-service.experiment.v2',
  gitRevision: args['expected-git'],
  modelSha256: args['expected-model-sha256'],
  wasmSha256: args['expected-wasm-sha256'],
  traceSha256: args['expected-trace-sha256'],
};

const findSummaries = async (directory) => {
  const found = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const target = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      found.push(...(await findSummaries(target)));
    } else if (entry.name === 'summary.json') {
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

const columns = [
  'valid',
  'invalidReasons',
  'protocolVersion',
  'runId',
  'deviceId',
  'operator',
  'trial',
  'gitRevision',
  'llamaCppRevision',
  'gitDirty',
  'scenario',
  'suite',
  'caseId',
  'requestedExecutionMode',
  'actualExecutionMode',
  'browser',
  'operatingSystem',
  'cpuModel',
  'logicalCpuCount',
  'totalMemoryBytes',
  'gpuVendor',
  'gpuArchitecture',
  'gpuDevice',
  'gpuDescription',
  'modelSource',
  'modelSizeBytes',
  'modelSha256',
  'traceId',
  'traceSha256',
  'wasmSha256',
  'modelLoadMs',
  'additionalClientAttachMs',
  'clientCount',
  'scheduler',
  'retention',
  'admission',
  'slice',
  'backgroundSlice',
  'foregroundActivationMs',
  'foregroundActivationP50Ms',
  'foregroundActivationP95Ms',
  'foregroundFirstTokenMs',
  'foregroundResumeTtftP50Ms',
  'foregroundResumeTtftP95Ms',
  'foregroundTotalWallMs',
  'backgroundTotalWallMs',
  'inputToNextPaintP95Ms',
  'sessionResumeToDispatchMs',
  'sessionResumeToFirstTokenMs',
  'resumeRestoreMs',
  'resumeRebuildMs',
  'resumedSessionSource',
  'sliceDurationP95Ms',
  'sliceDurationMaxMs',
  'stateReleaseCount',
  'releasedTokens',
  'admissionCommitCount',
  'admissionFallbackCount',
  'admissionRejectCount',
  'admissionRepairStepsP95',
  'admissionRepairStepsMax',
  'admissionBudgetViolationCount',
  'admissionLatencyP95Ms',
  'admissionDemotedTokens',
  'admissionCheckpointedTokens',
  'admissionRemovedStates',
  'requestSuccessRate',
  'requestFailureCount',
  'measuredCompletedRequestCount',
  'measuredRequestFailureCount',
  'measuredRequestSuccessRate',
  'workloadMakespanMs',
  'aggregateThroughputTokensPerSecond',
  'requestWallTimeP50Ms',
  'requestWallTimeP95Ms',
  'requestTtftP50Ms',
  'requestTtftP95Ms',
  'requestQueueWaitP95Ms',
  'foregroundRequestWallTimeP95Ms',
  'backgroundRequestWallTimeP95Ms',
  'overduePromotionCount',
  'totalPrefillMs',
  'totalDecodeMs',
  'totalCacheLoadMs',
  'totalRebuildMs',
  'totalParentRecoverMs',
  'totalPostMs',
  'schedulerDecisionCount',
  'schedulerDecisionTotalMs',
  'schedulerDecisionP95Ms',
  'cacheHintCount',
  'cacheHintTotalMs',
  'cacheHintP95Ms',
  'admissionTotalMs',
  'controlPlaneTotalMs',
  'controlPlanePct',
  'stateSnapshotCount',
  'peakSnapshotTokenBytes',
  'peakL1Tokens',
  'peakL2Tokens',
  'peakL3Tokens',
  'capacityViolationCount',
  'uiFrameSampleCount',
  'uiWorkloadBackend',
  'uiAverageFps',
  'uiFrameTimeP50Ms',
  'uiFrameTimeP95Ms',
  'uiFrameTimeP99Ms',
  'uiMaxFrameTimeMs',
  'uiLongFrameRatio',
  'finalDiskReads',
  'finalDiskWrites',
  'finalRestoreAttempts',
  'finalRestoreHitsL1',
  'finalRestoreHitsL2',
  'finalRestoreHitsL3',
  'finalRestoreMisses',
  'finalRestoreRebuilds',
  'operationErrorCount',
  'summaryPath',
];

const summaries = await findSummaries(input);
const rows = [];
for (const summaryPath of summaries.sort()) {
  const summary = JSON.parse(await readFile(summaryPath, 'utf8'));
  const webgpu = summary.environment?.browserRuntime?.webgpu ?? {};
  const operationErrorCount = summary.operationErrors?.length ?? 0;
  const controlledAdmissionFailure =
    summary.requestFailureCount > 0 &&
    summary.admissionRejectCount >= summary.requestFailureCount &&
    operationErrorCount === summary.requestFailureCount;
  const invalidReasons = [];
  if (summary.protocolVersion !== expected.protocolVersion) {
    invalidReasons.push('protocol-mismatch');
  }
  if (summary.gitDirty !== false) invalidReasons.push('dirty-worktree');
  if (summary.requestedExecutionMode !== summary.actualExecutionMode) {
    invalidReasons.push('execution-mode-mismatch');
  }
  if (operationErrorCount > 0 && !controlledAdmissionFailure) {
    invalidReasons.push('unexpected-operation-error');
  }
  if (summary.requestFailureCount > 0 && !controlledAdmissionFailure) {
    invalidReasons.push('unexpected-request-failure');
  }
  if (
    summary.requestedExecutionMode === 'webgpu' &&
    Number(summary.arguments?.['ui-work-items'] ?? 0) > 0 &&
    summary.uiWorkloadBackend !== 'webgpu-compute'
  ) {
    invalidReasons.push('ui-workload-not-webgpu');
  }
  if (expected.gitRevision && summary.gitRevision !== expected.gitRevision) {
    invalidReasons.push('git-revision-mismatch');
  }
  if (expected.modelSha256 && summary.model?.sha256 !== expected.modelSha256) {
    invalidReasons.push('model-sha256-mismatch');
  }
  if (
    expected.wasmSha256 &&
    summary.buildArtifacts?.wllamaWasmSha256 !== expected.wasmSha256
  ) {
    invalidReasons.push('wasm-sha256-mismatch');
  }
  if (
    expected.traceSha256 &&
    summary.scenario === 'trace' &&
    summary.trace?.sha256 !== expected.traceSha256
  ) {
    invalidReasons.push('trace-sha256-mismatch');
  }
  const valid = invalidReasons.length === 0;
  rows.push({
    valid,
    invalidReasons: invalidReasons.join(';'),
    protocolVersion: summary.protocolVersion,
    runId: summary.runId,
    deviceId: summary.deviceId,
    operator: summary.operator,
    trial: summary.trial,
    gitRevision: summary.gitRevision,
    llamaCppRevision: summary.llamaCppRevision,
    gitDirty: summary.gitDirty,
    scenario: summary.scenario,
    suite: summary.arguments?.suite,
    caseId:
      summary.arguments?.['case-id'] ??
      String(summary.runId).split('__').at(-2),
    requestedExecutionMode: summary.requestedExecutionMode,
    actualExecutionMode: summary.actualExecutionMode,
    browser: summary.browser,
    operatingSystem: summary.environment?.operatingSystem,
    cpuModel: summary.environment?.cpuModel,
    logicalCpuCount: summary.environment?.logicalCpuCount,
    totalMemoryBytes: summary.environment?.totalMemoryBytes,
    gpuVendor: webgpu.vendor,
    gpuArchitecture: webgpu.architecture,
    gpuDevice: webgpu.device,
    gpuDescription: webgpu.description,
    modelSource: summary.model?.source,
    modelSizeBytes: summary.model?.sizeBytes,
    modelSha256: summary.model?.sha256,
    traceId: summary.trace?.traceId,
    traceSha256: summary.trace?.sha256,
    wasmSha256: summary.buildArtifacts?.wllamaWasmSha256,
    modelLoadMs: summary.modelLoadMs,
    additionalClientAttachMs: summary.additionalClientAttachMs,
    clientCount: summary.clientCount,
    scheduler: summary.arguments?.scheduler,
    retention: summary.arguments?.retention,
    admission: summary.arguments?.admission,
    slice: summary.arguments?.slice,
    backgroundSlice: summary.arguments?.['background-slice'],
    foregroundActivationMs: summary.foregroundActivationMs,
    foregroundActivationP50Ms: summary.foregroundActivationP50Ms,
    foregroundActivationP95Ms: summary.foregroundActivationP95Ms,
    foregroundFirstTokenMs: summary.foregroundFirstTokenMs,
    foregroundResumeTtftP50Ms: summary.foregroundResumeTtftP50Ms,
    foregroundResumeTtftP95Ms: summary.foregroundResumeTtftP95Ms,
    foregroundTotalWallMs: summary.foregroundTotalWallMs,
    backgroundTotalWallMs: summary.backgroundTotalWallMs,
    inputToNextPaintP95Ms: summary.inputToNextPaintP95Ms,
    sessionResumeToDispatchMs: summary.sessionResumeToDispatchMs,
    sessionResumeToFirstTokenMs: summary.sessionResumeToFirstTokenMs,
    resumeRestoreMs: summary.resumeRestoreMs,
    resumeRebuildMs: summary.resumeRebuildMs,
    resumedSessionSource: summary.resumedSessionSource,
    sliceDurationP95Ms: summary.sliceDurationP95Ms,
    sliceDurationMaxMs: summary.sliceDurationMaxMs,
    stateReleaseCount: summary.stateReleaseCount,
    releasedTokens: summary.releasedTokens,
    admissionCommitCount: summary.admissionCommitCount,
    admissionFallbackCount: summary.admissionFallbackCount,
    admissionRejectCount: summary.admissionRejectCount,
    admissionRepairStepsP95: summary.admissionRepairStepsP95,
    admissionRepairStepsMax: summary.admissionRepairStepsMax,
    admissionBudgetViolationCount: summary.admissionBudgetViolationCount,
    admissionLatencyP95Ms: summary.admissionLatencyP95Ms,
    admissionDemotedTokens: summary.admissionDemotedTokens,
    admissionCheckpointedTokens: summary.admissionCheckpointedTokens,
    admissionRemovedStates: summary.admissionRemovedStates,
    requestSuccessRate: summary.requestSuccessRate,
    requestFailureCount: summary.requestFailureCount,
    measuredCompletedRequestCount: summary.measuredCompletedRequestCount,
    measuredRequestFailureCount: summary.measuredRequestFailureCount,
    measuredRequestSuccessRate: summary.measuredRequestSuccessRate,
    workloadMakespanMs: summary.workloadMakespanMs,
    aggregateThroughputTokensPerSecond:
      summary.aggregateThroughputTokensPerSecond,
    requestWallTimeP50Ms: summary.requestWallTimeP50Ms,
    requestWallTimeP95Ms: summary.requestWallTimeP95Ms,
    requestTtftP50Ms: summary.requestTtftP50Ms,
    requestTtftP95Ms: summary.requestTtftP95Ms,
    requestQueueWaitP95Ms: summary.requestQueueWaitP95Ms,
    foregroundRequestWallTimeP95Ms: summary.foregroundRequestWallTimeP95Ms,
    backgroundRequestWallTimeP95Ms: summary.backgroundRequestWallTimeP95Ms,
    overduePromotionCount: summary.overduePromotionCount,
    totalPrefillMs: summary.totalPrefillMs,
    totalDecodeMs: summary.totalDecodeMs,
    totalCacheLoadMs: summary.totalCacheLoadMs,
    totalRebuildMs: summary.totalRebuildMs,
    totalParentRecoverMs: summary.totalParentRecoverMs,
    totalPostMs: summary.totalPostMs,
    schedulerDecisionCount: summary.schedulerDecisionCount,
    schedulerDecisionTotalMs: summary.schedulerDecisionTotalMs,
    schedulerDecisionP95Ms: summary.schedulerDecisionP95Ms,
    cacheHintCount: summary.cacheHintCount,
    cacheHintTotalMs: summary.cacheHintTotalMs,
    cacheHintP95Ms: summary.cacheHintP95Ms,
    admissionTotalMs: summary.admissionTotalMs,
    controlPlaneTotalMs: summary.controlPlaneTotalMs,
    controlPlanePct: summary.controlPlanePct,
    stateSnapshotCount: summary.stateSnapshotCount,
    peakSnapshotTokenBytes: summary.peakSnapshotTokenBytes,
    peakL1Tokens: summary.peakL1Tokens,
    peakL2Tokens: summary.peakL2Tokens,
    peakL3Tokens: summary.peakL3Tokens,
    capacityViolationCount: summary.capacityViolationCount,
    uiFrameSampleCount: summary.uiFrameSampleCount,
    uiWorkloadBackend: summary.uiWorkloadBackend,
    uiAverageFps: summary.uiAverageFps,
    uiFrameTimeP50Ms: summary.uiFrameTimeP50Ms,
    uiFrameTimeP95Ms: summary.uiFrameTimeP95Ms,
    uiFrameTimeP99Ms: summary.uiFrameTimeP99Ms,
    uiMaxFrameTimeMs: summary.uiMaxFrameTimeMs,
    uiLongFrameRatio: summary.uiLongFrameRatio,
    finalDiskReads: summary.finalState?.tierStats?.diskReads,
    finalDiskWrites: summary.finalState?.tierStats?.diskWrites,
    finalRestoreAttempts: summary.finalState?.tierStats?.restoreAttempts,
    finalRestoreHitsL1: summary.finalState?.tierStats?.restoreHitsL1,
    finalRestoreHitsL2: summary.finalState?.tierStats?.restoreHitsL2,
    finalRestoreHitsL3: summary.finalState?.tierStats?.restoreHitsL3,
    finalRestoreMisses: summary.finalState?.tierStats?.restoreMisses,
    finalRestoreRebuilds: summary.finalState?.tierStats?.restoreRebuilds,
    operationErrorCount,
    summaryPath: path.relative(input, summaryPath),
  });
}

const content = [
  columns.join(','),
  ...rows.map((row) => columns.map((column) => csv(row[column])).join(',')),
].join('\n');
await writeFile(output, `${content}\n`);
console.log(
  JSON.stringify({
    input,
    output,
    runCount: rows.length,
    validCount: rows.filter((row) => row.valid).length,
  })
);
