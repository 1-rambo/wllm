import { median, pairedBootstrapDifference, summarize } from "./statistics.mjs";

const METRICS = [
  "ttftMs",
  "wallMs",
  "decodeTokensPerSecond",
  "priorityFirstTokenLatencyMs",
  "priorityCompletionMs",
  "priorityAdvantageGapMs",
  "frameP50Ms",
  "frameP95Ms",
  "frameP99Ms",
  "frameMaxMs",
  "over33Ratio",
  "over50Ratio",
  "units",
  "prefillTokens",
  "cacheTokens",
  "prefillProcessedTokens",
  "measuredPrefillMsPerToken",
  "replayedTokens",
  "recoveryFirstTokenLatencyMs",
  "recoveryCompletionMs"
];

export function buildSummary(records, options = {}) {
  const taskRecords = records.filter((record) => record.kind === "task" || record.kind === "recovery");
  const groups = groupRecords(taskRecords);
  const pairwise = buildPairwise(taskRecords, options.armPairs ?? []);
  const completeness = buildCompleteness(taskRecords, options.expected ?? null);
  const quality = summarizeQuality(records);
  return {
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    metrics: METRICS,
    groups,
    pairwise,
    completeness,
    quality,
    errorCount: records.filter(
      (record) =>
        record.kind === "failure" ||
        (record.error !== null && record.error !== undefined)
    ).length,
    recordCount: records.length
  };
}

function groupRecords(records) {
  const groups = new Map();
  for (const record of records) {
    const key = [
      record.arm,
      record.pressureId ?? "none",
      record.app ?? "all",
      record.role ?? "all"
    ].join("|");
    const group = groups.get(key) ?? {
      arm: record.arm,
      pressureId: record.pressureId ?? null,
      pressureTriangles: record.pressureTriangles ?? null,
      app: record.app ?? null,
      role: record.role ?? null,
      n: 0,
      errors: 0,
      metrics: {}
    };
    group.n += 1;
    if (record.error !== null && record.error !== undefined) group.errors += 1;
    for (const metric of METRICS) {
      const value = metricValue(record, metric);
      if (!Number.isFinite(value)) continue;
      group.metrics[metric] ??= [];
      group.metrics[metric].push(value);
    }
    groups.set(key, group);
  }
  return [...groups.values()].map((group) => ({
    ...group,
    metrics: Object.fromEntries(
      Object.entries(group.metrics).map(([metric, values]) => [metric, summarize(values)])
    )
  }));
}

function buildPairwise(records, armPairs) {
  const rows = [];
  for (const pair of armPairs) {
    const [leftArm, rightArm] = pair;
    const left = records.filter((record) => record.arm === leftArm);
    const right = records.filter((record) => record.arm === rightArm);
    const rightByKey = new Map(right.map((record) => [pairKey(record), record]));
    const paired = [];
    for (const leftRecord of left) {
      const rightRecord = rightByKey.get(pairKey(leftRecord));
      if (rightRecord) paired.push([leftRecord, rightRecord]);
    }
    const byPressure = new Map();
    for (const pairRecords of paired) {
      const pressureId = pairRecords[0].pressureId ?? "none";
      const group = byPressure.get(pressureId) ?? [];
      group.push(pairRecords);
      byPressure.set(pressureId, group);
    }
    for (const [pressureId, pairs] of byPressure) {
      rows.push(pairRow(leftArm, rightArm, pressureId, pairs));
    }
    if (paired.length > 0) {
      rows.push(pairRow(leftArm, rightArm, "overall", paired));
    }
  }
  return rows;
}

function pairRow(leftArm, rightArm, pressureId, pairs) {
  const metrics = {};
  for (const metric of METRICS) {
    const values = pairs.map(([leftRecord, rightRecord]) => [
      metricValue(leftRecord, metric),
      metricValue(rightRecord, metric)
    ]);
    const result = pairedBootstrapDifference(values);
    const baselineMedian = median(values.map(([, rightValue]) => rightValue));
    metrics[metric] = {
      ...result,
      relativeEffect:
        result.medianDifference === null ||
        baselineMedian === null ||
        baselineMedian === 0
          ? null
          : result.medianDifference / baselineMedian
    };
  }
  return {
    leftArm,
    rightArm,
    pressureId,
    n: pairs.length,
    metrics
  };
}

function buildCompleteness(records, expected) {
  if (expected === null) {
    return { expected: null, observed: records.length, complete: null };
  }
  return {
    expected,
    observed: records.length,
    complete: records.length >= expected
  };
}

function summarizeQuality(records) {
  const attempted = records.filter(
    (record) => typeof record.judgeValid === "boolean"
  );
  if (attempted.length === 0) {
    return { n: 0, scored: 0, valid: 0, validFraction: null, byArm: {} };
  }
  const scored = attempted.filter((record) =>
    Number.isFinite(record.judgeScore)
  );
  const byArm = {};
  for (const record of scored) {
    byArm[record.arm] ??= [];
    byArm[record.arm].push(record.judgeScore);
  }
  const valid = attempted.filter((record) => record.judgeValid === true).length;
  return {
    n: attempted.length,
    scored: scored.length,
    valid,
    validFraction: valid / attempted.length,
    byArm: Object.fromEntries(
      Object.entries(byArm).map(([arm, values]) => [arm, summarize(values)])
    )
  };
}

function pairKey(record) {
  return [
    record.model,
    record.platform,
    record.pressureId,
    record.repeat,
    record.app,
    record.taskId,
    record.role,
    record.phase
  ].join("|");
}

function metricValue(record, metric) {
  if (metric === "frameP50Ms") return record.frame?.p50Ms;
  if (metric === "frameP95Ms") return record.frame?.p95Ms;
  if (metric === "frameP99Ms") return record.frame?.p99Ms;
  if (metric === "frameMaxMs") return record.frame?.maxMs;
  if (metric === "over33Ratio") return record.frame?.over33Ratio;
  if (metric === "over50Ratio") return record.frame?.over50Ratio;
  return record[metric];
}
