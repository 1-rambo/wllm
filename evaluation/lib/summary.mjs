import { median, pairedBootstrapDifference, summarize } from "./statistics.mjs";
import { isValidClickResponse } from "./click-response.mjs";

const METRICS = [
  "ttftMs",
  "wallMs",
  "decodeTokensPerSecond",
  "clickResponseMs",
  "clickActivationMs",
  "clickFrameResponseMs",
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
  const clickResponse = summarizeClickResponse(taskRecords);
  return {
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    metrics: METRICS,
    groups,
    pairwise,
    clickResponse,
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

function summarizeClickResponse(records) {
  const armGroups = new Map();
  const pressureGroups = new Map();
  let attempts = 0;
  let valid = 0;
  let tasksAttempted = 0;
  let tasksValid = 0;

  for (const record of records) {
    const recordAttempts = Number(record.clickResponseAttempts ?? 0);
    const recordValid = Number(record.clickResponseValid ?? 0);
    if (!Number.isFinite(recordAttempts) || recordAttempts <= 0) continue;

    const validCount = Number.isFinite(recordValid) ? recordValid : 0;
    attempts += recordAttempts;
    valid += validCount;
    tasksAttempted += 1;
    if (validCount > 0) tasksValid += 1;

    for (const [collection, key] of [
      [armGroups, record.arm],
      [pressureGroups, `${record.arm}|${record.pressureId ?? "none"}`]
    ]) {
      const row = collection.get(key) ?? {
        arm: record.arm,
        pressureId: record.pressureId ?? null,
        pressureTriangles: record.pressureTriangles ?? null,
        tasksAttempted: 0,
        tasksValid: 0,
        attempts: 0,
        valid: 0,
        samples: {
          clickResponseMs: [],
          clickActivationMs: [],
          clickFrameResponseMs: []
        },
        taskMedians: []
      };
      row.tasksAttempted += 1;
      row.tasksValid += validCount > 0 ? 1 : 0;
      row.attempts += recordAttempts;
      row.valid += validCount;
      if (Number.isFinite(record.clickResponseMs)) {
        row.taskMedians.push(record.clickResponseMs);
      }
      for (const sample of Array.isArray(record.clickResponses) ? record.clickResponses : []) {
        if (!isValidClickResponse(sample)) continue;
        for (const [metric, values] of Object.entries(row.samples)) {
          if (Number.isFinite(sample[metric])) values.push(sample[metric]);
        }
      }
      collection.set(key, row);
    }
  }

  return {
    role: "exploratory",
    gate: false,
    attempts,
    valid,
    validFraction: attempts > 0 ? valid / attempts : null,
    tasksAttempted,
    tasksValid,
    byArm: finalizeClickGroups(armGroups),
    byArmPressure: finalizeClickGroups(pressureGroups)
  };
}

function finalizeClickGroups(groups) {
  return [...groups.values()].map((row) => ({
    arm: row.arm,
    pressureId: row.pressureId,
    pressureTriangles: row.pressureTriangles,
    tasksAttempted: row.tasksAttempted,
    tasksValid: row.tasksValid,
    attempts: row.attempts,
    valid: row.valid,
    validFraction: row.attempts > 0 ? row.valid / row.attempts : null,
    metrics: Object.fromEntries(
      Object.entries(row.samples).map(([metric, values]) => [metric, summarize(values)])
    ),
    taskMedianClickResponseMs: summarize(row.taskMedians)
  }));
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
