import { median } from "./statistics.mjs";

const MEMORY_PATTERN =
  /(Invalid magic number|memory access out of bounds|wasm memory|out of memory|OutOfMemory|OOM|device lost|memory-bounded failure|wllama has crashed|RuntimeError: unreachable|__wrap_abort|ggml_[a-z_]*abort|wasm-function)/i;

export function evaluateGates(options) {
  const { loaded, experimentId, records, summary, complete } = options;
  const definitions = loaded.preregistration.gates[experimentId] ?? [];
  const experiment = loaded.config.experiments[experimentId];
  const gates = definitions.map((definition) =>
    evaluateGate(definition, {
      records,
      summary,
      complete,
      experiment,
      judgeMinimumValidFraction:
        loaded.config.judge?.minimumValidFraction ?? 0.9
    })
  );
  return {
    schemaVersion: 1,
    experiment: experimentId,
    generatedAt: new Date().toISOString(),
    complete: complete === true,
    gates
  };
}

function evaluateGate(definition, context) {
  const base = {
    id: definition.id,
    type: definition.type,
    description: definition.description ?? null
  };
  switch (definition.type) {
    case "completeness":
      return completenessGate(base, context);
    case "resource-integrity":
      return resourceGate(base, context);
    case "priority-applied":
      return priorityGate(base, context);
    case "paired-directional":
      return pairedGate(base, definition, context);
    case "threshold":
      return thresholdGate(base, definition, context);
    case "quality-nonregression":
      return qualityGate(base, definition, context);
    case "lifecycle-observed":
      return lifecycleGate(base, context);
    case "output-consistency":
      return outputConsistencyGate(base, context);
    default:
      return {
        ...base,
        status: "incomplete",
        reason: `unsupported gate type ${definition.type}`
      };
  }
}

function completenessGate(base, { summary, complete }) {
  const expected = summary?.completeness?.expected ?? null;
  const observed = summary?.completeness?.observed ?? 0;
  const status =
    complete === true && expected !== null && observed >= expected
      ? "pass"
      : "incomplete";
  return {
    ...base,
    status,
    expected,
    observed,
    errors: summary?.errorCount ?? null
  };
}

function resourceGate(base, { records, complete }) {
  const memoryMatches = records
    .filter((record) => {
      const text = [
        record.error,
        record.resourceError,
        record.runtimeError,
        record.diagnostics?.runtime?.lastError
      ]
        .concat(Array.isArray(record.pageErrors) ? record.pageErrors : [])
        .filter(Boolean)
        .join("\n");
      return MEMORY_PATTERN.test(text);
    })
    .map((record) => ({
      arm: record.arm,
      app: record.app,
      taskId: record.taskId,
      error: record.error ?? record.resourceError ?? record.runtimeError ?? null
    }));
  const visibilityFailures = records.filter(
    (record) => record.pressureVisible === false
  );
  const lifecycleFailures = records.filter(
    (record) => record.lifecycleValid === false
  );
  const violations = [
    ...memoryMatches.map((match) => ({ type: "memory", ...match })),
    ...visibilityFailures.map((record) => ({
      type: "pressure-visibility",
      arm: record.arm,
      app: record.app,
      taskId: record.taskId,
      error: record.invalidReason ?? "pressure tab was not visible"
    })),
    ...lifecycleFailures.map((record) => ({
      type: "lifecycle",
      arm: record.arm,
      app: record.app,
      taskId: record.taskId,
      error: record.invalidReason ?? "lifecycle evidence incomplete"
    }))
  ];
  return {
    ...base,
    status:
      complete !== true
        ? "incomplete"
        : violations.length === 0
          ? "pass"
          : "fail",
    memoryMatches,
    matchCount: memoryMatches.length,
    pressureVisibleFailures: visibilityFailures.length,
    lifecycleFailures: lifecycleFailures.length,
    violations,
    checkedRecords: records.length
  };
}

function priorityGate(base, { records, complete }) {
  const requested = records.filter(
    (record) => record.priorityRequested === true && record.arm !== "wllama-per-tab"
  );
  const applied = requested.filter((record) => record.priorityApplied === true);
  const status =
    complete !== true
      ? "incomplete"
      : requested.length === 0
        ? "incomplete"
        : applied.length === requested.length
          ? "pass"
          : "fail";
  return {
    ...base,
    status,
    requested: requested.length,
    applied: applied.length,
    failed: requested.length - applied.length
  };
}

function pairedGate(base, definition, { summary, complete }) {
  const row = findPairRow(
    summary?.pairwise ?? [],
    definition.comparator,
    definition.metric
  );
  if (!row) {
    return {
      ...base,
      status: "incomplete",
      comparator: definition.comparator,
      metric: definition.metric,
      reason: "paired summary row not found"
    };
  }
  const metric = row.metrics?.[definition.metric];
  if (!metric || metric.n <= 0) {
    return {
      ...base,
      status: "incomplete",
      comparator: definition.comparator,
      metric: definition.metric,
      n: metric?.n ?? 0
    };
  }
  const direction = definition.direction ?? "lower";
  const minimum = definition.minimumRelativeEffect ?? 0;
  const relative = metric.relativeEffect;
  const observed = metric.medianDifference;
  const ci = metric.ci95 ?? [null, null];
  const directionPass = direction === "lower" ? observed < 0 : observed > 0;
  const magnitudePass =
    relative === null
      ? false
      : direction === "lower"
        ? relative <= -minimum
        : relative >= minimum;
  const ciPass =
    metric.n >= 3 &&
    (direction === "lower" ? ci[1] < 0 : ci[0] > 0);
  return {
    ...base,
    status:
      complete !== true
        ? "incomplete"
        : directionPass && magnitudePass && ciPass
          ? "pass"
          : "fail",
    comparator: definition.comparator,
    metric: definition.metric,
    direction,
    minimumRelativeEffect: minimum,
    n: metric.n,
    observed,
    relativeEffect: relative,
    ci95: ci,
    directionPass,
    magnitudePass,
    ciPass
  };
}

function thresholdGate(base, definition, { records, complete }) {
  let eligibleRecords = definition.arm
    ? records.filter((record) => record.arm === definition.arm)
    : records;
  if (definition.metric === "priorityTargetActiveAtRequest") {
    eligibleRecords = eligibleRecords.filter(
      (record) => record.arm !== "wllama-per-tab"
    );
  }
  const values = eligibleRecords
    .map((record) => metricValue(record, definition.metric))
    .filter((value) => Number.isFinite(value));
  if (values.length === 0) {
    return {
      ...base,
      status: "incomplete",
      metric: definition.metric,
      n: 0
    };
  }
  const ratioAggregation =
    definition.metric.endsWith("Rate") ||
    eligibleRecords.some(
      (record) => typeof record[definition.metric] === "boolean"
    );
  const observed = ratioAggregation
    ? values.reduce((sum, value) => sum + value, 0) / values.length
    : median(values);
  const numerator = ratioAggregation
    ? values.reduce((sum, value) => sum + value, 0)
    : null;
  const denominator = ratioAggregation ? values.length : null;
  const pass =
    definition.direction === "higher"
      ? observed >= definition.threshold
      : observed <= definition.threshold;
  return {
    ...base,
    status: complete !== true ? "incomplete" : pass ? "pass" : "fail",
    metric: definition.metric,
    arm: definition.arm ?? null,
    n: values.length,
    numerator,
    denominator,
    observed,
    threshold: definition.threshold,
    direction: definition.direction
  };
}

function qualityGate(
  base,
  definition,
  { summary, complete, judgeMinimumValidFraction }
) {
  const quality = summary?.quality ?? {};
  const byArm = quality.byArm ?? {};
  const left =
    byArm?.["ours"]?.median ?? byArm?.["ours-full"]?.median ?? null;
  const right = byArm?.[definition.comparator]?.median ?? null;
  const validFraction = quality.validFraction ?? null;
  const validFractionPass =
    validFraction !== null && validFraction >= judgeMinimumValidFraction;
  if (left === null || right === null || validFraction === null) {
    return {
      ...base,
      status: "incomplete",
      comparator: definition.comparator,
      metric: definition.metric,
      reason: "judge scores are absent",
      validFraction,
      minimumValidFraction: judgeMinimumValidFraction,
      n: quality.n ?? 0
    };
  }
  const difference = left - right;
  return {
    ...base,
    status:
      complete !== true
        ? "incomplete"
        : !validFractionPass
          ? "fail"
          : difference >= definition.minimumMargin
            ? "pass"
            : "fail",
    comparator: definition.comparator,
    metric: definition.metric,
    leftMedian: left,
    rightMedian: right,
    difference,
    minimumMargin: definition.minimumMargin,
    valid: quality.valid ?? 0,
    validFraction,
    minimumValidFraction: judgeMinimumValidFraction,
    validFractionPass,
    n: quality.n ?? 0
  };
}

function lifecycleGate(base, { records, complete }) {
  const recoveryRecords = records.filter((record) => record.kind === "recovery");
  const required = ["hidden", "frozen", "reload", "active"];
  const failures = [];
  const observed = [];
  for (const record of recoveryRecords) {
    const events = Array.isArray(record.lifecycleEvents)
      ? record.lifecycleEvents.map(String)
      : [];
    observed.push(...events);
    const missing = required.filter((event) => !events.includes(event));
    if (record.lifecycleValid !== true || missing.length > 0) {
      failures.push({
        arm: record.arm,
        model: record.model,
        app: record.app,
        taskId: record.taskId,
        repeat: record.repeat,
        pressureId: record.pressureId,
        lifecycleValid: record.lifecycleValid ?? null,
        missing
      });
    }
  }
  return {
    ...base,
    status:
      complete !== true || recoveryRecords.length === 0
        ? "incomplete"
        : failures.length === 0
          ? "pass"
          : "fail",
    required,
    observed: [...new Set(observed)],
    checked: recoveryRecords.length,
    failures
  };
}

function outputConsistencyGate(base, { records, complete }) {
  const checks = records.filter(
    (record) => typeof record.outputConsistent === "boolean"
  );
  const failures = checks.filter((record) => record.outputConsistent !== true);
  return {
    ...base,
    status:
      complete !== true
        ? "incomplete"
        : checks.length === 0
          ? "incomplete"
          : failures.length === 0
            ? "pass"
            : "fail",
    checked: checks.length,
    failures: failures.length
  };
}

function findPairRow(rows, comparator, metric) {
  const candidates = rows.filter(
    (row) => row.rightArm === comparator && row.metrics?.[metric] !== undefined
  );
  return (
    candidates.find((row) => row.pressureId === "overall") ??
    candidates[0] ??
    null
  );
}

function metricValue(record, metric) {
  if (metric === "backgroundCompletionRate") {
    if (record.role !== "background") return null;
    return record.completed === true ? 1 : 0;
  }
  const value = record[metric];
  if (typeof value === "boolean") return value ? 1 : 0;
  return value;
}
