const TASK_RECORD_KINDS = new Set(["task", "recovery"]);
const ACCEPTED_TASK_FAILURE_ARMS = new Set(["wllama-per-tab"]);

/**
 * Identify completed runs whose observed errors are task-level outcomes from
 * the per-tab baseline. Trial-level failures and errors in our runtime remain
 * technical failures that stop the handoff pipeline.
 */
export function assessTaskFailureOutcome(records, summary) {
  if (!Array.isArray(records)) {
    return {
      eligible: false,
      reason: "records.jsonl is unavailable",
      observedErrorCount: 0,
      summaryErrorCount: Number(summary?.errorCount ?? 0),
      acceptedTaskFailureCount: 0,
      unexpectedFailureCount: 0,
      failureArms: [],
      failureKinds: [],
      timeoutCount: 0,
      pressureVisibilityFailures: 0,
      lifecycleFailures: 0,
      unexpectedPressureVisibilityFailures: 0,
      unexpectedLifecycleFailures: 0
    };
  }

  const errorRecords = records.filter(hasRecordError);
  const acceptedTaskFailures = errorRecords.filter(isAcceptedTaskFailure);
  const unexpectedFailures = errorRecords.filter(
    (record) => !isAcceptedTaskFailure(record)
  );
  const observedErrorCount = errorRecords.length;
  const summaryErrorCount = Number(summary?.errorCount ?? observedErrorCount);
  const unexpectedPressureVisibilityFailures = records.filter(
    (record) =>
      record.pressureVisible === false && !isAcceptedTaskFailure(record)
  ).length;
  const unexpectedLifecycleFailures = records.filter(
    (record) =>
      record.lifecycleValid === false && !isAcceptedTaskFailure(record)
  ).length;
  const errorCountsMatch =
    Number.isFinite(summaryErrorCount) &&
    observedErrorCount === summaryErrorCount;
  const eligible =
    observedErrorCount > 0 &&
    errorCountsMatch &&
    unexpectedFailures.length === 0 &&
    unexpectedPressureVisibilityFailures === 0 &&
    unexpectedLifecycleFailures === 0;

  let reason = null;
  if (observedErrorCount === 0) reason = "no task-level failures were observed";
  else if (!errorCountsMatch) reason = "record and summary error counts differ";
  else if (unexpectedFailures.length > 0) {
    reason = "at least one failure is not an accepted per-tab task outcome";
  } else if (unexpectedPressureVisibilityFailures > 0) {
    reason = "pressure visibility evidence is invalid";
  } else if (unexpectedLifecycleFailures > 0) {
    reason = "lifecycle evidence is invalid";
  }

  return {
    eligible,
    reason,
    observedErrorCount,
    summaryErrorCount,
    acceptedTaskFailureCount: acceptedTaskFailures.length,
    unexpectedFailureCount: unexpectedFailures.length,
    failureArms: unique(errorRecords.map((record) => record.arm ?? "unknown")),
    failureKinds: unique(errorRecords.map((record) => record.kind ?? "unknown")),
    timeoutCount: acceptedTaskFailures.filter((record) =>
      /task timed out/i.test(String(record.error ?? ""))
    ).length,
    pressureVisibilityFailures: records.filter(
      (record) => record.pressureVisible === false
    ).length,
    lifecycleFailures: records.filter(
      (record) => record.lifecycleValid === false
    ).length,
    unexpectedPressureVisibilityFailures,
    unexpectedLifecycleFailures
  };
}

function isAcceptedTaskFailure(record) {
  return (
    TASK_RECORD_KINDS.has(record?.kind) &&
    ACCEPTED_TASK_FAILURE_ARMS.has(record?.arm) &&
    typeof record?.taskId === "string" &&
    record.taskId.length > 0 &&
    typeof record?.pressureId === "string" &&
    record.pressureId.length > 0
  );
}

function hasRecordError(record) {
  return (
    record?.kind === "failure" ||
    (record?.error !== null && record?.error !== undefined)
  );
}

function unique(values) {
  return [...new Set(values)];
}
