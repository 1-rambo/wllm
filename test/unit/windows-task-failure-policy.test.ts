import { describe, expect, it } from "vitest";

const policyUrl = new URL(
  "../../evaluation/windows/task-failure-policy.mjs",
  import.meta.url
).href;

async function assess(records: unknown[], summary: { errorCount?: number }) {
  const policy = await import(policyUrl);
  return policy.assessTaskFailureOutcome(records, summary);
}

function taskFailure(overrides: Record<string, unknown> = {}) {
  return {
    kind: "task",
    arm: "wllama-per-tab",
    pressureId: "pressure-500",
    taskId: "artifex-identity",
    error: "task timed out after 900000 ms",
    ...overrides
  };
}

describe("Windows task failure policy", () => {
  it("accepts complete per-tab baseline task failures as availability outcomes", async () => {
    const records = [
      taskFailure(),
      taskFailure({
        kind: "recovery",
        pressureId: "clear-return",
        taskId: "tavern-recovery",
        error: "wllama has crashed, please reload the module"
      })
    ];

    await expect(
      assess(records, { errorCount: 2 })
    ).resolves.toMatchObject({
      eligible: true,
      observedErrorCount: 2,
      summaryErrorCount: 2,
      acceptedTaskFailureCount: 2,
      unexpectedFailureCount: 0,
      timeoutCount: 1,
      failureArms: ["wllama-per-tab"],
      failureKinds: ["task", "recovery"]
    });
  });

  it("rejects trial-level failures instead of treating them as task outcomes", async () => {
    const assessment = await assess(
      [
        taskFailure(),
        {
          kind: "failure",
          arm: "wllama-per-tab",
          pressureId: "pressure-500",
          error: "browser context closed"
        }
      ],
      { errorCount: 2 }
    );

    expect(assessment.eligible).toBe(false);
    expect(assessment.unexpectedFailureCount).toBe(1);
    expect(assessment.failureKinds).toEqual(["task", "failure"]);
  });

  it("rejects failures in the ours arm", async () => {
    const assessment = await assess(
      [taskFailure({ arm: "ours" })],
      { errorCount: 1 }
    );

    expect(assessment.eligible).toBe(false);
    expect(assessment.unexpectedFailureCount).toBe(1);
    expect(assessment.failureArms).toEqual(["ours"]);
  });

  it("rejects mismatched record and summary error counts", async () => {
    const assessment = await assess(
      [taskFailure()],
      { errorCount: 2 }
    );

    expect(assessment.eligible).toBe(false);
    expect(assessment.reason).toBe("record and summary error counts differ");
  });

  it("allows a failed baseline task to carry its secondary visibility failure", async () => {
    const assessment = await assess(
      [taskFailure({ pressureVisible: false })],
      { errorCount: 1 }
    );

    expect(assessment.eligible).toBe(true);
    expect(assessment.pressureVisibilityFailures).toBe(1);
    expect(assessment.unexpectedPressureVisibilityFailures).toBe(0);
  });

  it("rejects invalid pressure or lifecycle evidence", async () => {
    const pressure = await assess(
      [taskFailure(), { kind: "task", pressureVisible: false }],
      { errorCount: 1 }
    );
    const lifecycle = await assess(
      [taskFailure(), { kind: "recovery", lifecycleValid: false }],
      { errorCount: 1 }
    );

    expect(pressure.eligible).toBe(false);
    expect(pressure.pressureVisibilityFailures).toBe(1);
    expect(lifecycle.eligible).toBe(false);
    expect(lifecycle.lifecycleFailures).toBe(1);
  });

  it("keeps strict runs on the normal technical-pass path", async () => {
    const assessment = await assess(
      [{ kind: "task", arm: "ours" }],
      { errorCount: 0 }
    );

    expect(assessment.eligible).toBe(false);
    expect(assessment.observedErrorCount).toBe(0);
  });

  it("requires task identity so a generic arm error cannot be accepted", async () => {
    const assessment = await assess(
      [taskFailure({ taskId: null })],
      { errorCount: 1 }
    );

    expect(assessment.eligible).toBe(false);
    expect(assessment.unexpectedFailureCount).toBe(1);
  });
});
