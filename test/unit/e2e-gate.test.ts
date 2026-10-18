import { describe, expect, it } from "vitest";
// @ts-expect-error The evaluation harness is plain ESM and has no ambient declarations.
import { evaluateGates } from "../../evaluation/lib/gate.mjs";

function loadedWithGate(metric: string) {
  return {
    preregistration: {
      gates: {
        "e2e-1": [
          {
            id: "active-target",
            type: "threshold",
            metric,
            threshold: 0.9,
            direction: "higher"
          }
        ]
      }
    },
    config: {
      experiments: { "e2e-1": {} },
      judge: { minimumValidFraction: 0.9 }
    }
  };
}

const summary = {
  completeness: { expected: 10, observed: 10 },
  errorCount: 0
};

describe("evaluateGates threshold aggregation", () => {
  it("treats boolean metrics as a ratio instead of dropping them", () => {
    const records = Array.from({ length: 10 }, (_, index) => ({
      arm: "ours",
      priorityTargetActiveAtRequest: index < 9
    }));
    records.push({
      arm: "wllama-per-tab",
      priorityTargetActiveAtRequest: false
    });
    const report = evaluateGates({
      loaded: loadedWithGate("priorityTargetActiveAtRequest"),
      experimentId: "e2e-1",
      records,
      summary,
      complete: true
    });

    expect(report.gates[0]).toMatchObject({
      id: "active-target",
      status: "pass",
      n: 10,
      observed: 0.9,
      threshold: 0.9
    });
  });

  it("fails when fewer than ninety percent of targets remain active", () => {
    const records = Array.from({ length: 10 }, (_, index) => ({
      arm: "ours",
      priorityTargetActiveAtRequest: index < 8
    }));
    records.push({
      arm: "wllama-per-tab",
      priorityTargetActiveAtRequest: false
    });
    const report = evaluateGates({
      loaded: loadedWithGate("priorityTargetActiveAtRequest"),
      experimentId: "e2e-1",
      records,
      summary,
      complete: true
    });

    expect(report.gates[0]).toMatchObject({
      id: "active-target",
      status: "fail",
      n: 10,
      observed: 0.8
    });
  });
});
