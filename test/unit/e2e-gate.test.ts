import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";
// @ts-expect-error The evaluation harness is plain ESM and has no ambient declarations.
import { evaluateGates } from "../../evaluation/lib/gate.mjs";

function loadedWithGate(metric: string, arm?: string) {
  return {
    preregistration: {
      gates: {
        "e2e-1": [
          {
            id: "active-target",
            type: "threshold",
            metric,
            threshold: 0.9,
            ...(arm ? { arm } : {}),
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

const e2eConfigUrl = new URL(
  "../../evaluation/config/e2e.json",
  import.meta.url
);

describe("paper-aligned E2E configuration", () => {
  it("uses five Artifex tabs, five Tavern tabs, and the full pressure ladder", async () => {
    const config = JSON.parse(await readFile(e2eConfigUrl, "utf8"));
    expect(config.layout).toEqual({
      appTabs: 10,
      artifexTabs: 5,
      pressureTabs: 1,
      totalTabs: 11
    });
    expect(config.pressure.schedule.map((entry: { triangles: number }) => entry.triangles))
      .toEqual([0, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 0]);
    expect(config.modes.formal).toMatchObject({
      appTabs: 10,
      artifexTabs: 5,
      pressureTabs: 1,
      totalTabs: 11,
      repeats: 3,
      judge: false
    });
    expect(config.execution).toMatchObject({ nParallel: 10, l1NodeCap: 10 });
  });
  it("limits threshold gates to the declared arm", () => {
    const records = [
      ...Array.from({ length: 9 }, () => ({
        arm: "ours",
        role: "background",
        backgroundCompletionRate: true,
        completed: true
      })),
      {
        arm: "ours",
        role: "background",
        backgroundCompletionRate: false,
        completed: false
      },
      {
        arm: "wllama-per-tab",
        role: "background",
        backgroundCompletionRate: false,
        completed: false
      }
    ];
    const loaded = loadedWithGate("backgroundCompletionRate", "ours");
    const report = evaluateGates({
      loaded,
      experimentId: "e2e-1",
      records,
      summary,
      complete: true
    });

    expect(report.gates[0]).toMatchObject({
      status: "pass",
      arm: "ours",
      n: 10,
      numerator: 9,
      denominator: 10,
      observed: 0.9
    });
  });
});
