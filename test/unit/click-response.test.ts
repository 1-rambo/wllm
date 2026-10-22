import { describe, expect, it } from "vitest";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
// @ts-expect-error The evaluation harness is plain ESM and has no ambient declarations.
import { clickResponseSeed, createClickSequencer, isValidClickResponse } from "../../evaluation/lib/click-response.mjs";
// @ts-expect-error The evaluation harness is plain ESM and has no ambient declarations.
import { buildSummary } from "../../evaluation/lib/summary.mjs";

describe("click-response sequencing", () => {
  it("is frozen as an exploratory E2E-1 measurement without a gate", async () => {
    const configUrl = new URL("../../evaluation/config/e2e.json", import.meta.url);
    const preregUrl = new URL(
      "../../evaluation/config/e2e-preregistration.json",
      import.meta.url
    );
    const configText = await readFile(configUrl, "utf8");
    const config = JSON.parse(configText);
    const preregistration = JSON.parse(await readFile(preregUrl, "utf8"));
    const configSha256 = createHash("sha256").update(configText).digest("hex");

    expect(preregistration.configSha256).toBe(configSha256);
    expect(config.clickResponse).toMatchObject({
      enabledForExperiments: ["e2e-1"],
      responseFrames: 2,
      affectsPriority: false
    });
    expect(preregistration.protocol.clickResponse).toMatchObject({
      experiment: "e2e-1",
      role: "exploratory",
      affectsPriority: false,
      gate: false
    });
    expect(
      preregistration.gates["e2e-1"].some(
        (gate: { id: string }) => gate.id.includes("click")
      )
    ).toBe(false);
  });

  it("is deterministic and only chooses active tabs without adjacent repeats", () => {
    const first = runSequence(20261022);
    const second = runSequence(20261022);
    const other = runSequence(20261023);

    expect(first).toEqual(second);
    expect(first).not.toEqual(other);
    expect(first).toHaveLength(10);
    for (let index = 1; index < first.length; index += 1) {
      expect(first[index]).not.toBe(first[index - 1]);
    }
    for (const value of first) {
      expect([0, 1, 2, 3]).toContain(value);
    }
  });

  it("stops at max clicks and avoids tabs outside the supplied active set", () => {
    const sequencer = createClickSequencer({
      indices: [0, 1, 2, 3, 3],
      seed: 4,
      initialDelayMs: 0,
      minIntervalMs: 1,
      maxIntervalMs: 2,
      maxClicks: 3
    });
    const sequence = [];
    for (let index = 0; index < 5; index += 1) {
      const value = sequencer.next(index < 2 ? [0, 2] : [1, 3]);
      if (value !== null) sequence.push(value);
    }

    expect(sequence).toHaveLength(3);
    expect(sequence.slice(0, 2).every((value) => value === 0 || value === 2)).toBe(true);
    expect(sequence[2]).toBeGreaterThanOrEqual(1);
    expect(sequence[2]).toBeLessThanOrEqual(3);
  });

  it("derives a stable per-model/trial seed", () => {
    expect(clickResponseSeed(7, 3, "llama")).toBe(
      clickResponseSeed(7, 3, "llama")
    );
    expect(clickResponseSeed(7, 4, "llama")).not.toBe(
      clickResponseSeed(7, 3, "llama")
    );
    expect(clickResponseSeed(7, 3, "qwen")).not.toBe(
      clickResponseSeed(7, 3, "llama")
    );
  });
});

describe("click-response validity and aggregation", () => {
  it("requires the requested rendered-frame count and a valid timing", () => {
    expect(isValidClickResponse(validSample({}))).toBe(true);
    expect(isValidClickResponse(validSample({ framesWaited: 1, responseFrames: 2 }))).toBe(false);
    expect(isValidClickResponse(validSample({ responseFrames: null }))).toBe(false);
    expect(isValidClickResponse(validSample({ error: "activation failed" }))).toBe(false);
    expect(isValidClickResponse(validSample({ clickResponseMs: null }))).toBe(false);
    expect(isValidClickResponse(validSample({ visibilityBefore: "visible" }))).toBe(true);
  });

  it("aggregates exploratory samples without changing task denominators", () => {
    const records = [
      taskRecord("ours", 80, 2, [
        validSample({ clickResponseMs: 70, clickActivationMs: 20, clickFrameResponseMs: 50 }),
        validSample({ clickResponseMs: 90, clickActivationMs: 30, clickFrameResponseMs: 60 })
      ]),
      taskRecord("wllama-per-tab", 160, 2, [
        validSample({ clickResponseMs: 150, clickActivationMs: 40, clickFrameResponseMs: 110 }),
        validSample({ clickResponseMs: 170, clickActivationMs: 50, clickFrameResponseMs: 120 })
      ])
    ];

    const summary = buildSummary(records, {
      armPairs: [["ours", "wllama-per-tab"]],
      expected: 2
    });
    const overall = summary.pairwise.find((row: { pressureId: string }) => row.pressureId === "overall");

    expect(summary.completeness).toMatchObject({ expected: 2, observed: 2, complete: true });
    expect(summary.clickResponse).toMatchObject({
      role: "exploratory",
      gate: false,
      attempts: 4,
      valid: 4,
      tasksAttempted: 2,
      tasksValid: 2
    });
    expect(summary.clickResponse.byArmPressure).toHaveLength(2);
    expect(overall.metrics.clickResponseMs).toMatchObject({
      n: 1,
      medianDifference: -80
    });
    expect(overall.metrics.clickFrameResponseMs.medianDifference).toBe(-60);
  });
});

function runSequence(seed: number): number[] {
  const sequencer = createClickSequencer({
    indices: [0, 1, 2, 3],
    seed,
    initialDelayMs: 0,
    minIntervalMs: 1,
    maxIntervalMs: 2,
    maxClicks: 10
  });
  const sequence = [];
  for (let index = 0; index < 10; index += 1) {
    const value = sequencer.next([0, 1, 2, 3]);
    if (value === null) break;
    sequence.push(value);
  }
  return sequence;
}

function validSample(overrides: Record<string, unknown>) {
  return {
    error: null,
    visibilityBefore: "hidden",
    visibilityState: "visible",
    visibilityChanged: true,
    responseFrames: 2,
    framesWaited: 2,
    clickResponseMs: 100,
    clickActivationMs: 25,
    clickFrameResponseMs: 75,
    ...overrides
  };
}

function taskRecord(
  arm: string,
  clickResponseMs: number,
  clickResponseValid: number,
  clickResponses: Array<Record<string, unknown>>
) {
  return {
    kind: "task",
    model: "llama-3.2-1b-instruct",
    platform: "win32",
    pressureId: "pressure-1000",
    pressureTriangles: 1000,
    repeat: 0,
    app: "artifex",
    taskId: "artifex-identity",
    role: "background",
    phase: "priority",
    arm,
    clickResponseAttempts: clickResponses.length,
    clickResponseValid,
    clickResponseMs,
    clickActivationMs: arm === "ours" ? 25 : 45,
    clickFrameResponseMs: arm === "ours" ? 55 : 115,
    clickResponses,
    error: null
  };
}
