import { describe, expect, it } from "vitest";
// @ts-expect-error The evaluation harness is plain ESM and has no ambient declarations.
import { normalizedTargets } from "../../evaluation/lib/targets.mjs";

describe("normalizedTargets", () => {
  const sequence = [0, 4, 7, 2, 9, 1, 6, 3, 8];

  it("deduplicates modulo collisions and includes the full target pool", () => {
    expect(normalizedTargets(sequence, 9, 9, 0)).toEqual([
      0, 4, 7, 2, 1, 6, 3, 8, 5
    ]);
  });

  it("rotates the selected target window across trials", () => {
    expect(normalizedTargets(sequence, 9, 3, 0)).toEqual([0, 4, 7]);
    expect(normalizedTargets(sequence, 9, 3, 1)).toEqual([4, 7, 2]);
    expect(normalizedTargets(sequence, 9, 3, 6)).toEqual([3, 8, 5]);
  });

  it("handles negative rotations and zero-sized pools", () => {
    expect(normalizedTargets(sequence, 9, 3, -1)).toEqual([5, 0, 4]);
    expect(normalizedTargets(sequence, 0, 3, 0)).toEqual([]);
  });
});
