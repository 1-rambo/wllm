import { describe, expect, it } from 'vitest';
import {
  contextWindowShape,
  estimateTextTokens,
  trimPrefixToTokenBudget,
} from '../../src/native/context-window';

/**
 * llama.cpp splits `n_ctx` across `n_parallel`, so the number a session can
 * actually use is the derived one. These tests pin the arithmetic and the
 * retention rule that keeps a long-lived conversation alive on a device
 * where the total context cannot grow without bound.
 */

describe('contextWindowShape', () => {
  it('divides the context the way llama.cpp does', () => {
    expect(
      contextWindowShape({ nCtx: 65_536, nParallel: 4, maxNewTokens: 256 })
    ).toMatchObject({
      nCtx: 65_536,
      nParallel: 4,
      perSlotTokens: 16_384,
      promptBudgetTokens: 16_384 - 256 - 64,
    });
  });

  it('reports the demo default as the 512-token slot it really is', () => {
    expect(
      contextWindowShape({ nCtx: 2_048, nParallel: 4, maxNewTokens: 8 })
    ).toMatchObject({ perSlotTokens: 512, promptBudgetTokens: 512 - 8 - 64 });
  });

  it('never returns a budget below one token', () => {
    expect(
      contextWindowShape({ nCtx: 512, nParallel: 4, maxNewTokens: 4_096 })
    ).toMatchObject({ perSlotTokens: 128, promptBudgetTokens: 1 });
  });

  it('treats a missing or nonsensical parallel count as one sequence', () => {
    expect(
      contextWindowShape({ nCtx: 8_192, nParallel: 0, maxNewTokens: 0 })
    ).toMatchObject({ nParallel: 1, perSlotTokens: 8_192 });
  });
});

describe('estimateTextTokens', () => {
  it('counts word and punctuation runs', () => {
    expect(estimateTextTokens('click(123)')).toBe(4);
    expect(estimateTextTokens('')).toBe(0);
    expect(estimateTextTokens('   ')).toBe(0);
  });
});

describe('trimPrefixToTokenBudget', () => {
  const turn = (index: number): string =>
    `turn ${index}\nGOAL\nbuy the cheapest red mug\nACCESSIBILITY TREE\n` +
    `${'link product '.repeat(20)}\n`;

  it('keeps the prefix untouched when it already fits', () => {
    const prefix = turn(1);
    expect(trimPrefixToTokenBudget(prefix, 1_000)).toEqual({
      text: prefix,
      droppedTokens: 0,
      droppedChars: 0,
    });
  });

  it('keeps the newest turns and starts on a line boundary', () => {
    const prefix = [turn(1), turn(2), turn(3), turn(4)].join('');
    // enough room for the newest turn plus a little slack
    const budget = estimateTextTokens(turn(4)) + 5;
    const trimmed = trimPrefixToTokenBudget(prefix, budget);

    expect(estimateTextTokens(trimmed.text)).toBeLessThanOrEqual(budget);
    expect(trimmed.droppedTokens).toBeGreaterThan(0);
    // the newest turn survives; the oldest does not
    expect(trimmed.text).toContain('turn 4');
    expect(trimmed.text).not.toContain('turn 1\n');
    // trimming must not leave half a record at the front
    expect(trimmed.text.startsWith('turn ')).toBe(true);
    // and the retained text is the tail of what was there before
    expect(prefix.endsWith(trimmed.text)).toBe(true);
  });

  it('drops everything when there is no room at all', () => {
    expect(trimPrefixToTokenBudget(turn(1), 0)).toMatchObject({ text: '' });
    expect(trimPrefixToTokenBudget(turn(1), -5).text).toBe('');
  });

  it('rechecks the estimate after the cut', () => {
    // punctuation-heavy text where characters per token is much lower than
    // prose, which is exactly where a single proportional cut overshoots
    const prefix = 'a, '.repeat(400);
    const trimmed = trimPrefixToTokenBudget(prefix, 25);
    expect(estimateTextTokens(trimmed.text)).toBeLessThanOrEqual(25);
  });
});
