import { describe, expect, test } from 'vitest';
import {
  compareEngineChatPriority,
  decodeSliceBudget,
  selectProtectedSessionNodeIds,
  shouldProtectSession,
} from './engine-chat-policy';

describe('decodeSliceBudget', () => {
  test('bounds finite generations by the configured slice', () => {
    expect(decodeSliceBudget(32, 100, false, false)).toBe(32);
    expect(decodeSliceBudget(32, 12, false, false)).toBe(12);
  });

  test('shortens background work while foreground work is waiting', () => {
    expect(decodeSliceBudget(128, 100, false, true, undefined, 8)).toBe(8);
    expect(decodeSliceBudget(128, 100, true, true, 64, 8)).toBe(64);
  });
});

describe('compareEngineChatPriority', () => {
  const base = {
    overdue: false,
    waitMs: 10,
    estimatedServiceMs: 20,
    createdAt: 1,
    id: 1,
  };

  test('selects foreground work first in foreground-aware mode', () => {
    expect(
      compareEngineChatPriority(
        { ...base, foreground: true },
        { ...base, foreground: false, id: 2 },
        true
      )
    ).toBeLessThan(0);
  });

  test('preserves overdue and age ordering inside a priority class', () => {
    expect(
      compareEngineChatPriority(
        { ...base, foreground: true, overdue: true, waitMs: 100 },
        { ...base, foreground: true, overdue: false, id: 2 },
        true
      )
    ).toBeLessThan(0);
  });
});

describe('shouldProtectSession', () => {
  const lifecycle = {
    foreground: false,
    lifecycleState: 'hidden' as const,
    changedAt: 90,
    lastForegroundAt: 80,
  };

  test('retains active TTL sessions and expires hidden ones', () => {
    expect(
      shouldProtectSession('ttl', { ...lifecycle, foreground: true }, 200, 10)
    ).toBe(true);
    expect(shouldProtectSession('ttl', lifecycle, 200, 10)).toBe(false);
  });

  test('uses recency and passive state for lifecycle-aware retention', () => {
    expect(
      shouldProtectSession(
        'lifecycle-aware',
        { ...lifecycle, lifecycleState: 'passive' },
        200,
        10
      )
    ).toBe(true);
    expect(shouldProtectSession('lifecycle-aware', lifecycle, 200, 10)).toBe(
      false
    );
  });
});

describe('selectProtectedSessionNodeIds', () => {
  const now = 1000;
  const hidden = (nodeId: number, lastForegroundAt: number) => ({
    nodeId,
    lifecycle: {
      foreground: false,
      lifecycleState: 'hidden' as const,
      changedAt: lastForegroundAt,
      lastForegroundAt,
    },
  });

  test('prioritizes passive sessions over newer hidden sessions', () => {
    const selected = selectProtectedSessionNodeIds(
      'lifecycle-aware',
      [
        hidden(1, 900),
        {
          nodeId: 2,
          lifecycle: {
            foreground: false,
            lifecycleState: 'passive',
            changedAt: 800,
            lastForegroundAt: 800,
          },
        },
      ],
      now,
      500,
      1
    );
    expect(selected).toEqual([2]);
  });

  test('uses recency to choose among TTL-protected sessions', () => {
    const selected = selectProtectedSessionNodeIds(
      'ttl',
      [hidden(1, 700), hidden(2, 950)],
      now,
      500,
      1
    );
    expect(selected).toEqual([2]);
  });
});
