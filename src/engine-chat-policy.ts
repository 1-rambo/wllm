export type EngineChatLifecycleState =
  | 'active'
  | 'passive'
  | 'hidden'
  | 'frozen';

export interface EngineChatPriorityView {
  foreground: boolean;
  overdue: boolean;
  waitMs: number;
  estimatedServiceMs: number;
  createdAt: number;
  id: number;
}

export type EngineChatRetentionPolicy =
  | 'evict-on-pause'
  | 'retain-until-full'
  | 'lru'
  | 'ttl'
  | 'lifecycle-aware';

export const shouldProtectSession = (
  policy: EngineChatRetentionPolicy,
  lifecycle:
    | {
        foreground: boolean;
        lifecycleState: EngineChatLifecycleState;
        changedAt: number;
        lastForegroundAt: number;
      }
    | undefined,
  now: number,
  ttlMs: number
): boolean => {
  if (!lifecycle || policy === 'lru') return false;
  if (policy === 'retain-until-full') return true;
  if (policy === 'evict-on-pause') return lifecycle.foreground;
  if (policy === 'ttl') {
    return lifecycle.foreground || now - lifecycle.changedAt <= ttlMs;
  }
  return (
    lifecycle.foreground ||
    lifecycle.lifecycleState === 'passive' ||
    now - lifecycle.lastForegroundAt <= ttlMs
  );
};

export const decodeSliceBudget = (
  baseBudget: number,
  remainingTokens: number,
  foreground: boolean,
  foregroundPending: boolean,
  foregroundBudget?: number,
  backgroundBudget?: number
): number => {
  const normalizedBase = Math.max(1, Math.floor(baseBudget));
  const selected =
    foreground && foregroundBudget !== undefined
      ? Math.max(1, Math.floor(foregroundBudget))
      : !foreground && foregroundPending
        ? Math.max(
            1,
            Math.floor(backgroundBudget ?? Math.min(16, normalizedBase))
          )
        : normalizedBase;
  return Number.isFinite(remainingTokens)
    ? Math.min(selected, Math.max(0, Math.floor(remainingTokens)))
    : selected;
};

export const compareEngineChatPriority = (
  left: EngineChatPriorityView,
  right: EngineChatPriorityView,
  foregroundAware: boolean
): number => {
  if (foregroundAware && left.foreground !== right.foreground) {
    return left.foreground ? -1 : 1;
  }
  if (left.overdue !== right.overdue) {
    return left.overdue ? -1 : 1;
  }
  if (left.overdue && left.waitMs !== right.waitMs) {
    return right.waitMs - left.waitMs;
  }
  if (left.estimatedServiceMs !== right.estimatedServiceMs) {
    return left.estimatedServiceMs - right.estimatedServiceMs;
  }
  if (left.waitMs !== right.waitMs) {
    return right.waitMs - left.waitMs;
  }
  if (left.createdAt !== right.createdAt) {
    return left.createdAt - right.createdAt;
  }
  return left.id - right.id;
};
