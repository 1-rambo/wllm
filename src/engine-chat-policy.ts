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

export interface EngineChatRetentionCandidate {
  nodeId: number;
  lifecycle:
    | {
        foreground: boolean;
        lifecycleState: EngineChatLifecycleState;
        changedAt: number;
        lastForegroundAt: number;
      }
    | undefined;
}

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

export const selectProtectedSessionNodeIds = (
  policy: EngineChatRetentionPolicy,
  candidates: EngineChatRetentionCandidate[],
  now: number,
  ttlMs: number,
  limit: number
): number[] => {
  if (policy === 'lru' || policy === 'evict-on-pause' || limit <= 0) return [];
  const eligible = candidates.filter((candidate) =>
    shouldProtectSession(policy, candidate.lifecycle, now, ttlMs)
  );
  if (policy === 'retain-until-full') {
    return eligible.map((candidate) => candidate.nodeId);
  }
  const lifecycleRank = (candidate: EngineChatRetentionCandidate): number => {
    if (policy !== 'lifecycle-aware') return 0;
    if (candidate.lifecycle?.foreground) return 4;
    if (candidate.lifecycle?.lifecycleState === 'active') return 3;
    if (candidate.lifecycle?.lifecycleState === 'passive') return 2;
    if (candidate.lifecycle?.lifecycleState === 'hidden') return 1;
    return 0;
  };
  return eligible
    .sort((left, right) => {
      const rankDelta = lifecycleRank(right) - lifecycleRank(left);
      if (rankDelta !== 0) return rankDelta;
      const foregroundDelta =
        (right.lifecycle?.lastForegroundAt ?? 0) -
        (left.lifecycle?.lastForegroundAt ?? 0);
      if (foregroundDelta !== 0) return foregroundDelta;
      const changedDelta =
        (right.lifecycle?.changedAt ?? 0) - (left.lifecycle?.changedAt ?? 0);
      if (changedDelta !== 0) return changedDelta;
      return left.nodeId - right.nodeId;
    })
    .slice(0, limit)
    .map((candidate) => candidate.nodeId);
};

export const combineHotNodeIds = (
  lifecycleNodeIds: number[],
  requestParentNodeIds: number[]
): number[] => {
  const result: number[] = [];
  const seen = new Set<number>();
  for (const nodeId of [...lifecycleNodeIds, ...requestParentNodeIds]) {
    if (nodeId < 1 || seen.has(nodeId)) continue;
    seen.add(nodeId);
    result.push(nodeId);
  }
  return result;
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
