import type {
  LifecycleRecord,
  RequestRecord,
  SessionKey,
  SessionNode,
} from '../../src/contracts';

export const fixedClock = {
  now: () => 1_000,
};

export function session(clientId: string, sessionId = 'main'): SessionKey {
  return { clientId, sessionId };
}

export function lifecycle(
  clientId: string,
  options: Partial<Omit<LifecycleRecord, 'clientId'>> = {}
): LifecycleRecord {
  return {
    clientId,
    epoch: 1,
    state: 'active',
    foreground: true,
    changedAt: 0,
    ...options,
  };
}

export function request(
  requestId: string,
  options: Partial<Omit<RequestRecord, 'requestId'>> = {}
): RequestRecord {
  const clientId =
    options.lifecycle?.clientId ??
    options.session?.clientId ??
    requestId.split(':')[0] ??
    'c1';
  return {
    requestId,
    session: session(clientId),
    parentNodeId: null,
    dependencyRequestIds: [],
    promptTokens: [1, 2, 3, 4],
    maxNewTokens: 4,
    lifecycle: lifecycle(clientId),
    createdAt: 0,
    enqueuedAt: 0,
    waitBudgetMs: 0,
    serviceEstimateMs: 0,
    serviceMs: 0,
    stage: 'prefill',
    prefillProcessed: 0,
    decodeGenerated: 0,
    tokenAccounting: 'native-token',
    nativeRequestId: null,
    slotId: null,
    outputChunks: [],
    ...options,
  };
}

export function node(
  nodeId: string,
  options: Partial<Omit<SessionNode, 'nodeId'>> = {}
): SessionNode {
  const clientId = options.session?.clientId ?? 'c1';
  return {
    nodeId,
    session: session(clientId),
    parentNodeId: null,
    turnText: nodeId,
    tokenIds: [1, 2, 3, 4],
    createdAt: 0,
    lastAccessAt: 0,
    accessCount: 0,
    ...options,
  };
}
