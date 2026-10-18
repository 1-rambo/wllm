import type {
  EnqueueResult,
  LifecycleRecord,
  ClientId,
  RequestId,
  RequestRecord,
  SchedulerConfig,
  SelectionResult,
  SessionKey,
} from '../contracts';

const DEFAULT_CONFIG: SchedulerConfig = {
  maxPending: 64,
  waitBaseMs: 25,
  waitDepthFactor: 15,
  serviceBoundMs: 300,
  waitBudgetCapMs: 5_000,
  graceMs: 40,
  retentionWindowMs: 30_000,
  maxProtectedHeads: 4,
  estimateWarmupRequests: 3,
  priorityFairnessMs: 750,
  priorityHandoffMs: 60_000,
  priorityDecayMs: 5_000,
  highPriorityBurstUnits: 2,
  residentBurstUnits: 1,
};

export class PriorityAwareScheduler {
  readonly config: SchedulerConfig;
  private readonly lifecycleByClient = new Map<string, LifecycleRecord>();
  private readonly lastHighPriorityAtByClient = new Map<ClientId, number>();
  private readonly lastPromotionAtByClient = new Map<ClientId, number>();
  private readonly requests = new Map<RequestId, RequestRecord>();
  private readonly sessionsByClient = new Map<string, Map<string, SessionKey>>();
  private readonly normalQueue: RequestId[] = [];
  private readonly overdueQueue: RequestId[] = [];
  private readonly completed = new Set<RequestId>();
  private readonly cancelled = new Set<RequestId>();
  private activeRequestId: RequestId | null = null;
  private graceUntil = 0;
  private serviceSamples = 0;
  // Burst accounting for the high-priority lane, so one long high-priority
  // request cannot starve every normal request indefinitely.
  private lastSelectedRequestId: RequestId | null = null;
  private consecutiveSelections = 0;
  private yieldHighPriority = false;

  constructor(config: Partial<SchedulerConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  updateLifecycle(record: LifecycleRecord, now = record.changedAt): boolean {
    const previous = this.lifecycleByClient.get(record.clientId);
    if (previous && record.epoch <= previous.epoch) {
      return false;
    }
    const next = { ...record };
    this.lifecycleByClient.set(record.clientId, next);

    // Priority is explicit and per session: several sessions may hold `high`
    // at the same time, so a promotion never demotes another session.
    const becameHigh =
      previous !== undefined && next.foreground && !previous.foreground;
    // Keep the last lifecycle update as the retention anchor. The separate
    // promotion map below changes only on an actual normal-to-high edge.
    this.lastHighPriorityAtByClient.set(next.clientId, now);
    if (becameHigh) {
      this.lastPromotionAtByClient.set(next.clientId, now);
    } else if (!next.foreground) {
      this.lastPromotionAtByClient.delete(next.clientId);
    }
    if (next.foreground) {
      this.graceUntil = Math.max(this.graceUntil, now + this.config.graceMs);
    }

    for (const request of this.requests.values()) {
      if (request.session.clientId === next.clientId) {
        if (next.foreground || request.lifecycle.epoch <= next.epoch) {
          request.lifecycle = { ...next };
        }
        if (becameHigh) {
          // A promotion owes its caller progress. Until this session produces
          // a token the debt outranks ordinary work.
          request.foregroundAtMs = now;
        }
        continue;
      }
    }
    return true;
  }

  enqueue(request: RequestRecord, now = request.enqueuedAt): EnqueueResult {
    if (this.requests.has(request.requestId)) {
      return { accepted: false, reason: 'duplicate' };
    }
    const lifecycle = this.lifecycleByClient.get(request.session.clientId);
    if (lifecycle && request.lifecycle.epoch < lifecycle.epoch) {
      return { accepted: false, reason: 'stale_lifecycle' };
    }
    if (this.requests.size >= this.config.maxPending) {
      return { accepted: false, reason: 'pending_limit' };
    }

    request.lifecycle = lifecycle
      ? { ...lifecycle }
      : { ...request.lifecycle };
    // The application may submit its first request after the priority write.
    // Carry the original transition time into that request so it still gets
    // the activation hand-off instead of looking like unrelated high-priority
    // work.
    if (request.lifecycle.foreground && request.foregroundAtMs === undefined) {
      const promotedAt = this.lastPromotionAtByClient.get(
        request.session.clientId
      );
      if (promotedAt !== undefined) {
        request.foregroundAtMs = promotedAt;
      }
    }
    request.enqueuedAt = now;
    request.waitBudgetMs = this.computeWaitBudget(request, now);
    request.serviceEstimateMs = this.estimateService(request);
    this.requests.set(request.requestId, request);
    this.rememberSession(request.session);
    this.normalQueue.push(request.requestId);
    return { accepted: true };
  }

  requeue(requestId: RequestId, now: number): boolean {
    const request = this.requests.get(requestId);
    if (!request) {
      return false;
    }
    if (this.activeRequestId === requestId) {
      this.activeRequestId = null;
    }
    this.removeFromQueues(requestId);
    request.enqueuedAt = now;
    request.waitBudgetMs = this.computeWaitBudget(request, now);
    request.serviceEstimateMs = this.estimateService(request);
    this.normalQueue.push(requestId);
    return true;
  }

  /**
   * Record that a unit produced a decode token for this request. It settles
   * the promotion debt: the caller has seen progress in the session it
   * promoted, so the scheduler stops holding a priority slot for it.
   */
  noteDecodeProgress(requestId: RequestId, now: number): void {
    const request = this.requests.get(requestId);
    if (!request) {
      return;
    }
    request.lastDecodeAtMs = now;
  }

  /**
   * True while a promoted session still owes its next token. The coordinator
   * uses it to isolate that acknowledgment from ordinary batched work without
   * cutting the promotion's first burst down to a single-token graph.
   */
  promotionPending(requestId: RequestId, now: number): boolean {
    const request = this.requests.get(requestId);
    if (!request) {
      return false;
    }
    const rank = this.priorityRank(request, now);
    return rank <= 1;
  }


  /**
   * True while a request belongs in the high-priority serving lane. Unlike
   * `promotionPending`, this remains true after the first token so an
   * answered promotion keeps its small batch instead of falling into the
   * ordinary background graph.
   */
  priorityServing(requestId: RequestId, now: number): boolean {
    const request = this.requests.get(requestId);
    return request !== undefined && this.priorityRank(request, now) <= 2;
  }

  /**
   * True while a promoted client still owes its next token. This remains
   * true across batch boundaries, including when the promoted request has
   * not been submitted yet. A long pressured prefill may therefore keep
   * activation preemption armed until the client actually makes progress.
   */
  hasPromotionDebt(clientId: ClientId, now: number): boolean {
    if (!this.lifecycleByClient.get(clientId)?.foreground) {
      return false;
    }
    const promotedAt = this.lastPromotionAtByClient.get(clientId);
    if (
      promotedAt === undefined ||
      now - promotedAt > this.config.priorityHandoffMs
    ) {
      return false;
    }
    return ![...this.requests.values()].some(
      (request) =>
        request.session.clientId === clientId &&
        request.lastDecodeAtMs !== undefined &&
        request.lastDecodeAtMs > promotedAt
    );
  }

  /**
   * Pending requests that still owe the human a token. These are the
   * requests the coordinator may advance in an express unit while an
   * ordinary background graph is already in flight.
   */
  pendingAcknowledgements(now: number): RequestRecord[] {
    return [...this.requests.values()]
      .filter((request) => this.priorityRank(request, now) <= 1)
      .sort((a, b) => this.compareRequests(a, b, now));
  }

  /**
   * Remove one specific request from the scheduler queues so an express
   * unit can own it until it reaches its next token boundary. Returns null
   * when the request is no longer queued (for example because the regular
   * pump admitted it first).
   */
  claim(requestId: RequestId): RequestRecord | null {
    const request = this.requests.get(requestId);
    if (!request) {
      return null;
    }
    if (
      !this.normalQueue.includes(requestId) &&
      !this.overdueQueue.includes(requestId)
    ) {
      return null;
    }
    if (
      request.dependencyRequestIds.some((dependencyId) =>
        this.isPending(dependencyId)
      )
    ) {
      return null;
    }
    this.removeFromQueues(requestId);
    return request;
  }

  selectNext(now: number): SelectionResult | null {
    this.promoteOverdue(now);
    const highPriorityPending = this.hasPendingHighPriority();
    if (!highPriorityPending && now < this.graceUntil) {
      return null;
    }

    const candidates = this.currentCandidates();
    const runnable: RequestRecord[] = [];
    const blocked: Array<{ request: RequestRecord; blockedBy: RequestId[] }> =
      [];

    for (const request of candidates) {
      const blockers = request.dependencyRequestIds.filter(
        (dependencyId) => this.isPending(dependencyId)
      );
      if (blockers.length === 0) {
        runnable.push(request);
      } else {
        blocked.push({ request, blockedBy: blockers });
      }
    }

    const pool = runnable.length > 0 ? runnable : blocked.map((item) => item.request);
    if (pool.length === 0) {
      return null;
    }
    // Once the high-priority lane has had its burst, give the head start to
    // normal work for one unit. A request that still owes its promotion debt
    // is never deferred this way.
    this.yieldHighPriority =
      this.consecutiveSelections >= Math.max(1, this.config.highPriorityBurstUnits);
    pool.sort((a, b) => this.compareRequests(a, b, now));
    const selected = pool[0];
    if (!selected) {
      return null;
    }
    this.consecutiveSelections =
      selected.requestId === this.lastSelectedRequestId
        ? this.consecutiveSelections + 1
        : 1;
    this.lastSelectedRequestId = selected.requestId;
    const blockedBy =
      blocked.find((item) => item.request.requestId === selected.requestId)
        ?.blockedBy ?? [];
    if (blockedBy.length === 0 && this.activeRequestId === null) {
      this.removeFromQueues(selected.requestId);
      this.activeRequestId = selected.requestId;
    }
    return {
      request: selected,
      runnable: blockedBy.length === 0,
      blockedBy,
    };
  }

  /**
   * Remove and return up to `limit` additional runnable requests that
   * satisfy `filter`. This is the admission side of cooperative batching:
   * callers first select the head request, then take peers of the same
   * lifecycle class and stage so one graph admission can cover the group.
   * The returned requests are removed from the scheduler queues until the
   * caller requeues or completes them.
   */
  selectBatch(
    now: number,
    limit: number,
    filter?: (request: RequestRecord) => boolean
  ): RequestRecord[] {
    const count = Math.max(0, Math.floor(limit));
    if (count === 0) {
      return [];
    }
    this.promoteOverdue(now);
    const highPriorityPending = this.hasPendingHighPriority();
    if (!highPriorityPending && now < this.graceUntil) {
      return [];
    }

    const runnable = this.currentCandidates()
      .filter((request) => filter === undefined || filter(request))
      .filter((request) =>
        request.dependencyRequestIds.every(
          (dependencyId) => !this.isPending(dependencyId)
        )
      )
      .sort((a, b) => this.compareRequests(a, b, now))
      .slice(0, count);
    for (const request of runnable) {
      this.removeFromQueues(request.requestId);
    }
    return runnable;
  }

  complete(requestId: RequestId): void {
    this.removeFromQueues(requestId);
    this.requests.delete(requestId);
    this.completed.add(requestId);
    if (this.activeRequestId === requestId) {
      this.activeRequestId = null;
    }
    this.serviceSamples += 1;
  }

  cancel(requestId: RequestId): boolean {
    const existed = this.requests.delete(requestId);
    this.removeFromQueues(requestId);
    if (this.activeRequestId === requestId) {
      this.activeRequestId = null;
    }
    if (existed) {
      this.cancelled.add(requestId);
    }
    return existed;
  }

  get(requestId: RequestId): RequestRecord | undefined {
    return this.requests.get(requestId);
  }

  get activeId(): RequestId | null {
    return this.activeRequestId;
  }

  lifecycleFor(clientId: string): LifecycleRecord | null {
    const lifecycle = this.lifecycleByClient.get(clientId);
    return lifecycle ? { ...lifecycle } : null;
  }

  get pendingCount(): number {
    return this.requests.size;
  }

  pendingRequests(): RequestRecord[] {
    return [...this.requests.values()];
  }

  completedRequests(): RequestId[] {
    return [...this.completed];
  }

  cancelledRequests(): RequestId[] {
    return [...this.cancelled];
  }

  removeClient(clientId: string): void {
    this.lifecycleByClient.delete(clientId);
    this.lastHighPriorityAtByClient.delete(clientId);
    this.lastPromotionAtByClient.delete(clientId);
    this.sessionsByClient.delete(clientId);
  }

  protectedSessionKeys(now: number): SessionKey[] {
    const bySession = new Map<string, { key: SessionKey; score: number }>();
    for (const sessions of this.sessionsByClient.values()) {
      for (const key of sessions.values()) {
        const lifecycle = this.lifecycleByClient.get(key.clientId);
        if (!lifecycle) {
          continue;
        }
        const lastHighPriorityAt = this.lastHighPriorityAtByClient.get(
          key.clientId
        );
        const promotedRecently =
          lastHighPriorityAt !== undefined &&
          now - lastHighPriorityAt <= this.config.retentionWindowMs;
        const eligible =
          lifecycle.foreground ||
          lifecycle.state === 'passive' ||
          promotedRecently;
        if (!eligible) {
          continue;
        }
        const keyString = `${key.clientId}\u0000${key.sessionId}`;
        const score =
          (lifecycle.foreground ? 1_000_000 : 0) +
          (lifecycle.state === 'active' ? 100_000 : 0) +
          (lastHighPriorityAt === undefined
            ? 0
            : Math.max(
                0,
                this.config.retentionWindowMs - (now - lastHighPriorityAt)
              ));
        const previous = bySession.get(keyString);
        if (!previous || previous.score < score) {
          bySession.set(keyString, { key, score });
        }
      }
    }
    return [...bySession.values()]
      .sort((a, b) => b.score - a.score)
      .slice(0, this.config.maxProtectedHeads)
      .map((entry) => entry.key);
  }

  hasPendingHighPriority(): boolean {
    return [...this.requests.values()].some(
      (request) => request.lifecycle.foreground
    );
  }

  private currentCandidates(): RequestRecord[] {
    const ids = [...this.overdueQueue, ...this.normalQueue];
    const unique = new Set(ids);
    if (this.activeRequestId) {
      unique.delete(this.activeRequestId);
    }
    return [...unique]
      .map((id) => this.requests.get(id))
      .filter((request): request is RequestRecord => request !== undefined);
  }

  private promoteOverdue(now: number): void {
    const remaining: RequestId[] = [];
    for (const requestId of this.normalQueue) {
      const request = this.requests.get(requestId);
      if (!request) {
        continue;
      }
      if (now - request.enqueuedAt >= request.waitBudgetMs) {
        if (!this.overdueQueue.includes(requestId)) {
          this.overdueQueue.push(requestId);
        }
      } else {
        remaining.push(requestId);
      }
    }
    this.normalQueue.splice(0, this.normalQueue.length, ...remaining);
  }

  private compareRequests(
    a: RequestRecord,
    b: RequestRecord,
    now: number
  ): number {
    const aAttention = this.priorityRank(a, now);
    const bAttention = this.priorityRank(b, now);
    const aRank = this.priorityRank(a, now);
    const bRank = this.priorityRank(b, now);
    if (aRank !== bRank) {
      return aRank - bRank;
    }
    if (aRank <= 1) {
      // Two unanswered promotions: the older one answers first so a burst of
      // promotions drains in the order the application issued them.
      const aPromoted = a.foregroundAtMs ?? 0;
      const bPromoted = b.foregroundAtMs ?? 0;
      if (aPromoted !== bPromoted) {
        return aPromoted - bPromoted;
      }
    }
    const aResident = a.nativeRequestId !== null;
    const bResident = b.nativeRequestId !== null;
    // Once a resident sequence starts a unit, let it keep its live slot
    // for a bounded run. Without this, every overdue cold request can
    // replace the resident on the next token and pay a checkpoint restore
    // after every tiny native unit. Foreground activation classes still
    // outrank this affinity because their priority ranks differ above.
    const residentBurstUnits = Math.max(
      1,
      Math.floor(this.config.residentBurstUnits)
    );
    if (
      aRank >= 2 &&
      bRank >= 2 &&
      this.lastSelectedRequestId !== null &&
      this.consecutiveSelections < residentBurstUnits
    ) {
      const aCurrentResident =
        aResident && a.requestId === this.lastSelectedRequestId;
      const bCurrentResident =
        bResident && b.requestId === this.lastSelectedRequestId;
      if (aCurrentResident !== bCurrentResident) {
        return aCurrentResident ? -1 : 1;
      }
    }
    // The priority rank already encodes the promoted session's preference
    // (2 for an answered promotion versus 3 for ordinary work), so the
    // remaining order is the wait-budget and service estimate below.
    const aOverdue = this.overdueQueue.includes(a.requestId);
    const bOverdue = this.overdueQueue.includes(b.requestId);
    if (aOverdue !== bOverdue) {
      return aOverdue ? -1 : 1;
    }
    if (aOverdue && bOverdue) {
      const waitDelta = this.wait(a, now) - this.wait(b, now);
      if (waitDelta !== 0) {
        return -waitDelta;
      }
    }
    // Cache affinity: a sequence that already owns a native slot should
    // finish its current decode burst before a cold session evicts it.
    // Without this, the shortest-work estimate can swap the resident set
    // on every token, turning a bounded L1 pool into checkpoint thrash.
    // A cold request that reaches its wait budget is handled by the overdue
    // branch above and still preempts, so affinity cannot starve it forever.
    if (aResident !== bResident) {
      return aResident ? -1 : 1;
    }
    const estimateDelta = a.serviceEstimateMs - b.serviceEstimateMs;
    if (estimateDelta !== 0) {
      return estimateDelta;
    }
    const waitDelta = this.wait(a, now) - this.wait(b, now);
    if (waitDelta !== 0) {
      return -waitDelta;
    }
    const creationDelta = a.createdAt - b.createdAt;
    if (creationDelta !== 0) {
      return creationDelta;
    }
    return a.requestId.localeCompare(b.requestId);
  }

  private wait(request: RequestRecord, now: number): number {
    return Math.max(0, now - request.enqueuedAt);
  }

  private computeWaitBudget(request: RequestRecord, now: number): number {
    const queueDepth = this.requests.size;
    const serviceBound = Math.max(
      this.config.serviceBoundMs,
      request.serviceEstimateMs
    );
    const depthPart = queueDepth * this.config.waitDepthFactor;
    const candidate = this.config.waitBaseMs + depthPart + serviceBound;
    return Math.min(this.config.waitBudgetCapMs, candidate);
  }

  private estimateService(request: RequestRecord): number {
    const requestCount = this.serviceSamples + 1;
    if (requestCount <= this.config.estimateWarmupRequests) {
      return this.config.serviceBoundMs;
    }
    const prefixCost = request.prefillProcessed * 1.1;
    const decodeCost = request.decodeGenerated * 1.4;
    return Math.max(1, prefixCost + decodeCost);
  }

  private isPending(requestId: RequestId): boolean {
    return this.requests.has(requestId);
  }

  /**
   * Priority rank for one request. Lower runs first:
   *  -1 - a deprioritised promotion that has waited past the fairness bound
   *   0 - a high-priority session that has not answered since its promotion
   *   1 - another promoted session that still owes its next token
   *   2 - a high-priority session that has already produced a token
   *   3 - everything else
   *
   * The two answer classes are what keep a promotion from being lost when it
   * lands in the middle of a prefill chunk and the developer promotes another
   * session before that chunk returns a token. Without them the session that
   * already answered runs the whole burst and the unanswered one waits behind
   * every later promotion.
   */
  private priorityRank(request: RequestRecord, now: number): number {
    const activatedAt = request.foregroundAtMs;
    const answeredAt = request.lastDecodeAtMs;
    const owed =
      activatedAt !== undefined &&
      (answeredAt === undefined || answeredAt < activatedAt);
    if (request.lifecycle.foreground) {
      if (owed) {
        return 0;
      }
      // The promoted session has answered. Its burst is bounded so normal
      // work still advances; after one normal unit it returns to its
      // ordinary priority.
      if (activatedAt !== undefined) {
        return this.yieldHighPriority &&
          request.requestId === this.lastSelectedRequestId
          ? 3
          : 2;
      }
      // Work queued into a session that was already foreground did not
      // consume a promotion edge, so retain the bounded burst behavior.
      return this.yieldHighPriority &&
        request.requestId === this.lastSelectedRequestId
        ? 3
        : 2;
    }
    if (!owed) {
      // A later session demotion affects new work, not the in-flight request
      // that already entered the serving path, but only for a bounded
      // hand-off: once the answering burst has had its decay window, a
      // request whose session is back to `normal` is ordinary work again.
      // Without the bound a session that was ever promoted holds the
      // priority lane for the rest of its request, so a chain of promotions
      // leaves several old targets competing with the current one and
      // spending the render budget on long priority-shaped units.
      if (activatedAt === undefined) {
        return 3;
      }
      const answeredForMs = now - (answeredAt ?? activatedAt);
      return answeredForMs > this.config.priorityDecayMs ? 3 : 2;
    }
    // A demoted promotion debt is only protected for a bounded hand-off
    // window: past it the request is ordinary work again, so one stuck
    // session cannot hold a priority slot forever. While the debt is still
    // current, an old promotion is raised after a bounded wait so a stream of
    // newer promotions cannot starve it indefinitely.
    const waitedMs = Math.max(0, now - (activatedAt ?? now));
    if (waitedMs > this.config.priorityHandoffMs) {
      return 3;
    }
    if (waitedMs >= this.config.priorityFairnessMs) {
      return -1;
    }
    return 1;
  }

  private removeFromQueues(requestId: RequestId): void {
    const normalIndex = this.normalQueue.indexOf(requestId);
    if (normalIndex >= 0) {
      this.normalQueue.splice(normalIndex, 1);
    }
    const overdueIndex = this.overdueQueue.indexOf(requestId);
    if (overdueIndex >= 0) {
      this.overdueQueue.splice(overdueIndex, 1);
    }
  }

  private rememberSession(key: SessionKey): void {
    let sessions = this.sessionsByClient.get(key.clientId);
    if (!sessions) {
      sessions = new Map();
      this.sessionsByClient.set(key.clientId, sessions);
    }
    sessions.set(key.sessionId, key);
  }
}
