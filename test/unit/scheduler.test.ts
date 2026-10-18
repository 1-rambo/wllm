import { describe, expect, it } from 'vitest';
import { PriorityAwareScheduler } from '../../src/scheduler/priority-aware-scheduler';
import { lifecycle, request, session } from '../helpers/factories';

describe('PriorityAwareScheduler', () => {
  it('rejects stale lifecycle epochs and updates queued requests', () => {
    const scheduler = new PriorityAwareScheduler({ graceMs: 0 });
    const current = lifecycle('c1', { epoch: 2, foreground: false });
    expect(scheduler.updateLifecycle(current, 10)).toBe(true);
    expect(
      scheduler.updateLifecycle(
        lifecycle('c1', { epoch: 1, foreground: true }),
        11
      )
    ).toBe(false);

    const stale = request('r1', {
      lifecycle: lifecycle('c1', { epoch: 1, foreground: true }),
    });
    expect(scheduler.enqueue(stale, 12)).toEqual({
      accepted: false,
      reason: 'stale_lifecycle',
    });

    const accepted = request('r2', {
      lifecycle: lifecycle('c1', { epoch: 2, foreground: false }),
    });
    expect(scheduler.enqueue(accepted, 12).accepted).toBe(true);

    scheduler.updateLifecycle(
      lifecycle('c1', { epoch: 3, foreground: true, state: 'active' }),
      13
    );
    expect(scheduler.get('r2')?.lifecycle).toMatchObject({
      epoch: 3,
      foreground: true,
      state: 'active',
    });
  });

  it('keeps concurrent high-priority sessions independent', () => {
    const scheduler = new PriorityAwareScheduler({ graceMs: 0 });
    scheduler.updateLifecycle(
      lifecycle('tab-a', { epoch: 2, foreground: true, state: 'active' }),
      0
    );
    scheduler.enqueue(
      request('tab-a:r1', {
        lifecycle: lifecycle('tab-a', { epoch: 2, foreground: true }),
      }),
      1
    );

    scheduler.updateLifecycle(
      lifecycle('tab-b', { epoch: 2, foreground: true, state: 'active' }),
      2
    );

    // Promoting one session must not demote another: the runtime has no
    // notion of a single foreground document any more.
    expect(scheduler.lifecycleFor('tab-a')).toMatchObject({
      foreground: true,
      state: 'active',
    });
    expect(scheduler.get('tab-a:r1')?.lifecycle).toMatchObject({
      foreground: true,
      state: 'active',
    });
    expect(scheduler.hasPendingHighPriority()).toBe(true);

    scheduler.enqueue(request('tab-b:r1'), 3);
    // Both sessions owe an unanswered promotion, so the older one drains
    // first and the newer one is not starved behind it.
    expect(scheduler.selectNext(4)?.request.requestId).toBe('tab-a:r1');
  });

  it('honors foreground grace and then prioritizes foreground work', () => {
    const scheduler = new PriorityAwareScheduler({ graceMs: 50 });
    scheduler.updateLifecycle(lifecycle('bg', { foreground: false }), 0);
    scheduler.enqueue(request('bg:r1', { createdAt: 0 }), 0);
    scheduler.updateLifecycle(lifecycle('fg', { foreground: true }), 100);

    expect(scheduler.selectNext(120)).toBeNull();
    scheduler.enqueue(request('fg:r1', { createdAt: 1 }), 121);
    expect(scheduler.selectNext(151)?.request.requestId).toBe('fg:r1');
  });

  it('does not let a blocked foreground request bypass a runnable dependency', () => {
    const scheduler = new PriorityAwareScheduler({ graceMs: 0 });
    scheduler.updateLifecycle(lifecycle('bg', { foreground: false }), 0);
    scheduler.updateLifecycle(lifecycle('fg', { foreground: true }), 0);

    scheduler.enqueue(request('dep', { session: session('bg') }), 0);
    scheduler.enqueue(
      request('child', {
        session: session('fg'),
        dependencyRequestIds: ['dep'],
      }),
      0
    );

    const first = scheduler.selectNext(1);
    expect(first).toMatchObject({
      runnable: true,
      request: { requestId: 'dep' },
    });
    scheduler.complete('dep');
    expect(scheduler.selectNext(2)).toMatchObject({
      runnable: true,
      request: { requestId: 'child' },
    });
  });

  it('promotes requests whose wait budget is exhausted', () => {
    const scheduler = new PriorityAwareScheduler({
      graceMs: 0,
      waitBaseMs: 1,
      waitDepthFactor: 0,
      serviceBoundMs: 1,
      waitBudgetCapMs: 10,
    });
    scheduler.updateLifecycle(lifecycle('c1', { foreground: false }), 0);
    scheduler.enqueue(request('old', { createdAt: 0 }), 0);
    scheduler.enqueue(request('new', { createdAt: 1 }), 10);

    expect(scheduler.selectNext(20)?.request.requestId).toBe('old');
  });

  it('keeps a resident sequence ahead of cold work until the cold wait budget expires', () => {
    const scheduler = new PriorityAwareScheduler({
      graceMs: 0,
      waitBaseMs: 10,
      waitDepthFactor: 0,
      serviceBoundMs: 10,
      waitBudgetCapMs: 100,
    });
    scheduler.updateLifecycle(lifecycle('c1', { foreground: false }), 0);
    scheduler.enqueue(request('resident'), 0);
    scheduler.enqueue(request('cold'), 0);
    const resident = scheduler.get('resident');
    if (resident) resident.nativeRequestId = 'resident';

    expect(scheduler.selectNext(1)?.request.requestId).toBe('resident');
    scheduler.requeue('resident', 1);
    expect(scheduler.selectNext(2)?.request.requestId).toBe('resident');
    scheduler.requeue('resident', 2);

    // The cold request has now waited past its budget, so affinity cannot
    // turn into starvation.
    expect(scheduler.selectNext(200)?.request.requestId).toBe('cold');
  });

  it('bounds resident affinity so overdue cold work eventually preempts', () => {
    const scheduler = new PriorityAwareScheduler({
      graceMs: 0,
      waitBaseMs: 1,
      waitDepthFactor: 0,
      serviceBoundMs: 1,
      waitBudgetCapMs: 1,
      residentBurstUnits: 3,
    });
    scheduler.updateLifecycle(lifecycle('c1', { foreground: false }), 0);
    const resident = request('resident', { session: session('c1') });
    scheduler.enqueue(resident, 0);
    scheduler.enqueue(request('cold', { session: session('c1') }), 0);
    resident.nativeRequestId = 'resident';

    expect(scheduler.selectNext(1)?.request.requestId).toBe('resident');
    scheduler.requeue('resident', 1);
    expect(scheduler.selectNext(2)?.request.requestId).toBe('resident');
    scheduler.requeue('resident', 2);
    expect(scheduler.selectNext(3)?.request.requestId).toBe('resident');
    scheduler.requeue('resident', 3);

    // The burst is exhausted, so the overdue cold request can take over.
    expect(scheduler.selectNext(4)?.request.requestId).toBe('cold');
    scheduler.requeue('cold', 4);
    const cold = scheduler.get('cold');
    if (cold) cold.nativeRequestId = 'cold';
    // The new resident starts its own bounded burst.
    expect(scheduler.selectNext(5)?.request.requestId).toBe('cold');
  });

  it('lets a foreground activation preempt a resident burst', () => {
    const scheduler = new PriorityAwareScheduler({
      graceMs: 0,
      waitBaseMs: 1,
      waitDepthFactor: 0,
      serviceBoundMs: 1,
      waitBudgetCapMs: 1,
      residentBurstUnits: 8,
    });
    scheduler.updateLifecycle(lifecycle('bg', { foreground: false }), 0);
    scheduler.updateLifecycle(lifecycle('fg', { foreground: false }), 0);
    const resident = request('resident', { session: session('bg') });
    scheduler.enqueue(resident, 0);
    scheduler.enqueue(
      request('fg:r1', {
        session: session('fg'),
        lifecycle: lifecycle('fg', { foreground: false }),
      }),
      0
    );
    resident.nativeRequestId = 'resident';
    expect(scheduler.selectNext(1)?.request.requestId).toBe('resident');
    scheduler.requeue('resident', 1);

    scheduler.updateLifecycle(
      lifecycle('fg', { epoch: 2, foreground: true }),
      2
    );
    expect(scheduler.selectNext(3)?.request.requestId).toBe('fg:r1');
  });

  it('keeps a completed session protectable during the retention window', () => {
    let now = 0;
    const scheduler = new PriorityAwareScheduler({
      graceMs: 0,
      retentionWindowMs: 100,
    });
    scheduler.updateLifecycle(
      lifecycle('c1', { foreground: true, changedAt: 0 }),
      0
    );
    scheduler.enqueue(request('r1', { session: session('c1', 'chat') }), now);
    expect(scheduler.protectedSessionKeys(now)).toEqual([
      session('c1', 'chat'),
    ]);

    scheduler.complete('r1');
    now = 50;
    expect(scheduler.protectedSessionKeys(now)).toEqual([
      session('c1', 'chat'),
    ]);
    scheduler.updateLifecycle(
      lifecycle('c1', { epoch: 2, foreground: false, state: 'hidden', changedAt: 100 }),
      100
    );
    expect(scheduler.protectedSessionKeys(200)).toEqual([
      session('c1', 'chat'),
    ]);
    expect(scheduler.protectedSessionKeys(201)).toEqual([]);
  });

  it('protects a recently hidden session for the retention window', () => {
    const scheduler = new PriorityAwareScheduler({
      graceMs: 0,
      retentionWindowMs: 100,
    });
    scheduler.updateLifecycle(
      lifecycle('c1', { foreground: true, changedAt: 0 }),
      0
    );
    scheduler.enqueue(request('r1', { session: session('c1', 'chat') }), 0);
    scheduler.updateLifecycle(
      lifecycle('c1', { epoch: 2, foreground: false, state: 'hidden', changedAt: 50 }),
      50
    );

    expect(scheduler.protectedSessionKeys(75)).toEqual([
      session('c1', 'chat'),
    ]);
    expect(scheduler.protectedSessionKeys(151)).toEqual([]);
  });

  it('rejects work after the pending limit is reached', () => {
    const scheduler = new PriorityAwareScheduler({ maxPending: 1 });
    scheduler.updateLifecycle(lifecycle('c1', { foreground: false }), 0);
    expect(scheduler.enqueue(request('r1'), 0).accepted).toBe(true);
    expect(scheduler.enqueue(request('r2'), 1)).toEqual({
      accepted: false,
      reason: 'pending_limit',
    });
  });

  it('drains unanswered promotions in the order the application issued them', () => {
    const scheduler = new PriorityAwareScheduler({
      graceMs: 0,
      priorityHandoffMs: 2_000,
    });
    scheduler.updateLifecycle(lifecycle('tab-a', { epoch: 1, foreground: false }), 0);
    scheduler.enqueue(request('tab-a:r1'), 10);
    // The application promotes tab-a after its request is already queued.
    scheduler.updateLifecycle(lifecycle('tab-a', { epoch: 2, foreground: true }), 100);

    // The application promotes tab-b before tab-a answered.
    scheduler.updateLifecycle(lifecycle('tab-b', { epoch: 1, foreground: false }), 290);
    scheduler.enqueue(request('tab-b:r1'), 295);
    scheduler.updateLifecycle(lifecycle('tab-b', { epoch: 2, foreground: true }), 300);

    // Both sessions hold `high` at the same time and both still owe a token,
    // so the promotion issued first is answered first...
    expect(scheduler.selectNext(320)?.request.requestId).toBe('tab-a:r1');
    scheduler.requeue('tab-a:r1', 330);
    scheduler.noteDecodeProgress('tab-a:r1', 335);

    // ...and the session promoted second goes next.
    expect(scheduler.selectNext(340)?.request.requestId).toBe('tab-b:r1');
    scheduler.requeue('tab-b:r1', 350);
    scheduler.noteDecodeProgress('tab-b:r1', 355);

    // Answering does not demote either session: both still hold `high`, so
    // the burst bound is what keeps normal work moving.
    expect(scheduler.lifecycleFor('tab-a')?.foreground).toBe(true);
    expect(scheduler.lifecycleFor('tab-b')?.foreground).toBe(true);
    expect(scheduler.hasPendingHighPriority()).toBe(true);
  });

  it('bounds how long the answered visible tab can hold the gpu', () => {
    const scheduler = new PriorityAwareScheduler({
      graceMs: 0,
      highPriorityBurstUnits: 2,
    });
    scheduler.updateLifecycle(lifecycle('fg', { foreground: true }), 0);
    scheduler.enqueue(request('fg:r1', { session: session('fg') }), 10);
    scheduler.updateLifecycle(
      lifecycle('bg', { epoch: 2, foreground: false }),
      20
    );
    scheduler.enqueue(
      request('bg:r1', {
        session: session('bg'),
        lifecycle: lifecycle('bg', { epoch: 2, foreground: false }),
      }),
      30
    );
    // The visible tab already answered, so the activation class does not apply.
    scheduler.noteDecodeProgress('fg:r1', 40);

    expect(scheduler.selectNext(50)?.request.requestId).toBe('fg:r1');
    scheduler.requeue('fg:r1', 60);
    expect(scheduler.selectNext(70)?.request.requestId).toBe('fg:r1');
    scheduler.requeue('fg:r1', 80);

    // Burst exhausted: hidden work gets one unit before the visible tab
    // resumes its priority.
    expect(scheduler.selectNext(90)?.request.requestId).toBe('bg:r1');
    scheduler.requeue('bg:r1', 100);
    expect(scheduler.selectNext(110)?.request.requestId).toBe('fg:r1');
  });

  it('does not invent an activation debt for work queued into a stale foreground tab', () => {
    const scheduler = new PriorityAwareScheduler({
      graceMs: 0,
      priorityHandoffMs: 2_000,
    });
    // The page that happened to be active when the trial booted reports
    // itself foreground before the human has clicked anything.
    scheduler.updateLifecycle(lifecycle('boot', { foreground: true }), 0);
    scheduler.enqueue(request('boot:r1', { session: session('boot') }), 10);
    expect(scheduler.get('boot:r1')?.foregroundAtMs).toBeUndefined();
  });

  it('drops a deprioritised promotion debt once the hand-off window expires', () => {
    const scheduler = new PriorityAwareScheduler({
      graceMs: 0,
      priorityHandoffMs: 500,
    });
    scheduler.updateLifecycle(lifecycle('tab-a', { epoch: 1, foreground: false }), 0);
    scheduler.enqueue(request('tab-a:r1'), 10);
    // The application promotes tab-a and then returns it to `normal` before
    // it produced a token, so the acknowledgment is only protected for the
    // bounded hand-off window.
    scheduler.updateLifecycle(lifecycle('tab-a', { epoch: 2, foreground: true }), 50);
    scheduler.updateLifecycle(lifecycle('tab-a', { epoch: 3, foreground: false }), 120);

    scheduler.updateLifecycle(lifecycle('tab-b', { epoch: 1, foreground: false }), 130);
    scheduler.enqueue(request('tab-b:r1'), 140);
    scheduler.updateLifecycle(lifecycle('tab-b', { epoch: 2, foreground: true }), 150);
    scheduler.noteDecodeProgress('tab-b:r1', 160);

    // Inside the window the unanswered debt still outranks the answered
    // high-priority session...
    expect(scheduler.selectNext(300)?.request.requestId).toBe('tab-a:r1');

    // ...and past it tab-a is ordinary work, so the high-priority session
    // wins even though tab-a never produced a token.
    expect(scheduler.selectNext(700)?.request.requestId).toBe('tab-b:r1');
  });

  it('keeps a demoted promotion debt through a long pressured prefill window', () => {
    const scheduler = new PriorityAwareScheduler({ graceMs: 0 });
    scheduler.updateLifecycle(
      lifecycle('tab-a', { epoch: 1, foreground: false }),
      0
    );
    scheduler.enqueue(request('tab-a:r1'), 10);
    scheduler.updateLifecycle(
      lifecycle('tab-a', { epoch: 2, foreground: true }),
      50
    );
    scheduler.updateLifecycle(
      lifecycle('tab-a', { epoch: 3, foreground: false }),
      1_000
    );

    scheduler.updateLifecycle(
      lifecycle('tab-b', { epoch: 1, foreground: false }),
      2_000
    );
    scheduler.enqueue(request('tab-b:r1'), 2_010);
    scheduler.updateLifecycle(
      lifecycle('tab-b', { epoch: 2, foreground: true }),
      2_020
    );
    scheduler.noteDecodeProgress('tab-b:r1', 2_030);

    // A pressured 1091-token prefill cannot answer inside the old 5 s
    // hand-off window. The bounded debt must survive one long prefill.
    expect(scheduler.selectNext(30_000)?.request.requestId).toBe('tab-a:r1');
  });

  it('keeps answered promotions in the priority lane for bounded sharing', () => {
    const scheduler = new PriorityAwareScheduler({ graceMs: 0 });
    scheduler.updateLifecycle(
      lifecycle('tab-a', { epoch: 1, foreground: false }),
      0
    );
    scheduler.enqueue(
      request('tab-a:r1', { createdAt: 100 }),
      10
    );
    scheduler.updateLifecycle(
      lifecycle('tab-a', { epoch: 2, foreground: true }),
      20
    );
    scheduler.noteDecodeProgress('tab-a:r1', 30);
    expect(scheduler.selectNext(35)?.request.requestId).toBe('tab-a:r1');
    scheduler.requeue('tab-a:r1', 36);

    scheduler.updateLifecycle(
      lifecycle('tab-b', { epoch: 1, foreground: false }),
      40
    );
    scheduler.enqueue(request('tab-b:r1', { createdAt: 0 }), 50);
    scheduler.updateLifecycle(
      lifecycle('tab-b', { epoch: 2, foreground: true }),
      60
    );
    scheduler.noteDecodeProgress('tab-b:r1', 70);

    // tab-b is newer and has an older creation time, but tab-a already owns
    // Both requests are in the priority lane. The older request wins here
    // by its wait order; lane ownership is not a completion lock.
    expect(scheduler.selectNext(80)?.request.requestId).toBe('tab-a:r1');
    expect(scheduler.priorityServing('tab-a:r1', 81)).toBe(true);
  });

  it('lets an answered promotion decay back to ordinary work after its window', () => {
    const scheduler = new PriorityAwareScheduler({
      graceMs: 0,
      priorityDecayMs: 200,
    });
    scheduler.updateLifecycle(
      lifecycle('tab-a', { epoch: 1, foreground: false }),
      0
    );
    scheduler.enqueue(request('tab-a:r1', { createdAt: 100 }), 10);
    scheduler.updateLifecycle(
      lifecycle('tab-a', { epoch: 2, foreground: true }),
      20
    );
    scheduler.noteDecodeProgress('tab-a:r1', 30);
    scheduler.updateLifecycle(
      lifecycle('tab-a', { epoch: 3, foreground: false }),
      40
    );

    // The answered burst keeps the priority serving class while the
    // promotion is still fresh...
    expect(scheduler.priorityServing('tab-a:r1', 220)).toBe(true);
    expect(scheduler.promotionPending('tab-a:r1', 220)).toBe(false);

    // ...and decays back to ordinary work once the window has passed, so a
    // chain of promotions cannot leave every past target holding the lane.
    expect(scheduler.priorityServing('tab-a:r1', 300)).toBe(false);
    expect(scheduler.promotionPending('tab-a:r1', 300)).toBe(false);
  });

  it('keeps a high-priority session in the lane while it stays promoted', () => {
    const scheduler = new PriorityAwareScheduler({
      graceMs: 0,
      priorityDecayMs: 200,
    });
    scheduler.updateLifecycle(
      lifecycle('tab-a', { epoch: 1, foreground: false }),
      0
    );
    scheduler.enqueue(request('tab-a:r1', { createdAt: 100 }), 10);
    scheduler.updateLifecycle(
      lifecycle('tab-a', { epoch: 2, foreground: true }),
      20
    );
    scheduler.noteDecodeProgress('tab-a:r1', 30);

    // The decay bound only applies once the application returns the session
    // to `normal`; an explicit high-priority session keeps its class.
    expect(scheduler.priorityServing('tab-a:r1', 10_000)).toBe(true);
  });

  it('keeps an answered in-flight promotion ahead after its session is demoted', () => {
    const scheduler = new PriorityAwareScheduler({ graceMs: 0 });
    scheduler.updateLifecycle(
      lifecycle('tab-a', { epoch: 1, foreground: false }),
      0
    );
    scheduler.enqueue(
      request('tab-a:r1', { createdAt: 100 }),
      10
    );
    scheduler.updateLifecycle(
      lifecycle('tab-a', { epoch: 2, foreground: true }),
      20
    );
    scheduler.noteDecodeProgress('tab-a:r1', 30);
    scheduler.updateLifecycle(
      lifecycle('tab-a', { epoch: 3, foreground: false }),
      40
    );

    scheduler.updateLifecycle(
      lifecycle('bg', { epoch: 1, foreground: false }),
      50
    );
    scheduler.enqueue(request('bg:r1', { createdAt: 0 }), 60);

    // The second request has an older creation time, so it would win if the
    // demotion erased the promoted request's serving class.
    expect(scheduler.selectNext(70)?.request.requestId).toBe('tab-a:r1');
  });

  it('raises an older deprioritised promotion after the fairness bound', () => {
    const scheduler = new PriorityAwareScheduler({
      graceMs: 0,
      priorityFairnessMs: 300,
      priorityHandoffMs: 5_000,
    });
    // tab-a is promoted, then returned to `normal` while it still owes a
    // token. tab-b and tab-c stay high priority.
    scheduler.updateLifecycle(lifecycle('tab-a', { epoch: 1, foreground: false }), 0);
    scheduler.enqueue(request('tab-a:r1'), 5);
    scheduler.updateLifecycle(lifecycle('tab-a', { epoch: 2, foreground: true }), 50);
    scheduler.updateLifecycle(lifecycle('tab-a', { epoch: 3, foreground: false }), 100);

    for (const [clientId, activatedAt, enqueuedAt] of [
      ['tab-b', 120, 100],
      ['tab-c', 220, 200],
    ] as Array<[string, number, number]>) {
      scheduler.updateLifecycle(
        lifecycle(clientId, { epoch: 1, foreground: false }),
        enqueuedAt
      );
      scheduler.enqueue(request(`${clientId}:r1`), enqueuedAt + 5);
      scheduler.updateLifecycle(
        lifecycle(clientId, { epoch: 2, foreground: true }),
        activatedAt
      );
    }

    // Inside the fairness bound, the unanswered high-priority promotions
    // lead.
    expect(scheduler.selectNext(300)?.request.requestId).toBe('tab-b:r1');
    scheduler.requeue('tab-b:r1', 305);
    scheduler.noteDecodeProgress('tab-b:r1', 306);

    // Once tab-a has waited past the bound it is raised, so a stream of later
    // promotions cannot starve it indefinitely.
    expect(scheduler.selectNext(350)?.request.requestId).toBe('tab-a:r1');
  });

  it('marks an unanswered foreground promotion as a pending handoff', () => {
    const scheduler = new PriorityAwareScheduler({ graceMs: 0 });
    scheduler.updateLifecycle(
      lifecycle('tab-a', { epoch: 1, foreground: false }),
      0
    );
    scheduler.enqueue(request('tab-a:r1'), 10);

    scheduler.updateLifecycle(
      lifecycle('tab-a', { epoch: 2, foreground: true, state: 'active' }),
      50
    );

    expect(scheduler.promotionPending('tab-a:r1', 51)).toBe(true);
    scheduler.noteDecodeProgress('tab-a:r1', 60);
    expect(scheduler.promotionPending('tab-a:r1', 61)).toBe(false);
  });

  it('carries a promotion into a request submitted after the priority write', () => {
    const scheduler = new PriorityAwareScheduler({ graceMs: 0 });
    scheduler.updateLifecycle(
      lifecycle('tab-a', { epoch: 1, foreground: false }),
      0
    );
    scheduler.updateLifecycle(
      lifecycle('tab-a', { epoch: 2, foreground: true, state: 'active' }),
      50
    );

    expect(
      scheduler.enqueue(
        request('tab-a:r1', {
          lifecycle: lifecycle('tab-a', {
            epoch: 2,
            foreground: true,
            state: 'active',
          }),
        }),
        60
      ).accepted
    ).toBe(true);
    expect(scheduler.get('tab-a:r1')?.foregroundAtMs).toBe(50);
    expect(scheduler.promotionPending('tab-a:r1', 61)).toBe(true);
  });
  it('tracks promotion debt before the promoted request is submitted', () => {
    const scheduler = new PriorityAwareScheduler({
      graceMs: 0,
      priorityHandoffMs: 50,
    });
    scheduler.updateLifecycle(
      lifecycle('tab-a', { epoch: 1, foreground: false }),
      0
    );
    scheduler.updateLifecycle(
      lifecycle('tab-a', { epoch: 2, foreground: true, state: 'active' }),
      10
    );

    expect(scheduler.hasPromotionDebt('tab-a', 11)).toBe(true);
    scheduler.enqueue(
      request('tab-a:r1', {
        lifecycle: lifecycle('tab-a', {
          epoch: 2,
          foreground: true,
          state: 'active',
        }),
      }),
      12
    );
    expect(scheduler.hasPromotionDebt('tab-a', 13)).toBe(true);
    scheduler.noteDecodeProgress('tab-a:r1', 20);
    expect(scheduler.hasPromotionDebt('tab-a', 21)).toBe(false);

    scheduler.updateLifecycle(
      lifecycle('tab-a', { epoch: 3, foreground: false }),
      22
    );
    scheduler.updateLifecycle(
      lifecycle('tab-a', { epoch: 4, foreground: true, state: 'active' }),
      30
    );
    expect(scheduler.hasPromotionDebt('tab-a', 81)).toBe(false);
  });
});
