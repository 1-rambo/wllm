import type { LifecycleState } from '../contracts';

/**
 * Demo-owned visibility policy.
 *
 * This is intentionally application code: the runtime exposes explicit
 * session priority, and this page decides how browser visibility should map
 * onto it. The engine page does not install an automatic tab-attention bridge.
 */
export interface VisibilityObservation {
  foreground: boolean;
  state: LifecycleState;
  reason: string;
}

export interface VisibilityPriorityOptions {
  onChange: (observation: VisibilityObservation) => void;
  forcedBackground?: () => boolean;
  documentTarget?: EventTarget;
  windowTarget?: EventTarget;
  isVisible?: () => boolean;
  hasFocus?: () => boolean;
}

export interface VisibilityPriorityObserver {
  current(reason?: string): VisibilityObservation;
  stop(): void;
}

export function observeVisibilityPriority(
  options: VisibilityPriorityOptions
): VisibilityPriorityObserver {
  const documentTarget = options.documentTarget ?? document;
  const windowTarget = options.windowTarget ?? window;
  const isVisible =
    options.isVisible ?? (() => document.visibilityState === 'visible');
  const hasFocus = options.hasFocus ?? (() => document.hasFocus());

  let frozen = false;

  const current = (reason = 'snapshot'): VisibilityObservation => {
    const visible = isVisible();
    const focused = hasFocus();
    const forced = options.forcedBackground?.() ?? false;
    const state: LifecycleState = frozen
      ? 'frozen'
      : !visible
        ? 'hidden'
        : focused && !forced
          ? 'active'
          : 'passive';
    return { foreground: state === 'active', state, reason };
  };

  const emit = (reason: string): void => {
    options.onChange(current(reason));
  };

  const listeners: Array<[EventTarget, string, EventListener]> = [];
  const on = (target: EventTarget, type: string, handler: () => void): void => {
    const listener: EventListener = () => handler();
    target.addEventListener(type, listener);
    listeners.push([target, type, listener]);
  };

  on(documentTarget, 'visibilitychange', () => emit('visibilitychange'));
  on(documentTarget, 'freeze', () => {
    frozen = true;
    emit('freeze');
  });
  on(documentTarget, 'resume', () => {
    frozen = false;
    emit('resume');
  });
  on(windowTarget, 'focus', () => emit('focus'));
  on(windowTarget, 'blur', () => emit('blur'));
  on(windowTarget, 'pageshow', () => emit('pageshow'));
  on(windowTarget, 'pagehide', () => emit('pagehide'));

  return {
    current,
    stop(): void {
      for (const [target, type, listener] of listeners) {
        target.removeEventListener(type, listener);
      }
      listeners.length = 0;
    },
  };
}
