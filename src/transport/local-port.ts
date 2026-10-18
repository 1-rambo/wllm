import type { MessagePortLike } from './protocol';

export class LocalMessagePort implements MessagePortLike {
  peer: LocalMessagePort | null = null;
  private readonly listeners = new Set<(event: { data: unknown }) => void>();
  private started = false;
  private readonly queued: unknown[] = [];

  postMessage(message: unknown): void {
    const peer = this.peer;
    if (!peer) {
      throw new Error('local message port is not connected');
    }
    queueMicrotask(() => peer.deliver(message));
  }

  addEventListener(
    type: 'message',
    listener: (event: { data: unknown }) => void
  ): void {
    if (type === 'message') {
      this.listeners.add(listener);
    }
  }

  start(): void {
    this.started = true;
    while (this.queued.length > 0) {
      const data = this.queued.shift();
      this.deliver(data);
    }
  }

  close(): void {
    this.listeners.clear();
  }

  private deliver(data: unknown): void {
    if (!this.started) {
      this.queued.push(data);
      return;
    }
    for (const listener of this.listeners) {
      listener({ data });
    }
  }
}

export function connectLocalPorts(
  left: LocalMessagePort,
  right: LocalMessagePort
): void {
  left.peer = right;
  right.peer = left;
}
