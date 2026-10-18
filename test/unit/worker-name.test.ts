import { describe, expect, it, vi } from 'vitest';
import { coordinatorWorkerName } from '../../src/coordinator/worker-name';

describe('coordinator worker name', () => {
  it('combines the library version with a digest of the worker source', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response('self.onconnect = () => {};\n'))
    );
    try {
      const url = new URL('https://example.test/coordinator.js');
      const first = await coordinatorWorkerName(url);
      const second = await coordinatorWorkerName(url);
      expect(first).toBe(second);
      expect(first).toMatch(/^wllm-[\w.-]+-[0-9a-f]{8,64}$/);
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it('separates builds whose worker source differs', async () => {
    try {
      vi.stubGlobal(
        'fetch',
        vi.fn(async () => new Response('self.onconnect = () => 1;\n'))
      );
      const url = new URL('https://example.test/coordinator.js');
      const one = await coordinatorWorkerName(url);
      vi.stubGlobal(
        'fetch',
        vi.fn(async () => new Response('self.onconnect = () => 2;\n'))
      );
      const two = await coordinatorWorkerName(url);
      expect(two).not.toBe(one);
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it('falls back to a url digest when the source cannot be fetched', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new Error('offline');
      })
    );
    try {
      const name = await coordinatorWorkerName(
        new URL('https://example.test/coordinator.js')
      );
      expect(name).toMatch(/^wllm-[\w.-]+-[0-9a-f]{8,64}$/);
    } finally {
      vi.unstubAllGlobals();
    }
  });
});
