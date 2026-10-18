import { describe, expect, it } from 'vitest';
import {
  readCoordinatorNumericOptions,
  writeCoordinatorNumericOptions,
} from '../../src/engines/coordinator-numeric-options';

describe('coordinator numeric options', () => {
  it('preserves the lease work-conserving quantum across the page boundary', () => {
    const options = {};
    readCoordinatorNumericOptions(
      new URLSearchParams({ leaseWorkConservingQuantumMs: '2500' }),
      options
    );
    expect(options).toMatchObject({ leaseWorkConservingQuantumMs: 2500 });

    const url = new URL('https://example.test/coordinator.shared-worker.js');
    writeCoordinatorNumericOptions(url, options);
    expect(url.searchParams.get('leaseWorkConservingQuantumMs')).toBe('2500');
  });

  it('still accepts an explicit zero quantum', () => {
    const options = {};
    readCoordinatorNumericOptions(
      new URLSearchParams({ leaseWorkConservingQuantumMs: '0' }),
      options
    );
    expect(options).toMatchObject({ leaseWorkConservingQuantumMs: 0 });
  });

  it('preserves the native queue-wait ceiling across the page boundary', () => {
    const options = {};
    readCoordinatorNumericOptions(
      new URLSearchParams({ queueWaitTimeoutMs: '300000' }),
      options
    );
    expect(options).toMatchObject({ queueWaitTimeoutMs: 300_000 });

    const url = new URL('https://example.test/coordinator.shared-worker.js');
    writeCoordinatorNumericOptions(url, options);
    expect(url.searchParams.get('queueWaitTimeoutMs')).toBe('300000');
  });
});
