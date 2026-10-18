import { describe, expect, it } from 'vitest';
import { BrowserLLMClient } from '../../src/client/browser-llm-client';
import { startCoordinatorPort } from '../../src/coordinator/coordinator-host';
import { SharedWorkerCoordinator } from '../../src/coordinator/shared-worker-coordinator';
import { FakeNativeEngine } from '../../src/native/fake-native-engine';
import { MemoryBlobStore } from '../../src/state/checkpoint-store';
import { connectLocalPorts, LocalMessagePort } from '../../src/transport/local-port';
import { lifecycle, request } from '../helpers/factories';
import type { CoordinatorEvent } from '../../src/transport/protocol';

describe('BrowserLLMClient integration', () => {
  it('attaches, submits, pumps, and receives completion over a message port', async () => {
    const [clientPort, workerPort] = [new LocalMessagePort(), new LocalMessagePort()];
    connectLocalPorts(clientPort, workerPort);

    const coordinator = new SharedWorkerCoordinator(
      new FakeNativeEngine({ prefillMsPerToken: 1, decodeMsPerToken: 1 }),
      new MemoryBlobStore(),
      {
        scheduler: { graceMs: 0 },
        execution: {
          maxPrefillTokens: 4,
          baseDecodeTokens: 4,
          foregroundDecodeTokens: 4,
          backgroundDecodeTokens: 4,
          prefillMsPerToken: 1,
          maxUnitWallTimeMs: 100,
        },
        macrotaskYield: async () => undefined,
      }
    );
    startCoordinatorPort(workerPort, coordinator);

    const completions: string[] = [];
    const client = new BrowserLLMClient(clientPort, {
      onUnit: (event) => {
        if (event.type === 'complete') {
          completions.push(event.output);
        }
      },
    });

    await client.attach('c1', lifecycle('c1'));
    await client.submit(request('c1:r1', { maxNewTokens: 2 }));
    await client.pump(2);
    await new Promise<void>((resolve) => setTimeout(resolve, 0));

    expect(completions).toHaveLength(1);
    expect(completions[0]).toContain('tok01');
  });

  it('forwards native chunks as coordinator token events', async () => {
    const coordinator = new SharedWorkerCoordinator(
      new FakeNativeEngine({ prefillMsPerToken: 1, decodeMsPerToken: 1 }),
      new MemoryBlobStore()
    );
    const events: CoordinatorEvent[] = [];
    coordinator.subscribe((event) => events.push(event));
    await coordinator.attach('c1', lifecycle('c1'));
    await coordinator.submit(request('c1:r1', { maxNewTokens: 2 }));

    coordinator.publishTokenChunk({
      requestId: 'c1:r1',
      text: 'hello',
      decodeGenerated: 1,
      elapsedMs: 12,
    });

    expect(events.at(-1)).toMatchObject({
      type: 'token',
      clientId: 'c1',
      requestId: 'c1:r1',
      text: 'hello',
      decodeGenerated: 1,
      elapsedMs: 12,
    });
  });
});
