import { describe, expect, it } from 'vitest';
import { FakeNativeEngine } from '../../src/native/fake-native-engine';
import {
  MessagePortRpcClient,
  NativeEngineProxy,
  NativeEngineWorkerServer,
} from '../../src/transport/native-worker';
import { connectLocalPorts, LocalMessagePort } from '../../src/transport/local-port';
import type { NativeWorkerEvent } from '../../src/transport/protocol';

describe('dedicated inference worker transport', () => {
  it('round-trips submit, advance, slot export, drop, and import', async () => {
    const [clientPort, serverPort] = [new LocalMessagePort(), new LocalMessagePort()];
    connectLocalPorts(clientPort, serverPort);

    const native = new FakeNativeEngine({ slotCount: 2 });
    new NativeEngineWorkerServer(serverPort, native);
    const proxy = new NativeEngineProxy(new MessagePortRpcClient(clientPort));

    await proxy.submit({
      requestId: 'r1',
      promptTokens: [1, 2, 3, 4],
      maxNewTokens: 100,
    });
    const advanced = await proxy.advance('r1', {
      maxPrefillTokens: 4,
      maxDecodeTokens: 0,
      wallTimeMs: 100,
    });
    expect(advanced).toMatchObject({
      requestId: 'r1',
      prefillProcessed: 4,
      stage: 'decode',
    });

    const slotId = advanced.slotId;
    expect(slotId).not.toBeNull();
    const checkpoint = await proxy.slotExport(slotId!);
    await proxy.slotDrop(slotId!);

    const imported = await proxy.slotImport(checkpoint);
    expect(imported).toMatchObject({
      inUse: true,
      tokens: 4,
      state: 'decode',
    });
    await expect(proxy.slotStats()).resolves.toHaveLength(2);
  });

  it('forwards native token chunks before the call result', async () => {
    const [clientPort, serverPort] = [
      new LocalMessagePort(),
      new LocalMessagePort(),
    ];
    connectLocalPorts(clientPort, serverPort);
    const events: NativeWorkerEvent[] = [];
    new MessagePortRpcClient(clientPort, (event) => events.push(event));

    serverPort.postMessage({
      type: 'native.chunk',
      event: {
        requestId: 'r1',
        text: 'hello',
        decodeGenerated: 1,
        elapsedMs: 12,
      },
    });
    await new Promise<void>((resolve) => setTimeout(resolve, 0));

    expect(events).toEqual([
      {
        type: 'native.chunk',
        event: {
          requestId: 'r1',
          text: 'hello',
          decodeGenerated: 1,
          elapsedMs: 12,
        },
      },
    ]);
  });

  it('rejects pending calls when the native port is dropped', async () => {
    const [clientPort, serverPort] = [
      new LocalMessagePort(),
      new LocalMessagePort(),
    ];
    connectLocalPorts(clientPort, serverPort);
    const client = new MessagePortRpcClient(clientPort);
    const pending = client.call('status', ['missing']);
    client.dispose(new Error('worker released'));
    await expect(pending).rejects.toThrow('worker released');
  });
});
