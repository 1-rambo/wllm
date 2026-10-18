import { describe, expect, it } from 'vitest';
import {
  checkpointChecksum,
  deserializeCheckpoint,
  makeCheckpointEnvelope,
  serializeCheckpoint,
} from '../../src/utils/serialization';
import { checksumBytes, decodeInt32Array, encodeInt32Array } from '../../src/utils/bytes';
import { session } from '../helpers/factories';

describe('checkpoint serialization', () => {
  it('round-trips a checkpoint without sharing mutable byte arrays', () => {
    const bytes = encodeInt32Array([10, 20, 30]);
    const envelope = makeCheckpointEnvelope(
      {
        bytes,
        tokens: [10, 20, 30],
        stage: 'decode',
        decodeGenerated: 3,
        formatVersion: 1,
        checksum: checksumBytes(bytes),
      },
      {
        checkpointKey: 'node:1',
        nodeId: 'node',
        session: session('c1'),
        modelHash: 'model-a',
        createdAt: 42,
      }
    );

    bytes[0] = 0;
    expect(envelope.bytes[0]).toBe(10);
    const restored = deserializeCheckpoint(serializeCheckpoint(envelope));
    expect(restored).toEqual(envelope);
    expect(decodeInt32Array(restored.bytes)).toEqual([10, 20, 30]);
    expect(checkpointChecksum(restored)).toBe(envelope.checksum);
  });

  it('drops a native park handle from a durable envelope', () => {
    const bytes = encodeInt32Array([1, 2, 3]);
    const kv = new Uint8Array([9, 9, 9, 9]);
    const envelope = makeCheckpointEnvelope(
      {
        bytes,
        kv,
        tokens: [1, 2, 3],
        stage: 'decode',
        decodeGenerated: 3,
        formatVersion: 2,
        checksum: checksumBytes(bytes),
        parkedHandle: 7,
        parkedByteLength: kv.byteLength,
      },
      {
        checkpointKey: 'node:2',
        nodeId: 'node',
        session: session('c1'),
        modelHash: 'model-a',
        createdAt: 43,
      }
    );

    expect(envelope.parkedHandle).toBeUndefined();
    expect(envelope.kv?.byteLength).toBe(kv.byteLength);
    const restored = deserializeCheckpoint(serializeCheckpoint(envelope));
    expect(restored.parkedHandle).toBeUndefined();
  });

  it('rejects malformed int32 payloads', () => {
    expect(() => decodeInt32Array(new Uint8Array([1, 2, 3]))).toThrow(
      'invalid int32 byte length'
    );
  });
});
