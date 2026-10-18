import type { CheckpointEnvelope, NativeSlotCheckpoint, SessionKey } from '../contracts';
import { checksumBytes } from './bytes';

export function makeCheckpointEnvelope(
  checkpoint: NativeSlotCheckpoint,
  input: {
    checkpointKey: string;
    nodeId: string;
    session: SessionKey;
    parentNodeId?: string | null;
    modelHash: string;
    createdAt: number;
  }
): CheckpointEnvelope {
  return {
    ...persistableCheckpoint(checkpoint),
    bytes: new Uint8Array(checkpoint.bytes),
    ...(checkpoint.kv ? { kv: new Uint8Array(checkpoint.kv) } : {}),
    tokens: [...checkpoint.tokens],
    ...input,
  };
}

// A tier-two handle is only meaningful inside the native instance that minted
// it. Durable envelopes outlive that instance, and the native copy is released
// once the blob is stored, so a persisted handle is a stale pointer that would
// hijack the restore path. Envelopes carry the serialized state only.
export function persistableCheckpoint(
  checkpoint: NativeSlotCheckpoint
): NativeSlotCheckpoint {
  const { parkedHandle, parkedByteLength, ...portable } = checkpoint;
  void parkedHandle;
  void parkedByteLength;
  return portable;
}

export function checkpointChecksum(checkpoint: NativeSlotCheckpoint): string {
  if (!checkpoint.kv || checkpoint.kv.byteLength === 0) {
    return checksumBytes(checkpoint.bytes);
  }
  const merged = new Uint8Array(
    checkpoint.bytes.byteLength + checkpoint.kv.byteLength
  );
  merged.set(checkpoint.bytes, 0);
  merged.set(checkpoint.kv, checkpoint.bytes.byteLength);
  return checksumBytes(merged);
}

export function serializeCheckpoint(envelope: CheckpointEnvelope): Uint8Array {
  return new TextEncoder().encode(
    JSON.stringify({
      ...envelope,
      bytes: bytesToBase64(envelope.bytes),
      kv: envelope.kv ? bytesToBase64(envelope.kv) : undefined,
    })
  );
}

export function deserializeCheckpoint(bytes: Uint8Array): CheckpointEnvelope {
  const parsed = JSON.parse(new TextDecoder().decode(bytes)) as Omit<
    CheckpointEnvelope,
    'bytes' | 'kv'
  > & { bytes: string; kv?: string };
  const { bytes: encodedBytes, kv: encodedKv, ...rest } = parsed;
  const envelope: CheckpointEnvelope = {
    ...rest,
    bytes: base64ToBytes(encodedBytes),
  };
  if (encodedKv) {
    envelope.kv = base64ToBytes(encodedKv);
  }
  return envelope;
}

function bytesToBase64(bytes: Uint8Array): string {
  let binary = '';
  const chunkSize = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    const chunk = bytes.subarray(offset, offset + chunkSize);
    binary += String.fromCharCode(...chunk);
  }
  return base64EncodeBinary(binary);
}

function base64ToBytes(value: string): Uint8Array {
  const alphabet =
    'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
  const clean = value.replace(/=+$/, '');
  let bits = 0;
  let bitCount = 0;
  const output: number[] = [];
  for (const character of clean) {
    const digit = alphabet.indexOf(character);
    if (digit < 0) throw new Error('invalid base64');
    bits = (bits << 6) | digit;
    bitCount += 6;
    if (bitCount >= 8) {
      bitCount -= 8;
      output.push((bits >> bitCount) & 0xff);
    }
  }
  return Uint8Array.from(output);
}

function base64EncodeBinary(binary: string): string {
  const alphabet =
    'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
  let output = '';
  for (let index = 0; index < binary.length; index += 3) {
    const a = binary.charCodeAt(index);
    const hasB = index + 1 < binary.length;
    const hasC = index + 2 < binary.length;
    const b = hasB ? binary.charCodeAt(index + 1) : 0;
    const c = hasC ? binary.charCodeAt(index + 2) : 0;
    const triple = (a << 16) | (b << 8) | c;
    output += alphabet[(triple >> 18) & 0x3f];
    output += alphabet[(triple >> 12) & 0x3f];
    output += hasB ? alphabet[(triple >> 6) & 0x3f] : '=';
    output += hasC ? alphabet[triple & 0x3f] : '=';
  }
  return output;
}
