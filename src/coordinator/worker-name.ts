/**
 * A SharedWorker is looked up by name and reused across tabs, so two builds
 * that disagree about the protocol must never share an instance. The name
 * combines the package version with a digest of the worker scope source the
 * browser will actually load, and it deliberately ignores the per-run URL
 * options so one configuration does not spawn a second coordinator.
 */
export async function coordinatorWorkerName(url: URL): Promise<string> {
  const version = typeof __WLLM_VERSION__ === 'string' ? __WLLM_VERSION__ : 'dev';
  const digest = await workerSourceDigest(url);
  return `wllm-${version}-${digest}`;
}

async function workerSourceDigest(url: URL): Promise<string> {
  try {
    const response = await fetch(url.href);
    if (response.ok) {
      const source = await response.text();
      if (source.length > 0) {
        return (await sha256Hex(source)).slice(0, 12);
      }
    }
  } catch {
    // fall through to the URL digest; the name still separates mismatched builds
  }
  return (await sha256Hex(url.href)).slice(0, 12);
}

async function sha256Hex(value: string): Promise<string> {
  const bytes = new TextEncoder().encode(value);
  const subtle = globalThis.crypto?.subtle;
  if (subtle) {
    const digest = await subtle.digest('SHA-256', bytes);
    return [...new Uint8Array(digest)]
      .map((byte) => byte.toString(16).padStart(2, '0'))
      .join('');
  }
  // FNV-1a fallback for contexts without WebCrypto.
  let hash = 0x811c9dc5;
  for (const byte of bytes) {
    hash ^= byte;
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash.toString(16).padStart(8, '0');
}
