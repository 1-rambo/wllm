import { checksumString } from './bytes';

export function modelIdentity(modelUrl: string, baseUrl?: string): string {
  let identity = modelUrl;
  try {
    identity = new URL(modelUrl, baseUrl).href;
  } catch {
    // Non-URL identifiers still need a stable hash in unit tests and custom
    // runtimes that provide their own model handle.
  }
  return checksumString(identity);
}
