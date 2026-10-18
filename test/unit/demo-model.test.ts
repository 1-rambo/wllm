import { describe, expect, it } from 'vitest';
import {
  defaultModelPath,
  defaultRuntimeConfig,
  demoModelManifest,
} from '../../src/config/demo-model';

describe('demo model manifest', () => {
  it('keeps the runtime default and fetch manifest in sync', () => {
    expect(defaultModelPath).toBe(
      `/models/${demoModelManifest.filename}`
    );
    expect(defaultRuntimeConfig).toBe(demoModelManifest.runtime);
  });

  it('describes a verifiable GGUF download', () => {
    expect(demoModelManifest.filename).toMatch(/\.gguf$/);
    expect(demoModelManifest.sizeBytes).toBeGreaterThan(0);
    expect(demoModelManifest.sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(demoModelManifest.downloadUrl).toMatch(/^https:\/\//);
    expect(demoModelManifest.license).not.toHaveLength(0);
    expect(demoModelManifest.licenseUrl).toMatch(/^https:\/\//);
  });

  it('has valid default llama.cpp dimensions', () => {
    expect(defaultRuntimeConfig.nCtx).toBeGreaterThan(0);
    expect(defaultRuntimeConfig.nBatch).toBeGreaterThan(0);
    expect(defaultRuntimeConfig.nParallel).toBeGreaterThan(0);
  });
});
