import manifest from '../../config/demo-model.json';

export interface DemoModelManifest {
  id: string;
  label: string;
  filename: string;
  sourceUrl: string;
  downloadUrl: string;
  license: string;
  licenseUrl: string;
  sizeBytes: number;
  sha256: string;
  runtime: {
    nCtx: number;
    nBatch: number;
    nParallel: number;
  };
}

export const demoModelManifest = manifest as DemoModelManifest;
export const defaultModelPath = `/models/${demoModelManifest.filename}`;
export const defaultRuntimeConfig = demoModelManifest.runtime;
