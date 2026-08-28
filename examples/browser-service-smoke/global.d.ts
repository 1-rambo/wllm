export {};

declare global {
  interface Window {
    wllmBrowserClient: import('../../src/browser-service-client').BrowserLLMClient;
    wllmBrowserReady: Promise<
      import('../../src/browser-service-protocol').BrowserServiceStatus
    >;
    wllmMeasureInteraction: () => Promise<number>;
    wllmBackgroundChunkCount: number;
    wllmExperimentEvents: import('../../src/browser-service-protocol').BrowserServiceEvent[];
    wllmClearExperimentEvents: () => void;
    wllmStartUiMeasurement: (workItems?: number) => Promise<void>;
    wllmStopUiMeasurement: () => Promise<{
      backend: 'webgpu-compute' | 'canvas2d';
      sampleCount: number;
      averageFps: number;
      frameTimeP50Ms: number;
      frameTimeP95Ms: number;
      frameTimeP99Ms: number;
      maxFrameTimeMs: number;
      longFrameRatio: number;
      workItems: number;
    }>;
  }
}
