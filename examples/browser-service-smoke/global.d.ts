export {};

declare global {
  interface Window {
    wllmBrowserClient: import('../../src/browser-service-client').BrowserLLMClient;
    wllmBrowserReady: Promise<
      import('../../src/browser-service-protocol').BrowserServiceStatus
    >;
    wllmMeasureInteraction: () => Promise<number>;
    wllmBackgroundChunkCount: number;
  }
}
