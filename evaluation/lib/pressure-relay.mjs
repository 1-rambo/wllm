import { samplePressure } from "./render-load.mjs";

function withTimeout(promise, timeoutMs, message) {
  const guarded = Promise.resolve(promise);
  guarded.catch(() => undefined);
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(message)), timeoutMs);
    guarded.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (error) => {
        clearTimeout(timer);
        reject(error);
      }
    );
  });
}

export class PressureRelay {
  constructor(options) {
    this.pressurePage = options.pressurePage;
    this.enginePages = options.enginePages ?? [];
    this.intervalMs = options.intervalMs ?? 250;
    this.onSample = options.onSample ?? (() => undefined);
    this.samples = [];
    this.timer = null;
    this.running = false;
  }

  async start() {
    if (this.running) return;
    this.running = true;
    const tick = async () => {
      if (!this.running) return;
      try {
        const sample = await withTimeout(
          samplePressure(this.pressurePage),
          5_000,
          "pressure sample timed out"
        );
        const row = {
          atMs: Date.now(),
          ...sample
        };
        this.samples.push(row);
        if (this.samples.length > 4096) this.samples.shift();
        await this.onSample(row);
        for (const page of this.enginePages) {
          if (!page.engine || page.engine !== "wllm") continue;
          await withTimeout(
            page.page.evaluate(
              async ({ samplePayload }) => {
                if (window.__wllmE2E?.reportRenderPressure) {
                  await window.__wllmE2E.reportRenderPressure(samplePayload);
                }
              },
              { samplePayload: row }
            ),
            2_000,
            "render-pressure relay timed out"
          ).catch(() => undefined);
        }
      } catch (error) {
        await this.onSample({
          atMs: Date.now(),
          error: error instanceof Error ? error.message : String(error)
        });
      } finally {
        if (this.running) {
          this.timer = setTimeout(tick, this.intervalMs);
        }
      }
    };
    // Keep start() behind the first measured broadcast. Trials otherwise race
    // inference against timer scheduling and can begin with a stale lease.
    await tick();
  }

  async stop() {
    this.running = false;
    if (this.timer !== null) {
      clearTimeout(this.timer);
      this.timer = null;
    }
  }
}
