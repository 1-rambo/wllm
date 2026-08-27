import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { copyFile, mkdir } from 'node:fs/promises';
import path from 'node:path';
import { chromium } from 'playwright';

const modelFile = process.env.WLLM_SMOKE_MODEL_FILE;
const modelUrl = modelFile
  ? 'http://127.0.0.1:8080/_tmp/browser-service-cpu/model.gguf'
  : (process.env.WLLM_SMOKE_MODEL_URL ??
    'https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf');
const candidates = [
  process.env.CHROMIUM_EXECUTABLE_PATH,
  `${process.env.HOME}/.cache/ms-playwright/chromium-1148/chrome-linux/chrome`,
  `${process.env.HOME}/.cache/ms-playwright/chromium-1217/chrome-linux64/chrome`,
  `${process.env.HOME}/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome`,
].filter(Boolean);
const executablePath = candidates.find((path) => existsSync(path));
const cpuBuild = process.env.WLLM_CPU_BUILD_DIR;

if (!executablePath) {
  throw new Error('Chromium is unavailable');
}
if (!cpuBuild) {
  throw new Error(
    'Set WLLM_CPU_BUILD_DIR to a source-built CPU-only Wasm directory'
  );
}

await mkdir('_tmp/browser-service-cpu', { recursive: true });
await copyFile(`${cpuBuild}/wllama.js`, '_tmp/browser-service-cpu/wllama.js');
await copyFile(
  `${cpuBuild}/wllama.wasm`,
  '_tmp/browser-service-cpu/wllama.wasm'
);
if (modelFile) {
  await copyFile(
    path.resolve(modelFile),
    '_tmp/browser-service-cpu/model.gguf'
  );
}

const server = spawn(process.execPath, ['scripts/http_server.js'], {
  cwd: new URL('../..', import.meta.url),
  stdio: ['ignore', 'pipe', 'inherit'],
});

try {
  await new Promise((resolve, reject) => {
    const timer = setTimeout(
      () => reject(new Error('server startup timeout')),
      10000
    );
    server.stdout.on('data', (chunk) => {
      if (String(chunk).includes('Server running')) {
        clearTimeout(timer);
        resolve();
      }
    });
    server.once('exit', (code) =>
      reject(new Error(`server exited with ${code}`))
    );
  });

  const browser = await chromium.launch({
    executablePath,
    headless: true,
    args: ['--no-sandbox'],
  });
  const context = await browser.newContext();
  const pageA = await context.newPage();
  const pageB = await context.newPage();
  pageA.on('console', (msg) => console.log('[A]', msg.type(), msg.text()));
  pageB.on('console', (msg) => console.log('[B]', msg.type(), msg.text()));
  pageA.on('pageerror', (err) => console.error('[A pageerror]', err));
  pageB.on('pageerror', (err) => console.error('[B pageerror]', err));
  const url = new URL('http://127.0.0.1:8080/examples/browser-service-smoke/');
  url.searchParams.set('module', '/_tmp/browser-service-cpu/wllama.js');
  url.searchParams.set('wasm', '/_tmp/browser-service-cpu/wllama.wasm');
  url.searchParams.set('foreground', '0');
  await pageA.goto(url.href);
  url.searchParams.set('foreground', '1');
  await pageB.goto(url.href);
  await Promise.all([
    pageA.evaluate(() => window.wllmBrowserReady),
    pageB.evaluate(() => window.wllmBrowserReady),
  ]);

  await Promise.all([
    pageA.evaluate(
      (url) =>
        window.wllmBrowserClient.loadModelFromUrl(url, {
          n_ctx: 4096,
          n_batch: 128,
          n_seq_max: 4,
        }),
      modelUrl
    ),
    pageB.evaluate(
      (url) =>
        window.wllmBrowserClient.loadModelFromUrl(url, {
          n_ctx: 4096,
          n_batch: 128,
          n_seq_max: 4,
        }),
      modelUrl
    ),
  ]);
  const tieredCache = {
    enabled: true,
    l1TokenCap: 512,
    l2TokenCap: 2048,
    l3TokenCap: 8192,
    admissionPolicy: 'monotonic',
    admissionMaxRepairSteps: 16,
  };
  await Promise.all([
    pageA.evaluate(
      (options) =>
        window.wllmBrowserClient.chatSessionInit(64 * 1024 * 1024, options),
      tieredCache
    ),
    pageB.evaluate(
      (options) =>
        window.wllmBrowserClient.chatSessionInit(64 * 1024 * 1024, options),
      tieredCache
    ),
  ]);

  await pageA.evaluate(() =>
    window.wllmBrowserClient.updateLifecycle({
      foreground: false,
      lifecycleState: 'hidden',
    })
  );
  await pageB.evaluate(() =>
    window.wllmBrowserClient.updateLifecycle({
      foreground: true,
      lifecycleState: 'active',
    })
  );

  const [background, foreground] = await Promise.all([
    pageA.evaluate(() =>
      window.wllmBrowserClient.chatFromNode(0, 'Write a short story.', {
        nPredict: 16,
        logicalSessionId: 'background',
      })
    ),
    pageB.evaluate(() =>
      window.wllmBrowserClient.chatFromNode(0, 'Say hello.', {
        nPredict: 4,
        logicalSessionId: 'foreground',
      })
    ),
  ]);
  const telemetry = await pageB.evaluate(() =>
    window.wllmBrowserClient.getTelemetry()
  );
  const eventTypes = new Set(telemetry.map((event) => event.type));
  for (const required of [
    'foreground_change',
    'request_arrive',
    'slice_dispatch',
    'request_complete',
  ]) {
    if (!eventTypes.has(required)) {
      throw new Error(`missing telemetry event ${required}`);
    }
  }
  console.log(
    JSON.stringify({
      ok: true,
      backgroundNode: background.nodeId,
      foregroundNode: foreground.nodeId,
      events: telemetry.length,
    })
  );
  await pageA.evaluate(() => window.wllmBrowserClient.exit());
  await pageB.evaluate(() => window.wllmBrowserClient.destroyService());
  await browser.close();
} finally {
  server.kill('SIGTERM');
}
