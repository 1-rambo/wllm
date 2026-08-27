import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { chromium } from 'playwright';

const candidates = [
  process.env.CHROMIUM_EXECUTABLE_PATH,
  `${process.env.HOME}/.cache/ms-playwright/chromium-1148/chrome-linux/chrome`,
  `${process.env.HOME}/.cache/ms-playwright/chromium-1217/chrome-linux64/chrome`,
  `${process.env.HOME}/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome`,
].filter(Boolean);
const executablePath = candidates.find((path) => existsSync(path));

if (!executablePath) {
  throw new Error(
    'Chromium is unavailable. Run `npx playwright install chromium` or set CHROMIUM_EXECUTABLE_PATH.'
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
  const url = 'http://127.0.0.1:8080/examples/browser-service-smoke/';
  await pageA.goto(`${url}?foreground=0`);
  await pageB.goto(`${url}?foreground=1`);
  await pageA.evaluate(() => window.wllmBrowserReady);
  await pageB.evaluate(() => window.wllmBrowserReady);
  const [statusA, status] = await Promise.all([
    pageA.evaluate(() => window.wllmBrowserClient.status()),
    pageB.evaluate(() => window.wllmBrowserClient.status()),
  ]);
  const registry = await Promise.all([
    pageA.evaluate(() => ({ ...localStorage })),
    pageB.evaluate(() => ({ ...localStorage })),
  ]);
  if (status.clientCount !== 2 || statusA.clientCount !== 2) {
    throw new Error(
      `expected 2 clients, received A=${statusA.clientCount}, B=${status.clientCount}`
    );
  }
  const debugSupported = await pageB.evaluate(
    () => typeof window.wllmBrowserClient.getDebugInfo === 'function'
  );
  if (!debugSupported) throw new Error('debug API unavailable');
  console.log(JSON.stringify({ ok: true, status }));
  await pageA.evaluate(() => window.wllmBrowserClient.exit());
  await pageB.evaluate(() => window.wllmBrowserClient.destroyService());
  await browser.close();
} finally {
  server.kill('SIGTERM');
}
