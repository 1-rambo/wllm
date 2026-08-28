import { spawn } from 'node:child_process';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createReadStream, existsSync } from 'node:fs';
import { copyFile, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { createServer } from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { chromium } from 'playwright';

const args = Object.fromEntries(
  process.argv.slice(2).map((arg) => {
    const [key, ...rest] = arg.replace(/^--/, '').split('=');
    return [key, rest.join('=') || 'true'];
  })
);
const backend = args.backend ?? 'wllama';
if (!['wllama', 'web-llm'].includes(backend)) {
  throw new Error('--backend must be wllama or web-llm');
}
const output = path.resolve(
  args.output ?? `artifacts/runs/webarena-${backend}-${Date.now()}.json`
);
const gitRevision = execFileSync('git', ['rev-parse', 'HEAD'], {
  encoding: 'utf8',
}).trim();
const gitDirty =
  execFileSync('git', ['status', '--porcelain'], {
    encoding: 'utf8',
  }).trim().length > 0;
if (gitDirty && args['allow-dirty'] !== 'true') {
  throw new Error(
    'Refusing to record a baseline from a dirty worktree. Commit/stash changes or pass --allow-dirty=true for development only.'
  );
}
const deviceMetadata = args['device-file']
  ? JSON.parse(await readFile(path.resolve(args['device-file']), 'utf8'))
  : null;
const deviceId = args['device-id'] ?? deviceMetadata?.deviceId ?? 'unassigned';
if (
  deviceMetadata?.deviceId &&
  String(deviceMetadata.deviceId) !== String(deviceId)
) {
  throw new Error('device.json deviceId does not match --device-id');
}
const sha256File = (filePath) =>
  new Promise((resolve, reject) => {
    const hash = createHash('sha256');
    const stream = createReadStream(filePath);
    stream.on('error', reject);
    stream.on('data', (chunk) => hash.update(chunk));
    stream.on('end', () => resolve(hash.digest('hex')));
  });
const reservePort = () =>
  new Promise((resolve, reject) => {
    const server = createServer();
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const address = server.address();
      if (!address || typeof address === 'string') {
        reject(new Error('cannot reserve port'));
        return;
      }
      server.close((error) => (error ? reject(error) : resolve(address.port)));
    });
  });
const port = Number(args.port ?? (await reservePort()));
const cwd = path.resolve('examples/webarena-latency-bench');
const localModelName = args['model-file']
  ? `__benchmark_model_${Date.now()}.gguf`
  : null;
const localModelPath = localModelName
  ? path.join(cwd, 'public', localModelName)
  : null;
if (localModelPath) {
  await mkdir(path.dirname(localModelPath), { recursive: true });
  await copyFile(path.resolve(args['model-file']), localModelPath);
}
const modelUrl = localModelName
  ? `http://127.0.0.1:${port}/${localModelName}`
  : args['model-url'];
const modelSha256 = args['model-file']
  ? await sha256File(path.resolve(args['model-file']))
  : null;
const wasmBuild = args['wasm-build'] ? path.resolve(args['wasm-build']) : null;
if (backend === 'wllama' && args.gpu === '0' && !wasmBuild) {
  throw new Error(
    '--wasm-build=<source-built CPU Wasm directory> is required when --gpu=0'
  );
}
const localWasmStem = wasmBuild ? `__benchmark_wllama_${Date.now()}` : null;
const localWasmModulePath = localWasmStem
  ? path.join(cwd, 'public', `${localWasmStem}.js`)
  : null;
const localWasmBinaryPath = localWasmStem
  ? path.join(cwd, 'public', `${localWasmStem}.wasm`)
  : null;
if (wasmBuild && localWasmModulePath && localWasmBinaryPath) {
  await Promise.all([
    copyFile(path.join(wasmBuild, 'wllama.js'), localWasmModulePath),
    copyFile(path.join(wasmBuild, 'wllama.wasm'), localWasmBinaryPath),
  ]);
}
const [wasmModuleSha256, wasmBinarySha256] = wasmBuild
  ? await Promise.all([
      sha256File(path.join(wasmBuild, 'wllama.js')),
      sha256File(path.join(wasmBuild, 'wllama.wasm')),
    ])
  : [null, null];
const browserCandidates = [
  args.browser,
  process.env.CHROMIUM_EXECUTABLE_PATH,
  chromium.executablePath(),
  `${process.env.HOME}/.cache/ms-playwright/chromium-1148/chrome-linux/chrome`,
  `${process.env.HOME}/.cache/ms-playwright/chromium-1217/chrome-linux64/chrome`,
  `${process.env.HOME}/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome`,
].filter(Boolean);
const executablePath = browserCandidates.find((candidate) => {
  return existsSync(candidate);
});
if (!executablePath) {
  throw new Error(
    'Chromium executable is unavailable. Run npx playwright install chromium or pass --browser=...'
  );
}
const server = spawn(
  process.platform === 'win32' ? 'npm.cmd' : 'npm',
  ['run', 'dev', '--', '--host', '127.0.0.1', '--port', String(port)],
  { cwd, stdio: ['ignore', 'pipe', 'inherit'] }
);
const waitForServer = () =>
  new Promise((resolve, reject) => {
    const timer = setTimeout(
      () => reject(new Error('Vite startup timeout')),
      30000
    );
    server.stdout.on('data', (chunk) => {
      if (String(chunk).includes('Local:')) {
        clearTimeout(timer);
        resolve();
      }
    });
    server.once('exit', (code) =>
      reject(new Error(`Vite exited with ${code}`))
    );
  });
let browser;
try {
  await waitForServer();
  browser = await chromium.launch({
    executablePath,
    headless: args.headless === 'true',
    args: [
      ...(os.platform() === 'linux' ? ['--no-sandbox'] : []),
      '--enable-unsafe-webgpu',
      '--ignore-gpu-blocklist',
    ],
  });
  const page = await browser.newPage({ acceptDownloads: false });
  page.on('pageerror', (error) =>
    console.error(`[pageerror] ${error.message}`)
  );
  page.on('console', (message) => {
    if (message.type() === 'error' || args.verbose === 'true') {
      console.error(`[browser:${message.type()}] ${message.text()}`);
    }
  });
  page.setDefaultTimeout(30000);
  page.setDefaultNavigationTimeout(30000);
  const url = new URL(`http://127.0.0.1:${port}/`);
  url.searchParams.set('backend', backend);
  url.searchParams.set('gpu', args.gpu ?? '1');
  if (modelUrl) url.searchParams.set('model', modelUrl);
  if (localWasmStem) {
    url.searchParams.set(
      'wllamaModule',
      `http://127.0.0.1:${port}/${localWasmStem}.js`
    );
    url.searchParams.set(
      'wllamaWasm',
      `http://127.0.0.1:${port}/${localWasmStem}.wasm`
    );
  }
  if (args['webllm-model']) {
    url.searchParams.set('webllmModel', args['webllm-model']);
  }
  url.searchParams.set('evalCount', args['eval-count'] ?? '120');
  url.searchParams.set('outputTokens', args['output-tokens'] ?? '48');
  await page.goto(url.href, { waitUntil: 'networkidle' });
  const browserRuntime = await page.evaluate(async () => {
    const result = {
      userAgent: navigator.userAgent,
      hardwareConcurrency: navigator.hardwareConcurrency,
      deviceMemoryGiB:
        'deviceMemory' in navigator ? Number(navigator.deviceMemory) : null,
      webgpu: {
        supported: Boolean(navigator.gpu),
        adapterAvailable: false,
        vendor: null,
        architecture: null,
        device: null,
        description: null,
      },
    };
    if (!navigator.gpu) return result;
    const adapter = await navigator.gpu.requestAdapter();
    result.webgpu.adapterAvailable = Boolean(adapter);
    if (adapter) {
      result.webgpu = {
        supported: true,
        adapterAvailable: true,
        vendor: adapter.info?.vendor || null,
        architecture: adapter.info?.architecture || null,
        device: adapter.info?.device || null,
        description: adapter.info?.description || null,
      };
    }
    return result;
  });
  if (
    args['require-webgpu'] !== 'false' &&
    !browserRuntime.webgpu.adapterAvailable
  ) {
    throw new Error('WebGPU adapter is required for the external baseline');
  }
  await page.getByRole('button', { name: 'Load Dataset' }).click();
  await page.waitForFunction(
    () =>
      !document.body.innerText.includes('Total loaded tasks: 0') ||
      Boolean(window.wllmWebArenaError)
  );
  const datasetError = await page.evaluate(() => window.wllmWebArenaError);
  if (datasetError) {
    throw new Error(`WebArena dataset loading failed: ${datasetError}`);
  }
  await page.getByRole('button', { name: 'Run Exp1' }).click();
  await page.waitForFunction(
    () =>
      Boolean(
        window.wllmWebArenaReport && window.wllmWebArenaReport.webarena
      ) || Boolean(window.wllmWebArenaError),
    undefined,
    { timeout: Number(args['timeout-ms'] ?? 7200000) }
  );
  const pageError = await page.evaluate(() => window.wllmWebArenaError);
  if (pageError) throw new Error(`WebArena benchmark failed: ${pageError}`);
  const report = await page.evaluate(() => window.wllmWebArenaReport);
  if (
    backend === 'wllama' &&
    args['require-webgpu'] !== 'false' &&
    report?.runtime?.usingWebGPU !== true
  ) {
    throw new Error(
      'WebGPU was required but the wllama runtime did not activate it'
    );
  }
  const wrapped = {
    protocolVersion: 'wllm.webarena-baseline.v1',
    gitRevision,
    gitDirty,
    deviceId,
    operator: args.operator ?? deviceMetadata?.operator ?? 'unassigned',
    deviceMetadata,
    backend,
    browser: await browser.version(),
    environment: {
      operatingSystem: `${os.platform()} ${os.release()} ${os.arch()}`,
      cpuModel: os.cpus()[0]?.model ?? 'unknown',
      logicalCpuCount: os.cpus().length,
      totalMemoryBytes: os.totalmem(),
      browserRuntime,
    },
    model: {
      source:
        backend === 'web-llm'
          ? (args['webllm-model'] ?? 'Llama-3.2-1B-Instruct-q4f32_1-MLC')
          : modelUrl,
      sha256: modelSha256,
    },
    wllamaRuntime:
      backend === 'wllama'
        ? {
            sourceBuild: wasmBuild,
            moduleSha256: wasmModuleSha256,
            wasmSha256: wasmBinarySha256,
          }
        : null,
    report,
  };
  await mkdir(path.dirname(output), { recursive: true });
  await writeFile(output, `${JSON.stringify(wrapped, null, 2)}\n`);
  console.log(JSON.stringify({ output, backend }));
} finally {
  await browser?.close().catch(() => {});
  if (server.exitCode === null && server.signalCode === null) {
    server.kill('SIGTERM');
  }
  if (localModelPath) {
    await rm(localModelPath, { force: true });
  }
  await Promise.all([
    localWasmModulePath
      ? rm(localWasmModulePath, { force: true })
      : Promise.resolve(),
    localWasmBinaryPath
      ? rm(localWasmBinaryPath, { force: true })
      : Promise.resolve(),
  ]);
}
