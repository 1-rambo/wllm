import { access, readdir } from "node:fs/promises";
import { mkdtemp, rm } from "node:fs/promises";
import { execFileSync } from "node:child_process";
import os from "node:os";
import path from "node:path";
import { chromium } from "playwright-core";

export async function findChromePath() {
  const configured = process.env.CHROME_PATH;
  if (configured) {
    await access(configured);
    return configured;
  }
  const candidates = chromeCandidates();
  for (const candidate of candidates) {
    try {
      await access(candidate);
      return candidate;
    } catch {
      // Try the next platform-specific location.
    }
  }
  for (const name of ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome"]) {
    try {
      const resolved = execFileSync(process.platform === "win32" ? "where" : "which", [name], {
        encoding: "utf8",
        stdio: ["ignore", "pipe", "ignore"]
      }).trim().split(/\r?\n/)[0];
      if (resolved) return resolved;
    } catch {
      // The command is not on PATH.
    }
  }
  throw new Error(
    "Chrome/Chromium not found. Install Chrome or set CHROME_PATH to its executable."
  );
}

export async function launchBrowserProcess(options = {}) {
  const executablePath = options.executablePath ?? await findChromePath();
  const headless = options.headless ?? process.env.WLLM_HEADLESS === "1";
  const args = [
    "--enable-unsafe-webgpu",
    "--enable-features=Vulkan,WebGPU,SharedArrayBuffer",
    "--disable-dev-shm-usage",
    "--disable-background-timer-throttling",
    "--disable-renderer-backgrounding",
    "--disable-backgrounding-occluded-windows",
    "--autoplay-policy=no-user-gesture-required"
  ];
  if (process.platform === "linux") {
    args.push("--no-sandbox");
  }
  return chromium.launch({
    executablePath,
    headless,
    args
  });
}

export async function launchEvaluationBrowser(options = {}) {
  const browser = await launchBrowserProcess(options);
  const context = await browser.newContext({
    viewport: { width: 1440, height: 980 },
    deviceScaleFactor: 1,
    acceptDownloads: false
  });
  return { browser, context };
}

export async function launchPersistentEvaluationBrowser(options = {}) {
  // A normal Chrome profile has usable persistent OPFS storage. Playwright's
  // incognito-style context can reject a multi-gigabyte GGUF load even when
  // navigator.storage.estimate() advertises a larger quota. Each caller gets
  // a fresh directory, so trials remain isolated without imposing incognito
  // storage limits on the natural per-tab wllama deployment.
  const executablePath = options.executablePath ?? await findChromePath();
  const headless = options.headless ?? process.env.WLLM_HEADLESS === "1";
  const userDataDir = await mkdtemp(path.join(os.tmpdir(), "wllm-eval-profile-"));
  const args = [
    "--enable-unsafe-webgpu",
    "--enable-features=Vulkan,WebGPU,SharedArrayBuffer",
    "--disable-dev-shm-usage",
    "--disable-background-timer-throttling",
    "--disable-renderer-backgrounding",
    "--disable-backgrounding-occluded-windows",
    "--autoplay-policy=no-user-gesture-required"
  ];
  if (process.platform === "linux") args.push("--no-sandbox");
  try {
    const context = await chromium.launchPersistentContext(userDataDir, {
      executablePath,
      headless,
      args,
      viewport: { width: 1440, height: 980 },
      deviceScaleFactor: 1,
      acceptDownloads: false
    });
    const browser = context.browser();
    return {
      browser,
      context,
      userDataDir,
      async close() {
        await withTimeout(context.close(), 15_000, "persistent browser close timed out")
          .catch(() => undefined);
        await rm(userDataDir, { recursive: true, force: true }).catch(() => undefined);
      }
    };
  } catch (error) {
    await rm(userDataDir, { recursive: true, force: true }).catch(() => undefined);
    throw error;
  }
}

export async function openAppPage(context, url, options = {}) {
  // context.newPage() has no Playwright timeout and can hang forever when the
  // browser is wedged; bound it so a stuck trial fails instead of stalling the run.
  const page = await withTimeout(
    context.newPage(),
    options.newPageTimeoutMs ?? 60_000,
    "context.newPage timed out"
  );
  page.setDefaultTimeout(options.timeoutMs ?? 300_000);
  page.setDefaultNavigationTimeout(options.timeoutMs ?? 300_000);
  await page.goto(url, { waitUntil: "domcontentloaded", timeout: options.timeoutMs ?? 300_000 });
  return page;
}

export function withTimeout(promise, timeoutMs, message) {
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

export async function closePageQuietly(page, timeoutMs = 5_000) {
  try {
    await Promise.race([
      page.close(),
      new Promise((resolve) => setTimeout(resolve, timeoutMs))
    ]);
  } catch {
    // A crashed or unresponsive page is already gone.
  }
}

export async function closeBrowserQuietly(browser, timeoutMs = 5_000) {
  try {
    await withTimeout(browser.close(), timeoutMs, "browser close timed out");
    return;
  } catch {
    // Chromium can stop responding after ten WebGPU tabs. The CLI performs a
    // final explicit exit after every artifact has been flushed.
  }
}

export async function listChromeProfilesRoot() {
  const root = path.join(process.env.HOME ?? process.env.USERPROFILE ?? ".", "Library", "Application Support", "Google", "Chrome");
  try {
    return await readdir(root);
  } catch {
    return [];
  }
}

function chromeCandidates() {
  if (process.platform === "darwin") {
    return [
      "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
      "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary",
      "/Applications/Chromium.app/Contents/MacOS/Chromium",
      "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"
    ];
  }
  if (process.platform === "win32") {
    const programFiles = process.env.ProgramFiles ?? "C:\\Program Files";
    const localAppData = process.env.LOCALAPPDATA ?? "";
    return [
      path.join(programFiles, "Google", "Chrome", "Application", "chrome.exe"),
      path.join(programFiles, "Google", "Chrome Beta", "Application", "chrome.exe"),
      path.join(programFiles, "Microsoft", "Edge", "Application", "msedge.exe"),
      path.join(localAppData, "Google", "Chrome", "Application", "chrome.exe"),
      path.join(localAppData, "Microsoft", "Edge", "Application", "msedge.exe")
    ];
  }
  return [
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "/snap/bin/chromium"
  ];
}
