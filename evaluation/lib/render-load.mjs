import { openAppPage } from "./browser.mjs";

export async function openPressurePage(context, baseUrl, options = {}) {
  const url = new URL("/render/pressure.html", baseUrl);
  url.searchParams.set("triangles", String(options.triangles ?? 0));
  url.searchParams.set("complexity", String(options.complexity ?? 128));
  const page = await openAppPage(context, url.toString(), {
    timeoutMs: options.timeoutMs ?? 300_000
  });
  await page.waitForFunction(
    () => window.__wllmPressure !== undefined,
    null,
    { timeout: options.timeoutMs ?? 300_000 }
  );
  return page;
}

export async function waitForPressureWebGPU(page, timeoutMs = 300_000) {
  await page.waitForFunction(
    () => {
      const pressure = window.__wllmPressure;
      return pressure !== undefined && pressure.mode !== "booting" && pressure.mode !== "unavailable" && pressure.mode !== "error";
    },
    null,
    { timeout: timeoutMs }
  );
  const error = await page.evaluate(() => window.__wllmPressure?.error ?? null);
  if (error !== null) {
    throw new Error(`pressure page failed: ${error}`);
  }
}

export async function setPressure(page, triangles) {
  await page.evaluate((value) => window.__wllmPressure.setTriangles(value), triangles);
}

export async function samplePressure(page) {
  return page.evaluate(() => window.__wllmPressure.snapshot());
}

export async function resetPressureStats(page) {
  await page.evaluate(() => window.__wllmPressure.resetStats());
}

export async function pressureFocus(page) {
  await page.bringToFront();
}

export async function stopPressure(page) {
  await page.evaluate(() => window.__wllmPressure.setTriangles(0));
}
