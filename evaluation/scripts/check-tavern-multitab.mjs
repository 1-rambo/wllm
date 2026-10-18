#!/usr/bin/env node
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createEvaluationServer } from "../lib/app-server.mjs";
import { launchBrowserProcess } from "../lib/browser.mjs";
import {
  samplePressure,
  waitForPressureWebGPU
} from "../lib/render-load.mjs";

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const implementationRoot = path.resolve(scriptDirectory, "../..");
const timestamp = new Date().toISOString().replaceAll(":", "-").replaceAll(".", "-");
const outputDirectory = path.join(
  implementationRoot,
  "artifacts/evaluation/visual-preflight",
  timestamp
);
await mkdir(outputDirectory, { recursive: true });

const server = await createEvaluationServer();
const browser = await launchBrowserProcess({ headless: false });
try {
  const context = await browser.newContext({
    viewport: { width: 1440, height: 980 },
    deviceScaleFactor: 1
  });
  await context.route("**/evaluation/app-adapter.js", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/javascript",
      body: ""
    })
  );

  const pages = [];
  const errors = [];
  for (let index = 0; index < 9; index += 1) {
    const page = await context.newPage();
    page.on("pageerror", (error) => errors.push({ tab: index, error: error.message }));
    page.on("requestfailed", (request) =>
      errors.push({
        tab: index,
        request: request.url(),
        error: request.failure()?.errorText ?? null
      })
    );
    await page.goto(
      `${server.url}/apps/tavern/index.html?multitab-check=${index}`,
      { waitUntil: "domcontentloaded", timeout: 120_000 }
    );
    await page.waitForSelector("#game-container canvas", {
      state: "attached",
      timeout: 30_000
    });
    pages.push(page);
  }

  const pressure = await context.newPage();
  pressure.on("pageerror", (error) => errors.push({ tab: "pressure", error: error.message }));
  pressure.on("console", (message) => {
    if (message.type() === "error") errors.push({ tab: "pressure", error: message.text() });
  });
  await pressure.goto(
    `${server.url}/render/pressure.html?triangles=1000&complexity=128`,
    { waitUntil: "domcontentloaded", timeout: 120_000 }
  );
  await pressure.waitForFunction(() => document.querySelector("canvas") !== null, null, {
    timeout: 30_000
  });
  await waitForPressureWebGPU(pressure, 60_000);
  await pressure.bringToFront();
  await pressure.waitForFunction(() => document.visibilityState === "visible", null, {
    timeout: 30_000
  });
  await pressure.waitForTimeout(1_500);
  const pressureBefore = await samplePressure(pressure);
  await pressure.waitForFunction(
    (initial) => window.__wllmPressure.snapshot().frameCount > initial + 1,
    pressureBefore.frameCount,
    { timeout: 30_000 }
  );
  const pressureAfter = await samplePressure(pressure);
  await pressure.screenshot({
    path: path.join(outputDirectory, "pressure-1000-triangles.png")
  });
  const pressureHud = await pressure.locator("#status").innerText();
  if (
    pressureAfter.mode !== "webgpu" ||
    pressureAfter.diagnosticPattern !== "visible-color-triangles-v1" ||
    pressureAfter.baseTriangleCount !== 1000 ||
    pressureAfter.instanceCopiesPerTriangle !== 128 ||
    pressureAfter.targetTriangles !== 1000 ||
    pressureAfter.submittedInstancesPerFrame !== 128_000 ||
    pressureAfter.visibilityState !== "visible" ||
    pressureAfter.frameCount <= pressureBefore.frameCount
  ) {
    throw new Error(`pressure visual check failed: ${JSON.stringify(pressureAfter)}`);
  }
  await pages[0].bringToFront();
  await pages[0].waitForTimeout(2_000);
  await pages[0].screenshot({
    path: path.join(outputDirectory, "tavern-9-tabs-pressure-1000.png")
  });
  const report = {
    checkedAt: new Date().toISOString(),
    outputDirectory,
    canvasCount: await pages[0].locator("#game-container canvas").count(),
    phaserVersion: await pages[0].evaluate(() => window.Phaser?.VERSION ?? null),
    canvasSize: await pages[0].locator("#game-container canvas").evaluate((canvas) => ({
      width: canvas.width,
      height: canvas.height
    })),
    pressure: {
      before: pressureBefore,
      after: pressureAfter,
      hud: pressureHud,
      screenshot: path.join(outputDirectory, "pressure-1000-triangles.png")
    },
    tavernScreenshot: path.join(outputDirectory, "tavern-9-tabs-pressure-1000.png"),
    errors
  };
  await writeFile(
    path.join(outputDirectory, "report.json"),
    `${JSON.stringify(report, null, 2)}\n`
  );
  console.log(JSON.stringify(report, null, 2));
} finally {
  await browser.close();
  await server.close();
}
