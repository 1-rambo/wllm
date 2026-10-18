#!/usr/bin/env node
import { mkdir } from "node:fs/promises";
import path from "node:path";
import { createEvaluationServer } from "../lib/app-server.mjs";
import { launchBrowserProcess } from "../lib/browser.mjs";

const outputDirectory = "/tmp/wllm-tavern-visual-check";
await mkdir(outputDirectory, { recursive: true });

const server = await createEvaluationServer();
const browser = await launchBrowserProcess({ headless: false });
try {
  const context = await browser.newContext({
    viewport: { width: 1440, height: 980 },
    deviceScaleFactor: 1
  })
  await context.route("**/evaluation/app-adapter.js", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/javascript",
      body: ""
    });
  });
  if (process.env.WITH_ADAPTER === "1") {
    await context.unroute("**/evaluation/app-adapter.js");
  }
  const page = await context.newPage();
  const consoleMessages = [];
  const pageErrors = [];
  const failedRequests = [];
  page.on("console", (message) => {
    consoleMessages.push({ type: message.type(), text: message.text() });
  });
  page.on("pageerror", (error) => pageErrors.push(error.message));
  page.on("requestfailed", (request) => {
    failedRequests.push({
      url: request.url(),
      error: request.failure()?.errorText ?? null
    });
  });

  const response = await page.goto(
    `${server.url}/apps/tavern/index.html?visual-check=1`,
    { waitUntil: "networkidle", timeout: 120_000 }
  );
  await page.waitForSelector("#game-container canvas", {
    state: "attached",
    timeout: 30_000
  }).catch(async (error) => {
    await page.screenshot({
      path: path.join(outputDirectory, "tavern-missing-canvas.png"),
      fullPage: false
    });
    console.log(
      JSON.stringify(
        {
          status: response?.status() ?? null,
          url: page.url(),
          title: await page.title(),
          canvasError: error instanceof Error ? error.message : String(error),
          dom: await page.evaluate(() => ({
            phaserLoaded: typeof window.Phaser !== "undefined",
            bodyText: document.body.innerText.slice(0, 500),
            gameContainerHtml: document.querySelector("#game-container")?.innerHTML ?? null,
            bodyBackground: getComputedStyle(document.body).backgroundColor
          })).catch((diagnosticError) => ({ error: diagnosticError.message })),
          consoleMessages,
          pageErrors,
          failedRequests
        },
        null,
        2
      )
    );
    throw error;
  });
  await page.waitForTimeout(3_000);

  const canvasState = await page.evaluate(() => {
    const canvas = document.querySelector("#game-container canvas");
    if (canvas === null) return { exists: false };
    const gl = canvas.getContext("webgl2") ?? canvas.getContext("webgl");
    let pixels = null;
    if (gl !== null) {
      const data = new Uint8Array(canvas.width * canvas.height * 4);
      gl.readPixels(
        0,
        0,
        canvas.width,
        canvas.height,
        gl.RGBA,
        gl.UNSIGNED_BYTE,
        data
      );
      let nonBlack = 0;
      let nonOpaque = 0;
      let max = 0;
      for (let index = 0; index < data.length; index += 4) {
        const value = data[index] + data[index + 1] + data[index + 2];
        if (value > 0) nonBlack += 1;
        if (data[index + 3] > 0) nonOpaque += 1;
        if (value > max) max = value;
      }
      pixels = {
        total: canvas.width * canvas.height,
        nonBlack,
        nonOpaque,
        nonBlackFraction: nonBlack / (canvas.width * canvas.height),
        maxRgbSum: max
      };
    }
    const dataUrl = canvas.toDataURL("image/png");
    return {
      exists: true,
      width: canvas.width,
      height: canvas.height,
      clientWidth: canvas.clientWidth,
      clientHeight: canvas.clientHeight,
      display: getComputedStyle(canvas).display,
      visibility: getComputedStyle(canvas).visibility,
      bodyBackground: getComputedStyle(document.body).backgroundColor,
      phaserVersion: window.Phaser?.VERSION ?? null,
      rendererType: window.Phaser?.WEBGL_RENDERER ?? null,
      dataUrlLength: dataUrl.length,
      dataUrlPrefix: dataUrl.slice(0, 64),
      pixels
    };
  });

  await page.screenshot({
    path: path.join(outputDirectory, "tavern.png"),
    fullPage: false
  });
  console.log(
    JSON.stringify(
      {
        status: response?.status() ?? null,
        url: page.url(),
        title: await page.title(),
        canvasState,
        consoleMessages,
        pageErrors,
        failedRequests
      },
      null,
      2
    )
  );
} finally {
  await browser.close();
  await server.close();
}
