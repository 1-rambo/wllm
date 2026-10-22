#!/usr/bin/env node
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { access, mkdir, stat } from "node:fs/promises";
import path from "node:path";
import {
  armFor,
  effectiveMode,
  experimentAppTabs,
  experimentFor,
  loadEvaluationConfig,
  modelById,
  modelUrlFor,
  modelsForExperiment,
  platformSupportsExperiment,
  pressureSchedule
} from "./lib/config.mjs";
import { artifactRoot, implRoot, platformId, timestampId } from "./lib/paths.mjs";
import { createEvaluationServer } from "./lib/app-server.mjs";
import { assertRuntimeBundleFresh } from "./lib/runtime-bundle.mjs";
import {
  closePageQuietly,
  closeBrowserQuietly,
  launchEvaluationBrowser,
  launchPersistentEvaluationBrowser,
  openAppPage
} from "./lib/browser.mjs";
import {
  openPressurePage,
  resetPressureStats,
  samplePressure,
  setPressure,
  waitForPressureWebGPU
} from "./lib/render-load.mjs";
import { PressureRelay } from "./lib/pressure-relay.mjs";
import { normalizedTargets } from "./lib/targets.mjs";
import {
  clickResponseSeed,
  createClickSequencer,
  isValidClickResponse
} from "./lib/click-response.mjs";
import { RunArtifacts, writeJsonAtomic } from "./lib/records.mjs";
import { buildSummary } from "./lib/summary.mjs";
import { evaluateGates } from "./lib/gate.mjs";
import { createJudgeRunner, scoreRecords } from "./lib/judge.mjs";
import { median } from "./lib/statistics.mjs";

const cli = parseArgs(process.argv.slice(2));
const loaded = await loadEvaluationConfig();
const experimentId = cli.experiment ?? "e2e-1";
const mode = cli.mode ?? "smoke";
await assertExperimentRunnable(loaded, experimentId, mode);
const experiment = experimentFor(loaded, experimentId);
const modeConfig = effectiveMode(loaded, mode);
const schedule = pressureSchedule(loaded, mode);
const layout = experimentAppTabs(loaded, mode);
const selectedModels = selectModels(loaded, experimentId, mode, cli);
if (selectedModels.length === 0) {
  throw new Error(`model selection is empty for ${experimentId}`);
}
if (cli.out && selectedModels.length !== 1) {
  throw new Error('--out is only valid with a single selected model; use --model <id>');
}

const startedAt = new Date();
const runtimeBundle = await assertRuntimeBundleFresh();
const server = await createEvaluationServer();
// Each measured trial gets its own browser process so a wedged browser or leftover
// GPU state from a previous trial cannot stall the next one. The judge keeps a
// lazily created browser of its own.
let browserState = null;
const ensureBrowserState = async () => {
  if (browserState === null) {
    browserState = await launchEvaluationBrowser({ headless: cli.headless });
  }
  return browserState;
};
const runRoot = path.join(implRoot, experiment.artifactRoot);
const runDirectories = [];

try {
  for (const model of selectedModels) {
    await assertModelReady(model);
    const outputDirectory = cli.out
      ? path.resolve(cli.out)
      : resolveRunDirectory(experiment, model, startedAt);
    runDirectories.push(outputDirectory);
    const artifacts = await new RunArtifacts(outputDirectory).open();
    await artifacts.writeJson("run.json", {
      schemaVersion: 2,
      experiment: experimentId,
      experimentLabel: experiment.label,
      mode,
      platform: platformId(),
      platformLabel: loaded.config.platforms[platformId()]?.label ?? platformId(),
      model: model.id,
      modelHash: model.sha256,
      modelFile: model.filename,
      promptFormat: model.promptFormat,
      modelLoadMode: "per-arm",
      modelLoadModes: Object.fromEntries(
        experiment.arms.map((armId) => [
          armId,
          modelLoadModeForArm(armFor(loaded, armId, experimentId))
        ])
      ),
      browserProfile: "isolated-persistent",
      execution: loaded.config.execution,
      executionByArm: Object.fromEntries(
        experiment.arms.map((armId) => [
          armId,
          executionForArm(
            armFor(loaded, armId, experimentId),
            loaded.config.execution
          )
        ])
      ),
      runtimeBundle,
      runtimeArmOverrides: loaded.runtimeArmOverrides,
      appServerRoots: server.roots,
      layout,
      clickResponse: Array.isArray(loaded.config.clickResponse?.enabledForExperiments) &&
        loaded.config.clickResponse.enabledForExperiments.includes(experimentId)
        ? loaded.config.clickResponse
        : null,
      recoveryLayout: experiment.scenario === "recovery"
        ? {
            nativeWorkerHost: "first-app-tab",
            targetPolicy: "non-host app tabs in load order",
            targetCandidates: Math.max(1, layout.appTabs - 1)
          }
        : null,
      schedule,
      command: process.argv.join(" "),
      startedAt: startedAt.toISOString(),
      config: path.relative(implRoot, loaded.file),
      configSha256: loaded.configSha256,
      runtimeConfig:
        loaded.runtimeConfigFile === null
          ? null
          : path.relative(implRoot, loaded.runtimeConfigFile),
      runtimeConfigSha256: loaded.runtimeConfigSha256,
      preregistration: path.relative(implRoot, loaded.preregistrationFile),
      preregistrationSha256: sha256(JSON.stringify(loaded.preregistration)),
      outputDirectory
    });
    try {
      const modelRecords = await runModel({
        loaded,
        experiment,
        mode,
        modeConfig,
        schedule,
        layout,
        model,
        headless: cli.headless,
        serverUrl: server.url,
        artifacts
      });
      let records = modelRecords;
      if (cli.judge && modeConfig.judge !== false && experiment.requiresJudge) {
        records = await runJudge({
          loaded,
          experiment,
          model,
          browser: (await ensureBrowserState()).browser,
          serverUrl: server.url,
          records
        });
        // The judge enriches records after records.jsonl has been streamed
        // to disk. Keep the enriched copy so a judge verdict can be traced
        // back to the raw model output that produced it.
        await artifacts.writeJsonl("judged-records.jsonl", records);
      }
      const summary = buildSummary(records, {
        armPairs: [[experiment.arms[0], experiment.arms[1]]],
        expected: expectedRecordCount(loaded, experimentId, mode, [model], layout)
      });
      const complete = summary.completeness.complete === true;
      const gateReport = evaluateGates({
        loaded,
        experimentId,
        records,
        summary,
        complete
      });
      await artifacts.writeJson("summary.json", summary);
      await artifacts.writeJson("gate-report.json", gateReport);
      await artifacts.writeJson("latest.json", {
        experiment: experimentId,
        model: model.id,
        run: path.basename(outputDirectory),
        updatedAt: new Date().toISOString()
      });
      console.log(`E2E RUN COMPLETE ${outputDirectory}`);
      console.log(
        `records=${records.length} complete=${complete} errors=${summary.errorCount}`
      );
      for (const gate of gateReport.gates) {
        console.log(`${gate.id}: ${gate.status}`);
      }
    } finally {
      await artifacts.close();
    }
  }
  await writeJsonAtomic(path.join(runRoot, "latest.json"), {
    experiment: experimentId,
    updatedAt: new Date().toISOString(),
    runs: runDirectories
  });
} catch (error) {
  const failureDirectory = runDirectories.at(-1) ?? path.join(runRoot, "failed", timestampId());
  await mkdir(failureDirectory, { recursive: true });
  await writeJsonAtomic(path.join(failureDirectory, "failure.json"), {
    experiment: experimentId,
    mode,
    at: new Date().toISOString(),
    error: error instanceof Error ? error.stack ?? error.message : String(error)
  });
  throw error;
} finally {
  if (browserState !== null) {
    await closeBrowserQuietly(browserState.browser);
  }
  await server.close().catch(() => undefined);
}

// Stage logging stays on stderr so a stalled trial is identifiable from the run log
// without instrumenting the measured paths themselves.
function stageLog(message) {
  console.error(`[e2e ${new Date().toISOString()}] ${message}`);
}

async function runModel(options) {
  const {
    loaded,
    experiment,
    mode,
    modeConfig,
    schedule,
    layout,
    model,
    headless,
    serverUrl,
    artifacts
  } = options;
  const records = [];
  for (let repeat = 0; repeat < modeConfig.repeats; repeat += 1) {
    for (let pressureIndex = 0; pressureIndex < schedule.length; pressureIndex += 1) {
      const pressure = schedule[pressureIndex];
      const trialOrdinal = repeat * schedule.length + pressureIndex;
      const arms = trialOrdinal % 2 === 0
        ? experiment.arms
        : [...experiment.arms].reverse();
      for (const [armIndex, armId] of arms.entries()) {
        const stage = `${armId} ${pressure.id} repeat=${repeat}`;
        stageLog(`trial start ${stage}`);
        // Use a fresh normal Chrome profile for every trial. wllama's native
        // per-tab deployment relies on OPFS, whose multi-gigabyte writes are
        // rejected by Playwright's incognito-style temporary context.
        const trialBrowser = await withTimeout(
          launchPersistentEvaluationBrowser({ headless }),
          60_000,
          "browser launch timed out"
        );
        const context = trialBrowser.context;
        stageLog(`trial browser launched ${stage}`);
        let trial = null;
        try {
          trial = await createTrial({
            loaded,
            experiment,
            mode,
            modeConfig,
            pressure,
            repeat,
            armId,
            trialOrdinal,
            armIndex,
            arm: armFor(loaded, armId, experimentId),
            model,
            layout,
            context,
            serverUrl
          });
          const result = experiment.scenario === "recovery"
            ? await runRecoveryTrial(trial)
            : await runPriorityTrial(trial);
          stageLog(`trial measured ${stage}`);
          for (const event of result.events) {
            await artifacts.event(event);
          }
          for (const record of result.records) {
            records.push(record);
            await artifacts.record(record);
          }
        } catch (error) {
          const evidence = trial === null
            ? null
            : await collectTrialFailureEvidence(trial);
          const failure = {
            kind: "failure",
            experiment: experiment.id,
            mode,
            platform: platformId(),
            model: model.id,
            arm: armId,
            pressureId: pressure.id,
            pressureTriangles: pressure.triangles,
            repeat,
            error: error instanceof Error ? error.message : String(error),
            evidence,
            at: new Date().toISOString()
          };
          records.push(failure);
          await artifacts.record(failure);
          await artifacts.event({ kind: "trial-failure", ...failure });
          console.error(`trial failed ${armId} ${pressure.id} repeat=${repeat}: ${failure.error}`);
          stageLog(`trial failed ${stage}: ${failure.error}`);
        } finally {
          if (trial !== null) {
            await withTimeout(
              trial.close(),
              15_000,
              "trial close timed out"
            ).catch(() => undefined);
          }
          await trialBrowser.close();
          stageLog(`trial teardown ${stage}: browser closed`);
        }
      }
    }
  }
  return records;
}

async function collectTrialFailureEvidence(trial) {
  const pages = await Promise.all(
    trial.appPages.map(async (entry) => {
      const [snapshot, diagnostics] = await Promise.all([
        withTimeout(pageCall(entry.page, "snapshot", null), 5_000, "failure snapshot timed out")
          .catch((error) => ({ error: error.message })),
        withTimeout(pageCall(entry.page, "diagnostics", null), 5_000, "failure diagnostics timed out")
          .catch((error) => ({ error: error.message }))
      ]);
      return {
        app: entry.app,
        tabIndex: entry.tabIndex,
        nativeHost: entry.nativeHost === true,
        pageErrors: [...entry.pageErrors],
        snapshot,
        diagnostics
      };
    })
  );
  const pressureVisibility = await pressureVisibilityState(trial.pressurePage);
  return { pages, pressureVisibilityState: pressureVisibility };
}

async function waitForAppReady(page, timeoutMs) {
  const startedAtMs = Date.now();
  let lastState = null;
  let loggedDiagnostics = false;
  while (Date.now() - startedAtMs < timeoutMs) {
    try {
      lastState = await page.evaluate(() => ({
        ready: window.__wllmE2E?.ready?.() === true,
        snapshot: window.__wllmE2E?.snapshot?.() ?? null,
        iframes: [...document.querySelectorAll("iframe")].map((frame) => {
          try {
            return {
              src: frame.src,
              status: frame.contentWindow?.__wllmEnginePage?.status?.() ?? null
            };
          } catch (error) {
            return {
              src: frame.src,
              error: error instanceof Error ? error.message : String(error)
            };
          }
        })
      }));
      if (lastState.ready) return;
      const adapterError = lastState.snapshot?.error ?? null;
      if (adapterError !== null) {
        throw new Error(`application adapter failed: ${adapterError}`);
      }
      const engineError = lastState.iframes
        .map((entry) => entry.status?.error ?? entry.error ?? null)
        .find((error) => error !== null);
      if (engineError) {
        throw new Error(`engine page failed: ${engineError}`);
      }
    } catch (error) {
      if (
        error instanceof Error &&
        (error.message.startsWith("application adapter failed:") ||
          error.message.startsWith("engine page failed:"))
      ) {
        throw error;
      }
    }
    if (!loggedDiagnostics && Date.now() - startedAtMs >= 15_000) {
      loggedDiagnostics = true;
      stageLog(`application readiness diagnostics: ${JSON.stringify(lastState)}`);
    }
    await delay(250);
  }
  throw new Error(
    `application did not become ready within ${timeoutMs} ms: ${JSON.stringify(lastState)}`
  );
}
async function createTrial(options) {
  const {
    loaded,
    experiment,
    mode,
    modeConfig,
    pressure,
    repeat,
    armId,
    arm,
    model,
    layout,
    context,
    serverUrl
  } = options;
  const trialOrdinal = options.trialOrdinal ?? 0;
  const definitions = appDefinitionsFor(layout, loaded.config);
  const appPages = [];
  let pressurePage = null;
  try {
    // Load every application while it owns the foreground tab. Loading an
    // app behind the pressure page can leave the wllama OPFS prewarm in a
    // background-throttled state, which later surfaces as a generic ready
    // timeout.
    for (const [tabIndex, definition] of definitions.entries()) {
      const sessionId = `${armId}-${definition.app}-${repeat}-${tabIndex}`;
      stageLog(
        `open app tab ${tabIndex + 1}/${definitions.length} ${definition.app} (${armId} ${pressure.id} repeat=${repeat})`
      );
      const pageUrl = new URL(
        appUrl(serverUrl, definition, {
          armId,
          arm,
          model,
          sessionId,
          modeConfig,
          execution: executionForArm(arm, loaded.config.execution)
        })
      );
      const page = await openAppPage(
        context,
        pageUrl.toString(),
        { timeoutMs: modeConfig.loadTimeoutMs }
      );
      page.setDefaultTimeout(modeConfig.taskTimeoutMs);
      page.setDefaultNavigationTimeout(modeConfig.loadTimeoutMs);
      const pageErrors = [];
      page.on("pageerror", (error) => pageErrors.push(error.message));
      page.on("console", (message) => {
        if (message.type() === "error") pageErrors.push(message.text());
      });
      const entry = {
        app: definition.app,
        task: definition.task,
        tabIndex,
        sessionId,
        page,
        pageErrors,
        engine: arm.engine,
        armId
      };
      appPages.push(entry);
      await ensureVisible(page, `${definition.app} app tab ${tabIndex}`);
      stageLog(`waiting for ${definition.app} tab ${tabIndex} readiness`);
      try {
        await waitForAppReady(page, modeConfig.loadTimeoutMs);
        await assertEngineConfiguration(
          page,
          arm,
          executionForArm(arm, loaded.config.execution)
        );
      } catch (error) {
        const detail = pageErrors.length === 0 ? "no page errors captured" : pageErrors.slice(-8).join(" | ");
        throw new Error(
          `${definition.app} tab ${tabIndex} was not ready within ${modeConfig.loadTimeoutMs} ms: ${detail}`
        );
      }
    }

    // Open the render load only after all model sessions are ready. Starting
    // it at zero also keeps model initialization outside the measured
    // pressure window; setPressure below establishes the requested load.
    stageLog(`opening pressure tab (${armId} ${pressure.id} repeat=${repeat})`);
    pressurePage = await openPressurePage(context, serverUrl, {
      triangles: 0,
      complexity: loaded.config.pressure.complexity,
      timeoutMs: modeConfig.loadTimeoutMs
    });
    await ensureVisible(pressurePage, "pressure tab");
    stageLog("waiting for pressure WebGPU readiness");
    await waitForPressureWebGPU(pressurePage, modeConfig.loadTimeoutMs);
    await withTimeout(
      resetPressureStats(pressurePage),
      8_000,
      "pressure stats reset timed out"
    );
    await withTimeout(
      setPressure(pressurePage, pressure.triangles),
      8_000,
      "pressure set timed out"
    );
    stageLog(`trial pages ready, pressure=${pressure.triangles} triangles`);
  } catch (error) {
    await Promise.all(appPages.map((entry) => closePageQuietly(entry.page)));
    if (pressurePage !== null) await closePageQuietly(pressurePage);
    throw error;
  }

  return {
    loaded,
    experiment,
    mode,
    modeConfig,
    pressure,
    repeat,
    armId,
    arm,
    model,
    context,
    serverUrl,
    pressurePage,
    appPages,
    trialOrdinal,
    async close() {
      await Promise.all(appPages.map((entry) => closePageQuietly(entry.page)));
      await closePageQuietly(pressurePage);
    }
  };
}

async function runPriorityTrial(trial) {
  const {
    loaded,
    experiment,
    mode,
    modeConfig,
    pressure,
    repeat,
    armId,
    arm,
    model,
    appPages,
    pressurePage
  } = trial;
  const trialOrdinal = trial.trialOrdinal ?? 0;
  const targetIndices = normalizedTargets(
    loaded.config.priority.sequence,
    appPages.length,
    modeConfig.maxSwitchesPerTrial,
    trialOrdinal
  );
  const targets = new Set(targetIndices);
  const events = [];
  const relay = new PressureRelay({
    pressurePage,
    enginePages: appPages,
    intervalMs: loaded.config.pressure.pollMs,
    onSample: (sample) => events.push({ kind: "pressure-sample", at: new Date().toISOString(), sample })
  });
  let relayStarted = false;
  try {
  await ensureVisible(pressurePage, "pressure tab");
  await relay.start();
  relayStarted = true;
  stageLog(`relay started (${armId} ${pressure.id} repeat=${repeat})`);
  const startWallMs = Date.now();
  const taskEntries = appPages.map((entry) => {
    const item = {
      entry,
      requestId: `${armId}-${entry.app}-${repeat}-${entry.tabIndex}-${Date.now()}`,
      finished: false,
      promise: null,
      task: entry.task
    };
    item.promise = settleWithTimeout(
      pageCall(entry.page, "runTask", {
        phase: "priority",
        id: entry.task.id,
        prompt: entry.task.prompt,
        systemPrompt: entry.task.systemPrompt,
        maxNewTokens: modeConfig.maxNewTokens,
        temperature: loaded.config.execution.temperature,
        seed: loaded.config.execution.seed + entry.tabIndex,
        requestId: `${armId}-${entry.app}-${repeat}-${entry.tabIndex}`
      }),
      modeConfig.taskTimeoutMs
    ).finally(() => {
      item.finished = true;
    });
    return item;
  });
  stageLog(`tasks started (${taskEntries.length}) (${armId} ${pressure.id} repeat=${repeat})`);
  // Start all sessions as normal, then promote the first target while work is
  // already in flight. This matches the public priority protocol and lets a
  // promotion take the same preemption path as later target switches.
  await delay(loaded.config.priority.startDelayMs ?? 0);
  const initialPriority = await setPriority(
    appPages[targetIndices[0]],
    "high",
    armId,
    `priority-${pressure.id}-${repeat}`
  );
  events.push(initialPriority);
  const switchPromise = switchPriorityTargets({
    appPages,
    targetIndices,
    intervalMs: modeConfig.switchIntervalMs,
    armId,
    pressure,
    repeat,
    events
  });
  const clickProbePromise = runClickResponseProbe({
    loaded,
    experimentId: experiment.id,
    armId,
    modelId: model.id,
    trialOrdinal,
    pressure,
    repeat,
    appPages,
    pressurePage,
    taskEntries,
    events
  });
  const settled = await Promise.all(taskEntries.map((item) => item.promise));
  stageLog(`tasks settled (${armId} ${pressure.id} repeat=${repeat})`);
  const clickResponses = await clickProbePromise.catch((error) => {
    events.push({
      kind: "click-response-probe-error",
      at: new Date().toISOString(),
      arm: armId,
      error: error instanceof Error ? error.message : String(error)
    });
    return [];
  });
  await switchPromise.catch(() => undefined);
  await ensureVisible(pressurePage, "pressure tab");
  await delay(loaded.config.pressure.settleMs);
  await relay.stop();
  const frame = await withTimeout(
    samplePressure(pressurePage),
    10_000,
    "pressure snapshot timed out"
  ).catch(() => null);
  const pressureVisibility = await pressureVisibilityState(pressurePage);
  const pressureValidityError = pressureInvalidReason(
    frame,
    pressureVisibility,
    pressure.triangles,
    loaded.config.pressure.complexity
  );
  const pressureValid = pressureValidityError === null;
  const records = [];
  for (const [index, item] of taskEntries.entries()) {
    const result = settled[index];
    const snapshot = await pageCall(item.entry.page, "snapshot", null, 10_000).catch(() => null);
    const diagnostics = await pageCall(item.entry.page, "diagnostics", null, 10_000).catch(() => null);
    const firstToken = result.progress?.find((event) => event.type === "token") ?? null;
    const highMarkers = (snapshot?.priorityMarkers ?? []).filter(
      (marker) => marker.level === "high"
    );
    const priorityMarker = highMarkers[0] ?? null;
    const markerBeforeToken = firstToken
      ? highMarkers
          .filter((candidate) => candidate.requestedAtMs <= firstToken.receivedAtMs)
          .at(-1) ?? null
      : null;
    const completed = result.error === null && result.stopReason !== "failed";
    const invalidReason = pressureValidityError;
    const trialError = result.error ?? invalidReason;
    const priorityRequested = targets.has(item.entry.tabIndex);
    const priorityRequestedAtWallMs =
      priorityMarker === null
        ? null
        : Number.isFinite(priorityMarker.atMs)
          ? priorityMarker.atMs
          : Number.isFinite(priorityMarker.requestedAtMs)
            ? startWallMs + priorityMarker.requestedAtMs
            : null;
    const priorityTargetActiveAtRequest =
      priorityRequested &&
      priorityMarker !== null &&
      Number.isFinite(result.elapsedMs)
        ? startWallMs + result.elapsedMs > priorityRequestedAtWallMs
        : null;
    const clickSamples = clickResponses.filter(
      (sample) => sample.tabIndex === item.entry.tabIndex
    );
    const validClickSamples = clickSamples.filter(
      (sample) =>
        isValidClickResponse(sample)
    );
    records.push({
      kind: "task",
      experiment: experiment.id,
      mode,
      platform: platformId(),
      model: model.id,
      arm: armId,
      engine: arm.engine,
      nativeWorkerHost:
        diagnostics?.coordinator?.runtime?.nativeWorkerHost ?? null,
      pressureId: pressure.id,
      pressureTriangles: pressure.triangles,
      repeat,
      trialKey: `${model.id}|${platformId()}|${pressure.id}|${repeat}`,
      app: item.entry.app,
      tabIndex: item.entry.tabIndex,
      sessionId: item.entry.sessionId,
      requestId: result.requestId ?? item.requestId,
      taskId: item.entry.task.id,
      role: targets.has(item.entry.tabIndex) ? "priority-target" : "background",
      targetTabIndices: targetIndices,
      targetPosition: targets.has(item.entry.tabIndex)
        ? targetIndices.indexOf(item.entry.tabIndex)
        : null,
      phase: "priority",
      prompt: item.entry.task.prompt,
      rubric: item.entry.task.rubric,
      // The engine-level prompt is what the two arms actually have to
      // evaluate. Recording it (and its hash) makes the "same prompt
      // across arms" claim auditable instead of assumed.
      promptText: typeof result.promptText === "string" ? result.promptText : null,
      promptTextHash:
        typeof result.promptText === "string" ? sha256(result.promptText) : null,
      applicationMessages: Array.isArray(result.applicationMessages)
        ? result.applicationMessages
        : null,
      text: result.text ?? "",
      outputHash: sha256(result.text ?? ""),
      startedAt: startWallMs,
      finishedAt: Date.now(),
      wallMs: result.elapsedMs ?? null,
      ttftMs: result.ttftMs ?? null,
      decodeMs: result.decodeMs ?? null,
      decodeTokens: result.decodeTokens ?? null,
      decodeTokensPerSecond:
        Number.isFinite(result.decodeMs) && result.decodeMs > 0 && Number.isFinite(result.decodeTokens)
          ? (result.decodeTokens / result.decodeMs) * 1000
          : null,
      prefillTokens: result.prefillTokens ?? null,
      cacheTokens: result.cacheTokens ?? null,
      prefillProcessedTokens: result.prefillProcessedTokens ?? null,
      measuredPrefillMsPerToken: result.measuredPrefillMsPerToken ?? null,
      units: result.units ?? null,
      stopReason: result.stopReason ?? null,
      leaseReason: maybeLeaseReason(diagnostics),
      lease: diagnostics?.coordinator?.state?.lease ?? null,
      completed,
      priorityRequested,
      priorityApplied: highMarkers.some(
        (marker) => marker.accepted === true && marker.applied === true
      ),
      priorityRequestedAtMs: priorityMarker?.requestedAtMs ?? null,
      priorityRequestedAtWallMs,
      priorityTargetActiveAtRequest,
      priorityFirstTokenLatencyMs:
        markerBeforeToken && firstToken
          ? firstToken.receivedAtMs - markerBeforeToken.requestedAtMs
          : null,
      clickResponseAttempts: clickSamples.length,
      clickResponseValid: validClickSamples.length,
      clickResponseMs: median(
        validClickSamples.map((sample) => sample.clickResponseMs)
      ),
      clickActivationMs: median(
        validClickSamples.map((sample) => sample.clickActivationMs)
      ),
      clickFrameResponseMs: median(
        validClickSamples.map((sample) => sample.clickFrameResponseMs)
      ),
      clickResponses: clickSamples,
      priorityCompletionMs:
        priorityTargetActiveAtRequest === true && Number.isFinite(result.elapsedMs)
          ? result.elapsedMs
          : null,
      pressureVisibilityState: pressureVisibility,
      pressureVisible: pressureValid,
      invalidReason,
      frame,
      memory: diagnostics?.coordinator?.state ?? null,
      diagnostics,
      progress: Array.isArray(result.progress) ? result.progress : [],
      priorityMarkers: Array.isArray(snapshot?.priorityMarkers)
        ? snapshot.priorityMarkers
        : [],
      pageErrors: item.entry.pageErrors,
      error: trialError
    });
  }
  const backgroundTtftMs = median(
    records
      .filter((record) => record.role === "background")
      .map((record) => record.ttftMs)
  );
  for (const record of records) {
    if (
      record.role !== "priority-target" ||
      !Number.isFinite(record.priorityFirstTokenLatencyMs) ||
      !Number.isFinite(backgroundTtftMs)
    ) {
      record.priorityAdvantageGapMs = null;
      continue;
    }
    record.priorityAdvantageGapMs =
      backgroundTtftMs - record.priorityFirstTokenLatencyMs;
  }
  return { records, events };
  } finally {
    if (relayStarted) await relay.stop();
  }
}

async function runRecoveryTrial(trial) {
  const {
    loaded,
    experiment,
    mode,
    modeConfig,
    pressure,
    repeat,
    armId,
    arm,
    model,
    appPages,
    pressurePage
  } = trial;
  // The first app tab owns the page-side native worker in the wllm arm.
  // Reloading that tab tests worker loss rather than user-session recovery,
  // so both arms select their lifecycle target from the remaining tabs.
  const targetCandidates = appPages.length > 1 ? appPages.slice(1) : appPages;
  const target = targetCandidates[repeat % targetCandidates.length];
  const events = [];
  const relay = new PressureRelay({
    pressurePage,
    enginePages: appPages,
    intervalMs: loaded.config.pressure.pollMs,
    onSample: (sample) => events.push({ kind: "pressure-sample", at: new Date().toISOString(), sample })
  });
  let relayStarted = false;
  try {
  await ensureVisible(pressurePage, "pressure tab before warm-up");
  await relay.start();
  relayStarted = true;
  stageLog(`relay started before warm-up (${armId} ${pressure.id} repeat=${repeat})`);
  await ensureVisible(target.page, "recovery target before warm-up");
  stageLog(`recovery warm-up start (${armId} ${pressure.id} repeat=${repeat})`);
  const warm = await withTimeout(pageCall(target.page, "runTask", {
    phase: "warm",
    id: `${target.task.id}-warm-${repeat}`,
    prompt: target.task.prompt,
    systemPrompt: target.task.systemPrompt,
    maxNewTokens: modeConfig.maxNewTokens,
    temperature: loaded.config.execution.temperature,
    seed: loaded.config.execution.seed
  }), modeConfig.taskTimeoutMs, "recovery warm-up task timed out");
  if (warm.error !== null) throw new Error(`warm-up failed: ${warm.error}`);
  stageLog(`recovery warm-up complete (${armId} ${pressure.id} repeat=${repeat})`);
  const warmHash = sha256(warm.text ?? "");
  const warmMessages = Array.isArray(warm.applicationMessages)
    ? warm.applicationMessages
    : [];
  const recoveryTurnPrompt = String(
    target.task.followUp ?? target.task.prompt
  );
  const continuationMessages =
    warmMessages.length > 0
      ? [
          ...warmMessages,
          { role: "assistant", content: String(warm.text ?? "") },
          { role: "user", content: recoveryTurnPrompt }
        ]
      : [];
  const recoveryHistoryHash = sha256(JSON.stringify(continuationMessages));
  const usesRetainedHistory =
    arm.engine === "wllm" && arm.completedTier !== "rebuild";
  const recoveryPromptMode = usesRetainedHistory
    ? "resident-history-turn-delta"
    : "full-history-replay";

  const measureOutputConsistency =
    loaded.config.recovery.measureOutputConsistency === true;
  let control = null;
  await ensureVisible(pressurePage, "pressure tab");
  const hiddenObserved = await waitForVisibility(target.page, "hidden");
  let hiddenEvidence = hiddenObserved ? "visibilitychange" : null;
  if (!hiddenObserved) {
    const inactive = await confirmInactiveTab(target.page);
    if (inactive.confirmed) {
      await pageCall(target.page, "markLifecycle", {
        type: "hidden",
        detail: {
          source: "inactive-tab-confirmation",
          activeUrl: inactive.activeUrl,
          targetUrl: inactive.targetUrl,
          visibilityState: inactive.visibilityState
        }
      });
      hiddenEvidence = "inactive-tab-confirmation";
    }
  }
  if (hiddenEvidence === null) throw new Error("target did not become hidden before lifecycle test");
  stageLog(`recovery lifecycle hidden confirmed (${hiddenEvidence})`);
  const background = appPages
    .filter((entry) => entry !== target)
    .map((entry) => ({
      entry,
      promise: settleWithTimeout(
        pageCall(entry.page, "runTask", {
          phase: "recovery-background",
          id: `${entry.task.id}-background-${repeat}`,
          prompt: entry.task.prompt,
          systemPrompt: entry.task.systemPrompt,
          maxNewTokens: modeConfig.maxNewTokens,
          temperature: loaded.config.execution.temperature,
          seed: loaded.config.execution.seed + entry.tabIndex
        }),
        modeConfig.taskTimeoutMs
      )
    }));
  stageLog(`recovery background tasks started (${background.length})`);
  const cdp = await target.page.context().newCDPSession(target.page);
  await cdp.send("Page.setWebLifecycleState", { state: "frozen" });
  await pageCall(target.page, "markLifecycle", {
    type: "cdp-frozen",
    detail: { source: "cdp" }
  });
  await delay(loaded.config.recovery.freezeMs);
  await cdp.send("Page.setWebLifecycleState", { state: "active" });
  stageLog("recovery lifecycle thawed");
  await pageCall(target.page, "markLifecycle", {
    type: "cdp-thaw",
    detail: { source: "cdp" }
  });
  await target.page.reload({
    waitUntil: "domcontentloaded",
    timeout: modeConfig.loadTimeoutMs
  });
  stageLog("recovery target reloaded; waiting for adapter readiness");
  await waitForAppReady(target.page, modeConfig.loadTimeoutMs);
  await assertEngineConfiguration(
    target.page,
    arm,
    executionForArm(arm, loaded.config.execution)
  );
  // Reloading a background page does not activate its window. Put the
  // restored application in front before submitting the resumed turn so
  // the measured lifecycle matches the foreground condition being claimed.
  await ensureVisible(target.page, "recovery target after reload");
  stageLog("recovery target ready after reload");
  await pageCall(target.page, "markLifecycle", {
    type: "cdp-active",
    detail: { source: "cdp" }
  });
  await delay(loaded.config.recovery.postFreezeMs);

  const recoveryPriority =
    arm.engine === "wllm" && arm.priorityMode === "enabled"
      ? await setPriority(
          target,
          "high",
          armId,
          `recovery-${pressure.id}-${repeat}`
        )
      : null;
  if (recoveryPriority !== null) events.push(recoveryPriority);

  const recoveryStartWallMs = Date.now();
  stageLog("recovery task submit");
  const recovered = await withTimeout(pageCall(target.page, "runTask", {
    phase: "recovery",
    id: `${target.task.id}-recovery-${repeat}`,
    prompt: recoveryTurnPrompt,
    systemPrompt: target.task.systemPrompt,
    maxNewTokens: modeConfig.maxNewTokens,
    temperature: loaded.config.execution.temperature,
    seed: loaded.config.execution.seed,
    continuationMessages,
    turnDeltaPrompt: usesRetainedHistory ? recoveryTurnPrompt : null
  }), modeConfig.taskTimeoutMs, "recovery task timed out");
  stageLog(`recovery task complete (${armId} ${pressure.id} repeat=${repeat})`);
  const backgroundSettled = await Promise.all(background.map((item) => item.promise));
  stageLog(`recovery background tasks settled (${background.length})`);
  await delay(loaded.config.pressure.settleMs);
  const frame = await withTimeout(
    samplePressure(pressurePage),
    10_000,
    "pressure snapshot timed out"
  ).catch(() => null);
  const pressureVisibility = await pressureVisibilityState(pressurePage);
  const pressureValidityError = pressureInvalidReason(
    frame,
    pressureVisibility,
    pressure.triangles,
    loaded.config.pressure.complexity
  );
  const pressureValid = pressureValidityError === null;
  await relay.stop();
  if (measureOutputConsistency && continuationMessages.length > 0) {
    stageLog(`recovery control probe start (${armId} ${pressure.id} repeat=${repeat})`);
    control = await withTimeout(pageCall(target.page, "runProbe", {
      sessionId: `${target.sessionId}-control-${repeat}`,
      messages: continuationMessages,
      maxNewTokens: modeConfig.maxNewTokens,
      temperature: loaded.config.execution.temperature,
      seed: loaded.config.execution.seed
    }), modeConfig.taskTimeoutMs, "recovery control probe timed out");
    if (control.error !== null) throw new Error(`recovery control failed: ${control.error}`);
    stageLog(`recovery control probe complete (${armId} ${pressure.id} repeat=${repeat})`);
  }
  const targetDiagnostics = await pageCall(target.page, "diagnostics", null).catch(() => null);
  const targetSnapshot = await pageCall(target.page, "snapshot", null).catch(() => null);
  const recoveryFirstToken =
    recovered.progress?.find((event) => event.type === "token") ?? null;
  const recoveryHighMarkers = (targetSnapshot?.priorityMarkers ?? []).filter(
    (marker) => marker.level === "high" && marker.accepted === true && marker.applied === true
  );
  const recoveryMarkerBeforeToken = recoveryFirstToken
    ? recoveryHighMarkers
        .filter(
          (candidate) => candidate.requestedAtMs <= recoveryFirstToken.receivedAtMs
        )
        .at(-1) ?? null
    : null;
  const recoveryPriorityRequested = recoveryPriority !== null;
  const recoveryPriorityApplied = recoveryHighMarkers.length > 0;
  const lifecycleLog = await pageCall(target.page, "getLifecycleLog", null).catch(() => []);
  const lifecycleEvents = lifecycleEventTypes(lifecycleLog);
  const lifecycleValid =
    lifecycleEvents.includes("hidden") &&
    lifecycleEvents.includes("frozen") &&
    lifecycleEvents.includes("reload") &&
    lifecycleEvents.includes("active");
  const continuationAvailable = continuationMessages.length > 0;
  const controlAvailable =
    !measureOutputConsistency ||
    (control !== null &&
      control.error === null &&
      typeof control.text === "string");
  const outputSimilarity =
    measureOutputConsistency && controlAvailable
      ? lexicalConsistency(control.text ?? "", recovered.text ?? "")
      : null;
  const outputConsistent =
    measureOutputConsistency && controlAvailable
      ? normalizeText(control.text ?? "") === normalizeText(recovered.text ?? "") ||
        outputSimilarity >= 0.65
      : null;
  const invalidReasons = [];
  if (!pressureValid) invalidReasons.push(pressureValidityError);
  if (!lifecycleValid) invalidReasons.push(`lifecycle evidence incomplete: ${lifecycleEvents.join(",") || "none"}`);
  if (!continuationAvailable) invalidReasons.push("warm application messages unavailable for recovery history");
  if (!controlAvailable) invalidReasons.push("pre-lifecycle recovery control output unavailable");
  const trialError = recovered.error ?? (invalidReasons.length > 0 ? invalidReasons.join("; ") : null);
  const replayed = replayedTokens(recovered);
  const completed = recovered.error === null && recovered.stopReason !== "failed";
  const records = [{
    kind: "recovery",
    experiment: experiment.id,
    mode,
    platform: platformId(),
    model: model.id,
    arm: armId,
    engine: arm.engine,
    nativeWorkerHost:
      targetDiagnostics?.coordinator?.runtime?.nativeWorkerHost ?? null,
    pressureId: pressure.id,
    pressureTriangles: pressure.triangles,
    repeat,
    trialKey: `${model.id}|${platformId()}|${pressure.id}|${repeat}`,
    app: target.app,
    tabIndex: target.tabIndex,
    sessionId: target.sessionId,
    requestId: recovered.requestId ?? null,
    taskId: target.task.id,
    role: "recovery-target",
    phase: "recovery",
    prompt: recoveryTurnPrompt,
    rubric: target.task.rubric,
    promptText: typeof recovered.promptText === "string" ? recovered.promptText : null,
    promptTextHash:
      typeof recovered.promptText === "string" ? sha256(recovered.promptText) : null,
    applicationMessages: Array.isArray(recovered.applicationMessages)
      ? recovered.applicationMessages
      : null,
    text: recovered.text ?? "",
    warmText: warm.text ?? "",
    warmOutputHash: warmHash,
    measureOutputConsistency,
    controlText: control?.text ?? null,
    controlOutputHash:
      controlAvailable && control !== null ? sha256(control.text ?? "") : null,
    recoveryTurnPrompt,
    recoveryHistoryHash,
    recoveryPromptMode,
    outputHash: sha256(recovered.text ?? ""),
    outputConsistent,
    outputSimilarity,
    lifecycleEvents,
    hiddenEvidence,
    lifecycleLog,
    lifecycleValid,
    pressureVisibilityState: pressureVisibility,
    pressureVisible: pressureValid,
    startedAt: recoveryStartWallMs,
    finishedAt: Date.now(),
    recoveryStartedAt: recoveryStartWallMs,
    recoveryFirstTokenLatencyMs: recovered.ttftMs ?? null,
    recoveryCompletionMs: recovered.elapsedMs ?? null,
    replayedTokens: replayed.value,
    replayedTokenBasis: replayed.basis,
    ttftMs: recovered.ttftMs ?? null,
    wallMs: recovered.elapsedMs ?? null,
    decodeMs: recovered.decodeMs ?? null,
    decodeTokens: recovered.decodeTokens ?? null,
    decodeTokensPerSecond:
      Number.isFinite(recovered.decodeMs) && recovered.decodeMs > 0 && Number.isFinite(recovered.decodeTokens)
        ? (recovered.decodeTokens / recovered.decodeMs) * 1000
        : null,
    measuredPrefillMsPerToken: recovered.measuredPrefillMsPerToken ?? null,
    prefillTokens: recovered.prefillTokens ?? null,
    cacheTokens: recovered.cacheTokens ?? null,
    prefillProcessedTokens: recovered.prefillProcessedTokens ?? null,
    units: recovered.units ?? null,
    stopReason: recovered.stopReason ?? null,
    leaseReason: maybeLeaseReason(targetDiagnostics),
    lease: targetDiagnostics?.coordinator?.state?.lease ?? null,
    completed,
    priorityRequested: recoveryPriorityRequested,
    priorityApplied: recoveryPriorityApplied,
    priorityMarker: recoveryPriority,
    priorityMarkers: recoveryHighMarkers,
    priorityFirstTokenLatencyMs:
      recoveryMarkerBeforeToken && recoveryFirstToken
        ? recoveryFirstToken.receivedAtMs - recoveryMarkerBeforeToken.requestedAtMs
        : null,
    priorityCompletionMs:
      recoveryPriorityRequested && Number.isFinite(recovered.elapsedMs)
        ? recovered.elapsedMs
        : null,
    frame,
    memory: targetDiagnostics?.coordinator?.state ?? null,
    diagnostics: targetDiagnostics,
    pageErrors: target.pageErrors,
    invalidReason: invalidReasons.length > 0 ? invalidReasons.join("; ") : null,
    error: trialError
  }];
  for (const [index, item] of background.entries()) {
    const result = backgroundSettled[index];
    const diagnostics = await pageCall(item.entry.page, "diagnostics", null).catch(() => null);
    records.push({
      kind: "task",
      experiment: experiment.id,
      mode,
      platform: platformId(),
      model: model.id,
      arm: armId,
      engine: arm.engine,
      nativeWorkerHost:
        diagnostics?.coordinator?.runtime?.nativeWorkerHost ?? null,
      pressureId: pressure.id,
      pressureTriangles: pressure.triangles,
      repeat,
      trialKey: `${model.id}|${platformId()}|${pressure.id}|${repeat}`,
      app: item.entry.app,
      tabIndex: item.entry.tabIndex,
      sessionId: item.entry.sessionId,
      taskId: item.entry.task.id,
      role: "background",
      phase: "recovery-background",
      prompt: item.entry.task.prompt,
      rubric: item.entry.task.rubric,
      text: result.text ?? "",
      outputHash: sha256(result.text ?? ""),
      completed: result.error === null && result.stopReason !== "failed",
      wallMs: result.elapsedMs ?? null,
      ttftMs: result.ttftMs ?? null,
      decodeTokens: result.decodeTokens ?? null,
      prefillTokens: result.prefillTokens ?? null,
      cacheTokens: result.cacheTokens ?? null,
      prefillProcessedTokens: result.prefillProcessedTokens ?? null,
      measuredPrefillMsPerToken: result.measuredPrefillMsPerToken ?? null,
      units: result.units ?? null,
      leaseReason: maybeLeaseReason(diagnostics),
      frame,
      memory: diagnostics?.coordinator?.state ?? null,
      diagnostics,
      pageErrors: item.entry.pageErrors,
      error: result.error ?? pressureValidityError
    });
  }
  return { records, events };
  } finally {
    if (relayStarted) await relay.stop();
  }
}

async function switchPriorityTargets(options) {
  const { appPages, targetIndices, intervalMs, armId, pressure, repeat, events } = options;
  let previous = targetIndices[0];
  for (const index of targetIndices.slice(1)) {
    await delay(intervalMs);
    if (appPages[previous]) {
      const normal = await setPriority(appPages[previous], "normal", armId, `priority-${pressure.id}-${repeat}`);
      events.push(normal);
    }
    const target = appPages[index];
    if (target) {
      const high = await setPriority(target, "high", armId, `priority-${pressure.id}-${repeat}`);
      events.push(high);
    }
    previous = index;
  }
  // The last rotated target keeps `high` until the trial ends: the measured
  // window must always contain exactly one prioritized session, otherwise the
  // render-pressure lease has no owner and every session degrades to the
  // background fallback policy.
}

async function runClickResponseProbe(options) {
  const {
    loaded,
    experimentId,
    armId,
    modelId,
    trialOrdinal,
    pressure,
    repeat,
    appPages,
    pressurePage,
    taskEntries,
    events
  } = options;
  const config = loaded.config.clickResponse;
  if (
    !config ||
    !Array.isArray(config.enabledForExperiments) ||
    !config.enabledForExperiments.includes(experimentId)
  ) {
    return [];
  }
  const sequencer = createClickSequencer({
    indices: appPages.map((entry) => entry.tabIndex),
    seed: clickResponseSeed(config.seed, trialOrdinal, modelId),
    initialDelayMs: config.initialDelayMs,
    minIntervalMs: config.intervalMinMs,
    maxIntervalMs: config.intervalMaxMs,
    maxClicks: config.maxClicksPerTrial
  });
  await delayWhileTasksRun(taskEntries, sequencer.initialDelayMs);
  const samples = [];
  while (true) {
    const activeIndices = taskEntries
      .filter((item) => item.finished !== true)
      .map((item) => item.entry.tabIndex);
    const tabIndex = sequencer.next(activeIndices);
    if (tabIndex === null) break;
    const entry = appPages.find((candidate) => candidate.tabIndex === tabIndex);
    if (!entry) break;
    const sample = await measureClickResponse({
      entry,
      pressurePage,
      config,
      armId,
      pressure,
      repeat
    });
    samples.push(sample);
    events.push({
      kind: "click-response",
      at: new Date(sample.requestedAtMs).toISOString(),
      ...sample
    });
    if (sample.error !== null) break;
    await delayWhileTasksRun(taskEntries, sequencer.intervalMs());
  }
  return samples;
}

async function measureClickResponse(options) {
  const { entry, pressurePage, config, armId, pressure, repeat } = options;
  let requestedAtMs = Date.now();
  let activatedAtMs = null;
  let responseAtMs = null;
  let visibilityBefore = null;
  let visibilityState = null;
  let pressureVisibilityBefore = null;
  let pressureVisibilityAfterClick = null;
  let framesWaited = 0;
  let pressureRestored = false;
  let error = null;
  try {
    [visibilityBefore, pressureVisibilityBefore] = await Promise.all([
      withTimeout(
        entry.page.evaluate(() => document.visibilityState),
        config.clickTimeoutMs,
        `clicked tab ${entry.tabIndex} visibility query timed out`
      ),
      pressureVisibilityState(pressurePage)
    ]);
    requestedAtMs = Date.now();
    await withTimeout(
      entry.page.bringToFront(),
      config.clickTimeoutMs,
      `click activation timed out for tab ${entry.tabIndex}`
    );
    activatedAtMs = Date.now();
    const frames = await withTimeout(
      waitForRenderedFrames(entry.page, config.responseFrames),
      config.frameTimeoutMs,
      `clicked tab ${entry.tabIndex} produced no rendered frame`
    );
    framesWaited = frames.frames;
    visibilityState = frames.visibilityState;
    if (visibilityState !== "visible") {
      throw new Error(`clicked tab visibility was ${visibilityState}`);
    }
    responseAtMs = Date.now();
    pressureVisibilityAfterClick = await pressureVisibilityState(pressurePage);
  } catch (clickError) {
    error = clickError instanceof Error ? clickError.message : String(clickError);
  } finally {
    if (config.restorePressureAfterClick !== false) {
      try {
        await ensureVisible(pressurePage, "pressure tab");
        pressureRestored = true;
      } catch (restoreError) {
        pressureRestored = false;
        error ??= restoreError instanceof Error
          ? restoreError.message
          : String(restoreError);
      }
    } else {
      pressureRestored = true;
    }
  }
  return {
    arm: armId,
    pressureId: pressure.id,
    pressureTriangles: pressure.triangles,
    repeat,
    app: entry.app,
    tabIndex: entry.tabIndex,
    taskId: entry.task.id,
    sessionId: entry.sessionId,
    requestedAtMs,
    activatedAtMs,
    responseAtMs,
    visibilityBefore,
    pressureVisibilityBefore,
    pressureVisibilityAfterClick,
    clickActivationMs:
      activatedAtMs === null ? null : activatedAtMs - requestedAtMs,
    clickResponseMs:
      responseAtMs === null ? null : responseAtMs - requestedAtMs,
    clickFrameResponseMs:
      activatedAtMs === null || responseAtMs === null
        ? null
        : responseAtMs - activatedAtMs,
    visibilityState,
    visibilityChanged:
      visibilityBefore === "hidden" && visibilityState === "visible",
    responseFrames: config.responseFrames,
    framesWaited,
    pressureRestored,
    affectsPriority: config.affectsPriority === true,
    error
  };
}

function waitForRenderedFrames(page, frameCount) {
  return page.evaluate((target) => new Promise((resolve) => {
    let frames = 0;
    const tick = () => {
      frames += 1;
      if (frames >= target) {
        resolve({ frames, visibilityState: document.visibilityState });
        return;
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }), frameCount);
}

async function delayWhileTasksRun(taskEntries, durationMs) {
  const deadline = Date.now() + Math.max(0, durationMs);
  while (Date.now() < deadline) {
    if (taskEntries.every((item) => item.finished === true)) return;
    await delay(Math.min(100, deadline - Date.now()));
  }
}

async function setPriority(entry, level, armId, phase) {
  const result = await pageCall(entry.page, "setPriority", level, 8_000).catch((error) => ({
    accepted: false,
    applied: false,
    error: error instanceof Error ? error.message : String(error)
  }));
  return {
    kind: "priority",
    at: new Date().toISOString(),
    phase,
    arm: armId,
    app: entry.app,
    tabIndex: entry.tabIndex,
    level,
    ...result
  };
}

async function runJudge(options) {
  const { loaded, experiment, model, browser, serverUrl, records } = options;
  const judgeModelId = loaded.config.judge.model;
  const judgeModel = modelById(loaded.models, judgeModelId);
  await assertModelReady(judgeModel);
  const context = await withTimeout(
    browser.newContext({
      viewport: { width: 1440, height: 980 },
      deviceScaleFactor: 1
    }),
    60_000,
    "judge browser.newContext timed out"
  );
  const runner = await createJudgeRunner({
    context,
    baseUrl: serverUrl,
    modelUrl: modelUrlFor(judgeModel),
    config: loaded.config.judge,
    timeoutMs: 300_000
  });
  try {
    return await scoreRecords(runner, records, {
      sampleFraction: loaded.config.judge.sampleFraction,
      onProgress: (current, total, record) => {
        console.log(`judge ${current}/${total} ${record.arm} ${record.taskId}`);
      }
    });
  } finally {
    await runner.close();
    await context.close().catch(() => undefined);
  }
}

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

async function pageCall(page, method, argument, timeoutMs = null) {
  const invocation = page.evaluate(
    async ({ methodName, value }) => {
      const target = window.__wllmE2E;
      if (!target || typeof target[methodName] !== "function") {
        throw new Error(`adapter method ${methodName} is unavailable`);
      }
      return target[methodName](value);
    },
    { methodName: method, value: argument }
  );
  if (timeoutMs === null) return invocation;
  return withTimeout(invocation, timeoutMs, `adapter method ${method} timed out`);
}

async function assertEngineConfiguration(page, arm, execution) {
  const expected = {
    modelLoadMode: modelLoadModeForArm(arm),
    nCtx: execution.nCtx,
    nParallel: execution.nParallel,
    queueWaitTimeoutMs: execution.queueWaitTimeoutMs,
    webgpuCommandSubmitBatchSize: execution.webgpuCommandSubmitBatchSize,
    webgpuMaxInflightBatches: execution.webgpuMaxInflightBatches,
    ...(arm.engine === "wllm"
      ? {
          leaseWorkConservingBarrier:
            execution.leaseWorkConservingBarrier,
          leaseBackgroundBarrier: execution.leaseBackgroundBarrier,
          leaseWorkConservingQuantumMs:
            execution.leaseWorkConservingQuantumMs,
          schedulerForegroundBurstUnits:
            execution.schedulerForegroundBurstUnits,
          schedulerResidentBurstUnits:
            execution.schedulerResidentBurstUnits,
          schedulerRetentionWindowMs:
            execution.schedulerRetentionWindowMs
        }
      : {})
  };
  const observed = await withTimeout(
    page.evaluate((keys) => {
      const frame = document.querySelector("#wllm-e2e-engine");
      if (!(frame instanceof HTMLIFrameElement)) return null;
      const url = new URL(frame.src);
      return Object.fromEntries(
        keys.map((key) => [key, url.searchParams.get(key)])
      );
    }, Object.keys(expected)),
    10_000,
    "engine configuration audit timed out"
  );
  if (observed === null) {
    throw new Error("evaluation engine iframe is unavailable");
  }
  const mismatches = [];
  for (const [key, value] of Object.entries(expected)) {
    if (value === undefined) continue;
    const normalized =
      typeof value === "boolean" ? (value ? "1" : "0") : String(value);
    if (observed[key] !== normalized) {
      mismatches.push(
        `${key}: expected ${normalized}, got ${observed[key] ?? "missing"}`
      );
    }
  }
  if (mismatches.length > 0) {
    throw new Error(
      `engine configuration was not forwarded: ${mismatches.join("; ")}`
    );
  }
  return observed;
}

async function ensureVisible(page, label) {
  await withTimeout(page.bringToFront(), 30_000, `${label} bringToFront timed out`);
  const visible = await waitForVisibility(page, "visible");
  if (!visible) throw new Error(`${label} did not become visible`);
}

async function confirmInactiveTab(page) {
  if (process.platform !== "darwin") {
    return { confirmed: false, reason: "tab-activity-probe-unavailable" };
  }
  try {
    const activeUrl = execFileSync(
      "osascript",
      ["-e", 'tell application "Google Chrome" to get URL of active tab of front window'],
      { encoding: "utf8", timeout: 8_000 }
    ).trim();
    const targetUrl = page.url();
    const visibilityState = await page.evaluate(() => document.visibilityState);
    return {
      confirmed:
        activeUrl.length > 0 && canonicalUrl(activeUrl) !== canonicalUrl(targetUrl),
      activeUrl,
      targetUrl,
      visibilityState,
      reason: "front-window-active-tab"
    };
  } catch (error) {
    return {
      confirmed: false,
      reason: error instanceof Error ? error.message : String(error)
    };
  }
}

function canonicalUrl(value) {
  try {
    const parsed = new URL(value);
    parsed.hash = "";
    return parsed.toString();
  } catch {
    return value;
  }
}

async function waitForVisibility(page, expected, timeoutMs = 30_000) {
  return page
    .waitForFunction(
      (value) => document.visibilityState === value,
      expected,
      { timeout: timeoutMs }
    )
    .then(() => true)
    .catch(() => false);
}

async function pressureVisibilityState(page) {
  return withTimeout(
    page.evaluate(() => document.visibilityState),
    8_000,
    "pressure visibility query timed out"
  ).catch(() => null);
}

function pressureFrameValid(frame, triangles, complexity) {
  const expectedInstances = Math.max(1, Math.floor(triangles * complexity));
  return (
    frame !== null &&
    frame.mode === "webgpu" &&
    frame.error === null &&
    frame.diagnosticPattern === "visible-color-triangles-v1" &&
    frame.baseTriangleCount === triangles &&
    frame.instanceCopiesPerTriangle === complexity &&
    frame.targetTriangles === triangles &&
    frame.complexity === complexity &&
    frame.drawsPerFrame === triangles * complexity &&
    frame.submittedInstancesPerFrame === expectedInstances &&
    frame.frameCount > 0
  );
}

function pressureInvalidReason(frame, visibility, triangles, complexity) {
  if (visibility !== "visible") {
    return `pressure tab visibility was ${visibility ?? "unknown"}`;
  }
  if (!pressureFrameValid(frame, triangles, complexity)) {
    return [
      "pressure visual frame was invalid",
      `mode=${frame?.mode ?? "missing"}`,
      `pattern=${frame?.diagnosticPattern ?? "missing"}`,
      `triangles=${frame?.targetTriangles ?? "missing"}`,
      `instances=${frame?.submittedInstancesPerFrame ?? "missing"}`,
      `frames=${frame?.frameCount ?? "missing"}`
    ].join(" ");
  }
  return null;
}
function maybeLeaseReason(diagnostics) {
  return (
    diagnostics?.coordinator?.state?.lease?.reason ??
    diagnostics?.metrics?.lease?.reason ??
    diagnostics?.status?.lease?.reason ??
    null
  );
}

function lifecycleEventTypes(log) {
  const entries = Array.isArray(log) ? log : [];
  const types = entries.map((entry) => String(entry?.type ?? ""));
  const normalized = [];
  if (types.includes("hidden")) normalized.push("hidden");
  if (types.includes("freeze") || types.includes("cdp-frozen")) {
    normalized.push("frozen");
  }
  if (types.filter((type) => type === "load").length >= 2 || types.includes("reload")) {
    normalized.push("reload");
  }
  if (types.includes("resume") || types.includes("cdp-active")) {
    normalized.push("active");
  }
  return normalized;
}

function replayedTokens(result) {
  const processed = Number(result?.prefillProcessedTokens);
  const cache = Number(result?.cacheTokens);
  const prefill = Number(result?.prefillTokens);
  const accounting = {
    prefillTokens: Number.isFinite(prefill) ? prefill : null,
    cacheTokens: Number.isFinite(cache) ? cache : null,
    prefillProcessedTokens: Number.isFinite(processed) ? processed : null
  };
  if (Number.isFinite(processed) && Number.isFinite(cache)) {
    return {
      value: Math.max(0, processed),
      basis: "nativePromptN",
      ...accounting
    };
  }
  if (Number.isFinite(processed)) {
    return {
      value: Math.max(0, processed),
      basis: "prefillProcessedTokens",
      ...accounting
    };
  }
  if (Number.isFinite(prefill)) {
    return {
      value: Math.max(0, prefill),
      basis: "prefillTokens",
      ...accounting
    };
  }
  return { value: null, basis: "unavailable", ...accounting };
}

function settleWithTimeout(promise, timeoutMs) {
  let timer = null;
  const guarded = Promise.resolve(promise).catch((error) => failureResult(error));
  const timeout = new Promise((resolve) => {
    timer = setTimeout(
      () => resolve(failureResult(new Error(`task timed out after ${timeoutMs} ms`))),
      timeoutMs
    );
  });
  return Promise.race([guarded, timeout]).finally(() => {
    if (timer !== null) clearTimeout(timer);
  });
}

function failureResult(error) {
  return {
    requestId: null,
    text: "",
    elapsedMs: null,
    ttftMs: null,
    decodeMs: null,
    decodeTokens: null,
    prefillTokens: null,
    cacheTokens: null,
    prefillProcessedTokens: null,
    measuredPrefillMsPerToken: null,
    units: null,
    stopReason: "failed",
    progress: [],
    error: error instanceof Error ? error.message : String(error)
  };
}

function appDefinitionsFor(layout, config) {
  const definitions = [];
  for (let index = 0; index < layout.appTabs; index += 1) {
    const app = index < layout.artifexTabs ? "artifex" : "tavern";
    const appConfig = config.apps[app];
    definitions.push({
      app,
      task: appConfig.tasks[index % appConfig.tasks.length]
    });
  }
  return definitions;
}

function modelLoadModeForArm(arm) {
  // wllama's standard deployment uses its persistent model cache, which also
  // lets multiple tabs open the same downloaded GGUF. The shared coordinator
  // needs only one model load per trial, so it keeps the direct path.
  return arm.engine === "wllama" ? "cache" : "direct";
}

function executionForArm(arm, execution) {
  if (arm.engine !== "wllama") return execution;
  // The shared runtime divides one context into parallel logical sessions.
  // Each per-tab wllama engine owns exactly one of those session slots.
  const sharedParallel = Math.max(1, Math.floor(execution.nParallel ?? 1));
  return {
    ...execution,
    nCtx: Math.max(1, Math.floor(execution.nCtx / sharedParallel)),
    nParallel: 1
  };
}

function appUrl(baseUrl, definition, options) {
  const entry = definition.app === "artifex"
    ? "/apps/artifex/game.html"
    : "/apps/tavern/index.html";
  const url = new URL(entry, baseUrl);
  url.searchParams.set("app", definition.app);
  if (definition.app === "artifex") {
    url.searchParams.set("director", "off");
  }
  url.searchParams.set("arm", options.armId);
  url.searchParams.set("engine", options.arm.engine);
  url.searchParams.set("enginePage", options.arm.enginePage);
  url.searchParams.set("sessionId", options.sessionId);
  url.searchParams.set("modelUrl", modelUrlFor(options.model));
  url.searchParams.set("modelLoadMode", modelLoadModeForArm(options.arm));
  url.searchParams.set("promptFormat", options.model.promptFormat);
  url.searchParams.set("priorityMode", options.arm.priorityMode ?? "enabled");
  url.searchParams.set("initialPriority", options.arm.initialPriority ?? "normal");
  url.searchParams.set("executionMode", options.arm.executionMode ?? "deadline");
  url.searchParams.set("completedTier", options.arm.completedTier ?? "L2");
  url.searchParams.set("parkCompleted", options.arm.parkCompleted === false ? "0" : "1");
  url.searchParams.set(
    "preemptOnActivation",
    options.execution.preemptOnActivation === false ? "0" : "1"
  );
  url.searchParams.set("maxNewTokens", String(options.modeConfig.maxNewTokens));
  url.searchParams.set("temperature", String(options.execution.temperature ?? 0));
  url.searchParams.set("seed", String(options.execution.seed ?? 20261015));
  url.searchParams.set("nCtx", String(options.execution.nCtx));
  url.searchParams.set("nBatch", String(options.execution.nBatch));
  url.searchParams.set("nParallel", String(options.execution.nParallel));
  for (const key of [
    "queueWaitTimeoutMs",
    "webgpuCommandSubmitBatchSize",
    "webgpuMaxInflightBatches",
  ]) {
    const value = options.execution[key];
    if (typeof value === "number" && Number.isFinite(value)) {
      url.searchParams.set(key, String(value));
    }
  }
  for (const key of [
    "leaseWorkConservingBarrier",
    "leaseBackgroundBarrier",
  ]) {
    const value = options.execution[key];
    if (typeof value === "boolean") {
      url.searchParams.set(key, value ? "1" : "0");
    }
  }
  url.searchParams.set(
    "leaseBatchSize",
    String(options.execution.leaseBatchSize ?? 2)
  );
  const staleSignalShape = options.execution.leaseStaleSignalShape;
  if (staleSignalShape === "barrier" || staleSignalShape === "bounded") {
    url.searchParams.set("leaseStaleSignalShape", staleSignalShape);
  }
  url.searchParams.set(
    "leaseWorkConservingQuantumMs",
    String(options.execution.leaseWorkConservingQuantumMs ?? 1000)
  );
  for (const key of [
    "schedulerForegroundBurstUnits",
    "schedulerResidentBurstUnits",
    "schedulerRetentionWindowMs",
  ]) {
    const value = options.execution[key];
    if (typeof value === "number" && Number.isFinite(value)) {
      url.searchParams.set(key, String(value));
    }
  }
  url.searchParams.set(
    "batchHandoff",
    options.execution.batchHandoff === true ? "1" : "0"
  );
  url.searchParams.set(
    "handoffBatchSize",
    String(options.execution.handoffBatchSize ?? 2)
  );
  url.searchParams.set(
    "expressHandoff",
    options.execution.expressHandoff === true ? "1" : "0"
  );
  url.searchParams.set(
    "handoffPrefillTokens",
    String(options.execution.handoffPrefillTokens ?? 64)
  );
  const l1NodeCap = Math.max(1, Math.floor(options.execution.l1NodeCap ?? 1));
  const residentSlots = Math.max(
    1,
    Math.min(l1NodeCap, Math.floor(options.execution.nParallel ?? 1))
  );
  const l1TokenCap =
    options.execution.l1TokenCap ??
    (options.execution.l1TokenCapPerSession ?? 4096) * residentSlots;
  const l2TokenCap =
    options.execution.l2TokenCap ??
    (options.execution.l2TokenCapPerSession ?? 4096) * residentSlots;
  url.searchParams.set("l1NodeCap", String(residentSlots));
  url.searchParams.set("l1TokenCap", String(l1TokenCap));
  url.searchParams.set("l2TokenCap", String(l2TokenCap));
  url.searchParams.set("l3ByteCap", String(options.execution.l3ByteCap));
  url.searchParams.set("repairBudget", String(options.execution.repairBudget));
  url.searchParams.set("timeoutMs", String(options.modeConfig.taskTimeoutMs));
  return url.toString();
}

function resolveRunDirectory(experiment, model, startedAt) {
  const root = path.join(implRoot, experiment.artifactRoot);
  if (experiment.platformScope === "ablation") {
    return path.join(root, model.id, timestampId(startedAt));
  }
  return path.join(root, platformId(), model.id, timestampId(startedAt));
}

function expectedRecordCount(loaded, experimentId, mode, selectedModels, layout) {
  const experiment = experimentFor(loaded, experimentId);
  return selectedModels.length
    * experiment.arms.length
    * effectiveMode(loaded, mode).repeats
    * pressureSchedule(loaded, mode).length
    * layout.appTabs;
}


function selectModels(loaded, experimentId, mode, cli) {
  const available = modelsForExperiment(loaded, experimentId);
  if (cli.model) {
    const selected = available.find((model) => model.id === cli.model);
    if (!selected) throw new Error(`model ${cli.model} is not in ${experimentId}'s model set`);
    return [selected];
  }
  if (mode === "smoke" || mode === "pilot") {
    return [available[0]];
  }
  return cli.allModels || mode === "formal" ? available : [available[0]];
}

async function assertModelReady(model) {
  const file = path.join(implRoot, "public", "models", model.filename);
  try {
    await access(file);
  } catch {
    throw new Error(
      `model ${model.id} is missing at ${file}. Run "node evaluation/scripts/prepare-models.mjs --model ${model.id}".`
    );
  }
}

async function assertExperimentRunnable(loaded, experimentId, mode) {
  if (!platformSupportsExperiment(loaded, experimentId)) {
    const platform = loaded.config.platforms[platformId()]?.label ?? platformId();
    throw new Error(`${experimentId} is not configured to run on ${platform}`);
  }
  if (!loaded.config.modes[mode]) throw new Error(`unknown mode: ${mode}`);
}

function sha256(value) {
  return createHash("sha256").update(String(value)).digest("hex");
}

function normalizeText(value) {
  return String(value).replace(/\s+/g, " ").trim();
}

function semanticTokens(value) {
  return (normalizeText(value)
    .toLowerCase()
    .replace(/\*[^*]*\*/g, " ")
    .match(/[a-z0-9']+/g) ?? []);
}

function lexicalConsistency(left, right) {
  const a = semanticTokens(left);
  const b = semanticTokens(right);
  if (a.length === 0 || b.length === 0) return 0;
  let previous = new Array(b.length + 1).fill(0);
  for (const leftToken of a) {
    const current = new Array(b.length + 1).fill(0);
    for (let index = 0; index < b.length; index += 1) {
      current[index + 1] = leftToken === b[index]
        ? previous[index] + 1
        : Math.max(previous[index + 1], current[index]);
    }
    previous = current;
  }
  return (2 * previous[b.length]) / (a.length + b.length);
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function parseArgs(argv) {
  const result = {
    experiment: null,
    mode: null,
    model: null,
    out: null,
    judge: false,
    headless: false,
    allModels: false
  };
  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index];
    if (value === "--experiment") result.experiment = argv[++index];
    else if (value === "--mode") result.mode = argv[++index];
    else if (value === "--model") result.model = argv[++index];
    else if (value === "--out") result.out = argv[++index];
    else if (value === "--no-judge") result.judge = false;
    else if (value === "--judge") result.judge = true;
    else if (value === "--headless") result.headless = true;
    else if (value === "--headed") result.headless = false;
    else if (value === "--all-models") result.allModels = true;
  }
  return result;
}

// All artifacts and summaries are flushed before this point. Playwright can
// retain a dead browser transport after Chrome exits, so terminate the CLI
// explicitly instead of letting that handle leak block the next experiment.
if (process.env.WLLM_E2E_KEEP_ALIVE !== "1") {
  await new Promise((resolve) => setTimeout(resolve, 50));
  process.exit(process.exitCode ?? 0);
}
