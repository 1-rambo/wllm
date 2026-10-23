#!/usr/bin/env node
import { spawn } from "node:child_process";
import { createWriteStream } from "node:fs";
import {
  access,
  mkdir,
  readFile,
  readdir,
  rename,
  rm,
  statfs,
  writeFile
} from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import readline from "node:readline/promises";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const evaluationRoot = path.resolve(here, "..");
const implRoot = path.resolve(evaluationRoot, "..");
const handoffRoot = path.join(implRoot, "handoff", "windows");
const modelSets = {
  small: ["llama-3.2-1b-instruct", "qwen2.5-1.5b-instruct"],
  all: [
    "llama-3.2-1b-instruct",
    "llama-3.2-3b-instruct",
    "qwen2.5-1.5b-instruct",
    "qwen2.5-3b-instruct"
  ]
};
const e2eConfig = JSON.parse(
  await readFile(path.join(evaluationRoot, "config", "e2e.json"), "utf8")
);
const expectedFormalRecordCount =
  2 *
  e2eConfig.modes.formal.repeats *
  e2eConfig.pressure.schedule.length *
  e2eConfig.modes.formal.appTabs;
const expectedSmokeRecordCount =
  2 *
  e2eConfig.modes.smoke.repeats *
  e2eConfig.pressure.smokeSchedule.length *
  e2eConfig.modes.smoke.appTabs;

const args = parseArgs(process.argv.slice(2));
if (args.help) {
  printHelp();
  process.exit(0);
}

if (args.dryRun) {
  printDryRun();
  process.exit(0);
}

if (args.packageOnly) {
  const session = await loadSessionForPackage();
  const packagePath = await finalizeSession(session.state, session.directory);
  process.exitCode = packagePath || args.noPackage ? 0 : 1;
  process.exit();
}

if (process.platform !== "win32") {
  throw new Error(
    `This runner only supports native Windows. Current platform: ${process.platform}. ` +
      "Use the normal evaluation commands on macOS."
  );
}

const requestedModelIds = modelSets[args.modelSet];
if (!requestedModelIds) {
  throw new Error(`Unknown model set: ${args.modelSet}`);
}
const requestedPreparedModels = requestedModelIds;
const session = await createOrResumeSession(
  requestedModelIds,
  requestedPreparedModels
);
const persistedModelSet = modelSets[session.state.modelSet];
if (!persistedModelSet) {
  throw new Error(`Unknown persisted model set: ${session.state.modelSet}`);
}
const modelIds = validatePersistedModels(
  session.state.models,
  persistedModelSet
);
const preparedModels = validatePersistedModels(session.state.preparedModels);
let exitCode = 0;

try {
  await prepareEnvironment(session);
  await runPipeline(session, modelIds, preparedModels);
} catch (error) {
  exitCode = 1;
  session.state.overallStatus = "failed";
  session.state.failure = error instanceof Error
    ? { message: error.message, stack: error.stack ?? null }
    : { message: String(error) };
  console.error("");
  console.error(`RUN STOPPED: ${session.state.failure.message}`);
} finally {
  if (session.state.overallStatus === "running") {
    const currentStages = latestStages(session.state.stages);
    const unfinishedStages = currentStages.filter(
      (stage) => stage.status !== "completed"
    );
    const formalIssues = currentStages.filter(
      (stage) =>
        stage.mode === "formal" &&
        ["completed-with-errors", "incomplete"].includes(stage.resultStatus)
    );
    session.state.overallStatus =
      unfinishedStages.length > 0
        ? "partial"
        : formalIssues.length > 0 || session.state.formalCommandFailures > 0
          ? "completed-with-errors"
          : "completed";
  }
  session.state.completedAt = new Date().toISOString();
  const packagePath = await finalizeSession(session.state, session.directory);
  if (!packagePath && !args.noPackage) {
    if (exitCode === 0) exitCode = 1;
  }
  console.log("");
  console.log(`SESSION: ${session.directory}`);
  if (packagePath) {
    console.log(`SEND THIS FILE: ${packagePath}`);
  } else {
    console.log("Result packaging failed. Send the entire session directory.");
  }
  if (!args.noOpen && process.platform === "win32") {
    openExplorer(packagePath ?? session.directory);
  }
}

process.exitCode = exitCode;
await new Promise((resolve) => setTimeout(resolve, 100));

async function prepareEnvironment(session) {
  console.log("");
  console.log("Checking Windows environment...");
  const environment = await collectEnvironment();
  if (
    session.state.runnerCommit &&
    environment.implCommit !== session.state.runnerCommit
  ) {
    throw new Error(
      `Implementation commit changed during resume: expected ${session.state.runnerCommit}, found ${environment.implCommit ?? "unknown"}. ` +
        "Check out the original handoff tag/commit and resume again."
    );
  }
  session.state.runnerCommit ??= environment.implCommit;
  session.state.environment = environment;
  await writeJson(path.join(session.directory, "environment.json"), environment);
  await saveState(session);

  const errors = [];
  if (environment.nodeMajor < 20) {
    errors.push(`Node.js 20 or newer is required. Found ${environment.node}.`);
  }
  if (!environment.git) errors.push("Git is not available on PATH.");
  if (!environment.implCommit) {
    errors.push("The runner is not inside a Git checkout, so the implementation commit cannot be recorded.");
  }
  if (!environment.chromePath) {
    errors.push(
      "Google Chrome was not found. Install Chrome or set CHROME_PATH before running."
    );
  }
  if (environment.memoryGiB < 15) {
    errors.push(`At least 16 GiB RAM is required. Found ${environment.memoryGiB.toFixed(1)} GiB.`);
  }
  if (environment.freeDiskGiB < 20) {
    errors.push(
      `At least 20 GiB free disk is required. Found ${environment.freeDiskGiB.toFixed(1)} GiB.`
    );
  }
  if (errors.length > 0) {
    throw new Error(errors.join("\n"));
  }
  process.env.CHROME_PATH = environment.chromePath;

  console.log(`  Host: ${environment.hostname}`);
  console.log(`  Node: ${environment.node}`);
  console.log(`  Chrome: ${environment.chromeVersion ?? "unknown"}`);
  console.log(`  GPU: ${environment.gpus.map((gpu) => gpu.name).join("; ") || "unknown"}`);
  console.log(`  Memory: ${environment.memoryGiB.toFixed(1)} GiB`);
  console.log(`  Free disk: ${environment.freeDiskGiB.toFixed(1)} GiB`);
  if (environment.onAcPower === false) {
    console.log("  WARNING: the machine appears to be on battery power.");
  }
  if (environment.implWorktreeClean === false) {
    console.log("  WARNING: the implementation worktree contains local changes.");
  }
  console.log("");
  console.log("Before the run:");
  console.log("  - Keep the computer on AC power.");
  console.log("  - Disable sleep and keep the screen unlocked.");
  console.log("  - Close games, video calls, and other GPU-heavy applications.");
  console.log("  - Do not minimize Chrome while the experiment is running.");
  console.log("  - Keep only one wLLM runner window open.");
  console.log("");
  if (!args.yes && process.stdin.isTTY) {
    const prompt = readline.createInterface({
      input: process.stdin,
      output: process.stdout
    });
    await prompt.question("Press Enter to start, or close this window to cancel. ");
    prompt.close();
  }
}

async function runPipeline(session, selectedModels, modelsToPrepare) {
  await runLoggedStage(session, {
    id: "bootstrap:npm-ci",
    label: "Install pinned Node dependencies",
    command: npmCommand(),
    args: ["ci"]
  });

  for (const model of modelsToPrepare) {
    await runLoggedStage(session, {
      id: `bootstrap:model:${model}`,
      label: `Prepare model ${model}`,
      command: process.execPath,
      args: ["evaluation/scripts/prepare-models.mjs", "--model", model]
    });
  }

  await runLoggedStage(session, {
    id: "bootstrap:apps",
    label: "Prepare and build ArtifexWebGame and Holtwick Tavern",
    command: npmCommand(),
    args: ["run", "evaluation:prepare"]
  });

  for (const [id, label, command, commandArgs] of [
    ["verify:independence", "Verify evaluation independence", npmCommand(), ["run", "evaluation:verify"]],
    ["verify:typecheck", "Typecheck", npmCommand(), ["run", "typecheck"]],
    ["verify:test", "Run unit and integration tests", npmCommand(), ["run", "test"]],
    ["verify:model", "Verify demo model", npmCommand(), ["run", "model:check"]],
    [
      "verify:prebuilt",
      "Verify the pinned prebuilt runtime",
      npmCommand(),
      ["run", "verify:prebuilt"]
    ],
    ["verify:environment", "Check browser and evaluation files", npmCommand(), ["run", "evaluation:check"]]
  ]) {
    await runLoggedStage(session, { id, label, command, args: commandArgs });
  }

  const smokeModel = selectedModels[0];
  for (const experiment of ["e2e-1", "e2e-2"]) {
    await runE2EStage(session, {
      experiment,
      model: smokeModel,
      mode: "smoke"
    });
  }

  let formalCommandFailures = 0;
  for (const experiment of ["e2e-1", "e2e-2"]) {
    for (const model of selectedModels) {
      try {
        await runE2EStage(session, {
          experiment,
          model,
          mode: "formal"
        });
      } catch (error) {
        formalCommandFailures += 1;
        console.error("");
        console.error(
          `FORMAL RUN FAILED: ${experiment} / ${model}: ` +
            (error instanceof Error ? error.message : String(error))
        );
        console.error("The failed run is preserved. Continuing with the next formal run.");
      }
    }
  }
  session.state.formalCommandFailures = formalCommandFailures;
  if (formalCommandFailures > 0) {
    session.state.overallStatus = "partial";
  }
}

function e2eOutputDirectory(session, experiment, mode, model, attempt) {
  return path.join(
    session.directory,
    "runs",
    experiment,
    mode,
    model,
    `attempt-${String(attempt).padStart(2, "0")}`
  );
}

function resolveAttemptValue(value, attempt) {
  return typeof value === "function" ? value(attempt) : value;
}

function isReusableStage(stage, options) {
  if (stage.status !== "completed") return false;
  return options.reusable ? options.reusable(stage) : true;
}

async function runE2EStage(session, options) {
  const { experiment, model, mode } = options;
  const stageId = `run:${experiment}:${mode}:${model}`;
  const stage = await runLoggedStage(session, {
    id: stageId,
    label: `${experiment} ${mode} ${model}`,
    command: process.execPath,
    args: (attempt) => [
      "evaluation/scripts/run.mjs",
      "--experiment",
      experiment,
      "--mode",
      mode,
      "--model",
      model,
      "--out",
      e2eOutputDirectory(session, experiment, mode, model, attempt),
      "--no-judge"
    ],
    metadata: (attempt) => ({
      kind: "e2e",
      experiment,
      model,
      mode,
      outputDirectory: e2eOutputDirectory(
        session,
        experiment,
        mode,
        model,
        attempt
      )
    }),
    // A completed command is evidence even when its records contain failures.
    // Resume keeps those artifacts instead of silently producing a fresh run.
    reusable: (candidate) => candidate.status === "completed"
  });

  if (stage.status !== "completed") {
    throw new Error(`${stageId} did not complete successfully. See ${stage.logFile}.`);
  }

  const validation = await inspectRun(stage.outputDirectory, { experiment, mode });
  stage.validation = validation;
  const recordsComplete = validation.complete && validation.failure === null;
  stage.resultStatus = mode === "smoke"
    ? recordsComplete && validation.errorCount === 0
      ? "smoke-complete"
      : "smoke-failed"
    : !recordsComplete
      ? "incomplete"
      : validation.errorCount > 0
        ? "completed-with-errors"
        : "complete";
  stage.finishedAt = new Date().toISOString();
  await saveState(session);

  if (mode === "formal") {
    await captureReport(
      session,
      experiment,
      model,
      stage.outputDirectory,
      stage.attempt
    );
  }
  console.log(
    `  ${stageId}: records=${validation.recordCount ?? "?"}/${validation.expectedRecordCount ?? "?"}, ` +
      `complete=${validation.complete}, errors=${validation.errorCount}, ` +
      `result=${stage.resultStatus}`
  );
  if (validation.failure) {
    console.log(`  ${stageId}: failure=${validation.failure}`);
  }

  if (mode === "smoke" && stage.resultStatus === "smoke-failed") {
    throw new Error(
      `${stageId} smoke failed: records=${validation.recordCount ?? "?"}, ` +
        `expected=${validation.expectedRecordCount ?? "?"}, ` +
        `complete=${validation.complete}, errors=${validation.errorCount}.`
    );
  }
}

async function runLoggedStage(session, options) {
  const previousAttempts = session.state.stages.filter(
    (stage) => stage.id === options.id
  );
  const existing = previousAttempts.at(-1);
  if (existing && isReusableStage(existing, options)) {
    console.log(`SKIP ${options.id} (completed, attempt ${existing.attempt ?? 1})`);
    return existing;
  }
  if (existing?.status === "completed") {
    existing.status = "incomplete";
    existing.resultStatus = "unverified";
  }

  const attempt = previousAttempts.length + 1;
  const commandArgs = resolveAttemptValue(options.args, attempt);
  const metadata = resolveAttemptValue(options.metadata, attempt) ?? {};
  const baseLogName = `${safeName(options.id)}-attempt-${attempt}`;

  const logDirectory = path.join(session.directory, "logs");
  await mkdir(logDirectory, { recursive: true });
  const logFile = path.join(logDirectory, `${baseLogName}.log`);
  const stage = {
    id: options.id,
    attempt,
    attemptId: `${options.id}:attempt-${attempt}`,
    label: options.label,
    status: "running",
    command: [options.command, ...commandArgs].join(" "),
    cwd: implRoot,
    logFile,
    startedAt: new Date().toISOString(),
    finishedAt: null,
    exitCode: null,
    ...metadata
  };
  session.state.stages.push(stage);
  await saveState(session);

  console.log("");
  console.log(`STAGE ${options.id} (attempt ${attempt})`);
  console.log(`  ${stage.command}`);
  const log = createWriteStream(logFile, { flags: "a" });
  log.write(`# ${stage.startedAt}\n# ${stage.command}\n`);
  const child = spawn(options.command, commandArgs, {
    cwd: implRoot,
    env: {
      ...process.env,
      CHROME_PATH: process.env.CHROME_PATH ?? "",
      FORCE_COLOR: "0",
      NO_COLOR: "1"
    },
    shell: process.platform === "win32" && /\.cmd$/i.test(options.command),
    windowsHide: false,
    stdio: ["inherit", "pipe", "pipe"]
  });

  child.stdout.on("data", (chunk) => {
    process.stdout.write(chunk);
    log.write(chunk);
  });
  child.stderr.on("data", (chunk) => {
    process.stderr.write(chunk);
    log.write(chunk);
  });

  const result = await new Promise((resolve) => {
    child.on("error", (error) => {
      resolve({ code: 1, signal: null, error });
    });
    child.on("exit", (code, signal) => resolve({ code, signal }));
  });
  await new Promise((resolve) => log.end(resolve));

  stage.finishedAt = new Date().toISOString();
  stage.exitCode = result.code;
  stage.signal = result.signal;
  stage.status = result.code === 0 ? "completed" : "failed";
  await saveState(session);

  if (stage.status !== "completed") {
    throw new Error(
      `${options.id} failed with exit code ${result.code ?? "none"}` +
        (result.signal ? ` and signal ${result.signal}` : "") +
        (result.error ? `: ${result.error.message}` : "") +
        `. See ${logFile}.`
    );
  }
  return stage;
}

async function captureReport(session, experiment, model, runDirectory, attempt) {
  const reportDirectory = path.join(session.directory, "reports");
  await mkdir(reportDirectory, { recursive: true });
  const reportFile = path.join(
    reportDirectory,
    `${experiment}-${safeName(model)}-attempt-${attempt}.txt`
  );
  const result = await captureCommand(
    process.execPath,
    ["evaluation/scripts/report.mjs", runDirectory],
    { cwd: implRoot }
  );
  await writeFile(
    reportFile,
    `${result.stdout}${result.stderr ? `\nSTDERR\n${result.stderr}` : ""}`,
    "utf8"
  );
}

async function inspectRun(runDirectory, options = {}) {
  const { mode = null } = options;
  const summary = await readJsonIfPresent(path.join(runDirectory, "summary.json"));
  const records = await readJsonlIfPresent(path.join(runDirectory, "records.jsonl"));
  const runMetadata = await readJsonIfPresent(path.join(runDirectory, "run.json"));
  const failure = await readJsonIfPresent(path.join(runDirectory, "failure.json"));
  const complete = summary?.completeness?.complete === true;
  const observedRecordErrors = Array.isArray(records)
    ? records.filter(
        (record) =>
          record?.kind === "failure" ||
          (record?.error !== null && record?.error !== undefined)
      ).length
    : 0;
  const errorCount = Number(summary?.errorCount ?? observedRecordErrors);
  const expectedRecordCount =
    summary?.completeness?.expected ??
    (mode === "formal"
      ? expectedFormalRecordCount
      : mode === "smoke"
        ? expectedSmokeRecordCount
        : null);
  return {
    complete,
    recordCount: summary?.recordCount ?? records?.length ?? null,
    expectedRecordCount,
    errorCount,
    failure: failure?.error ?? null,
    quality: summary?.quality ?? null,
    platform: runMetadata?.platform ?? null,
    runtimeConfigSha256: runMetadata?.runtimeConfigSha256 ?? null,
    configSha256: runMetadata?.configSha256 ?? null,
    filesPresent: {
      summary: summary !== null,
      records: records !== null,
      run: runMetadata !== null
    }
  };
}

async function finalizeSession(state, sessionDirectory) {
  const resultsFile = path.join(sessionDirectory, "RESULTS.md");
  const feedbackFile = path.join(sessionDirectory, "FEEDBACK.md");
  const packagePath = path.join(handoffRoot, `${state.runId}-results.zip`);
  state.resultsFile = resultsFile;
  state.feedbackFile = feedbackFile;
  state.packagePath = packagePath;
  await ensureFeedbackFile(feedbackFile, state.runId);
  await writeResults(state, resultsFile);
  await saveState({ state, directory: sessionDirectory });
  if (args.noPackage) {
    console.log(`RESULTS: ${resultsFile}`);
    return null;
  }
  await rm(packagePath, { force: true });
  const command = [
    "-NoProfile",
    "-NonInteractive",
    "-Command",
    `Compress-Archive -Path (Join-Path ${quotePowerShell(sessionDirectory)} '*') -DestinationPath ${quotePowerShell(packagePath)} -Force`
  ];
  const result = await captureCommand("powershell.exe", command, {
    cwd: implRoot,
    allowFailure: true
  });
  if (result.code !== 0) {
    console.error(`Failed to create result zip: ${result.stderr || result.stdout}`);
    console.log(`RESULTS: ${resultsFile}`);
    return null;
  }
  console.log(`RESULTS: ${resultsFile}`);
  return packagePath;
}

async function ensureFeedbackFile(file, runId) {
  try {
    await access(file);
    return;
  } catch {
    await writeFile(
      file,
      [
        `# Collaborator feedback: ${runId}`,
        "",
        "- Anything unusual during the run:",
        "- Was the computer interrupted or put to sleep:",
        "- Other observations:",
        ""
      ].join("\n"),
      "utf8"
    );
  }
}

async function writeResults(state, resultsFile) {
  const environment = state.environment ?? {};
  const stages = state.stages ?? [];
  const currentStages = latestStages(stages);
  const e2eStages = currentStages.filter((stage) => stage.kind === "e2e");
  const smokeStages = e2eStages.filter((stage) => stage.mode === "smoke");
  const formalStages = e2eStages.filter((stage) => stage.mode === "formal");
  const lines = [
    "# wLLM Windows E2E handoff",
    "",
    `- Run ID: \`${state.runId}\``,
    `- Status: **${state.overallStatus}**`,
    `- Started: ${state.startedAt}`,
    `- Finished: ${state.completedAt ?? new Date().toISOString()}`,
    `- Model set: \`${state.modelSet}\``,
    `- Models: ${(state.models ?? []).join(", ")}`,
    `- Operator: ${state.operator}`,
    ""
  ];
  if (state.note) {
    lines.push(`Operator note: ${state.note}`, "");
  }
  lines.push(
    "## Machine",
    "",
    `- Host: ${environment.hostname ?? "unknown"}`,
    `- OS: ${environment.os ?? "unknown"}`,
    `- CPU: ${environment.cpu ?? "unknown"}`,
    `- Memory: ${formatNumber(environment.memoryGiB)} GiB`,
    `- GPUs: ${(environment.gpus ?? []).map((gpu) => `${gpu.name} (${gpu.driverVersion ?? "driver unknown"})`).join("; ") || "unknown"}`,
    `- Chrome: ${environment.chromeVersion ?? environment.chromePath ?? "unknown"}`,
    `- Node: ${environment.node ?? "unknown"}`,
    `- Implementation commit: ${environment.implCommit ?? "unknown"}`,
    `- AC power observed: ${environment.onAcPower === false ? "no" : "yes"}`,
    ""
  );
  lines.push(
    "## Smoke results",
    "",
    "| Experiment | Model | Records | Expected | Complete | Errors | Result | Artifact |",
    "|---|---|---:|---:|---|---:|---|---|"
  );
  if (smokeStages.length === 0) {
    lines.push("| none | none | - | - | - | - | no smoke run | - |");
  }
  for (const stage of smokeStages) {
    const validation = stage.validation ?? {};
    const artifact = relativeOrAbsolute(stage.outputDirectory, path.dirname(resultsFile));
    lines.push(
      `| ${stage.experiment} | ${stage.model} | ${validation.recordCount ?? "-"} | ` +
        `${validation.expectedRecordCount ?? "-"} | ${validation.complete ?? "-"} | ` +
        `${validation.errorCount ?? "-"} | ${stage.resultStatus ?? stage.status} | \`${artifact}\` |`
    );
  }
  lines.push(
    "",
    "## Formal results",
    "",
    "| Experiment | Model | Attempt | Records | Expected | Complete | Errors | Result | Artifact |",
    "|---|---|---:|---:|---:|---|---:|---|---|"
  );
  if (formalStages.length === 0) {
    lines.push("| none | none | - | - | - | - | - | no formal run | - |");
  }
  for (const stage of formalStages) {
    const validation = stage.validation ?? {};
    const artifact = relativeOrAbsolute(stage.outputDirectory, path.dirname(resultsFile));
    lines.push(
      `| ${stage.experiment} | ${stage.model} | ${stage.attempt ?? 1} | ` +
        `${validation.recordCount ?? "-"} | ${validation.expectedRecordCount ?? "-"} | ` +
        `${validation.complete ?? "-"} | ${validation.errorCount ?? "-"} | ` +
        `${stage.resultStatus ?? stage.status} | \`${artifact}\` |`
    );
  }
  if (
    formalStages.some((stage) =>
      ["completed-with-errors", "incomplete"].includes(stage.resultStatus)
    )
  ) {
    lines.push(
      "",
      "Runs marked `completed-with-errors` or `incomplete` are reported directly.",
      "Their raw records and logs are preserved; failed tasks remain in the fixed denominator."
    );
  }

  const attemptHistory = stages.filter(
    (stage) => (stage.attempt ?? 1) > 1 || stage.status !== "completed"
  );
  if (attemptHistory.length > 0) {
    lines.push("## Attempt history", "");
    for (const stage of attemptHistory) {
      lines.push(
        `- \`${stage.id}\` attempt ${stage.attempt ?? 1}: ${stage.status}` +
          `${stage.resultStatus ? `/${stage.resultStatus}` : ""}, ` +
          `exit=${stage.exitCode ?? "n/a"}, ` +
          `log=\`${relativeOrAbsolute(stage.logFile, path.dirname(resultsFile))}\`` +
          `${stage.outputDirectory ? `, artifact=\`${relativeOrAbsolute(stage.outputDirectory, path.dirname(resultsFile))}\`` : ""}`
      );
    }
    lines.push("");
  }

  const unfinishedStages = currentStages.filter(
    (stage) => stage.status !== "completed"
  );
  if (unfinishedStages.length > 0) {
    lines.push("## Current incomplete stages", "");
    for (const stage of unfinishedStages) {
      lines.push(`- \`${stage.id}\` attempt ${stage.attempt ?? 1}: ${stage.status}`);
    }
    lines.push("");
  }
  if (state.failure) {
    lines.push("## Fatal error", "", "```text", state.failure.message, "```", "");
  }
  lines.push(
    "## Send back",
    "",
    `Send the sibling file \`${state.runId}-results.zip\` to the experiment owner.`,
    "The zip contains this report, logs, raw records, summaries, and every attempt.",
    "Do not delete or merge partial runs.",
    "For comments, edit `FEEDBACK.md` and rerun `Package-Windows-Results.cmd`.",
    ""
  );
  await writeFile(resultsFile, `${lines.join("\n")}\n`, "utf8");
}

async function createOrResumeSession(requestedModels, requestedPreparedModels) {
  await mkdir(handoffRoot, { recursive: true });
  let sessionDirectory;
  let state;
  if (args.resume) {
    sessionDirectory = args.session
      ? path.join(handoffRoot, safeName(args.session))
      : await findLatestSession();
    if (!sessionDirectory) {
      throw new Error("No previous Windows handoff session was found to resume.");
    }
    state = await readJson(path.join(sessionDirectory, "handoff-state.json"));
    state.stages ??= [];
    if (!Array.isArray(state.models) || state.models.length === 0) {
      state.models = modelSets[state.modelSet] ?? [];
    }
    if (!Array.isArray(state.preparedModels) || state.preparedModels.length === 0) {
      state.preparedModels = unique(state.models);
    }
    normalizeStageAttempts(state);
    for (const stage of state.stages) {
      if (stage.status === "running") stage.status = "interrupted";
    }
    state.overallStatus = "running";
    state.completedAt = null;
    state.failure = null;
    console.log(`Resuming ${state.runId}`);
    console.log(`  Persisted model set: ${state.modelSet}`);
    console.log(`  Persisted models: ${(state.models ?? []).join(", ")}`);
    if (args.modelSet !== state.modelSet) {
      console.log(
        `  NOTE: ignoring --model-set ${args.modelSet}; resume keeps ${state.modelSet}.`
      );
    }
    await saveState({ state, directory: sessionDirectory });
    return { state, directory: sessionDirectory };
  }

  const runId = safeName(
    args.session ??
      `${os.hostname()}-${new Date().toISOString().replace(/[-:]/g, "").replace(/\..+/, "")}`
  );
  sessionDirectory = path.join(handoffRoot, runId);
  const existingState = await readJsonIfPresent(
    path.join(sessionDirectory, "handoff-state.json")
  );
  if (existingState) {
    throw new Error(
      `Session ${runId} already exists. Use --resume to continue it.`
    );
  }
  await mkdir(sessionDirectory, { recursive: true });
  state = {
    schemaVersion: 1,
    runId,
    overallStatus: "running",
    startedAt: new Date().toISOString(),
    completedAt: null,
    platform: process.platform,
    hostname: os.hostname(),
    modelSet: args.modelSet,
    models: requestedModels,
    preparedModels: requestedPreparedModels,
    operator: args.operator || process.env.USERNAME || os.userInfo().username,
    note: args.note || "",
    failure: null,
    stages: [],
    resultsFile: null,
    packagePath: null
  };
  await saveState({ state, directory: sessionDirectory });
  return { state, directory: sessionDirectory };
}

async function loadSessionForPackage() {
  const sessionDirectory = args.session
    ? path.join(handoffRoot, safeName(args.session))
    : await findLatestSession();
  if (!sessionDirectory) {
    throw new Error("No Windows handoff session was found.");
  }
  const state = await readJson(path.join(sessionDirectory, "handoff-state.json"));
  normalizeStageAttempts(state);
  return {
    state,
    directory: sessionDirectory
  };
}

async function findLatestSession() {
  const entries = await readdir(handoffRoot, { withFileTypes: true }).catch(() => []);
  const sessions = [];
  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    const directory = path.join(handoffRoot, entry.name);
    const state = await readJsonIfPresent(path.join(directory, "handoff-state.json"));
    if (state?.runId) {
      sessions.push({ directory, startedAt: state.startedAt ?? "" });
    }
  }
  sessions.sort((left, right) => right.startedAt.localeCompare(left.startedAt));
  return sessions[0]?.directory ?? null;
}

function normalizeStageAttempts(state) {
  const counts = new Map();
  for (const stage of state.stages ?? []) {
    const count = (counts.get(stage.id) ?? 0) + 1;
    counts.set(stage.id, count);
    if (!Number.isInteger(stage.attempt) || stage.attempt < 1) {
      stage.attempt = count;
    }
    stage.attemptId ??= `${stage.id}:attempt-${stage.attempt}`;
  }
}

function latestStages(stages = []) {
  const latest = new Map();
  for (const stage of stages) latest.set(stage.id, stage);
  return [...latest.values()];
}

function validatePersistedModels(models, allowedModels = null) {
  const validIds = new Set(Object.values(modelSets).flat());
  if (!Array.isArray(models) || models.length === 0) {
    throw new Error("The handoff session does not record any models.");
  }
  const unknown = models.filter((model) => !validIds.has(model));
  if (unknown.length > 0) {
    throw new Error(`The handoff session contains unknown models: ${unknown.join(", ")}`);
  }
  if (allowedModels) {
    const allowed = new Set(allowedModels);
    const outsideSet = models.filter((model) => !allowed.has(model));
    if (outsideSet.length > 0) {
      throw new Error(
        `The handoff session model set is inconsistent: ${outsideSet.join(", ")}`
      );
    }
  }
  return unique(models);
}

async function saveState(session) {
  const file = path.join(session.directory, "handoff-state.json");
  const temporary = `${file}.tmp`;
  await writeFile(temporary, `${JSON.stringify(session.state, null, 2)}\n`, "utf8");
  await rename(temporary, file);
}

async function collectEnvironment() {
  const memoryBytes = os.totalmem();
  const disk = await statfs(implRoot);
  const chromePath = await findChrome();
  const gpus = await powershellJson(
    "Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,AdapterRAM | ConvertTo-Json -Compress"
  ).then((value) => value ? (Array.isArray(value) ? value : [value]) : []);
  const chromeVersion = chromePath
    ? await captureCommand(
        "powershell.exe",
        [
          "-NoProfile",
          "-NonInteractive",
          "-Command",
          `(Get-Item ${quotePowerShell(chromePath)}).VersionInfo.ProductVersion`
        ],
        { cwd: implRoot, allowFailure: true }
      ).then((result) => result.stdout.trim() || null)
    : null;
  const battery = await powershellJson(
    "Get-CimInstance Win32_Battery | Select-Object -First 1 BatteryStatus | ConvertTo-Json -Compress"
  );
  const gitStatus = await captureCommand("git", ["status", "--short"], {
    cwd: implRoot,
    allowFailure: true
  });
  return {
    capturedAt: new Date().toISOString(),
    hostname: os.hostname(),
    os: `${os.type()} ${os.release()} ${os.arch()}`,
    node: process.version,
    nodeMajor: Number(process.versions.node.split(".")[0]),
    npm: await commandVersion(npmCommand()),
    git: await commandVersion("git"),
    implCommit: await commandVersion("git", ["rev-parse", "HEAD"]),
    implWorktreeClean:
      gitStatus.code === 0 ? gitStatus.stdout.trim().length === 0 : null,
    cpu: os.cpus()[0]?.model?.trim() ?? null,
    logicalProcessors: os.cpus().length,
    memoryGiB: memoryBytes / 1024 ** 3,
    freeDiskGiB: (Number(disk.bavail) * Number(disk.bsize)) / 1024 ** 3,
    chromePath,
    chromeVersion,
    gpus,
    onAcPower: battery ? Number(battery.BatteryStatus) === 2 : true
  };
}

async function findChrome() {
  const candidates = [
    process.env.CHROME_PATH,
    path.join(process.env.ProgramFiles ?? "C:\\Program Files", "Google", "Chrome", "Application", "chrome.exe"),
    path.join(process.env["ProgramFiles(x86)"] ?? "C:\\Program Files (x86)", "Google", "Chrome", "Application", "chrome.exe"),
    path.join(process.env.LOCALAPPDATA ?? "", "Google", "Chrome", "Application", "chrome.exe")
  ].filter(Boolean);
  for (const candidate of candidates) {
    try {
      await access(candidate);
      return candidate;
    } catch {
      // Try the next standard Chrome location.
    }
  }
  return null;
}

async function powershellJson(command) {
  const result = await captureCommand(
    "powershell.exe",
    ["-NoProfile", "-NonInteractive", "-Command", command],
    { cwd: implRoot, allowFailure: true }
  );
  if (result.code !== 0 || !result.stdout.trim()) return null;
  try {
    return JSON.parse(result.stdout);
  } catch {
    return null;
  }
}

async function commandVersion(command, commandArgs = ["--version"]) {
  const result = await captureCommand(command, commandArgs, {
    cwd: implRoot,
    allowFailure: true
  });
  return result.code === 0 ? result.stdout.trim() || result.stderr.trim() : null;
}

async function captureCommand(command, commandArgs, options = {}) {
  return new Promise((resolve) => {
    const child = spawn(command, commandArgs, {
      cwd: options.cwd ?? implRoot,
      env: { ...process.env, FORCE_COLOR: "0", NO_COLOR: "1" },
      shell: process.platform === "win32" && /\.cmd$/i.test(command),
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"]
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.on("error", (error) => {
      resolve({ code: 1, stdout, stderr: `${stderr}${error.message}` });
    });
    child.on("exit", (code) => {
      resolve({ code: code ?? 1, stdout, stderr });
    });
  });
}

async function readJson(file) {
  return JSON.parse(await readFile(file, "utf8"));
}

async function readJsonlIfPresent(file) {
  try {
    const content = await readFile(file, "utf8");
    return content
      .split(/\r?\n/)
      .filter((line) => line.trim().length > 0)
      .map((line) => JSON.parse(line));
  } catch (error) {
    if (error?.code === "ENOENT") return null;
    throw error;
  }
}

async function readJsonIfPresent(file) {
  try {
    return await readJson(file);
  } catch (error) {
    if (error?.code === "ENOENT") return null;
    throw error;
  }
}

async function writeJson(file, value) {
  await mkdir(path.dirname(file), { recursive: true });
  await writeFile(file, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function parseArgs(argv) {
  const result = {
    help: false,
    dryRun: false,
    resume: false,
    packageOnly: false,
    noPackage: false,
    noOpen: false,
    yes: false,
    modelSet: "all",
    session: null,
    operator: null,
    note: ""
  };
  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index];
    if (value === "--help" || value === "-h") result.help = true;
    else if (value === "--dry-run") result.dryRun = true;
    else if (value === "--resume") result.resume = true;
    else if (value === "--package-only") result.packageOnly = true;
    else if (value === "--no-package") result.noPackage = true;
    else if (value === "--no-open") result.noOpen = true;
    else if (value === "--yes") result.yes = true;
    else if (value === "--model-set") {
      result.modelSet = requireOptionValue(argv, ++index, value);
    } else if (value === "--session") {
      result.session = requireOptionValue(argv, ++index, value);
    } else if (value === "--operator") {
      result.operator = requireOptionValue(argv, ++index, value);
    } else if (value === "--note") {
      result.note = requireOptionValue(argv, ++index, value);
    }
    else throw new Error(`Unknown argument: ${value}`);
  }
  return result;
}

function requireOptionValue(argv, index, option) {
  const value = argv[index];
  if (value === undefined || value.startsWith("--")) {
    throw new Error(`Missing value for ${option}`);
  }
  return value;
}

function printDryRun() {
  const models = modelSets[args.modelSet];
  if (!models) throw new Error(`Unknown model set: ${args.modelSet}`);
  console.log("Windows E2E dry run");
  console.log(`  Platform: ${process.platform}`);
  console.log(`  Model set: ${args.modelSet}`);
  console.log(`  Models: ${models.join(", ")}`);
  console.log(
    `  Formal records per experiment/model: ${expectedFormalRecordCount}`
  );
  console.log("  Stages:");
  for (const experiment of ["e2e-1", "e2e-2"]) {
    console.log(`    ${experiment} smoke ${models[0]}`);
  }
  for (const experiment of ["e2e-1", "e2e-2"]) {
    for (const model of models) {
      console.log(`    ${experiment} formal ${model}`);
    }
  }
}

function printHelp() {
  console.log(`Usage:
  Run-Windows-E2E.cmd [options]

Options:
  --model-set all       Run all four configured models (default)
  --model-set small     Run only Llama-1B and Qwen-1.5B for debugging
  --resume              Resume the latest session without rerunning completed stages
  --session <id>        Select a named session
  --operator <name>     Record the operator name
  --note "text"         Add a short note to RESULTS.md
  --yes                 Skip the interactive start prompt
  --package-only        Repackage an existing session
  --no-package          Do not create a result zip
  --no-open             Do not open Explorer after completion
  --dry-run             Print the planned stages without running them
  --help                Show this help

The default handoff runs one simple smoke for each experiment, followed by
one formal repeat for e2e-1 and e2e-2 across all four paper models.
Each formal run is judge-free and has a fixed denominator of ${expectedFormalRecordCount} records.
Results are written under:
  impl\\handoff\\windows\\`);
}

function npmCommand() {
  return "npm.cmd";
}

function unique(values) {
  return [...new Set(values)];
}

function safeName(value) {
  return String(value).replace(/[^A-Za-z0-9._-]+/g, "-");
}

function quotePowerShell(value) {
  return `'${String(value).replaceAll("'", "''")}'`;
}

function relativeOrAbsolute(target, from) {
  if (!target) return "";
  return path.relative(from, target).replaceAll("\\", "/");
}

function formatNumber(value) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(2) : "unknown";
}

function openExplorer(target) {
  try {
    const child = spawn("explorer.exe", [target], {
      detached: true,
      stdio: "ignore",
      windowsHide: true
    });
    child.unref();
  } catch {
    // Opening Explorer is a convenience only.
  }
}
