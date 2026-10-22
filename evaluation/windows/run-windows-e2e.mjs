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
const judgeModel = "llama-3.2-3b-instruct";
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
const preregistration = JSON.parse(
  await readFile(
    path.join(evaluationRoot, "config", "e2e-preregistration.json"),
    "utf8"
  )
);
const expectedFormalRecordCount =
  2 *
  e2eConfig.modes.formal.repeats *
  e2eConfig.pressure.schedule.length *
  e2eConfig.layout.appTabs;
const expectedGateCountByExperiment = Object.fromEntries(
  ["e2e-1", "e2e-2"].map((experiment) => [
    experiment,
    preregistration.gates[experiment].length
  ])
);

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
const requestedPreparedModels = unique([...requestedModelIds, judgeModel]);
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
    const gateFailures = currentStages.filter(
      (stage) => stage.resultStatus === "gate-fail"
    );
    session.state.overallStatus =
      unfinishedStages.length > 0
        ? "partial"
        : gateFailures.length > 0
          ? "completed-with-gate-failures"
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

  for (const experiment of ["e2e-1", "e2e-2"]) {
    for (const model of selectedModels) {
      await runE2EStage(session, {
        experiment,
        model,
        mode: "smoke",
        judge: false
      });
      if (!args.skipPilot) {
        await runE2EStage(session, {
          experiment,
          model,
          mode: "pilot",
          judge: false
        });
      }
    }
  }

  let formalCommandFailures = 0;
  for (const experiment of ["e2e-1", "e2e-2"]) {
    for (const model of selectedModels) {
      try {
        await runE2EStage(session, {
          experiment,
          model,
          mode: "formal",
          judge: true
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
  const {
    experiment,
    model,
    mode,
    judge
  } = options;
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
      ...(judge ? ["--judge"] : ["--no-judge"])
    ],
    metadata: (attempt) => ({
      kind: "e2e",
      experiment,
      model,
      mode,
      judge,
      outputDirectory: e2eOutputDirectory(
        session,
        experiment,
        mode,
        model,
        attempt
      )
    }),
    reusable: (candidate) => candidate.validation?.technicalPass === true
  });

  if (stage.status !== "completed") {
    throw new Error(`${stageId} did not complete successfully. See ${stage.logFile}.`);
  }

  if (stage.validation?.technicalPass === true) {
    if (mode === "formal") {
      await captureReport(
        session,
        experiment,
        model,
        stage.outputDirectory,
        stage.attempt
      );
    }
    return;
  }

  const validation = await inspectRun(stage.outputDirectory, { experiment, mode });
  stage.validation = validation;
  stage.resultStatus = validation.technicalPass
    ? validation.formalGatePass
      ? "pass"
      : mode === "formal"
        ? "gate-fail"
        : "technical-pass"
    : "incomplete";
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
  if (!validation.technicalPass) {
    stage.status = "incomplete";
    stage.resultStatus = "incomplete";
    await saveState(session);
    throw new Error(
      `${stageId} is incomplete: records=${validation.recordCount ?? "?"}, ` +
        `expected=${validation.expectedRecordCount ?? "?"}, ` +
        `complete=${validation.complete}, errors=${validation.errorCount ?? "?"}, ` +
        `gates=${validation.gates.length}/${validation.expectedGateCount ?? "?"}.`
    );
  }
  console.log(
    `  ${stageId}: records=${validation.recordCount ?? "?"}, ` +
      `complete=${validation.complete}, gates=${formatGateCounts(validation.gateCounts)}`
  );
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
  const { experiment = null, mode = null } = options;
  const summary = await readJsonIfPresent(path.join(runDirectory, "summary.json"));
  const gateReport = await readJsonIfPresent(path.join(runDirectory, "gate-report.json"));
  const runMetadata = await readJsonIfPresent(path.join(runDirectory, "run.json"));
  const failure = await readJsonIfPresent(path.join(runDirectory, "failure.json"));
  const gates = Array.isArray(gateReport?.gates) ? gateReport.gates : [];
  const gateCounts = { pass: 0, fail: 0, incomplete: 0, skipped: 0, other: 0 };
  for (const gate of gates) {
    const status = String(gate.status ?? "").toLowerCase();
    if (Object.hasOwn(gateCounts, status) && status !== "other") {
      gateCounts[status] += 1;
    } else {
      gateCounts.other += 1;
    }
  }
  const complete = summary?.completeness?.complete === true;
  const errorCount = Number(summary?.errorCount ?? 0);
  const expectedRecordCount = summary?.completeness?.expected ?? null;
  const expectedGateCount =
    mode === "formal" ? expectedGateCountByExperiment[experiment] ?? null : null;
  const protocolMatches =
    mode !== "formal" ||
    (expectedRecordCount === expectedFormalRecordCount &&
      gates.length === expectedGateCount);
  return {
    complete,
    recordCount: summary?.recordCount ?? null,
    expectedRecordCount,
    expectedGateCount,
    errorCount,
    failure: failure?.error ?? null,
    gateCounts,
    formalGatePass:
      gates.length > 0 &&
      gates.every((gate) => String(gate.status).toLowerCase() === "pass"),
    gates: gates.map((gate) => ({
      id: gate.id,
      status: gate.status,
      numerator: gate.numerator ?? null,
      denominator: gate.denominator ?? null
    })),
    quality: summary?.quality ?? null,
    platform: runMetadata?.platform ?? null,
    runtimeConfigSha256: runMetadata?.runtimeConfigSha256 ?? null,
    configSha256: runMetadata?.configSha256 ?? null,
    filesPresent: {
      summary: summary !== null,
      gateReport: gateReport !== null,
      run: runMetadata !== null
    },
    technicalPass:
      summary !== null &&
      runMetadata !== null &&
      complete &&
      errorCount === 0 &&
      !failure &&
      protocolMatches &&
      (mode !== "formal" || gateReport !== null)
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
    "## Formal results",
    "",
    "| Experiment | Model | Attempt | Records | Expected | Complete | Errors | Gates | Result | Artifact |",
    "|---|---|---:|---:|---:|---|---:|---|---|---|"
  );
  if (formalStages.length === 0) {
    lines.push("| none | none | - | - | - | - | - | - | no formal run | - |");
  }
  for (const stage of formalStages) {
    const validation = stage.validation ?? {};
    const artifact = relativeOrAbsolute(stage.outputDirectory, path.dirname(resultsFile));
    lines.push(
      `| ${stage.experiment} | ${stage.model} | ${stage.attempt ?? 1} | ` +
        `${validation.recordCount ?? "-"} | ${validation.expectedRecordCount ?? "-"} | ` +
        `${validation.complete ?? "-"} | ${validation.errorCount ?? "-"} | ` +
        `${formatGateCounts(validation.gateCounts)} | ` +
        `${stage.resultStatus ?? stage.status} | \`${artifact}\` |`
    );
  }
  lines.push("", "## Gate details", "");
  for (const stage of formalStages) {
    lines.push(
      `### ${stage.experiment} / ${stage.model} / attempt ${stage.attempt ?? 1}`,
      ""
    );
    if (!stage.validation?.gates?.length) {
      lines.push(
        `No gate report was produced. Stage status: ${stage.status}.`,
        ""
      );
      continue;
    }
    for (const gate of stage.validation.gates) {
      lines.push(`- \`${gate.id}\`: **${gate.status}**`);
    }
    lines.push("");
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
    "The zip contains this report, logs, raw records, summaries, gate reports, and every attempt.",
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
      state.preparedModels = unique([...state.models, judgeModel]);
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
    skipPilot: false,
    modelSet: "small",
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
    else if (value === "--skip-pilot") result.skipPilot = true;
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
  console.log(`  Judge model: ${judgeModel}`);
  console.log("  Stages:");
  for (const experiment of ["e2e-1", "e2e-2"]) {
    for (const model of models) {
      console.log(`    ${experiment} smoke ${model}`);
      if (!args.skipPilot) console.log(`    ${experiment} pilot ${model}`);
      console.log(`    ${experiment} formal ${model}`);
    }
  }
}

function printHelp() {
  console.log(`Usage:
  Run-Windows-E2E.cmd [options]

Options:
  --model-set small     Run Llama-1B and Qwen-1.5B (default)
  --model-set all       Run all four configured models
  --skip-pilot          Skip pilot runs after smoke
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

The default handoff runs e2e-1 and e2e-2 for two small models. Formal runs
include the configured judge model. Results are written under:
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

function formatGateCounts(counts) {
  if (!counts) return "n/a";
  return `${counts.pass}P/${counts.fail}F/${counts.incomplete}I/${counts.skipped}S`;
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
