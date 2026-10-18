#!/usr/bin/env node
import { readdir, readFile, stat } from "node:fs/promises";
import path from "node:path";
import { loadEvaluationConfig, experimentFor } from "../lib/config.mjs";
import { artifactRoot, implRoot } from "../lib/paths.mjs";
import { buildSummary } from "../lib/summary.mjs";

const cli = parseArgs(process.argv.slice(2));
const loaded = await loadEvaluationConfig();
const runDirectory = await resolveRunDirectory(cli, loaded);
const [records, summaryFile, gateReport, runMetadata] = await Promise.all([
  readRecords(path.join(runDirectory, "records.jsonl")),
  readJsonIfPresent(path.join(runDirectory, "summary.json")),
  readJsonIfPresent(path.join(runDirectory, "gate-report.json")),
  readJsonIfPresent(path.join(runDirectory, "run.json"))
]);

const experimentId =
  runMetadata?.experiment ?? gateReport?.experiment ?? cli.experiment ?? "e2e-1";
const experiment = experimentFor(loaded, experimentId);
const summary =
  summaryFile ??
  buildSummary(records, {
    armPairs: experiment.arms.length >= 2 ? [[experiment.arms[0], experiment.arms[1]]] : []
  });

if (cli.json) {
  process.stdout.write(
    `${JSON.stringify(
      {
        runDirectory,
        run: runMetadata,
        summary,
        gates: gateReport,
        records
      },
      null,
      2
    )}\n`
  );
  process.exit(0);
}

printHeader({ runDirectory, experiment, runMetadata, summary, gateReport, records });
printQuality(summary, records);
printServing(summary, records);
printFrames(summary, records);
printPairwise(summary, experiment);
printGates(gateReport);

function parseArgs(argv) {
  const options = { json: false };
  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index];
    switch (value) {
      case "--json":
        options.json = true;
        break;
      case "--experiment":
        options.experiment = argv[++index];
        break;
      case "--model":
        options.model = argv[++index];
        break;
      case "--platform":
        options.platform = argv[++index];
        break;
      default:
        if (!value.startsWith("--")) {
          options.input = value;
        }
        break;
    }
  }
  return options;
}

async function resolveRunDirectory(options, config) {
  if (options.input) {
    const resolved = path.resolve(options.input);
    const info = await stat(resolved).catch(() => null);
    if (!info) throw new Error(`run path does not exist: ${resolved}`);
    if (info.isDirectory()) {
      return latestInsideDirectory(resolved);
    }
    const pointer = JSON.parse(await readFile(resolved, "utf8"));
    return resolvePointer(resolved, pointer, config, options);
  }
  const latestFile = await findLatestPointer(config);
  if (!latestFile) {
    throw new Error(
      `no latest.json found under ${artifactRoot}; pass a run directory explicitly`
    );
  }
  const pointer = JSON.parse(await readFile(latestFile, "utf8"));
  return resolvePointer(latestFile, pointer, config, options);
}

async function latestInsideDirectory(directory) {
  const latest = path.join(directory, "latest.json");
  const info = await stat(latest).catch(() => null);
  if (!info) return directory;
  const pointer = JSON.parse(await readFile(latest, "utf8"));
  if (typeof pointer.run === "string") {
    const candidate = path.join(directory, pointer.run);
    const candidateInfo = await stat(candidate).catch(() => null);
    if (candidateInfo?.isDirectory()) return candidate;
  }
  return directory;
}

async function resolvePointer(pointerFile, pointer, config, options) {
  let candidates = [];
  if (Array.isArray(pointer.runs) && pointer.runs.length > 0) {
    candidates = pointer.runs.map((entry) => path.resolve(entry));
  } else if (typeof pointer.run === "string") {
    candidates = [path.resolve(path.dirname(pointerFile))];
  } else {
    throw new Error(`unsupported latest.json shape: ${pointerFile}`);
  }
  const filtered = [];
  for (const candidate of candidates) {
    const metadata = await readJsonIfPresent(path.join(candidate, "run.json"));
    if (options.experiment && metadata?.experiment !== options.experiment) continue;
    if (options.model && metadata?.model !== options.model) continue;
    if (options.platform && metadata?.platform !== options.platform) continue;
    filtered.push(candidate);
  }
  if (filtered.length === 0) {
    throw new Error(`latest pointer has no run matching the requested filters`);
  }
  return filtered.at(-1);
}

async function findLatestPointer(config) {
  const candidates = [];
  for (const experiment of Object.values(config.config.experiments)) {
    const root = path.join(implRoot, experiment.artifactRoot);
    await collectLatestFiles(root, candidates);
  }
  candidates.sort((left, right) => right.mtimeMs - left.mtimeMs);
  return candidates[0]?.file ?? null;
}

async function collectLatestFiles(directory, output) {
  const entries = await readdir(directory, { withFileTypes: true }).catch(() => []);
  for (const entry of entries) {
    const file = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      await collectLatestFiles(file, output);
    } else if (entry.isFile() && entry.name === "latest.json") {
      const info = await stat(file);
      output.push({ file, mtimeMs: info.mtimeMs });
    }
  }
}

async function readRecords(file) {
  const text = await readFile(file, "utf8").catch((error) => {
    if (error?.code === "ENOENT") return "";
    throw error;
  });
  return text
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line, index) => {
      try {
        return JSON.parse(line);
      } catch (error) {
        throw new Error(`invalid JSON in ${file}:${index + 1}: ${error.message}`);
      }
    });
}

async function readJsonIfPresent(file) {
  try {
    return JSON.parse(await readFile(file, "utf8"));
  } catch (error) {
    if (error?.code === "ENOENT") return null;
    throw error;
  }
}

function printHeader({
  runDirectory,
  experiment,
  runMetadata,
  summary,
  gateReport,
  records
}) {
  const taskRecords = records.filter(
    (record) => record.kind === "task" || record.kind === "recovery"
  );
  const failureRecords = taskRecords.filter((record) => record.error != null);
  console.log("RUN");
  console.log(`directory: ${runDirectory}`);
  console.log(`experiment: ${experiment.id} (${experiment.label})`);
  if (runMetadata) {
    console.log(
      `platform: ${runMetadata.platform ?? "unknown"}  model: ${runMetadata.model ?? "unknown"}`
    );
    console.log(
      `mode: ${runMetadata.mode ?? "unknown"}  config: ${runMetadata.configSha256 ?? "unknown"}`
    );
  }
  console.log(
    `records: ${summary.recordCount ?? "-"}  task/recovery: ${taskRecords.length}  errors: ${failureRecords.length}`
  );
  if (summary.completeness) {
    console.log(
      `completeness: observed=${summary.completeness.observed ?? "-"} expected=${summary.completeness.expected ?? "-"} complete=${summary.completeness.complete ?? "unknown"}`
    );
  }
  if (gateReport) {
    const failed = gateReport.gates?.filter((gate) => gate.status !== "pass") ?? [];
    console.log(
      `gates: ${gateReport.gates?.length ?? 0} total, ${failed.length} not pass`
    );
  }
  console.log("");
}

function printQuality(summary, records) {
  console.log("TASK QUALITY");
  const attempted = records.filter(
    (record) => typeof record.judgeValid === "boolean"
  );
  const quality = summary.quality ?? {};
  if (attempted.length === 0) {
    console.log("no judge records");
    console.log("");
    return;
  }
  console.log(
    `valid: ${quality.valid ?? 0}/${quality.n ?? attempted.length} (${formatRatio(quality.validFraction)})`
  );
  if (Number.isFinite(quality.scored) && quality.scored !== quality.n) {
    console.log(`numeric scores: ${quality.scored}/${quality.n}`);
  }
  for (const [arm, values] of Object.entries(quality.byArm ?? {}).sort()) {
    console.log(
      `${arm}: n=${values.n} median=${formatNumber(values.median)} mean=${formatNumber(values.mean)} p95=${formatNumber(values.p95)}`
    );
  }
  console.log("");
}

function printServing(summary, records) {
  console.log("SERVING AND PRIORITY");
  for (const group of summary.groups ?? []) {
    if (!["priority-target", "recovery-target", "background", null].includes(group.role)) {
      continue;
    }
    const metrics = group.metrics ?? {};
    console.log(
      [
        `arm=${group.arm}`,
        `pressure=${group.pressureId ?? "none"}`,
        `app=${group.app ?? "all"}`,
        `role=${group.role ?? "all"}`,
        `n=${group.n}`,
        `ttft=${formatNumber(metrics.ttftMs?.median)}`,
        `priority_first=${formatNumber(metrics.priorityFirstTokenLatencyMs?.median)}`,
        `priority_complete=${formatNumber(metrics.priorityCompletionMs?.median)}`,
        `recovery_ttft=${formatNumber(metrics.recoveryFirstTokenLatencyMs?.median)}`,
        `recovery_complete=${formatNumber(metrics.recoveryCompletionMs?.median)}`,
        `wall=${formatNumber(metrics.wallMs?.median)}`,
        `decode_tok_s=${formatNumber(metrics.decodeTokensPerSecond?.median)}`,
        `replayed=${formatNumber(metrics.replayedTokens?.median)}`,
        `prefill_ms_token=${formatNumber(metrics.measuredPrefillMsPerToken?.median)}`,
        `units=${formatNumber(metrics.units?.median)}`,
        `prefill=${formatNumber(metrics.prefillTokens?.median)}`,
        `cache=${formatNumber(metrics.cacheTokens?.median)}`,
        `processed=${formatNumber(metrics.prefillProcessedTokens?.median)}`,
        `lease=${leaseSummary(records, group)}`
      ].join("  ")
    );
  }
  console.log("");
}

function printFrames(summary, records) {
  console.log("FRAME PERFORMANCE");
  for (const group of summary.groups ?? []) {
    const metrics = group.metrics ?? {};
    console.log(
      [
        `arm=${group.arm}`,
        `pressure=${group.pressureId ?? "none"}`,
        `app=${group.app ?? "all"}`,
        `role=${group.role ?? "all"}`,
        `n=${group.n}`,
        `p50=${formatNumber(metrics.frameP50Ms?.median)}`,
        `p95=${formatNumber(metrics.frameP95Ms?.median)}`,
        `p99=${formatNumber(metrics.frameP99Ms?.median)}`,
        `max=${formatNumber(metrics.frameMaxMs?.median)}`,
        `over33=${formatNumber(metrics.over33Ratio?.median)}`,
        `over50=${formatNumber(metrics.over50Ratio?.median)}`
      ].join("  ")
    );
  }
  console.log("");
}

function printPairwise(summary, experiment) {
  console.log("PAIRED CONTRASTS");
  const pairs = (summary.pairwise ?? []).filter(
    (row) => row.leftArm === experiment.arms[0]
  );
  if (pairs.length === 0) {
    console.log("no paired rows");
    console.log("");
    return;
  }
  const metrics = [
    "priorityFirstTokenLatencyMs",
    "priorityCompletionMs",
    "recoveryFirstTokenLatencyMs",
    "recoveryCompletionMs",
    "replayedTokens",
    "ttftMs",
    "wallMs",
    "decodeTokensPerSecond",
    "frameP95Ms"
  ];
  for (const row of pairs) {
    console.log(`pressure=${row.pressureId} n=${row.n} left=${row.leftArm} right=${row.rightArm}`);
    for (const metric of metrics) {
      const value = row.metrics?.[metric];
      if (!value || value.n === 0) continue;
      console.log(
        `  ${metric}: diff=${formatNumber(value.medianDifference)} CI95=[${formatNumber(value.ci95?.[0])}, ${formatNumber(value.ci95?.[1])}] rel=${formatRatio(value.relativeEffect)}`
      );
    }
  }
  console.log("");
}

function printGates(gateReport) {
  console.log("GATES");
  if (!gateReport) {
    console.log("gate-report.json is missing");
    console.log("");
    return;
  }
  for (const gate of gateReport.gates ?? []) {
    console.log(`${gate.id}: ${gate.status}`);
    const fields = Object.entries(gate)
      .filter(([key]) => !["id", "status", "type", "description"].includes(key))
      .filter(([, value]) => ["string", "number", "boolean"].includes(typeof value));
    if (fields.length > 0) {
      console.log(
        `  ${fields.map(([key, value]) => `${key}=${String(value)}`).join("  ")}`
      );
    }
  }
  console.log("");
}


function leaseSummary(records, group) {
  const matching = records.filter(
    (record) =>
      record.arm === group.arm &&
      (record.pressureId ?? "none") === (group.pressureId ?? "none") &&
      (record.app ?? null) === (group.app ?? null) &&
      (record.role ?? null) === (group.role ?? null)
  );
  const counts = new Map();
  for (const record of matching) {
    const reason = record.leaseReason ?? "unknown";
    counts.set(reason, (counts.get(reason) ?? 0) + 1);
  }
  return [...counts.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([reason, count]) => `${reason}:${count}`)
    .join(",");
}

function formatNumber(value) {
  if (!Number.isFinite(value)) return "-";
  if (Math.abs(value) >= 1000) return value.toFixed(1);
  if (Math.abs(value) >= 10) return value.toFixed(2);
  return value.toFixed(3);
}

function formatRatio(value) {
  if (!Number.isFinite(value)) return "-";
  return `${(value * 100).toFixed(1)}%`;
}
