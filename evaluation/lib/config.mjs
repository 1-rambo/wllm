import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { configRoot, platformId } from "./paths.mjs";

export async function readJson(file) {
  return JSON.parse(await readFile(file, "utf8"));
}

export async function loadEvaluationConfig() {
  const file = path.join(configRoot, "e2e.json");
  const preregistrationFile = path.join(configRoot, "e2e-preregistration.json");
  const modelsFile = path.join(configRoot, "models.json");
  const [configText, preregistrationText, models] = await Promise.all([
    readFile(file, "utf8"),
    readFile(preregistrationFile, "utf8"),
    readJson(modelsFile)
  ]);
  const config = JSON.parse(configText);
  const preregistration = JSON.parse(preregistrationText);
  const configSha256 = sha256(configText);
  if (
    typeof preregistration.configSha256 === "string" &&
    preregistration.configSha256 !== configSha256
  ) {
    throw new Error(
      `frozen preregistration does not match e2e.json: ` +
        `expected ${preregistration.configSha256}, got ${configSha256}`
    );
  }

  const runtimeConfigFile = path.join(configRoot, "e2e-runtime.json");
  const runtimeConfigText = await readOptionalText(runtimeConfigFile);
  const runtimeConfig =
    runtimeConfigText === null ? null : JSON.parse(runtimeConfigText);
  const effectiveConfig = applyRuntimeExecutionOverrides(config, runtimeConfig);
  return {
    file,
    preregistrationFile,
    runtimeConfigFile: runtimeConfigText === null ? null : runtimeConfigFile,
    runtimeConfigSha256:
      runtimeConfigText === null ? null : sha256(runtimeConfigText),
    modelsFile,
    config: effectiveConfig,
    frozenConfig: config,
    runtimeArmOverrides: runtimeConfig?.armOverrides ?? {},
    preregistration,
    models,
    configSha256
  };
}

async function readOptionalText(file) {
  try {
    return await readFile(file, "utf8");
  } catch (error) {
    if (error && typeof error === "object" && error.code === "ENOENT") return null;
    throw error;
  }
}

function applyRuntimeExecutionOverrides(config, overrides) {
  if (overrides === null) return config;
  if (typeof overrides !== "object" || Array.isArray(overrides)) {
    throw new Error("e2e-runtime.json must contain an object");
  }
  const allowed = new Set(["execution", "armOverrides"]);
  for (const key of Object.keys(overrides)) {
    if (!allowed.has(key)) {
      throw new Error(
        `e2e-runtime.json cannot override frozen key: ${key}`
      );
    }
  }
  const execution = overrides.execution;
  if (execution !== undefined) {
    if (
      typeof execution !== "object" ||
      execution === null ||
      Array.isArray(execution)
    ) {
      throw new Error("e2e-runtime.json execution must be an object");
    }
  }
  validateRuntimeArmOverrides(config, overrides.armOverrides);
  if (execution === undefined) return config;
  return {
    ...config,
    execution: { ...config.execution, ...execution }
  };
}

function validateRuntimeArmOverrides(config, armOverrides) {
  if (armOverrides === undefined) return;
  if (
    typeof armOverrides !== "object" ||
    armOverrides === null ||
    Array.isArray(armOverrides)
  ) {
    throw new Error("e2e-runtime.json armOverrides must be an object");
  }
  const allowedFields = new Set(["completedTier", "parkCompleted"]);
  for (const [experimentId, experimentOverrides] of Object.entries(armOverrides)) {
    const experiment = config.experiments[experimentId];
    if (!experiment) {
      throw new Error(`e2e-runtime.json armOverrides has unknown experiment: ${experimentId}`);
    }
    if (
      typeof experimentOverrides !== "object" ||
      experimentOverrides === null ||
      Array.isArray(experimentOverrides)
    ) {
      throw new Error(
        `e2e-runtime.json armOverrides.${experimentId} must be an object`
      );
    }
    for (const [armId, override] of Object.entries(experimentOverrides)) {
      if (!experiment.arms.includes(armId)) {
        throw new Error(
          `e2e-runtime.json armOverrides.${experimentId} has unknown arm: ${armId}`
        );
      }
      if (typeof override !== "object" || override === null || Array.isArray(override)) {
        throw new Error(
          `e2e-runtime.json armOverrides.${experimentId}.${armId} must be an object`
        );
      }
      for (const [key, value] of Object.entries(override)) {
        if (!allowedFields.has(key)) {
          throw new Error(
            `e2e-runtime.json armOverrides.${experimentId}.${armId} cannot override ${key}`
          );
        }
        if (
          key === "completedTier" &&
          !["L1", "L2", "L3", "rebuild"].includes(value)
        ) {
          throw new Error(
            `e2e-runtime.json armOverrides.${experimentId}.${armId}.completedTier is invalid`
          );
        }
        if (key === "parkCompleted" && typeof value !== "boolean") {
          throw new Error(
            `e2e-runtime.json armOverrides.${experimentId}.${armId}.parkCompleted must be boolean`
          );
        }
      }
    }
  }
}

export function sha256(value) {
  return createHash("sha256").update(value).digest("hex");
}

export function modelById(models, id) {
  const model = models.models.find((candidate) => candidate.id === id);
  if (!model) {
    throw new Error(`unknown model id: ${id}`);
  }
  return model;
}

export function modelsForExperiment(loaded, experimentId) {
  const experiment = loaded.config.experiments[experimentId];
  if (!experiment) throw new Error(`unknown experiment: ${experimentId}`);
  const ids = loaded.config.modelSets[experiment.modelSet];
  if (!Array.isArray(ids) || ids.length === 0) {
    throw new Error(`experiment ${experimentId} has no model set`);
  }
  return ids.map((id) => modelById(loaded.models, id));
}

export function experimentFor(loaded, id) {
  const experiment = loaded.config.experiments[id];
  if (!experiment) throw new Error(`unknown experiment: ${id}`);
  return experiment;
}

export function armFor(loaded, id, experimentId) {
  const arm = loaded.config.arms[id];
  if (!arm) throw new Error(`unknown arm: ${id}`);
  return {
    ...arm,
    ...(loaded.runtimeArmOverrides?.[experimentId]?.[id] ?? {})
  };
}

export function effectiveMode(loaded, mode) {
  const value = loaded.config.modes[mode];
  if (!value) throw new Error(`unknown mode: ${mode}`);
  return value;
}

export function pressureSchedule(loaded, mode) {
  const modeConfig = effectiveMode(loaded, mode);
  const key = modeConfig.schedule ?? "schedule";
  const schedule = loaded.config.pressure[key];
  if (!Array.isArray(schedule)) {
    throw new Error(`pressure schedule ${key} is not defined`);
  }
  return schedule;
}

export function platformSupportsExperiment(loaded, experimentId) {
  const experiment = experimentFor(loaded, experimentId);
  const scope = experiment.platformScope;
  const entry = loaded.config.platforms[platformId()];
  if (!entry) return false;
  return entry[scope] === true;
}

export function experimentAppTabs(loaded, mode) {
  const modeConfig = effectiveMode(loaded, mode);
  return {
    appTabs: modeConfig.appTabs ?? loaded.config.layout.appTabs,
    artifexTabs:
      modeConfig.artifexTabs ?? loaded.config.layout.artifexTabs,
    pressureTabs:
      modeConfig.pressureTabs ?? loaded.config.layout.pressureTabs,
    totalTabs: modeConfig.totalTabs ?? loaded.config.layout.totalTabs
  };
}

export function modelUrlFor(model) {
  return `/models/${model.filename}`;
}

export function expectedRecordsForTrial(loaded, experimentId, mode) {
  const experiment = experimentFor(loaded, experimentId);
  const layout = experimentAppTabs(loaded, mode);
  const schedule = pressureSchedule(loaded, mode);
  const repeats = effectiveMode(loaded, mode).repeats;
  const models = modelsForExperiment(loaded, experimentId).length;
  const arms = experiment.arms.length;
  return {
    models,
    arms,
    repeats,
    pressures: schedule.length,
    appTabs: layout.appTabs,
    total: models * arms * repeats * schedule.length * layout.appTabs
  };
}
