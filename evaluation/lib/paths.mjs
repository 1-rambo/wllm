import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));

export const evaluationRoot = path.resolve(here, "..");
export const implRoot = path.resolve(evaluationRoot, "..");
export const configRoot = path.join(evaluationRoot, "config");
export const assetsRoot = path.join(evaluationRoot, "assets", "e2e");
export const distRoot = path.join(implRoot, "dist-demo");
export const modelsRoot = path.join(implRoot, "public", "models");
export const artifactRoot = path.join(implRoot, "artifacts", "evaluation");
export const preparedManifestPath = path.join(
  artifactRoot,
  "prepared-apps.json"
);

export function resolveImpl(...segments) {
  return path.join(implRoot, ...segments);
}

export function resolveEvaluation(...segments) {
  return path.join(evaluationRoot, ...segments);
}

export function timestampId(date = new Date()) {
  return date.toISOString().replaceAll(":", "-");
}

export function platformId() {
  return process.platform;
}

export function npmCommand(platform = process.platform) {
  return platform === "win32" ? "npm.cmd" : "npm";
}

export function nodeCommand() {
  return process.execPath;
}
