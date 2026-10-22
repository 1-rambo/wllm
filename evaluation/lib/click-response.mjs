import { seededRandom } from "./statistics.mjs";

/**
 * Drive a deterministic randomized visit order over active app tabs.
 * Clicks never repeat the previous tab while another active tab remains.
 */
export function createClickSequencer(options) {
  const indices = [...new Set(options.indices.map((value) => Math.trunc(value)))]
    .filter((value) => value >= 0)
    .sort((left, right) => left - right);
  const random = seededRandom(options.seed);
  const maxClicks = Math.max(0, Math.trunc(options.maxClicks));
  const minIntervalMs = Math.max(0, Math.trunc(options.minIntervalMs));
  const maxIntervalMs = Math.max(minIntervalMs, Math.trunc(options.maxIntervalMs));
  let pool = [];
  let last = null;
  let clicks = 0;

  return {
    initialDelayMs: Math.max(0, Math.trunc(options.initialDelayMs)),

    next(activeIndices) {
      if (clicks >= maxClicks) return null;
      const allowed = new Set(indices);
      const active = [...new Set(activeIndices.map((value) => Math.trunc(value)))]
        .filter((value) => allowed.has(value))
        .sort((left, right) => left - right);
      if (active.length === 0) return null;

      const activeSet = new Set(active);
      pool = pool.filter((value) => activeSet.has(value) && value !== last);
      if (pool.length === 0) {
        const candidates = active.filter((value) => value !== last);
        if (candidates.length === 0) return null;
        pool = shuffle(candidates, random);
      }

      last = pool.shift() ?? null;
      if (last === null) return null;
      clicks += 1;
      return last;
    },

    intervalMs() {
      return minIntervalMs + Math.floor(random() * (maxIntervalMs - minIntervalMs + 1));
    },

    get count() {
      return clicks;
    }
  };
}

export function clickResponseSeed(baseSeed, trialOrdinal, modelId) {
  // Deliberately exclude the arm: both arms see the same planned visit
  // order for a model/trial whenever the same tabs remain active.
  let hash = 2166136261;
  const value = `${modelId}|${trialOrdinal}`;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (Math.trunc(baseSeed) ^ hash) >>> 0;
}

export function isValidClickResponse(sample) {
  // Playwright Chrome can report every page in a window as visible even
  // after a real bringToFront activation. Keep visibility as diagnostics and
  // validate the actual activation path by the requested frame count.
  return (
    sample?.error == null &&
    Number.isFinite(sample.framesWaited) &&
    Number.isFinite(sample.responseFrames) &&
    sample.framesWaited >= sample.responseFrames &&
    Number.isFinite(sample.clickResponseMs)
  );
}

function shuffle(values, random) {
  const output = [...values];
  for (let index = output.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(random() * (index + 1));
    [output[index], output[swapIndex]] = [output[swapIndex], output[index]];
  }
  return output;
}
