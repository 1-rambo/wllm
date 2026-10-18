import { openAppPage, closePageQuietly } from "./browser.mjs";

export async function createJudgeRunner(options) {
  const page = await openAppPage(
    options.context,
    judgeUrl(options.baseUrl, options.modelUrl, options.config),
    { timeoutMs: options.timeoutMs ?? 300_000 }
  );
  await page.waitForFunction(
    () => typeof window.__wllmEnginePage?.prewarm === "function",
    null,
    { timeout: options.timeoutMs ?? 300_000 }
  );
  const engine = await page.evaluate(() => window.__wllmEnginePage);
  await page.evaluate(async () => {
    await window.__wllmEnginePage.prewarm();
    await window.__wllmEnginePage.createSession({ sessionId: "judge" });
  });
  return {
    page,
    engine,
    async score(record) {
      return scoreOne(page, record, options.config);
    },
    async close() {
      await closePageQuietly(page);
    }
  };
}

export async function scoreRecords(runner, records, options = {}) {
  const eligible = records.filter(
    (record) =>
      (record.kind === "task" || record.kind === "recovery") &&
      record.error === null &&
      typeof record.text === "string" &&
      record.text.trim() !== ""
  );
  const fraction = Math.max(0, Math.min(1, options.sampleFraction ?? 0.35));
  const limit = Math.max(1, Math.ceil(eligible.length * fraction));
  const sampled = eligible.slice(0, limit);
  const scored = new Map();
  for (const [index, record] of sampled.entries()) {
    const result = await runner.score(record);
    scored.set(recordKey(record), result);
    if (typeof options.onProgress === "function") {
      options.onProgress(index + 1, sampled.length, record, result);
    }
  }
  return records.map((record) => {
    const result = scored.get(recordKey(record));
    if (!result) return record;
    return {
      ...record,
      judgeScore: result.score,
      judgeValid: result.valid,
      judgeRaw: result.raw,
      judgeError: result.error,
      judgeFallbackUsed: result.fallbackUsed === true,
      judgePrimaryRaw: result.primaryRaw ?? null
    };
  });
}


async function scoreOne(page, record, config) {
  try {
    const primary = await completeJudgePrompt(page, primaryPrompt(record), config);
    const primaryResult = parseJudgeResult(primary);
    if (primaryResult.valid || primaryResult.raw.trim() !== "") {
      return primaryResult;
    }

    // Some deterministic prompt/response combinations make the judge model
    // emit only EOS. Retry once with the same rubric, request, response, and
    // scoring instruction in a different field order. The primary prompt is
    // unchanged for every response that produces any output.
    const fallback = await completeJudgePrompt(
      page,
      fallbackPrompt(record),
      config
    );
    return {
      ...parseJudgeResult(fallback),
      fallbackUsed: true,
      primaryRaw: primaryResult.raw
    };
  } catch (error) {
    return {
      valid: false,
      score: null,
      raw: "",
      error: error instanceof Error ? error.message : String(error)
    };
  }
}

function primaryPrompt(record) {
  return [
    "You are a blind evaluator for a browser application task.",
    "Do not reward verbosity. A response that breaks the requested setting or persona is invalid.",
    "",
    `Rubric: ${record.rubric ?? "Follow the user request in the application context."}`,
    `User request: ${record.prompt ?? ""}`,
    `Assistant response: ${record.text ?? ""}`,
    "",
    // The instruction sits after the response on purpose. A completion model
    // handed a prompt that ends in an unfinished assistant sentence keeps
    // writing that sentence instead of evaluating it, which is what made the
    // earlier judge invalid on long in-character responses.
    "Evaluate the assistant response above on a scale from 0 to 10.",
    "Reply with exactly two lines and nothing else:",
    "SCORE: <number>",
    "REASON: <short reason>"
  ].join("\n");
}

function fallbackPrompt(record) {
  return [
    "You are a blind evaluator for a browser application task.",
    "Do not reward verbosity. A response that breaks the requested setting or persona is invalid.",
    "",
    `Assistant response: ${record.text ?? ""}`,
    `User request: ${record.prompt ?? ""}`,
    `Rubric: ${record.rubric ?? "Follow the user request in the application context."}`,
    "",
    "Evaluate the assistant response above on a scale from 0 to 10.",
    "Reply with exactly two lines and nothing else:",
    "SCORE: <number>",
    "REASON: <short reason>"
  ].join("\n");
}

async function completeJudgePrompt(page, promptText, config) {
  return page.evaluate(
    async ({ promptText, maxNewTokens, temperature, seed }) => {
      return window.__wllmEnginePage.generate("judge", {
        requestId: `judge-${Date.now()}-${Math.random().toString(36).slice(2)}`,
        promptText,
        maxNewTokens,
        temperature,
        seed
      });
    },
    {
      promptText,
      maxNewTokens: config?.maxNewTokens ?? 96,
      temperature: config?.temperature ?? 0,
      seed: config?.seed ?? 20261015
    }
  );
}

function parseJudgeResult(result) {
  // Accept the plain `SCORE: 8` line and the markdown `**SCORE:** 8` the
  // model sometimes emits; both state the score explicitly.
  const match = /SCORE\s*:?\s*\*{0,2}\s*(-?\d+(?:\.\d+)?)/i.exec(
    result.text ?? ""
  );
  const score = match ? Number(match[1]) : null;
  return {
    valid: score !== null && Number.isFinite(score) && score >= 0 && score <= 10,
    score,
    raw: result.text ?? "",
    error: null
  };
}

function judgeUrl(baseUrl, modelUrl, config) {
  const url = new URL("/engines/wllama.html", baseUrl);
  url.searchParams.set("modelUrl", new URL(modelUrl, baseUrl).toString());
  url.searchParams.set("nCtx", "2048");
  url.searchParams.set("nBatch", "128");
  url.searchParams.set("nParallel", "1");
  url.searchParams.set("temperature", String(config?.temperature ?? 0));
  url.searchParams.set("seed", String(config?.seed ?? 20261015));
  url.searchParams.set("frameMonitor", "0");
  // Independent judgements must not inherit each other's prompt cache: the
  // wllama page caches the prompt prefix across calls by default, and a
  // stale prefix is what let an earlier judge continue the previous
  // response instead of scoring it.
  url.searchParams.set("cachePrompt", "0");
  url.searchParams.set("modelLoadMode", "direct");
  return url.toString();
}

function recordKey(record) {
  return [
    record.arm,
    record.model,
    record.platform,
    record.pressureId,
    record.repeat,
    record.app,
    record.taskId,
    record.role,
    record.phase
  ].join("|");
}
