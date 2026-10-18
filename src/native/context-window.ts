/**
 * Per-slot context budgeting for the native GGUF engine.
 *
 * llama.cpp divides `n_ctx` across `n_parallel` sequences, so one session can
 * only ever use `n_ctx / n_parallel` tokens. A long-lived session carries its
 * conversation across turns, which makes the resident sequence grow every
 * turn; without a budget the run dies inside llama.cpp halfway through a
 * request, and the failure names a number nobody configured. These helpers
 * make the real per-slot budget explicit, keep the newest part of a retained
 * prefix, and let the engine refuse an impossible prompt before spending a
 * minute of units on it.
 */

export const DEFAULT_CONTEXT_RESERVE_TOKENS = 64;

export interface ContextWindowShape {
  nCtx: number;
  nParallel: number;
  // what one sequence slot can hold, as llama.cpp computes it
  perSlotTokens: number;
  // what this request may spend on its prompt once generation and safety
  // reserves are set aside
  promptBudgetTokens: number;
  reserveTokens: number;
}

export function contextWindowShape(input: {
  nCtx: number;
  nParallel: number;
  maxNewTokens: number;
  reserveTokens?: number;
}): ContextWindowShape {
  const nCtx = Math.max(1, Math.floor(input.nCtx));
  const nParallel = Math.max(1, Math.floor(input.nParallel));
  const reserveTokens = Math.max(
    0,
    Math.floor(input.reserveTokens ?? DEFAULT_CONTEXT_RESERVE_TOKENS)
  );
  const perSlotTokens = Math.max(1, Math.floor(nCtx / nParallel));
  const maxNewTokens = Math.max(0, Math.floor(input.maxNewTokens));
  return {
    nCtx,
    nParallel,
    perSlotTokens,
    promptBudgetTokens: Math.max(1, perSlotTokens - maxNewTokens - reserveTokens),
    reserveTokens,
  };
}

/**
 * Word and punctuation runs. The native tokenizer is the only exact count, but
 * this estimate is what lets the engine decide before submitting work, and it
 * tracks llama.cpp's BPE closely enough for a budget with a reserve.
 */
export function estimateTextTokens(text: string): number {
  if (text.length === 0) {
    return 0;
  }
  const chunks = text.match(/\w+|[^\s\w]/g);
  return chunks === null ? 0 : chunks.length;
}

export interface PrefixTrimResult {
  text: string;
  droppedTokens: number;
  droppedChars: number;
}

/**
 * Keep the tail of `prefix` inside `budgetTokens`.
 *
 * The tail is what a session still needs: the newest turn, its action error
 * and the observation it acts on. Trimming starts at a line boundary so the
 * retained text never begins mid-record, and the estimator is rechecked after
 * the cut because characters and tokens are not proportional.
 */
export function trimPrefixToTokenBudget(
  prefix: string,
  budgetTokens: number
): PrefixTrimResult {
  const totalTokens = estimateTextTokens(prefix);
  if (prefix.length === 0 || totalTokens <= budgetTokens) {
    return { text: prefix, droppedTokens: 0, droppedChars: 0 };
  }
  if (budgetTokens <= 0) {
    return { text: '', droppedTokens: totalTokens, droppedChars: prefix.length };
  }

  const charsPerToken = prefix.length / Math.max(1, totalTokens);
  let cut = Math.max(
    0,
    prefix.length - Math.floor(budgetTokens * charsPerToken)
  );
  const newline = prefix.indexOf('\n', cut);
  if (newline >= 0 && newline - cut <= 512) {
    cut = newline + 1;
  }

  let text = prefix.slice(cut);
  // Re-check: the ratio estimate can overshoot on text whose last turns are
  // punctuation-heavy. Ten iterations is far more than any real prompt needs.
  for (let attempt = 0; attempt < 10; attempt += 1) {
    if (estimateTextTokens(text) <= budgetTokens || text.length === 0) {
      break;
    }
    const drop = Math.max(1, Math.ceil(text.length * 0.1));
    text = text.slice(drop);
    cut += drop;
  }

  return {
    text,
    droppedTokens: Math.max(0, totalTokens - estimateTextTokens(text)),
    droppedChars: cut,
  };
}
