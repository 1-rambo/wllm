/*
 * Local replacement for the WebLLM module used by the static Tavern demo.
 *
 * The evaluation server rewrites the tavern's remote `esm.run` import to
 * this file. The replacement keeps the OpenAI-compatible streaming surface
 * the app expects, but routes generation through the application adapter,
 * which in turn uses the public engine-page API. No product source is
 * imported here.
 */

function adapter() {
  return window.__wllmE2E ?? null;
}

async function waitForAdapter() {
  const started = performance.now();
  while (performance.now() - started < 10 * 60_000) {
    const value = adapter();
    if (value !== null) return value;
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  throw new Error("wllm evaluation adapter did not become available");
}

function formatMessages(messages) {
  const format = window.__wllmE2E?.config?.promptFormat ?? "qwen-chatml";
  if (format === "llama3-chat") {
    const parts = ["<|begin_of_text|>"];
    for (const message of messages ?? []) {
      parts.push(
        `<|start_header_id|>${message.role}<|end_header_id|>\n\n${message.content}<|eot_id|>`
      );
    }
    parts.push("<|start_header_id|>assistant<|end_header_id|>\n\n");
    return parts.join("");
  }
  const parts = [];
  for (const message of messages ?? []) {
    parts.push(`<|im_start|>${message.role}\n${message.content}\n<|im_end|>\n`);
  }
  parts.push("<|im_start|>assistant\n");
  return parts.join("");
}

function splitText(text) {
  const chunks = [];
  const size = 8;
  for (let index = 0; index < text.length; index += size) {
    chunks.push(text.slice(index, index + size));
  }
  if (chunks.length === 0) chunks.push("");
  return chunks;
}

export async function CreateMLCEngine(modelId, options = {}) {
  const runtime = await waitForAdapter();
  await runtime.initialize();
  if (options.initProgressCallback) {
    options.initProgressCallback({ progress: 1, text: "evaluation adapter ready" });
  }
  return {
    modelId,
    chat: {
      completions: {
        async create(request = {}) {
          const result = await runtime.generateFromMessages(request.messages ?? [], {
            requestId: `tavern-${Date.now()}`,
            promptFormat: runtime.config.promptFormat,
            maxNewTokens: request.max_tokens ?? runtime.config.maxNewTokens,
            temperature: request.temperature ?? runtime.config.temperature
          });
          if (result.error !== null) throw new Error(result.error);
          if (request.stream === false) {
            return {
              choices: [
                {
                  message: { role: "assistant", content: result.text },
                  finish_reason: result.stopReason ?? "stop"
                }
              ]
            };
          }
          const chunks = splitText(result.text);
          return (async function* stream() {
            for (const chunk of chunks) {
              yield { choices: [{ delta: { content: chunk }, finish_reason: null }] };
            }
            yield { choices: [{ delta: {}, finish_reason: result.stopReason ?? "stop" }] };
          })();
        }
      }
    }
  };
}
