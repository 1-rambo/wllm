import type { SamplingParams } from './types';

// Message content types

export interface ChatCompletionMessageText {
  type: 'text';
  text: string;
}

export interface ChatCompletionMessageImage {
  type: 'image';
  data: ArrayBuffer;
}

export interface ChatCompletionMessageAudio {
  type: 'audio';
  data: ArrayBuffer;
}

export type ChatCompletionMessageContent =
  | ChatCompletionMessageText
  | ChatCompletionMessageImage
  | ChatCompletionMessageAudio;

// Tool definitions

export interface ChatCompletionToolFunctionParameters {
  type: 'object';
  properties: Record<
    string,
    {
      type: string;
      description?: string;
      enum?: string[];
      [key: string]: unknown;
    }
  >;
  required?: string[];
  additionalProperties?: boolean;
}

export interface ChatCompletionToolFunction {
  name: string;
  description?: string;
  parameters?: ChatCompletionToolFunctionParameters;
  strict?: boolean;
}

export interface ChatCompletionTool {
  type: 'function';
  function: ChatCompletionToolFunction;
}

export type ChatCompletionToolChoice =
  | 'none'
  | 'auto'
  | 'required'
  | { type: 'function'; function: { name: string } };

// Message types

export interface ChatCompletionSystemMessage {
  role: 'system';
  content: string;
  name?: string;
}

export interface ChatCompletionUserMessage {
  role: 'user';
  content: string | ChatCompletionMessageContent[];
  name?: string;
}

export interface ChatCompletionToolCall {
  id: string;
  type: 'function';
  function: {
    name: string;
    arguments: string; // JSON-encoded string
  };
}

export interface ChatCompletionAssistantMessage {
  role: 'assistant';
  content?: string | null;
  name?: string;
  tool_calls?: ChatCompletionToolCall[];
}

export interface ChatCompletionToolMessage {
  role: 'tool';
  content: string;
  tool_call_id: string;
}

export type ChatCompletionMessage =
  | ChatCompletionSystemMessage
  | ChatCompletionUserMessage
  | ChatCompletionAssistantMessage
  | ChatCompletionToolMessage;

// Request params

export type ChatCompletionParams = {
  messages: ChatCompletionMessage[];
  stream?: boolean;
  model?: string;
  abortSignal?: AbortSignal;
  // sampling
  temperature?: number;
  max_tokens?: number;
  // stop?: string | string[];
  // n?: number;
  logprobs?: boolean;
  top_logprobs?: number;
  logit_bias?: Record<string, number>;
  // tools
  tools?: ChatCompletionTool[];
  tool_choice?: ChatCompletionToolChoice;
  // parallel_tool_calls?: boolean;
  // response format
  response_format?: {
    type: 'text' | 'json_object' | 'json_schema';
    json_schema?: { name: string; schema: unknown; strict?: boolean };
  };
  // user-facing
  user?: string;
  // llama-server-specific
  chat_template_kwargs?: Record<string, any>;
  cache_prompt?: boolean;
  return_tokens?: boolean;
  timings_per_token?: boolean;
  return_progress?: boolean;
} & SamplingParams;

// Response types----------

export interface ChatCompletionLogprob {
  token: string;
  logprob: number;
  bytes: number[] | null;
}

export interface ChatCompletionLogprobsContent extends ChatCompletionLogprob {
  top_logprobs: ChatCompletionLogprob[];
}

export interface ChatCompletionChoiceLogprobs {
  content: ChatCompletionLogprobsContent[] | null;
  refusal: ChatCompletionLogprobsContent[] | null;
}

export interface ChatCompletionChoice {
  index: number;
  message: ChatCompletionAssistantMessage;
  finish_reason: 'stop' | 'length' | 'tool_calls' | 'content_filter' | null;
  logprobs: ChatCompletionChoiceLogprobs | null;
}

export interface ChatCompletionUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  prompt_tokens_details?: { cached_tokens: number; audio_tokens: number };
  completion_tokens_details?: {
    reasoning_tokens: number;
    audio_tokens: number;
    accepted_prediction_tokens: number;
    rejected_prediction_tokens: number;
  };
}

/** Response when stream=false (or omitted) */
export interface ChatCompletionResponse {
  id: string;
  object: 'chat.completion';
  created: number;
  model: string;
  choices: ChatCompletionChoice[];
  usage: ChatCompletionUsage;
  system_fingerprint?: string;
  service_tier?: string;
}

// Streaming response types

export interface ChatCompletionChunkDelta {
  role?: 'assistant';
  content?: string | null;
  tool_calls?: Array<{
    index: number;
    id?: string;
    type?: 'function';
    function?: { name?: string; arguments?: string };
  }>;
  refusal?: string | null;
}

export interface ChatCompletionChunkChoice {
  index: number;
  delta: ChatCompletionChunkDelta;
  finish_reason: 'stop' | 'length' | 'tool_calls' | 'content_filter' | null;
  logprobs: ChatCompletionChoiceLogprobs | null;
}

export interface ChatMessagePromptProgress {
  total: number;
  cache: number;
  processed: number;
  time_ms: number;
}

export interface ResultTimings {
  cache_n: number;
  prompt_n: number;
  prompt_ms: number;
  prompt_per_token_ms: number;
  prompt_per_second: number;
  predicted_n: number;
  predicted_ms: number;
  predicted_per_token_ms: number;
  predicted_per_second: number;
  /**
   * True when the native server stopped the task at a token boundary
   * because it consumed the per-unit time budget. The slot keeps its KV
   * cache, so the next bounded unit resumes from the same prefix.
   */
  bounded_stop?: boolean;
  /** Exact native budget that closed a bounded stop. */
  bounded_stop_reason?:
    | 'prefill-budget'
    | 'decode-budget'
    | 'unit-time-budget'
    | 'predict-time-budget';
}

/** Response when stream=true — one chunk per SSE event */
export interface ChatCompletionChunk {
  id: string;
  object: 'chat.completion.chunk';
  created: number;
  model: string;
  choices: ChatCompletionChunkChoice[];
  usage?: ChatCompletionUsage | null;
  timings?: ResultTimings;
  prompt_progress?: ChatMessagePromptProgress;
}

// Raw (text) completion

export type RawCompletionParams = {
  prompt: string | string[];
  stream?: boolean;
  model?: string;
  abortSignal?: AbortSignal;
  suffix?: string;
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  n?: number;
  logprobs?: number | null;
  echo?: boolean;
  stop?: string | string[];
  presence_penalty?: number;
  frequency_penalty?: number;
  best_of?: number;
  logit_bias?: Record<string, number>;
  seed?: number;
  return_progress?: boolean;
  // bounded serving: pin this request to a serving slot. The slot keeps its
  // prompt cache alive until it is explicitly exported or dropped.
  id_slot?: number;
  // bounded serving: evaluate at most this many uncached prompt tokens in
  // this request. The slot keeps the evaluated prefix for the next unit.
  n_prefill_max?: number;
  // bounded serving: decode at most this many tokens in this request, then
  // stop at a token boundary with the slot KV retained for the next unit.
  n_decode_max?: number;
  // bounded serving: maximum wall-clock time for the native task unit in
  // milliseconds. It covers prompt and decode and is checked at token
  // boundaries so the slot KV remains resumable.
  t_unit_max_ms?: number;
} & SamplingParams;

export interface RawCompletionChoice {
  text: string;
  index: number;
  finish_reason: 'stop' | 'length' | 'content_filter' | null;
  logprobs: {
    tokens: string[];
    token_logprobs: number[];
    top_logprobs: Array<Record<string, number>>;
    text_offset: number[];
  } | null;
}

/** Response when stream=false */
export interface RawCompletionResponse {
  id: string;
  object: 'text_completion';
  created: number;
  model: string;
  choices: RawCompletionChoice[];
  usage: ChatCompletionUsage;
  system_fingerprint?: string;
  timings?: ResultTimings;
}

/** One chunk when stream=true */
export interface RawCompletionChunk {
  id: string;
  object: 'text_completion';
  created: number;
  model: string;
  choices: Array<{
    text: string;
    index: number;
    finish_reason: 'stop' | 'length' | 'content_filter' | null;
    logprobs: null;
  }>;
  usage?: ChatCompletionUsage | null;
  timings?: ResultTimings;
  prompt_progress?: ChatMessagePromptProgress;
}

// Embeddings

export interface EmbeddingCreateParams {
  input: string | string[] | number[] | number[][];
  model?: string;
  encoding_format?: 'float' | 'base64';
  // dimensions?: number; // unsupported by llama.cpp
  // user?: string;
}

export interface Embedding {
  object: 'embedding';
  index: number;
  embedding: number[] | string; // float array or base64 string depending on encoding_format
}

export interface EmbeddingUsage {
  prompt_tokens: number;
  total_tokens: number;
}

export interface CreateEmbeddingResponse {
  object: 'list';
  data: Embedding[];
  model: string;
  usage: EmbeddingUsage;
}

// Reranking (NOT official OAI-compat, but is a commonly-used API schema)

export interface RerankParams {
  query: string;
  documents: string[];
  top_n?: number;
}

export interface RerankResult {
  index: number;
  relevance_score: number;
}

export interface RerankUsage {
  prompt_tokens: number;
  total_tokens: number;
}

export interface RerankResponse {
  model: string;
  object: 'list';
  usage: RerankUsage;
  results: RerankResult[];
}
