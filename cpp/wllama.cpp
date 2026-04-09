#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <stdio.h>
#include <atomic>
#include <chrono>

#include <stdlib.h>
#include <unistd.h>

#ifdef __EMSCRIPTEN__
#include <malloc.h>
#include <emscripten/emscripten.h>
#endif

// #define GLUE_DEBUG(...) fprintf(stderr, "@@ERROR@@" __VA_ARGS__)

#include "llama.h"
#include "helpers/wcommon.h"
#include "actions.hpp"

#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define WLLAMA_ACTION(name)                 \
  else if (action == #name)                 \
  {                                         \
    auto res = action_##name(app, req_raw); \
    res.handler.serialize(output_buffer);   \
  }

static void llama_log_callback_logTee(ggml_log_level level, const char *text, void *user_data)
{
  (void)user_data;
  const char *lvl = "@@DEBUG";
  size_t len = strlen(text);
  if (len == 0 || text[len - 1] != '\n')
  {
    // do not print if the line does not terminate with \n
    return;
  }
  if (level == GGML_LOG_LEVEL_ERROR)
  {
    lvl = "@@ERROR";
  }
  else if (level == GGML_LOG_LEVEL_WARN)
  {
    lvl = "@@WARN";
  }
  else if (level == GGML_LOG_LEVEL_INFO)
  {
    lvl = "@@INFO";
  }
  fprintf(stderr, "%s@@%s", lvl, text);
}

static void printStr(ggml_log_level level, const char *text)
{
  std::string str = std::string(text) + "\n";
  llama_log_callback_logTee(level, str.c_str(), nullptr);
}

static glue_outbuf output_buffer;
static app_t app;
static std::atomic<uint64_t> g_action_trace_seq{1};
static std::string g_last_action_error;

static int64_t action_trace_now_ms()
{
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

static bool action_trace_is_key_action(const std::string &action)
{
  return action == "chat_start" ||
         action == "chat_finish" ||
         action == "decode" ||
         action == "sampling_sample";
}

static std::vector<char> input_buffer;
// second argument is dummy
extern "C" const char *wllama_malloc(size_t size, uint32_t)
{
  if (input_buffer.size() < size)
  {
    input_buffer.resize(size);
  }
  return input_buffer.data();
}

extern "C" const char *wllama_start()
{
  try
  {
    g_last_action_error.clear();
    llama_backend_init();
    // std::cerr << llama_print_system_info() << "\n";
    llama_log_set(llama_log_callback_logTee, nullptr);
    std::cerr << "[WLLAMA_TRACE_SCHEMA] version=2026-04-03-decode-enter-return-v2" << std::endl;
    wllama_malloc(1024, 0);
    return "{\"success\":true}";
  }
  catch (std::exception &e)
  {
    printStr(GGML_LOG_LEVEL_ERROR, e.what());
    return "{\"error\":true}";
  }
}

extern "C" const char *wllama_get_last_error()
{
  return g_last_action_error.c_str();
}

extern "C" const char *wllama_action(const char *name, const char *req_raw)
{
  const uint64_t trace_id = g_action_trace_seq.fetch_add(1);
  const int64_t started_at_ms = action_trace_now_ms();
  std::string action = name ? std::string(name) : std::string();
  uint32_t *output_len = reinterpret_cast<uint32_t *>(const_cast<char *>(req_raw));
  const bool trace_key_action = action_trace_is_key_action(action);
  const int64_t slow_action_threshold_ms = 100;

  if (trace_key_action)
  {
    std::cerr << "[WLLAMA_ACTION_TRACE] id=" << trace_id
              << " phase=entry action=" << action << std::endl;
  }

  try
  {
    g_last_action_error.clear();
    if (action.empty())
    {
      printStr(GGML_LOG_LEVEL_ERROR, "Empty action");
      abort();
    }

    WLLAMA_ACTION(load)
    WLLAMA_ACTION(set_options)
    WLLAMA_ACTION(sampling_init)
    WLLAMA_ACTION(sampling_sample)
    WLLAMA_ACTION(sampling_accept)
    WLLAMA_ACTION(get_vocab)
    WLLAMA_ACTION(lookup_token)
    WLLAMA_ACTION(tokenize)
    WLLAMA_ACTION(detokenize)
    WLLAMA_ACTION(decode)
    WLLAMA_ACTION(encode)
    WLLAMA_ACTION(get_logits)
    WLLAMA_ACTION(embeddings)
    WLLAMA_ACTION(chat_format)
    WLLAMA_ACTION(kv_remove)
    WLLAMA_ACTION(kv_clear)
    else if (action == "chat_init")
    {
      auto res = action_tree_init(app, req_raw);
      res.handler.serialize(output_buffer);
    }
    else if (action == "chat_state")
    {
      auto res = action_tree_state(app, req_raw);
      res.handler.serialize(output_buffer);
    }
    else if (action == "chat_set_active")
    {
      auto res = action_tree_switch(app, req_raw);
      res.handler.serialize(output_buffer);
    }
    else if (action == "chat_delete")
    {
      auto res = action_tree_delete(app, req_raw);
      res.handler.serialize(output_buffer);
    }
    else if (action == "chat_reset")
    {
      auto res = action_tree_reset(app, req_raw);
      res.handler.serialize(output_buffer);
    }
    else if (action == "chat_start")
    {
      auto res = action_tree_chat_start(app, req_raw);
      res.handler.serialize(output_buffer);
    }
    else if (action == "chat_finish")
    {
      auto res = action_tree_chat_finish(app, req_raw);
      res.handler.serialize(output_buffer);
    }
    WLLAMA_ACTION(current_status)
    WLLAMA_ACTION(perf_context)
    WLLAMA_ACTION(perf_reset)
    // WLLAMA_ACTION(session_save)
    // WLLAMA_ACTION(session_load)
    WLLAMA_ACTION(test_benchmark)
    WLLAMA_ACTION(test_perplexity)

    else
    {
      printStr(GGML_LOG_LEVEL_ERROR, (std::string("Unknown action: ") + name).c_str());
      abort();
    }

    // length of response is written inside input_buffer
    if (output_len != nullptr)
    {
      output_len[0] = static_cast<uint32_t>(output_buffer.data.size());
    }

    const char *output_ptr = output_buffer.data.data();
    if (output_ptr == nullptr || output_buffer.data.empty())
    {
      g_last_action_error = std::string("action=") + action + " err=empty_output_buffer";
      if (output_len != nullptr)
      {
        output_len[0] = 0;
      }
      std::cerr << "[WLLAMA_ACTION_TRACE] id=" << trace_id
                << " phase=error action=" << action
                << " elapsedMs=" << (action_trace_now_ms() - started_at_ms)
                << " err=empty_output_buffer" << std::endl;
      return nullptr;
    }

    const int64_t elapsed_ms = action_trace_now_ms() - started_at_ms;

    if (trace_key_action || elapsed_ms >= slow_action_threshold_ms)
    {
      std::cerr << "[WLLAMA_ACTION_TRACE] id=" << trace_id
                << " phase=exit action=" << action
                << " elapsedMs=" << elapsed_ms
                << " outBytes=" << output_buffer.data.size() << std::endl;
    }
    return output_ptr;
  }
  catch (std::exception &e)
  {
    g_last_action_error = std::string("action=") + action + " err=" + e.what();
    if (output_len != nullptr)
    {
      output_len[0] = 0;
    }
    std::cerr << "[WLLAMA_ACTION_TRACE] id=" << trace_id
              << " phase=error action=" << action
              << " elapsedMs=" << (action_trace_now_ms() - started_at_ms)
              << " err=" << e.what() << std::endl;
    printStr(GGML_LOG_LEVEL_ERROR, e.what());
    return nullptr;
  }
  catch (...)
  {
    g_last_action_error = std::string("action=") + action + " err=unknown";
    if (output_len != nullptr)
    {
      output_len[0] = 0;
    }
    std::cerr << "[WLLAMA_ACTION_TRACE] id=" << trace_id
              << " phase=error action=" << action
              << " elapsedMs=" << (action_trace_now_ms() - started_at_ms)
              << " err=unknown" << std::endl;
    printStr(GGML_LOG_LEVEL_ERROR, "Unknown error in wllama_action");
    return nullptr;
  }
}

extern "C" const char *wllama_exit()
{
  try
  {
    free_all(app);
    llama_backend_free();
    return "{\"success\":true}";
  }
  catch (std::exception &e)
  {
    printStr(GGML_LOG_LEVEL_ERROR, e.what());
    return "{\"error\":true}";
  }
}

extern "C" const char *wllama_debug()
{
  auto get_mem_total = [&]()
  {
#ifdef __EMSCRIPTEN__
    return EM_ASM_INT(return HEAP8.length);
#else
    return 0;
#endif
  };
  auto get_mem_free = [&]()
  {
#ifdef __EMSCRIPTEN__
    auto i = mallinfo();
    unsigned int total_mem = get_mem_total();
    unsigned int dynamic_top = (unsigned int)sbrk(0);
    return total_mem - dynamic_top + i.fordblks;
#else
    return 0;
#endif
  };
  /*json res = json{
      {"mem_total_MB", get_mem_total() / 1024 / 1024},
      {"mem_free_MB", get_mem_free() / 1024 / 1024},
      {"mem_used_MB", (get_mem_total() - get_mem_free()) / 1024 / 1024},
  };
  result = std::string(res.dump());
  return result.c_str();*/
  return nullptr;
}

int main()
{
  std::cerr << "Unused\n";
  return 0;
}
