/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
extern "C" {
#include "oneflow/api/python/stack_getter.h"
}

#include <pybind11/pybind11.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/common/env_var/debug_mode.h"
#include "oneflow/core/common/singleton.h"
#include "oneflow/core/framework/shut_down_util.h"
#include "oneflow/core/common/foreign_lock_helper.h"
#include "oneflow/core/common/foreign_stack_getter.h"
#include "fmt/core.h"
#include "fmt/color.h"
#include "fmt/ostream.h"

namespace py = pybind11;

namespace oneflow {

const int kDefaultStackSize = 15;
using Stack = small_vector<std::pair<PyCodeObject*, int>, kDefaultStackSize>;

class PyStackGetter final : public ForeignStackGetter {
 public:
  // indended to be called in main thread.
  std::string GetCurrentStack(size_t max_size) const override {
    if (IsShuttingDown()) { return ""; }

    Stack stack;
    PyFrameObject* frame = PyEval_GetFrame();
    CHECK_NOTNULL(frame);
    for (int i = 0; frame != nullptr && i < max_size; frame = frame->f_back, ++i) {
      int lineno = PyFrame_GetLineNumber(frame);
      // hold the PyCodeObject instead of the PyFrameObject because the latter holds
      // references to the local objects in the frame.
      stack.emplace_back(frame->f_code, lineno);
    }
    return GetFormattedStack(stack);
  }

  // "happy path", performance is important.
  // indended to be called in main thread.
  void RecordCurrentStack(StackId id) override {
    if (IsShuttingDown()) { return; }

    std::lock_guard<std::mutex> lock(mutex_);
    auto& id2stack = id2stack_arr_[index_];
    id2stack.first = id;
    for (auto& f_code_and_line_no : id2stack.second) { Py_DECREF(f_code_and_line_no.first); }
    id2stack.second.clear();
    for (PyFrameObject* frame = CHECK_NOTNULL(PyEval_GetFrame());
         frame != nullptr && (id2stack.second.size() < kDefaultStackSize || IsInDebugMode());
         frame = frame->f_back) {
      int lineno = PyFrame_GetLineNumber(frame);
      // hold the PyCodeObject instead of the PyFrameObject because the latter holds
      // references to the local objects in the frame.
      id2stack.second.emplace_back(frame->f_code, lineno);
      Py_INCREF(frame->f_code);
    }
    index_ = (index_ + 1) % id2stack_arr_.max_size();
  }

  // "bad path", performance is not important.
  std::string GetFormatted(StackId id) const override {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string result = "  <unknown>\n";
    for (const auto& pair : id2stack_arr_) {
      if (pair.first == id) {
        const auto& stack = pair.second;
        result = GetFormattedStack(stack);
        if (stack.size() == kDefaultStackSize && !IsInDebugMode()) {
          fmt::format_to(
              std::back_inserter(result),
              "The Python stack may be truncated, you might want to enable the debug mode (by "
              "setting the environment variable `{}` to 1) and get the full stack.",
              fmt::styled("ONEFLOW_DEBUG", fmt::emphasis::bold | fmt::fg(fmt::color::dark_gray)));
        }
        break;
      }
    }
    return result;
  }

 private:
  std::string GetFormattedStack(const Stack& stack) const {
    std::string buffer;
    // NOTE: intentionally not using CHECK_JUST here, because CHECK_JUST may also
    // call current function (StackGetter::Print) and cause infinite loop.
    // NOLINTNEXTLINE
    Singleton<ForeignLockHelper>::Get()->WithScopedAcquire([&]() -> Maybe<void> {
      for (const auto& pair : stack) {
        PyCodeObject* f_code = pair.first;
        int lineno = pair.second;
        const std::string filename =
            PyBytes_AS_STRING(PyUnicode_AsEncodedString(f_code->co_filename, "utf-8", "~E~"));
        const std::string funcname =
            PyBytes_AS_STRING(PyUnicode_AsEncodedString(f_code->co_name, "utf-8", "~E~"));
        const std::string line_text = [&]() -> std::string {
          std::string line_text;
          std::ifstream ifs(filename);
          if (!ifs.is_open()) { return "<unknown>"; }
          for (int j = 0; j < lineno; ++j) { std::getline(ifs, line_text); }
          line_text.erase(line_text.find_last_not_of(' ') + 1);  // suffixing spaces
          line_text.erase(0, line_text.find_first_not_of(' '));  // prefixing spaces
          return line_text;
        }();
        // immitate python's stack trace format
        fmt::format_to(std::back_inserter(buffer), "  File \"{}\", line {}, in {}\n    {}\n",
                       filename, lineno, funcname, line_text);
      }
      return Maybe<void>::Ok();
    });
    return buffer;
  };
  std::array<std::pair<StackId, Stack>, 20000> id2stack_arr_;
  int64_t index_ = 0;
  mutable std::mutex mutex_;
};

ONEFLOW_API_PYBIND11_MODULE("", m) {
  m.def("RegisterStackGetter", []() {
    Singleton<ForeignStackGetter>::Delete();
    Singleton<ForeignStackGetter>::SetAllocated(new PyStackGetter());
    enable_eval_frame_shim_for_current_thread();
  });
  m.def("GetCurrentStack", [](size_t max_size) {
    return Singleton<ForeignStackGetter>::Get()->GetCurrentStack(max_size);
  });
}

}  // namespace oneflow