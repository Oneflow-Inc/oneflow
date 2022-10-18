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

#include <utility>
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

namespace python {

class Frame : public oneflow::Frame {
 public:
  Frame(PyCodeObject* code, int lineno, std::shared_ptr<Frame> back)
      : code(code), lineno(lineno), back(std::move(back)) {
    Py_INCREF(code);
  }
  OF_DISALLOW_COPY(Frame);
  ~Frame() { Py_DECREF(code); }

  PyCodeObject* const code;
  const int lineno;
  std::shared_ptr<Frame> back;

 private:
};

}  // namespace python

class PyStackGetter final : public ForeignStackGetter {
 public:
  // indended to be called in main thread.
  std::shared_ptr<Frame> GetCurrentFrame() const override { return cur_frame; }

  std::string GetFormattedStack(std::shared_ptr<const Frame> frame) const override {
    std::string buffer;
    std::shared_ptr<const python::Frame> py_frame =
        std::dynamic_pointer_cast<const python::Frame>(frame);
    // NOTE: intentionally not using CHECK_JUST here, because CHECK_JUST may also
    // call current function (StackGetter::Print) and cause infinite loop.
    // NOLINTNEXTLINE
    Singleton<ForeignLockHelper>::Get()->WithScopedAcquire([&]() -> Maybe<void> {
      while (py_frame != nullptr) {
        const auto* code_object = py_frame->code;
        const auto& lineno = py_frame->lineno;
        const char* filename =
            PyBytes_AsString(PyUnicode_AsEncodedString(code_object->co_filename, "utf-8", "~E~"));
        const char* funcname =
            PyBytes_AsString(PyUnicode_AsEncodedString(code_object->co_name, "utf-8", "~E~"));
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
        py_frame = py_frame->back;
      }
      return Maybe<void>::Ok();
    });
    return buffer;
  };
  void PushFrame(PyFrameObject* frame) {
    cur_frame = std::make_shared<python::Frame>(frame->f_code, PyFrame_GetLineNumber(frame), cur_frame);
  }
  void PopFrame() {
    if (cur_frame != nullptr) { cur_frame = cur_frame->back; }
  }
 private:
  std::shared_ptr<oneflow::python::Frame> cur_frame;
};

ONEFLOW_API_PYBIND11_MODULE("", m) {
  m.def("RegisterStackGetter", []() {
    Singleton<ForeignStackGetter>::Delete();
    Singleton<ForeignStackGetter>::SetAllocated(new PyStackGetter());
    enable_eval_frame_shim_for_current_thread();
  });
  m.def("GetCurrentStack", []() {
    auto* tmp = Singleton<ForeignStackGetter>::Get();
    return tmp->GetFormattedStack(tmp->GetCurrentFrame());
  });
}

}  // namespace oneflow

void push_frame(PyFrameObject* frame) {
  auto* stack_getter = dynamic_cast<oneflow::PyStackGetter*>(oneflow::Singleton<oneflow::ForeignStackGetter>::Get());
  stack_getter->PushFrame(frame);
}

void pop_frame() {
  auto* stack_getter = dynamic_cast<oneflow::PyStackGetter*>(oneflow::Singleton<oneflow::ForeignStackGetter>::Get());
  stack_getter->PopFrame();
}
