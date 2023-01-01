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

#include <utility>

#include "fmt/core.h"
#include "fmt/color.h"
#include "fmt/ostream.h"
#include "pybind11/pybind11.h"

#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/common/env_var/debug_mode.h"
#include "oneflow/core/common/singleton.h"
#include "oneflow/core/framework/shut_down_util.h"
#include "oneflow/core/common/foreign_lock_helper.h"
#include "oneflow/extension/stack/foreign_stack_getter.h"
#include "oneflow/extension/stack/python/custom_eval_frame.h"

namespace py = pybind11;

namespace oneflow {

namespace {
std::string RawPyUnicodeToStdString(const PyObject* py_str) {
  return PyBytes_AsString(PyUnicode_AsEncodedString(const_cast<PyObject*>(py_str), "utf-8", "~E~"));
}
static constexpr auto* PyUnicodeToStdString =
    DECORATE(&RawPyUnicodeToStdString, ThreadLocalCachedCopiable);
}  // namespace

class PyFrame final
    : public Frame,
      public intrusive::EnableObjectPool<PyFrame, intrusive::kThreadUnsafeAndDisableDestruct> {
 public:
  PyFrame() : EnableObjectPool(), lineno(0) {}
  void __Init__(PyCodeObject* code, int lineno, intrusive::shared_ptr<PyFrame> back) {
    this->filename = PyUnicodeToStdString(code->co_filename);
    this->funcname = PyUnicodeToStdString(code->co_name);
    this->lineno = lineno;
    this->back = std::move(back);
  }
  OF_DISALLOW_COPY_AND_MOVE(PyFrame);
  ~PyFrame() = default;

  std::string filename;
  std::string funcname;
  int lineno;
  intrusive::shared_ptr<PyFrame> back;
};

class PyStackGetter final : public ForeignStackGetter {
 public:
  PyStackGetter() {
    auto* frame = PyEval_GetFrame();
    // Get the first frame. It assumes `import oneflow` is called in global scope,
    while (frame->f_back != nullptr) { frame = frame->f_back; }
    current_frame_ = object_pool_.make_shared(frame->f_code, 0, nullptr);
  }
  // indended to be called in main thread.
  intrusive::shared_ptr<Frame> GetCurrentFrame() const override {
    if (IsShuttingDown() || !current_frame_) { return nullptr; }
    // See `RecordAndEvalFrame` for documentation.
    current_frame_->lineno = PyFrame_GetLineNumber(PyEval_GetFrame());
    return intrusive::shared_ptr<Frame>(current_frame_.get());
  }

  std::string GetFormattedStack(intrusive::shared_ptr<Frame> frame) const override {
    if (frame == nullptr) { return "  <unknown>\n"; }
    std::string buffer;
    const auto* py_frame = dynamic_cast<const PyFrame*>(frame.get());
    py::gil_scoped_acquire acquire;
    while (py_frame != nullptr) {
      const auto& lineno = py_frame->lineno;
      const std::string line_text = [&]() -> std::string {
        std::string line_text;
        std::ifstream ifs(py_frame->filename);
        if (!ifs.is_open()) { return "<unknown>"; }
        for (int j = 0; j < lineno; ++j) { std::getline(ifs, line_text); }
        line_text.erase(line_text.find_last_not_of(' ') + 1);  // suffixing spaces
        line_text.erase(0, line_text.find_first_not_of(' '));  // prefixing spaces
        return line_text;
      }();
      // immitate python's stack trace format
      fmt::format_to(std::back_inserter(buffer), "  File \"{}\", line {}, in {}\n    {}\n",
                     py_frame->filename, lineno, py_frame->funcname, line_text);
      py_frame = py_frame->back.get();
    }
    return buffer;
  };

#if PY_VERSION_HEX >= 0x03090000
  PyObject* RecordAndEvalFrame(PyThreadState* tstate, PyFrameObject* frame,
#else
  PyObject* RecordAndEvalFrame(PyFrameObject* frame,
#endif
                               int throw_flag) {
    // Example:
    // >> def f(): # Line 1
    // >>   pass   # Line 2
    // >> f()      # Line 3
    //
    // When we call f(), `RecordAndEvalFrame` is triggered and the `frame`
    // argument is the frame of function `f`, which is Line 1 at that time. It is not
    // useful to us, but we can adjust it in `GetCurrentFrame` method.
    //
    PushFrame(frame);
#if PY_VERSION_HEX >= 0x03090000
    if (tstate == NULL) { tstate = PyThreadState_GET(); }
    PyObject* ret = _PyEval_EvalFrameDefault(tstate, frame, throw_flag);
#else
    PyObject* ret = _PyEval_EvalFrameDefault(frame, throw_flag);
#endif
    PopFrame();
    return ret;
  }

 private:
  intrusive::shared_ptr<PyFrame> current_frame_;

  intrusive::ObjectPool<PyFrame, intrusive::kThreadUnsafeAndDisableDestruct> object_pool_;

  void PushFrame(PyFrameObject* frame) {
    if (auto* f = frame->f_back) { current_frame_->lineno = PyFrame_GetLineNumber(f); }
    current_frame_ = object_pool_.make_shared(frame->f_code, 0, current_frame_);
  }
  void PopFrame() {
    CHECK_NOTNULL(current_frame_);
    current_frame_ = current_frame_->back;
  }
};

#if PY_VERSION_HEX >= 0x03090000
PyObject* RecordAndEvalFrame(PyThreadState* tstate, PyFrameObject* frame,
#else
PyObject* RecordAndEvalFrame(PyFrameObject* frame,
#endif
                             int throw_flag) {
  using namespace oneflow;
  return dynamic_cast<PyStackGetter*>(Singleton<ForeignStackGetter>::Get())
#if PY_VERSION_HEX >= 0x03090000
      ->RecordAndEvalFrame(tstate, frame, throw_flag);
#else
      ->RecordAndEvalFrame(frame, throw_flag);
#endif
}

void RegisterPyStackGetter() {
  if (!IsPythonStackGetterEnabled()) { return; }
  Singleton<ForeignStackGetter>::Delete();
  Singleton<ForeignStackGetter>::SetAllocated(new PyStackGetter());
  EnableCustomEvalFrameForCurrentThread(&RecordAndEvalFrame);
}

}  // namespace oneflow
