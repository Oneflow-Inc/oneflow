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

#include "oneflow/extension/stack/python/stack_getter.h"

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
#include "oneflow/core/common/env_var/debug_mode.h"
#include "oneflow/core/job/graph_scope_vars.h"
#include "oneflow/extension/stack/foreign_stack_getter.h"
#include "oneflow/extension/stack/python/custom_eval_frame.h"

namespace py = pybind11;

namespace oneflow {

namespace {
std::string PyUnicodeToStdString(const PyObject* py_str) {
  return PyBytes_AsString(PyUnicode_AsEncodedString(const_cast<PyObject*>(py_str), "utf-8", "~E~"));
}
}  // namespace

class PyFrame final : public Frame {
 public:
  // There is no need to increase the reference count of these cpython objects
  // because they must be alive during the lifetime of `PyFrame`.
  PyFrame(PyFrameObject* frame, std::shared_ptr<PyFrame> back)
      : filename(frame->f_code->co_filename),
        funcname(frame->f_code->co_name),
        cpython_frame(frame),
        lineno(0),
        back(std::move(back)) {}
  ~PyFrame() = default;
  OF_DISALLOW_COPY_AND_MOVE(PyFrame);

  PyObject* filename;
  PyObject* funcname;
  PyFrameObject* cpython_frame;
  int lineno;
  std::shared_ptr<PyFrame> back;
};

class PyStackGetter final : public ForeignStackGetter {
 public:
  PyStackGetter() {
    auto* frame = PyEval_GetFrame();
    // Get the first frame. It assumes `import oneflow` is called in global scope,
    while (frame->f_back != nullptr) { frame = frame->f_back; }
    current_frame_ = std::make_shared<PyFrame>(frame, nullptr);
  }
  // indended to be called in main thread.
  std::shared_ptr<Frame> GetCurrentFrame() const override {
    if (IsShuttingDown() || !current_frame_) { return nullptr; }
    // See `RecordAndEvalFrame` for documentation.
    current_frame_->lineno = PyFrame_GetLineNumber(current_frame_->cpython_frame);
    return current_frame_;
  }

  // bad path, performance is not a concern.
  std::string GetFormattedStack(std::shared_ptr<Frame> frame) const override {
    if (frame == nullptr) { return "  <unknown>\n"; }
    std::string buffer;
    const auto* py_frame = dynamic_cast<const PyFrame*>(frame.get());
    py::gil_scoped_acquire acquire;
    while (py_frame != nullptr) {
      const auto& lineno = py_frame->lineno;
      const std::string line_text = [&]() -> std::string {
        std::string line_text;
        std::ifstream ifs(PyUnicodeToStdString(py_frame->filename));
        if (!ifs.is_open()) { return "<unknown>"; }
        for (int j = 0; j < lineno; ++j) { std::getline(ifs, line_text); }
        line_text.erase(line_text.find_last_not_of(' ') + 1);  // suffixing spaces
        line_text.erase(0, line_text.find_first_not_of(' '));  // prefixing spaces
        return line_text;
      }();
      // immitate python's stack trace format
      fmt::format_to(std::back_inserter(buffer), "  File \"{}\", line {}, in {}\n    {}\n",
                     PyUnicodeToStdString(py_frame->filename), lineno,
                     PyUnicodeToStdString(py_frame->funcname), line_text);
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
  std::shared_ptr<PyFrame> current_frame_;

  void PushFrame(PyFrameObject* frame) {
    if (auto* f = frame->f_back) { current_frame_->lineno = PyFrame_GetLineNumber(f); }
    current_frame_ = std::make_shared<PyFrame>(frame, current_frame_);
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

namespace {

// get a formatted stack frame representation
std::string get_python_frame_str_repr(PyFrameObject* frame) {
  if (frame == NULL) return "";
  std::string buffer;
  PyCodeObject* code = frame->f_code;
  std::string file_name = PyUnicodeToStdString(code->co_filename);
  std::string code_name = PyUnicodeToStdString(code->co_name);
  int line_number = PyFrame_GetLineNumber(frame);

  fmt::format_to(std::back_inserter(buffer), "File \"{}\", line {}, in {}", file_name, line_number,
                 code_name);

  std::string line_text;
  const bool debug_mode = GetGraphDebugMode() || IsInDebugMode();
  if (debug_mode) {
    const auto& GetCurSrc = [&file_name, line_number]() -> std::string {
      std::string line_text;
      std::ifstream ifs(file_name);
      if (!ifs.is_open()) { return "<unknown>"; }
      for (int j = 0; j < line_number; ++j) { std::getline(ifs, line_text); }
      line_text.erase(line_text.find_last_not_of(' ') + 1);  // suffixing spaces
      line_text.erase(0, line_text.find_first_not_of(' '));  // prefixing spaces
      return line_text;
    };
    line_text = GetCurSrc();
    buffer += ", source < " + line_text + " >; ";
  } else {
    buffer += "; ";
  }

  return buffer;
}

bool check_if_python_file_should_be_filtered(const std::string& path) {
  const auto& paths_to_be_kept = GetPythonPathsToBeKeptForDebugging();
  for (int i = 0; i < paths_to_be_kept.size(); ++i) {
    const std::string& path_to_be_kept = paths_to_be_kept[i];
    if (path.size() > path_to_be_kept.size()) {
      if (path.substr(0, path_to_be_kept.size()) == path_to_be_kept) { return false; }
    }
  }

  const auto& paths_to_be_filtered = GetPythonPathsToBeFilteredForDebugging();
  for (int i = 0; i < paths_to_be_filtered.size(); ++i) {
    const std::string& path_to_be_filtered = paths_to_be_filtered[i];
    if (path.size() > path_to_be_filtered.size()) {
      if (path.substr(0, path_to_be_filtered.size()) == path_to_be_filtered) { return true; }
    }
  }

  return false;
}

bool check_if_frame_should_be_filtered(PyFrameObject* frame) {
  std::string frame_file_name = PyUnicodeToStdString(frame->f_code->co_filename);
  return check_if_python_file_should_be_filtered(frame_file_name);
}

bool check_if_should_skip_this_frame(PyFrameObject* frame) {
  const bool only_user_py_stack = GetGraphDebugOnlyUserPyStack();
  if (only_user_py_stack) { return check_if_frame_should_be_filtered(frame); }
  return false;
}

int32_t get_cur_stack_depth() {
  int32_t current_stack_depth = 0;
  PyFrameObject* f = PyEval_GetFrame();
  while (f) {
    if (check_if_should_skip_this_frame(f)) {
      f = f->f_back;
      continue;
    }

    current_stack_depth++;
    f = f->f_back;
  }
  return current_stack_depth;
}

std::string get_cur_frame_stack_str() {
  const int32_t max_stack_depth = GetGraphDebugMaxPyStackDepth();
  std::string cur_f_str;
  PyFrameObject* cur_frame = PyEval_GetFrame();

  int i = 0;
  while (i < max_stack_depth) {
    if (cur_frame == NULL) break;

    if (check_if_should_skip_this_frame(cur_frame)) {
      cur_frame = cur_frame->f_back;
      continue;
    }
    cur_f_str += get_python_frame_str_repr(cur_frame);
    cur_frame = cur_frame->f_back;
    i++;
  }

  // show how may stack frames remain to be shown in debug mode
  const bool debug_mode = GetGraphDebugMode() || IsInDebugMode();
  if (debug_mode) {
    const int32_t current_stack_depth = get_cur_stack_depth();
    if (current_stack_depth > max_stack_depth) {
      cur_f_str += "... " + std::to_string(current_stack_depth - max_stack_depth) + " more";
    }
  } else {
    if (cur_frame != NULL) { cur_f_str += " ... more"; }
  }

  return cur_f_str;
}

}  // namespace

PythonFrameGuard::PythonFrameGuard() {
  if (OF_PREDICT_FALSE(LazyMode::is_enabled())) {
    prev_frame_str_ = DispatchFrame::get_str();
    DispatchFrame::set_str(get_cur_frame_stack_str());
  }
}
PythonFrameGuard::~PythonFrameGuard() {
  if (OF_PREDICT_FALSE(LazyMode::is_enabled())) { DispatchFrame::set_str(prev_frame_str_); }
}

}  // namespace oneflow
