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
#ifndef ONEFLOW_API_PYTHON_FUNCTIONAL_PYTHON_FRAME_H_
#define ONEFLOW_API_PYTHON_FUNCTIONAL_PYTHON_FRAME_H_

#include <Python.h>
#include <cstdint>
#include <string>
#include <vector>

#include "oneflow/api/python/functional/common.h"
#include "oneflow/core/framework/op_interpreter/dispatch_frame.h"
#include "oneflow/core/job/graph_scope_vars.h"
#include "oneflow/core/profiler/profiler.h"

namespace oneflow {
namespace one {
namespace functional {

namespace {

// get a formatted stack frame representation
// example: Python Stack[-10]: '__call__' at '.../graph/graph.py': line 219
std::string get_python_frame_str_repr(int32_t stack_index, PyFrameObject* frame) {
  if (frame == NULL) return "";
  PyCodeObject* code = frame->f_code;
  std::string repr = "Python Stack[" + std::to_string(stack_index) + "]: ";
  std::string file_name = PyObjectToReprStr(code->co_filename);
  std::string code_name = PyObjectToReprStr(code->co_name);
  int line_number = PyFrame_GetLineNumber(frame);

  return repr + code_name + " at " + file_name + ": line " + std::to_string(line_number) + "; ";
}

// all the files except those specified in paths_to_be_kepted in 'oneflow/python' should be filtered
const static std::vector<std::string> paths_to_be_filtered = {ONEFLOW_PYTHON_BASE_DIR};

// keep the files in 'python/oneflow/test' and 'python/oneflow/nn/modules' for running and debugging
// tests
const static std::vector<std::string> paths_to_be_kepted = {
    std::string(ONEFLOW_PYTHON_BASE_DIR) + "/oneflow/test",
    std::string(ONEFLOW_PYTHON_BASE_DIR) + "/oneflow/nn/modules"};

bool check_if_python_file_should_be_filtered(const std::string& path) {
  for (int i = 0; i < paths_to_be_kepted.size(); ++i) {
    const std::string& path_to_keep = paths_to_be_kepted[i];
    if (path.size() > path_to_keep.size()) {
      if (path.substr(0, path_to_keep.size()) == path_to_keep) { return false; }
    }
  }

  for (int i = 0; i < paths_to_be_filtered.size(); ++i) {
    const std::string& path_to_filter = paths_to_be_filtered[i];
    if (path.size() > path_to_filter.size()) {
      if (path.substr(0, path_to_filter.size()) == path_to_filter) { return true; }
    }
  }

  return false;
}

bool check_if_frame_should_be_filtered(PyFrameObject* frame) {
  std::string frame_file_name = PyObjectToReprStr(frame->f_code->co_filename);
  frame_file_name = frame_file_name.substr(1, frame_file_name.size() - 2);  // get rid of ' '
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

    const int32_t stack_index = (-1) * i - 1;

    if (check_if_should_skip_this_frame(cur_frame)) {
      cur_frame = cur_frame->f_back;
      continue;
    }

    i++;
    cur_f_str = get_python_frame_str_repr(stack_index, cur_frame) + cur_f_str;
    cur_frame = cur_frame->f_back;
  }

  const bool debug_mode =
      GetGraphDebugMode();  // show how may stack frames remain to be shown in debug mode
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

class PythonFrameGuard {
 public:
  PythonFrameGuard() {
    if (OF_PREDICT_FALSE(LazyMode::is_enabled())) {
      prev_frame_str_ = DispatchFrame::get_str();
      DispatchFrame::set_str(get_cur_frame_stack_str());
    }
  }
  ~PythonFrameGuard() {
    if (OF_PREDICT_FALSE(LazyMode::is_enabled())) { DispatchFrame::set_str(prev_frame_str_); }
  }

 private:
  std::string prev_frame_str_;
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_API_PYTHON_FUNCTIONAL_PYTHON_FRAME_H_
