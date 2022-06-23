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

#include "oneflow/api/python/functional/common.h"
#include "oneflow/core/framework/op_interpreter/dispatch_frame.h"
#include "oneflow/core/job/graph_scope_vars.h"

namespace oneflow {
namespace one {
namespace functional {

namespace {
std::string get_cur_frame_stack_str(int32_t max_stack_depth) {
  std::string cur_f_str;
  PyFrameObject* cur_frame = PyEval_GetFrame();
  for (int32_t i = 0; i < max_stack_depth; i++) {
    if (cur_frame == NULL) break;
    const int32_t stack_index = (-1) * i - 1;
    cur_f_str = "Python Stack[" + std::to_string(stack_index)
                + "]: " + PyObjectToReprStr((PyObject*)cur_frame) + "; " + cur_f_str;
    cur_frame = cur_frame->f_back;
  }
  return cur_f_str;
}

int32_t get_cur_stack_depth() {
  int32_t current_stack_depth = 0;
  PyFrameObject* f = PyEval_GetFrame();
  while (f) {
    current_stack_depth++;
    f = f->f_back;
  }
  return current_stack_depth;
}

std::string get_cur_frame_stack_str() {
  const bool debug_mode = GetGraphDebugMode();
  const int32_t max_stack_depth = GetGraphDebugMaxPyStackDepth();
  if (debug_mode) {  // show more info for the stack trace in debug mode
    int32_t current_stack_depth = get_cur_stack_depth();
    std::string cur_f_str = get_cur_frame_stack_str(max_stack_depth);
    if (current_stack_depth > max_stack_depth) {  // show how many stack depth remaining to be shown
      int32_t remaining_stack_depth = current_stack_depth - max_stack_depth;
      cur_f_str += " ... " + std::to_string(remaining_stack_depth) + " more; ";
    }
    return cur_f_str;
  }

  return get_cur_frame_stack_str(max_stack_depth);
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
