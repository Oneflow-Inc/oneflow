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

namespace oneflow {
namespace one {
namespace functional {

namespace {

inline PyFrameObject* get_frame_back(PyFrameObject* frame) {
  assert(frame != NULL);
  PyFrameObject* back = frame->f_back;
  if (back != NULL) { Py_XINCREF(back); }
  return back;
}

inline std::string get_cur_frame_stack_str() {
  PyFrameObject* cur_frame = PyEval_GetFrame();
  if (cur_frame == NULL) return "";
  std::string cur_f_str = "Python stack[-1]: " + PyObjectToReprStr((PyObject*)cur_frame);

  PyFrameObject* back_frame = get_frame_back(cur_frame);
  if (back_frame == NULL) return cur_f_str;
  std::string back_f_str = PyObjectToReprStr((PyObject*)back_frame);
  cur_f_str = "Python stack[-2]: " + back_f_str + "; " + cur_f_str;
  Py_XDECREF(back_frame);

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
