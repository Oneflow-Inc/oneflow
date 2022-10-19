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
#include "oneflow/api/python/stack_getter.h"

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <frameobject.h>
#include <pystate.h>
// see https://bugs.python.org/issue23644 for why this file is written
// as .c instead of .cpp
#define Py_BUILD_CORE
#include "internal/pycore_pystate.h"
#undef Py_BUILD_CORE

#if PY_VERSION_HEX >= 0x03090000
static PyObject* custom_eval_frame(PyThreadState* tstate, PyFrameObject* frame,
#else
static PyObject* custom_eval_frame(PyFrameObject* frame,
#endif
                                         int throw_flag) {
  // filter out functions like Python _bootstrap, _shutdown
  if (frame->f_back != NULL) {
    push_frame(frame->f_back);
  }
#if PY_VERSION_HEX >= 0x03090000
  if (tstate == NULL) { tstate = PyThreadState_GET(); }
  PyObject* ret = _PyEval_EvalFrameDefault(tstate, frame, throw_flag);
#else
  PyObject* ret = _PyEval_EvalFrameDefault(frame, throw_flag);
#endif
  pop_frame();
  return ret;
}

inline static void enable_custom_eval_frame(PyThreadState* tstate) {
#if PY_VERSION_HEX >= 0x03090000
  if (_PyInterpreterState_GetEvalFrameFunc(tstate->interp) != &custom_eval_frame) {
    _PyInterpreterState_SetEvalFrameFunc(tstate->interp, &custom_eval_frame);
  }
#else
  if (tstate->interp->eval_frame != &custom_eval_frame) {
    // First call
    tstate->interp->eval_frame = &custom_eval_frame;
  }
#endif
}

void enable_custom_eval_frame_for_current_thread() {
  return enable_custom_eval_frame(PyThreadState_GET());
}
