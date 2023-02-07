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

// see https://bugs.python.org/issue23644 for why this file is written
// as .c instead of .cpp

#include "oneflow/extension/stack/python/custom_eval_frame.h"

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <frameobject.h>
#include <pystate.h>
// see https://bugs.python.org/issue35886
#if PY_VERSION_HEX >= 0x03080000
#define Py_BUILD_CORE
#include "internal/pycore_pystate.h"
#undef Py_BUILD_CORE
#endif

inline static void EnableCustomEvalFrame(PyThreadState* tstate, _PyFrameEvalFunction eval_func) {
#if PY_VERSION_HEX >= 0x03090000
  if (_PyInterpreterState_GetEvalFrameFunc(tstate->interp) != eval_func) {
    _PyInterpreterState_SetEvalFrameFunc(tstate->interp, eval_func);
  }
#else
  if (tstate->interp->eval_frame != eval_func) {
    tstate->interp->eval_frame = eval_func;
  }
#endif
}

void EnableCustomEvalFrameForCurrentThread(_PyFrameEvalFunction eval_func) {
  return EnableCustomEvalFrame(PyThreadState_GET(), eval_func);
}
