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
#ifndef ONEFLOW_API_PYTHON_CUSTOM_EVAL_FRAME_H_
#define ONEFLOW_API_PYTHON_CUSTOM_EVAL_FRAME_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <Python.h>
#undef _PyGC_FINALIZED

#if PY_VERSION_HEX >= 0x03090000
typedef PyObject* (*PyFrameEvalFunc)(struct _ts*, struct _frame*, int);
#else
typedef PyObject* (*PyFrameEvalFunc)(struct _frame*, int);
#endif
void EnableCustomEvalFrameForCurrentThread(PyFrameEvalFunc eval_func);

#ifdef __cplusplus
}
#endif

#endif  // ONEFLOW_API_PYTHON_CUSTOM_EVAL_FRAME_H_
