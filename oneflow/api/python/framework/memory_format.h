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
#ifndef ONEFLOW_API_PYTHON_FRAMEWORK_MEMORY_FORMAT_H_
#define ONEFLOW_API_PYTHON_FRAMEWORK_MEMORY_FORMAT_H_

#include <Python.h>
#include <pybind11/pybind11.h>

#include "oneflow/core/common/memory_format.pb.h"

namespace oneflow {

typedef struct PyMemoryFormatObject {
  PyTypeObject ob_type;
  MemoryFormat memory_format;
} PyMemoryFormatObject;

bool PyMemoryFormat_Check(PyObject*);

inline MemoryFormat PyMemoryFormat_Unpack(PyObject* self) {
  return ((PyMemoryFormatObject*)self)->memory_format;
}

PyObject* PyMemoryFormat_New(MemoryFormat memory_format);

}  // namespace oneflow

#endif  // ONEFLOW_API_PYTHON_FRAMEWORK_MEMORY_FORMAT_H_
