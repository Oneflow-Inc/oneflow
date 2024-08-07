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
#ifndef ONEFLOW_API_PYTHON_FRAMEWORK_SIZE_H_
#define ONEFLOW_API_PYTHON_FRAMEWORK_SIZE_H_
#include <type_traits>
#include <Python.h>
#undef _PyGC_FINALIZED
#include <pybind11/pybind11.h>
#include "oneflow/core/common/shape.h"

namespace oneflow {

typedef struct {
  PyTupleObject ob_base;
} TensorSize;

extern PyTypeObject TensorSize_Type;

int TensorSize_Check(PyObject* p);

PyObject* TensorSize_New(Py_ssize_t len);
PyObject* TensorSize_NewFromShape(const Shape& size);

Shape TensorSize_AsShape(PyObject* self);

}  // namespace oneflow

#endif  // ONEFLOW_API_PYTHON_FRAMEWORK_SIZE_H_
