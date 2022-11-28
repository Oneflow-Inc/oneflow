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
#ifndef ONEFLOW_API_PYTHON_FRAMEWORK_TENSOR_H_
#define ONEFLOW_API_PYTHON_FRAMEWORK_TENSOR_H_

#include <Python.h>

#include "oneflow/core/framework/tensor.h"

namespace oneflow {
namespace one {

typedef struct {
  PyObject_HEAD;
  std::shared_ptr<Tensor> data;
} PyTensorObject;

extern PyTypeObject* PyTensorObject_Type;
extern PyTypeObject* PyParameterObject_Type;

inline bool PyTensorMetaClass_CheckExact(PyObject* obj) {
  return obj == (PyObject*)PyTensorObject_Type;
}

inline bool PyTensor_Check(PyObject* op) { return PyObject_TypeCheck(op, PyTensorObject_Type); }

inline bool PyTensor_CheckExact(PyObject* op) {
  return op->ob_type == PyTensorObject_Type || op->ob_type == PyParameterObject_Type;
}

inline std::shared_ptr<Tensor>& PyTensor_Unpack(PyObject* op) {
  assert(PyTensor_Check(op));
  return ((PyTensorObject*)op)->data;
}

PyObject* PyTensor_New(const std::shared_ptr<Tensor>& data);
PyObject* PyParameter_New(const std::shared_ptr<Parameter>& data);
PyObject* PyParameter_New(const std::shared_ptr<Tensor>& data, bool requires_grad);

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_API_PYTHON_FRAMEWORK_TENSOR_H_
