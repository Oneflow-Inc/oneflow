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
#ifndef ONEFLOW_API_PYTHON_FRAMEWORK_PY_TENSOR_H_
#define ONEFLOW_API_PYTHON_FRAMEWORK_PY_TENSOR_H_

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

}  // namespace one
}  // namespace oneflow

#define PyTensor_Check(op) PyObject_TypeCheck(op, oneflow::one::PyTensorObject_Type)
#define PyTensor_CheckExact(op) Py_IS_TYPE(op, oneflow::one::PyTensorObject_Type)

#define _PyTensor_CAST(op) (assert(PyTensor_Check(op)), (oneflow::one::PyTensorObject*)(op))
#define PyTensor_Unpack(op) _PyTensor_CAST(op)->data

PyObject* PyTensor_New(const std::shared_ptr<oneflow::one::Tensor>& data);
PyObject* PyParameter_New(const std::shared_ptr<oneflow::one::Parameter>& data);
PyObject* PyParameter_New(const std::shared_ptr<oneflow::one::Tensor>& data, bool requires_grad);

#endif  // ONEFLOW_API_PYTHON_FRAMEWORK_PY_TENSOR_H_
