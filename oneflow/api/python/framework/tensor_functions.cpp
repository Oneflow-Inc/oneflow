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

#include <Python.h>
#include "oneflow/api/python/exception/exception.h"
#include "oneflow/api/python/functional/common.h"
#include "oneflow/api/python/functional/functional_api.yaml.pybind.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

using functional::PyObjectPtr;

#define ASSERT(x) (x).GetOrThrow()
#define ASSERT_PTR(x) (x).GetPtrOrThrow()

static PyObject* PyTensorObject_reshape(PyObject* self, PyObject* args) {
  HANDLE_ERRORS
  PyObject* new_shape = NULL;
  if (PyTuple_Size(args) == 1) {
    new_shape = PyTuple_GetItem(args, 0);
    if (PyLong_Check(new_shape)) new_shape = PyTuple_Pack(1, new_shape);
  } else {
    new_shape = args;
  }
  PyObject* tuple = PyTuple_Pack(2, self, new_shape);
  return functional::reshape(NULL, tuple, NULL);
  Py_DECREF(new_shape);
  Py_DECREF(tuple);
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_reshape_as(PyObject* self, PyObject* args) {
  HANDLE_ERRORS
  auto tensor = PyTensor_Unpack(self);
  PyObject* other = PyTuple_GetItem(args, 0);
  auto result = ASSERT_PTR(functional::Reshape(tensor, *PyTensor_Unpack(other)->shape()));
  return PyTensor_New(result);
  END_HANDLE_ERRORS
}

PyMethodDef PyTensorObject_extra_methods[] = {
    {"reshape", PyTensorObject_reshape, METH_VARARGS, NULL},
    {"reshape_as", PyTensorObject_reshape_as, METH_VARARGS, NULL},
    {NULL},
};

}  // namespace one
}  // namespace oneflow
