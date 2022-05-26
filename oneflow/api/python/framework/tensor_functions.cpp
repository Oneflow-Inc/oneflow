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
#include "oneflow/core/common/shape_vec.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/common/shape.h"

namespace oneflow {
namespace one {

using functional::PyObjectPtr;

#define ASSERT(x) (x).GetOrThrow()
#define ASSERT_PTR(x) (x).GetPtrOrThrow()

static PyObject* PyTensorObject_reshape(PyObject* self, PyObject* args) {
  HANDLE_ERRORS
  PyObject* shape = args;
  if (PyTuple_Size(args) == 1) {
    PyObject* item = PyTuple_GetItem(args, 0);
    if (!PyLong_Check(item)) { shape = item; }
  }
  CHECK_OR_THROW(functional::PyLongSequenceCheck(shape))
      << "reshape(): argument 'shape' must be tuple of ints, but found "
      << functional::PyStringAsString(PyObject_Str((PyObject*)Py_TYPE(shape)));
  const auto& dims = functional::PyUnpackLongSequence<int64_t>(shape);
  DimVector dim(dims.begin(), dims.end());
  return PyTensor_New(ASSERT_PTR(functional::Reshape(PyTensor_Unpack(self), Shape(dim))));
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_reshape_as(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  auto tensor = PyTensor_Unpack(self);
  PyObject* other = NULL;
  static const char* keywords[2] = {"other", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|:reshape_as", const_cast<char**>(keywords),
                                   &other)) {
    return NULL;
  }
  return PyTensor_New(ASSERT_PTR(functional::Reshape(tensor, *PyTensor_Unpack(other)->shape())));
  END_HANDLE_ERRORS
}

PyMethodDef PyTensorObject_extra_methods[] = {
    {"reshape", PyTensorObject_reshape, METH_VARARGS, NULL},
    {"reshape_as", (PyCFunction)PyTensorObject_reshape_as, METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL},
};

}  // namespace one
}  // namespace oneflow
