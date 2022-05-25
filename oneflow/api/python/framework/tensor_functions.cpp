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

#define ASSERT(x) (x).GetOrThrow()
#define ASSERT_PTR(x) (x).GetPtrOrThrow()

int64_t unpack_long(PyObject* self) {
  int overflow = -1;
  long long val = PyLong_AsLongLongAndOverflow(self, &overflow);
  if (val == -1 && PyErr_Occurred()) {
    THROW(RuntimeError) << "unpack_long >> Python exception occurs. python type:"
                        << Py_TYPE(self)->tp_name;
  }
  if (overflow != 0) { THROW(RuntimeError) << "unpack_long >> Overflow when unpacking long"; }
  return (int64_t)val;
}

static PyObject* PyTensorObject_reshape(PyObject* self, PyObject* args) {
  HANDLE_ERRORS
  auto tensor = PyTensor_Unpack(self);
  if (!PyTuple_Check(args)) {
    THROW(TypeError) << "reshape(): argument 'shape' must be tuple of ints, but found "
                     << Py_TYPE(args)->tp_name;
  }

  size_t size = (size_t)PyTuple_Size(args);
  if (size == -1) { size = tensor->ndim(); }
  PyObject* shape = PyTuple_GetItem(args, 0);
  if (PyList_Check(shape)) {
    size = (size_t)PyList_Size(shape);
    DimVector vec(size);
    for (int i = 0; i < size; ++i) { vec.at(i) = unpack_long(PyList_GetItem(shape, i)); }
    auto result = ASSERT_PTR(functional::Reshape(tensor, Shape(vec)));
    return PyTensor_New(result);

  } else {
    if (PyLong_Check(shape)) {
      DimVector vec(size);
      for (int i = 0; i < size; ++i) { vec.at(i) = unpack_long(PyTuple_GetItem(args, i)); }
      auto result = ASSERT_PTR(functional::Reshape(tensor, Shape(vec)));
      return PyTensor_New(result);
    } else {
      size = (size_t)PyTuple_Size(shape);
      DimVector vec(size);
      for (int i = 0; i < size; ++i) { vec.at(i) = unpack_long(PyTuple_GetItem(shape, i)); }
      auto result = CHECK_JUST(functional::Reshape(tensor, Shape(vec)));
      return PyTensor_New(result);
    }
  }
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_reshape_as(PyObject* self, PyObject* args) {
  HANDLE_ERRORS
  auto tensor = PyTensor_Unpack(self);
  PyObject* other = PyTuple_GetItem(args, 0);
  auto result = CHECK_JUST(functional::Reshape(tensor, *PyTensor_Unpack(other)->shape()));
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
