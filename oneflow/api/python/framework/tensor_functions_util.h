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
#include <string>
#include "oneflow/api/python/exception/exception.h"
#include "oneflow/api/python/functional/common.h"
#include "oneflow/core/common/error.pb.h"
#include "oneflow/core/common/throw.h"

namespace oneflow {
namespace one {

using functional::PyObjectPtr;

std::string PyUnpack_String(PyObject* obj) {
  CHECK_OR_THROW(PyUnicode_Check(obj)) << "PyUnpack_String(): expect a PyUnicode object";
  Py_ssize_t size = -1;
  const char* data = PyUnicode_AsUTF8AndSize(obj, &size);
  CHECK_NOTNULL_OR_THROW(data) << "error unpacking string as utf-8";
  return std::string(data, (size_t)size);
}

// For signature like Tensor.reshape(*shape), this function can handle these cases:
// 1. parse positional arguments only case, like Tensor.reshape(1, 2)
// 2. parse keyword arguments only case, like Tensor.reshape(shape=(1, 2))
// 3. raise Error for multiple arguments case, like Tensor.reshape(1, shape=(1, ))
// 4. return empty tuple for empty arguments, like Tensor.reshape()
PyObject* PyParseArgs(PyObject* args, PyObject* kwargs, const char* func_name,
                      const std::string& param_name) {
  PyObject* args_obj = NULL;
  // Tensor.reshape(shape=(1, 2)), get (1, 2) for kwargs["shape"]
  if (kwargs != NULL) {
    PyObject* key = nullptr;
    PyObject* value = nullptr;
    Py_ssize_t pos = 0;
    while (PyDict_Next(kwargs, &pos, &key, &value)) {
      CHECK_OR_THROW(args_obj == NULL)
          << Error::TypeError() << func_name << "() got multiple values for argument '"
          << param_name << "' or get invalid argument";
      CHECK_EQ_OR_THROW(PyUnpack_String(key), param_name)
          << Error::TypeError() << func_name << "() got an unexpected keyword argument "
          << PyUnpack_String(key);
      args_obj = value;
    }
  }
  if (PyTuple_GET_SIZE(args) != 0) {
    CHECK_OR_THROW(args_obj == NULL)
        << Error::TypeError() << func_name << "() got multiple values for argument '" << param_name
        << "' or get invalid argument";
    if (PyTuple_Size(args) == 1 && functional::PyShapeSequenceCheck(args)) {
      args_obj = PyTuple_GET_ITEM(args, 0);
    } else {
      args_obj = args;
    }
  };
  if (args_obj == NULL) { args_obj = args; }
  return args_obj;
}

}  // namespace one
}  // namespace oneflow