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
#ifndef ONEFLOW_API_PYTHON_FRAMEWORK_TENSORTYPE_H_
#define ONEFLOW_API_PYTHON_FRAMEWORK_TENSORTYPE_H_

#include <Python.h>
#include <object.h>
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/framework/dtype.h"

namespace oneflow {
namespace one {

// extern PyTypeObject* PyTensorObject_Type;
// extern PyTypeObject* PyParameterObject_Type;
extern PyTypeObject* PyTensortypeObject_Type;

extern PyTypeObject* PyByteTensortypeObject_Type;   // uint8
extern PyTypeObject* PyCharTensortypeObject_Type;   // int8
extern PyTypeObject* PyShortTensortypeObject_Type;  // int16
extern PyTypeObject* PyIntTensortypeObject_Type;    // int32
extern PyTypeObject* PyLongTensortypeObject_Type;   // int64

extern PyTypeObject* PyHalfTensortypeObject_Type;
extern PyTypeObject* PyFloatTensortypeObject_Type;
extern PyTypeObject* PyDoubleTensortypeObject_Type;

bool PyTensortype_Check(PyObject* op);
Symbol<DType> TensortypeToDType(PyObject*);
PyObject* DTypeToTensortype(Symbol<DType>);
// static Symbol<DType> TensortypeToDType(PyObject* type);

inline bool PyByteTensor_Check(PyObject* op) {
  return PyObject_TypeCheck(op, PyByteTensortypeObject_Type);
}
inline bool PyCharTensor_Check(PyObject* op) {
  return PyObject_TypeCheck(op, PyCharTensortypeObject_Type);
}
inline bool PyShortTensor_Check(PyObject* op) {
  return PyObject_TypeCheck(op, PyShortTensortypeObject_Type);
}
inline bool PyIntTensor_Check(PyObject* op) {
  return PyObject_TypeCheck(op, PyIntTensortypeObject_Type);
}
inline bool PyLongTensor_Check(PyObject* op) {
  return PyObject_TypeCheck(op, PyLongTensortypeObject_Type);
}

inline bool PyHalfTensor_Check(PyObject* op) {
  return PyObject_TypeCheck(op, PyHalfTensortypeObject_Type);
}
inline bool PyFloatTensor_Check(PyObject* op) {
  return PyObject_TypeCheck(op, PyFloatTensortypeObject_Type);
}
inline bool PyDoubleTensor_Check(PyObject* op) {
  return PyObject_TypeCheck(op, PyDoubleTensortypeObject_Type);
}

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_API_PYTHON_FRAMEWORK_TENSOR_H_
