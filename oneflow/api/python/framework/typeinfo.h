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

#ifndef ONEFLOW_API_PYTHON_FRAMEWORK_TYPEINFO_H_
#define ONEFLOW_API_PYTHON_FRAMEWORK_TYPEINFO_H_

#include <Python.h>
#include "oneflow/core/common/throw.h"
#include "oneflow/core/framework/dtype.h"

namespace oneflow {
namespace one {

typedef struct {
  PyObject_HEAD;
  Symbol<DType> dtype;
} PyDTypeInfo;

extern PyTypeObject PyIInfoType;
extern PyTypeObject PyFInfoType;

inline bool PyIInfo_Check(PyObject* obj) { return PyObject_TypeCheck(obj, &PyIInfoType); }
inline bool PyFInfo_Check(PyObject* obj) { return PyObject_TypeCheck(obj, &PyFInfoType); }
inline bool PyDTypeInfo_Check(PyObject* obj) { return PyIInfo_Check(obj) || PyFInfo_Check(obj); }

inline Symbol<DType> PyDTypeInfo_UnpackDType(PyObject* obj) {
  assert(PyDTypeInfo_Check(obj));
  return ((PyDTypeInfo*)obj)->dtype;
}

inline DataType PyDTypeInfo_UnpackDataType(PyObject* obj) {
  assert(PyDTypeInfo_Check(obj));
  return ((PyDTypeInfo*)obj)->dtype->data_type();
}

}  // namespace one
}  // namespace oneflow
#endif  // ONEFLOW_API_PYTHON_FRAMEWORK_TYPEINFO_H_
