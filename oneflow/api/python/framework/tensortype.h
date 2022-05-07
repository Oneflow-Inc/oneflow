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
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/tensor.h"

namespace oneflow {
namespace one {

// extern PyTypeObject* PyTensorObject_Type;
// extern PyTypeObject* PyParameterObject_Type;
extern PyTypeObject* PyTensortypeObject_Type;

extern PyHeapTypeObject* PyTensortypeMetaclass_Type;  // cpu Tensor
extern PyTypeObject* PyByteTensortypeObject_Type;     // uint8
extern PyTypeObject* PyCharTensortypeObject_Type;     // int8
extern PyTypeObject* PyShortTensortypeObject_Type;    // int16
extern PyTypeObject* PyIntTensortypeObject_Type;      // int32
extern PyTypeObject* PyLongTensortypeObject_Type;     // int64
extern PyTypeObject* PyHalfTensortypeObject_Type;     // float16
extern PyTypeObject* PyFloatTensortypeObject_Type;    // float32
extern PyTypeObject* PyDoubleTensortypeObject_Type;   // float64

extern PyHeapTypeObject* PyCudaTensortypeMetaclass_Type;  // cuda Tensor
extern PyTypeObject* PyCudaByteTensortypeObject_Type;     // cuda.uint8
extern PyTypeObject* PyCudaCharTensortypeObject_Type;     // cuda.int8
extern PyTypeObject* PyCudaShortTensortypeObject_Type;    // cuda.int16
extern PyTypeObject* PyCudaIntTensortypeObject_Type;      // cuda.int32
extern PyTypeObject* PyCudaLongTensortypeObject_Type;     // cuda.int64
extern PyTypeObject* PyCudaHalfTensortypeObject_Type;     // cuda.float16
extern PyTypeObject* PyCudaFloatTensortypeObject_Type;    // cuda.float32
extern PyTypeObject* PyCudaDoubleTensortypeObject_Type;   // cuda.float64

inline bool PyTensortype_Check(PyObject* op) {
  return PyObject_TypeCheck(op, &PyTensortypeMetaclass_Type->ht_type)
         || PyObject_TypeCheck(op, &PyCudaTensortypeMetaclass_Type->ht_type);
};
inline bool PyTensortype_CheckExact(PyObject* op) {
  return op->ob_type == &PyTensortypeMetaclass_Type->ht_type
         || op->ob_type == &PyCudaTensortypeMetaclass_Type->ht_type;
}

Symbol<DType> TensortypeToDType(PyObject*);
DeviceType TensortypeToDevice(PyObject*);
PyObject* GetTensortype(const Symbol<DType>&, const Maybe<Symbol<Device>>&);

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_API_PYTHON_FRAMEWORK_TENSOR_H_
