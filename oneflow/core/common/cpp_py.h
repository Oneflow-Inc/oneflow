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
#ifndef ONEFLOW_CORE_COMMON_CPP_PY_H_
#define ONEFLOW_CORE_COMMON_CPP_PY_H_

#include "oneflow/core/common/data_type.h"
#include "oneflow/core/framework/util.h"
#include "oneflow/core/framework/tensor.h"
extern "C" {
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
}

namespace oneflow {
namespace {
void OFDataTypeToNumpyType(DataType of_data_type, int* out_numpy_type) {
  switch (of_data_type) {
    case DataType::kFloat: *out_numpy_type = NPY_FLOAT32; break;
    case DataType::kDouble: *out_numpy_type = NPY_FLOAT64; break;
    case DataType::kInt8: *out_numpy_type = NPY_INT8; break;
    case DataType::kInt32: *out_numpy_type = NPY_INT32; break;
    case DataType::kInt64: *out_numpy_type = NPY_INT64; break;
    case DataType::kUInt8: *out_numpy_type = NPY_UINT8; break;
    case DataType::kFloat16: *out_numpy_type = NPY_FLOAT16; break;
    default:
      LOG(FATAL) << "OneFlow data type " << DataType_Name(of_data_type)
                 << " is not valid to numpy data type.";
  }
}
}  // namespace

template<typename T>
void TensorToNumpy(const user_op::Tensor* tensor, PyObject* arg) {
  if (tensor == nullptr) {
    Py_INCREF(Py_None);
    arg = Py_None;
    return;
  }
  int type_num = -1;
  OFDataTypeToNumpyType(tensor->data_type(), &type_num);
  int dim_size = tensor->shape().NumAxes();
  npy_intp dims[dim_size];
  FOR_RANGE(size_t, i, 0, dim_size) { dims[i] = tensor->shape().At(i); }
  void* data = static_cast<void*>(const_cast<T*>(tensor->dptr<T>()));
  auto* np_array =
      reinterpret_cast<PyArrayObject*>(PyArray_SimpleNewFromData(dim_size, dims, type_num, data));
  // Numpy will not release the data
  PyArray_CLEARFLAGS(np_array, NPY_ARRAY_OWNDATA);
  arg = reinterpret_cast<PyObject*>(np_array);
}

template<typename T>
void NumpyToTensor(PyObject* arg, user_op::Tensor* tensor) {}
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_CPP_PY_H_