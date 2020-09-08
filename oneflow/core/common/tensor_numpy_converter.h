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
#ifndef ONEFLOW_CORE_COMMON_TENSOR_NUMPY_CONVERTER_H_
#define ONEFLOW_CORE_COMMON_TENSOR_NUMPY_CONVERTER_H_

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "oneflow/core/common/data_type.h"
#include "oneflow/core/framework/util.h"
#include "oneflow/core/framework/tensor.h"

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
                 << " is not valid to Numpy data type.";
  }
}

void NumpyTypeToOFDataType(PyArrayObject* array, DataType* of_data_type) {
  int py_array_type = PyArray_TYPE(array);
  switch (py_array_type) {
    case NPY_FLOAT32: *of_data_type = DataType::kFloat; break;
    case NPY_FLOAT64: *of_data_type = DataType::kDouble; break;
    case NPY_INT8: *of_data_type = DataType::kInt8; break;
    case NPY_INT32: *of_data_type = DataType::kInt32; break;
    case NPY_INT64: *of_data_type = DataType::kInt64; break;
    case NPY_UINT8: *of_data_type = DataType::kUInt8; break;
    case NPY_FLOAT16: *of_data_type = DataType::kFloat16; break;
    default:
      LOG(FATAL) << "Numpy data type " << py_array_type << " is not valid to OneFlow data type.";
  }
}
}  // namespace

template<typename T>
void TensorToNumpy(const user_op::Tensor* tensor, PyObject** arg_ptr) {
  if (tensor == nullptr) {
    Py_INCREF(Py_None);
    *arg_ptr = Py_None;
    return;
  }
  int type_num = -1;
  OFDataTypeToNumpyType(tensor->data_type(), &type_num);
  LOG(INFO) << "Tensor data type " << DataType_Name(tensor->data_type()) << " Numpy type "
            << type_num;
  int dim_size = tensor->shape().NumAxes();
  npy_intp dims[dim_size];
  FOR_RANGE(size_t, i, 0, dim_size) { dims[i] = tensor->shape().At(i); }
  void* data = static_cast<void*>(const_cast<T*>(tensor->dptr<T>()));
  auto* np_array =
      reinterpret_cast<PyArrayObject*>(PyArray_SimpleNewFromData(dim_size, dims, type_num, data));
  void* ptr = PyArray_DATA(np_array);
  // Numpy will not release the data
  PyArray_CLEARFLAGS(np_array, NPY_ARRAY_OWNDATA);
  *arg_ptr = reinterpret_cast<PyObject*>(np_array);
}

template<typename T>
void NumpyToTensor(PyObject* arg, user_op::Tensor* tensor) {
  PyObject* ro_array = PyArray_FromAny(arg, nullptr, 0, 0, NPY_ARRAY_CARRAY_RO, nullptr);
  PyArrayObject* array = reinterpret_cast<PyArrayObject*>(ro_array);

  DataType of_data_type = DataType::kFloat;
  NumpyTypeToOFDataType(array, &of_data_type);
  CHECK_EQ(of_data_type, tensor->data_type())
      << "Numpy to OneFlow data type " << DataType_Name(of_data_type)
      << " is not equal to OneFlow tensor data type " << DataType_Name(tensor->data_type());

  int64_t array_elem_cnt = 1;
  FOR_RANGE(int, i, 0, PyArray_NDIM(array)) { array_elem_cnt *= PyArray_SHAPE(array)[i]; }
  CHECK_EQ(array_elem_cnt, tensor->shape().elem_cnt())
      << "Numpy array element count " << array_elem_cnt
      << " is not equal to OneFlow tensor element count " << tensor->shape().elem_cnt();

  void* array_data_void = PyArray_DATA(array);
  T* array_data = static_cast<T*>(array_data_void);
  FOR_RANGE(int64_t, i, 0, array_elem_cnt) { tensor->mut_dptr<T>()[i] = array_data[i]; }
}
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_TENSOR_NUMPY_CONVERTER_H_