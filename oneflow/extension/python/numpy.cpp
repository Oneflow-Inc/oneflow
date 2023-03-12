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
#include <pybind11/pybind11.h>
#include "oneflow/core/common/stride.h"
#include "oneflow/core/common/throw.h"
#include "oneflow/core/common/registry_error.h"
#include "oneflow/extension/python/numpy_internal.h"

namespace py = pybind11;

namespace oneflow {

namespace numpy {

NumPyArrayInternal::NumPyArrayInternal(PyObject* obj, const std::function<void()>& deleter)
    : obj_((PyArrayObject*)obj), deleter_(deleter) {
  CHECK_OR_THROW(PyArray_Check(obj)) << "The object is not a numpy array.";
  CHECK_OR_THROW(PyArray_ISCONTIGUOUS(obj_)) << "Contiguous array is expected.";
  size_ = PyArray_SIZE(obj_);
  data_ = PyArray_DATA(obj_);
}

NumPyArrayInternal::~NumPyArrayInternal() {
  if (deleter_) { deleter_(); }
}

Maybe<int> OFDataTypeToNumpyType(DataType of_data_type) {
  switch (of_data_type) {
    case DataType::kBool: return NPY_BOOL;
    case DataType::kFloat: return NPY_FLOAT32;
    case DataType::kDouble: return NPY_FLOAT64;
    case DataType::kInt8: return NPY_INT8;
    case DataType::kInt32: return NPY_INT32;
    case DataType::kInt64: return NPY_INT64;
    case DataType::kUInt8: return NPY_UINT8;
    case DataType::kFloat16: return NPY_FLOAT16;
    case DataType::kComplex64: return NPY_COMPLEX64;
    case DataType::kComplex128: return NPY_COMPLEX128;
    default:
      return Error::InvalidValueError() << "OneFlow data type " << DataType_Name(of_data_type)
                                        << " is not valid to Numpy data type.";
  }
}

Maybe<DataType> NumpyTypeToOFDataType(int np_type) {
  switch (np_type) {
    case NPY_BOOL: return DataType::kBool;
    case NPY_FLOAT32: return DataType::kFloat;
    case NPY_FLOAT64: return DataType::kDouble;
    case NPY_INT8: return DataType::kInt8;
    case NPY_INT32: return DataType::kInt32;
    case NPY_INT64:
    case NPY_LONGLONG: return DataType::kInt64;
    case NPY_UINT8: return DataType::kUInt8;
    case NPY_FLOAT16: return DataType::kFloat16;
    case NPY_COMPLEX64: return DataType::kComplex64;
    case NPY_COMPLEX128: return DataType::kComplex128;
    default:
      return Error::InvalidValueError() << "Numpy data type " << std::to_string(np_type)
                                        << " is not valid to OneFlow data type.";
  }
}

Maybe<DataType> GetOFDataTypeFromNpArray(PyArrayObject* array) {
  int np_array_type = PyArray_TYPE(array);
  return NumpyTypeToOFDataType(np_array_type);
}

std::vector<size_t> OFShapeToNumpyShape(const DimVector& fixed_vec) {
  size_t ndim = fixed_vec.size();
  auto result = std::vector<size_t>(ndim);
  for (int i = 0; i < ndim; i++) { result[i] = fixed_vec.at(i); }
  return result;
}

// NumPy strides use bytes. OneFlow strides use element counts.
std::vector<size_t> OFStrideToNumpyStride(const Stride& stride, const DataType data_type) {
  size_t ndim = stride.size();
  auto result = std::vector<size_t>(ndim);
  int byte_per_elem = GetSizeOfDataType(data_type);
  for (int i = 0; i < ndim; i++) { result[i] = stride.at(i) * byte_per_elem; }
  return result;
}

bool PyArrayCheckLongScalar(PyObject* obj) {
  return PyArray_CheckScalar(obj) && PyDataType_ISINTEGER(PyArray_DescrFromScalar(obj));
}

bool PyArrayCheckFloatScalar(PyObject* obj) {
  return PyArray_CheckScalar(obj) && PyDataType_ISFLOAT(PyArray_DescrFromScalar(obj));
}

bool PyArrayCheckBoolScalar(PyObject* obj) {
  return PyArray_CheckScalar(obj) && PyDataType_ISBOOL(PyArray_DescrFromScalar(obj));
}

bool PyArrayCheckComplexScalar(PyObject* obj) {
  return PyArray_CheckScalar(obj) && PyDataType_ISCOMPLEX(PyArray_DescrFromScalar(obj));
}

// Executing any numpy c api before _import_array() results in segfault
// NOTE: this InitNumpyCAPI() works because of `PY_ARRAY_UNIQUE_SYMBOL`
// defined in numpy_internal.h
// Reference:
// https://numpy.org/doc/stable/reference/c-api/array.html#importing-the-api
Maybe<void> InitNumpyCAPI() {
  CHECK_ISNULL_OR_RETURN(PyArray_API);
  CHECK_EQ_OR_RETURN(_import_array(), 0)
      << ". Unable to import Numpy array, try to upgrade Numpy version!";
  return Maybe<void>::Ok();
}

}  // namespace numpy
}  // namespace oneflow
