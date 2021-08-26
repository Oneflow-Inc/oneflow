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
#include "oneflow/extension/python/numpy.h"

namespace oneflow {

namespace numpy {

Maybe<int> OFDataTypeToNumpyType(DataType of_data_type) {
  switch (of_data_type) {
    case DataType::kFloat: return NPY_FLOAT32;
    case DataType::kDouble: return NPY_FLOAT64;
    case DataType::kInt8: return NPY_INT8;
    case DataType::kInt32: return NPY_INT32;
    case DataType::kInt64: return NPY_INT64;
    case DataType::kUInt8: return NPY_UINT8;
    case DataType::kFloat16: return NPY_FLOAT16;
    default:
      return Error::ValueError("OneFlow data type " + DataType_Name(of_data_type)
                               + " is not valid to Numpy data type.");
  }
}

Maybe<DataType> NumpyTypeToOFDataType(int np_type) {
  switch (np_type) {
    case NPY_FLOAT32: return DataType::kFloat;
    case NPY_FLOAT64: return DataType::kDouble;
    case NPY_INT8: return DataType::kInt8;
    case NPY_INT32: return DataType::kInt32;
    case NPY_INT64: return DataType::kInt64;
    case NPY_UINT8: return DataType::kUInt8;
    case NPY_FLOAT16: return DataType::kFloat16;
    default:
      return Error::ValueError("Numpy data type " + std::to_string(np_type)
                               + " is not valid to OneFlow data type.");
  }
}

Maybe<DataType> GetOFDataTypeFromNpArray(PyArrayObject* array) {
  int np_array_type = PyArray_TYPE(array);
  return NumpyTypeToOFDataType(np_array_type);
}

}  // namespace numpy
}  // namespace oneflow
