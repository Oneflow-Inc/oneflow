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
#include "oneflow/cambricon/cnnl/cnnl_types.h"

#include "oneflow/core/common/throw.h"

namespace oneflow {

cnnlDataType_t ConvertToCnnlDataType(DataType dtype) {
  switch (dtype) {
    case DataType::kFloat: return CNNL_DTYPE_FLOAT;
    case DataType::kDouble: return CNNL_DTYPE_DOUBLE;
    case DataType::kFloat16: return CNNL_DTYPE_HALF;
    case DataType::kBool: return CNNL_DTYPE_BOOL;
    case DataType::kInt8: return CNNL_DTYPE_INT8;
    case DataType::kInt16: return CNNL_DTYPE_INT16;
    case DataType::kInt32: return CNNL_DTYPE_INT32;
    case DataType::kInt64: return CNNL_DTYPE_INT64;
    case DataType::kChar:
    case DataType::kUInt8: return CNNL_DTYPE_UINT8;
    case DataType::kUInt16: return CNNL_DTYPE_UINT16;
    case DataType::kUInt32: return CNNL_DTYPE_UINT32;
    case DataType::kUInt64: return CNNL_DTYPE_UINT64;
    default:
      THROW(RuntimeError) << "Can not convert oneflow " << DataType_Name(dtype)
                          << " to CNNL data type.";
      return CNNL_DTYPE_INVALID;
  }
}

}  // namespace oneflow
