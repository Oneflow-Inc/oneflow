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
#ifndef ONEFLOW_CORE_DEVICE_NCCL_UTIL_H_
#define ONEFLOW_CORE_DEVICE_NCCL_UTIL_H_

#include "oneflow/core/register/blob.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/device/cuda_util.h"

#ifdef WITH_CUDA

#include <cuda.h>
#if CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#endif  // CUDA_VERSION >= 11000

#endif  // WITH_CUDA

namespace oneflow {

#ifdef WITH_CUDA

inline ncclDataType_t GetNcclDataType(const DataType& dt) {
  switch (dt) {
#define NCCL_DATA_TYPE_CASE(dtype) \
  case DataType::k##dtype: return ncclDataType_t::nccl##dtype
    NCCL_DATA_TYPE_CASE(Char);
    NCCL_DATA_TYPE_CASE(Float);
    NCCL_DATA_TYPE_CASE(Double);
    NCCL_DATA_TYPE_CASE(Int8);
    NCCL_DATA_TYPE_CASE(Int32);
    NCCL_DATA_TYPE_CASE(Int64);
    NCCL_DATA_TYPE_CASE(Float16);
    case DataType::kBool: return ncclDataType_t::ncclUint8;
#if defined(__CUDA_BF16_TYPES_EXIST__) && NCCL_VERSION_CODE >= 21003
    case DataType::kBFloat16: return ncclBfloat16;
#endif
    case DataType::kUInt8: return ncclUint8;
    case DataType::kUInt32: return ncclUint32;
    case DataType::kUInt64: return ncclUint64;
    default: UNIMPLEMENTED();
  }
  return ncclDataType_t::ncclFloat;
}

std::string NcclUniqueIdToString(const ncclUniqueId& unique_id);

void NcclUniqueIdFromString(const std::string& str, ncclUniqueId* unique_id);

#define HAS_NCCL_SEND_RECV NCCL_VERSION_CODE > 2700

#endif  // WITH_CUDA

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_NCCL_UTIL_H_
