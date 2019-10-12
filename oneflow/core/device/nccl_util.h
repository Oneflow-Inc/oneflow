#ifndef ONEFLOW_CORE_DEVICE_NCCL_UTIL_H_
#define ONEFLOW_CORE_DEVICE_NCCL_UTIL_H_

#include "oneflow/core/register/blob.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/device/cuda_util.h"

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
    NCCL_DATA_TYPE_CASE(Float16);
    default: UNIMPLEMENTED();
  }
}

void NcclCheck(ncclResult_t error);

#endif  // WITH_CUDA

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_NCCL_UTIL_H_
