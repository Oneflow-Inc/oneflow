#ifndef ONEFLOW_CORE_DEVICE_NCCL_UTIL_H_
#define ONEFLOW_CORE_DEVICE_NCCL_UTIL_H_

#include "nccl.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

inline ncclDataType_t GetNcclDataType(const DataType &dt) {
  switch (dt) {
    case DataType::kChar: return ncclDataType_t::ncclChar;
    case DataType::kFloat: return ncclDataType_t::ncclFloat;
    case DataType::kDouble: return ncclDataType_t::ncclDouble;
    case DataType::kInt8: return ncclDataType_t::ncclInt8;
    case DataType::kInt32: return ncclDataType_t::ncclInt32;
    default: UNIMPLEMENTED();
  }
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_NCCL_UTIL_H_
