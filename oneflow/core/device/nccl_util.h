#ifndef ONEFLOW_CORE_DEVICE_NCCL_UTIL_H_
#define ONEFLOW_CORE_DEVICE_NCCL_UTIL_H_

#include "nccl.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

inline ncclDataType_t GetNcclDataType(const DataType &dt) {
  switch (dt) {
#define NCCL_DATA_TYPE_CASE(dtype) \
  case DataType::k##dtype: return ncclDataType_t::nccl##dtype
    NCCL_DATA_TYPE_CASE(Char);
    NCCL_DATA_TYPE_CASE(Float);
    NCCL_DATA_TYPE_CASE(Double);
    NCCL_DATA_TYPE_CASE(Int8);
    NCCL_DATA_TYPE_CASE(Int32);
    default: UNIMPLEMENTED();
  }
}

void NcclCheck(ncclResult_t error);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_NCCL_UTIL_H_
