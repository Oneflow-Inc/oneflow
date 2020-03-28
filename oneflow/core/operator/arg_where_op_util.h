#ifndef ONEFLOW_CORE_OPERATOR_ARG_WHERE_OP_UTIL_H_
#define ONEFLOW_CORE_OPERATOR_ARG_WHERE_OP_UTIL_H_

#include "oneflow/core/common/data_type.h"

namespace oneflow {

void InferArgWhereWorkspaceSizeInBytes(DeviceType device_type, DataType value_type,
                                       DataType index_type, int32_t num_axes, int64_t n,
                                       int64_t* workspace_bytes);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_ARG_WHERE_OP_UTIL_H_
