#ifndef ONEFLOW_CORE_OPERATOR_UNIQUE_OP_UTIL_H_
#define ONEFLOW_CORE_OPERATOR_UNIQUE_OP_UTIL_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

struct UniqueOpUtil {
  static void GetWorkspaceSizeInBytes(DeviceType device_type, DataType value_type,
                                      DataType index_type, int64_t n,
                                      int64_t* workspace_size_in_bytes);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_UNIQUE_OP_UTIL_H_
