#ifndef ONEFLOW_CORE_OPERATOR_INDEXED_SLICES_REDUCE_SUM_OP_UTIL_H_
#define ONEFLOW_CORE_OPERATOR_INDEXED_SLICES_REDUCE_SUM_OP_UTIL_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

struct IndexedSlicesReduceSumOpUtil {
  static void GetReduceSumWorkspaceSizeInBytes(DeviceType device_type, DataType value_type,
                                               DataType index_type, int64_t n, int64_t m,
                                               int64_t* workspace_size_in_bytes);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_INDEXED_SLICES_REDUCE_SUM_OP_UTIL_H_
