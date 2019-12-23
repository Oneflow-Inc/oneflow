#ifndef ONEFLOW_CORE_OPERATOR_RADIX_SORT_OP_UTIL_H_
#define ONEFLOW_CORE_OPERATOR_RADIX_SORT_OP_UTIL_H_

#include "oneflow/core/common/data_type.pb.h"

namespace oneflow {

size_t InferTempStorageForSortingPairsAscendingAtCompile(int32_t num_row, int32_t num_col,
                                                         DataType key_data_type);
size_t InferTempStorageForSortingPairsDescendingAtCompile(int32_t num_row, int32_t num_col,
                                                          DataType key_data_type);
size_t InferTempStorageForSortingKeysAscendingAtCompile(int32_t num_row, int32_t num_col,
                                                        DataType key_data_type);
size_t InferTempStorageForSortingKeysDescendingAtCompile(int32_t num_row, int32_t num_col,
                                                         DataType key_data_type);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_RADIX_SORT_OP_UTIL_H_
