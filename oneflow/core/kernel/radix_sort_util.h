#ifndef RADIX_SORT_UTIL_H_
#define RADIX_SORT_UTIL_H_

#include "oneflow/core/common/data_type.pb.h"

namespace oneflow {

size_t InferTempStorageForRadixSort(int32_t num_row, int32_t num_col, DataType data_type);
}

#endif
