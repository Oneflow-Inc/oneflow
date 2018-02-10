#ifndef ONEFLOW_CORE_COMMON_OPERATOR_UTIL_H_
#define ONEFLOW_CORE_COMMON_OPERATOR_UTIL_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

void GetWindowedOutputSize(int64_t input_size, int32_t filter_size,
                           int32_t stride, const std::string& padding_type,
                           int64_t* output_size, int32_t* pad_small_side,
                           int32_t* pad_large_side);

void GetWindowedOutputSize(int64_t input_size, int32_t filter_size,
                           int32_t stride, const std::string& padding_type,
                           int64_t* output_size, int32_t* padding_size);

void GetWindowedOutputSize(int64_t input_size, int32_t filter_size,
                           int32_t dilation_rate, int32_t stride,
                           const std::string& padding_type,
                           int64_t* output_size, int32_t* pad_small_side,
                           int32_t* pad_large_side);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_OPERATOR_UTIL_H_
