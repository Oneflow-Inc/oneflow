#ifndef ONEFLOW_CORE_COMMON_OPERATOR_UTIL_H_
#define ONEFLOW_CORE_COMMON_OPERATOR_UTIL_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

void GetWindowedOutputSize(int64_t input_size, int64_t filter_size,
                           int64_t stride, const std::string& padding_type,
                           int64_t* output_size, int64_t* padding_before,
                           int64_t* padding_after);

void GetWindowedOutputSize(int64_t input_size, int64_t filter_size,
                           int64_t stride, const std::string& padding_type,
                           int64_t* output_size, int64_t* padding_size);

void GetWindowedOutputSize(int64_t input_size, int64_t filter_size,
                           int64_t dilation_rate, int64_t stride,
                           const std::string& padding_type,
                           int64_t* output_size, int64_t* padding_before,
                           int64_t* padding_after);

void Get3DOutputSize(const std::vector<int64_t>& in,
                     const std::vector<int64_t>& pool_size,
                     const std::vector<int64_t>& strides,
                     const std::string& padding_type, std::vector<int64_t>* out,
                     std::vector<int64_t>* padding);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_OPERATOR_UTIL_H_
