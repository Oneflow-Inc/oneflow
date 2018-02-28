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

void GetWindowedOutputSize(int64_t input_size, int64_t filter_size,
                           int64_t stride, const std::string& padding_type,
                           int64_t* output_size, int64_t* pad_small_side,
                           int64_t* pad_large_side);

void GetWindowedOutputSize(int64_t input_size, int64_t filter_size,
                           int64_t stride, const std::string& padding_type,
                           int64_t* output_size, int64_t* padding_size);

void GetWindowedOutputSize(int64_t input_size, int64_t filter_size,
                           int64_t dilation_rate, int64_t stride,
                           const std::string& padding_type,
                           int64_t* output_size, int64_t* pad_small_side,
                           int64_t* pad_large_side);

void Get3DOutputSize(const std::vector<int64_t>& in,
                     const std::vector<int64_t>& pool_size,
                     const std::vector<int64_t>& strides,
                     const std::string& padding_type, std::vector<int64_t>* out,
                     std::vector<int64_t>* padding);

void Get3DOutputSize(const std::vector<int64_t>& in,
                     const std::vector<int64_t>& pool_size,
                     const std::vector<int64_t>& strides,
                     const std::string& padding_type, std::vector<int64_t>* out,
                     std::vector<int64_t>* padding_before,
                     std::vector<int64_t>* padding_after);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_OPERATOR_UTIL_H_
