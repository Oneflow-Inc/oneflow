#ifndef ONEFLOW_CORE_COMMON_OPERATOR_UTIL_H_
#define ONEFLOW_CORE_COMMON_OPERATOR_UTIL_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

void GetWindowedOutputSize(int32_t input_size, int32_t filter_size,
                           int32_t stride, const std::string& padding_type,
                           int32_t* output_size, int32_t* padding_before,
                           int32_t* padding_after);

void GetWindowedOutputSize(int32_t input_size, int32_t filter_size,
                           int32_t stride, const std::string& padding_type,
                           int32_t* output_size, int32_t* padding_size);

void GetWindowedOutputSize(int32_t input_size, int32_t filter_size,
                           int32_t dilation_rate, int32_t stride,
                           const std::string& padding_type,
                           int32_t* output_size, int32_t* padding_before,
                           int32_t* padding_after);

void Get3DOutputSize(const std::vector<int32_t>& in,
                     const std::vector<int32_t>& pool_size,
                     const std::vector<int32_t>& strides,
                     const std::string& padding_type, std::vector<int32_t>* out,
                     std::vector<int32_t>* padding);

void Get3DOutputSize(const std::vector<int32_t>& in,
                     const std::vector<int32_t>& pool_size,
                     const std::vector<int32_t>& strides,
                     const std::string& padding_type, std::vector<int32_t>* out,
                     std::vector<int32_t>* padding_before,
                     std::vector<int32_t>* padding_after);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_OPERATOR_UTIL_H_
