#ifndef ONEFLOW_CUSTOMIZED_OPS_NN_UTIL_H_
#define ONEFLOW_CUSTOMIZED_OPS_NN_UTIL_H_

#include "oneflow/core/framework/framework.h"

namespace oneflow {

void CalcOutAndPadding(int64_t input_size, int32_t filter_size, int32_t dilation_rate,
                       int32_t stride, const std::string& padding_type, int64_t* output_size,
                       int32_t* padding_before, int32_t* padding_after);

void CalcSamePadding(int64_t input_size, int32_t filter_size, int32_t dilation_rate, int32_t stride,
                     const std::string& padding_type, int32_t* padding_small,
                     int32_t* padding_large);

void CalcConvOut(int64_t input_size, int32_t filter_size, int32_t dilation_rate, int32_t stride,
                 int32_t pad, int64_t* output_size);

const size_t IdxOffset(const std::string& data_format);

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_OPS_NN_UTIL_H_
