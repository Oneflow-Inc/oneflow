/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_USER_OPS_NN_UTIL_H_
#define ONEFLOW_USER_OPS_NN_UTIL_H_

#include "oneflow/core/framework/framework.h"

namespace oneflow {

Maybe<void> CalcOutAndPadding(int64_t input_size, int32_t filter_size, int32_t dilation_rate,
                              int32_t stride, const std::string& padding_type, int64_t* output_size,
                              int32_t* padding_before, int32_t* padding_after);

Maybe<void> CalcSamePadding(int64_t input_size, int32_t filter_size, int32_t dilation_rate,
                            int32_t stride, int32_t* padding_small, int32_t* padding_large);

Maybe<void> CalcConvOut(int64_t input_size, int32_t filter_size, int32_t dilation_rate,
                        int32_t stride, int32_t padding_before, int64_t* output_size);

const size_t IdxOffset(const std::string& data_format);
const int32_t ChannelIdx(const std::string& data_format, int32_t num_axes);

}  // namespace oneflow

#endif  // ONEFLOW_USER_OPS_NN_UTIL_H_
