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
#include "oneflow/user/ops/nn_util.h"

namespace oneflow {

Maybe<void> CalcOutAndPadding(int64_t input_size, int32_t filter_size, int32_t dilation_rate,
                              int32_t stride, const std::string& padding_type, int64_t* output_size,
                              int32_t* padding_before, int32_t* padding_after) {
  CHECK_GT_OR_RETURN(stride, 0);
  CHECK_GE_OR_RETURN(dilation_rate, 1);

  int32_t effective_filter_size = (filter_size - 1) * dilation_rate + 1;
  if (padding_type == "valid") {
    if (output_size) { *output_size = (input_size - effective_filter_size + stride) / stride; }
    if (padding_before) { *padding_before = 0; }
    if (padding_after) { *padding_after = 0; }
  } else if (padding_type == "same") {
    int64_t tmp_output_size = (input_size + stride - 1) / stride;
    if (output_size) { *output_size = tmp_output_size; }
    const int32_t padding_needed = std::max(
        0,
        static_cast<int32_t>((tmp_output_size - 1) * stride + effective_filter_size - input_size));
    // For odd values of total padding, add more padding at the 'right'
    // side of the given dimension.
    if (padding_before) { *padding_before = padding_needed / 2; }
    if (padding_after) { *padding_after = padding_needed - padding_needed / 2; }
  } else {
    UNIMPLEMENTED();
  }
  if (output_size) { CHECK_GE_OR_RETURN((*output_size), 0); }
  return Maybe<void>::Ok();
}

Maybe<void> CalcSamePadding(int64_t input_size, int32_t filter_size, int32_t dilation_rate,
                            int32_t stride, int32_t* padding_small, int32_t* padding_large) {
  CHECK_GT_OR_RETURN(stride, 0);
  CHECK_GE_OR_RETURN(dilation_rate, 1);

  int32_t effective_filter_size = (filter_size - 1) * dilation_rate + 1;
  int64_t tmp_output_size = (input_size + stride - 1) / stride;
  const int32_t padding_needed = std::max(
      0, static_cast<int32_t>((tmp_output_size - 1) * stride + effective_filter_size - input_size));
  if (padding_small) { *padding_small = padding_needed / 2; }
  if (padding_large) { *padding_large = padding_needed - padding_needed / 2; }
  return Maybe<void>::Ok();
}

Maybe<void> CalcConvOut(int64_t input_size, int32_t filter_size, int32_t dilation_rate,
                        int32_t stride, int32_t padding_before, int64_t* output_size) {
  CHECK_GT_OR_RETURN(stride, 0);
  CHECK_GE_OR_RETURN(dilation_rate, 1);

  int32_t effective_filter_size = (filter_size - 1) * dilation_rate + 1;
  if (output_size) {
    *output_size = (input_size + 2 * padding_before - effective_filter_size + stride) / stride;
    CHECK_GE_OR_RETURN((*output_size), 0);
  }
  return Maybe<void>::Ok();
}

const size_t IdxOffset(const std::string& data_format) {
  if (data_format == "channels_first") {
    return 2;
  } else if (data_format == "channels_last") {
    return 1;
  } else {
    UNIMPLEMENTED();
  }
}

const int32_t ChannelIdx(const std::string& data_format, int32_t num_axes) {
  if (data_format == "channels_first") {
    return 1;
  } else if (data_format == "channels_last") {
    return num_axes - 1;
  } else {
    UNIMPLEMENTED();
  }
}

}  // namespace oneflow
