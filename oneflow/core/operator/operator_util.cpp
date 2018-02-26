#include "oneflow/core/operator/operator_util.h"

namespace oneflow {

void GetWindowedOutputSize(int64_t input_size, int32_t filter_size,
                           int32_t dilation_rate, int32_t stride,
                           const std::string& padding_type,
                           int64_t* output_size, int32_t* pad_small_side,
                           int32_t* pad_large_side) {
  CHECK_GT(stride, 0);
  CHECK_GE(dilation_rate, 1);

  int32_t effective_filter_size = (filter_size - 1) * dilation_rate + 1;
  if (padding_type == "valid") {
    *output_size = (input_size - effective_filter_size + stride) / stride;
    if (pad_small_side) { *pad_small_side = 0; }
    if (pad_large_side) { *pad_large_side = 0; }
  } else if (padding_type == "same") {
    *output_size = (input_size + stride - 1) / stride;
    const int32_t padding_needed =
        std::max(static_cast<int32_t>(0),
                 static_cast<int32_t>((*output_size - 1) * stride
                                      + effective_filter_size - input_size));
    // For odd values of total padding, add more padding at the 'right'
    // side of the given dimension.
    if (pad_small_side) { *pad_small_side = padding_needed / 2; }
    if (pad_large_side) { *pad_large_side = padding_needed - *pad_large_side; }
  } else {
    UNEXPECTED_RUN();
  }
  CHECK_GE((*output_size), 0);
}

void GetWindowedOutputSize(int64_t input_size, int32_t filter_size,
                           int32_t stride, const std::string& padding_type,
                           int64_t* output_size, int32_t* pad_small_side,
                           int32_t* pad_large_side) {
  GetWindowedOutputSize(input_size, filter_size, 1, stride, padding_type,
                        output_size, pad_small_side, pad_large_side);
}

void GetWindowedOutputSize(int64_t input_size, int32_t filter_size,
                           int32_t stride, const std::string& padding_type,
                           int64_t* output_size, int32_t* padding_size) {
  GetWindowedOutputSize(input_size, filter_size, stride, padding_type,
                        output_size, padding_size, nullptr);
}

}  // namespace oneflow
