#include "oneflow/customized/ops/nn_util.h"

namespace oneflow {

void CalcOutAndPadding(int64_t input_size, int32_t filter_size, int32_t dilation_rate,
                       int32_t stride, const std::string& padding_type, int64_t* output_size,
                       int32_t* padding_before, int32_t* padding_after) {
  CHECK_GT(stride, 0);
  CHECK_GE(dilation_rate, 1);

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
  if (output_size) { CHECK_GE((*output_size), 0); }
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

}  // namespace oneflow
