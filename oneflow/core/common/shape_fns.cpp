#include "oneflow/core/common/shape_fns.h"

namespace oneflow {

void GetWindowedOutputSizeVerboseV2(int32_t input_size, int32_t filter_size,
                                    int32_t dilation_rate, int32_t stride,
                                    const std::string& padding_type,
                                    int32_t* output_size,
                                    int32_t* padding_before,
                                    int32_t* padding_after) {
  if (stride <= 0) { LOG(FATAL) << "Stride must be > 0, but got " << stride; }
  if (dilation_rate < 1) {
    LOG(FATAL) << "Dilation rate must be >= 1, but got " << dilation_rate;
  }

  // See also the parallel implementation in GetWindowedOutputSizeFromDimsV2.
  int32_t effective_filter_size = (filter_size - 1) * dilation_rate + 1;
  if (padding_type == "valid") {
    *output_size = (input_size - effective_filter_size + stride) / stride;
    *padding_before = *padding_after = 0;
  } else if (padding_type == "same") {
    *output_size = (input_size + stride - 1) / stride;
    const int32_t padding_needed = std::max(
        static_cast<int32_t>(0),
        (*output_size - 1) * stride + effective_filter_size - input_size);
    // For odd values of total padding, add more padding at the 'right'
    // side of the given dimension.
    *padding_before = padding_needed / 2;
    *padding_after = padding_needed - *padding_before;
  } else {
    UNEXPECTED_RUN();
  }
  if (*output_size < 0) {
    LOG(FATAL) << "computed output size would be negative";
  }
}

void GetWindowedOutputSizeVerbose(int32_t input_size, int32_t filter_size,
                                  int32_t stride,
                                  const std::string& padding_type,
                                  int32_t* output_size, int32_t* padding_before,
                                  int32_t* padding_after) {
  GetWindowedOutputSizeVerboseV2(input_size, filter_size,
                                 /*dilation_rate=*/1, stride, padding_type,
                                 output_size, padding_before, padding_after);
}

void GetWindowedOutputSize(int32_t input_size, int32_t filter_size,
                           int32_t stride, const std::string& padding_type,
                           int32_t* output_size, int32_t* padding_size) {
  int32_t padding_after_unused;
  GetWindowedOutputSizeVerbose(input_size, filter_size, stride, padding_type,
                               output_size, padding_size,
                               &padding_after_unused);
}

void GetWindowedOutputSizeV2(int32_t input_size, int32_t filter_size,
                             int32_t dilation_rate, int32_t stride,
                             const std::string& padding_type,
                             int32_t* output_size, int32_t* padding_size) {
  int32_t padding_after_unused;
  GetWindowedOutputSizeVerboseV2(input_size, filter_size, dilation_rate, stride,
                                 padding_type, output_size, padding_size,
                                 &padding_after_unused);
}

}  // namespace oneflow
