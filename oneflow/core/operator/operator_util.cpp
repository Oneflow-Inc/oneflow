#include "oneflow/core/operator/operator_util.h"

namespace oneflow {

void GetWindowedOutputSize(int64_t input_size, int64_t filter_size,
                           int64_t dilation_rate, int64_t stride,
                           const std::string& padding_type,
                           int64_t* output_size, int64_t* padding_before,
                           int64_t* padding_after) {
  CHECK_GT(stride, 0);
  CHECK_GE(dilation_rate, 1);

  int64_t effective_filter_size = (filter_size - 1) * dilation_rate + 1;
  if (padding_type == "valid") {
    *output_size = (input_size - effective_filter_size + stride) / stride;
    if (padding_before) { *padding_before = 0; }
    if (padding_after) { *padding_after = 0; }
  } else if (padding_type == "same") {
    *output_size = (input_size + stride - 1) / stride;
    const int64_t padding_needed = std::max(
        static_cast<int64_t>(0),
        (*output_size - 1) * stride + effective_filter_size - input_size);
    // For odd values of total padding, add more padding at the 'right'
    // side of the given dimension.
    if (padding_before) { *padding_before = padding_needed / 2; }
    if (padding_after) { *padding_after = padding_needed - *padding_before; }
  } else {
    UNEXPECTED_RUN();
  }
  CHECK_GE((*output_size), 0);
}

void GetWindowedOutputSize(int64_t input_size, int64_t filter_size,
                           int64_t stride, const std::string& padding_type,
                           int64_t* output_size, int64_t* padding_before,
                           int64_t* padding_after) {
  GetWindowedOutputSize(input_size, filter_size, 1, stride, padding_type,
                        output_size, padding_before, padding_after);
}

void GetWindowedOutputSize(int64_t input_size, int64_t filter_size,
                           int64_t stride, const std::string& padding_type,
                           int64_t* output_size, int64_t* padding_size) {
  GetWindowedOutputSize(input_size, filter_size, stride, padding_type,
                        output_size, padding_size, nullptr);
}

void Get3DOutputSize(const std::vector<int64_t>& in,
                     const std::vector<int64_t>& pool_size,
                     const std::vector<int64_t>& strides,
                     const std::string& padding_type, std::vector<int64_t>* out,
                     std::vector<int64_t>* padding) {
  Get3DOutputSize(in, pool_size, strides, padding_type, out, padding, nullptr);
}

void Get3DOutputSize(const std::vector<int64_t>& in,
                     const std::vector<int64_t>& pool_size,
                     const std::vector<int64_t>& strides,
                     const std::string& padding_type, std::vector<int64_t>* out,
                     std::vector<int64_t>* padding_before,
                     std::vector<int64_t>* padding_after) {
  CHECK(out);
  out->clear();
  out->resize(3);
  if (padding_before) {
    padding_before->clear();
    padding_before->resize(3);
  }
  if (padding_after) {
    padding_after->clear();
    padding_after->resize(3);
  }
  FOR_RANGE(size_t, i, 0, 3) {
    int64_t* out_ptr = &(*out).at(i);
    int64_t* padding_before_ptr =
        padding_before ? (&(*padding_before).at(i)) : nullptr;
    int64_t* padding_after_ptr =
        padding_after ? (&(*padding_after).at(i)) : nullptr;
    GetWindowedOutputSize(in.at(i), pool_size.at(i), strides.at(i),
                          padding_type, out_ptr, padding_before_ptr,
                          padding_after_ptr);
  }
}

}  // namespace oneflow
