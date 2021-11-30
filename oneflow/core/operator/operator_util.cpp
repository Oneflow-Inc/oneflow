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
#include "oneflow/core/operator/operator_util.h"
#include "oneflow/core/framework/user_op_conf.h"

namespace oneflow {

size_t DhwOffset(const std::string& data_format) {
  if (data_format == "channels_first") {
    return 2;
  } else if (data_format == "channels_last") {
    return 1;
  } else {
    UNIMPLEMENTED();
  }
}

std::vector<int32_t> Get3DVecInOpConf(const PbRf<int32_t>& field_vals, int32_t NDims) {
  std::vector<int32_t> vec;
  vec.reserve(3);
  FOR_RANGE(uint8_t, dim, 0, 3) {
    int64_t index = static_cast<int64_t>(dim) - (3 - NDims);
    if (index < 0) {
      vec.emplace_back(1);
    } else {
      vec.emplace_back(field_vals.Get(index));
    }
  }
  return vec;
}

int64_t GetInDim(const ShapeView& shape, const std::string& data_format, int32_t dim,
                 int32_t NDims) {
  int64_t offset = 0;
  if (data_format == "channels_last") {
    offset = 1;
  } else if (data_format == "channels_first") {
    offset = 2;
  } else {
    UNIMPLEMENTED();
  }
  int64_t index = offset + static_cast<int64_t>(dim) - static_cast<int64_t>(3 - NDims);
  if (index < offset) {
    return 1;
  } else {
    return shape.At(index);
  }
}

void GetWindowedOutputSize(int64_t input_size, int32_t filter_size, int32_t dilation_rate,
                           int32_t stride, const std::string& padding_type, bool ceil_mode,
                           int64_t* output_size, int32_t* padding_before, int32_t* padding_after) {
  CHECK_GT(stride, 0);
  CHECK_GE(dilation_rate, 1);

  int32_t effective_filter_size = (filter_size - 1) * dilation_rate + 1;
  if (padding_type == "customized") {
    if (output_size) {
      *output_size = (input_size + *padding_before + *padding_after - effective_filter_size + stride
                      + (ceil_mode ? stride - 1 : 0))
                     / stride;
      CHECK_GE((*output_size), 0);
    }
  } else if (padding_type == "valid") {
    if (output_size) { *output_size = (input_size - effective_filter_size + stride) / stride; }
    if (padding_before) { *padding_before = 0; }
    if (padding_after) { *padding_after = 0; }
  } else {
    int64_t tmp_output_size = (input_size + stride - 1) / stride;
    if (output_size) { *output_size = tmp_output_size; }
    const int32_t padding_needed = std::max(
        0,
        static_cast<int32_t>((tmp_output_size - 1) * stride + effective_filter_size - input_size));
    const int32_t padding_small = padding_needed / 2;
    const int32_t padding_large = padding_needed - padding_needed / 2;
    if (padding_type == "same_upper") {
      if (padding_before) { *padding_before = padding_small; }
      if (padding_after) { *padding_after = padding_large; }
    } else if (padding_type == "same_lower") {
      if (padding_before) { *padding_before = padding_large; }
      if (padding_after) { *padding_after = padding_small; }
    } else {
      UNIMPLEMENTED();
    }
  }
  if (output_size) { CHECK_GE((*output_size), 0); }
}

void GetWindowedOutputSize(int64_t input_size, int32_t filter_size, int32_t dilation_rate,
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

void GetWindowedOutputSize(int64_t input_size, int32_t filter_size, int32_t stride,
                           const std::string& padding_type, int64_t* output_size,
                           int32_t* padding_before, int32_t* padding_after) {
  GetWindowedOutputSize(input_size, filter_size, 1, stride, padding_type, output_size,
                        padding_before, padding_after);
}

void GetWindowedOutputSize(int64_t input_size, int32_t filter_size, int32_t stride,
                           const std::string& padding_type, int64_t* output_size,
                           int32_t* padding_size) {
  GetWindowedOutputSize(input_size, filter_size, stride, padding_type, output_size, padding_size,
                        nullptr);
}

void Get3DOutputSize(const DimVector& in, const std::vector<int32_t>& pool_size,
                     const std::vector<int32_t>& strides, const std::string& padding_type,
                     DimVector* out, std::vector<int32_t>* padding) {
  Get3DOutputSize(in, pool_size, strides, padding_type, out, padding, nullptr, nullptr);
}

void Get3DOutputSize(const DimVector& in, const std::vector<int32_t>& pool_size,
                     const std::vector<int32_t>& strides, const std::string& padding_type,
                     DimVector* out, std::vector<int32_t>* padding_before,
                     std::vector<int32_t>* padding_after) {
  Get3DOutputSize(in, pool_size, strides, padding_type, out, padding_before, padding_after,
                  nullptr);
}

void Get3DOutputSize(const DimVector& in, const std::vector<int32_t>& pool_size,
                     const std::vector<int32_t>& strides, const std::string& padding_type,
                     DimVector* out, std::vector<int32_t>* padding_before,
                     std::vector<int32_t>* padding_after, std::vector<int32_t>* dilation_rate) {
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
    int32_t* padding_before_ptr = padding_before ? (&(*padding_before).at(i)) : nullptr;
    int32_t* padding_after_ptr = padding_after ? (&(*padding_after).at(i)) : nullptr;
    if (dilation_rate) {
      GetWindowedOutputSize(in.at(i), pool_size.at(i), dilation_rate->at(i), strides.at(i),
                            padding_type, out_ptr, padding_before_ptr, padding_after_ptr);
    } else {
      GetWindowedOutputSize(in.at(i), pool_size.at(i), strides.at(i), padding_type, out_ptr,
                            padding_before_ptr, padding_after_ptr);
    }
  }
}

void Get3DOutputSize(const DimVector& in, const std::vector<int32_t>& pool_size,
                     const std::vector<int32_t>& strides, const std::string& padding_type,
                     const bool ceil_mode, std::vector<int32_t>* dilation_rate, DimVector* out,
                     std::vector<int32_t>* padding_before, std::vector<int32_t>* padding_after) {
  CHECK(out);
  out->clear();
  out->resize(3);
  FOR_RANGE(size_t, i, 0, 3) {
    int64_t* out_ptr = &(*out).at(i);
    if (dilation_rate) {
      GetWindowedOutputSize(in.at(i), pool_size.at(i), dilation_rate->at(i), strides.at(i),
                            padding_type, ceil_mode, out_ptr, &(padding_before->at(i)),
                            &(padding_after->at(i)));
    } else {
      GetWindowedOutputSize(in.at(i), pool_size.at(i), 1, strides.at(i), padding_type, ceil_mode,
                            out_ptr, &(padding_before->at(i)), &(padding_after->at(i)));
    }
  }
}

void GetConvOutAndPad(const ShapeView& in_blob_shape, const PbMessage& conv_conf, DimVector* out,
                      std::vector<int32_t>* pad_small_side, std::vector<int32_t>* pad_large_side) {
  int32_t opkernel_dim = in_blob_shape.NumAxes() - 2;
  if (out) { out->assign(opkernel_dim, 0); }
  if (pad_small_side) { pad_small_side->assign(opkernel_dim, 0); }
  if (pad_large_side) { pad_large_side->assign(opkernel_dim, 0); }
  const auto& data_format = GetValFromPbMessage<std::string>(conv_conf, "data_format");
  const std::string& padding = GetValFromPbMessage<std::string>(conv_conf, "padding");
  const PbRf<int32_t>& dilation_rate = GetPbRfFromPbMessage<int32_t>(conv_conf, "dilation_rate");
  const auto& strides = GetPbRfFromPbMessage<int32_t>(conv_conf, "strides");
  const PbRf<int32_t>& kernel_size = GetPbRfFromPbMessage<int32_t>(conv_conf, "kernel_size");
  FOR_RANGE(int32_t, i, 0, opkernel_dim) {
    GetWindowedOutputSize(in_blob_shape.At(DhwOffset(data_format) + i), kernel_size.Get(i),
                          dilation_rate.Get(i), strides.Get(i), padding,
                          out ? &(out->at(i)) : nullptr,
                          pad_small_side ? &(pad_small_side->at(i)) : nullptr,
                          pad_large_side ? &(pad_large_side->at(i)) : nullptr);
  }
}

void GetConvOutAndPad(const ShapeView& in_blob_shape, const user_op::UserOpConfWrapper& conv_conf,
                      DimVector* out, std::vector<int32_t>* pad_small_side,
                      std::vector<int32_t>* pad_large_side) {
  int32_t opkernel_dim = in_blob_shape.NumAxes() - 2;
  if (out) { out->assign(opkernel_dim, 0); }
  if (pad_small_side) { pad_small_side->assign(opkernel_dim, 0); }
  if (pad_large_side) { pad_large_side->assign(opkernel_dim, 0); }
  const auto& data_format = conv_conf.attr<std::string>("data_format");
  const auto& padding = conv_conf.attr<std::string>("padding");
  const auto& strides = conv_conf.attr<std::vector<int32_t>>("strides");
  const auto& dilation_rate = conv_conf.attr<std::vector<int32_t>>("dilation_rate");
  const auto& kernel_size = conv_conf.attr<std::vector<int32_t>>("kernel_size");
  FOR_RANGE(int32_t, i, 0, opkernel_dim) {
    GetWindowedOutputSize(in_blob_shape.At(DhwOffset(data_format) + i), kernel_size.at(i),
                          dilation_rate.at(i), strides.at(i), padding,
                          out ? &(out->at(i)) : nullptr,
                          pad_small_side ? &(pad_small_side->at(i)) : nullptr,
                          pad_large_side ? &(pad_large_side->at(i)) : nullptr);
  }
}

}  // namespace oneflow
