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
#include "oneflow/xrt/tvm/ops/nn_util.h"

namespace oneflow {
namespace xrt {
namespace of_tvm {

tvm::Array<tvm::relay::IndexExpr> Calc2DPadding(const std::string& padding_type,
                                                const std::vector<int32_t>& input_size,
                                                const std::vector<int32_t>& filter_size,
                                                const std::vector<int32_t>& stride,
                                                const std::vector<int32_t>& dilation) {
  if (padding_type == "valid") { return tvm::Array<tvm::relay::IndexExpr>({0, 0, 0, 0}); }

  tvm::Array<tvm::relay::IndexExpr> padding;
  for (int i = 0; i < 2; ++i) {
    // calc logic is copied from operator/operator_util.cpp:GetWindowedOutputSize()
    // TODO: (bowenc) try to reuse code above
    const int32_t effective_filter_size = (filter_size.at(i) - 1) * dilation.at(i) + 1;
    const int32_t tmp_output_size = (input_size.at(i) + stride.at(i) - 1) / stride.at(i);
    const int32_t padding_needed = std::max(
        0, (tmp_output_size - 1) * stride.at(i) + effective_filter_size - input_size.at(i));

    const int32_t padding_small = padding_needed / 2;
    const int32_t padding_large = padding_needed - padding_needed / 2;
    int32_t padding_before = 0;
    int32_t padding_after = 0;

    if (padding_type == "same_upper") {
      padding_before = padding_small;
      padding_after = padding_large;
    } else if (padding_type == "same_lower") {
      padding_before = padding_large;
      padding_after = padding_small;
    } else {
      UNIMPLEMENTED() << "padding_type " << padding_type << "not suported.";
    }
    padding.push_back(padding_before);
    padding.push_back(padding_after);
  }
  return padding;
}

tvm::Array<tvm::relay::IndexExpr> Calc2DPadding4Pool(const std::string& data_format,
                                                     const std::string& padding_format,
                                                     const Shape& in_shape,
                                                     const std::vector<int32_t>& pool_size,
                                                     const std::vector<int32_t>& stride) {
  if (padding_format == "valid") { return tvm::Array<tvm::relay::IndexExpr>({0, 0, 0, 0}); }

  auto Int64VecToInt32Vec = [](const std::vector<int64_t>& vec) -> std::vector<int32_t> {
    std::vector<int32_t> ret;
    for (int64_t val : vec) { ret.push_back(static_cast<int32_t>(val)); }
    return ret;
  };
  std::vector<int32_t> input_size;
  if (data_format == "NCHW") {
    input_size = Int64VecToInt32Vec(std::vector<int64_t>{in_shape.At(2), in_shape.At(3)});
  } else {
    input_size = Int64VecToInt32Vec(std::vector<int64_t>{in_shape.At(1), in_shape.At(2)});
  }
  auto padding4 =
      Calc2DPadding(padding_format, input_size, pool_size, stride, std::vector<int32_t>{1, 1});
  // order is {top, left, bottom, right}
  return {padding4[1], padding4[3], padding4[0], padding4[2]};
}

}  // namespace of_tvm
}  // namespace xrt
}  // namespace oneflow
