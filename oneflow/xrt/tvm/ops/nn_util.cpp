#include "oneflow/xrt/tvm/ops/nn_util.h"

namespace oneflow {
namespace xrt {
namespace of_tvm {

tvm::Array<tvm::relay::IndexExpr> Calc2DPadding(const std::string& padding_format,
    const std::vector<int32_t>& input_size, const std::vector<int32_t>& filter_size,
    const std::vector<int32_t>& stride, const std::vector<int32_t>& dilation) {
  if (padding_format == "valid") { return tvm::Array<tvm::relay::IndexExpr>({0,0,0,0}); }
  CHECK_EQ("same", padding_format);

  tvm::Array<tvm::relay::IndexExpr> padding;
  for (int i = 0; i < 2; ++i) {
    // calc logic is copied from operator/operator_util.cpp:GetWindowedOutputSize()
    int32_t effective_filter_size = (filter_size.at(i) - 1) * dilation.at(i) + 1;
    int32_t tmp_output_size = (input_size.at(i) + stride.at(i) - 1) / stride.at(i);
    int32_t padding_needed = std::max(0,
        (tmp_output_size - 1) * stride.at(i) + effective_filter_size - input_size.at(i));
    int32_t padding_before = padding_needed / 2;
    int32_t padding_after = padding_needed - padding_before;
    padding.push_back(padding_before);
    padding.push_back(padding_after);
  }
  return padding;
}

tvm::Array<tvm::relay::IndexExpr> Calc2DPadding4Pool(const std::string& data_format,
    const std::string& padding_format, const Shape& in_shape,
    const std::vector<int32_t>& pool_size, const std::vector<int32_t>& stride) {
  if (padding_format == "valid") { return tvm::Array<tvm::relay::IndexExpr>({0,0,0,0}); }
  CHECK_EQ("same", padding_format);

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
  auto padding4 = Calc2DPadding(padding_format, input_size, pool_size, stride,
      std::vector<int32_t>{1, 1});
  LOG(INFO) << "padding4: " << padding4[0] << "," <<
    padding4[1] << "," << padding4[2] << "," << padding4[3];
  // order is {top, left, bottom, right}
  return {padding4[1], padding4[3], padding4[0], padding4[2]};
}

}
}
}
