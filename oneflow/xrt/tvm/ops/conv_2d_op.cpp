#include "oneflow/xrt/tvm/ops/op_kernel.h"
#include <tvm/relay/attrs/nn.h>
#include "oneflow/xrt/tvm/ops/nn_util.h"

namespace oneflow {
namespace xrt {
namespace of_tvm {

namespace {

std::string GetKernelLayout(const std::string& data_format) {
  if (data_format == "NCHW") { return "OIHW"; }
  else { return "OHWI"; }
}

tvm::Array<tvm::relay::IndexExpr> Calc2DPadding4Conv(const std::string& data_format,
    const std::string& padding_format, const Shape& in_shape, const Shape& weight_shape,
    const std::vector<int32_t>& stride, const std::vector<int32_t>& dilation) {
  if (padding_format == "valid") { return tvm::Array<tvm::relay::IndexExpr>({0, 0}); }

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
  std::vector<int32_t> filter_size;
  if (GetKernelLayout(data_format) == "OIHW") {
    filter_size = Int64VecToInt32Vec(
        std::vector<int64_t>{weight_shape.At(2), weight_shape.At(3)});
  } else {
    filter_size = Int64VecToInt32Vec(
      std::vector<int64_t>{weight_shape.At(1), weight_shape.At(2)});
  }
  auto padding4 = Calc2DPadding(padding_format, input_size, filter_size, stride, dilation);
  // only need padding_after for conv
  return {padding4[1], padding4[3]};
}

}

class Conv2DOp final : public TVMOpKernel {
 public:
  void Compile(TVMOpContext* ctx) override {
    LOG(WARNING) << ctx->DebugStr();
    
    tvm::Array<tvm::relay::Expr> inputs;
    inputs.push_back(ctx->GetExpr4InputName("in_0"));
    inputs.push_back(ctx->GetExpr4InputName("weight_0"));

    auto conv_attrs = tvm::runtime::make_object<tvm::relay::Conv2DAttrs>();
    {
      std::string data_format = ctx->Attr<std::string>("data_format");
      CHECK(data_format == "channels_last" || data_format == "channels_first")
        << "Wrong data_format: " << data_format;
      if (data_format == "channels_first") {
        data_format = "NCHW";
      } else {
        data_format = "NHWC";
      }

      conv_attrs->data_layout = data_format;
      conv_attrs->channels = ctx->Attr<int32_t>("filters");
      conv_attrs->kernel_layout = GetKernelLayout(data_format);

      std::vector<int32_t> stride = ctx->Attr<std::vector<int32_t>>("strides");
      CHECK_EQ(2, stride.size());
      conv_attrs->strides = tvm::Array<tvm::relay::IndexExpr>({stride.at(0), stride.at(1)});

      std::vector<int32_t> dilation = ctx->Attr<std::vector<int32_t>>("dilation_rate");
      CHECK_EQ(2, dilation.size());
      conv_attrs->dilation = tvm::Array<tvm::relay::IndexExpr>({dilation.at(0), dilation.at(1)});

      std::vector<int32_t> padding = ctx->Attr<std::vector<int32_t>>("padding_before");
      conv_attrs->padding = tvm::Array<tvm::relay::IndexExpr>({padding.at(0), padding.at(1)});

      std::vector<int32_t> kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
      CHECK_EQ(2, kernel_size.size());
      conv_attrs->kernel_size =
        tvm::Array<tvm::relay::IndexExpr>({kernel_size.at(0), kernel_size.at(1)});

      // though the default value of groups is 1 in tvm::relay::Conv2DAttrs,
      // but we still need to set it explicitly
      conv_attrs->groups = 1;
    }

    auto conv_op = tvm::relay::Op::Get("nn.conv2d");
    auto conv = tvm::relay::Call(conv_op, inputs, tvm::Attrs(conv_attrs), {});

    ctx->SetExpr4OutputName("out_0", std::move(conv));
  }
};

REGISTER_TVM_OP_KERNEL(Conv2D, Conv2DOp).Finalize();

}  // namespace of_tvm
}  // namespace xrt
}  // namespace oneflow
