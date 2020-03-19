#include "oneflow/xrt/tvm/ops/tvm_op_kernel.h"
#include <tvm/relay/attrs/nn.h>

namespace oneflow {
namespace xrt {
namespace of_tvm {

namespace {

std::string GetKernelLayout(const std::string& data_format) {
  if (data_format == "NCHW") { return "OIHW"; }
  else { return "OHWI"; }
}

tvm::Array<tvm::relay::IndexExpr> CalcPadding(const std::string& data_format, 
    const std::string& padding_format, const Shape& in_shape, const Shape& weight_shape,
    const std::vector<int32_t>& stride, const std::vector<int32_t>& dilation) {
  if (padding_format == "valid") {
    return tvm::Array<tvm::relay::IndexExpr>({0, 0});
  } else {
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
    std::vector<int32_t> filter_size;
    if (GetKernelLayout(data_format) == "OIHW") {
      filter_size = Int64VecToInt32Vec(
          std::vector<int64_t>{weight_shape.At(2), weight_shape.At(3)});
    } else {
      filter_size = Int64VecToInt32Vec(
        std::vector<int64_t>{weight_shape.At(1), weight_shape.At(2)});
    }

    tvm::Array<tvm::relay::IndexExpr> padding;
    for (int i = 0; i < 2; ++i) {
      // calc logic is copied from operator/operator_util.cpp:GetWindowedOutputSize()
      int32_t effective_filter_size = (filter_size.at(i) - 1) * dilation.at(i) + 1;
      int32_t tmp_output_size = (input_size.at(i) + stride.at(i) - 1) / stride.at(i);
      int32_t padding_needed = std::max(0,
          (tmp_output_size - 1) * stride.at(i) + effective_filter_size - input_size.at(i));
      int32_t padding_after = padding_needed - padding_needed / 2;
      padding.push_back(padding_after);
    }
    return padding;
  }
}

}

class Conv2DOp final : public TVMOpKernel {
 public:
  void Compile(TVMOpContext* ctx) override {
    tvm::Array<tvm::relay::Expr> inputs;
    inputs.push_back(ctx->GetExpr4InputName("in"));
    inputs.push_back(ctx->GetExpr4InputName("weight"));

    auto conv_attrs = tvm::make_node<tvm::relay::Conv2DAttrs>();
    {
      std::string data_format = ctx->GetAttr<std::string>("data_format");
      CHECK(data_format == "channels_last" || data_format == "channels_first") 
        << "Wrong data_format: " << data_format;
      if (data_format == "channels_first") {
        data_format = "NCHW";
      } else {
        data_format = "NHWC";
      }

      conv_attrs->data_layout = data_format;
      conv_attrs->channels = ctx->GetAttr<int32_t>("filters");
      conv_attrs->kernel_layout = GetKernelLayout(data_format);

      std::vector<int32_t> stride = ctx->GetAttr<std::vector<int32_t>>("strides");
      CHECK_EQ(2, stride.size());
      conv_attrs->strides = tvm::Array<tvm::relay::IndexExpr>({stride.at(0), stride.at(1)});

      std::vector<int32_t> dilation = ctx->GetAttr<std::vector<int32_t>>("dilation_rate");
      CHECK_EQ(2, dilation.size());
      conv_attrs->dilation = tvm::Array<tvm::relay::IndexExpr>({dilation.at(0), dilation.at(1)});

      conv_attrs->padding = CalcPadding(data_format, ctx->GetAttr<std::string>("padding"),
          ctx->GetShape4InputName("in"), ctx->GetShape4InputName("weight"), stride, dilation);
      LOG(INFO) << "TVMLOG padding: " << conv_attrs->padding[0] << ',' << conv_attrs->padding[1];

      std::vector<int32_t> kernel_size = ctx->GetAttr<std::vector<int32_t>>("kernel_size");
      CHECK_EQ(2, kernel_size.size());
      conv_attrs->kernel_size =
        tvm::Array<tvm::relay::IndexExpr>({kernel_size.at(0), kernel_size.at(1)});

      // though the default value of groups is 1 in tvm::relay::Conv2DAttrs,
      // but we still need to set it explicitly
      conv_attrs->groups = 1;
    }

    auto conv_op = tvm::relay::Op::Get("nn.conv2d");
    auto conv = tvm::relay::CallNode::make(conv_op, inputs, tvm::Attrs(conv_attrs), {});

    bool use_bias = ctx->GetAttr<bool>("use_bias");
    if (use_bias) {
      auto bias_add_attrs = tvm::make_node<tvm::relay::BiasAddAttrs>();
      bias_add_attrs->axis = 1;

      auto bias_add_in = ctx->GetExpr4InputName("bias");
      auto bias_add_op = tvm::relay::Op::Get("nn.bias_add");
      auto bias_add = tvm::relay::CallNode::make(
          bias_add_op, {conv, bias_add_in}, tvm::Attrs(bias_add_attrs), {});
      ctx->set_op_expr(bias_add);
    } else {
      ctx->set_op_expr(conv);
    }
  }
};

REGISTER_TVM_OP_KERNEL(Conv2D, Conv2DOp).EnableTrainPhase().Finalize();

}
}
}
