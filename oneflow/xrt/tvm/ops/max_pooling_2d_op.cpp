#include "oneflow/xrt/tvm/ops/tvm_op_kernel.h"
#include <tvm/relay/attrs/nn.h>
#include "oneflow/xrt/tvm/ops/nn_util.h"

namespace oneflow {
namespace xrt {
namespace of_tvm {

class MaxPooling2DOp final : public TVMOpKernel {
 public:
  void Compile(TVMOpContext* ctx) override {
    tvm::Array<tvm::relay::Expr> node_inputs;
    node_inputs.push_back(ctx->GetExpr4InputName("in"));

    auto attrs = tvm::make_node<tvm::relay::MaxPool2DAttrs>();
    {
      std::string data_format = ctx->GetAttr<std::string>("data_format");
      CHECK(data_format == "channels_last" || data_format == "channels_first") 
        << "Wrong data_format: " << data_format;
      if (data_format == "channels_first") {
        data_format = "NCHW";
      } else {
        data_format = "NHWC";
      }
      attrs->layout = data_format;

      std::vector<int32_t> strides = ctx->GetAttr<std::vector<int32_t>>("strides");
      CHECK_EQ(2, strides.size());
      attrs->strides = tvm::Array<tvm::relay::IndexExpr>({strides.at(0), strides.at(1)});

      std::vector<int32_t> pool_size = ctx->GetAttr<std::vector<int32_t>>("pool_size");
      CHECK_EQ(2, pool_size.size());
      attrs->pool_size = tvm::Array<tvm::relay::IndexExpr>({pool_size.at(0), pool_size.at(1)});

      attrs->padding = Calc2DPadding4Pool(data_format, ctx->GetAttr<std::string>("padding"),
          ctx->GetShape4InputName("in"), pool_size, strides);

      attrs->ceil_mode = false;
    }

    auto op = tvm::relay::Op::Get("nn.max_pool2d");
    auto expr = tvm::relay::CallNode::make(op, node_inputs, tvm::Attrs(attrs), {});
    ctx->SetExpr4OutputName("out", std::move(expr));
  }
};

REGISTER_TVM_OP_KERNEL(MaxPooling2D, MaxPooling2DOp).Finalize();

}
}
}
