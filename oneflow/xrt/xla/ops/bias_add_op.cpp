#include "oneflow/xrt/xla/ops/op_context.h"
#include "oneflow/xrt/xla/ops/op_kernel.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace oneflow {
namespace xrt {
namespace mola {

class BiasAddOp : public XlaOpKernel {
 public:
  void Compile(XlaOpContext *ctx) override {
    Shape in_shape = ctx->InputShape("a_0");
    Shape bias_shape = ctx->InputShape("b_0");
    CHECK_GE(in_shape.NumAxes(), 2);
    CHECK_EQ(bias_shape.NumAxes(), 1);

    CHECK_EQ(ctx->InputType("a_0"), ctx->InputType("b_0"));

    xla::XlaOp in = ctx->Input("a_0");
    xla::XlaOp bias = ctx->Input("b_0");

    // Channel dim for NCHW data formart
    int channel_dim = 1;
    ctx->SetOutput("out_0", xla::Add(in, bias, {channel_dim}));
  }
};

REGISTER_XLA_OP_KERNEL(BiasAdd, BiasAddOp).Finalize();

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow
