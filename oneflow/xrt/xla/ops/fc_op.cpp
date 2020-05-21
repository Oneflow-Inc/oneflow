#include "oneflow/xrt/xla/ops/op_context.h"
#include "oneflow/xrt/xla/ops/op_kernel.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace oneflow {
namespace xrt {
namespace mola {

class FullyConnectedOp : public XlaOpKernel {
 public:
  void Compile(XlaOpContext *ctx) override {
    xla::XlaOp in = ctx->Input("in");
    xla::XlaOp weight = xla::Transpose(ctx->Input("weight"), {1, 0});
    xla::XlaOp result = xla::Dot(in, weight);

    if (ctx->Attr<bool>("use_bias")) { result = xla::Add(result, ctx->Input("bias")); }
    ctx->SetOutput("out", result);
  }
};

REGISTER_XLA_OP_KERNEL(FullyConnected, FullyConnectedOp).Finalize();

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow
