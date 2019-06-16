#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "oneflow/core/compiler/of2xla/xla_op_compiler_registry.h"
#include "oneflow/core/compiler/of2xla/xla_op_compiler.h"
#include "oneflow/core/compiler/of2xla/xla_op_context.h"

namespace oneflow {
namespace mola {

class FullyConnectedOp : public XlaOpCompiler {
 public:
  void Compile(XlaOpContext *ctx) override {
    xla::XlaOp in = ctx->Input("in");
    xla::XlaOp weight = ctx->Input("weight");
    xla::XlaOp result = xla::Dot(in, weight);

    if (ctx->GetAttr<bool>("use_bias")) {
      int feature_dim = 1;
      result = xla::Add(result, ctx->Input("bias"), {feature_dim});
    }
    ctx->SetOutput("out", result);
  }
};

REGISTER_XLA_OP_COMPILER(FullyConnected, FullyConnectedOp)

}  // namespace mola
}  // namespace oneflow
