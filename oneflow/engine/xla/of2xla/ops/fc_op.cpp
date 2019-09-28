#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "oneflow/engine/xla/of2xla/xla_op_compiler_registry.h"
#include "oneflow/engine/xla/of2xla/xla_op_compiler.h"
#include "oneflow/engine/xla/of2xla/xla_op_context.h"

namespace oneflow {
namespace mla {

class FullyConnectedOp : public XlaOpCompiler {
 public:
  void Compile(XlaOpContext *ctx) override {
    xla::XlaOp in = ctx->Input("in");
    xla::XlaOp weight = xla::Transpose(ctx->Input("weight"), {1, 0});
    xla::XlaOp result = xla::Dot(in, weight);

    if (ctx->GetAttr<bool>("use_bias")) {
      result = xla::Add(result, ctx->Input("bias"));
    }
    ctx->SetOutput("out", result);
  }
};

REGISTER_XLA_OP_COMPILER(FullyConnected, FullyConnectedOp);

}  // namespace mla
}  // namespace oneflow
