#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "oneflow/engine/xla/of2xla/xla_op_compiler_registry.h"
#include "oneflow/engine/xla/of2xla/xla_op_compiler.h"
#include "oneflow/engine/xla/of2xla/xla_op_context.h"

namespace oneflow {
namespace mola {

class BiasAddOp : public XlaOpCompiler {
 public:
  void Compile(XlaOpContext *ctx) override {
    Shape in_shape = ctx->InputShape("a");
    Shape bias_shape = ctx->InputShape("b");
    CHECK_GE(in_shape.NumAxes(), 2);
    CHECK_EQ(bias_shape.NumAxes(), 1);

    xla::XlaOp in = ctx->Input("a");
    xla::XlaOp bias = ctx->Input("b");
    
    // Channel dim for NCHW data formart
    int channel_dim = 1;
    ctx->SetOutput("out", xla::Add(in, bias, {channel_dim}));
  }
};

REGISTER_XLA_OP_COMPILER(BiasAdd, BiasAddOp)

}  // namespace mola
}  // namespace oneflow
