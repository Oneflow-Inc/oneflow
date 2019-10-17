#include "oneflow/xrt/of2xla/xla_op_compiler.h"
#include "oneflow/xrt/of2xla/xla_op_compiler_registry.h"
#include "oneflow/xrt/of2xla/xla_op_context.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

#include "oneflow/xrt/of2xla/xla_helpers.h"

namespace oneflow {
namespace mola {

class ReshapeOp : public XlaOpCompiler {
 public:
  void Compile(XlaOpContext *ctx) override {
    Shape in_shape = ctx->InputShape("in");
    Shape shape = ctx->OutputShape("out");
    CHECK_EQ(shape.Count(0), in_shape.Count(0));

    ctx->SetOutput("out", Reshape(ctx->Input("in"), shape));
  }
};

REGISTER_XLA_OP_COMPILER(Reshape, ReshapeOp);

class ReshapeLikeOp : public XlaOpCompiler {
 public:
  void Compile(XlaOpContext *ctx) override {
    Shape x_shape = ctx->InputShape("x");
    Shape like_shape = ctx->InputShape("like");
    CHECK_EQ(x_shape.Count(0), like_shape.Count(0));

    ctx->SetOutput("y", Reshape(ctx->Input("x"), like_shape));
  }
};

REGISTER_XLA_OP_COMPILER(ReshapeLike, ReshapeLikeOp);

}  // namespace mola
}  // namespace oneflow
