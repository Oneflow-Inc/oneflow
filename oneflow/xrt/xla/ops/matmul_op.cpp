#include "oneflow/xrt/xla/op_context.h"
#include "oneflow/xrt/xla/ops/op_compiler.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace oneflow {
namespace xrt {
namespace mola {

class MatMulOp : public OpCompiler {
 public:
  void Compile(OpContext *ctx) override {
    Shape a_shape = ctx->InputShape("a");
    Shape b_shape = ctx->InputShape("b");
    CHECK_GE(a_shape.NumAxes(), 2);
    CHECK_EQ(a_shape.NumAxes(), b_shape.NumAxes());

    if (a_shape.NumAxes() > 2) {
      auto batch_matmul_compiler =
          CreateOpCompiler(ctx->backend(), "BatchMatMul");
      batch_matmul_compiler->Compile(ctx);
      return;
    }

    bool transpose_a = ctx->GetAttr<bool>("transpose_a");
    bool transpose_b = ctx->GetAttr<bool>("transpose_b");

    xla::XlaOp a = ctx->Input("a");
    xla::XlaOp b = ctx->Input("b");

    auto lhs = transpose_a ? xla::Transpose(a, {1, 0}) : a;
    auto rhs = transpose_b ? xla::Transpose(b, {1, 0}) : b;
    ctx->SetOutput("out", xla::Dot(lhs, rhs));
  }
};

REGISTER_XLA_OP_COMPILER(MatMul, MatMulOp).Finalize();

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow
