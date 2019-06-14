#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "oneflow/core/compiler/of2xla/xla_op_compiler_registry.h"
#include "oneflow/core/compiler/of2xla/xla_op_compiler.h"
#include "oneflow/core/compiler/of2xla/xla_op_context.h"

namespace oneflow {
namespace mola {

class MatMulOp : public XlaOpCompiler {
 public:
  void Compile(XlaOpContext *ctx) override {
    bool transpose_a = ctx->GetAttr<bool>("transpose_a");
    bool transpose_b = ctx->GetAttr<bool>("transpose_b");

    xla::XlaOp a = ctx->Input(0);
    xla::XlaOp b = ctx->Input(1);

    auto lhs = transpose_a ? xla::Transpose(a, {1, 0}) : a;
    auto rhs = transpose_b ? xla::Transpose(b, {1, 0}) : b;
    ctx->SetOutput(0, xla::Dot(lhs, rhs));
  }
};

REGISTER_XLA_OP_COMPILER(MatMul, MatMulOp);

}  // namespace mola
}  // namespace oneflow
