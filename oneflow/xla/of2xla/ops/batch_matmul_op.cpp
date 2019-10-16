#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "oneflow/xla/of2xla/xla_op_compiler_registry.h"
#include "oneflow/xla/of2xla/xla_op_compiler.h"
#include "oneflow/xla/of2xla/xla_op_context.h"

namespace oneflow {
namespace mola {

class BatchMatMulOp : public XlaOpCompiler {
 public:
  void Compile(XlaOpContext *ctx) override {
    Shape shape_a = ctx->InputShape("a");
    Shape shape_b = ctx->InputShape("b");
    CHECK_EQ(shape_a.NumAxes(), shape_b.NumAxes());
    CHECK_GT(shape_a.NumAxes(), 2);

    bool transpose_a = ctx->GetAttr<bool>("transpose_a");
    bool transpose_b = ctx->GetAttr<bool>("transpose_b");

    xla::XlaOp a = ctx->Input("a");
    xla::XlaOp b = ctx->Input("b");

    // int axis = shape_a.NumAxes();
    // std::vector<long long int> permute_a(axis), permute_b(axis);
    // std::iota(permute_a.begin(), permute_a.end(), 0);
    // std::iota(permute_b.begin(), permute_b.end(), 0);
    // if (transpose_a) {
    //   permute_a[axis - 1] = axis - 2;
    //   permute_a[axis - 2] = axis - 1;
    // }
    // if (transpose_b) {
    //   permute_b[axis - 1] = axis - 2;
    //   permute_b[axis - 2] = axis - 1;
    // }

    // auto lhs = transpose_a ? xla::Transpose(a, permute_a) : a;
    // auto rhs = transpose_b ? xla::Transpose(b, permute_b) : b;
    
    // ctx->SetOutput("out", xla::BatchDot(lhs, rhs));
  
    ctx->SetOutput("out", xla::BatchDot(a, transpose_a, b, transpose_b));
  }
};

REGISTER_XLA_OP_COMPILER(BatchMatMul, BatchMatMulOp);

}  // namespace mola
}  // namespace oneflow
