#include "oneflow/xrt/xla/ops/op_context.h"
#include "oneflow/xrt/xla/ops/op_kernel.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace oneflow {
namespace xrt {
namespace mola {

class BatchMatMulOp : public XlaOpKernel {
 public:
  void Compile(XlaOpContext *ctx) override {
    Shape shape_a = ctx->InputShape("a");
    Shape shape_b = ctx->InputShape("b");
    CHECK_EQ(shape_a.NumAxes(), shape_b.NumAxes());
    CHECK_GT(shape_a.NumAxes(), 2);

    bool transpose_a = ctx->Attr<bool>("transpose_a");
    bool transpose_b = ctx->Attr<bool>("transpose_b");

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

REGISTER_XLA_OP_KERNEL(BatchMatMul, BatchMatMulOp).Finalize();

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow
