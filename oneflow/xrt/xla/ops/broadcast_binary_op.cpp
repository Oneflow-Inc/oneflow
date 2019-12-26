#include "oneflow/core/common/shape_view.h"
#include "oneflow/xrt/xla/ops/op_context.h"
#include "oneflow/xrt/xla/ops/op_kernel.h"

#include "oneflow/xrt/xla/ops/binary_op.h"
#include "oneflow/xrt/xla/xla_helpers.h"

namespace oneflow {
namespace xrt {
namespace mola {

template<typename BinaryOp>
class BcastBinaryOp : public XlaOpKernel {
 public:
  void Compile(XlaOpContext *ctx) override {
    Shape shape_a = ctx->InputShape("a");
    Shape shape_b = ctx->InputShape("b");

    int axes = std::max(shape_a.NumAxes(), shape_b.NumAxes());
    shape_a = CreateLeftExtendedShape(ShapeView(shape_a), axes);
    shape_b = CreateLeftExtendedShape(ShapeView(shape_b), axes);

    xla::XlaOp a = Reshape(ctx->Input("a"), shape_a);
    xla::XlaOp b = Reshape(ctx->Input("b"), shape_b);
    ctx->SetOutput("out", BinaryOp()(a, b));
  }
};

REGISTER_XLA_OP_KERNEL(BcastAdd, BcastBinaryOp<op::Add>).Finalize();
REGISTER_XLA_OP_KERNEL(BcastMul, BcastBinaryOp<op::Mul>).Finalize();
REGISTER_XLA_OP_KERNEL(BcastDiv, BcastBinaryOp<op::Div>).Finalize();

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow
