#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "oneflow/engine/xla/of2xla/xla_op_compiler_registry.h"
#include "oneflow/engine/xla/of2xla/xla_op_compiler.h"
#include "oneflow/engine/xla/of2xla/xla_op_context.h"

#include "oneflow/engine/xla/of2xla/xla_helpers.h"
#include "oneflow/engine/xla/of2xla/ops/binary_op.h"

namespace oneflow {
namespace mla {

template <typename BinaryOp>
class BcastBinaryOp : public XlaOpCompiler {
 public:
  void Compile(XlaOpContext *ctx) override {
    Shape shape_a = ctx->InputShape("a");
    Shape shape_b = ctx->InputShape("b");

    int axes = std::max(shape_a.NumAxes(), shape_b.NumAxes());
    shape_a = shape_a.CreateLeftExtendedShape(axes);
    shape_b = shape_b.CreateLeftExtendedShape(axes);

    xla::XlaOp a = Reshape(ctx->Input("a"), shape_a);
    xla::XlaOp b = Reshape(ctx->Input("b"), shape_b);

//    std::vector<long long> bcast_dimensions_a(axes);
//    std::vector<long long> bcast_dimensions_b(axes);
//    for (int i = 0; i < axes; ++i) {
//      int64_t dim = std::max(shape_a.At(i), shape_b.At(i));
//      bcast_dimensions_a[i] = dim / shape_a.At(i);
//      bcast_dimensions_b[i] = dim / shape_b.At(i);
//    }
//
//    if (NeedBroadcast(bcast_dimensions_a)) {
//      a = xla::Broadcast(a, bcast_dimensions_a);
//    }
//    if (NeedBroadcast(bcast_dimensions_b)) {
//      b = xla::Broadcast(b, bcast_dimensions_b);
//    }
    ctx->SetOutput("out", BinaryOp()(a, b));
  }

  bool NeedBroadcast(const std::vector<long long> &bcast_dimensions) const {
    bool need_broadcast = false;
    for (auto dim : bcast_dimensions) {
      need_broadcast = need_broadcast || (dim > 1);
    }
    return need_broadcast;
  }
};

REGISTER_XLA_OP_COMPILER(BcastAdd, BcastBinaryOp<op::Add>);
REGISTER_XLA_OP_COMPILER(BcastMul, BcastBinaryOp<op::Mul>);
REGISTER_XLA_OP_COMPILER(BcastDiv, BcastBinaryOp<op::Div>);

}  // namespace mla
}  // namespace oneflow
