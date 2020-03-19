#include "oneflow/xrt/tvm/ops/tvm_op_kernel.h"

namespace oneflow {
namespace xrt {
namespace of_tvm {

class ReshapeLikeOp final : public TVMOpKernel {
 public:
  void Compile(TVMOpContext* ctx) override {
    tvm::Array<tvm::relay::Expr> node_inputs;
    node_inputs.push_back(ctx->GetExpr4InputName("x"));
    node_inputs.push_back(ctx->GetExpr4InputName("like"));

    const Shape& x_shape = ctx->GetShape4InputName("x");
    const Shape& like_shape = ctx->GetShape4InputName("like");
    CHECK_EQ(x_shape.elem_cnt(), like_shape.elem_cnt());

    auto op = tvm::relay::Op::Get("reshape_like");
    auto expr = tvm::relay::CallNode::make(op, node_inputs, tvm::Attrs(), {});
    ctx->set_op_expr(expr);
  }
};

REGISTER_TVM_OP_KERNEL(ReshapeLike, ReshapeLikeOp).EnableTrainPhase().Finalize();

}
}
}
