#include "oneflow/xrt/tvm/ops/tvm_op_kernel.h"
#include <tvm/relay/attrs/transform.h>

namespace oneflow {
namespace xrt {
namespace of_tvm {

class TransposeOp final : public TVMOpKernel {
 public:
  void Compile(TVMOpContext* ctx) override {
    tvm::Array<tvm::relay::Expr> node_inputs;
    node_inputs.push_back(ctx->GetExpr4InputName("in"));

    const Shape& in_shape = ctx->GetShape4InputName("in");
    std::vector<int32_t> axes = ctx->GetAttr<std::vector<int32_t>>("perm");
    CHECK_EQ(axes.size(), in_shape.NumAxes());

    tvm::Array<tvm::Integer> tvm_axes;
    for (int32_t dim : axes) { tvm_axes.push_back(dim); }
    auto transpose_attrs = tvm::make_node<tvm::relay::TransposeAttrs>();
    transpose_attrs->axes = tvm_axes;

    auto op = tvm::relay::Op::Get("transpose");
    auto expr = tvm::relay::CallNode::make(op, node_inputs, tvm::Attrs(transpose_attrs), {});
    ctx->SetExpr4OutputName("out", std::move(expr));
  }
};

REGISTER_TVM_OP_KERNEL(Transpose, TransposeOp).Finalize();

}
}
}
