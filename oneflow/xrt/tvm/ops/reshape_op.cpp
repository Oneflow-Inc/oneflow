#include "oneflow/xrt/tvm/ops/tvm_op_kernel.h"
#include <tvm/relay/attrs/transform.h>

namespace oneflow {
namespace xrt {
namespace of_tvm {

class ReshapeOp final : public TVMOpKernel {
 public:
  void Compile(TVMOpContext* ctx) override {
    tvm::Array<tvm::relay::Expr> node_inputs;
    node_inputs.push_back(ctx->GetExpr4InputName("in"));

    const Shape& in_shape = ctx->GetShape4InputName("in");
    const Shape& conf_shape = ctx->GetAttr<Shape>("shape");
    CHECK_EQ(in_shape.elem_cnt(), conf_shape.elem_cnt());

    tvm::Array<tvm::Integer> tvm_conf_shape;
    for (int64_t dim : conf_shape.dim_vec()) {
      tvm_conf_shape.push_back(static_cast<int32_t>(dim));
    }
    auto reshape_attrs = tvm::make_node<tvm::relay::ReshapeAttrs>();
    reshape_attrs->newshape = tvm_conf_shape;

    auto op = tvm::relay::Op::Get("reshape");
    auto expr = tvm::relay::CallNode::make(op, node_inputs, tvm::Attrs(reshape_attrs), {});
    ctx->SetExpr4OutputName("out", std::move(expr));
  }
};

REGISTER_TVM_OP_KERNEL(Reshape, ReshapeOp).Finalize();

}
}
}
