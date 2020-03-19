#include "oneflow/xrt/tvm/ops/tvm_op_kernel.h"
#include <tvm/relay/attrs/nn.h>

namespace oneflow {
namespace xrt {
namespace of_tvm {

class BiasAddOp final : public TVMOpKernel {
 public:
  void Compile(TVMOpContext* ctx) override {
    tvm::Array<tvm::relay::Expr> node_inputs;
    node_inputs.push_back(ctx->GetExpr4InputName("a"));
    node_inputs.push_back(ctx->GetExpr4InputName("b"));

    auto bias_add_attrs = tvm::make_node<tvm::relay::BiasAddAttrs>();
    bias_add_attrs->axis = ctx->GetAttr<int32_t>("axis");

    auto op = tvm::relay::Op::Get("nn.bias_add");
    auto expr = tvm::relay::CallNode::make(op, node_inputs, tvm::Attrs(bias_add_attrs), {});
    ctx->set_op_expr(expr);
  }
};

REGISTER_TVM_OP_KERNEL(BiasAdd, BiasAddOp).EnableTrainPhase().Finalize();

}
}
}
