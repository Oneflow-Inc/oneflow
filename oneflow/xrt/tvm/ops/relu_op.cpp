#include "oneflow/xrt/tvm/ops/tvm_op_kernel.h"

namespace oneflow {
namespace xrt {
namespace of_tvm {

class ReluOp final : public TVMOpKernel {
 public:
  void Compile(TVMOpContext* ctx) override {
    tvm::Array<tvm::relay::Expr> node_inputs;
    node_inputs.push_back(ctx->GetExpr4InputName("in"));

    auto op = tvm::relay::Op::Get("nn.relu");
    auto expr = tvm::relay::CallNode::make(op, node_inputs, tvm::Attrs(), {});
    ctx->SetExpr4OutputName("out", std::move(expr));
  }
};

REGISTER_TVM_OP_KERNEL(Relu, ReluOp).Finalize();

}
}
}
