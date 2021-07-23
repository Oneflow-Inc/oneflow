#include "oneflow/xrt/tvm/ops/op_kernel.h"

namespace oneflow {
namespace xrt {
namespace of_tvm {

class ReluOp final : public TVMOpKernel {
 public:
  void Compile(TVMOpContext* ctx) override {
    LOG(WARNING) << ctx->DebugStr();
    tvm::Array<tvm::relay::Expr> node_inputs;
    node_inputs.push_back(ctx->GetExpr4InputName("in_0"));

    auto op = tvm::relay::Op::Get("nn.relu");
    auto expr = tvm::relay::Call(op, node_inputs, tvm::Attrs(), {});
    ctx->SetExpr4OutputName("out_0", std::move(expr));
  }
};

REGISTER_TVM_OP_KERNEL(Relu, ReluOp).Finalize();

}  // namespace of_tvm
}  // namespace xrt
}  // namespace oneflow
