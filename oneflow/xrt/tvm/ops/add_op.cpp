#include "oneflow/xrt/tvm/ops/tvm_op_kernel.h"

namespace oneflow {
namespace xrt {
namespace of_tvm {

class AddOp final : public TVMOpKernel {
 public:
  void Compile(TVMOpContext* ctx) override {
    int32_t num_inputs = ctx->num_inputs();
    CHECK_EQ(2, num_inputs) << "TVM only support Add operator with 2 inputs";

    tvm::Array<tvm::relay::Expr> node_inputs;
    node_inputs.push_back(ctx->GetExpr4InputName("in_0"));
    node_inputs.push_back(ctx->GetExpr4InputName("in_1"));

    auto op = tvm::relay::Op::Get("add");
    auto expr = tvm::relay::CallNode::make(op, node_inputs, tvm::Attrs(), {});
    ctx->SetExpr4OutputName("out", std::move(expr));
  }
};

REGISTER_TVM_OP_KERNEL(Add, AddOp).Finalize();

}
}
}
