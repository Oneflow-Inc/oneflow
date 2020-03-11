#include "oneflow/xrt/tvm/ops/tvm_op_kernel.h"

namespace oneflow {
namespace xrt {
namespace of_tvm {

class ReluOp final : public TVMOpKernel {
 public:
  void Compile(TVMOpContext* ctx) override {
    auto op = tvm::relay::Op::Get("nn.relu");
    auto expr = tvm::relay::CallNode::make(op, ctx->node_inputs(), tvm::Attrs(), {});
    ctx->set_op_expr(expr);
  }
};

REGISTER_TVM_OP_KERNEL(Relu, ReluOp).EnableTrainPhase().Finalize();

}
}
}
