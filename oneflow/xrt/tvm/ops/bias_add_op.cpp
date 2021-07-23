#include "oneflow/xrt/tvm/ops/op_kernel.h"
#include <tvm/relay/attrs/nn.h>

namespace oneflow {
namespace xrt {
namespace of_tvm {

class BiasAddOp final : public TVMOpKernel {
 public:
  void Compile(TVMOpContext* ctx) override {
    LOG(WARNING) << ctx->DebugStr();
    tvm::Array<tvm::relay::Expr> node_inputs;
    node_inputs.push_back(ctx->GetExpr4InputName("a_0"));
    node_inputs.push_back(ctx->GetExpr4InputName("b_0"));

    auto bias_add_attrs = tvm::runtime::make_object<tvm::relay::BiasAddAttrs>();
    bias_add_attrs->axis = ctx->Attr<int32_t>("axis");

    auto op = tvm::relay::Op::Get("nn.bias_add");
    auto expr = tvm::relay::Call(op, node_inputs, tvm::Attrs(bias_add_attrs), {});
    ctx->SetExpr4OutputName("out_0", std::move(expr));
  }
};

REGISTER_TVM_OP_KERNEL(BiasAdd, BiasAddOp).Finalize();

}  // namespace of_tvm
}  // namespace xrt
}  // namespace oneflow
