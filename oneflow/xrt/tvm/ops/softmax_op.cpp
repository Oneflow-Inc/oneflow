#include "oneflow/xrt/tvm/ops/op_kernel.h"
#include <tvm/relay/attrs/nn.h>

namespace oneflow {
namespace xrt {
namespace of_tvm {

class SoftmaxOp final : public TVMOpKernel {
 public:
  void Compile(TVMOpContext* ctx) override {
    tvm::Array<tvm::relay::Expr> node_inputs;
    node_inputs.push_back(ctx->GetExpr4InputName("in_0"));

    auto softmax_attrs = tvm::runtime::make_object<tvm::relay::SoftmaxAttrs>();
    softmax_attrs->axis = ctx->Attr<int32_t>("axis");

    auto op = tvm::relay::Op::Get("nn.softmax");
    auto expr = tvm::relay::Call(op, node_inputs, tvm::Attrs(softmax_attrs), {});
    ctx->SetExpr4OutputName("out_0", std::move(expr));
  }
};

REGISTER_TVM_OP_KERNEL(Softmax, SoftmaxOp).Finalize();

}  // namespace of_tvm
}  // namespace xrt
}  // namespace oneflow
