#include "oneflow/xrt/tvm/ops/op_kernel.h"
#include <tvm/relay/attrs/nn.h>

namespace oneflow {
namespace xrt {
namespace of_tvm {

class BatchNormOp final : public TVMOpKernel {
 public:
  void Compile(TVMOpContext* ctx) override {
    LOG(WARNING) << ctx->DebugStr();

    tvm::Array<tvm::relay::Expr> inputs;
    inputs.push_back(ctx->GetExpr4InputName("x_0"));
    inputs.push_back(ctx->GetExpr4InputName("gamma_0"));
    inputs.push_back(ctx->GetExpr4InputName("beta_0"));
    inputs.push_back(ctx->GetExpr4InputName("moving_mean_0"));
    inputs.push_back(ctx->GetExpr4InputName("moving_variance_0"));
    // TODO: handle training
    auto attrs = tvm::runtime::make_object<tvm::relay::BatchNormAttrs>();
    attrs->axis = ctx->Attr<int32_t>("axis");
    attrs->epsilon = ctx->Attr<float>("epsilon");
    attrs->center = true;
    attrs->scale = true;

    const auto& op = tvm::relay::Op::Get("nn.batch_norm");
    auto bn_op = tvm::relay::Call(op, inputs, tvm::Attrs(attrs), {});

    // auto bn = tvm::relay::Call(bn_op, inputs, tvm::Attrs(bn_attrs), {});

    // // There is no TOpPattern attr registered for nn.batch_norm, which leads to the attr missing
    // // error when we call relay.build().
    // // But nn.batch_norm always get unpacked by SimplifyInference Pass in tvm,
    // // and SimplifyInference takes effect only when we solely need the 1st output of bn.
    // // Thus, we should return the 1st output of nn.batch_norm instead of itself here.
    // auto n = tvm::relay::TupleGetItem(bn, 0);
    ctx->SetExpr4OutputName("y_0", std::move(bn_op));
  }
};

REGISTER_TVM_OP_KERNEL(Normalization, BatchNormOp).Finalize();

}  // namespace of_tvm
}  // namespace xrt
}  // namespace oneflow

