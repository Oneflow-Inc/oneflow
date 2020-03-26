#include "oneflow/xrt/tvm/ops/tvm_op_kernel.h"
#include <tvm/relay/attrs/nn.h>

namespace oneflow {
namespace xrt {
namespace of_tvm {

class BatchNormOp final : public TVMOpKernel {
 public:
  void Compile(TVMOpContext* ctx) override {
    tvm::Array<tvm::relay::Expr> inputs;
    inputs.push_back(ctx->GetExpr4InputName("in"));
    inputs.push_back(ctx->GetExpr4InputName("gamma"));
    inputs.push_back(ctx->GetExpr4InputName("beta"));
    inputs.push_back(ctx->GetExpr4InputName("moving_mean"));
    inputs.push_back(ctx->GetExpr4InputName("moving_variance"));

    auto bn_attrs = tvm::make_node<tvm::relay::BatchNormAttrs>();
    {
      bn_attrs->axis = ctx->GetAttr<int32_t>("axis");
      bn_attrs->epsilon = ctx->GetAttr<float>("epsilon");
      bn_attrs->center = ctx->GetAttr<bool>("center");
      bn_attrs->scale = ctx->GetAttr<bool>("scale");
    }

    auto bn_op = tvm::relay::Op::Get("nn.batch_norm");
    //TODO(niuchong): handle multi-output for batch_norm when inference
    auto bn = tvm::relay::CallNode::make(bn_op, inputs, tvm::Attrs(bn_attrs), {});

    // There is no TOpPattern attr registered for nn.batch_norm, which leads to the attr missing
    // error when we call relay.build().
    // But nn.batch_norm always get unpacked by SimplifyInference Pass in tvm,
    // and SimplifyInference takes effect only when we solely need the 1st output of bn.
    // Thus, we should return the 1st output of nn.batch_norm instead of itself here.
    auto n = tvm::make_node<tvm::relay::TupleGetItemNode>();
    n->tuple = std::move(bn);
    n->index = 0;
    ctx->SetExpr4OutputName("out", std::move(tvm::relay::TupleGetItem(n)));
  }
};

REGISTER_TVM_OP_KERNEL(Normalization, BatchNormOp).Finalize();

}
}
}
