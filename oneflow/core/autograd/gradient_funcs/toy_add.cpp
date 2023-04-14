#include "oneflow/core/framework/op_expr_grad_function.h"

namespace oneflow {
namespace one {

struct ToyAddCaptureState : public AutoGradCaptureState {
  std::vector<bool> requires_grad;
};

class ToyAdd : public OpExprGradFunction<ToyAddCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(ToyAddCaptureState* ctx, const TensorTuple& inputs, const TensorTuple& outputs,
                      const AttrMap& attrs) const override {
    ctx->requires_grad.resize(inputs.size());
    for (int i =0  ; i<inputs.size(); ++i) {
      ctx->requires_grad[i] = inputs.at(i)->requires_grad();
    }
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const ToyAddCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
    in_grads->resize(2);
    for(int i = 0;i< 2;++i){
        if(ctx->requires_grad.at(i)) {in_grads->at(i) = out_grads.at(0);}
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("toy_add", ToyAdd);

}  // namespace one
}  // namespace oneflow
