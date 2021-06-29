#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct TriuInterpState : public OpExprInterpState {
  bool requires_grad; 
  int64_t diagonal; 
};

class Triu : public OpExprGradFunction<TriuInterpState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(TriuInterpState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const TriuInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> Triu::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> Triu::Capture(TriuInterpState* ctx, const TensorTuple& inputs,
                                  const TensorTuple& outputs, const AttrMap& attrs) const {
  ctx->requires_grad = inputs.at(0)->requires_grad();
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->diagonal = JUST(composed_attrs.GetAttr<int64_t>("diagonal"));
  return Maybe<void>::Ok();
}

Maybe<void> Triu::Apply(const TriuInterpState* ctx, const TensorTuple& out_grads,
                                TensorTuple* in_grads) const {
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);
  in_grads->resize(1);
  if (ctx->requires_grad) {
    in_grads->at(0) = JUST(functional::Triu(out_grads.at(0), ctx->diagonal));
  }
return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("triu", Triu);

}  // namespace one
}  // namespace oneflow