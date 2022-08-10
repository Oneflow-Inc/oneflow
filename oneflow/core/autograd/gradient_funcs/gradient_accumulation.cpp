/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct GradAccRepeatCaptureState : public AutoGradCaptureState {
  int32_t repeat_num = 1;
};

class GradAccRepeat : public OpExprGradFunction<GradAccRepeatCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(GradAccRepeatCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const GradAccRepeatCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> GradAccRepeat::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> GradAccRepeat::Capture(GradAccRepeatCaptureState* ctx, const TensorTuple& inputs,
                                   const TensorTuple& outputs, const AttrMap& attrs) const {
  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->repeat_num = JUST(composed_attrs.GetAttr<int32_t>("repeat_num"));
  return Maybe<void>::Ok();
}

Maybe<void> GradAccRepeat::Apply(const GradAccRepeatCaptureState* ctx, const TensorTuple& out_grads,
                                 TensorTuple* in_grads) const {
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
  in_grads->resize(1);
  (*in_grads)[0] = JUST(functional::GradAccCollect(out_grads[0], ctx->repeat_num));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("repeat", GradAccRepeat);

struct GradAccCollectCaptureState : public AutoGradCaptureState {
  int32_t max_acc_num = 1;
};

class GradAccCollect : public OpExprGradFunction<GradAccCollectCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(GradAccCollectCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const GradAccCollectCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> GradAccCollect::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> GradAccCollect::Capture(GradAccCollectCaptureState* ctx, const TensorTuple& inputs,
                                    const TensorTuple& outputs, const AttrMap& attrs) const {
  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->max_acc_num = JUST(composed_attrs.GetAttr<int32_t>("max_acc_num"));
  return Maybe<void>::Ok();
}

Maybe<void> GradAccCollect::Apply(const GradAccCollectCaptureState* ctx,
                                  const TensorTuple& out_grads, TensorTuple* in_grads) const {
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
  in_grads->resize(1);
  (*in_grads)[0] = JUST(functional::GradAccRepeat(out_grads[0], ctx->max_acc_num));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("acc", GradAccCollect);

struct GradAccPackCaptureState : public AutoGradCaptureState {
  int32_t pack_num = 1;
};

class GradAccPack : public OpExprGradFunction<GradAccPackCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(GradAccPackCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const GradAccPackCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> GradAccPack::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> GradAccPack::Capture(GradAccPackCaptureState* ctx, const TensorTuple& inputs,
                                 const TensorTuple& outputs, const AttrMap& attrs) const {
  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->pack_num = JUST(composed_attrs.GetAttr<int32_t>("pack_num"));
  return Maybe<void>::Ok();
}

Maybe<void> GradAccPack::Apply(const GradAccPackCaptureState* ctx, const TensorTuple& out_grads,
                               TensorTuple* in_grads) const {
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
  in_grads->resize(1);
  (*in_grads)[0] = JUST(functional::GradAccUnpack(out_grads[0], ctx->pack_num));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("pack", GradAccPack);

struct GradAccUnpackCaptureState : public AutoGradCaptureState {
  int32_t unpack_num = 1;
};

class GradAccUnpack : public OpExprGradFunction<GradAccUnpackCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(GradAccUnpackCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const GradAccUnpackCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> GradAccUnpack::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> GradAccUnpack::Capture(GradAccUnpackCaptureState* ctx, const TensorTuple& inputs,
                                   const TensorTuple& outputs, const AttrMap& attrs) const {
  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->unpack_num = JUST(composed_attrs.GetAttr<int32_t>("unpack_num"));
  return Maybe<void>::Ok();
}

Maybe<void> GradAccUnpack::Apply(const GradAccUnpackCaptureState* ctx, const TensorTuple& out_grads,
                                 TensorTuple* in_grads) const {
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
  in_grads->resize(1);
  (*in_grads)[0] = JUST(functional::GradAccPack(out_grads[0], ctx->unpack_num));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("unpack", GradAccUnpack);

}  // namespace one
}  // namespace oneflow
