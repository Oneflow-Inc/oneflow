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

enum class AmpIdentityType {
  kWhite = 0,
  kBlack,
};

struct AmpIdentityCaptureState : public AutoGradCaptureState {};

template<AmpIdentityType type>
class AmpIdentityGrad : public OpExprGradFunction<AmpIdentityCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(AmpIdentityCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const AmpIdentityCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->resize(1);
    if (type == AmpIdentityType::kWhite) {
      (*in_grads)[0] = JUST(functional::AmpWhiteIdentity(out_grads[0]));
    } else if (type == AmpIdentityType::kBlack) {
      (*in_grads)[0] = JUST(functional::AmpBlackIdentity(out_grads[0]));
    } else {
      (*in_grads)[0] = out_grads[0];
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("amp_white_identity", AmpIdentityGrad<AmpIdentityType::kWhite>);
REGISTER_OP_EXPR_GRAD_FUNCTION("amp_black_identity", AmpIdentityGrad<AmpIdentityType::kBlack>);

}  // namespace one
}  // namespace oneflow
