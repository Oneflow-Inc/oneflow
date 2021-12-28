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
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct AdaptivePoolCaptureState : public AutoGradCaptureState {
  bool requires_grad;
};

class AdaptivePoolNdGrad : public OpExprGradFunction<AdaptivePoolCaptureState> {
 public:
  AdaptivePoolNdGrad(const std::string& mode, const int& ndims) : mode_(mode), ndims_(ndims) {}
  Maybe<void> Capture(AdaptivePoolCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpBase* ctx) const override;
  Maybe<void> Apply(const AdaptivePoolCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  std::string mode_;
  int32_t ndims_;
};

Maybe<void> AdaptivePoolNdGrad::Capture(AdaptivePoolCaptureState* state, const TensorTuple& inputs,
                                        const TensorTuple& outputs, const OpBase* ctx) const {
  state->requires_grad = inputs.at(0)->requires_grad();
  if (!state->requires_grad) { return Maybe<void>::Ok(); }

  state->SaveTensorForBackward(inputs.at(0));
  return Maybe<void>::Ok();
}

Maybe<void> AdaptivePoolNdGrad::Apply(const AdaptivePoolCaptureState* state,
                                      const TensorTuple& out_grads, TensorTuple* in_grads) const {
  if (!state->requires_grad) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);

  const std::shared_ptr<oneflow::one::Tensor>& x = state->SavedTensors().at(0);
  in_grads->resize(1);
  in_grads->at(0) = JUST(functional::AdaptivePoolNdGrad(x, out_grads.at(0), mode_, ndims_));
  return Maybe<void>::Ok();
}

class AdaptiveAvgPool1dGrad final : public AdaptivePoolNdGrad {
 public:
  AdaptiveAvgPool1dGrad() : AdaptivePoolNdGrad("avg", 1) {}
};

class AdaptiveAvgPool2dGrad final : public AdaptivePoolNdGrad {
 public:
  AdaptiveAvgPool2dGrad() : AdaptivePoolNdGrad("avg", 2) {}
};

class AdaptiveAvgPool3dGrad final : public AdaptivePoolNdGrad {
 public:
  AdaptiveAvgPool3dGrad() : AdaptivePoolNdGrad("avg", 3) {}
};

REGISTER_OP_EXPR_GRAD_FUNCTION("adaptive_avg_pool1d", AdaptiveAvgPool1dGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("adaptive_avg_pool2d", AdaptiveAvgPool2dGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("adaptive_avg_pool3d", AdaptiveAvgPool3dGrad);

}  // namespace one
}  // namespace oneflow
