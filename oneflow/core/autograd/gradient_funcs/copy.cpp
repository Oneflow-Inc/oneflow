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
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct CopyCaptureState : public AutoGradCaptureState {
  std::string device_type;
  int64_t device_id;
};

class Copy : public OpExprGradFunction<CopyCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(CopyCaptureState* ctx, const TensorTuple& inputs, const TensorTuple& outputs,
                      const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);  // NOLINT(maybe-need-error-msg)
    if (inputs[0]->is_global()) {
      ctx->device_type = JUST(inputs[0]->parallel_desc())->device_tag();
      ctx->device_id = 0;  // global tensor only has one local device
    } else {
      ctx->device_type = JUST(inputs[0]->device())->type();
      ctx->device_id = JUST(inputs[0]->device())->device_id();
    }
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const CopyCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->resize(1);
    (*in_grads)[0] = JUST(
        functional::Copy(out_grads[0], ctx->device_type, ctx->device_id, /*pin_memory=*/false));
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("copy", Copy);

}  // namespace one
}  // namespace oneflow
