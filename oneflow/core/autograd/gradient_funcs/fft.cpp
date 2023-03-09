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
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow{
namespace one {

struct FftCaptureState : public AutoGradCaptureState {
    std::vector<size_t> indices;
    std::vector<bool> requires_grad;
};

class Fft : public OpExprGradFunction<FftCaptureState> {
public:
    Maybe<void> Init(const OpExpr& op) override {
        const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
        CHECK_NOTNULL_OR_RETURN(fw_op_expr);
        base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
        return Maybe<void>::Ok();
    }

    Maybe<void> Capture(FftCaptureState* ctx, const TensorTuple& inputs,
                        const TensorTuple& outputs, const AttrMap& attrs) const override {
        CHECK_EQ_OR_RETURN(inputs.size(), 2);
        ctx->indices.resize(inputs.size());
        ctx->requires_grad.resize(inputs.size());
        for (int i = 0; i < inputs.size(); ++i){
            ctx->requires_grad[i] = inputs.at(i)->requires_grad();
            if (ctx->requires_grad[i]){
                ctx->indices[i] = ctx->SaveTensorForBackward(inputs.at(inputs.size() - 1 - i));
            }
        }
        return Maybe<void>::Ok();
    }

    Maybe<void> Apply(const FftCaptureState* ctx, const TensorTuple& out_grads,
                        TensorTuple* in_grads) const override {
        CHECK_EQ_OR_RETURN(out_grads.size(), 1);
        in_grads->resize(ctx->requires_grad.size());
        for (int i = 0; i < ctx->requires_grad.size(); ++i){
            if (ctx->requires_grad.at(i)){
                in_grads->at(i) = JUST(functional::Fft(out_grads.at(0), ctx->SavedTensors().at(ctx->indices[i])));
            }
        }
        return Maybe<void>::Ok();
    }

private:
    AttrMap base_attrs_;

};

REGISTER_OP_EXPR_GRAD_FUNCTION("fft", Fft);

}   // namespace oneflow

}   // namespace oneflow