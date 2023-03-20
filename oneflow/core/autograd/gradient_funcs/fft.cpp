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
#include <string>
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/functional_api.yaml.h"

namespace oneflow {
namespace one {

struct FftR2CCaptureState : public AutoGradCaptureState {
  bool requires_grad;
  bool onesided;
  bool forward;
  std::vector<int64_t> dims;
  std::string norm_str;
};

#if 0
class FftR2C : public OpExprGradFunction<FftR2CCaptureState> {
public:
    Maybe<void> Init(const OpExpr& op) override {
        const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
        CHECK_NOTNULL_OR_RETURN(fw_op_expr);
        base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
        return Maybe<void>::Ok();
    }

    Maybe<void> Capture(FftR2CCaptureState* ctx, const TensorTuple& inputs,
                        const TensorTuple& outputs, const AttrMap& attrs) const override {
        
        
        CHECK_EQ_OR_RETURN(inputs.size(), 1);
        ctx->requires_grad = inputs.at(0)->requires_grad();
        ctx->onesided = JUST(attrs.GetAttr<bool>("onesided"));
        ctx->forward = JUST(attrs.GetAttr<bool>("forward"));
        ctx->dims = JUST(attrs.GetAttr<std::vector<int64_t>>("forward"));
        ctx->norm_str = JUST(attrs.GetAttr<std::string>("norm"));

        return Maybe<void>::Ok();
    }

    Maybe<void> Apply(const FftR2CCaptureState* ctx, const TensorTuple& out_grads,
                        TensorTuple* in_grads) const override {
        // CHECK_EQ_OR_RETURN(out_grads.size(), 1);
        // in_grads->resize(ctx->requires_grad.size());
        // for (int i = 0; i < ctx->requires_grad.size(); ++i){
        //     if (ctx->requires_grad.at(i)){
        //         in_grads->at(i) = JUST(functional::Fft(out_grads.at(0), ctx->SavedTensors().at(ctx->indices[i])));
        //     }
        // }
        // TO-DO add gradient logic
        CHECK_EQ_OR_RETURN(out_grads.size(), 1);
        in_grads->resize(1);
        in_grads->at(0) = functional::FftR2CGrad(out_grads.at(0), ctx->dims, ctx->norm_str, !(ctx->forward));
        return Maybe<void>::Ok();

        if (!ctx->onesided){

        }
        
        return Maybe<void>::Ok();
    }

private:
    AttrMap base_attrs_;

};
#endif

struct FftC2CCaptureState : public AutoGradCaptureState {
  bool requires_grad;
  bool forward;
  std::vector<int64_t> dims;
  std::string norm_str;
};

class FftC2C : public OpExprGradFunction<FftC2CCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(FftC2CCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    ctx->requires_grad = inputs.at(0)->requires_grad();
    ctx->forward = JUST(attrs.GetAttr<bool>("forward"));
    ctx->dims = JUST(attrs.GetAttr<std::vector<int64_t>>("forward"));
    ctx->norm_str = JUST(attrs.GetAttr<std::string>("norm"));

    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const FftC2CCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    // TO-DO add gradient logic
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    in_grads->resize(1);
    in_grads->at(0) =
        JUST(functional::FftC2CGrad(out_grads.at(0), ctx->dims, ctx->norm_str, !(ctx->forward)));
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

// REGISTER_OP_EXPR_GRAD_FUNCTION("fft_r2c", FftR2C);   TO-DO
REGISTER_OP_EXPR_GRAD_FUNCTION("fft_c2c", FftC2C);

}  // namespace one

}  // namespace oneflow