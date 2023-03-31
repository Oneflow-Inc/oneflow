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
#include "oneflow/core/common/optional.h"
#include "oneflow/core/framework/attr_map.h"
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
  DimVector input_shape_vec;
  std::string norm_str;
};

#if 1
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
        ctx->dims = JUST(attrs.GetAttr<std::vector<int64_t>>("dims"));
        ctx->norm_str = JUST(attrs.GetAttr<std::string>("norm"));
        ctx->input_shape_vec = inputs.at(0)->shape()->dim_vec();

        return Maybe<void>::Ok();
    }

    Maybe<void> Apply(const FftR2CCaptureState* ctx, const TensorTuple& out_grads,
                        TensorTuple* in_grads) const override {
        std::cout << "=========== [FftR2C Op Backward] ===========" << std::endl;
        CHECK_EQ_OR_RETURN(out_grads.size(), 1);
        in_grads->resize(1);
        if (!ctx->onesided){
            auto complex_grad = JUST(functional::FftC2C(out_grads.at(0), NullOpt, ctx->dims, ctx->norm_str, /*forward*/ false, /*is_grad_fn*/ true));
            in_grads->at(0) = JUST(functional::Real(complex_grad));
        }
        else{
          // CHECK_OR_THROW(false) << "UNIMPLEMENTED";
          Shape input_shape(ctx->input_shape_vec);
          int64_t last_dim = ctx->dims.back();
          int64_t last_dim_size = input_shape.At(last_dim);
          int64_t zero_length = last_dim_size - out_grads.at(0)->dim(last_dim);
          if (zero_length > 0){
            std::vector<int64_t> fft_dims {last_dim};
            std::vector<int64_t> fft_shapes {last_dim_size};
            auto complex_full_grad = JUST(functional::FftC2C(out_grads.at(0), fft_shapes, fft_dims, ctx->norm_str, /*forward*/ false, /*is_grad_fn*/ true));
            in_grads->at(0) = JUST(functional::Real(complex_full_grad));
          }
          else{
            // do c2c and slice
            // const auto& in_grad_sizes = in_grads->at(0)->shape()->dim_vec();
            auto complex_grad = JUST(functional::FftC2C(in_grads->at(0), NullOpt, ctx->dims, ctx->norm_str, /*forward*/ false, /*is_grad_fn*/ true));
            std::vector<int64_t> slice_st(input_shape.begin(), input_shape.end());
            std::vector<int64_t> slice_end(input_shape.begin(), input_shape.end());
            std::vector<int64_t> slice_step(input_shape.size(), 1);
            auto sliced_tensor = JUST(functional::Slice(complex_grad, slice_st, slice_end, slice_step, false));
            in_grads->at(0) = sliced_tensor;
          }
          // if (zero_length > 0){
          //   // do pad and c2c
          //   std::vector<int64_t> pad_amount(in_grad_sizes.size() * 2, 0);
          //   auto pad_idx = pad_amount.size() - 2 * last_dim - 1;
          //   pad_amount[pad_idx] = zero_length;
          //   auto complex_full_grad = JUST(functional::ConstantPad(out_grads.at(0), pad_amount, 0));
          //   in_grads->at(0) = functioanl::FftC2C(complex_full_grad, )
          // }
          // else{
          //   // do c2c and slice
          //   auto complex_grad = JUST(functional::FftC2C(in_grads->at(0), NullOpt, ctx->dims, ctx->norm_str, /*forward*/ false, /*is_grad_fn*/ true));
          //   std::vector<int64_t> slice_st(in_grad_sizes.begin(), in_grad_sizes.end());
          //   std::vector<int64_t> slice_end(in_grad_sizes.begin(), in_grad_sizes.end());
          //   std::vector<int64_t> slice_step(in_grad_sizes.size(), 1);
          //   auto sliced_tensor = JUST(functional::Slice(complex_grad, slice_st, slice_end, slice_step, false));
          //   in_grads->at(0) = sliced_tensor;
          // }
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
    ComposedAttrMap composed_attrs(attrs, base_attrs_);

    ctx->requires_grad = inputs.at(0)->requires_grad();

    ctx->forward = JUST(composed_attrs.GetAttr<bool>("forward"));
    ctx->dims = JUST(attrs.GetAttr<std::vector<int64_t>>("dims"));
    ctx->norm_str = JUST(attrs.GetAttr<std::string>("norm"));

    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const FftC2CCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    // TO-DO add gradient logic
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    // std::vector<int64_t> n (out_grads.at(0)->ndim());
    // for (int i = 0; i < ctx->dims.size(); i++){
    //   n[i] = out_grads.at(0)->dim(ctx->dims[i]);
    // }
    in_grads->resize(1);
    in_grads->at(0) = JUST(functional::FftC2C(out_grads.at(0), NullOpt, ctx->dims, ctx->norm_str,
                                              /*forward*/ !(ctx->forward), /*is_grad_fn*/ true));
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("fft_r2c", FftR2C);
REGISTER_OP_EXPR_GRAD_FUNCTION("fft_c2c", FftC2C);

}  // namespace one

}  // namespace oneflow