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

class FftR2C : public OpExprGradFunction<FftR2CCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);
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
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    in_grads->resize(1);
    if (!ctx->onesided) {
      std::cout << "=========== [FftR2C Op Backward] !ctx->onesided ===========" << std::endl;
      // different from torch -- we set `forward` is true
      auto complex_grad =
          JUST(functional::FftC2C(out_grads.at(0), NullOpt, ctx->dims, ctx->norm_str,
                                  /*forward*/ !(ctx->forward), /*is_grad_fn*/ true));
      in_grads->at(0) = JUST(functional::Real(complex_grad));
    } else {
      std::cout << "=========== [FftR2C Op Backward] ctx->onesided ===========" << std::endl;
      Shape input_shape(ctx->input_shape_vec);
      int64_t last_dim = ctx->dims.back();
      int64_t last_dim_size = input_shape.At(last_dim);
      int64_t zero_length = last_dim_size - out_grads.at(0)->dim(last_dim);
      if (zero_length > 0) {
        std::cout << "=========== [FftR2C Op Backward] ctx->onesided, zero_length > 0 ==========="
                  << std::endl;
        std::vector<int64_t> fft_dims = ctx->dims;
        std::vector<int64_t> fft_shapes(fft_dims.size(), 0);
        FOR_RANGE(size_t, i, 0, fft_dims.size()) { fft_shapes[i] = input_shape[fft_dims[i]]; }
        auto complex_full_grad =
            JUST(functional::FftC2C(out_grads.at(0), fft_shapes, ctx->dims, ctx->norm_str,
                                    /*forward*/ !(ctx->forward), /*is_grad_fn*/ true));
        in_grads->at(0) = JUST(functional::Real(complex_full_grad));
      } else {
        // do c2c and slice
        // const auto& in_grad_sizes = in_grads->at(0)->shape()->dim_vec();
        std::cout << "=========== [FftR2C Op Backward] ctx->onesided, zero_length <= 0 ==========="
                  << std::endl;
        auto complex_grad =
            JUST(functional::FftC2C(out_grads.at(0), NullOpt, ctx->dims, ctx->norm_str,
                                    /*forward*/ !(ctx->forward), /*is_grad_fn*/ true));
        std::vector<int64_t> slice_st(input_shape.size(), 0);
        std::vector<int64_t> slice_end(input_shape.begin(), input_shape.end());
        std::vector<int64_t> slice_step(input_shape.size(), 1);
        auto sliced_tensor =
            JUST(functional::Slice(complex_grad, slice_st, slice_end, slice_step, false));
        in_grads->at(0) = sliced_tensor;
      }
    }

    return Maybe<void>::Ok();
  }
};

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
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(FftC2CCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);

    ctx->requires_grad = inputs.at(0)->requires_grad();

    ctx->forward = JUST(attrs.GetAttr<bool>("forward"));
    ctx->dims = JUST(attrs.GetAttr<std::vector<int64_t>>("dims"));
    ctx->norm_str = JUST(attrs.GetAttr<std::string>("norm"));

    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const FftC2CCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    in_grads->resize(1);
    in_grads->at(0) = JUST(functional::FftC2C(out_grads.at(0), NullOpt, ctx->dims, ctx->norm_str,
                                              /*forward*/ !(ctx->forward), /*is_grad_fn*/ true));
    return Maybe<void>::Ok();
  }
};

struct FftC2RCaptureState : public AutoGradCaptureState {
  bool requires_grad;
  bool forward;
  std::vector<int64_t> dims;
  std::string norm_str;
  int64_t last_dim_size;
  DimVector input_shape_vec;
};

class FftC2R : public OpExprGradFunction<FftC2RCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(FftC2RCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    ctx->requires_grad = inputs.at(0)->requires_grad();
    ctx->forward = JUST(attrs.GetAttr<bool>("forward"));
    ctx->dims = JUST(attrs.GetAttr<std::vector<int64_t>>("dims"));
    ctx->norm_str = JUST(attrs.GetAttr<std::string>("norm"));
    ctx->last_dim_size = JUST(attrs.GetAttr<int64_t>("last_dim_size"));
    ctx->input_shape_vec = inputs.at(0)->shape()->dim_vec();

    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const FftC2RCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    in_grads->resize(1);
    auto complex_grad = JUST(functional::FftR2C(out_grads.at(0), NullOpt, ctx->dims, ctx->norm_str,
                                                /*onesided=*/true, ctx->forward));
    Shape input_shape(ctx->input_shape_vec);
    int64_t last_dim = ctx->dims.back();
    auto double_length = out_grads.at(0)->dim(last_dim) - complex_grad->dim(last_dim);
    auto in_grad = complex_grad;

    // mul by 2, and slice
    if (double_length > 0) {
      in_grad = JUST(functional::Narrow(complex_grad, last_dim, 1,
                                        double_length));  // will change shape of in_grad
      in_grad = JUST(functional::ScalarMul(in_grad, 2, /*inplace*/ true));
    }

    std::vector<int64_t> slice_st(input_shape.size(), 0);
    std::vector<int64_t> slice_end(input_shape.begin(), input_shape.end());
    std::vector<int64_t> slice_step(input_shape.size(), 1);
    auto sliced_tensor =
        JUST(functional::Slice(complex_grad, slice_st, slice_end, slice_step, false));

    in_grads->at(0) = sliced_tensor;
    return Maybe<void>::Ok();
  }

};

REGISTER_OP_EXPR_GRAD_FUNCTION("fft_r2c", FftR2C);
REGISTER_OP_EXPR_GRAD_FUNCTION("fft_c2c", FftC2C);
REGISTER_OP_EXPR_GRAD_FUNCTION("fft_c2r", FftC2R);

}  // namespace one

}  // namespace oneflow