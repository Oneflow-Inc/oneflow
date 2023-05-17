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
  bool requires_grad = false;
  bool onesided = false;
  std::vector<int64_t> dims;
  DimVector input_shape_vec;
  int32_t norm_mode = 0;
};

class FftR2C : public OpExprGradFunction<FftR2CCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(FftR2CCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1) << "RuntimeError: assert `inputs.size() == 1`";
    ctx->requires_grad = JUST(oneflow::VectorAt(inputs, 0))->requires_grad();
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

    ctx->onesided = JUST(attrs.GetAttr<bool>("onesided"));
    ctx->dims = JUST(attrs.GetAttr<std::vector<int64_t>>("dims"));
    ctx->norm_mode = JUST(attrs.GetAttr<int32_t>("norm_mode"));
    ctx->input_shape_vec = JUST(oneflow::VectorAt(inputs, 0))->shape()->dim_vec();

    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const FftR2CCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1) << "RuntimeError: assert `out_grads.size() == 1`";
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

    in_grads->resize(1);
    if (!ctx->onesided) {
      auto complex_grad = JUST(functional::FftC2C(JUST(oneflow::VectorAt(out_grads, 0)), NullOpt,
                                                  ctx->dims, ctx->norm_mode,
                                                  /*forward=*/false, /*normalized=*/false));
      JUST(oneflow::VectorAt(*in_grads, 0)) = JUST(functional::Real(complex_grad));
    } else {
      std::vector<int64_t> fft_dims = ctx->dims;
      std::vector<int64_t> fft_shapes(fft_dims.size(), 0);
      FOR_RANGE(size_t, i, 0, fft_dims.size()) {
        fft_shapes[i] = ctx->input_shape_vec[fft_dims[i]];
      }

      // fill the last dim
      bool must_copy = false;
      auto x_sizes = JUST(oneflow::VectorAt(out_grads, 0))->shape()->dim_vec();
      std::vector<int64_t> pad_amount(x_sizes.size() * 2, 0);
      int64_t last_dim = ctx->dims.back();
      if (x_sizes[last_dim] < ctx->input_shape_vec[last_dim]) {
        must_copy = true;
        auto pad_idx = pad_amount.size() - 2 * last_dim - 1;
        pad_amount[pad_idx] = ctx->input_shape_vec[last_dim] - x_sizes[last_dim];
      }
      auto complex_full_grad =
          must_copy
              ? JUST(functional::ConstantPad(JUST(oneflow::VectorAt(out_grads, 0)), pad_amount, 0))
              : JUST(oneflow::VectorAt(out_grads, 0));
      complex_full_grad =
          JUST(functional::FftC2C(complex_full_grad, NullOpt, ctx->dims, ctx->norm_mode,
                                  /*forward=*/false, /*normalized=*/false));

      JUST(oneflow::VectorAt(*in_grads, 0)) = JUST(functional::Real(complex_full_grad));
    }

    return Maybe<void>::Ok();
  }
};

struct FftC2CCaptureState : public AutoGradCaptureState {
  bool requires_grad = false;
  bool forward = false;
  std::vector<int64_t> dims;
  int32_t norm_mode = 0;
};

class FftC2C : public OpExprGradFunction<FftC2CCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(FftC2CCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1) << "RuntimeError: assert `inputs.size() == 1`";

    ctx->requires_grad = JUST(oneflow::VectorAt(inputs, 0))->requires_grad();
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

    ctx->forward = JUST(attrs.GetAttr<bool>("forward"));
    ctx->dims = JUST(attrs.GetAttr<std::vector<int64_t>>("dims"));
    ctx->norm_mode = JUST(attrs.GetAttr<int32_t>("norm_mode"));

    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const FftC2CCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1) << "RuntimeError: assert `out_grads.size() == 1`";
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

    in_grads->resize(1);
    JUST(oneflow::VectorAt(*in_grads, 0)) = JUST(functional::FftC2C(
        JUST(oneflow::VectorAt(out_grads, 0)), NullOpt, ctx->dims, ctx->norm_mode,
        /*forward=*/!(ctx->forward), /*normalized=*/false));
    return Maybe<void>::Ok();
  }
};

struct FftC2RCaptureState : public AutoGradCaptureState {
  bool requires_grad = false;
  std::vector<int64_t> dims;
  int32_t norm_mode = 0;
  int64_t last_dim_size = 1;
  DimVector input_shape_vec;
};

class FftC2R : public OpExprGradFunction<FftC2RCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(FftC2RCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1) << "RuntimeError: assert `inputs.size() == 1`";
    ctx->requires_grad = JUST(oneflow::VectorAt(inputs, 0))->requires_grad();
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

    ctx->dims = JUST(attrs.GetAttr<std::vector<int64_t>>("dims"));
    ctx->norm_mode = JUST(attrs.GetAttr<int32_t>("norm_mode"));
    ctx->last_dim_size = JUST(attrs.GetAttr<int64_t>("last_dim_size"));
    ctx->input_shape_vec = JUST(oneflow::VectorAt(inputs, 0))->shape()->dim_vec();

    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const FftC2RCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1) << "RuntimeError: out_grads.size() == 1";
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

    in_grads->resize(1);

    // NOTE: set `forward` True to prevent conjugating result
    auto complex_grad = JUST(functional::FftR2C(
        JUST(oneflow::VectorAt(out_grads, 0)), NullOpt, ctx->dims, ctx->norm_mode,
        /*onesided=*/true, /*forward=*/true, /*normalized=*/false));  // no need conj
    Shape input_shape(ctx->input_shape_vec);
    int64_t last_dim = ctx->dims.back();
    auto double_length =
        JUST(oneflow::VectorAt(out_grads, 0))->dim(last_dim) - complex_grad->dim(last_dim);
    auto in_grad = complex_grad;

    // Mul by 2, and slice
    if (double_length > 0) {
      in_grad = JUST(functional::Narrow(complex_grad, last_dim, 1,
                                        double_length));  // will change shape of in_grad
      in_grad = JUST(functional::ScalarMul(in_grad, 2, /*inplace=*/true));
    }

    std::vector<int64_t> slice_st(input_shape.size(), 0);
    std::vector<int64_t> slice_end(input_shape.begin(), input_shape.end());
    std::vector<int64_t> slice_step(input_shape.size(), 1);
    auto sliced_tensor =
        JUST(functional::Slice(complex_grad, slice_st, slice_end, slice_step, false));

    JUST(oneflow::VectorAt(*in_grads, 0)) = sliced_tensor;
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("fft_r2c", FftR2C);
REGISTER_OP_EXPR_GRAD_FUNCTION("fft_c2c", FftC2C);
REGISTER_OP_EXPR_GRAD_FUNCTION("fft_c2r", FftC2R);

}  // namespace one

}  // namespace oneflow