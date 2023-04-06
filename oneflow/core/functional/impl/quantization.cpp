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

#include "oneflow/core/functional/impl/binary_functor.h"

#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/mutable_attr_map.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/functional/function_library.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

class MinMaxObserverFunctor {
 public:
  MinMaxObserverFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("min_max_observer")
                         .Input("in")
                         .Output("scale")
                         .Output("zero_point")
                         .Build());
  }
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& in,
                                const std::string& quantization_formula,
                                const int32_t& quantization_bit,
                                const std::string& quantization_scheme,
                                const bool& per_layer_quantization) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("quantization_formula", "quantization_bit",
                                                 "quantization_scheme", "per_layer_quantization");
    attrs.SetAllAttrs(quantization_formula, quantization_bit, quantization_scheme,
                      per_layer_quantization);
    return OpInterpUtil::Dispatch<TensorTuple>(*op_, {in}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class MovingAverageMinMaxObserverFunctor {
 public:
  MovingAverageMinMaxObserverFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("moving_average_min_max_observer")
                         .Input("in")
                         .Input("current_train_step")
                         .Input("moving_max")
                         .Input("moving_min")
                         .Output("scale")
                         .Output("zero_point")
                         .Build());
  }
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& in,
                                const std::shared_ptr<one::Tensor>& current_train_step,
                                const std::shared_ptr<one::Tensor>& moving_max,
                                const std::shared_ptr<one::Tensor>& moving_min,
                                const bool& training, const int64_t& stop_update_after_iters,
                                const std::string& quantization_formula,
                                const int32_t& quantization_bit,
                                const std::string& quantization_scheme,
                                const float& momentum) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("training", "quantization_formula",
                                                 "stop_update_after_iters", "quantization_bit",
                                                 "quantization_scheme", "momentum");
    attrs.SetAllAttrs(training, quantization_formula, stop_update_after_iters, quantization_bit,
                      quantization_scheme, momentum);
    return OpInterpUtil::Dispatch<TensorTuple>(
        *op_, {in, current_train_step, moving_max, moving_min}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class FakeQuantizationFunctor {
 public:
  FakeQuantizationFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("fake_quantization")
                         .Input("in")
                         .Input("scale")
                         .Input("zero_point")
                         .Output("out")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& in,
                           const std::shared_ptr<one::Tensor>& scale,
                           const std::shared_ptr<one::Tensor>& zero_point,
                           const std::string& quantization_formula, const int32_t& quantization_bit,
                           const std::string& quantization_scheme) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("quantization_formula", "quantization_bit",
                                                 "quantization_scheme");
    attrs.SetAllAttrs(quantization_formula, quantization_bit, quantization_scheme);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {in, scale, zero_point}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class QuantizationFunctor {
 public:
  QuantizationFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("quantization")
                         .Input("in")
                         .Input("scale")
                         .Input("zero_point")
                         .Output("out")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& in,
                           const std::shared_ptr<one::Tensor>& scale,
                           const std::shared_ptr<one::Tensor>& zero_point,
                           const std::string quantization_formula, const int32_t& quantization_bit,
                           const std::string quantization_scheme) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("quantization_formula", "quantization_bit",
                                                 "quantization_scheme");
    attrs.SetAllAttrs(quantization_formula, quantization_bit, quantization_scheme);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {in, scale, zero_point}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class GroupwiseDequantizeFunctor {
 public:
  GroupwiseDequantizeFunctor() {
    symmetric_op_ = CHECK_JUST(
        one::OpBuilder("groupwise_dequantize").Input("in").Input("scale").Output("out").Build());
    asymmetric_op_ = CHECK_JUST(one::OpBuilder("groupwise_dequantize")
                                    .Input("in")
                                    .Input("scale")
                                    .Input("zero")
                                    .Output("out")
                                    .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& in,
                           const std::shared_ptr<one::Tensor>& scale,
                           const Optional<one::Tensor>& zero, const int32_t& num_bits,
                           const bool& symmetric, const int64_t& group_dim,
                           const int64_t& group_size) const {
    auto& attrs =
        THREAD_CACHED_MUTABLE_ATTR_MAP("num_bits", "symmetric", "group_dim", "group_size");
    CHECK_OR_RETURN(num_bits == 4 || num_bits == 8) << "num_bits should be 4 or 8.";
    CHECK_GE_OR_RETURN(in->shape()->NumAxes(), 1)
        << "The number of dimensions for tensor in should be greater than or equal to 1.";
    const int64_t regularized_group_dim =
        group_dim < 0 ? in->shape()->NumAxes() + group_dim : group_dim;
    CHECK_OR_RETURN(regularized_group_dim >= 0 && regularized_group_dim < in->shape()->NumAxes())
        << "group_dim should be in range [-" << in->shape()->NumAxes() << ","
        << in->shape()->NumAxes() << ").";
    const int64_t group_dim_size =
        in->shape()->At(regularized_group_dim)
        * (regularized_group_dim == in->shape()->NumAxes() - 1 ? 8 / num_bits : 1);
    const int64_t regularized_group_size = group_size < 0 ? group_dim_size : group_size;
    CHECK_OR_RETURN(regularized_group_size > 0 && regularized_group_size <= group_dim_size)
        << "group_size should be in range (0," << group_dim_size << "].";
    CHECK_EQ_OR_RETURN(group_dim_size % regularized_group_size, 0)
        << "group_size should be a divisor of " << group_dim_size << ".";
    const int64_t num_groups = group_dim_size / regularized_group_size;
    if (symmetric) {
      CHECK_OR_RETURN(in->dtype()->data_type() == DataType::kUInt8
                      || in->dtype()->data_type() == DataType::kInt8)
          << "The dtype of tensor in should be int8 or uint8.";
    } else {
      CHECK_OR_RETURN(in->dtype()->data_type() == DataType::kUInt8)
          << "The dtype of tensor in should be uint8.";
    }
    CHECK_EQ_OR_RETURN(scale->shape()->NumAxes(), in->shape()->NumAxes())
        << "The number of dimensions of tensor scale should be equal to tensor in.";
    for (int64_t i = 0; i < in->shape()->NumAxes(); ++i) {
      if (i == regularized_group_dim) {
        CHECK_EQ_OR_RETURN(scale->shape()->At(i), num_groups)
            << "The size of the " << i << "-th dimension of tensor scale should be equal to "
            << num_groups;
      } else if (i == in->shape()->NumAxes() - 1) {
        CHECK_EQ_OR_RETURN(scale->shape()->At(i), in->shape()->At(i) * (8 / num_bits))
            << "The size of the " << i << "-th dimension of tensor scale should be equal to "
            << in->shape()->At(i) * (8 / num_bits) << ".";
      } else {
        CHECK_EQ_OR_RETURN(scale->shape()->At(i), in->shape()->At(i))
            << "The size of the " << i
            << "-th dimension of tensor scale should be equal to tensor in.";
      }
    }
    if (!symmetric) {
      CHECK_OR_RETURN(zero) << "When symmetric is False, tensor zero should be specified.";
      CHECK_OR_RETURN(JUST(zero)->dtype() == scale->dtype())
          << "The dtype of the zero tensor should be the same as the scale "
             "tensor.";
      CHECK_OR_RETURN(*JUST(zero)->shape() == *scale->shape())
          << "The shape of zero tensor should be equal to tensor scale.";
    } else {
      CHECK_OR_RETURN(!zero) << "When symmetric is True, tensor zero should be None.";
    }
    attrs.SetAllAttrs(num_bits, symmetric, regularized_group_dim, regularized_group_size);
    if (symmetric) {
      return OpInterpUtil::Dispatch<Tensor>(*symmetric_op_, {in, scale}, attrs);
    } else {
      return OpInterpUtil::Dispatch<Tensor>(*asymmetric_op_, {in, scale, JUST(zero)}, attrs);
    }
  }

 private:
  std::shared_ptr<OpExpr> symmetric_op_;
  std::shared_ptr<OpExpr> asymmetric_op_;
};

class FusedLinearWithGroupwiseQuantizedWeightFunctor {
 public:
  FusedLinearWithGroupwiseQuantizedWeightFunctor() {
    symmetric_with_bias_op_ =
        CHECK_JUST(one::OpBuilder("fused_linear_with_groupwise_quantized_weight")
                       .Input("x")
                       .Input("w")
                       .Input("w_scale")
                       .Input("b")
                       .Output("out")
                       .Build());
    symmetric_without_bias_op_ =
        CHECK_JUST(one::OpBuilder("fused_linear_with_groupwise_quantized_weight")
                       .Input("x")
                       .Input("w")
                       .Input("w_scale")
                       .Output("out")
                       .Build());
    asymmetric_with_bias_op_ =
        CHECK_JUST(one::OpBuilder("fused_linear_with_groupwise_quantized_weight")
                       .Input("x")
                       .Input("w")
                       .Input("w_scale")
                       .Input("w_zero")
                       .Input("b")
                       .Output("out")
                       .Build());
    asymmetric_without_bias_op_ =
        CHECK_JUST(one::OpBuilder("fused_linear_with_groupwise_quantized_weight")
                       .Input("x")
                       .Input("w")
                       .Input("w_scale")
                       .Input("w_zero")
                       .Output("out")
                       .Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& w,
                           const std::shared_ptr<one::Tensor>& w_scale,
                           const Optional<one::Tensor>& w_zero, const Optional<one::Tensor>& b,
                           const int32_t& num_bits, const bool& symmetric, const int64_t& group_dim,
                           const int64_t& group_size) const {
    CHECK_GE_OR_RETURN(x->shape()->NumAxes(), 2)
        << "The number of dimensions for tensor x should be greater than or equal to 2.";
    const int64_t m = x->shape()->Count(0, x->shape()->NumAxes() - 1);
    const int64_t k = x->shape()->At(x->shape()->NumAxes() - 1);
    CHECK_OR_RETURN(num_bits == 4 || num_bits == 8) << "num_bits should be 4 or 8.";
    CHECK_EQ_OR_RETURN(w->shape()->NumAxes(), 2)
        << "The number of dimensions for tensor w should be equal to 2.";
    CHECK_EQ_OR_RETURN(k % (8 / num_bits), 0)
        << "The size of the last dimension of x should be a multiple of (8/num_bits).";
    CHECK_EQ_OR_RETURN(w->shape()->At(1), k / (8 / num_bits))
        << "The size of second dimension of tensor w should be equal to " << k / (8 / num_bits);
    const int64_t n = w->shape()->At(0);
    const int64_t regularized_group_dim =
        group_dim < 0 ? w->shape()->NumAxes() + group_dim : group_dim;
    CHECK_OR_RETURN(regularized_group_dim == 0 || regularized_group_dim == 1)
        << "group_dim should be in range [-2,2).";
    const int64_t group_dim_size = regularized_group_dim == 0 ? n : k;
    const int64_t regularized_group_size = group_size < 0 ? group_dim_size : group_size;
    CHECK_OR_RETURN(regularized_group_size > 0 && regularized_group_size <= group_dim_size)
        << "group_size should be in range (0," << group_dim_size << "].";
    CHECK_EQ_OR_RETURN(group_dim_size % regularized_group_size, 0)
        << "group_size should be a divisor of " << group_dim_size << ".";
    const int64_t num_groups = group_dim_size / regularized_group_size;
    if (symmetric) {
      CHECK_OR_RETURN(w->dtype()->data_type() == DataType::kUInt8
                      || w->dtype()->data_type() == DataType::kInt8)
          << "The dtype of tensor w should be int8 or uint8.";
    } else {
      CHECK_OR_RETURN(w->dtype()->data_type() == DataType::kUInt8)
          << "The dtype of tensor w should be uint8.";
    }
    CHECK_EQ_OR_RETURN(w_scale->shape()->NumAxes(), 2)
        << "The number of dimensions of tensor w_scale should be equal to 2.";
    for (int64_t i = 0; i < 2; ++i) {
      if (i == regularized_group_dim) {
        CHECK_EQ_OR_RETURN(w_scale->shape()->At(i), num_groups)
            << "The size of the " << i << "-th dimension of tensor w_scale should be equal to "
            << num_groups;
      } else if (i == 1) {
        CHECK_EQ_OR_RETURN(w_scale->shape()->At(i), k)
            << "The size of the " << i << "-th dimension of tensor w_scale should be equal to " << k
            << ".";
      } else {
        CHECK_EQ_OR_RETURN(w_scale->shape()->At(i), w->shape()->At(i))
            << "The size of the " << i
            << "-th dimension of tensor w_scale should be equal to tensor w.";
      }
    }
    CHECK_OR_RETURN(w_scale->dtype() == x->dtype())
        << "The dtype of the w_scale tensor should be the same as the x tensor.";
    if (!symmetric) {
      CHECK_OR_RETURN(w_zero) << "When symmetric is False, tensor w_zero should be specified.";
      CHECK_OR_RETURN(JUST(w_zero)->dtype() == w_scale->dtype())
          << "The dtype of the w_zero tensor should be the same as the w_scale "
             "tensor.";
      CHECK_OR_RETURN(*JUST(w_zero)->shape() == *w_scale->shape())
          << "The shape of w_zero tensor should be equal to tensor w_scale.";
    } else {
      CHECK_OR_RETURN(!w_zero) << "When symmetric is True, tensor w_zero should be None.";
    }

    if (b) {
      CHECK_OR_RETURN(JUST(b)->dtype() == x->dtype())
          << "The dtype of the b tensor should be the same as the x tensor.";
      CHECK_EQ_OR_RETURN(JUST(b)->shape()->NumAxes(), 1)
          << "The number of dimensions for tensor b should be equal to 1.";
      CHECK_EQ_OR_RETURN(JUST(b)->shape()->At(0), n)
          << "The size of first dimension of tensor b should be equal to the size of first "
             "dimension of tensor w";
    }

    if (m > 8) {
      const auto w_dequantized = JUST(functional::GroupwiseDequantize(
          w, w_scale, w_zero, num_bits, symmetric, group_dim, group_size));
      if (b) {
        return JUST(functional::FusedMatmulBias(x, w_dequantized, JUST(b), Optional<one::Tensor>(),
                                                1.0, 1.0));
      } else {
        return JUST(functional::MatMul(x, w_dequantized, false, true, 1.0));
      }
    }
    auto& attrs =
        THREAD_CACHED_MUTABLE_ATTR_MAP("num_bits", "symmetric", "group_dim", "group_size");

    attrs.SetAllAttrs(num_bits, symmetric, regularized_group_dim, regularized_group_size);

    if (symmetric) {
      if (b) {
        return OpInterpUtil::Dispatch<Tensor>(*symmetric_with_bias_op_, {x, w, w_scale, JUST(b)},
                                              attrs);
      } else {
        return OpInterpUtil::Dispatch<Tensor>(*symmetric_without_bias_op_, {x, w, w_scale}, attrs);
      }
    } else {
      if (b) {
        return OpInterpUtil::Dispatch<Tensor>(*asymmetric_with_bias_op_,
                                              {x, w, w_scale, JUST(w_zero), JUST(b)}, attrs);
      } else {
        return OpInterpUtil::Dispatch<Tensor>(*asymmetric_without_bias_op_,
                                              {x, w, w_scale, JUST(w_zero)}, attrs);
      }
    }
  }

 private:
  std::shared_ptr<OpExpr> symmetric_with_bias_op_;
  std::shared_ptr<OpExpr> symmetric_without_bias_op_;
  std::shared_ptr<OpExpr> asymmetric_with_bias_op_;
  std::shared_ptr<OpExpr> asymmetric_without_bias_op_;
};

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) { m.add_functor<impl::FakeQuantizationFunctor>("FakeQuantization"); };
ONEFLOW_FUNCTION_LIBRARY(m) { m.add_functor<impl::QuantizationFunctor>("Quantization"); };
ONEFLOW_FUNCTION_LIBRARY(m) { m.add_functor<impl::MinMaxObserverFunctor>("MinMaxObserver"); };
ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::MovingAverageMinMaxObserverFunctor>("MovingAverageMinMaxObserver");
};
ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::GroupwiseDequantizeFunctor>("GroupwiseDequantize");
  m.add_functor<impl::FusedLinearWithGroupwiseQuantizedWeightFunctor>(
      "FusedLinearWithGroupwiseQuantizedWeight");
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
