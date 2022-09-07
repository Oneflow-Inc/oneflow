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

#include "oneflow/core/common/scalar.h"
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/mutable_attr_map.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/functional/function_library.h"
#include "oneflow/core/functional/sequence_function.h"
#include "oneflow/core/functional/functional_api.yaml.h"
#include "oneflow/core/functional/impl/common.h"
#include "oneflow/core/functional/impl/unary_functor.h"
#include "oneflow/core/common/container_util.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

class ConvBiasGradFunctor {
 public:
  ConvBiasGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("conv_bias_grad").Input("dy").Output("bias_diff").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy, const int32_t& num_spatial_dims,
                           const std::string& data_format) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("num_spatial_dims", "data_format");
    attrs.SetAllAttrs(num_spatial_dims, data_format);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ConvFilterGradFunctor {
 public:
  ConvFilterGradFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("conv_filter_grad").Input("dy").Input("x").Output("filter_diff").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& x, const int32_t& num_spatial_dims,
                           const std::vector<int32_t>& kernel_size,
                           const std::vector<int32_t>& strides,
                           const std::vector<int32_t>& padding_before,
                           const std::vector<int32_t>& dilation_rate, const int32_t& groups,
                           const std::string& data_format) const {
    auto& attrs =
        THREAD_CACHED_MUTABLE_ATTR_MAP("num_spatial_dims", "kernel_size", "strides",
                                       "padding_before", "dilation_rate", "groups", "data_format");
    attrs.SetAllAttrs(num_spatial_dims, kernel_size, strides, padding_before, dilation_rate, groups,
                      data_format);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ConvDataGradFunctor {
 public:
  ConvDataGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("conv_data_grad")
                         .Input("dy")
                         .Input("filter")
                         .Input("x_like")
                         .Output("dx")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& weight,
                           const std::shared_ptr<one::Tensor>& x, const int32_t& num_spatial_dims,
                           const std::vector<int32_t>& kernel_size,
                           const std::vector<int32_t>& strides,
                           const std::vector<int32_t>& padding_before,
                           const std::vector<int32_t>& dilation_rate, const int32_t& groups,
                           const std::string& data_format) const {
    auto& attrs =
        THREAD_CACHED_MUTABLE_ATTR_MAP("num_spatial_dims", "kernel_size", "strides",
                                       "padding_before", "dilation_rate", "groups", "data_format");
    attrs.SetAllAttrs(num_spatial_dims, kernel_size, strides, padding_before, dilation_rate, groups,
                      data_format);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, weight, JUST(x->detach())}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class EmbeddingGradFunctor {
 public:
  EmbeddingGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("embedding_grad")
                         .Input("dy")
                         .Input("weight")
                         .Input("indices")
                         .Output("dx")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& weight,
                           const std::shared_ptr<one::Tensor>& indices, const int64_t& padding_idx,
                           const bool& scale_grad_by_freq) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("padding_idx", "scale_grad_by_freq");
    attrs.SetAllAttrs(padding_idx, scale_grad_by_freq);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, weight, indices}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class MaxPoolNdGradFunctor {
 public:
  MaxPoolNdGradFunctor() {
    for (int ndims = 1; ndims <= 3; ++ndims) {
      const auto& op_type_name = GetOpTypeName(ndims);
      op_expr_map_[op_type_name] = CHECK_JUST(
          one::OpBuilder(op_type_name).Input("x").Input("indice").Input("dy").Output("dx").Build());
    }
  }
  static std::string GetOpTypeName(const int32_t& ndims) {
    return "max_pool_" + std::to_string(ndims) + "d_grad";
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& indice,
                           const std::shared_ptr<one::Tensor>& dy, const int32_t& ndims,
                           const std::string& data_format, const std::vector<int32_t>& padding,
                           const std::vector<int32_t>& kernel_size,
                           const std::vector<int32_t>& stride, const std::vector<int32_t>& dilation,
                           const bool& return_indices, const bool& ceil_mode) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("data_format", "padding", "kernel_size", "stride",
                                                 "dilation", "return_indices", "ceil_mode");
    attrs.SetAllAttrs(data_format, padding, kernel_size, stride, dilation, return_indices,
                      ceil_mode);
    const auto& op_type_name = GetOpTypeName(ndims);
    const auto& it = op_expr_map_.find(op_type_name);
    CHECK_OR_RETURN(it != op_expr_map_.end())
        << Error::RuntimeError() << "Encounter unsupported op " << op_type_name
        << " in MaxPoolNdGradFunctor.";
    CHECK_NOTNULL_OR_RETURN(it->second);  // NOLINT(maybe-need-error-msg)
    return OpInterpUtil::Dispatch<Tensor>(*it->second, {x, indice, dy}, attrs);
  }

 protected:
  std::unordered_map<std::string, std::shared_ptr<OpExpr>> op_expr_map_;
};

class TFPoolNdGradFunctor {
 public:
  TFPoolNdGradFunctor() {
    for (const auto& mode : {"tf_max", "tf_avg"}) {
      for (int ndims = 1; ndims <= 3; ++ndims) {
        const auto& op_type_name = GetOpTypeName(mode, ndims);
        op_expr_map_[op_type_name] = CHECK_JUST(
            one::OpBuilder(op_type_name).Input("x").Input("y").Input("dy").Output("dx").Build());
      }
    }
  }
  static std::string GetOpTypeName(const std::string& mode, const int32_t& ndims) {
    return mode + "_pool_" + std::to_string(ndims) + "d_grad";
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& y,
                           const std::shared_ptr<one::Tensor>& dy, const std::string& mode,
                           const int32_t& ndims, const std::string& data_format,
                           const std::string& padding, const std::vector<int32_t>& padding_before,
                           const std::vector<int32_t>& padding_after,
                           const std::vector<int32_t>& pool_size,
                           const std::vector<int32_t>& strides, const bool& ceil_mode) const {
    auto& attrs =
        THREAD_CACHED_MUTABLE_ATTR_MAP("data_format", "padding", "padding_before", "padding_after",
                                       "pool_size", "strides", "ceil_mode");
    attrs.SetAllAttrs(data_format, padding, padding_before, padding_after, pool_size, strides,
                      ceil_mode);
    const auto& op_type_name = GetOpTypeName(mode, ndims);
    const auto& it = op_expr_map_.find(op_type_name);
    CHECK_OR_RETURN(it != op_expr_map_.end())
        << Error::RuntimeError() << "Encounter unsupported op " << op_type_name
        << " in TFPoolNdGradFunctor.";
    CHECK_NOTNULL_OR_RETURN(it->second);  // NOLINT(maybe-need-error-msg)
    return OpInterpUtil::Dispatch<Tensor>(*it->second, {x, y, dy}, attrs);
  }

 protected:
  std::unordered_map<std::string, std::shared_ptr<OpExpr>> op_expr_map_;
};

class AdaptivePoolNdGradFunctor {
 public:
  AdaptivePoolNdGradFunctor() {
    for (const auto& mode : {"avg"}) {
      for (int ndims = 1; ndims <= 3; ++ndims) {
        const auto& op_type_name = GetOpTypeName(mode, ndims);
        op_expr_map_[op_type_name] =
            CHECK_JUST(one::OpBuilder(op_type_name).Input("x").Input("dy").Output("dx").Build());
      }
    }
  }
  static std::string GetOpTypeName(const std::string& mode, const int32_t& ndims) {
    return "adaptive_" + mode + "_pool" + std::to_string(ndims) + "d_grad";
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& dy, const std::string& mode,
                           const int32_t& ndims) const {
    const auto& op_type_name = GetOpTypeName(mode, ndims);
    const auto& it = op_expr_map_.find(op_type_name);
    CHECK_OR_RETURN(it != op_expr_map_.end())
        << Error::RuntimeError() << "Encounter unsupported op " << op_type_name
        << " in AdaptivePoolNdGradFunctor.";
    CHECK_NOTNULL_OR_RETURN(it->second);  // NOLINT(maybe-need-error-msg)
    return OpInterpUtil::Dispatch<Tensor>(*it->second, {x, dy});
  }

 protected:
  std::unordered_map<std::string, std::shared_ptr<OpExpr>> op_expr_map_;
};

class SparseCrossEntropyGradFunctor {
 public:
  SparseCrossEntropyGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("sparse_cross_entropy_grad")
                         .Input("prediction")
                         .Input("label")
                         .Input("dy")
                         .Output("prediction_diff")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& prediction,
                           const std::shared_ptr<one::Tensor>& label,
                           const std::shared_ptr<one::Tensor>& dy, const int64_t& depth) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("depth");
    attrs.SetAllAttrs(depth);

    return OpInterpUtil::Dispatch<Tensor>(*op_, {prediction, label, dy}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class SparseCrossEntropyMsGradFunctor {
 public:
  SparseCrossEntropyMsGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("sparse_cross_entropy_ms_grad")
                         .Input("prediction")
                         .Input("label")
                         .Input("dy")
                         .Output("prediction_diff")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& prediction,
                           const std::shared_ptr<one::Tensor>& label,
                           const std::shared_ptr<one::Tensor>& dy, const int64_t& depth) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("depth");
    attrs.SetAllAttrs(depth);

    return OpInterpUtil::Dispatch<Tensor>(*op_, {prediction, label, dy}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class SparseSoftmaxCrossEntropyGrad {
 public:
  SparseSoftmaxCrossEntropyGrad() {
    op_ = CHECK_JUST(one::OpBuilder("sparse_softmax_cross_entropy_grad")
                         .Input("prob")
                         .Input("label")
                         .Input("dy")
                         .Output("prediction_diff")
                         .Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& prob,
                           const std::shared_ptr<one::Tensor>& label, const int64_t& depth) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("depth");
    attrs.SetAllAttrs(depth);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {prob, label, dy}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class SparseSoftmaxCrossEntropyMsGrad {
 public:
  SparseSoftmaxCrossEntropyMsGrad() {
    op_ = CHECK_JUST(one::OpBuilder("sparse_softmax_cross_entropy_ms_grad")
                         .Input("prob")
                         .Input("label")
                         .Input("dy")
                         .Output("prediction_diff")
                         .Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& prob,
                           const std::shared_ptr<one::Tensor>& label, const int64_t& depth) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("depth");
    attrs.SetAllAttrs(depth);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {prob, label, dy}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class SmoothL1LossGradFunctor {
 public:
  SmoothL1LossGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("smooth_l1_loss_grad")
                         .Input("dy")
                         .Input("input")
                         .Input("target")
                         .Output("dx")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& target, const float& beta) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("beta");
    attrs.SetAllAttrs(beta);

    return OpInterpUtil::Dispatch<one::Tensor>(*op_, {dy, input, target}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class KLDivLossGradFunctor {
 public:
  KLDivLossGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("kl_div_loss_grad")
                         .Input("dy")
                         .Input("input")
                         .Input("target")
                         .Output("dx")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& target,
                           const bool log_target) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("log_target");
    attrs.SetAllAttrs(log_target);

    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, input, target}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class KLDivLossTargetGradFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& target,
                           const bool log_target) const {
    if (log_target) {
      return functional::sequence_function(functional::Sub)
          .then([](const std::shared_ptr<Tensor>& input) {
            return functional::ScalarAdd(1, input, /*alpha=*/Scalar(1));
          })
          .then(std::bind(functional::Mul, std::placeholders::_1, JUST(functional::Exp(target))))
          .then(std::bind(functional::Mul, std::placeholders::_1, dy))
          .call(target, input, /*alpha=*/1, /*inplace=*/false);
    } else {
      return functional::sequence_function(functional::Log)
          .then([](const std::shared_ptr<Tensor>& input) {
            return functional::ScalarAdd(1, input, /*alpha=*/Scalar(1));
          })
          .then(std::bind(functional::Sub, std::placeholders::_1, input, /*alpha=*/1,
                          /*inplace=*/false))
          .then(std::bind(functional::Mul, std::placeholders::_1, dy))
          .call(target);
    }
  }
};

class NLLGradFunctor {
 public:
  NLLGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("nll_grad")
                         .Input("out_grad")
                         .Input("input")
                         .Input("target")
                         .Output("in_grad")
                         .Build());

    op_weight_ = CHECK_JUST(one::OpBuilder("nll_grad")
                                .Input("out_grad")
                                .Input("input")
                                .Input("target")
                                .Input("weight")
                                .Output("in_grad")
                                .Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& out_grad,
                           const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& target,
                           const Optional<one::Tensor>& weight, const int64_t ignore_index) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("ignore_index");
    attrs.SetAllAttrs(ignore_index);

    if (weight) {
      return OpInterpUtil::Dispatch<one::Tensor>(
          *op_weight_, {out_grad, input, target, JUST(JUST(weight)->detach())}, attrs);
    } else {
      return OpInterpUtil::Dispatch<one::Tensor>(*op_, {out_grad, input, target}, attrs);
    }
  }

 private:
  std::shared_ptr<OpExpr> op_;
  std::shared_ptr<OpExpr> op_weight_;
};

class BinaryCrossEntropyLossGradFunctor {
 public:
  BinaryCrossEntropyLossGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("binary_cross_entropy_grad")
                         .Input("dy")
                         .Input("input")
                         .Input("target")
                         .Output("dx")
                         .Build());
    op_weight_ = CHECK_JUST(one::OpBuilder("binary_cross_entropy_grad")
                                .Input("dy")
                                .Input("input")
                                .Input("target")
                                .Input("weight")
                                .Output("dx")
                                .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& target,
                           const Optional<one::Tensor>& weight) const {
    if (weight) {
      return OpInterpUtil::Dispatch<one::Tensor>(*op_weight_, {dy, input, target, JUST(weight)});
    } else {
      return OpInterpUtil::Dispatch<one::Tensor>(*op_, {dy, input, target});
    }
  }

 private:
  std::shared_ptr<OpExpr> op_;
  std::shared_ptr<OpExpr> op_weight_;
};

class BinaryCrossEntropyLossTargetGradFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& target,
                           const Optional<one::Tensor>& weight) const {
    auto grad = functional::sequence_function(functional::Reciprocal)
                    .then([](const std::shared_ptr<Tensor>& input) {
                      return functional::ScalarAdd(-1, input, /*alpha=*/Scalar(1));
                    })
                    .then(functional::Log)
                    .then(std::bind(functional::Mul, dy, std::placeholders::_1))
                    .call(input);
    if (weight) {
      return functional::Mul(JUST(grad), JUST(weight));
    } else {
      return grad;
    }
  }
};

class BinaryCrossEntropyWithLogitsLossGradFunctor {
 public:
  BinaryCrossEntropyWithLogitsLossGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("binary_cross_entropy_with_logits_grad")
                         .Input("dy")
                         .Input("input")
                         .Input("target")
                         .Output("dx")
                         .Build());
    op_weight_ = CHECK_JUST(one::OpBuilder("binary_cross_entropy_with_logits_grad")
                                .Input("dy")
                                .Input("input")
                                .Input("target")
                                .Input("weight")
                                .Output("dx")
                                .Build());
    op_pos_ = CHECK_JUST(one::OpBuilder("binary_cross_entropy_with_logits_grad")
                             .Input("dy")
                             .Input("input")
                             .Input("target")
                             .Input("pos_weight")
                             .Output("dx")
                             .Build());
    op_weight_pos_ = CHECK_JUST(one::OpBuilder("binary_cross_entropy_with_logits_grad")
                                    .Input("dy")
                                    .Input("input")
                                    .Input("target")
                                    .Input("weight")
                                    .Input("pos_weight")
                                    .Output("dx")
                                    .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& target,
                           const Optional<one::Tensor>& weight,
                           const Optional<one::Tensor>& pos_weight) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("has_pos_weight");
    attrs.SetAllAttrs(pos_weight.has_value());

    if (weight) {
      if (pos_weight) {
        return OpInterpUtil::Dispatch<one::Tensor>(
            *op_weight_pos_, {dy, input, target, JUST(weight), JUST(pos_weight)}, attrs);
      } else {
        return OpInterpUtil::Dispatch<one::Tensor>(*op_weight_, {dy, input, target, JUST(weight)},
                                                   attrs);
      }
    } else {
      if (pos_weight) {
        return OpInterpUtil::Dispatch<one::Tensor>(*op_pos_, {dy, input, target, JUST(pos_weight)},
                                                   attrs);
      } else {
        return OpInterpUtil::Dispatch<one::Tensor>(*op_, {dy, input, target}, attrs);
      }
    }
  }

 private:
  std::shared_ptr<OpExpr> op_;
  std::shared_ptr<OpExpr> op_weight_;
  std::shared_ptr<OpExpr> op_pos_;
  std::shared_ptr<OpExpr> op_weight_pos_;
};

class BinaryCrossEntropyWithLogitsLossTargetGradFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& target,
                           const Optional<one::Tensor>& weight,
                           const Optional<one::Tensor>& pos_weight) const {
    if (pos_weight) {
      auto sig = JUST(functional::Sigmoid(input));
      auto one_sub_sig = JUST(functional::ScalarSub(1, sig, /*alpha=*/1));
      auto a = JUST(functional::sequence_function(functional::Log)
                        .then(std::bind(functional::Mul, std::placeholders::_1, JUST(pos_weight)))
                        .call(sig));
      auto b = JUST(functional::Log(one_sub_sig));
      auto grad = functional::Sub(b, a, /*alpha=*/1, false);

      return weight ? functional::Mul(JUST(grad), JUST(weight)) : grad;
    } else {
      auto grad = functional::sequence_function(functional::Negative)
                      .then(std::bind(functional::Mul, std::placeholders::_1, dy))
                      .call(input);
      return weight ? functional::Mul(JUST(grad), JUST(weight)) : grad;
    }
  }
};

class BinaryCrossEntropyWithLogitsReduceMeanLossGradFunctor {
 public:
  BinaryCrossEntropyWithLogitsReduceMeanLossGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("binary_cross_entropy_with_logits_reduce_mean_grad")
                         .Input("dy")
                         .Input("input")
                         .Input("target")
                         .Output("dx")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& target) const {
    return OpInterpUtil::Dispatch<one::Tensor>(*op_, {dy, input, target});
  }

 private:
  std::shared_ptr<OpExpr> op_;
  std::shared_ptr<OpExpr> op_weight_;
  std::shared_ptr<OpExpr> op_pos_;
  std::shared_ptr<OpExpr> op_weight_pos_;
};

class BinaryCrossEntropyWithLogitsReduceMeanLossTargetGradFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& target) const {
    return functional::sequence_function(functional::Mul)
        .then(functional::Negative)
        .call(dy, input);
  }
};

class CombinedMarginLossGradFunctor {
 public:
  CombinedMarginLossGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("combined_margin_loss_grad")
                         .Input("dy")
                         .Input("label")
                         .Input("theta")
                         .Output("dx")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& label,
                           const std::shared_ptr<one::Tensor>& theta, const float& m1,
                           const float& m2, const float& m3, const int64_t& depth) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("m1", "m2", "m3", "depth");
    attrs.SetAllAttrs(m1, m2, m3, depth);
    return OpInterpUtil::Dispatch<one::Tensor>(*op_, {dy, label, theta}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class AffineGridGradFunctor {
 public:
  AffineGridGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("affine_grid_grad").Input("dgrid").Output("dtheta").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dgrid, const Shape& size,
                           const bool& align_corners) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("size", "align_corners");
    attrs.SetAllAttrs(size, align_corners);
    return OpInterpUtil::Dispatch<one::Tensor>(*op_, {dgrid}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class GridSampleGradFunctor {
 public:
  GridSampleGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("grid_sample_grad")
                         .Input("doutput")
                         .Input("input")
                         .Input("grid")
                         .Output("dinput")
                         .Output("dgrid")
                         .Build());
  }
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& doutput,
                                const std::shared_ptr<one::Tensor>& input,
                                const std::shared_ptr<one::Tensor>& grid,
                                const std::string& interpolation_mode,
                                const std::string& padding_mode, const bool& align_corners) const {
    auto& attrs =
        THREAD_CACHED_MUTABLE_ATTR_MAP("interpolation_mode", "padding_mode", "align_corners");
    attrs.SetAllAttrs(interpolation_mode, padding_mode, align_corners);
    return OpInterpUtil::Dispatch<one::TensorTuple>(*op_, {doutput, input, grid}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class CtcLossGradFunctor {
 public:
  CtcLossGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("ctc_loss_grad")
                         .Input("grad_out")
                         .Input("log_probs")
                         .Input("targets")
                         .Input("input_lengths")
                         .Input("target_lengths")
                         .Input("loss")
                         .Input("alpha")
                         .Output("grad")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& grad_out,
                           const std::shared_ptr<one::Tensor>& log_probs,
                           const std::shared_ptr<one::Tensor>& targets,
                           const std::shared_ptr<one::Tensor>& input_lengths,
                           const std::shared_ptr<one::Tensor>& target_lengths,
                           const std::shared_ptr<one::Tensor>& loss,
                           const std::shared_ptr<one::Tensor>& alpha, const int32_t& blank,
                           const bool& zero_infinity, const int64_t& max_target_length) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("blank", "zero_infinity", "max_target_length");
    attrs.SetAllAttrs(blank, zero_infinity, max_target_length);
    return OpInterpUtil::Dispatch<one::Tensor>(
        *op_, {grad_out, log_probs, targets, input_lengths, target_lengths, loss, alpha}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class PadGradFunctor {
 public:
  PadGradFunctor() {
    reflect_pad1d_grad_ =
        CHECK_JUST(one::OpBuilder("reflection_pad1d_grad").Input("dy").Output("dx").Build());
    reflect_pad2d_grad_ =
        CHECK_JUST(one::OpBuilder("reflection_pad2d_grad").Input("dy").Output("dx").Build());
    replicate_pad1d_grad_ =
        CHECK_JUST(one::OpBuilder("replication_pad1d_grad").Input("dy").Output("dx").Build());
    replicate_pad2d_grad_ =
        CHECK_JUST(one::OpBuilder("replication_pad2d_grad").Input("dy").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy, const std::vector<int64_t>& pad,
                           const std::string& mode, const Scalar& value) const {
    const int64_t ndim = dy->shape()->NumAxes();
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("padding");
    attrs.SetAllAttrs(pad);
    if (mode == "reflect") {
      if (ndim == 3) {
        return OpInterpUtil::Dispatch<Tensor>(*reflect_pad1d_grad_, {dy}, attrs);
      } else if (ndim == 4) {
        return OpInterpUtil::Dispatch<Tensor>(*reflect_pad2d_grad_, {dy}, attrs);
      } else {
        UNIMPLEMENTED_THEN_RETURN() << "only 3D/4D reflect padding are supported for now";
      }

    } else if (mode == "replicate") {
      if (ndim == 3) {
        return OpInterpUtil::Dispatch<Tensor>(*replicate_pad1d_grad_, {dy}, attrs);
      } else if (ndim == 4) {
        return OpInterpUtil::Dispatch<Tensor>(*replicate_pad2d_grad_, {dy}, attrs);
      } else {
        UNIMPLEMENTED_THEN_RETURN() << "only 3D/4D replicate padding are supported for now";
      }
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "Pad mode is " << mode
                                  << ", but only constant, reflect and replicate are valid.";
    }
  }

 private:
  std::shared_ptr<OpExpr> reflect_pad1d_grad_;
  std::shared_ptr<OpExpr> reflect_pad2d_grad_;
  std::shared_ptr<OpExpr> replicate_pad1d_grad_;
  std::shared_ptr<OpExpr> replicate_pad2d_grad_;
};

class AvgPoolNdGradFunctor {
 public:
  AvgPoolNdGradFunctor() {
    for (int ndims = 1; ndims <= 3; ++ndims) {
      const auto& op_type_name = GetOpTypeName(ndims);
      op_expr_map_[op_type_name] =
          CHECK_JUST(one::OpBuilder(op_type_name).Input("x").Input("dy").Output("dx").Build());
    }
  }
  static std::string GetOpTypeName(const int32_t& ndims) {
    return "avg_pool_" + std::to_string(ndims) + "d_grad";
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& dy, const int32_t& ndims,
                           const std::string& data_format, const std::vector<int32_t>& padding,
                           const std::vector<int32_t>& kernel_size,
                           const std::vector<int32_t>& stride, const bool& ceil_mode,
                           const bool& count_include_pad, const int32_t& divisor_override) const {
    auto& attrs =
        THREAD_CACHED_MUTABLE_ATTR_MAP("data_format", "padding", "kernel_size", "stride",
                                       "ceil_mode", "count_include_pad", "divisor_override");
    attrs.SetAllAttrs(data_format, padding, kernel_size, stride, ceil_mode, count_include_pad,
                      divisor_override);
    const auto& op_type_name = GetOpTypeName(ndims);
    const auto& it = op_expr_map_.find(op_type_name);
    CHECK_OR_RETURN(it != op_expr_map_.end())
        << Error::RuntimeError() << "Encounter unsupported op " << op_type_name
        << " in AvgPoolNdGradFunctor.";
    CHECK_NOTNULL_OR_RETURN(it->second);  // NOLINT(maybe-need-error-msg)
    return OpInterpUtil::Dispatch<Tensor>(*it->second, {x, dy}, attrs);
  }

 protected:
  std::unordered_map<std::string, std::shared_ptr<OpExpr>> op_expr_map_;
};

class NormalizationGradFunctor {
 public:
  NormalizationGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("normalization_grad")
                         .Input("dy")
                         .Input("x")
                         .Input("mean")
                         .Input("inv_variance")
                         .Input("gamma")
                         .Output("dx")
                         .Output("gamma_diff")
                         .Output("beta_diff")
                         .Build());
  }
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& grad,
                                const std::shared_ptr<one::Tensor>& x,
                                const std::shared_ptr<one::Tensor>& mean,
                                const std::shared_ptr<one::Tensor>& inv_variance,
                                const std::shared_ptr<one::Tensor>& gamma, const float& epsilon,
                                const int32_t& axis) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("epsilon", "axis");
    attrs.SetAllAttrs(epsilon, axis);
    return OpInterpUtil::Dispatch<TensorTuple>(*op_, {grad, x, mean, inv_variance, gamma}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class NormalizationAddReluGradFunctor {
 public:
  NormalizationAddReluGradFunctor() {
    addend_op_ = CHECK_JUST(one::OpBuilder("normalization_add_relu_grad")
                                .Input("x")
                                .Input("dy")
                                .Input("mean")
                                .Input("inv_variance")
                                .Input("gamma")
                                .Input("beta")
                                .Input("reserve_space")
                                .Input("y")
                                .Output("dx")
                                .Output("gamma_diff")
                                .Output("beta_diff")
                                .Output("addend_diff")
                                .Build());
    no_addend_op_ = CHECK_JUST(one::OpBuilder("normalization_add_relu_grad")
                                   .Input("x")
                                   .Input("dy")
                                   .Input("mean")
                                   .Input("inv_variance")
                                   .Input("gamma")
                                   .Input("beta")
                                   .Input("reserve_space")
                                   .Input("y")
                                   .Output("dx")
                                   .Output("gamma_diff")
                                   .Output("beta_diff")
                                   .Build());
  }
  Maybe<TensorTuple> operator()(
      const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& grad,
      const std::shared_ptr<one::Tensor>& mean, const std::shared_ptr<one::Tensor>& inv_variance,
      const std::shared_ptr<one::Tensor>& gamma, const std::shared_ptr<one::Tensor>& beta,
      const std::shared_ptr<one::Tensor>& reserve_space, const std::shared_ptr<one::Tensor>& y,
      const int32_t& axis, const float& epsilon, bool has_addend) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("axis", "epsilon");
    attrs.SetAllAttrs(axis, epsilon);
    if (has_addend) {
      return OpInterpUtil::Dispatch<TensorTuple>(
          *addend_op_, {x, grad, mean, inv_variance, gamma, beta, reserve_space, y}, attrs);
    } else {
      return OpInterpUtil::Dispatch<TensorTuple>(
          *no_addend_op_, {x, grad, mean, inv_variance, gamma, beta, reserve_space, y}, attrs);
    }
  }

 private:
  std::shared_ptr<OpExpr> addend_op_;
  std::shared_ptr<OpExpr> no_addend_op_;
};

class LayerNormGradFunctor {
 public:
  LayerNormGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("layer_norm_grad")
                         .Input("dy")
                         .Input("x")
                         .Input("mean")
                         .Input("inv_variance")
                         .Output("dx")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& mean,
                           const std::shared_ptr<one::Tensor>& inv_variance,
                           const int64_t& begin_norm_axis, const double& epsilon) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("begin_norm_axis", "epsilon");
    attrs.SetAllAttrs(begin_norm_axis, epsilon);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, x, mean, inv_variance}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class LayerNormAffineGradFunctor {
 public:
  LayerNormAffineGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("layer_norm_grad")
                         .Input("dy")
                         .Input("x")
                         .Input("mean")
                         .Input("inv_variance")
                         .Input("gamma")
                         .Output("dx")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& mean,
                           const std::shared_ptr<one::Tensor>& inv_variance,
                           const std::shared_ptr<one::Tensor>& gamma,
                           const int64_t& begin_norm_axis, const double& epsilon) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("begin_norm_axis", "epsilon");
    attrs.SetAllAttrs(begin_norm_axis, epsilon);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, x, mean, inv_variance, gamma}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class LayerNormParamGradFunctor {
 public:
  LayerNormParamGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("layer_norm_param_grad")
                         .Input("dy")
                         .Input("x")
                         .Input("mean")
                         .Input("inv_variance")
                         .Output("gamma_diff")
                         .Output("beta_diff")
                         .Build());
  }
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& dy,
                                const std::shared_ptr<one::Tensor>& x,
                                const std::shared_ptr<one::Tensor>& mean,
                                const std::shared_ptr<one::Tensor>& inv_variance,
                                const int64_t& begin_params_axis) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("begin_params_axis");
    attrs.SetAllAttrs(begin_params_axis);
    return OpInterpUtil::Dispatch<TensorTuple>(*op_, {dy, x, mean, inv_variance}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class BroadcastMatmulGradBFunctor {
 public:
  BroadcastMatmulGradBFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("broadcast_matmul_grad_b").Input("a").Input("b").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& a,
                           const std::shared_ptr<one::Tensor>& b, double alpha) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("alpha");
    attrs.SetAllAttrs(alpha);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {a, b}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class FusedScaleTrilSoftmaxMaskScaleGradFunctor {
 public:
  FusedScaleTrilSoftmaxMaskScaleGradFunctor() {
    fused_op_ = CHECK_JUST(one::OpBuilder("fused_tril_scale_softmax_mask_scale_grad")
                               .Input("softmax_y")
                               .Input("dy")
                               .Input("mask")
                               .Output("dx")
                               .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& softmax_y,
                           const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& mask, const int64_t diagonal,
                           const float tril_scale_value, const float mask_scale_value) const {
    auto& fused_attrs =
        THREAD_CACHED_MUTABLE_ATTR_MAP("diagonal", "tril_scale_value", "mask_scale_value");
    fused_attrs.SetAllAttrs(diagonal, tril_scale_value, mask_scale_value);
    return OpInterpUtil::Dispatch<Tensor>(*fused_op_, {softmax_y, dy, mask}, fused_attrs);
  }

 private:
  std::shared_ptr<OpExpr> fused_op_;
};

class FusedScaleMaskSoftmaxGradFunctor {
 public:
  FusedScaleMaskSoftmaxGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("fused_scale_mask_softmax_grad")
                         .Input("y")
                         .Input("dy")
                         .Input("mask")
                         .Output("dx")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& y,
                           const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& mask, const float& scale) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("scale_value");
    attrs.SetAllAttrs(scale);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {y, dy, mask}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class FusedScaleMaskSoftmaxDropoutGradFunctor {
 public:
  FusedScaleMaskSoftmaxDropoutGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("fused_scale_mask_softmax_dropout_grad")
                         .Input("softmax_y")
                         .Input("dy")
                         .Input("mask")
                         .Input("dropout_mask")
                         .Output("dx")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& softmax_y,
                           const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& mask,
                           const std::shared_ptr<one::Tensor>& dropout_mask, const float& scale,
                           const float& dropout_scale) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("scale_value", "dropout_scale_value");
    attrs.SetAllAttrs(scale, dropout_scale);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {softmax_y, dy, mask, dropout_mask}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class CublasBiasAddReluMatmulGradFunctor {
 public:
  CublasBiasAddReluMatmulGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("cublas_bias_add_relu_matmul_grad")
                         .Input("dy")
                         .Input("weight")
                         .Input("aux")
                         .Output("d_grad")
                         .Output("d_bias")
                         .Build());
  }
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& dy,
                                const std::shared_ptr<one::Tensor>& weight,
                                const std::shared_ptr<one::Tensor>& aux,
                                const double& alpha) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("alpha");
    attrs.SetAllAttrs(alpha);
    return OpInterpUtil::Dispatch<TensorTuple>(*op_, {dy, weight, aux}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class CublasMatmulBiasAddGradFunctor {
 public:
  CublasMatmulBiasAddGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("cublas_matmul_bias_add_grad")
                         .Input("dy")
                         .Input("x")
                         .Output("w_grad")
                         .Output("b_grad")
                         .Build());
  }
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& dy,
                                const std::shared_ptr<one::Tensor>& x) const {
    return OpInterpUtil::Dispatch<TensorTuple>(*op_, {dy, x});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class FusedReluDropoutGradFunctor {
 public:
  FusedReluDropoutGradFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("fused_relu_dropout_grad").Input("dy").Input("mask").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& mask, const float& scale) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("scale");
    attrs.SetAllAttrs(scale);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, mask}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class FusedDotFeatureInteractionGradFunctor {
 public:
  FusedDotFeatureInteractionGradFunctor() {
    ops_has_output_concat_grad_.resize(kMaxInputCount);
    ops_no_output_concat_grad_.resize(kMaxInputCount);
    for (int n = 0; n < ops_has_output_concat_grad_.size(); ++n) {
      ops_has_output_concat_grad_[n] =
          CHECK_JUST(one::OpBuilder("fused_dot_feature_interaction_grad")
                         .Input("dy")
                         .Input("features", n + 1)
                         .Output("features_grad", n + 1)
                         .Output("output_concat_grad")
                         .Build());
    }
    for (int n = 0; n < ops_no_output_concat_grad_.size(); ++n) {
      ops_no_output_concat_grad_[n] =
          CHECK_JUST(one::OpBuilder("fused_dot_feature_interaction_grad")
                         .Input("dy")
                         .Input("features", n + 1)
                         .Output("features_grad", n + 1)
                         .Build());
    }
  }

  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& dy, const TensorTuple& features,
                                const bool& has_output_concat, const bool& self_interaction,
                                const int32_t& output_concat_grad_dim,
                                const std::string& pooling) const {
    auto& attrs =
        THREAD_CACHED_MUTABLE_ATTR_MAP("self_interaction", "output_concat_grad_dim", "pooling");
    attrs.SetAllAttrs(self_interaction, output_concat_grad_dim, pooling);
    CHECK_OR_RETURN(pooling == "sum" || pooling == "none")
        << Error::RuntimeError() << "pooling should be sum or none, but get " << pooling << ". ";
    const int64_t n_features_grad = features.size();
    CHECK_LE_OR_RETURN(n_features_grad, kMaxInputCount)
        << Error::RuntimeError() << "The number of tensors in features should be less than 128.";
    TensorTuple inputs(n_features_grad + 1);
    inputs[0] = dy;
    for (int32_t i = 0; i < n_features_grad; ++i) { inputs[i + 1] = features[i]; }
    if (has_output_concat) {
      return OpInterpUtil::Dispatch<TensorTuple>(
          *JUST(oneflow::VectorAt(ops_has_output_concat_grad_, n_features_grad - 1)), inputs,
          attrs);
    } else {
      return OpInterpUtil::Dispatch<TensorTuple>(
          *JUST(oneflow::VectorAt(ops_no_output_concat_grad_, n_features_grad - 1)), inputs, attrs);
    }
  }

 private:
  std::vector<std::shared_ptr<OpExpr>> ops_has_output_concat_grad_;
  std::vector<std::shared_ptr<OpExpr>> ops_no_output_concat_grad_;
};

class FusedCrossFeatureInteractionV1GradFunctor {
 public:
  FusedCrossFeatureInteractionV1GradFunctor() {
    v1_grad_op_ = CHECK_JUST(one::OpBuilder("fused_cross_feature_interaction_v1_grad")
                                 .Input("dy")
                                 .Input("weight")
                                 .Input("x")
                                 .Input("x0")
                                 .Input("matmul_result")
                                 .Output("dx")
                                 .Output("dw")
                                 .Output("dx0")
                                 .Output("dbias")
                                 .Build());
  }

  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& dy,
                                const std::shared_ptr<one::Tensor>& weight,
                                const std::shared_ptr<one::Tensor>& x,
                                const std::shared_ptr<one::Tensor>& x0,
                                const std::shared_ptr<one::Tensor>& matmul_result) const {
    return OpInterpUtil::Dispatch<TensorTuple>(*v1_grad_op_, {dy, weight, x, x0, matmul_result});
  }

 private:
  std::shared_ptr<OpExpr> v1_grad_op_;
};

class FusedCrossFeatureInteractionV2GradFunctor {
 public:
  FusedCrossFeatureInteractionV2GradFunctor() {
    v2_grad_op_ = CHECK_JUST(one::OpBuilder("fused_cross_feature_interaction_v2_grad")
                                 .Input("dy")
                                 .Input("weight")
                                 .Input("bias")
                                 .Input("x")
                                 .Input("x0")
                                 .Input("matmul_result")
                                 .Output("dx")
                                 .Output("dw")
                                 .Output("dx0")
                                 .Output("dbias")
                                 .Build());
  }

  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& dy,
                                const std::shared_ptr<one::Tensor>& weight,
                                const std::shared_ptr<one::Tensor>& bias,
                                const std::shared_ptr<one::Tensor>& x,
                                const std::shared_ptr<one::Tensor>& x0,
                                const std::shared_ptr<one::Tensor>& matmul_result) const {
    return OpInterpUtil::Dispatch<TensorTuple>(*v2_grad_op_,
                                               {dy, weight, bias, x, x0, matmul_result});
  }

 private:
  std::shared_ptr<OpExpr> v2_grad_op_;
};

class MatrixVectorProductGradAFunctor {
 public:
  MatrixVectorProductGradAFunctor() {
    matrix_vector_product_grad_a_op_ = CHECK_JUST(
        one::OpBuilder("matrix_vector_product_grad_a").Input("dy").Input("b").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& b) const {
    return OpInterpUtil::Dispatch<Tensor>(*matrix_vector_product_grad_a_op_, {dy, b});
  }

 private:
  std::shared_ptr<OpExpr> matrix_vector_product_grad_a_op_;
};

class MatrixVectorProductGradBFunctor {
 public:
  MatrixVectorProductGradBFunctor() {
    matrix_vector_product_grad_b_op_ = CHECK_JUST(
        one::OpBuilder("matrix_vector_product_grad_b").Input("dy").Input("a").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& a) const {
    return OpInterpUtil::Dispatch<Tensor>(*matrix_vector_product_grad_b_op_, {dy, a});
  }

 private:
  std::shared_ptr<OpExpr> matrix_vector_product_grad_b_op_;
};

class VectorMatrixProductGradAFunctor {
 public:
  VectorMatrixProductGradAFunctor() {
    vector_matrix_product_grad_a_op_ = CHECK_JUST(
        one::OpBuilder("vector_matrix_product_grad_a").Input("dy").Input("b").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& b) const {
    return OpInterpUtil::Dispatch<Tensor>(*vector_matrix_product_grad_a_op_, {dy, b});
  }

 private:
  std::shared_ptr<OpExpr> vector_matrix_product_grad_a_op_;
};

class VectorMatrixProductGradBFunctor {
 public:
  VectorMatrixProductGradBFunctor() {
    vector_matrix_product_grad_b_op_ = CHECK_JUST(
        one::OpBuilder("vector_matrix_product_grad_b").Input("dy").Input("a").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& a) const {
    return OpInterpUtil::Dispatch<Tensor>(*vector_matrix_product_grad_b_op_, {dy, a});
  }

 private:
  std::shared_ptr<OpExpr> vector_matrix_product_grad_b_op_;
};

class FusedMLPGradFunctor {
 public:
  FusedMLPGradFunctor() {
#if CUDA_VERSION >= 11060
    fused_op_.resize(kMaxInputCount /*the maximum number of layers*/);
    for (int n = 1; n < fused_op_.size(); ++n) {
      fused_op_[n] = CHECK_JUST(one::OpBuilder("cublas_fused_mlp_grad")
                                    .Input("dy")
                                    .Input("x")
                                    .Input("weights", n)
                                    .Input("cublas_aux", n)
                                    .Input("hidden", n)
                                    .Output("d_x")
                                    .Output("d_biases", n)
                                    .Output("d_weights", n)
                                    .Build());
    }
#endif
  }
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& dy,
                                const std::shared_ptr<one::Tensor>& x, const TensorTuple& weights,
                                const TensorTuple& cublas_aux, const TensorTuple& hidden,
                                const std::vector<float>& alpha_list) const {
    const int64_t weight_size = weights.size();
    CHECK_EQ_OR_RETURN(alpha_list.size(), weight_size - 1)
        << "Alpha list size should be equal to weight_size - 1. ";
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("alpha_list");
    attrs.SetAllAttrs(alpha_list);
    TensorTuple input(2 + 3 * weight_size);
    input[0] = dy;
    input[1] = x;
    std::copy(weights.begin(), weights.end(), input.begin() + 2);
    std::copy(cublas_aux.begin(), cublas_aux.end(), input.begin() + 2 + weight_size);
    std::copy(hidden.begin(), hidden.end(), input.begin() + 2 + 2 * weight_size);
#if CUDA_VERSION >= 11060
    return OpInterpUtil::Dispatch<TensorTuple>(*fused_op_[weight_size], input, attrs);
#endif
    UNIMPLEMENTED_THEN_RETURN() << "Only Support in CUDA_VERSION >= 11060";
  }

 private:
#if CUDA_VERSION >= 11060
  std::vector<std::shared_ptr<OpExpr>> fused_op_;
#endif
};

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::ConvBiasGradFunctor>("ConvBiasGrad");
  m.add_functor<impl::ConvFilterGradFunctor>("ConvFilterGrad");
  m.add_functor<impl::ConvDataGradFunctor>("ConvDataGrad");
  m.add_functor<impl::EmbeddingGradFunctor>("EmbeddingGrad");
  m.add_functor<impl::TFPoolNdGradFunctor>("TFPoolNdGrad");
  m.add_functor<impl::AdaptivePoolNdGradFunctor>("AdaptivePoolNdGrad");
  m.add_functor<impl::KLDivLossGradFunctor>("KLDivLossGrad");
  m.add_functor<impl::KLDivLossTargetGradFunctor>("KLDivLossTargetGrad");
  m.add_functor<impl::NLLGradFunctor>("NLLGrad");
  m.add_functor<impl::BinaryCrossEntropyLossGradFunctor>("BinaryCrossEntropyLossGrad");
  m.add_functor<impl::BinaryCrossEntropyLossTargetGradFunctor>("BinaryCrossEntropyLossTargetGrad");
  m.add_functor<impl::BinaryCrossEntropyWithLogitsLossGradFunctor>(
      "BinaryCrossEntropyWithLogitsLossGrad");
  m.add_functor<impl::BinaryCrossEntropyWithLogitsLossTargetGradFunctor>(
      "BinaryCrossEntropyWithLogitsLossTargetGrad");
  m.add_functor<impl::SparseCrossEntropyGradFunctor>("SparseCrossEntropyGrad");
  m.add_functor<impl::SparseCrossEntropyMsGradFunctor>("SparseCrossEntropyMsGrad");
  m.add_functor<impl::SparseSoftmaxCrossEntropyGrad>("SparseSoftmaxCrossEntropyGrad");
  m.add_functor<impl::SparseSoftmaxCrossEntropyMsGrad>("SparseSoftmaxCrossEntropyMsGrad");
  m.add_functor<impl::SmoothL1LossGradFunctor>("SmoothL1LossGrad");
  m.add_functor<impl::CombinedMarginLossGradFunctor>("CombinedMarginLossGrad");
  m.add_functor<impl::AffineGridGradFunctor>("AffineGridGrad");
  m.add_functor<impl::GridSampleGradFunctor>("GridSampleGrad");
  m.add_functor<impl::MaxPoolNdGradFunctor>("MaxPoolNdGrad");
  m.add_functor<impl::PadGradFunctor>("PadGrad");
  m.add_functor<impl::AvgPoolNdGradFunctor>("AvgPoolNdGrad");
  m.add_functor<impl::NormalizationGradFunctor>("NormalizationGrad");
  m.add_functor<impl::NormalizationAddReluGradFunctor>("NormalizationAddReluGrad");
  m.add_functor<impl::LayerNormGradFunctor>("LayerNormGrad");
  m.add_functor<impl::LayerNormAffineGradFunctor>("LayerNormAffineGrad");
  m.add_functor<impl::LayerNormParamGradFunctor>("LayerNormParamGrad");
  m.add_functor<impl::BroadcastMatmulGradBFunctor>("BroadcastMatmulGradB");
  m.add_functor<impl::CtcLossGradFunctor>("CtcLossGrad");
  m.add_functor<impl::FusedScaleTrilSoftmaxMaskScaleGradFunctor>(
      "FusedScaleTrilSoftmaxMaskScaleGrad");
  m.add_functor<impl::FusedScaleMaskSoftmaxGradFunctor>("FusedScaleMaskSoftmaxGrad");
  m.add_functor<impl::FusedScaleMaskSoftmaxDropoutGradFunctor>("FusedScaleMaskSoftmaxDropoutGrad");
  m.add_functor<impl::CublasBiasAddReluMatmulGradFunctor>("CublasBiasAddReluMatmulGrad");
  m.add_functor<impl::CublasMatmulBiasAddGradFunctor>("CublasMatmulBiasAddGrad");
  m.add_functor<impl::FusedReluDropoutGradFunctor>("FusedReluDropoutGrad");
  m.add_functor<impl::FusedDotFeatureInteractionGradFunctor>("FusedDotFeatureInteractionGrad");
  m.add_functor<impl::FusedCrossFeatureInteractionV1GradFunctor>(
      "FusedCrossFeatureInteractionV1Grad");
  m.add_functor<impl::FusedCrossFeatureInteractionV2GradFunctor>(
      "FusedCrossFeatureInteractionV2Grad");
  m.add_functor<impl::FusedMLPGradFunctor>("FusedMLPGrad");
  m.add_functor<impl::BinaryCrossEntropyWithLogitsReduceMeanLossGradFunctor>(
      "BinaryCrossEntropyWithLogitsReduceMeanLossGrad");
  m.add_functor<impl::BinaryCrossEntropyWithLogitsReduceMeanLossTargetGradFunctor>(
      "BinaryCrossEntropyWithLogitsReduceMeanLossTargetGrad");
  m.add_functor<impl::MatrixVectorProductGradAFunctor>("MatrixVectorProductGradA");
  m.add_functor<impl::MatrixVectorProductGradBFunctor>("MatrixVectorProductGradB");
  m.add_functor<impl::VectorMatrixProductGradAFunctor>("VectorMatrixProductGradA");
  m.add_functor<impl::VectorMatrixProductGradBFunctor>("VectorMatrixProductGradB");
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
