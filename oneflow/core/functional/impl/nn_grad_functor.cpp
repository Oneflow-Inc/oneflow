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
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/functional/function_library.h"
#include "oneflow/core/functional/impl/common.h"
#include "oneflow/core/functional/impl/unary_functor.h"

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
    auto ctx = std::make_shared<ConvBiasGradOpInterpCtx>();
    ctx->num_spatial_dims = num_spatial_dims;
    ctx->data_format = data_format;
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy}, ctx);
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
    auto ctx = std::make_shared<ConvFilterGradOpInterpCtx>();
    ctx->num_spatial_dims = num_spatial_dims;
    ctx->kernel_size = kernel_size;
    ctx->strides = strides;
    ctx->padding_before = padding_before;
    ctx->dilation_rate = dilation_rate;
    ctx->groups = groups;
    ctx->data_format = data_format;
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, x}, ctx);
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
    auto ctx = std::make_shared<ConvDataGradOpInterpCtx>();
    ctx->num_spatial_dims = num_spatial_dims;
    ctx->kernel_size = kernel_size;
    ctx->strides = strides;
    ctx->padding_before = padding_before;
    ctx->dilation_rate = dilation_rate;
    ctx->groups = groups;
    ctx->data_format = data_format;
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, weight, x}, ctx);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class PoolingNdGradFunctor {
 public:
  PoolingNdGradFunctor() {
    for (const auto& mode : {"max"}) {
      for (int ndims = 1; ndims <= 3; ++ndims) {
        const auto& op_type_name = GetOpTypeName(mode, ndims);
        op_expr_map_[op_type_name] = CHECK_JUST(one::OpBuilder(op_type_name)
                                                    .Input("x")
                                                    .Input("y")
                                                    .Input("indice")
                                                    .Input("dy")
                                                    .Output("dx")
                                                    .Build());
      }
    }
  }
  static std::string GetOpTypeName(const std::string& mode, const int32_t& ndims) {
    return mode + "pool_" + std::to_string(ndims) + "d_grad";
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& y,
                           const std::shared_ptr<one::Tensor>& indice,
                           const std::shared_ptr<one::Tensor>& dy, const std::string& mode,
                           const int32_t& ndims, const std::string& data_format,
                           const std::vector<int32_t>& padding,
                           const std::vector<int32_t>& kernel_size,
                           const std::vector<int32_t>& stride, const std::vector<int32_t>& dilation,
                           const bool& return_indices, const bool& ceil_mode) const {
    auto ctx = std::make_shared<PoolingNdGradOpInterpCtx>();
    ctx->data_format = data_format;
    ctx->padding = padding;
    ctx->kernel_size = kernel_size;
    ctx->stride = stride;
    ctx->dilation = dilation;
    ctx->return_indices = return_indices;
    ctx->ceil_mode = ceil_mode;
    const auto& op_type_name = GetOpTypeName(mode, ndims);
    const auto& it = op_expr_map_.find(op_type_name);
    CHECK_OR_RETURN(it != op_expr_map_.end())
        << "Encounter unsupported op " << op_type_name << " in PoolingNdGradFunctor.";
    CHECK_NOTNULL_OR_RETURN(it->second);
    return OpInterpUtil::Dispatch<Tensor>(*it->second, {x, y, indice, dy}, ctx);
  }

 protected:
  std::unordered_map<std::string, std::shared_ptr<OpExpr>> op_expr_map_;
};

class PoolNdGradFunctor {
 public:
  PoolNdGradFunctor() {
    for (const auto& mode : {"max", "avg"}) {
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
    auto ctx = std::make_shared<PoolNdGradOpInterpCtx>();
    ctx->data_format = data_format;
    ctx->padding = padding;
    ctx->padding_before = padding_before;
    ctx->padding_after = padding_after;
    ctx->pool_size = pool_size;
    ctx->strides = strides;
    ctx->ceil_mode = ceil_mode;
    const auto& op_type_name = GetOpTypeName(mode, ndims);
    const auto& it = op_expr_map_.find(op_type_name);
    CHECK_OR_RETURN(it != op_expr_map_.end())
        << "Encounter unsupported op " << op_type_name << " in PoolNdGradFunctor.";
    CHECK_NOTNULL_OR_RETURN(it->second);
    return OpInterpUtil::Dispatch<Tensor>(*it->second, {x, y, dy}, ctx);
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
        << "Encounter unsupported op " << op_type_name << " in AdaptivePoolNdGradFunctor.";
    CHECK_NOTNULL_OR_RETURN(it->second);
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
    auto ctx = std::make_shared<SparseCrossEntropyGradOpInterpCtx>();
    ctx->depth = depth;
    return OpInterpUtil::Dispatch<Tensor>(*op_, {prediction, label, dy}, ctx);
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
    auto ctx = std::make_shared<SparseCrossEntropyMsGradOpInterpCtx>();
    ctx->depth = depth;
    return OpInterpUtil::Dispatch<Tensor>(*op_, {prediction, label, dy}, ctx);
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
    auto ctx = std::make_shared<SparseSoftmaxCrossEntropyGradOpInterpCtx>();
    ctx->depth = depth;
    return OpInterpUtil::Dispatch<Tensor>(*op_, {prob, label, dy}, ctx);
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
                           const std::shared_ptr<one::Tensor>& target, const float& beta,
                           const std::string& reduction) const {
    auto ctx = std::make_shared<SmoothL1LossGradOpInterpCtx>();
    ctx->beta = beta;
    ctx->reduction = reduction;
    return OpInterpUtil::Dispatch<one::Tensor>(*op_, {dy, input, target}, ctx);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class KLDivLossGradFunctor {
 public:
  KLDivLossGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("kl_div_loss_grad")
                         .Input("input")
                         .Input("target")
                         .Input("dy")
                         .Output("dx")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& target, const bool log_target,
                           const std::string& reduction) const {
    auto ctx = std::make_shared<KLDivLossGradOpInterpCtx>();
    ctx->log_target = log_target;
    ctx->reduction = reduction;
    return OpInterpUtil::Dispatch<Tensor>(*op_, {input, target, dy}, ctx);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class NllLossGradFunctor {
 public:
  NllLossGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("nll_grad")
                         .Input("input")
                         .Input("target")
                         .Input("total_weight")
                         .Input("dy")
                         .Output("dx")
                         .Build());
    op_weight_ = CHECK_JUST(one::OpBuilder("nll_grad")
                                .Input("input")
                                .Input("target")
                                .Input("total_weight")
                                .Input("weight")
                                .Input("dy")
                                .Output("dx")
                                .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& target,
                           const Optional<one::Tensor>& weight,
                           const std::shared_ptr<one::Tensor>& total_weight,
                           const int64_t ignore_index, const std::string& reduction) const {
    auto ctx = std::make_shared<NllGradOpInterpCtx>();
    ctx->ignore_index = ignore_index;
    ctx->reduction = reduction;
    if (weight) {
      return OpInterpUtil::Dispatch<one::Tensor>(
          *op_weight_, {input, target, total_weight, JUST(weight), dy}, ctx);
    } else {
      return OpInterpUtil::Dispatch<one::Tensor>(*op_, {input, target, total_weight, dy}, ctx);
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
                         .Input("input")
                         .Input("target")
                         .Input("dy")
                         .Output("dx")
                         .Build());
    op_weight_ = CHECK_JUST(one::OpBuilder("binary_cross_entropy_grad")
                                .Input("input")
                                .Input("target")
                                .Input("weight")
                                .Input("dy")
                                .Output("dx")
                                .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& target,
                           const Optional<one::Tensor>& weight,
                           const std::string& reduction) const {
    auto ctx = std::make_shared<BinaryCrossEntropyLossGradOpInterpCtx>();
    ctx->reduction = reduction;
    if (weight) {
      return OpInterpUtil::Dispatch<one::Tensor>(*op_weight_, {input, target, JUST(weight), dy},
                                                 ctx);
    } else {
      return OpInterpUtil::Dispatch<one::Tensor>(*op_, {input, target, dy}, ctx);
    }
  }

 private:
  std::shared_ptr<OpExpr> op_;
  std::shared_ptr<OpExpr> op_weight_;
};

class BinaryCrossEntropyWithLogitsLossGradFunctor {
 public:
  BinaryCrossEntropyWithLogitsLossGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("binary_cross_entropy_with_logits_grad")
                         .Input("input")
                         .Input("target")
                         .Input("dy")
                         .Output("dx")
                         .Build());
    op_weight_ = CHECK_JUST(one::OpBuilder("binary_cross_entropy_with_logits_grad")
                                .Input("input")
                                .Input("target")
                                .Input("weight")
                                .Input("dy")
                                .Output("dx")
                                .Build());
    op_pos_ = CHECK_JUST(one::OpBuilder("binary_cross_entropy_with_logits_grad")
                             .Input("input")
                             .Input("target")
                             .Input("pos_weight")
                             .Input("dy")
                             .Output("dx")
                             .Build());
    op_weight_pos_ = CHECK_JUST(one::OpBuilder("binary_cross_entropy_with_logits_grad")
                                    .Input("input")
                                    .Input("target")
                                    .Input("weight")
                                    .Input("pos_weight")
                                    .Input("dy")
                                    .Output("dx")
                                    .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& target,
                           const Optional<one::Tensor>& weight,
                           const Optional<one::Tensor>& pos_weight,
                           const std::string& reduction) const {
    auto ctx = std::make_shared<BinaryCrossEntropyWithLogitsLossGradOpInterpCtx>();
    ctx->reduction = reduction;
    ctx->has_pos_weight = pos_weight.has_value();
    if (weight) {
      if (pos_weight) {
        return OpInterpUtil::Dispatch<one::Tensor>(
            *op_weight_pos_, {input, target, JUST(weight), JUST(pos_weight), dy}, ctx);
      } else {
        return OpInterpUtil::Dispatch<one::Tensor>(*op_weight_, {input, target, JUST(weight), dy},
                                                   ctx);
      }
    } else {
      if (pos_weight) {
        return OpInterpUtil::Dispatch<one::Tensor>(*op_pos_, {input, target, JUST(pos_weight), dy},
                                                   ctx);
      } else {
        return OpInterpUtil::Dispatch<one::Tensor>(*op_, {input, target, dy}, ctx);
      }
    }
  }

 private:
  std::shared_ptr<OpExpr> op_;
  std::shared_ptr<OpExpr> op_weight_;
  std::shared_ptr<OpExpr> op_pos_;
  std::shared_ptr<OpExpr> op_weight_pos_;
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
    auto ctx = std::make_shared<CombinedMarginLossGradOpInterpCtx>();
    ctx->m1 = m1;
    ctx->m2 = m2;
    ctx->m3 = m3;
    ctx->depth = depth;
    return OpInterpUtil::Dispatch<one::Tensor>(*op_, {dy, label, theta}, ctx);
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
    auto ctx = std::make_shared<AffineGridGradOpInterpCtx>();
    ctx->size = size;
    ctx->align_corners = align_corners;
    return OpInterpUtil::Dispatch<one::Tensor>(*op_, {dgrid}, ctx);
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
    auto ctx = std::make_shared<GridSampleGradOpInterpCtx>();
    ctx->interpolation_mode = interpolation_mode;
    ctx->padding_mode = padding_mode;
    ctx->align_corners = align_corners;
    return OpInterpUtil::Dispatch<one::TensorTuple>(*op_, {doutput, input, grid}, ctx);
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
    auto ctx = std::make_shared<CtcLossGradOpInterpCtx>();
    ctx->blank = blank;
    ctx->zero_infinity = zero_infinity;
    ctx->max_target_length = max_target_length;
    return OpInterpUtil::Dispatch<one::Tensor>(
        *op_, {grad_out, log_probs, targets, input_lengths, target_lengths, loss, alpha}, ctx);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class PadGradFunctor {
 public:
  PadGradFunctor() {
    pad_grad_ = CHECK_JUST(one::OpBuilder("pad_grad").Input("dy").Output("dx").Build());
    reflect_pad_grad_ =
        CHECK_JUST(one::OpBuilder("reflection_pad2d_grad").Input("dy").Output("dx").Build());
    replicate_pad_grad_ =
        CHECK_JUST(one::OpBuilder("replication_pad2d_grad").Input("dy").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy, const std::vector<int64_t>& pad,
                           const std::string& mode, const Scalar& value) const {
    const int64_t ndim = dy->shape()->NumAxes();
    size_t padding_size = 2 * ndim;
    CHECK_LE_OR_RETURN(pad.size(), padding_size)
        << "Pad size should less than or equal to input axes * 2.";

    autp ctx = std::make_shared<PadGradOpInterpCtx>();
    ctx->padding = pad;
    if (mode == "constant") {
      std::vector<int64_t> pad_before(ndim, 0);
      std::vector<int64_t> pad_after(ndim, 0);
      const int64_t pad_pair = pad.size() / 2;
      for (int64_t i = 0; i < pad_pair; ++i) {
        pad_before[ndim - i - 1] = pad[2 * i];
        pad_after[ndim - i - 1] = pad[2 * i + 1];
      }
      ctx->padding_before = pad_before;
      ctx->padding_after = pad_after;

      if (IsFloatingDataType(dy->dtype()->data_type())) {
        ctx->floating_constant_value = JUST(value.As<double>());
        ctx->integral_constant_value = 0;
      } else if (IsIntegralDataType(dy->dtype()->data_type())) {
        ctx->floating_constant_value = 0;
        ctx->integral_constant_value = JUST(value.As<int64_t>());
      }
      return OpInterpUtil::Dispatch<Tensor>(*pad_grad_, {dy}, ctx);
    } else if (mode == "reflect") {
      return OpInterpUtil::Dispatch<Tensor>(*reflect_pad_grad_, {dy}, ctx);
    } else if (mode == "replicate") {
      return OpInterpUtil::Dispatch<Tensor>(*replicate_pad_grad_, {dy}, ctx);
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "Pad mode is " << mode
                                  << ", but only constant, reflect and replicate are valid.";
    }
  }

 private:
  std::shared_ptr<OpExpr> pad_grad_;
  std::shared_ptr<OpExpr> reflect_pad_grad_;
  std::shared_ptr<OpExpr> replicate_pad_grad_;
};

class AvgPoolingNdGradFunctor {
 public:
  AvgPoolingNdGradFunctor() {
    for (int ndims = 1; ndims <= 3; ++ndims) {
      const auto& op_type_name = GetOpTypeName(ndims);
      op_expr_map_[op_type_name] = CHECK_JUST(
          one::OpBuilder(op_type_name).Input("x").Input("y").Input("dy").Output("dx").Build());
    }
  }
  static std::string GetOpTypeName(const int32_t& ndims) {
    return "avgpool_" + std::to_string(ndims) + "d_grad";
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& y,
                           const std::shared_ptr<one::Tensor>& dy, const int32_t& ndims,
                           const std::string& data_format, const std::vector<int32_t>& padding,
                           const std::vector<int32_t>& kernel_size,
                           const std::vector<int32_t>& stride, const bool& ceil_mode,
                           const bool& count_include_pad, const int64_t& divisor_override) const {
    auto ctx = std::make_shared<AvgPoolingNdGradOpInterpCtx>();
    ctx->data_format = data_format;
    ctx->padding = padding;
    ctx->kernel_size = kernel_size;
    ctx->stride = stride;
    ctx->ceil_mode = ceil_mode;
    ctx->count_include_pad = count_include_pad;
    ctx->divisor_override = divisor_override;
    const auto& op_type_name = GetOpTypeName(ndims);
    const auto& it = op_expr_map_.find(op_type_name);
    CHECK_OR_RETURN(it != op_expr_map_.end())
        << "Encounter unsupported op " << op_type_name << " in PoolingNdGradFunctor.";
    CHECK_NOTNULL_OR_RETURN(it->second);
    return OpInterpUtil::Dispatch<Tensor>(*it->second, {x, y, dy}, ctx);
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
    auto ctx = std::make_shared<NormalizationGradOpInterpCtx>();
    ctx->epsilon = epsilon;
    ctx->axis = axis;
    return OpInterpUtil::Dispatch<TensorTuple>(*op_, {grad, x, mean, inv_variance, gamma}, ctx);
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
  }
  Maybe<TensorTuple> operator()(
      const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& grad,
      const std::shared_ptr<one::Tensor>& mean, const std::shared_ptr<one::Tensor>& inv_variance,
      const std::shared_ptr<one::Tensor>& gamma, const std::shared_ptr<one::Tensor>& beta,
      const std::shared_ptr<one::Tensor>& reserve_space, const std::shared_ptr<one::Tensor>& y,
      const int32_t& axis, const float& epsilon) const {
    auto ctx = std::make_shared<NormalizationAddReluGradOpInterpCtx>();
    ctx->axis = axis;
    ctx->epsilon = epsilon;
    return OpInterpUtil::Dispatch<TensorTuple>(
        *addend_op_, {x, grad, mean, inv_variance, gamma, beta, reserve_space, y}, ctx);
  }

 private:
  std::shared_ptr<OpExpr> addend_op_;
};

class LayerNormGradFunctor {
 public:
  LayerNormGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("layer_norm_grad")
                         .Input("x")
                         .Input("mean")
                         .Input("inv_variance")
                         .Input("dy")
                         .Output("dx")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& mean,
                           const std::shared_ptr<one::Tensor>& inv_variance,
                           const std::shared_ptr<one::Tensor>& dy, const int64_t& begin_norm_axis,
                           const double& epsilon) const {
    auto ctx = std::make_shared<LayerNormGradOpInterpCtx>();
    ctx->begin_norm_axis = begin_norm_axis;
    ctx->epsilon = epsilon;
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x, mean, inv_variance, dy}, ctx);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class LayerNormParamGradFunctor {
 public:
  LayerNormParamGradFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("layer_norm_param_grad").Input("dy").Output("normalized_diff").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy, const int64_t& begin_params_axis,
                           const double& epsilon) const {
    auto ctx = std::make_shared<LayerNormParamGradOpInterpCtx>();
    ctx->begin_norm_axis = begin_norm_axis;
    ctx->epsilon = epsilon;
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy}, ctx);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class LayerNormAffineParamGradFunctor {
 public:
  LayerNormAffineParamGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("layer_norm_param_grad")
                         .Input("dy")
                         .Input("gamma")
                         .Input("normalized")
                         .Output("gamma_diff")
                         .Output("beta_diff")
                         .Output("normalized_diff")
                         .Output("reduce_buf")
                         .Build());
  }
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& dy,
                                const std::shared_ptr<one::Tensor>& gamma,
                                const std::shared_ptr<one::Tensor>& normalized,
                                const int64_t& begin_params_axis, const double& epsilon) const {
    auto ctx = std::make_shared<LayerNormParamGradOpInterpCtx>();
    ctx->begin_norm_axis = begin_norm_axis;
    ctx->epsilon = epsilon;
    return OpInterpUtil::Dispatch<TensorTuple>(*op_, {dy, gamma, normalized}, ctx);
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
    auto ctx = std::make_shared<BroadcastMatmulGradBOpInterpCtx>();
    ctx->alpha = alpha;
    return OpInterpUtil::Dispatch<Tensor>(*op_, {a, b}, ctx);
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
    auto ctx = std::make_shared<FusedScaleTrilSoftmaxMaskScaleGradOpInterpCtx>();
    ctx->diagonal = diagonal;
    ctx->tril_scale_value = tril_scale_value;
    ctx->mask_scale_value = mask_scale_value;
    return OpInterpUtil::Dispatch<Tensor>(*fused_op_, {softmax_y, dy, mask}, ctx);
  }

 private:
  std::shared_ptr<OpExpr> fused_op_;
};

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::ConvBiasGradFunctor>("ConvBiasGrad");
  m.add_functor<impl::ConvFilterGradFunctor>("ConvFilterGrad");
  m.add_functor<impl::ConvDataGradFunctor>("ConvDataGrad");
  m.add_functor<impl::PoolNdGradFunctor>("PoolNdGrad");
  m.add_functor<impl::AdaptivePoolNdGradFunctor>("AdaptivePoolNdGrad");
  m.add_functor<impl::KLDivLossGradFunctor>("KLDivLossGrad");
  m.add_functor<impl::NllLossGradFunctor>("NllLossGrad");
  m.add_functor<impl::BinaryCrossEntropyLossGradFunctor>("BinaryCrossEntropyLossGrad");
  m.add_functor<impl::BinaryCrossEntropyWithLogitsLossGradFunctor>(
      "BinaryCrossEntropyWithLogitsLossGrad");
  m.add_functor<impl::SparseCrossEntropyGradFunctor>("SparseCrossEntropyGrad");
  m.add_functor<impl::SparseCrossEntropyMsGradFunctor>("SparseCrossEntropyMsGrad");
  m.add_functor<impl::SparseSoftmaxCrossEntropyGrad>("SparseSoftmaxCrossEntropyGrad");
  m.add_functor<impl::SmoothL1LossGradFunctor>("SmoothL1LossGrad");
  m.add_functor<impl::CombinedMarginLossGradFunctor>("CombinedMarginLossGrad");
  m.add_functor<impl::AffineGridGradFunctor>("AffineGridGrad");
  m.add_functor<impl::GridSampleGradFunctor>("GridSampleGrad");
  m.add_functor<impl::PoolingNdGradFunctor>("PoolingNdGrad");
  m.add_functor<impl::PadGradFunctor>("PadGrad");
  m.add_functor<impl::AvgPoolingNdGradFunctor>("AvgPoolingNdGrad");
  m.add_functor<impl::NormalizationGradFunctor>("NormalizationGrad");
  m.add_functor<impl::NormalizationAddReluGradFunctor>("NormalizationAddReluGrad");
  m.add_functor<impl::LayerNormGradFunctor>("LayerNormGrad");
  m.add_functor<impl::LayerNormParamGradFunctor>("LayerNormParamGrad");
  m.add_functor<impl::LayerNormAffineParamGradFunctor>("LayerNormAffineParamGrad");
  m.add_functor<impl::BroadcastMatmulGradBFunctor>("BroadcastMatmulGradB");
  m.add_functor<impl::CtcLossGradFunctor>("CtcLossGrad");
  m.add_functor<impl::FusedScaleTrilSoftmaxMaskScaleGradFunctor>(
      "FusedScaleTrilSoftmaxMaskScaleGrad");
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
