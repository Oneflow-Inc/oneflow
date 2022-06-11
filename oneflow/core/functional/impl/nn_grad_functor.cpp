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
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/functional/function_library.h"
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("num_spatial_dims", num_spatial_dims));
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("num_spatial_dims", num_spatial_dims));
    JUST(attrs.SetAttr<std::vector<int32_t>>("kernel_size", kernel_size));
    JUST(attrs.SetAttr<std::vector<int32_t>>("strides", strides));
    JUST(attrs.SetAttr<std::vector<int32_t>>("padding_before", padding_before));
    JUST(attrs.SetAttr<std::vector<int32_t>>("dilation_rate", dilation_rate));
    JUST(attrs.SetAttr<int32_t>("groups", groups));
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("num_spatial_dims", num_spatial_dims));
    JUST(attrs.SetAttr<std::vector<int32_t>>("kernel_size", kernel_size));
    JUST(attrs.SetAttr<std::vector<int32_t>>("strides", strides));
    JUST(attrs.SetAttr<std::vector<int32_t>>("padding_before", padding_before));
    JUST(attrs.SetAttr<std::vector<int32_t>>("dilation_rate", dilation_rate));
    JUST(attrs.SetAttr<int32_t>("groups", groups));
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, weight, x}, attrs);
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("padding_idx", padding_idx));
    JUST(attrs.SetAttr<bool>("scale_grad_by_freq", scale_grad_by_freq));
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    JUST(attrs.SetAttr<std::vector<int32_t>>("padding", padding));
    JUST(attrs.SetAttr<std::vector<int32_t>>("kernel_size", kernel_size));
    JUST(attrs.SetAttr<std::vector<int32_t>>("stride", stride));
    JUST(attrs.SetAttr<std::vector<int32_t>>("dilation", dilation));
    JUST(attrs.SetAttr<bool>("return_indices", return_indices));
    JUST(attrs.SetAttr<bool>("ceil_mode", ceil_mode));
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    JUST(attrs.SetAttr<std::string>("padding", padding));
    JUST(attrs.SetAttr<std::vector<int32_t>>("padding_before", padding_before));
    JUST(attrs.SetAttr<std::vector<int32_t>>("padding_after", padding_after));
    JUST(attrs.SetAttr<std::vector<int32_t>>("pool_size", pool_size));
    JUST(attrs.SetAttr<std::vector<int32_t>>("strides", strides));
    JUST(attrs.SetAttr<bool>("ceil_mode", ceil_mode));
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("depth", depth));

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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("depth", depth));

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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("depth", depth));
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<float>("beta", beta));

    return OpInterpUtil::Dispatch<one::Tensor>(*op_, {dy, input, target}, attrs);
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
                           const std::shared_ptr<one::Tensor>& target,
                           const bool log_target) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<bool>("log_target", log_target));

    return OpInterpUtil::Dispatch<Tensor>(*op_, {input, target, dy}, attrs);
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
                           const int64_t ignore_index) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("ignore_index", ignore_index));

    if (weight) {
      return OpInterpUtil::Dispatch<one::Tensor>(
          *op_weight_, {input, target, total_weight, JUST(weight), dy}, attrs);
    } else {
      return OpInterpUtil::Dispatch<one::Tensor>(*op_, {input, target, total_weight, dy}, attrs);
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
                           const Optional<one::Tensor>& weight) const {
    MutableAttrMap attrs;

    if (weight) {
      return OpInterpUtil::Dispatch<one::Tensor>(*op_weight_, {input, target, JUST(weight), dy},
                                                 attrs);
    } else {
      return OpInterpUtil::Dispatch<one::Tensor>(*op_, {input, target, dy}, attrs);
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
                           const Optional<one::Tensor>& pos_weight) const {
    MutableAttrMap attrs;

    JUST(attrs.SetAttr<bool>("has_pos_weight", pos_weight.has_value()));

    if (weight) {
      if (pos_weight) {
        return OpInterpUtil::Dispatch<one::Tensor>(
            *op_weight_pos_, {input, target, JUST(weight), JUST(pos_weight), dy}, attrs);
      } else {
        return OpInterpUtil::Dispatch<one::Tensor>(*op_weight_, {input, target, JUST(weight), dy},
                                                   attrs);
      }
    } else {
      if (pos_weight) {
        return OpInterpUtil::Dispatch<one::Tensor>(*op_pos_, {input, target, JUST(pos_weight), dy},
                                                   attrs);
      } else {
        return OpInterpUtil::Dispatch<one::Tensor>(*op_, {input, target, dy}, attrs);
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<float>("m1", m1));
    JUST(attrs.SetAttr<float>("m2", m2));
    JUST(attrs.SetAttr<float>("m3", m3));
    JUST(attrs.SetAttr<int64_t>("depth", depth));
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<Shape>("size", size));
    JUST(attrs.SetAttr<bool>("align_corners", align_corners));
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::string>("interpolation_mode", interpolation_mode));
    JUST(attrs.SetAttr<std::string>("padding_mode", padding_mode));
    JUST(attrs.SetAttr<bool>("align_corners", align_corners));
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("blank", blank));
    JUST(attrs.SetAttr<bool>("zero_infinity", zero_infinity));
    JUST(attrs.SetAttr<int64_t>("max_target_length", max_target_length));
    return OpInterpUtil::Dispatch<one::Tensor>(
        *op_, {grad_out, log_probs, targets, input_lengths, target_lengths, loss, alpha}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class PadGradFunctor {
 public:
  PadGradFunctor() {
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
        << Error::RuntimeError() << "Pad size should less than or equal to input axes * 2.";
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::vector<int64_t>>("padding", pad));
    if (mode == "reflect") {
      return OpInterpUtil::Dispatch<Tensor>(*reflect_pad_grad_, {dy}, attrs);
    } else if (mode == "replicate") {
      return OpInterpUtil::Dispatch<Tensor>(*replicate_pad_grad_, {dy}, attrs);
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "Pad mode is " << mode
                                  << ", but only constant, reflect and replicate are valid.";
    }
  }

 private:
  std::shared_ptr<OpExpr> reflect_pad_grad_;
  std::shared_ptr<OpExpr> replicate_pad_grad_;
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    JUST(attrs.SetAttr<std::vector<int32_t>>("padding", padding));
    JUST(attrs.SetAttr<std::vector<int32_t>>("kernel_size", kernel_size));
    JUST(attrs.SetAttr<std::vector<int32_t>>("stride", stride));
    JUST(attrs.SetAttr<bool>("ceil_mode", ceil_mode));
    JUST(attrs.SetAttr<bool>("count_include_pad", count_include_pad));
    JUST(attrs.SetAttr<int32_t>("divisor_override", divisor_override));
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<float>("epsilon", epsilon));
    JUST(attrs.SetAttr<int32_t>("axis", axis));
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("axis", axis));
    JUST(attrs.SetAttr<float>("epsilon", epsilon));
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("begin_norm_axis", begin_norm_axis));
    JUST(attrs.SetAttr<double>("epsilon", epsilon));
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("begin_norm_axis", begin_norm_axis));
    JUST(attrs.SetAttr<double>("epsilon", epsilon));
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
                                const int64_t& begin_params_axis, const double& epsilon) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("begin_params_axis", begin_params_axis));
    JUST(attrs.SetAttr<double>("epsilon", epsilon));
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
    MutableAttrMap attr;
    JUST(attr.SetAttr<double>("alpha", alpha));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {a, b}, attr);
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
    MutableAttrMap fused_attrs;
    JUST(fused_attrs.SetAttr<int64_t>("diagonal", diagonal));
    JUST(fused_attrs.SetAttr<float>("tril_scale_value", tril_scale_value));
    JUST(fused_attrs.SetAttr<float>("mask_scale_value", mask_scale_value));
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
    MutableAttrMap attrs_;
    JUST(attrs_.SetAttr<float>("scale_value", scale));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {y, dy, mask}, attrs_);
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
    MutableAttrMap attrs_;
    JUST(attrs_.SetAttr<float>("scale_value", scale));
    JUST(attrs_.SetAttr<float>("dropout_scale_value", dropout_scale));

    return OpInterpUtil::Dispatch<Tensor>(*op_, {softmax_y, dy, mask, dropout_mask}, attrs_);
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<double>("alpha", alpha));
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
    MutableAttrMap attr_map;
    JUST(attr_map.SetAttr<float>("scale", scale));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, mask}, attr_map);
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<bool>("self_interaction", self_interaction));
    JUST(attrs.SetAttr<int32_t>("output_concat_grad_dim", output_concat_grad_dim));
    JUST(attrs.SetAttr<std::string>("pooling", pooling));
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

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::ConvBiasGradFunctor>("ConvBiasGrad");
  m.add_functor<impl::ConvFilterGradFunctor>("ConvFilterGrad");
  m.add_functor<impl::ConvDataGradFunctor>("ConvDataGrad");
  m.add_functor<impl::EmbeddingGradFunctor>("EmbeddingGrad");
  m.add_functor<impl::TFPoolNdGradFunctor>("TFPoolNdGrad");
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
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
