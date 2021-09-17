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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    JUST(attrs.SetAttr<std::vector<int32_t>>("padding", padding));
    JUST(attrs.SetAttr<std::vector<int32_t>>("kernel_size", kernel_size));
    JUST(attrs.SetAttr<std::vector<int32_t>>("stride", stride));
    JUST(attrs.SetAttr<std::vector<int32_t>>("dilation", dilation));
    JUST(attrs.SetAttr<bool>("return_indices", return_indices));
    JUST(attrs.SetAttr<bool>("ceil_mode", ceil_mode));
    const auto& op_type_name = GetOpTypeName(mode, ndims);
    const auto& it = op_expr_map_.find(op_type_name);
    CHECK_OR_RETURN(it != op_expr_map_.end())
        << "Encounter unsupported op " << op_type_name << " in PoolingNdGradFunctor.";
    CHECK_NOTNULL_OR_RETURN(it->second);
    return OpInterpUtil::Dispatch<Tensor>(*it->second, {x, y, indice, dy}, attrs);
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
        << "Encounter unsupported op " << op_type_name << " in PoolNdGradFunctor.";
    CHECK_NOTNULL_OR_RETURN(it->second);
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
        << "Encounter unsupported op " << op_type_name << " in AdaptivePoolNdGradFunctor.";
    CHECK_NOTNULL_OR_RETURN(it->second);
    return OpInterpUtil::Dispatch<Tensor>(*it->second, {x, dy});
  }

 protected:
  std::unordered_map<std::string, std::shared_ptr<OpExpr>> op_expr_map_;
};

class SmoothL1LossGradFunctor {
 public:
  SmoothL1LossGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("smooth_l1_loss_grad")
                         .Input("loss_grad")
                         .Input("prediction")
                         .Input("label")
                         .Output("prediction_grad")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& loss_grad,
                           const std::shared_ptr<one::Tensor>& prediction,
                           const std::shared_ptr<one::Tensor>& label, const float& beta) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<float>("beta", beta));
    return OpInterpUtil::Dispatch<one::Tensor>(*op_, {loss_grad, prediction, label}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::vector<int64_t>>("padding", pad));
    if (mode == "constant") {
      std::vector<int64_t> pad_before(ndim, 0);
      std::vector<int64_t> pad_after(ndim, 0);
      const int64_t pad_pair = pad.size() / 2;
      for (int64_t i = 0; i < pad_pair; ++i) {
        pad_before[ndim - i - 1] = pad[2 * i];
        pad_after[ndim - i - 1] = pad[2 * i + 1];
      }
      JUST(attrs.SetAttr<std::vector<int64_t>>("padding_before", pad_before));
      JUST(attrs.SetAttr<std::vector<int64_t>>("padding_after", pad_after));

      if (IsFloatingDataType(dy->dtype()->data_type())) {
        JUST(attrs.SetAttr<double>("floating_constant_value", JUST(value.As<double>())));
        JUST(attrs.SetAttr<int64_t>("integral_constant_value", 0));
      } else if (IsIntegralDataType(dy->dtype()->data_type())) {
        JUST(attrs.SetAttr<double>("floating_constant_value", 0));
        JUST(attrs.SetAttr<int64_t>("integral_constant_value", JUST(value.As<int64_t>())));
      }
      return OpInterpUtil::Dispatch<Tensor>(*pad_grad_, {dy}, attrs);
    } else if (mode == "reflect") {
      return OpInterpUtil::Dispatch<Tensor>(*reflect_pad_grad_, {dy}, attrs);
    } else if (mode == "replicate") {
      return OpInterpUtil::Dispatch<Tensor>(*replicate_pad_grad_, {dy}, attrs);
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    JUST(attrs.SetAttr<std::vector<int32_t>>("padding", padding));
    JUST(attrs.SetAttr<std::vector<int32_t>>("kernel_size", kernel_size));
    JUST(attrs.SetAttr<std::vector<int32_t>>("stride", stride));
    JUST(attrs.SetAttr<bool>("ceil_mode", ceil_mode));
    JUST(attrs.SetAttr<bool>("count_include_pad", count_include_pad));
    JUST(attrs.SetAttr<int64_t>("divisor_override", divisor_override));
    const auto& op_type_name = GetOpTypeName(ndims);
    const auto& it = op_expr_map_.find(op_type_name);
    CHECK_OR_RETURN(it != op_expr_map_.end())
        << "Encounter unsupported op " << op_type_name << " in PoolingNdGradFunctor.";
    CHECK_NOTNULL_OR_RETURN(it->second);
    return OpInterpUtil::Dispatch<Tensor>(*it->second, {x, y, dy}, attrs);
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
  }
  Maybe<TensorTuple> operator()(
      const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& grad,
      const std::shared_ptr<one::Tensor>& mean, const std::shared_ptr<one::Tensor>& inv_variance,
      const std::shared_ptr<one::Tensor>& gamma, const std::shared_ptr<one::Tensor>& beta,
      const std::shared_ptr<one::Tensor>& reserve_space, const std::shared_ptr<one::Tensor>& y,
      const int32_t& axis, const float& epsilon) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("axis", axis));
    JUST(attrs.SetAttr<float>("epsilon", epsilon));
    return OpInterpUtil::Dispatch<TensorTuple>(
        *addend_op_, {x, grad, mean, inv_variance, gamma, beta, reserve_space, y}, attrs);
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("begin_norm_axis", begin_norm_axis));
    JUST(attrs.SetAttr<double>("epsilon", epsilon));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x, mean, inv_variance, dy}, attrs);
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("begin_params_axis", begin_params_axis));
    JUST(attrs.SetAttr<double>("epsilon", epsilon));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy}, attrs);
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("begin_params_axis", begin_params_axis));
    JUST(attrs.SetAttr<double>("epsilon", epsilon));
    return OpInterpUtil::Dispatch<TensorTuple>(*op_, {dy, gamma, normalized}, attrs);
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

class WhereScalarXFunctor {
 public:
  WhereScalarXFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("where_scalar_x").Input("condition").Input("y").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& condition, const Scalar& scalar,
                           const std::shared_ptr<one::Tensor>& y) const {
    MutableAttrMap attrs;
    if (scalar.IsFloatingPoint()) {
      JUST(attrs.SetAttr<double>("float_operand", JUST(scalar.As<double>())));
      JUST(attrs.SetAttr<bool>("has_float_operand", true));
      JUST(attrs.SetAttr<bool>("has_int_operand", false));
    } else if (scalar.IsIntegral()) {
      JUST(attrs.SetAttr<int64_t>("int_operand", JUST(scalar.As<int64_t>())));
      JUST(attrs.SetAttr<bool>("has_float_operand", false));
      JUST(attrs.SetAttr<bool>("has_int_operand", true));
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "The scalar in Where shoule be float or int.";
    }
    return OpInterpUtil::Dispatch<Tensor>(*op_, {condition, y}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class WhereScalarYFunctor {
 public:
  WhereScalarYFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("where_scalar_y").Input("condition").Input("x").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& condition,
                           const std::shared_ptr<one::Tensor>& x, const Scalar& scalar) const {
    MutableAttrMap attrs;
    if (scalar.IsFloatingPoint()) {
      JUST(attrs.SetAttr<double>("float_operand", JUST(scalar.As<double>())));
      JUST(attrs.SetAttr<bool>("has_float_operand", true));
      JUST(attrs.SetAttr<bool>("has_int_operand", false));
    } else if (scalar.IsIntegral()) {
      JUST(attrs.SetAttr<int64_t>("int_operand", JUST(scalar.As<int64_t>())));
      JUST(attrs.SetAttr<bool>("has_float_operand", false));
      JUST(attrs.SetAttr<bool>("has_int_operand", true));
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "The scalar in Where shoule be float or int.";
    }
    return OpInterpUtil::Dispatch<Tensor>(*op_, {condition, x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class WhereScalarXYFunctor {
 public:
  WhereScalarXYFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("where_scalar_xy").Input("condition").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& condition, const Scalar& x_scalar,
                           const Scalar& y_scalar) const {
    MutableAttrMap attrs;
    if (x_scalar.IsFloatingPoint() && y_scalar.IsFloatingPoint()) {
      JUST(attrs.SetAttr<double>("x_float_operand", JUST(x_scalar.As<double>())));
      JUST(attrs.SetAttr<double>("y_float_operand", JUST(y_scalar.As<double>())));
      JUST(attrs.SetAttr<bool>("has_x_float_operand", true));
      JUST(attrs.SetAttr<bool>("has_y_float_operand", true));
      JUST(attrs.SetAttr<bool>("has_x_int_operand", false));
      JUST(attrs.SetAttr<bool>("has_y_int_operand", false));
    } else if (x_scalar.IsIntegral() && y_scalar.IsIntegral()) {
      JUST(attrs.SetAttr<int64_t>("x_int_operand", JUST(x_scalar.As<int64_t>())));
      JUST(attrs.SetAttr<int64_t>("y_int_operand", JUST(y_scalar.As<int64_t>())));
      JUST(attrs.SetAttr<bool>("has_x_float_operand", false));
      JUST(attrs.SetAttr<bool>("has_y_float_operand", false));
      JUST(attrs.SetAttr<bool>("has_x_int_operand", true));
      JUST(attrs.SetAttr<bool>("has_y_int_operand", true));
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "The scalar in Where shoule be float or int.";
    }
    return OpInterpUtil::Dispatch<Tensor>(*op_, {condition}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class SliceGradBaseFunctor {
 public:
  SliceGradBaseFunctor() = default;
  virtual ~SliceGradBaseFunctor() = default;
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& like,
                           const std::vector<int64_t>& start, const std::vector<int64_t>& stop,
                           const std::vector<int64_t>& step) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::vector<int64_t>>("start", start));
    JUST(attrs.SetAttr<std::vector<int64_t>>("stop", stop));
    JUST(attrs.SetAttr<std::vector<int64_t>>("step", step));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, like}, attrs);
  }

 protected:
  std::shared_ptr<OpExpr> op_;
};

class SliceGradFunctor : public SliceGradBaseFunctor {
 public:
  SliceGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("slice_grad").Input("dy").Input("like").Output("dx").Build());
  }
};

class NarrowGradFunctor {
 public:
  NarrowGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("narrow_grad").Input("dy").Input("like").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& like, const int64_t& dim,
                           const int64_t& start, const int64_t& length) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("dim", dim));
    JUST(attrs.SetAttr<int64_t>("start", start));
    JUST(attrs.SetAttr<int64_t>("length", length));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, like}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class FlipGradFunctor {
 public:
  FlipGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("flip_grad").Input("dy").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::vector<int32_t>& dims) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::vector<int32_t>>("dims", dims));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class UpsampleLinear1DGradFunctor {
 public:
  UpsampleLinear1DGradFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("upsample_linear_1d_grad").Input("dy").Input("x").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& x, const float& scale_factor,
                           const bool& align_corners, const std::string& data_format) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<float>("scale_factor", scale_factor));
    JUST(attrs.SetAttr<bool>("align_corners", align_corners));
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class UpsampleBilinear2DGradFunctor {
 public:
  UpsampleBilinear2DGradFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("upsample_bilinear_2d_grad").Input("dy").Input("x").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& x, const float& height_scale,
                           const float& width_scale, const bool& align_corners,
                           const std::string& data_format) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<float>("height_scale", height_scale));
    JUST(attrs.SetAttr<float>("width_scale", width_scale));
    JUST(attrs.SetAttr<bool>("align_corners", align_corners));
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class UpsampleNearest1DGradFunctor {
 public:
  UpsampleNearest1DGradFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("upsample_nearest_1d_grad").Input("dy").Input("x").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& x, const float& scale_factor,
                           const std::string& data_format) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<float>("scale_factor", scale_factor));
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class UpsampleNearest2DGradFunctor {
 public:
  UpsampleNearest2DGradFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("upsample_nearest_2d_grad").Input("dy").Input("x").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& x, const float& height_scale,
                           const float& width_scale, const std::string& data_format) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<float>("height_scale", height_scale));
    JUST(attrs.SetAttr<float>("width_scale", width_scale));
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class UpsampleNearest3DGradFunctor {
 public:
  UpsampleNearest3DGradFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("upsample_nearest_3d_grad").Input("dy").Input("x").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& x, const float& depth_scale,
                           const float& height_scale, const float& width_scale,
                           const std::string& data_format) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<float>("depth_scale", depth_scale));
    JUST(attrs.SetAttr<float>("height_scale", height_scale));
    JUST(attrs.SetAttr<float>("width_scale", width_scale));
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class UpsampleBicubic2DGradFunctor {
 public:
  UpsampleBicubic2DGradFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("upsample_bicubic_2d_grad").Input("dy").Input("x").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& x, const float& height_scale,
                           const float& width_scale, const bool& align_corners,
                           const std::string& data_format) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<float>("height_scale", height_scale));
    JUST(attrs.SetAttr<float>("width_scale", width_scale));
    JUST(attrs.SetAttr<bool>("align_corners", align_corners));
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class UpsampleTrilinear3DGradFunctor {
 public:
  UpsampleTrilinear3DGradFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("upsample_trilinear_3d_grad").Input("dy").Input("x").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& x, const float& depth_scale,
                           const float& height_scale, const float& width_scale,
                           const bool& align_corners, const std::string& data_format) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<float>("depth_scale", depth_scale));
    JUST(attrs.SetAttr<float>("height_scale", height_scale));
    JUST(attrs.SetAttr<float>("width_scale", width_scale));
    JUST(attrs.SetAttr<bool>("align_corners", align_corners));
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class DiagGradFunctor {
 public:
  DiagGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("diag_grad").Input("dy").Input("in").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& x, const int32_t& diagonal) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("diagonal", diagonal));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ElementwiseMinimumGradFunctor {
 public:
  ElementwiseMinimumGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("elementwise_minimum_backward")
                         .Input("dz")
                         .Input("x")
                         .Input("y")
                         .Output("dx")
                         .Output("dy")
                         .Build());
  }
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& dz,
                                const std::shared_ptr<one::Tensor>& x,
                                const std::shared_ptr<one::Tensor>& y) const {
    return OpInterpUtil::Dispatch<TensorTuple>(*op_, {dz, x, y});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ElementwiseMaximumGradFunctor {
 public:
  ElementwiseMaximumGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("elementwise_maximum_backward")
                         .Input("dz")
                         .Input("x")
                         .Input("y")
                         .Output("dx")
                         .Output("dy")
                         .Build());
  }
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& dz,
                                const std::shared_ptr<one::Tensor>& x,
                                const std::shared_ptr<one::Tensor>& y) const {
    return OpInterpUtil::Dispatch<TensorTuple>(*op_, {dz, x, y});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class DivGradFunctor {
 public:
  DivGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("broadcast_div_grad")
                         .Input("dz")
                         .Input("z")
                         .Input("y")
                         .Output("dy")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dz,
                           const std::shared_ptr<one::Tensor>& z,
                           const std::shared_ptr<one::Tensor>& y) const {
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dz, z, y});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class BroadcastPowXGradFunctor {
 public:
  BroadcastPowXGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("broadcast_pow_x_grad")
                         .Input("dz")
                         .Input("x")
                         .Input("y")
                         .Input("z")
                         .Output("dx")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dz,
                           const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& y,
                           const std::shared_ptr<one::Tensor>& z) const {
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dz, x, y, z});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class BroadcastPowYGradFunctor {
 public:
  BroadcastPowYGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("broadcast_pow_y_grad")
                         .Input("dz")
                         .Input("x")
                         .Input("y")
                         .Input("z")
                         .Output("dy")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dz,
                           const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& y,
                           const std::shared_ptr<one::Tensor>& z) const {
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dz, x, y, z});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::ConvBiasGradFunctor>("ConvBiasGrad");
  m.add_functor<impl::ConvFilterGradFunctor>("ConvFilterGrad");
  m.add_functor<impl::ConvDataGradFunctor>("ConvDataGrad");
  m.add_functor<impl::PoolNdGradFunctor>("PoolNdGrad");
  m.add_functor<impl::AdaptivePoolNdGradFunctor>("AdaptivePoolNdGrad");
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
  m.add_functor<impl::WhereScalarXFunctor>("WhereScalarX");
  m.add_functor<impl::WhereScalarYFunctor>("WhereScalarY");
  m.add_functor<impl::WhereScalarXYFunctor>("WhereScalarXY");
  m.add_functor<impl::SliceGradFunctor>("SliceGrad");
  m.add_functor<impl::NarrowGradFunctor>("NarrowGrad");
  m.add_functor<impl::FlipGradFunctor>("FlipGrad");
  m.add_functor<impl::UpsampleLinear1DGradFunctor>("UpsampleLinear1DGrad");
  m.add_functor<impl::UpsampleNearest1DGradFunctor>("UpsampleNearest1DGrad");
  m.add_functor<impl::UpsampleNearest2DGradFunctor>("UpsampleNearest2DGrad");
  m.add_functor<impl::UpsampleBilinear2DGradFunctor>("UpsampleBilinear2DGrad");
  m.add_functor<impl::UpsampleBicubic2DGradFunctor>("UpsampleBicubic2DGrad");
  m.add_functor<impl::UpsampleNearest3DGradFunctor>("UpsampleNearest3DGrad");
  m.add_functor<impl::UpsampleTrilinear3DGradFunctor>("UpsampleTrilinear3DGrad");
  m.add_functor<impl::DiagGradFunctor>("DiagGrad");
  m.add_functor<impl::ElementwiseMinimumGradFunctor>("ElementwiseMinGrad");
  m.add_functor<impl::ElementwiseMaximumGradFunctor>("ElementwiseMaxGrad");
  m.add_functor<impl::BroadcastPowXGradFunctor>("BroadcastPowXGrad");
  m.add_functor<impl::BroadcastPowYGradFunctor>("BroadcastPowYGrad");
  m.add_functor<impl::DivGradFunctor>("DivGrad");
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
