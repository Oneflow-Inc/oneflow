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

#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/common/scalar.h"
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/tensor_util.h"
#include "oneflow/core/framework/op_interpreter.h"
#include "oneflow/core/framework/random_generator.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/function_library.h"
#include "oneflow/core/functional/sequence_function.h"
#include "oneflow/core/functional/impl/common.h"
#include "oneflow/core/functional/impl/unary_functor.h"
#include "oneflow/core/job/lazy_mode.h"
#include "oneflow/user/kernels/random_mask_like_kernel.h"
#include "oneflow/user/kernels/dropout_kernel.h"
#include "oneflow/core/register/ofblob.h"
namespace oneflow {
namespace one {
namespace functional {

namespace impl {

class BiasAddFunctor {
 public:
  BiasAddFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("bias_add").Input("a").Input("b").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& bias, const int32_t& axis) const {
    MutableAttrMap attrs;
    int32_t axis_val = axis;
    if (axis_val < 0) {
      const int64_t num_axes = x->shape()->NumAxes();
      axis_val += num_axes;
    }
    JUST(attrs.SetAttr<int32_t>("axis", axis_val));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x, bias}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ConvBaseFunctor {
 public:
  explicit ConvBaseFunctor(const int& num_spatial_dims) : num_spatial_dims_(num_spatial_dims) {
    bias_op_ = CHECK_JUST(one::OpBuilder("bias_add").Input("a").Input("b").Output("out").Build());
  }
  virtual ~ConvBaseFunctor() = default;
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& weight,
                           const Optional<one::Tensor>& bias, const std::vector<int32_t>& stride,
                           const std::vector<int32_t>& padding,
                           const std::vector<int32_t>& dilation, const int32_t& groups,
                           const std::string& channel_pos) const {
    MutableAttrMap conv_attrs;
    std::vector<int32_t> kernel_size_vec(num_spatial_dims_);
    int32_t kernel_idx_offset = 2;
    if (channel_pos == "channels_last") { kernel_idx_offset = 1; }

    for (int i = 0; i < num_spatial_dims_; i++) {
      kernel_size_vec.at(i) = ((weight->shape())->At(i + kernel_idx_offset));
    }
    JUST(conv_attrs.SetAttr<int32_t>("filters", (weight->shape())->At(0)));
    JUST(conv_attrs.SetAttr<std::vector<int32_t>>("padding_before", padding));
    JUST(conv_attrs.SetAttr<std::vector<int32_t>>("kernel_size", kernel_size_vec));
    JUST(conv_attrs.SetAttr<std::vector<int32_t>>("strides", stride));
    JUST(conv_attrs.SetAttr<std::vector<int32_t>>("dilation_rate", dilation));
    JUST(conv_attrs.SetAttr<int32_t>("groups", groups));
    JUST(conv_attrs.SetAttr<std::string>("data_format", channel_pos));
    const std::shared_ptr<one::Tensor>& conv_out =
        JUST(OpInterpUtil::Dispatch<Tensor>(*conv_op_, {x, weight}, conv_attrs));
    if (bias) {
      MutableAttrMap bias_attrs;
      JUST(bias_attrs.SetAttr<int32_t>("axis", 1));
      return OpInterpUtil::Dispatch<Tensor>(*bias_op_, {conv_out, JUST(bias)}, bias_attrs);
    } else {
      return conv_out;
    }
  }

 protected:
  std::shared_ptr<OpExpr> conv_op_;
  std::shared_ptr<OpExpr> bias_op_;
  int32_t num_spatial_dims_;
};

class Conv1dFunctor : public ConvBaseFunctor {
 public:
  Conv1dFunctor() : ConvBaseFunctor(/*num_spatial_dims_=*/1) {
    conv_op_ =
        CHECK_JUST(one::OpBuilder("conv1d").Input("in").Input("weight").Output("out").Build());
  }
};

class Conv2dFunctor : public ConvBaseFunctor {
 public:
  Conv2dFunctor() : ConvBaseFunctor(/*num_spatial_dims_=*/2) {
    conv_op_ =
        CHECK_JUST(one::OpBuilder("conv2d").Input("in").Input("weight").Output("out").Build());
  }
};

class Conv3dFunctor : public ConvBaseFunctor {
 public:
  Conv3dFunctor() : ConvBaseFunctor(/*num_spatial_dims_=*/3) {
    conv_op_ =
        CHECK_JUST(one::OpBuilder("conv3d").Input("in").Input("weight").Output("out").Build());
  }
};

class DeConvBaseFunctor {
 public:
  explicit DeConvBaseFunctor() {
    bias_op_ = CHECK_JUST(one::OpBuilder("bias_add").Input("a").Input("b").Output("out").Build());
  }
  virtual ~DeConvBaseFunctor() = default;
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& weight,
                           const Optional<one::Tensor>& bias, const int32_t& filters,
                           const std::vector<int32_t>& padding, const std::string& data_format,
                           const std::vector<int32_t>& kernel_size,
                           const std::vector<int32_t>& output_padding,
                           const std::vector<int32_t>& strides,
                           const std::vector<int32_t>& dilation, const int32_t& groups) const {
    MutableAttrMap deconv_attrs;
    JUST(deconv_attrs.SetAttr<int32_t>("filters", filters));
    JUST(deconv_attrs.SetAttr<std::vector<int32_t>>("padding_before", padding));
    JUST(deconv_attrs.SetAttr<std::vector<int32_t>>("kernel_size", kernel_size));
    JUST(deconv_attrs.SetAttr<std::vector<int32_t>>("output_padding", output_padding));
    JUST(deconv_attrs.SetAttr<std::vector<int32_t>>("strides", strides));
    JUST(deconv_attrs.SetAttr<std::vector<int32_t>>("dilation_rate", dilation));
    JUST(deconv_attrs.SetAttr<int32_t>("groups", groups));
    JUST(deconv_attrs.SetAttr<std::string>("data_format", data_format));
    std::shared_ptr<one::Tensor> deconv_out = nullptr;
    deconv_out = JUST(OpInterpUtil::Dispatch<Tensor>(*deconv_op_, {x, weight}, deconv_attrs));
    if (bias) {
      MutableAttrMap bias_attrs;
      JUST(bias_attrs.SetAttr<int32_t>("axis", 1));
      return OpInterpUtil::Dispatch<Tensor>(*bias_op_, {deconv_out, JUST(bias)}, bias_attrs);
    } else {
      return deconv_out;
    }
  }

 protected:
  std::shared_ptr<OpExpr> deconv_op_;
  std::shared_ptr<OpExpr> bias_op_;
};

class DeConv1dFunctor : public DeConvBaseFunctor {
 public:
  DeConv1dFunctor() {
    deconv_op_ =
        CHECK_JUST(one::OpBuilder("deconv1d").Input("in").Input("weight").Output("out").Build());
  }
};

class DeConv2dFunctor : public DeConvBaseFunctor {
 public:
  DeConv2dFunctor() {
    deconv_op_ =
        CHECK_JUST(one::OpBuilder("deconv2d").Input("in").Input("weight").Output("out").Build());
  }
};

class DeConv3dFunctor : public DeConvBaseFunctor {
 public:
  DeConv3dFunctor() {
    deconv_op_ =
        CHECK_JUST(one::OpBuilder("deconv3d").Input("in").Input("weight").Output("out").Build());
  }
};

class MatMulFunctor {
 public:
  MatMulFunctor() {
    matmul_op_ = CHECK_JUST(one::OpBuilder("matmul").Input("a").Input("b").Output("out").Build());
    batch_matmul_op_ =
        CHECK_JUST(one::OpBuilder("batch_matmul").Input("a").Input("b").Output("out").Build());
    bcast_matmul_op_ =
        CHECK_JUST(one::OpBuilder("broadcast_matmul").Input("a").Input("b").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& a,
                           const std::shared_ptr<one::Tensor>& b, const bool& transpose_a,
                           const bool& transpose_b, const double& alpha) const {
    const auto& a_shape = a->shape();
    const auto& b_shape = b->shape();

    // TODO(): Support 1-d tensor by dot.
    CHECK_GE_OR_RETURN(a_shape->NumAxes(), 2) << "Tensor a's dim should >= 2";
    CHECK_GE_OR_RETURN(b_shape->NumAxes(), 2) << "Tensor b's dim should >= 2";

    MutableAttrMap attrs;
    JUST(attrs.SetAttr<bool>("transpose_a", transpose_a));
    JUST(attrs.SetAttr<bool>("transpose_b", transpose_b));
    JUST(attrs.SetAttr<double>("alpha", alpha));
    if (a_shape->NumAxes() != b_shape->NumAxes()) {
      CHECK_EQ_OR_RETURN(b_shape->NumAxes(), 2)
          << "Not support number of dimensions of a being less than number of dimensions of b!";
      return OpInterpUtil::Dispatch<Tensor>(*bcast_matmul_op_, {a, b}, attrs);
    }
    if (a_shape->NumAxes() > 2) {
      return OpInterpUtil::Dispatch<Tensor>(*batch_matmul_op_, {a, b}, attrs);
    }
    return OpInterpUtil::Dispatch<Tensor>(*matmul_op_, {a, b}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> matmul_op_;
  std::shared_ptr<OpExpr> batch_matmul_op_;
  std::shared_ptr<OpExpr> bcast_matmul_op_;
};

class BatchMatMulFunctor {
 public:
  BatchMatMulFunctor() {
    batch_matmul_op_ =
        CHECK_JUST(one::OpBuilder("batch_matmul").Input("a").Input("b").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& a,
                           const std::shared_ptr<one::Tensor>& b, const bool& transpose_a,
                           const bool& transpose_b, const double& alpha) const {
    const auto& a_shape = a->shape();
    const auto& b_shape = b->shape();
    CHECK_GE_OR_RETURN(a_shape->NumAxes(), 3) << "Tensor a's dim should >= 3";
    CHECK_GE_OR_RETURN(b_shape->NumAxes(), 3) << "Tensor b's dim should >= 3";
    CHECK_GE_OR_RETURN(a_shape->At(0), b_shape->At(0))
        << "batch dim not match, please check input!";
    CHECK_GE_OR_RETURN(a_shape->At(2), b_shape->At(1))
        << "matmul dim not match, please check input!";
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<bool>("transpose_a", transpose_a));
    JUST(attrs.SetAttr<bool>("transpose_b", transpose_b));
    JUST(attrs.SetAttr<double>("alpha", alpha));
    return OpInterpUtil::Dispatch<Tensor>(*batch_matmul_op_, {a, b}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> batch_matmul_op_;
};

class LayerNormFunctor {
 public:
  LayerNormFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("layer_norm")
                         .Input("x")
                         .Output("y")
                         .Output("mean")
                         .Output("inv_variance")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const int64_t& begin_norm_axis,
                           const int64_t& begin_params_axis, const double& epsilon) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("begin_norm_axis", begin_norm_axis));
    JUST(attrs.SetAttr<int64_t>("begin_params_axis", begin_params_axis));
    JUST(attrs.SetAttr<double>("epsilon", epsilon));
    JUST(attrs.SetAttr<bool>("center", false));
    JUST(attrs.SetAttr<bool>("scale", false));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class LayerNormAffineFunctor {
 public:
  LayerNormAffineFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("layer_norm")
                         .Input("x")
                         .Input("gamma")
                         .Input("beta")
                         .Output("y")
                         .Output("mean")
                         .Output("inv_variance")
                         .Output("normalized")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& gamma,
                           const std::shared_ptr<one::Tensor>& beta, const int64_t& begin_norm_axis,
                           const int64_t& begin_params_axis, const double& epsilon) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("begin_norm_axis", begin_norm_axis));
    JUST(attrs.SetAttr<int64_t>("begin_params_axis", begin_params_axis));
    JUST(attrs.SetAttr<double>("epsilon", epsilon));
    JUST(attrs.SetAttr<bool>("center", true));
    JUST(attrs.SetAttr<bool>("scale", true));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x, gamma, beta}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class PoolNDFunctor {
 public:
  PoolNDFunctor() = default;
  virtual ~PoolNDFunctor() = default;
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::vector<int32_t>& kernel_size,
                           const std::vector<int32_t>& strides, const std::string& padding,
                           const std::vector<int32_t>& padding_before,
                           const std::vector<int32_t>& padding_after,
                           const std::string& data_format, const bool& ceil_mode) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::vector<int32_t>>("pool_size", kernel_size));
    JUST(attrs.SetAttr<std::vector<int32_t>>("strides", strides));
    JUST(attrs.SetAttr<std::string>("padding", padding));
    JUST(attrs.SetAttr<std::vector<int32_t>>("padding_before", padding_before));
    JUST(attrs.SetAttr<std::vector<int32_t>>("padding_after", padding_after));
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    JUST(attrs.SetAttr<bool>("ceil_mode", ceil_mode));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 protected:
  std::shared_ptr<OpExpr> op_;
};

class PoolingNDFunctor {
 public:
  PoolingNDFunctor() = default;
  virtual ~PoolingNDFunctor() = default;
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& x,
                                const std::vector<int32_t>& kernel_size,
                                const Optional<std::vector<int32_t>>& stride,
                                const std::vector<int32_t>& padding,
                                const std::vector<int32_t>& dilation, const bool& return_indices,
                                const bool& ceil_mode, const std::string& data_format) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    JUST(attrs.SetAttr<std::vector<int32_t>>("padding", padding));
    JUST(attrs.SetAttr<std::vector<int32_t>>("kernel_size", kernel_size));
    if (stride.has_value()) {
      JUST(attrs.SetAttr<std::vector<int32_t>>("stride", *JUST(stride)));
    } else {
      JUST(attrs.SetAttr<std::vector<int32_t>>(
          "stride", kernel_size));  // If stride is None, we set it as kernel_size to align Pytorch.
    }
    JUST(attrs.SetAttr<std::vector<int32_t>>("dilation", dilation));
    JUST(attrs.SetAttr<bool>("return_indices", return_indices));
    JUST(attrs.SetAttr<bool>("ceil_mode", ceil_mode));
    return OpInterpUtil::Dispatch<TensorTuple>(*op_, {x}, attrs);
  }

 protected:
  std::shared_ptr<OpExpr> op_;
};

class TFAvgPool2DFunctor : public PoolNDFunctor {
 public:
  TFAvgPool2DFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("tf_avg_pool_2d").Input("x").Output("y").Build());
  }
};

class TFMaxPool2DFunctor : public PoolNDFunctor {
 public:
  TFMaxPool2DFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("tf_max_pool_2d").Input("x").Output("y").Build());
  }
};

class Maxpool1DFunctor : public PoolingNDFunctor {
 public:
  Maxpool1DFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("maxpool_1d").Input("x").Output("y").Output("indice").Build());
  }
};

class Maxpool2DFunctor : public PoolingNDFunctor {
 public:
  Maxpool2DFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("maxpool_2d").Input("x").Output("y").Output("indice").Build());
  }
};

class Maxpool3DFunctor : public PoolingNDFunctor {
 public:
  Maxpool3DFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("maxpool_3d").Input("x").Output("y").Output("indice").Build());
  }
};

class AdaptivePoolNDFunctor {
 public:
  AdaptivePoolNDFunctor() = default;
  virtual ~AdaptivePoolNDFunctor() = default;
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::vector<int64_t>& output_size) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::vector<int64_t>>("output_size", output_size));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 protected:
  std::shared_ptr<OpExpr> op_;
};

class AdaptiveAvgPool1DFunctor : public AdaptivePoolNDFunctor {
 public:
  AdaptiveAvgPool1DFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("adaptive_avg_pool1d").Input("x").Output("y").Build());
  }
};

class AdaptiveAvgPool2DFunctor : public AdaptivePoolNDFunctor {
 public:
  AdaptiveAvgPool2DFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("adaptive_avg_pool2d").Input("x").Output("y").Build());
  }
};

class AdaptiveAvgPool3DFunctor : public AdaptivePoolNDFunctor {
 public:
  AdaptiveAvgPool3DFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("adaptive_avg_pool3d").Input("x").Output("y").Build());
  }
};

class LossFunctorBase {
 public:
  Maybe<Tensor> apply_reduction(const Maybe<Tensor>& x, const std::string& reduction) const {
    CHECK_OR_RETURN(reduction == "none" || reduction == "sum" || reduction == "mean")
        << "Reduction should be none, sum or mean.";
    if (reduction == "sum") { return functional::ReduceSum(JUST(x), {}, false); }
    if (reduction == "mean") { return functional::ReduceMean(JUST(x), {}, false); }
    return x;
  }

 protected:
  LossFunctorBase() = default;
  virtual ~LossFunctorBase() = default;
};

class MseLossFunctor : public LossFunctorBase {
 public:
  MseLossFunctor() {}
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& target,
                           const std::string& reduction) const {
    const auto out =
        sequence_function(functional::Sub).then(functional::Square).call(input, target);
    return apply_reduction(out, reduction);
  }
};

class L1LossFunctor : public LossFunctorBase {
 public:
  L1LossFunctor() {}
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& target,
                           const std::string& reduction) const {
    const auto out = sequence_function(functional::Sub).then(functional::Abs).call(input, target);
    return apply_reduction(out, reduction);
  }
};

class SmoothL1LossFunctor : LossFunctorBase {
 public:
  SmoothL1LossFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("smooth_l1_loss").Input("input").Input("target").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& target, const float& beta,
                           const std::string& reduction) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<float>("beta", beta));
    return apply_reduction(OpInterpUtil::Dispatch<Tensor>(*op_, {input, target}, attrs), reduction);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class KLDivLossFunctor : public LossFunctorBase {
 public:
  KLDivLossFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("kl_div_loss").Input("input").Input("target").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& target, const bool log_target,
                           const std::string& reduction) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<bool>("log_target", log_target));
    return apply_reduction(OpInterpUtil::Dispatch<Tensor>(*op_, {input, target}, attrs), reduction);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class MarginRankingLossFunctor : public LossFunctorBase {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input_1,
                           const std::shared_ptr<one::Tensor>& input_2,
                           const std::shared_ptr<one::Tensor>& target, const float margin,
                           const std::string& reduction) const {
    const auto out =
        sequence_function(functional::Sub)
            .then(functional::Negative)
            .then(std::bind(functional::Mul, target, std::placeholders::_1))
            .then([&margin](const std::shared_ptr<one::Tensor>& x) {
              return functional::ScalarAdd(x, Scalar(margin), /*alpha=*/1, /*inplace=*/true);
            })
            .then(std::bind(functional::Clamp, std::placeholders::_1, Scalar(0), NullOpt))
            .call(input_1, input_2);
    return apply_reduction(out, reduction);
  }
};

class BinaryCrossEntropyLossFunctor : public LossFunctorBase {
 public:
  BinaryCrossEntropyLossFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("binary_cross_entropy")
                         .Input("input")
                         .Input("target")
                         .Output("out")
                         .Build());
    op_weight_ = CHECK_JUST(one::OpBuilder("binary_cross_entropy")
                                .Input("input")
                                .Input("target")
                                .Input("weight")
                                .Output("out")
                                .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& target,
                           const Optional<one::Tensor>& weight,
                           const std::string& reduction) const {
    MutableAttrMap attrs;
    auto out =
        weight ? OpInterpUtil::Dispatch<Tensor>(*op_weight_, {input, target, JUST(weight)}, attrs)
               : OpInterpUtil::Dispatch<Tensor>(*op_, {input, target}, attrs);
    return apply_reduction(out, reduction);
  }

 private:
  std::shared_ptr<OpExpr> op_;
  std::shared_ptr<OpExpr> op_weight_;
};

class BinaryCrossEntropyWithLogitsLossFunctor : public LossFunctorBase {
 public:
  BinaryCrossEntropyWithLogitsLossFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("binary_cross_entropy_with_logits")
                         .Input("input")
                         .Input("target")
                         .Output("out")
                         .Build());
    op_weight_ = CHECK_JUST(one::OpBuilder("binary_cross_entropy_with_logits")
                                .Input("input")
                                .Input("target")
                                .Input("weight")
                                .Output("out")
                                .Build());
    op_pos_ = CHECK_JUST(one::OpBuilder("binary_cross_entropy_with_logits")
                             .Input("input")
                             .Input("target")
                             .Input("pos_weight")
                             .Output("out")
                             .Build());
    op_weight_pos_ = CHECK_JUST(one::OpBuilder("binary_cross_entropy_with_logits")
                                    .Input("input")
                                    .Input("target")
                                    .Input("weight")
                                    .Input("pos_weight")
                                    .Output("out")
                                    .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& target,
                           const Optional<one::Tensor>& weight,
                           const Optional<one::Tensor>& pos_weight,
                           const std::string& reduction) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<bool>("has_pos_weight", pos_weight.has_value()));

    std::shared_ptr<Tensor> out;
    if (weight) {
      if (pos_weight) {
        out = JUST(OpInterpUtil::Dispatch<Tensor>(
            *op_weight_pos_, {input, target, JUST(weight), JUST(pos_weight)}, attrs));
      } else {
        out =
            JUST(OpInterpUtil::Dispatch<Tensor>(*op_weight_, {input, target, JUST(weight)}, attrs));
      }
    } else {
      if (pos_weight) {
        out = JUST(
            OpInterpUtil::Dispatch<Tensor>(*op_pos_, {input, target, JUST(pos_weight)}, attrs));
      } else {
        out = JUST(OpInterpUtil::Dispatch<Tensor>(*op_, {input, target}, attrs));
      }
    }
    return apply_reduction(out, reduction);
  }

 private:
  std::shared_ptr<OpExpr> op_;
  std::shared_ptr<OpExpr> op_weight_;
  std::shared_ptr<OpExpr> op_pos_;
  std::shared_ptr<OpExpr> op_weight_pos_;
};

class NllLossFunctor {
 public:
  NllLossFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("nll")
                         .Input("input")
                         .Input("target")
                         .Output("out")
                         .Output("total_weight")
                         .Build());
    op_weight_ = CHECK_JUST(one::OpBuilder("nll")
                                .Input("input")
                                .Input("target")
                                .Input("weight")
                                .Output("out")
                                .Output("total_weight")
                                .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& target,
                           const Optional<one::Tensor>& weight, const int64_t& ignore_index,
                           const std::string& reduction) const {
    CHECK_OR_RETURN(reduction == "none" || reduction == "sum" || reduction == "mean")
        << "Reduction should be none, sum or mean.";

    const auto& input_shape = input->shape();
    const auto& target_shape = target->shape();
    CHECK_LE_OR_RETURN(input_shape->NumAxes(), 5);
    CHECK_EQ_OR_RETURN(input_shape->NumAxes() - 1, target_shape->NumAxes());

    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("ignore_index", ignore_index));

    std::vector<int> input_perm(input_shape->dim_vec().size(), 0);
    input_perm[input_perm.size() - 1] = 1;
    for (size_t i = 1; i < input_perm.size() - 1; ++i) { input_perm[i] = i + 1; }

    const auto input_ = JUST(sequence_function(functional::Transpose)
                                 .then(std::bind(functional::Reshape, std::placeholders::_1,
                                                 Shape({-1, input_shape->At(1)})))
                                 .call(input, input_perm));
    auto target_ = JUST(functional::Flatten(target, 0, target_shape->NumAxes() - 1));

    std::shared_ptr<TensorTuple> kernel_result;
    std::shared_ptr<Tensor> result;
    if (weight) {
      kernel_result = JUST(
          OpInterpUtil::Dispatch<TensorTuple>(*op_weight_, {input_, target_, JUST(weight)}, attrs));
    } else {
      kernel_result = JUST(OpInterpUtil::Dispatch<TensorTuple>(*op_, {input_, target_}, attrs));
    }
    result = JUST(functional::Reshape(kernel_result->at(0), *target_shape));
    if (reduction == "none") { return result; }

    result = JUST(functional::ReduceSum(result, {}, false));

    if (reduction == "sum") { return result; }

    return functional::Div(result, kernel_result->at(1));
  }

 private:
  std::shared_ptr<OpExpr> op_;
  std::shared_ptr<OpExpr> op_weight_;
};

class CrossEntropyFunctor {
 public:
  CrossEntropyFunctor() {
    op_log_softmax_ = CHECK_JUST(one::OpBuilder("log_softmax").Input("in").Output("prob").Build());
    op_nll_ = CHECK_JUST(one::OpBuilder("nll")
                             .Input("input")
                             .Input("target")
                             .Output("out")
                             .Output("total_weight")
                             .Build());
    op_nll_weight_ = CHECK_JUST(one::OpBuilder("nll")
                                    .Input("input")
                                    .Input("target")
                                    .Input("weight")
                                    .Output("out")
                                    .Output("total_weight")
                                    .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& target,
                           const Optional<one::Tensor>& weight, const int64_t& ignore_index,
                           const std::string& reduction) const {
    CHECK_OR_RETURN(reduction == "none" || reduction == "sum" || reduction == "mean")
        << "Reduction should be none, sum or mean.";
    const auto& input_shape = input->shape();
    const auto& target_shape = target->shape();
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("ignore_index", ignore_index));

    std::vector<int> input_perm(input_shape->dim_vec().size(), 0);
    input_perm[input_perm.size() - 1] = 1;
    for (size_t i = 1; i < input_perm.size() - 1; ++i) { input_perm[i] = i + 1; }

    const auto input_ = JUST(sequence_function(functional::Transpose)
                                 .then(std::bind(functional::Reshape, std::placeholders::_1,
                                                 Shape({-1, input_shape->At(1)})))
                                 .then([this](const std::shared_ptr<one::Tensor>& x) {
                                   return OpInterpUtil::Dispatch<Tensor>(*op_log_softmax_, {x});
                                 })
                                 .call(input, input_perm));

    const auto target_ = JUST(functional::Flatten(target, 0, target->shape()->NumAxes() - 1));

    std::shared_ptr<TensorTuple> kernel_result;
    std::shared_ptr<Tensor> result;
    if (weight) {
      kernel_result = JUST(OpInterpUtil::Dispatch<TensorTuple>(
          *op_nll_weight_, {input_, target_, JUST(weight)}, attrs));
    } else {
      kernel_result = JUST(OpInterpUtil::Dispatch<TensorTuple>(*op_nll_, {input_, target_}, attrs));
    }
    result = JUST(functional::Reshape(kernel_result->at(0), *target_shape));
    if (reduction == "none") { return result; }

    result = JUST(functional::ReduceSum(result, {}, false));
    if (reduction == "sum") { return result; }

    return functional::Div(result, kernel_result->at(1));
  }

 private:
  std::shared_ptr<OpExpr> op_log_softmax_;
  std::shared_ptr<OpExpr> op_nll_;
  std::shared_ptr<OpExpr> op_nll_weight_;
};

class SparseCrossEntropyFunctor {
 public:
  SparseCrossEntropyFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("sparse_cross_entropy")
                         .Input("prediction")
                         .Input("label")
                         .Output("out")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& prediction,
                           const std::shared_ptr<one::Tensor>& label, const int64_t& depth) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("depth", depth));

    return OpInterpUtil::Dispatch<Tensor>(*op_, {prediction, label}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class SparseCrossEntropyMsFunctor {
 public:
  SparseCrossEntropyMsFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("sparse_cross_entropy_ms")
                         .Input("prediction")
                         .Input("label")
                         .Output("out")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& prediction,
                           const std::shared_ptr<one::Tensor>& label, const int64_t& depth) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("depth", depth));

    return OpInterpUtil::Dispatch<Tensor>(*op_, {prediction, label}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class SparseSoftmaxCrossEntropyFunctor {
 public:
  SparseSoftmaxCrossEntropyFunctor() {
    // SparseSoftmaxCrossEntropy
    op_sparse_softmax_cross_entropy_ = CHECK_JUST(one::OpBuilder("sparse_softmax_cross_entropy")
                                                      .Input("prediction")
                                                      .Input("label")
                                                      .Output("prob")
                                                      .Output("out")
                                                      .Build());
    // lazy model SparseSoftmaxCrossEntropyMs
    op_sparse_softmax_cross_entropy_ms_ =
        CHECK_JUST(one::OpBuilder("sparse_softmax_cross_entropy_ms")
                       .Input("prediction")
                       .Input("label")
                       .Output("prob")
                       .Output("out")
                       .Build());
    // eager model SparseSoftmaxCrossEntropyMs
    op_reduce_max_device_stage_ = CHECK_JUST(one::OpBuilder("reduce_max_device_stage")
                                                 .Input("in")
                                                 .Output("out")
                                                 .Output("mask")
                                                 .Output("count")
                                                 .Build());
    op_reduce_max_global_stage_ = CHECK_JUST(one::OpBuilder("reduce_max_global_stage")
                                                 .Input("in")
                                                 .Input("device_count")
                                                 .Output("out")
                                                 .Output("mask")
                                                 .Build());
    op_sparse_cross_entropy_ms_ = CHECK_JUST(one::OpBuilder("sparse_cross_entropy_ms")
                                                 .Input("prediction")
                                                 .Input("label")
                                                 .Output("out")
                                                 .Build());
    op_broadcast_sub_ =
        CHECK_JUST(one::OpBuilder("broadcast_sub").Input("x").Input("y").Output("z").Build());
    op_broadcast_div_ =
        CHECK_JUST(one::OpBuilder("broadcast_div").Input("x").Input("y").Output("z").Build());
    op_reduce_sum_ = CHECK_JUST(
        one::OpBuilder("reduce_sum").Input("input_tensor").Output("output_tensor").Build());
    op_exp_ = CHECK_JUST(one::OpBuilder("exp").Input("x").Output("y").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& logits,
                           const std::shared_ptr<one::Tensor>& label) const {
    if (JUST(RunWithMsVersion(logits, label))) {
      if (LazyMode::is_enabled()) {
        return LazySparseSoftmaxCrossEntropyMsOperator(logits, label);
      } else {
        return EagerSparseSoftmaxCrossEntropyMsOperator(logits, label);
      }
    } else {
      return SparseSoftmaxCrossEntropyOperator(logits, label);
    }
  }

  Maybe<bool> RunWithMsVersion(const std::shared_ptr<one::Tensor>& logits,
                               const std::shared_ptr<one::Tensor>& label) const {
    if (!(logits->is_consistent() && label->is_consistent())) { return false; }

    if (logits->shape()->NumAxes() != 2) { return false; }

    const cfg::NdSbp& logits_nd_sbp = *(JUST(logits->nd_sbp()));
    const int32_t split_axis = logits->shape()->NumAxes() - 1;
    bool has_split_axis_parallel = false;
    for (int64_t i = 0; i < logits_nd_sbp.sbp_parallel_size(); ++i) {
      const auto& sbp = logits_nd_sbp.sbp_parallel(i);
      if (sbp.has_split_parallel() && sbp.split_parallel().axis() == split_axis) {
        has_split_axis_parallel = true;
      } else {
        if (sbp.has_partial_sum_parallel()) { return false; }
      }
    }
    if (!has_split_axis_parallel) { return false; }

    return true;
  }

  Maybe<Tensor> SparseSoftmaxCrossEntropyOperator(const std::shared_ptr<one::Tensor>& logits,
                                                  const std::shared_ptr<one::Tensor>& label) const {
    MutableAttrMap attrs;
    int64_t depth = logits->shape()->At(logits->shape()->NumAxes() - 1);
    JUST(attrs.SetAttr<int64_t>("depth", depth));
    const auto& result = JUST(OpInterpUtil::Dispatch<TensorTuple>(*op_sparse_softmax_cross_entropy_,
                                                                  {logits, label}, attrs));
    return result->at(1);
  }

  Maybe<Tensor> LazySparseSoftmaxCrossEntropyMsOperator(
      const std::shared_ptr<one::Tensor>& logits, const std::shared_ptr<one::Tensor>& label) const {
    MutableAttrMap attrs;
    int64_t depth = logits->shape()->At(logits->shape()->NumAxes() - 1);
    JUST(attrs.SetAttr<int64_t>("depth", depth));
    const auto& result = JUST(OpInterpUtil::Dispatch<TensorTuple>(
        *op_sparse_softmax_cross_entropy_ms_, {logits, label}, attrs));
    return result->at(1);
  }

  Maybe<Tensor> EagerSparseSoftmaxCrossEntropyMsOperator(
      const std::shared_ptr<one::Tensor>& logits, const std::shared_ptr<one::Tensor>& label) const {
    // op_reduce_max_device_stage_
    MutableAttrMap attrs;
    int64_t depth = logits->shape()->At(logits->shape()->NumAxes() - 1);
    int32_t axis = logits->shape()->NumAxes() - 1;
    JUST(attrs.SetAttr<std::vector<int32_t>>("axis", {axis}));
    const auto& max_device_stage =
        JUST(OpInterpUtil::Dispatch<TensorTuple>(*op_reduce_max_device_stage_, {logits}, attrs));
    std::shared_ptr<Tensor> max_global_stage_input0 = max_device_stage->at(0);
    std::shared_ptr<Tensor> max_global_stage_input1 = max_device_stage->at(2);

    const cfg::NdSbp& logits_nd_sbp = *(JUST(logits->nd_sbp()));
    std::vector<Symbol<cfg::SbpParallel>> s0b_sbp_parallels;
    std::vector<Symbol<cfg::SbpParallel>> s0s1_sbp_parallels;
    if (logits_nd_sbp.sbp_parallel_size() == 2) {
      cfg::SbpParallel sbp;
      sbp.mutable_broadcast_parallel();
      s0b_sbp_parallels.emplace_back(logits_nd_sbp.sbp_parallel(0));
      s0b_sbp_parallels.emplace_back(sbp);
      s0s1_sbp_parallels.emplace_back(logits_nd_sbp.sbp_parallel(0));
      s0s1_sbp_parallels.emplace_back(logits_nd_sbp.sbp_parallel(1));
      max_global_stage_input0 = JUST(functional::ToConsistent(
          max_device_stage->at(0), JUST(max_device_stage->at(0)->parallel_desc()),
          s0b_sbp_parallels, s0s1_sbp_parallels));
      max_global_stage_input1 = JUST(functional::ToConsistent(
          max_device_stage->at(2), JUST(max_device_stage->at(0)->parallel_desc()),
          s0b_sbp_parallels, s0s1_sbp_parallels));
    }
    // op_reduce_max_global_stage_
    attrs.clear();
    JUST(attrs.SetAttr<std::vector<int32_t>>("axis", {axis}));
    JUST(attrs.SetAttr<bool>("keepdims", true));
    const auto& max_global_stage = JUST(OpInterpUtil::Dispatch<TensorTuple>(
        *op_reduce_max_global_stage_, {max_global_stage_input0, max_global_stage_input1}, attrs));
    auto& broadcast_sub_input = max_global_stage->at(0);
    if (logits_nd_sbp.sbp_parallel_size() == 2) {
      broadcast_sub_input = JUST(functional::ToConsistent(
          broadcast_sub_input, JUST(max_device_stage->at(0)->parallel_desc()), s0b_sbp_parallels,
          s0b_sbp_parallels));
    }
    // op_broadcast_sub_
    attrs.clear();
    const auto& output_broadcast_sub = JUST(OpInterpUtil::Dispatch<TensorTuple>(
        *op_broadcast_sub_, {logits, broadcast_sub_input}, attrs));
    // op_exp_
    const auto& output_exp =
        JUST(OpInterpUtil::Dispatch<TensorTuple>(*op_exp_, {output_broadcast_sub->at(0)}, attrs));
    // op_reduce_sum_
    JUST(attrs.SetAttr<std::vector<int32_t>>("axis", {axis}));
    JUST(attrs.SetAttr<bool>("keepdims", true));
    const auto& output_reduce_sum =
        JUST(OpInterpUtil::Dispatch<TensorTuple>(*op_reduce_sum_, {output_exp->at(0)}, attrs));
    std::shared_ptr<Tensor> broadcast_div_input1 = output_reduce_sum->at(0);
    if (logits_nd_sbp.sbp_parallel_size() == 2) {
      std::vector<Symbol<cfg::SbpParallel>> empty_grad_sbp_parallels;
      broadcast_div_input1 = JUST(functional::ToConsistent(
          output_reduce_sum->at(0), JUST(output_reduce_sum->at(0)->parallel_desc()),
          s0b_sbp_parallels, s0b_sbp_parallels));
    }
    // op_broadcast_div_
    attrs.clear();
    const auto& predictions = JUST(OpInterpUtil::Dispatch<TensorTuple>(
        *op_broadcast_div_, {output_exp->at(0), broadcast_div_input1}, attrs));
    // op_sparse_cross_entropy_ms_
    JUST(attrs.SetAttr<int64_t>("depth", depth));
    const auto& output = JUST(OpInterpUtil::Dispatch<Tensor>(*op_sparse_cross_entropy_ms_,
                                                             {predictions->at(0), label}, attrs));
    return output;
  }

 private:
  // SparseSoftmaxCrossEntropy
  std::shared_ptr<OpExpr> op_sparse_softmax_cross_entropy_;
  // lazy model SparseSoftmaxCrossEntropyMs
  std::shared_ptr<OpExpr> op_sparse_softmax_cross_entropy_ms_;
  // SparseSoftmaxCrossEntropyMs
  std::shared_ptr<OpExpr> op_reduce_max_device_stage_;
  std::shared_ptr<OpExpr> op_reduce_max_global_stage_;
  std::shared_ptr<OpExpr> op_broadcast_sub_;
  std::shared_ptr<OpExpr> op_exp_;
  std::shared_ptr<OpExpr> op_reduce_sum_;
  std::shared_ptr<OpExpr> op_broadcast_div_;
  std::shared_ptr<OpExpr> op_sparse_cross_entropy_ms_;
};

class SoftmaxCrossEntropyFunctor {
 public:
  SoftmaxCrossEntropyFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("softmax_cross_entropy")
                         .Input("prediction")
                         .Input("label")
                         .Output("out")
                         .Output("prob")
                         .Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& logits,
                           const std::shared_ptr<one::Tensor>& label) const {
    return OpInterpUtil::Dispatch<Tensor>(*op_, {logits, label});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class SoftmaxCrossEntropyGradFunctor {
 public:
  SoftmaxCrossEntropyGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("softmax_cross_entropy_grad")
                         .Input("dy")
                         .Input("label")
                         .Input("prob")
                         .Output("prediction_diff")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& label,
                           const std::shared_ptr<one::Tensor>& prob) const {
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, label, prob});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class CombinedMarginLossFunctor {
 public:
  CombinedMarginLossFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("combined_margin_loss")
                         .Input("x")
                         .Input("label")
                         .Output("y")
                         .Output("theta")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& label, const float& m1,
                           const float& m2, const float& m3) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<float>("m1", m1));
    JUST(attrs.SetAttr<float>("m2", m2));
    JUST(attrs.SetAttr<float>("m3", m3));
    JUST(attrs.SetAttr<int64_t>("depth", x->shape()->At(1)));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x, label}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class CtcLossFunctor {
 public:
  CtcLossFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("ctc_loss")
                         .Input("log_probs")
                         .Input("targets")
                         .Input("input_lengths")
                         .Input("target_lengths")
                         .Output("loss")
                         .Output("alpha")
                         .Build());
    op_xdivy_ = CHECK_JUST(one::OpBuilder("xdivy").Input("x").Input("y").Output("z").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& log_probs,
                           const std::shared_ptr<one::Tensor>& targets,
                           const std::shared_ptr<one::Tensor>& input_lengths,
                           const std::shared_ptr<one::Tensor>& target_lengths,
                           const int64_t& max_target_length, const int& blank,
                           const bool& zero_infinity, const std::string& reduction) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("max_target_length", max_target_length));
    JUST(attrs.SetAttr<int32_t>("blank", blank));
    JUST(attrs.SetAttr<bool>("zero_infinity", zero_infinity));
    auto out = JUST(OpInterpUtil::Dispatch<Tensor>(
        *op_, {log_probs, targets, input_lengths, target_lengths}, attrs));
    if (zero_infinity) {
      const auto create_constant = [&](const Scalar& scalar) -> Maybe<Tensor> {
        return functional::Constant(*out->shape(), scalar, out->dtype(), JUST(out->device()));
      };

      out = JUST(sequence_function(functional::Constant)
                     .then(std::bind(functional::BroadcastEqual, out, std::placeholders::_1))
                     .then(std::bind(functional::Where, std::placeholders::_1,
                                     JUST(create_constant(Scalar(0))), out))
                     .call(*out->shape(), Scalar(std::numeric_limits<double>::infinity()),
                           out->dtype(), JUST(out->device())));
    }
    CHECK_OR_RETURN([&]() -> bool {
      if ((reduction != "none") && (reduction != "sum") && (reduction != "mean")) return false;
      return true;
    }());
    if (reduction == "sum") { return functional::ReduceSum(out, {}, false); }
    if (reduction == "mean") {
      return sequence_function(functional::Clamp)
          .then(std::bind(functional::Cast, std::placeholders::_1, log_probs->dtype()))
          .then([&](const std::shared_ptr<one::Tensor>& x) {
            return OpInterpUtil::Dispatch<Tensor>(*op_xdivy_, {out, x});
          })
          .then(std::bind(functional::ReduceMean, std::placeholders::_1, std::vector<int32_t>({}),
                          false))
          .call(target_lengths, Scalar(1), NullOpt);
    }
    return out;
  }

 private:
  std::shared_ptr<OpExpr> op_;
  std::shared_ptr<OpExpr> op_xdivy_;
};

class TripletMarginLossFunctor {
 public:
  TripletMarginLossFunctor() {}

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& anchor,
                           const std::shared_ptr<one::Tensor>& positive,
                           const std::shared_ptr<one::Tensor>& negative, const float& margin,
                           const float& p, const float& eps, const bool& swap,
                           const std::string& reduction) const {
    int32_t dim_norm = anchor->ndim() - 1;
    std::vector<int32_t> dim(1, dim_norm);
    CHECK_OR_RETURN([&]() -> bool {
      if ((reduction != "none") && (reduction != "sum") && (reduction != "mean")) return false;
      return true;
    }());
    auto da_p = JUST(VectorNorm(JUST(ScalarAdd(eps, JUST(Sub(anchor, positive)), /*alpha=*/1)), p,
                                dim, /*keepdim=*/false, anchor->dtype()));
    auto da_n = JUST(VectorNorm(JUST(ScalarAdd(eps, JUST(Sub(anchor, negative)), /*alpha=*/1)), p,
                                dim, /*keepdim=*/false, anchor->dtype()));
    if (swap) {
      auto distance_swap =
          JUST(VectorNorm(JUST(ScalarAdd(eps, JUST(Sub(positive, negative)), /*alpha=*/1)), p, dim,
                          /*keepdim=*/false, positive->dtype()));
      da_n = JUST(Minimum(distance_swap, da_n));
    }
    auto triplet_loss =
        JUST(Clamp(JUST(ScalarAdd(JUST(Sub(da_p, da_n)), margin, /*alpha=*/1, /*inplace=*/false)),
                   /*min=*/0.0, NullOpt));
    int32_t ndim = triplet_loss->ndim() - 1;
    std::vector<int32_t> axis(1, ndim);

    if (reduction == "mean") {
      triplet_loss = JUST(ReduceMean(triplet_loss, axis, /*keepdim=*/false));
    } else if (reduction == "sum") {
      triplet_loss = JUST(ReduceSum(triplet_loss, axis, /*keepdim=*/false));
    }
    return triplet_loss;
  }
};

class AffineGridFunctor {
 public:
  AffineGridFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("affine_grid").Input("theta").Output("grid").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& theta, const Shape& size,
                           const bool& align_corners) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<Shape>("size", size));
    JUST(attrs.SetAttr<bool>("align_corners", align_corners));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {theta}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class GridSampleFunctor {
 public:
  GridSampleFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("grid_sample").Input("input").Input("grid").Output("output").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& grid,
                           const std::string& interpolation_mode, const std::string& padding_mode,
                           const bool& align_corners) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::string>("interpolation_mode", interpolation_mode));
    JUST(attrs.SetAttr<std::string>("padding_mode", padding_mode));
    JUST(attrs.SetAttr<bool>("align_corners", align_corners));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {input, grid}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class NormalizationFunctor {
 public:
  NormalizationFunctor() {
    norm_eval_op_ = CHECK_JUST(one::OpBuilder("normalization")
                                   .Input("x")
                                   .Input("moving_mean")
                                   .Input("moving_variance")
                                   .Input("gamma")
                                   .Input("beta")
                                   .Output("y")
                                   .Attr("training", false)
                                   .Build());
    norm_training_stats_op_ = CHECK_JUST(one::OpBuilder("normalization")
                                             .Input("x")
                                             .Input("moving_mean")
                                             .Input("moving_variance")
                                             .Input("gamma")
                                             .Input("beta")
                                             .Output("y")
                                             .Output("mean")
                                             .Output("inv_variance")
                                             .Attr("training", true)
                                             .Build());
    norm_training_no_stats_op_ = CHECK_JUST(one::OpBuilder("normalization")
                                                .Input("x")
                                                .Input("gamma")
                                                .Input("beta")
                                                .Output("y")
                                                .Output("mean")
                                                .Output("inv_variance")
                                                .Attr("training", true)
                                                .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const Optional<one::Tensor>& moving_mean,
                           const Optional<one::Tensor>& moving_variance,
                           const std::shared_ptr<one::Tensor>& gamma,
                           const std::shared_ptr<one::Tensor>& beta, const int32_t& axis,
                           const float& epsilon, const float& momentum,
                           const bool& training) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("axis", axis));
    JUST(attrs.SetAttr<float>("epsilon", epsilon));
    // convert torch momentum to tensorflow momentum
    JUST(attrs.SetAttr<float>("momentum", 1.0 - momentum));

    CHECK_OR_RETURN((moving_mean && moving_variance) || (!moving_mean && !moving_variance))
        << "Both moving_mean and moving_variance should be None or Tensor.";
    if (!training) {
      CHECK_OR_RETURN(moving_mean && moving_variance)
          << "Must have moving_mean and moving_variance in eval mode.";
      return OpInterpUtil::Dispatch<one::Tensor>(
          *norm_eval_op_, {x, JUST(moving_mean), JUST(moving_variance), gamma, beta}, attrs);
    }
    if (moving_mean) {
      return OpInterpUtil::Dispatch<one::Tensor>(
          *norm_training_stats_op_, {x, JUST(moving_mean), JUST(moving_variance), gamma, beta},
          attrs);
    }
    return OpInterpUtil::Dispatch<one::Tensor>(*norm_training_no_stats_op_, {x, gamma, beta},
                                               attrs);
  }

 private:
  std::shared_ptr<OpExpr> norm_eval_op_;
  std::shared_ptr<OpExpr> norm_training_stats_op_;
  std::shared_ptr<OpExpr> norm_training_no_stats_op_;
};

class NormalizationAddReluFunctor {
 public:
  NormalizationAddReluFunctor() {
    norm_eval_op_ = CHECK_JUST(one::OpBuilder("normalization")
                                   .Input("x")
                                   .Input("moving_mean")
                                   .Input("moving_variance")
                                   .Input("gamma")
                                   .Input("beta")
                                   .Output("y")
                                   .Attr("training", false)
                                   .Build());
    relu_op_ = CHECK_JUST(one::OpBuilder("relu").Input("x").Output("y").Build());
    add_op_ = CHECK_JUST(one::OpBuilder("add_n").Input("in", 2).Output("out").Build());
    fused_norm_training_stats_op_ = CHECK_JUST(one::OpBuilder("normalization_add_relu")
                                                   .Input("x")
                                                   .Input("moving_mean")
                                                   .Input("moving_variance")
                                                   .Input("gamma")
                                                   .Input("beta")
                                                   .Output("y")
                                                   .Output("reserve_space")
                                                   .Output("mean")
                                                   .Output("inv_variance")
                                                   .Attr("training", true)
                                                   .Build());
    fused_addend_norm_training_stats_op_ = CHECK_JUST(one::OpBuilder("normalization_add_relu")
                                                          .Input("x")
                                                          .Input("addend")
                                                          .Input("moving_mean")
                                                          .Input("moving_variance")
                                                          .Input("gamma")
                                                          .Input("beta")
                                                          .Output("y")
                                                          .Output("reserve_space")
                                                          .Output("mean")
                                                          .Output("inv_variance")
                                                          .Attr("training", true)
                                                          .Build());
    fused_norm_training_no_stats_op_ = CHECK_JUST(one::OpBuilder("normalization_add_relu")
                                                      .Input("x")
                                                      .Input("gamma")
                                                      .Input("beta")
                                                      .Output("y")
                                                      .Output("reserve_space")
                                                      .Output("mean")
                                                      .Output("inv_variance")
                                                      .Attr("training", true)
                                                      .Build());
    fused_addend_norm_training_no_stats_op_ = CHECK_JUST(one::OpBuilder("normalization_add_relu")
                                                             .Input("x")
                                                             .Input("addend")
                                                             .Input("gamma")
                                                             .Input("beta")
                                                             .Output("y")
                                                             .Output("reserve_space")
                                                             .Output("mean")
                                                             .Output("inv_variance")
                                                             .Attr("training", true)
                                                             .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const Optional<one::Tensor>& addend,
                           const Optional<one::Tensor>& moving_mean,
                           const Optional<one::Tensor>& moving_variance,
                           const std::shared_ptr<one::Tensor>& gamma,
                           const std::shared_ptr<one::Tensor>& beta, const int32_t& axis,
                           const float& epsilon, const float& momentum,
                           const bool& is_training) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("axis", axis));
    JUST(attrs.SetAttr<float>("epsilon", epsilon));
    // convert torch momentum to tensorflow momentum
    JUST(attrs.SetAttr<float>("momentum", 1.0f - momentum));

    CHECK_OR_RETURN((moving_mean && moving_variance) || (!moving_mean && !moving_variance))
        << "Both moving_mean and moving_variance should be None or Tensor.";
    if (!is_training) {
      CHECK_OR_RETURN(moving_mean && moving_variance)
          << "Must have moving_mean and moving_variance in eval mode.";
      const auto& normalize_result = JUST(OpInterpUtil::Dispatch<one::Tensor>(
          *norm_eval_op_, {x, JUST(moving_mean), JUST(moving_variance), gamma, beta}, attrs));
      if (addend) {
        const auto& add_result =
            JUST(OpInterpUtil::Dispatch<one::Tensor>(*add_op_, {normalize_result, JUST(addend)}));
        return OpInterpUtil::Dispatch<one::Tensor>(*relu_op_, {add_result});
      } else {
        return OpInterpUtil::Dispatch<one::Tensor>(*relu_op_, {normalize_result});
      }
    } else if (moving_mean) {
      if (addend) {
        return OpInterpUtil::Dispatch<one::Tensor>(
            *fused_addend_norm_training_stats_op_,
            {x, JUST(addend), JUST(moving_mean), JUST(moving_variance), gamma, beta}, attrs);
      } else {
        return OpInterpUtil::Dispatch<one::Tensor>(
            *fused_norm_training_stats_op_,
            {x, JUST(moving_mean), JUST(moving_variance), gamma, beta}, attrs);
      }
    } else {
      if (addend) {
        return OpInterpUtil::Dispatch<one::Tensor>(*fused_addend_norm_training_no_stats_op_,
                                                   {x, JUST(addend), gamma, beta}, attrs);
      } else {
        return OpInterpUtil::Dispatch<one::Tensor>(*fused_norm_training_no_stats_op_,
                                                   {x, gamma, beta}, attrs);
      }
    }
  }

 private:
  std::shared_ptr<OpExpr> norm_eval_op_;
  std::shared_ptr<OpExpr> relu_op_;
  std::shared_ptr<OpExpr> add_op_;
  std::shared_ptr<OpExpr> fused_norm_training_stats_op_;
  std::shared_ptr<OpExpr> fused_addend_norm_training_stats_op_;
  std::shared_ptr<OpExpr> fused_norm_training_no_stats_op_;
  std::shared_ptr<OpExpr> fused_addend_norm_training_no_stats_op_;
};

class PadFunctor {
 public:
  PadFunctor() {
    pad_ = CHECK_JUST(one::OpBuilder("pad").Input("x").Output("y").Build());
    reflect_pad_ = CHECK_JUST(one::OpBuilder("reflection_pad2d").Input("x").Output("y").Build());
    replicate_pad_ = CHECK_JUST(one::OpBuilder("replication_pad2d").Input("x").Output("y").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const std::vector<int64_t>& pad,
                           const std::string& mode, const Scalar& value) const {
    const int64_t ndim = x->shape()->NumAxes();
    CHECK_LE_OR_RETURN(ndim, 5) << "Dimension of input tensor should less than or equal to 5";
    CHECK_LE_OR_RETURN(pad.size(), 2 * ndim)
        << "Pad size should less than or equal to input axes * 2.";
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::vector<int64_t>>("padding", pad));
    if (mode == "constant") {
      CHECK_EQ_OR_RETURN(pad.size() % 2, 0)
          << "Length of pad must be even but instead it equals " << pad.size();
      if (IsFloatingDataType(x->dtype()->data_type())) {
        JUST(attrs.SetAttr<double>("floating_constant_value", JUST(value.As<double>())));
        JUST(attrs.SetAttr<int64_t>("integral_constant_value", 0));
      } else if (IsIntegralDataType(x->dtype()->data_type())) {
        JUST(attrs.SetAttr<double>("floating_constant_value", 0));
        JUST(attrs.SetAttr<int64_t>("integral_constant_value", JUST(value.As<int64_t>())));
      } else {
        UNIMPLEMENTED_THEN_RETURN() << "Data type should be floating or integral type.";
      }

      std::vector<int64_t> pad_before(ndim, 0);
      std::vector<int64_t> pad_after(ndim, 0);
      const int64_t pad_pair = pad.size() / 2;
      for (int64_t i = 0; i < pad_pair; ++i) {
        pad_before[ndim - i - 1] = pad[2 * i];
        pad_after[ndim - i - 1] = pad[2 * i + 1];
      }
      JUST(attrs.SetAttr<std::vector<int64_t>>("padding_before", pad_before));
      JUST(attrs.SetAttr<std::vector<int64_t>>("padding_after", pad_after));
      return OpInterpUtil::Dispatch<Tensor>(*pad_, {x}, attrs);

    } else if (mode == "reflect") {
      const int64_t pad_h = x->shape()->dim_vec().at(2);
      const int64_t pad_w = x->shape()->dim_vec().at(3);
      CHECK_OR_RETURN(pad[2] < pad_h && pad[3] < pad_h && pad[0] < pad_w && pad[1] < pad_w)
          << "padding size should be less than the corresponding input dimension!";
      return OpInterpUtil::Dispatch<Tensor>(*reflect_pad_, {x}, attrs);
    } else if (mode == "replicate") {
      return OpInterpUtil::Dispatch<Tensor>(*replicate_pad_, {x}, attrs);
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "Pad mode is " << mode
                                  << ", but only constant, reflect and replicate are valid.";
    }
  }

 private:
  std::shared_ptr<OpExpr> pad_;
  std::shared_ptr<OpExpr> reflect_pad_;
  std::shared_ptr<OpExpr> replicate_pad_;
};

class DropoutFunctor {
 public:
  DropoutFunctor() {
    dropout_op_ =
        CHECK_JUST(one::OpBuilder("dropout").Input("in").Output("out").Output("mask").Build());
    dropout_addend_op_ = CHECK_JUST(one::OpBuilder("dropout")
                                        .Input("in")
                                        .Input("_add_to_output")
                                        .Output("out")
                                        .Output("mask")
                                        .Build());
    add_op_ = CHECK_JUST(one::OpBuilder("add_n").Input("in", 2).Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const Optional<one::Tensor>& addend, const float& p,
                           const bool& training, const Optional<one::Generator>& generator) const {
    const auto gen = generator.value_or(JUST(one::DefaultAutoGenerator()));
    const auto& dropout_state = std::make_shared<FusedDropoutKernelState>(gen);
    MutableAttrMap dropout_attrs;
    JUST(dropout_attrs.SetAttr<float>("rate", p));
    if (addend) {
      if ((!training) || p == 0.0) {
        return OpInterpUtil::Dispatch<Tensor>(*add_op_, {x, JUST(addend)});
      } else {
        return OpInterpUtil::Dispatch<Tensor>(*dropout_addend_op_, {x, JUST(addend)},
                                              OpExprInterpContext(dropout_attrs, dropout_state));
      }
    } else {
      if (!training || p == 0.0) {
        return x;
      } else {
        return OpInterpUtil::Dispatch<Tensor>(*dropout_op_, {x},
                                              OpExprInterpContext(dropout_attrs, dropout_state));
      }
    }
  }

 private:
  std::shared_ptr<OpExpr> dropout_op_;
  std::shared_ptr<OpExpr> dropout_addend_op_;
  std::shared_ptr<OpExpr> add_op_;
};

class DropoutGradFunctor {
 public:
  DropoutGradFunctor() {
    dropout_grad_op_ =
        CHECK_JUST(one::OpBuilder("dropout_grad").Input("dy").Input("mask").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& mask, const float& scale) const {
    MutableAttrMap dropout_grad_attrs;
    JUST(dropout_grad_attrs.SetAttr<float>("scale", scale));
    return OpInterpUtil::Dispatch<Tensor>(*dropout_grad_op_, {dy, mask}, dropout_grad_attrs);
  }

 private:
  std::shared_ptr<OpExpr> dropout_grad_op_;
};

class AvgPoolingNDFunctor {
 public:
  AvgPoolingNDFunctor() = default;
  virtual ~AvgPoolingNDFunctor() = default;
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::vector<int32_t>& kernel_size,
                           const Optional<std::vector<int32_t>>& stride,
                           const std::vector<int32_t>& padding, const bool& ceil_mode,
                           const bool& count_include_pad, const int64_t& divisor_override,
                           const std::string& data_format) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    JUST(attrs.SetAttr<std::vector<int32_t>>("padding", padding));
    JUST(attrs.SetAttr<std::vector<int32_t>>("kernel_size", kernel_size));
    if (stride.has_value()) {
      JUST(attrs.SetAttr<std::vector<int32_t>>("stride", *JUST(stride)));
    } else {
      JUST(attrs.SetAttr<std::vector<int32_t>>(
          "stride", kernel_size));  // If stride is None, we set it as kernel_size to align Pytorch.
    }
    JUST(attrs.SetAttr<bool>("ceil_mode", ceil_mode));
    JUST(attrs.SetAttr<bool>("count_include_pad", count_include_pad));
    JUST(attrs.SetAttr<int64_t>("divisor_override", divisor_override));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 protected:
  std::shared_ptr<OpExpr> op_;
};

class Avgpool1DFunctor : public AvgPoolingNDFunctor {
 public:
  Avgpool1DFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("avgpool_1d").Input("x").Output("y").Build());
  }
};

class Avgpool2DFunctor : public AvgPoolingNDFunctor {
 public:
  Avgpool2DFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("avgpool_2d").Input("x").Output("y").Build());
  }
};

class Avgpool3DFunctor : public AvgPoolingNDFunctor {
 public:
  Avgpool3DFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("avgpool_3d").Input("x").Output("y").Build());
  }
};

class UnfoldFunctor {
 public:
  UnfoldFunctor() {
    unfold_op_ = CHECK_JUST(one::OpBuilder("unfold").Input("x").Output("y").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const std::string& data_format,
                           const std::vector<int32_t>& kernel_size,
                           const std::vector<int32_t>& dilation_rate,
                           const std::vector<int32_t>& padding,
                           const std::vector<int32_t>& strides) const {
    const auto& x_shape = x->shape();
    // Only Support 4d tensor now.
    CHECK_EQ_OR_RETURN(x_shape->NumAxes(), 4) << "Input Tensor dim should == 4";
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    JUST(attrs.SetAttr<std::vector<int32_t>>("kernel_size", kernel_size));
    JUST(attrs.SetAttr<std::vector<int32_t>>("dilation_rate", dilation_rate));
    JUST(attrs.SetAttr<std::vector<int32_t>>("padding", padding));
    JUST(attrs.SetAttr<std::vector<int32_t>>("strides", strides));

    return OpInterpUtil::Dispatch<Tensor>(*unfold_op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> unfold_op_;
};

class FoldFunctor {
 public:
  FoldFunctor() { fold_op_ = CHECK_JUST(one::OpBuilder("fold").Input("x").Output("y").Build()); }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const std::string& data_format,
                           const std::vector<int32_t>& output_size,
                           const std::vector<int32_t>& kernel_size,
                           const std::vector<int32_t>& dilation_rate,
                           const std::vector<int32_t>& padding,
                           const std::vector<int32_t>& strides) const {
    const auto& x_shape = x->shape();
    // Only Support 3d tensor fold now. format is (N, C*K*K, L)
    CHECK_EQ_OR_RETURN(x_shape->NumAxes(), 3) << "Input Tensor dim should == 3";
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    JUST(attrs.SetAttr<std::vector<int32_t>>("output_size", output_size));
    JUST(attrs.SetAttr<std::vector<int32_t>>("kernel_size", kernel_size));
    JUST(attrs.SetAttr<std::vector<int32_t>>("dilation_rate", dilation_rate));
    JUST(attrs.SetAttr<std::vector<int32_t>>("padding", padding));
    JUST(attrs.SetAttr<std::vector<int32_t>>("strides", strides));

    return OpInterpUtil::Dispatch<Tensor>(*fold_op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> fold_op_;
};

class OneHotFunctor {
 public:
  OneHotFunctor() {
    one_hot_op_ = CHECK_JUST(one::OpBuilder("one_hot").Input("indices").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input, const int64_t& num_classes,
                           const Scalar& on_value, const Scalar& off_value) const {
    if (IsFloatingDataType(input->dtype()->data_type())) {
      OF_RUNTIME_ERROR() << "one_hot is only applicable to index tensor.";
    }
    MutableAttrMap attrs;
    if (num_classes == -1) {
      std::vector<int32_t> axis(input->ndim());
      std::iota(axis.begin(), axis.end(), 0);
      auto tensor_max = JUST(functional::ReduceMax(input, axis, false));

      int64_t max = 0;
      const auto& callback =
          std::make_shared<std::function<void(uint64_t)>>([&](uint64_t of_blob_ptr) {
            auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
            of_blob->AutoMemCopyTo<int64_t>(&max,
                                            1);  // copy 1 scalar(int64_t) tensor's value to max
          });
      JUST(SyncAccessTensorWithTimeOut(tensor_max, callback, "const"));
      JUST(attrs.SetAttr<int64_t>("depth", max + 1));

    } else {
      JUST(attrs.SetAttr<int64_t>("depth", num_classes));
    }
    // Refer to: https://github.com/Oneflow-Inc/oneflow/pull/5315/files#r755823506
    bool is_on_value_double = on_value.IsFloatingPoint();
    bool is_off_value_double = off_value.IsFloatingPoint();
    if (is_on_value_double || is_off_value_double) {
      JUST(attrs.SetAttr<DataType>("dtype", kFloat));
      JUST(attrs.SetAttr<double>("floating_on_value", JUST(on_value.As<double>())));
      JUST(attrs.SetAttr<double>("floating_off_value", JUST(off_value.As<double>())));
      JUST(attrs.SetAttr<int64_t>("integer_on_value", 0));
      JUST(attrs.SetAttr<int64_t>("integer_off_value", 0));
    } else {
      JUST(attrs.SetAttr<DataType>("dtype", kInt64));
      JUST(attrs.SetAttr<double>("floating_on_value", 0));
      JUST(attrs.SetAttr<double>("floating_off_value", 0));
      JUST(attrs.SetAttr<int64_t>("integer_on_value", JUST(on_value.As<int64_t>())));
      JUST(attrs.SetAttr<int64_t>("integer_off_value", JUST(off_value.As<int64_t>())));
    }
    return OpInterpUtil::Dispatch<Tensor>(*one_hot_op_, {input}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> one_hot_op_;
};

class L2NormalizeFunctor {
 public:
  L2NormalizeFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("l2_normalize").Input("x").Output("y").Output("square_x_sum").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input, const int32_t& axis,
                           const float& epsilon) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("axis", 0));
    JUST(attrs.SetAttr<float>("epsilon", epsilon));

    if (axis != 0) {
      std::vector<int> input_perm(input->shape()->dim_vec().size(), 0);
      for (size_t i = 0; i < input_perm.size(); ++i) { input_perm[i] = static_cast<int>(i); }
      std::swap(input_perm[0], input_perm[static_cast<size_t>(axis)]);

      const auto result = JUST(OpInterpUtil::Dispatch<TensorTuple>(
          *op_, {JUST(functional::Transpose(input, input_perm))}, attrs));
      return functional::Transpose(result->at(0), input_perm);
    }

    return OpInterpUtil::Dispatch<Tensor>(*op_, {input}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class NormalizeFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input, const float& p,
                           const int32_t& dim, const float& eps) const {
    return SequenceFunction<Maybe<Tensor>(const std::shared_ptr<Tensor>&, const float&,
                                          const int32_t&)>(
               [](const auto& x, const float& p, const int32_t& dim) -> Maybe<Tensor> {
                 return functional::ScalarNorm(x, p, dim, true, NullOpt);
               })
        .then([&](const auto& x) { return functional::Clamp(x, eps, NullOpt); })
        .then([&](const auto& x) { return functional::Div(input, x); })
        .call(input, p, dim);
  }
};

class FusedSelfAttentionFunctor {
 public:
  FusedSelfAttentionFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("fused_self_attention_query_mul_key_and_value")
                         .Input("hidden_states")
                         .Output("query_mul_key")
                         .Output("value")
                         .Build());
  }
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& hidden_states,
                                const int64_t& head_size, const float& alpha) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("head_size", head_size));
    JUST(attrs.SetAttr<float>("alpha", alpha));
    return OpInterpUtil::Dispatch<TensorTuple>(*op_, {hidden_states}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class FusedSelfAttentionGradFunctor {
 public:
  FusedSelfAttentionGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("fused_self_attention_query_mul_key_and_value_grad")
                         .Input("query_mul_key_grad")
                         .Input("value_grad")
                         .Input("hidden_states")
                         .Output("hidden_states_grad")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& query_mul_key_grad,
                           const std::shared_ptr<one::Tensor>& value_grad,
                           const std::shared_ptr<one::Tensor>& hidden_states,
                           const float& alpha) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<float>("alpha", alpha));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {query_mul_key_grad, value_grad, hidden_states},
                                          attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class FusedScaleTrilSoftmaxMaskScaleFunctor {
 public:
  FusedScaleTrilSoftmaxMaskScaleFunctor() {
    random_mask_like_op_ =
        CHECK_JUST(one::OpBuilder("random_mask_like").Input("like").Output("out").Build());
    fused_op_ = CHECK_JUST(one::OpBuilder("fused_tril_scale_softmax_mask_scale")
                               .Input("x")
                               .Input("mask")
                               .Output("y")
                               .Output("softmax_y")
                               .Build());
  }
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& x, const float p,
                                const int64_t diagonal, const float tril_scale_value,
                                const Optional<one::Generator>& generator) const {
    const auto gen = generator.value_or(JUST(one::DefaultAutoGenerator()));
    MutableAttrMap random_mask_like_attrs;
    JUST(random_mask_like_attrs.SetAttr<float>("rate", p));
    JUST(random_mask_like_attrs.SetAttr<int64_t>("seed", gen->current_seed()));
    const auto& random_mask_like_state = std::make_shared<RandomMaskLikeKernelState>(gen);

    const auto& mask = JUST(OpInterpUtil::Dispatch<Tensor>(
        *random_mask_like_op_, {x},
        OpExprInterpContext(random_mask_like_attrs, random_mask_like_state)));

    float mask_scale_value = 1.0;
    if (p != 1.0) { mask_scale_value = 1.0 / (1.0 - p); }
    MutableAttrMap fused_attrs;
    JUST(fused_attrs.SetAttr<int64_t>("diagonal", diagonal));
    JUST(fused_attrs.SetAttr<float>("tril_scale_value", tril_scale_value));
    JUST(fused_attrs.SetAttr<float>("mask_scale_value", mask_scale_value));

    return OpInterpUtil::Dispatch<TensorTuple>(*fused_op_, {x, mask}, fused_attrs);
  }

 private:
  std::shared_ptr<OpExpr> fused_op_;
  std::shared_ptr<OpExpr> random_mask_like_op_;
};

class L2NormalizeGradFunctor {
 public:
  L2NormalizeGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("l2_normalize_grad")
                         .Input("dy")
                         .Input("y")
                         .Input("square_x_sum")
                         .Output("dx")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& y,
                           const std::shared_ptr<one::Tensor>& square_x_sum, const int32_t& axis,
                           const float& epsilon) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("axis", axis));
    JUST(attrs.SetAttr<float>("epsilon", epsilon));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, y, square_x_sum}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class FusedBiasAddGeluFunctor {
 public:
  FusedBiasAddGeluFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("fused_bias_add_gelu").Input("a").Input("b").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& a,
                           const std::shared_ptr<one::Tensor>& b, const int32_t& axis) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("axis", axis));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {a, b}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class FusedBiasAddGeluGradFunctor {
 public:
  FusedBiasAddGeluGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("fused_bias_add_gelu_grad")
                         .Input("a")
                         .Input("b")
                         .Input("dy")
                         .Output("dx")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& a,
                           const std::shared_ptr<one::Tensor>& b,
                           const std::shared_ptr<one::Tensor>& dy, const int32_t& axis) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("axis", axis));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {a, b, dy}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class FusedBiasAddDropoutFunctor {
 public:
  FusedBiasAddDropoutFunctor() {
    random_mask_like_op_ =
        CHECK_JUST(one::OpBuilder("random_mask_like").Input("like").Output("out").Build());
    fused_bias_add_mask_scale_op_ = CHECK_JUST(one::OpBuilder("fused_bias_add_mask_scale")
                                                   .Input("a")
                                                   .Input("b")
                                                   .Input("mask")
                                                   .Output("out")
                                                   .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& a,
                           const std::shared_ptr<one::Tensor>& b, const float& p,
                           const int32_t& axis, const Optional<one::Generator>& generator) const {
    const auto gen = generator.value_or(JUST(one::DefaultAutoGenerator()));
    MutableAttrMap random_mask_like_attrs;
    JUST(random_mask_like_attrs.SetAttr<float>("rate", p));
    JUST(random_mask_like_attrs.SetAttr<int64_t>("seed", gen->current_seed()));
    const auto& random_mask_like_state = std::make_shared<RandomMaskLikeKernelState>(gen);

    float scale = 0.0;
    if (p != 1.0) { scale = 1.0 / (1.0 - p); }
    MutableAttrMap fused_bias_add_mask_attrs;
    JUST(fused_bias_add_mask_attrs.SetAttr<float>("scale", scale));
    int32_t axis_val = axis;
    if (axis_val < 0) {
      const int64_t num_axes = a->shape()->NumAxes();
      axis_val += num_axes;
    }
    JUST(fused_bias_add_mask_attrs.SetAttr<int32_t>("axis", axis_val));
    if (p >= 0.0) {
      return SequenceFunction<Maybe<Tensor>()>([&]() -> Maybe<Tensor> {
               return OpInterpUtil::Dispatch<Tensor>(
                   *random_mask_like_op_, {a},
                   OpExprInterpContext(random_mask_like_attrs, random_mask_like_state));
             })
          .then([&](const std::shared_ptr<one::Tensor>& x) {
            return OpInterpUtil::Dispatch<Tensor>(*fused_bias_add_mask_scale_op_, {a, b, x},
                                                  fused_bias_add_mask_attrs);
          })
          .call();
    } else {
      return functional::BiasAdd(a, b, axis_val);
    }
  }

 private:
  std::shared_ptr<OpExpr> random_mask_like_op_;
  std::shared_ptr<OpExpr> fused_bias_add_mask_scale_op_;
};

class FusedScaleTrilFunctor {
 public:
  FusedScaleTrilFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("fused_scale_tril").Input("in").Output("out").Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const int64_t& diagonal,
                           const Scalar& fill_value, const Scalar& scale) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("diagonal", diagonal));
    bool is_fill_value_double = fill_value.IsFloatingPoint();
    bool is_scale_double = scale.IsFloatingPoint();
    if (is_fill_value_double) {
      JUST(attrs.SetAttr<double>("floating_fill_value", JUST(fill_value.As<double>())));
      JUST(attrs.SetAttr<int64_t>("integer_fill_value", 0));
      JUST(attrs.SetAttr<bool>("is_floating_fill_value", true));
    } else {
      JUST(attrs.SetAttr<double>("floating_fill_value", 0));
      JUST(attrs.SetAttr<int64_t>("integer_fill_value", JUST(fill_value.As<int64_t>())));
      JUST(attrs.SetAttr<bool>("is_floating_fill_value", false));
    }

    if (is_scale_double) {
      JUST(attrs.SetAttr<double>("floating_scale_value", JUST(scale.As<double>())));
      JUST(attrs.SetAttr<int64_t>("integer_scale_value", 0));
      JUST(attrs.SetAttr<bool>("is_floating_scale_value", true));
    } else {
      JUST(attrs.SetAttr<double>("floating_scale_value", 0));
      JUST(attrs.SetAttr<int64_t>("integer_scale_value", JUST(scale.As<int64_t>())));
      JUST(attrs.SetAttr<bool>("is_floating_scale_value", false));
    }
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class FusedScaleMaskSoftmaxFunctor {
 public:
  FusedScaleMaskSoftmaxFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("fused_scale_mask_softmax").Input("x").Input("mask").Output("y").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& mask, const float& fill_value,
                           const float& scale) const {
    MutableAttrMap attrs_;
    JUST(attrs_.SetAttr<float>("scale_value", scale));
    JUST(attrs_.SetAttr<float>("mask_fill_value", fill_value));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x, mask}, attrs_);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class FusedScaleMaskSoftmaxDropoutFunctor {
 public:
  FusedScaleMaskSoftmaxDropoutFunctor() {
    random_mask_like_op_ =
        CHECK_JUST(one::OpBuilder("random_mask_like").Input("like").Output("out").Build());
    fused_scale_mask_softmax_dropout_op_ =
        CHECK_JUST(one::OpBuilder("fused_scale_mask_softmax_dropout")
                       .Input("x")
                       .Input("mask")
                       .Input("dropout_mask")
                       .Output("y")
                       .Output("softmax_y")
                       .Build());
  }
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& x,
                                const std::shared_ptr<one::Tensor>& mask, const float& fill_value,
                                const float& scale, const float& p, const bool& training,
                                const Optional<one::Generator>& generator) const {
    float rate = p;
    if (!training) rate = 0.0;
    const auto gen = generator.value_or(JUST(one::DefaultAutoGenerator()));
    MutableAttrMap random_mask_like_attrs;
    JUST(random_mask_like_attrs.SetAttr<float>("rate", rate));
    JUST(random_mask_like_attrs.SetAttr<int64_t>("seed", gen->current_seed()));
    const auto& random_mask_like_state = std::make_shared<RandomMaskLikeKernelState>(gen);

    const auto& dropout_mask = JUST(OpInterpUtil::Dispatch<Tensor>(
        *random_mask_like_op_, {x},
        OpExprInterpContext(random_mask_like_attrs, random_mask_like_state)));

    float dropout_scale = 0.0;
    if (rate != 1.0) { dropout_scale = 1.0 / (1.0 - rate); }
    MutableAttrMap fused_scale_mask_softmax_dropout_attrs;
    JUST(fused_scale_mask_softmax_dropout_attrs.SetAttr<float>("scale_value", scale));
    JUST(fused_scale_mask_softmax_dropout_attrs.SetAttr<float>("mask_fill_value", fill_value));
    JUST(fused_scale_mask_softmax_dropout_attrs.SetAttr<float>("dropout_scale_value",
                                                               dropout_scale));

    return OpInterpUtil::Dispatch<TensorTuple>(*fused_scale_mask_softmax_dropout_op_,
                                               {x, mask, dropout_mask},
                                               fused_scale_mask_softmax_dropout_attrs);
  }

 private:
  std::shared_ptr<OpExpr> random_mask_like_op_;
  std::shared_ptr<OpExpr> fused_scale_mask_softmax_dropout_op_;
};

class CtcGreedyDecoderFunctor {
 public:
  CtcGreedyDecoderFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("ctc_greedy_decoder")
                         .Input("log_probs")
                         .Input("input_lengths")
                         .Output("decoded")
                         .Output("neg_sum_logits")
                         .Build());
  }
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& log_probs,
                                const std::shared_ptr<one::Tensor>& input_lengths,
                                const bool& merge_repeated) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<bool>("merge_repeated", merge_repeated));
    return OpInterpUtil::Dispatch<TensorTuple>(*op_, {log_probs, input_lengths}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class PartialFCSampleFunctor {
 public:
  PartialFCSampleFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("distributed_partial_fc_sample")
                         .Input("weight")
                         .Input("label")
                         .Output("mapped_label")
                         .Output("sampled_label")
                         .Output("sampled_weight")
                         .Build());
  }
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& wegiht,
                                const std::shared_ptr<one::Tensor>& label,
                                const int64_t& num_sample) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("num_sample", num_sample));
    return OpInterpUtil::Dispatch<TensorTuple>(*op_, {wegiht, label}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class PariticalFCSampleDisableBoxing {
 public:
  PariticalFCSampleDisableBoxing() {
    op_ = CHECK_JUST(one::OpBuilder("distributed_partial_fc_sample_disable_boxing")
                         .Input("sampled_weight_diff")
                         .Input("sampled_label")
                         .Output("boxing_disabled_sampled_weight_diff")
                         .Output("boxing_disabled_sampled_label")
                         .Build());
  }
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& sampled_weight_diff,
                                const std::shared_ptr<one::Tensor>& sampled_label) const {
    return OpInterpUtil::Dispatch<TensorTuple>(*op_, {sampled_weight_diff, sampled_label});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class NmsFunctor {
 public:
  NmsFunctor() { op_ = CHECK_JUST(one::OpBuilder("nms").Input("in").Output("out").Build()); }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const float& iou_threshold,
                           const int32_t& keep_n) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<float>("iou_threshold", iou_threshold));
    JUST(attrs.SetAttr<int32_t>("keep_n", keep_n));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class RoiAlignFunctor {
 public:
  RoiAlignFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("roi_align").Input("x").Input("rois").Output("y").Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& rois, const float& spatial_scale,
                           const int32_t& pooled_h, const int32_t& pooled_w,
                           const int32_t& sampling_ratio, const bool& aligned) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<float>("spatial_scale", spatial_scale));
    JUST(attrs.SetAttr<int32_t>("pooled_h", pooled_h));
    JUST(attrs.SetAttr<int32_t>("pooled_w", pooled_w));
    JUST(attrs.SetAttr<int32_t>("sampling_ratio", sampling_ratio));
    JUST(attrs.SetAttr<bool>("aligned", aligned));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x, rois}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class RoiAlignGradFunctor {
 public:
  RoiAlignGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("roi_align_grad")
                         .Input("dy")
                         .Input("x_like")
                         .Input("rois")
                         .Output("dx")
                         .Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& x_like,
                           const std::shared_ptr<one::Tensor>& rois, const float& spatial_scale,
                           const int32_t& pooled_h, const int32_t& pooled_w,
                           const int32_t& sampling_ratio, const bool& aligned) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<float>("spatial_scale", spatial_scale));
    JUST(attrs.SetAttr<int32_t>("pooled_h", pooled_h));
    JUST(attrs.SetAttr<int32_t>("pooled_w", pooled_w));
    JUST(attrs.SetAttr<int32_t>("sampling_ratio", sampling_ratio));
    JUST(attrs.SetAttr<bool>("aligned", aligned));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, x_like, rois}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::BiasAddFunctor>("BiasAdd");
  m.add_functor<impl::Conv1dFunctor>("Conv1d");
  m.add_functor<impl::Conv2dFunctor>("Conv2d");
  m.add_functor<impl::Conv3dFunctor>("Conv3d");
  m.add_functor<impl::DeConv1dFunctor>("Deconv1d");
  m.add_functor<impl::DeConv2dFunctor>("Deconv2d");
  m.add_functor<impl::DeConv3dFunctor>("Deconv3d");
  m.add_functor<impl::MatMulFunctor>("MatMul");
  m.add_functor<impl::BatchMatMulFunctor>("BatchMatMul");
  m.add_functor<impl::LayerNormFunctor>("LayerNorm");
  m.add_functor<impl::LayerNormAffineFunctor>("LayerNormAffine");
  m.add_functor<impl::TFAvgPool2DFunctor>("AvgPool2D");
  m.add_functor<impl::Maxpool1DFunctor>("Maxpool1D");
  m.add_functor<impl::Maxpool2DFunctor>("Maxpool2D");
  m.add_functor<impl::Maxpool3DFunctor>("Maxpool3D");
  m.add_functor<impl::TFMaxPool2DFunctor>("MaxPool2D");
  m.add_functor<impl::AdaptiveAvgPool1DFunctor>("AdaptiveAvgPool1D");
  m.add_functor<impl::AdaptiveAvgPool2DFunctor>("AdaptiveAvgPool2D");
  m.add_functor<impl::AdaptiveAvgPool3DFunctor>("AdaptiveAvgPool3D");
  m.add_functor<impl::L1LossFunctor>("L1Loss");
  m.add_functor<impl::MseLossFunctor>("MseLoss");
  m.add_functor<impl::KLDivLossFunctor>("KLDivLoss");
  m.add_functor<impl::NllLossFunctor>("NllLoss");
  m.add_functor<impl::BinaryCrossEntropyLossFunctor>("BinaryCrossEntropyLoss");
  m.add_functor<impl::BinaryCrossEntropyWithLogitsLossFunctor>("BinaryCrossEntropyWithLogitsLoss");
  m.add_functor<impl::SparseCrossEntropyFunctor>("SparseCrossEntropy");
  m.add_functor<impl::SparseCrossEntropyMsFunctor>("SparseCrossEntropyMs");
  m.add_functor<impl::CrossEntropyFunctor>("CrossEntropy");
  m.add_functor<impl::SparseSoftmaxCrossEntropyFunctor>("SparseSoftmaxCrossEntropy");
  m.add_functor<impl::SoftmaxCrossEntropyFunctor>("SoftmaxCrossEntropy");
  m.add_functor<impl::SoftmaxCrossEntropyGradFunctor>("SoftmaxCrossEntropyGrad");
  m.add_functor<impl::SmoothL1LossFunctor>("SmoothL1Loss");
  m.add_functor<impl::CombinedMarginLossFunctor>("CombinedMarginLoss");
  m.add_functor<impl::TripletMarginLossFunctor>("TripletMarginLoss");
  m.add_functor<impl::MarginRankingLossFunctor>("MarginRankingLoss");
  m.add_functor<impl::CtcLossFunctor>("CtcLoss");
  m.add_functor<impl::AffineGridFunctor>("AffineGrid");
  m.add_functor<impl::GridSampleFunctor>("GridSample");
  m.add_functor<impl::NormalizationFunctor>("Normalization");
  m.add_functor<impl::NormalizationAddReluFunctor>("NormalizationAddRelu");
  m.add_functor<impl::PadFunctor>("Pad");
  m.add_functor<impl::DropoutFunctor>("Dropout");
  m.add_functor<impl::DropoutGradFunctor>("DropoutGrad");
  m.add_functor<impl::Avgpool1DFunctor>("Avgpool1D");
  m.add_functor<impl::Avgpool2DFunctor>("Avgpool2D");
  m.add_functor<impl::Avgpool3DFunctor>("Avgpool3D");
  m.add_functor<impl::UnfoldFunctor>("Unfold");
  m.add_functor<impl::FoldFunctor>("Fold");
  m.add_functor<impl::OneHotFunctor>("OneHot");
  m.add_functor<impl::FusedSelfAttentionFunctor>("FusedSelfAttention");
  m.add_functor<impl::FusedSelfAttentionGradFunctor>("FusedSelfAttentionGrad");
  m.add_functor<impl::NormalizeFunctor>("Normalize");
  m.add_functor<impl::L2NormalizeFunctor>("L2Normalize");
  m.add_functor<impl::L2NormalizeGradFunctor>("L2NormalizeGrad");
  m.add_functor<impl::FusedBiasAddGeluFunctor>("FusedBiasAddGelu");
  m.add_functor<impl::FusedBiasAddGeluGradFunctor>("FusedBiasAddGeluGrad");
  m.add_functor<impl::FusedBiasAddDropoutFunctor>("FusedBiasAddDropout");
  m.add_functor<impl::FusedScaleMaskSoftmaxFunctor>("FusedScaleMaskSoftmax");
  m.add_functor<impl::FusedScaleMaskSoftmaxDropoutFunctor>("FusedScaleMaskSoftmaxDropout");
  m.add_functor<impl::FusedScaleTrilSoftmaxMaskScaleFunctor>("FusedScaleTrilSoftmaxMaskScale");
  m.add_functor<impl::FusedScaleTrilFunctor>("FusedScaleTril");
  m.add_functor<impl::CtcGreedyDecoderFunctor>("CtcGreedyDecoder");
  m.add_functor<impl::PartialFCSampleFunctor>("DistributedPariticalFCSample");
  m.add_functor<impl::PariticalFCSampleDisableBoxing>("DistributedPariticalFCSampleDisableBoxing");
  m.add_functor<impl::NmsFunctor>("Nms");
  m.add_functor<impl::RoiAlignFunctor>("RoiAlign");
  m.add_functor<impl::RoiAlignGradFunctor>("RoiAlignGrad");
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
