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

#include "oneflow/core/common/optional.h"
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/op_interpreter.h"
#include "oneflow/core/framework/random_generator.h"
#include "oneflow/core/functional/function_library.h"
#include "oneflow/core/functional/impl/common.h"
#include "oneflow/core/functional/impl/unary_functor.h"
#include "oneflow/core/functional/scalar.h"
#include "oneflow/user/kernels/random_mask_like_kernel.h"

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
    JUST(attrs.SetAttr<int32_t>("axis", axis));
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
                           const std::vector<int32_t>& dilation, const int32_t& groups) const {
    MutableAttrMap conv_attrs;
    std::vector<int32_t> kernel_size_vec(num_spatial_dims_);
    for (int i = 0; i < num_spatial_dims_; i++) {
      kernel_size_vec.at(i) = ((weight->shape())->At(i + 2));
    }
    JUST(conv_attrs.SetAttr<int32_t>("filters", (weight->shape())->At(0)));
    JUST(conv_attrs.SetAttr<std::vector<int32_t>>("padding_before", padding));
    JUST(conv_attrs.SetAttr<std::vector<int32_t>>("kernel_size", kernel_size_vec));
    JUST(conv_attrs.SetAttr<std::vector<int32_t>>("strides", stride));
    JUST(conv_attrs.SetAttr<std::vector<int32_t>>("dilation_rate", dilation));
    JUST(conv_attrs.SetAttr<int32_t>("groups", groups));
    JUST(conv_attrs.SetAttr<std::string>("data_format", std::string("channels_first")));
    const std::shared_ptr<one::Tensor>& conv_out =
        JUST(OpInterpUtil::Dispatch<Tensor>(*conv_op_, {x, weight}, conv_attrs));
    if (bias) {
      MutableAttrMap bias_attrs;
      JUST(bias_attrs.SetAttr<int32_t>("axis", 1));
      return OpInterpUtil::Dispatch<Tensor>(*bias_op_, {conv_out, JUST(bias.value())}, bias_attrs);
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
                                const std::string& data_format, const std::vector<int32_t>& padding,
                                const std::vector<int32_t>& kernel_size,
                                const std::vector<int32_t>& stride,
                                const std::vector<int32_t>& dilation, const bool& return_indices,
                                const bool& ceil_mode) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    JUST(attrs.SetAttr<std::vector<int32_t>>("padding", padding));
    JUST(attrs.SetAttr<std::vector<int32_t>>("kernel_size", kernel_size));
    JUST(attrs.SetAttr<std::vector<int32_t>>("stride", stride));
    JUST(attrs.SetAttr<std::vector<int32_t>>("dilation", dilation));
    JUST(attrs.SetAttr<bool>("return_indices", return_indices));
    JUST(attrs.SetAttr<bool>("ceil_mode", ceil_mode));
    return OpInterpUtil::Dispatch<TensorTuple>(*op_, {x}, attrs);
  }

 protected:
  std::shared_ptr<OpExpr> op_;
};

class AvgPool2DFunctor : public PoolNDFunctor {
 public:
  AvgPool2DFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("avg_pool_2d").Input("x").Output("y").Build());
  }
};

class MaxPool2DFunctor : public PoolNDFunctor {
 public:
  MaxPool2DFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("max_pool_2d").Input("x").Output("y").Build());
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

class SparseSoftmaxCrossEntropyFunctor {
 public:
  SparseSoftmaxCrossEntropyFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("sparse_softmax_cross_entropy")
                         .Input("prediction")
                         .Input("label")
                         .Output("out")
                         .Output("prob")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& logits,
                           const std::shared_ptr<one::Tensor>& label, const int64_t& depth) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("depth", depth));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {logits, label}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class SmoothL1LossFunctor {
 public:
  SmoothL1LossFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("smooth_l1_loss").Input("prediction").Input("label").Output("loss").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& logits,
                           const std::shared_ptr<one::Tensor>& label, const float& beta) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<float>("beta", beta));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {logits, label}, attrs);
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
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& x,
                                const std::shared_ptr<one::Tensor>& label, const float& m1,
                                const float& m2, const float& m3, const int64_t& depth) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<float>("m1", m1));
    JUST(attrs.SetAttr<float>("m2", m2));
    JUST(attrs.SetAttr<float>("m3", m3));
    JUST(attrs.SetAttr<int64_t>("depth", depth));
    return OpInterpUtil::Dispatch<TensorTuple>(*op_, {x, label}, attrs);
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
                           const bool& is_training) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("axis", axis));
    JUST(attrs.SetAttr<float>("epsilon", epsilon));
    JUST(attrs.SetAttr<float>("momentum", momentum));

    CHECK_OR_RETURN((moving_mean && moving_variance) || (!moving_mean && !moving_variance))
        << "Both moving_mean and moving_variance should be None or Tensor.";
    if (!is_training) {
      CHECK_OR_RETURN(moving_mean && moving_variance)
          << "Must have moving_mean and moving_variance in eval mode.";
      return OpInterpUtil::Dispatch<one::Tensor>(
          *norm_eval_op_,
          {x, JUST(moving_mean.value()), JUST(moving_variance.value()), gamma, beta}, attrs);
    }
    if (moving_mean) {
      return OpInterpUtil::Dispatch<one::Tensor>(
          *norm_training_stats_op_,
          {x, JUST(moving_mean.value()), JUST(moving_variance.value()), gamma, beta}, attrs);
    }
    return OpInterpUtil::Dispatch<one::Tensor>(*norm_training_no_stats_op_, {x, gamma, beta},
                                               attrs);
  }

 private:
  std::shared_ptr<OpExpr> norm_eval_op_;
  std::shared_ptr<OpExpr> norm_training_stats_op_;
  std::shared_ptr<OpExpr> norm_training_no_stats_op_;
};

class PadFunctor {
 public:
  PadFunctor() {
    constant_pad_1d_ = CHECK_JUST(one::OpBuilder("constant_pad1d").Input("x").Output("y").Build());
    constant_pad_2d_ = CHECK_JUST(one::OpBuilder("constant_pad2d").Input("x").Output("y").Build());
    constant_pad_3d_ = CHECK_JUST(one::OpBuilder("constant_pad3d").Input("x").Output("y").Build());
    reflect_pad_ = CHECK_JUST(one::OpBuilder("reflection_pad2d").Input("x").Output("y").Build());
    replicate_pad_ = CHECK_JUST(one::OpBuilder("replication_pad2d").Input("x").Output("y").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const std::vector<int64_t>& pad,
                           const std::string& mode, const Scalar& value) const {
    size_t padding_size = 2 * x->shape()->NumAxes();
    CHECK_LE_OR_RETURN(pad.size(), padding_size)
        << "Pad size should less than or equal to input axes * 2.";
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::vector<int64_t>>("padding", pad));
    if (mode == "constant") {
      if (IsFloatingDataType(x->dtype()->data_type())) {
        JUST(attrs.SetAttr<double>("floating_value", JUST(value.As<double>())));
        JUST(attrs.SetAttr<int64_t>("integral_value", 0));
      } else if (IsIntegralDataType(x->dtype()->data_type())) {
        JUST(attrs.SetAttr<double>("floating_value", 0));
        JUST(attrs.SetAttr<int64_t>("integral_value", JUST(value.As<int64_t>())));
      } else {
        UNIMPLEMENTED_THEN_RETURN() << "Data type should be floating or integral type.";
      }
      switch (x->shape()->NumAxes()) {
        case 3: return OpInterpUtil::Dispatch<Tensor>(*constant_pad_1d_, {x}, attrs);
        case 4: return OpInterpUtil::Dispatch<Tensor>(*constant_pad_2d_, {x}, attrs);
        case 5: return OpInterpUtil::Dispatch<Tensor>(*constant_pad_3d_, {x}, attrs);
        default:
          UNIMPLEMENTED_THEN_RETURN() << "Pad mode is " << mode << ", but " << x->shape()->NumAxes()
                                      << "d-tensor is not support yet! ";
      }

    } else if (mode == "reflect") {
      return OpInterpUtil::Dispatch<Tensor>(*reflect_pad_, {x}, attrs);
    } else if (mode == "replicate") {
      return OpInterpUtil::Dispatch<Tensor>(*replicate_pad_, {x}, attrs);
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "Pad mode is " << mode
                                  << ", but only constant, reflect and replicate are valid.";
    }
  }

 private:
  std::shared_ptr<OpExpr> constant_pad_1d_;
  std::shared_ptr<OpExpr> constant_pad_2d_;
  std::shared_ptr<OpExpr> constant_pad_3d_;
  std::shared_ptr<OpExpr> reflect_pad_;
  std::shared_ptr<OpExpr> replicate_pad_;
};

class DropoutFunctor {
 public:
  DropoutFunctor() {
    random_mask_like_op_ =
        CHECK_JUST(one::OpBuilder("random_mask_like").Input("like").Output("out").Build());
    dropout_op_ =
        CHECK_JUST(one::OpBuilder("dropout").Input("in").Input("mask").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const float& p,
                           const Optional<one::Generator>& generator) const {
    MutableAttrMap random_mask_like_attrs;
    JUST(random_mask_like_attrs.SetAttr<float>("rate", p));

    std::shared_ptr<one::Generator> gen;
    if (!generator) {
      gen = JUST(one::DefaultAutoGenerator());
    } else {
      gen = JUST(generator.value());
    }

    JUST(random_mask_like_attrs.SetAttr<int64_t>("seed", gen->current_seed()));
    const auto& random_mask_like_state = std::make_shared<RandomMaskLikeKernelState>(gen);

    const auto& mask = JUST(OpInterpUtil::Dispatch<Tensor>(
        *random_mask_like_op_, {x},
        OpExprInterpContext(random_mask_like_attrs, random_mask_like_state)));
    float scale = 1.0;
    if (p != 1.0) { scale = 1.0 / (1.0 - p); }
    MutableAttrMap dropout_attrs;
    JUST(dropout_attrs.SetAttr<float>("scale", scale));
    return OpInterpUtil::Dispatch<Tensor>(*dropout_op_, {x, mask}, dropout_attrs);
  }

 private:
  std::shared_ptr<OpExpr> random_mask_like_op_;
  std::shared_ptr<OpExpr> dropout_op_;
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
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const std::string& data_format,
                           const std::vector<int32_t>& padding,
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
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const int64_t& num_classes,
                           const Scalar& on_value, const Scalar& off_value) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("depth", num_classes));
    bool is_on_value_double = on_value.IsFloatingPoint();
    bool is_off_value_double = off_value.IsFloatingPoint();
    if (is_on_value_double || is_off_value_double) {
      JUST(attrs.SetAttr<DataType>("dtype", kDouble));
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
    return OpInterpUtil::Dispatch<Tensor>(*one_hot_op_, {x}, attrs);
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
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& input, const int32_t& axis,
                                const float& epsilon) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("axis", axis));
    JUST(attrs.SetAttr<float>("epsilon", epsilon));
    return OpInterpUtil::Dispatch<TensorTuple>(*op_, {input}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
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
    MutableAttrMap random_mask_like_attrs;
    JUST(random_mask_like_attrs.SetAttr<float>("rate", p));
    std::shared_ptr<one::Generator> gen;
    if (!generator) {
      gen = JUST(one::DefaultAutoGenerator());
    } else {
      gen = JUST(generator.value());
    }
    JUST(random_mask_like_attrs.SetAttr<int64_t>("seed", gen->current_seed()));
    const auto& random_mask_like_state = std::make_shared<RandomMaskLikeKernelState>(gen);
    const auto& mask = JUST(OpInterpUtil::Dispatch<Tensor>(
        *random_mask_like_op_, {a},
        OpExprInterpContext(random_mask_like_attrs, random_mask_like_state)));
    float scale = 1.0;
    if (p != 1.0) { scale = 1.0 / (1.0 - p); }
    MutableAttrMap fused_bias_add_mask_attrs;
    JUST(fused_bias_add_mask_attrs.SetAttr<float>("scale", scale));
    JUST(fused_bias_add_mask_attrs.SetAttr<int32_t>("axis", axis));
    return OpInterpUtil::Dispatch<Tensor>(*fused_bias_add_mask_scale_op_, {a, b, mask},
                                          fused_bias_add_mask_attrs);
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

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::BiasAddFunctor>("BiasAdd");
  m.add_functor<impl::Conv1dFunctor>("Conv1d");
  m.add_functor<impl::Conv2dFunctor>("Conv2d");
  m.add_functor<impl::Conv3dFunctor>("Conv3d");
  m.add_functor<impl::MatMulFunctor>("MatMul");
  m.add_functor<impl::BatchMatMulFunctor>("BatchMatMul");
  m.add_functor<impl::LayerNormFunctor>("LayerNorm");
  m.add_functor<impl::LayerNormAffineFunctor>("LayerNormAffine");
  m.add_functor<impl::AvgPool2DFunctor>("AvgPool2D");
  m.add_functor<impl::Maxpool1DFunctor>("Maxpool1D");
  m.add_functor<impl::Maxpool2DFunctor>("Maxpool2D");
  m.add_functor<impl::Maxpool3DFunctor>("Maxpool3D");
  m.add_functor<impl::MaxPool2DFunctor>("MaxPool2D");
  m.add_functor<impl::AdaptiveAvgPool1DFunctor>("AdaptiveAvgPool1D");
  m.add_functor<impl::AdaptiveAvgPool2DFunctor>("AdaptiveAvgPool2D");
  m.add_functor<impl::AdaptiveAvgPool3DFunctor>("AdaptiveAvgPool3D");
  m.add_functor<impl::SparseSoftmaxCrossEntropyFunctor>("SparseSoftmaxCrossEntropy");
  m.add_functor<impl::SmoothL1LossFunctor>("SmoothL1Loss");
  m.add_functor<impl::CombinedMarginLossFunctor>("CombinedMarginLoss");
  m.add_functor<impl::NormalizationFunctor>("Normalization");
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
  m.add_functor<impl::L2NormalizeFunctor>("L2Normalize");
  m.add_functor<impl::L2NormalizeGradFunctor>("L2NormalizeGrad");
  m.add_functor<impl::FusedBiasAddGeluFunctor>("FusedBiasAddGelu");
  m.add_functor<impl::FusedBiasAddGeluGradFunctor>("FusedBiasAddGeluGrad");
  m.add_functor<impl::FusedBiasAddDropoutFunctor>("FusedBiasAddDropout");
  m.add_functor<impl::FusedScaleTrilFunctor>("FusedScaleTril");
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
