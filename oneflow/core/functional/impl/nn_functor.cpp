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
#include "oneflow/core/common/error.h"
#include "oneflow/core/common/maybe.h"
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
#include "oneflow/core/common/container_util.h"
#include "oneflow/user/kernels/distributions/common.h"
#include "oneflow/core/framework/nd_sbp.h"

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
  explicit DeConvBaseFunctor(const int& num_spatial_dims) : num_spatial_dims_(num_spatial_dims) {
    bias_op_ = CHECK_JUST(one::OpBuilder("bias_add").Input("a").Input("b").Output("out").Build());
  }
  virtual ~DeConvBaseFunctor() = default;
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& weight,
                           const Optional<one::Tensor>& bias, const std::vector<int32_t>& stride,
                           const std::vector<int32_t>& padding,
                           const std::vector<int32_t>& output_padding, const int32_t& groups,
                           const std::vector<int32_t>& dilation,
                           const std::string& data_format) const {
    MutableAttrMap deconv_attrs;
    std::vector<int32_t> kernel_size_vec(num_spatial_dims_);
    int32_t kernel_idx_offset = 2;
    if (data_format == "channels_last") { kernel_idx_offset = 1; }
    for (int i = 0; i < num_spatial_dims_; i++) {
      kernel_size_vec[i] = ((weight->shape())->At(i + kernel_idx_offset));
    }

    JUST(deconv_attrs.SetAttr<int32_t>("filters", (weight->shape())->At(1) * groups));
    JUST(deconv_attrs.SetAttr<std::vector<int32_t>>("padding_before", padding));
    JUST(deconv_attrs.SetAttr<std::vector<int32_t>>("kernel_size", kernel_size_vec));
    JUST(deconv_attrs.SetAttr<std::vector<int32_t>>("output_padding", output_padding));
    JUST(deconv_attrs.SetAttr<std::vector<int32_t>>("strides", stride));
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
  int32_t num_spatial_dims_;
};

class DeConv1dFunctor : public DeConvBaseFunctor {
 public:
  DeConv1dFunctor() : DeConvBaseFunctor(/*num_spatial_dims_=*/1) {
    deconv_op_ =
        CHECK_JUST(one::OpBuilder("deconv1d").Input("in").Input("weight").Output("out").Build());
  }
};

class DeConv2dFunctor : public DeConvBaseFunctor {
 public:
  DeConv2dFunctor() : DeConvBaseFunctor(/*num_spatial_dims_=*/2) {
    deconv_op_ =
        CHECK_JUST(one::OpBuilder("deconv2d").Input("in").Input("weight").Output("out").Build());
  }
};

class DeConv3dFunctor : public DeConvBaseFunctor {
 public:
  DeConv3dFunctor() : DeConvBaseFunctor(/*num_spatial_dims_=*/3) {
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
    CHECK_EQ_OR_RETURN(a_shape->NumAxes(), 3)
        << Error::RuntimeError() << "Expected 3-dimensional tensor, but got " << a_shape->NumAxes()
        << "-dimensional tensor for argument #1";
    CHECK_EQ_OR_RETURN(b_shape->NumAxes(), 3)
        << Error::RuntimeError() << "Expected 3-dimensional tensor, but got " << b_shape->NumAxes()
        << "-dimensional tensor for argument #2";
    CHECK_EQ_OR_RETURN(a_shape->At(0), b_shape->At(0))
        << Error::RuntimeError() << "Batch dim not match, please check input!";
    CHECK_EQ_OR_RETURN(a_shape->At(2), b_shape->At(1))
        << Error::RuntimeError() << "Matmul dim not match, please check input!";
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<bool>("transpose_a", transpose_a));
    JUST(attrs.SetAttr<bool>("transpose_b", transpose_b));
    JUST(attrs.SetAttr<double>("alpha", alpha));
    return OpInterpUtil::Dispatch<Tensor>(*batch_matmul_op_, {a, b}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> batch_matmul_op_;
};

class FusedMLPFunctor {
 public:
  FusedMLPFunctor() {
#if CUDA_VERSION >= 11040
    fused_op_.resize(kMaxInputCount /*the maximum number of inputs*/);
    for (int n = 1; n < fused_op_.size(); ++n) {
      fused_op_[n] = CHECK_JUST(one::OpBuilder("cublas_fused_mlp")
                                    .Input("x")
                                    .Input("weights", n)
                                    .Input("biases", n)
                                    .Output("out")
                                    .Output("cublas_aux", n)
                                    .Output("hidden", n)
                                    .Build());
    }
#endif
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const TensorTuple& weights,
                           const TensorTuple& biases, bool skip_final_activation) const {
    const int64_t weight_size = weights.size();
    const int64_t bias_size = biases.size();
    CHECK_GE_OR_RETURN(weight_size, 1) << "The number of weights should be greater equal than 1. ";
    CHECK_EQ_OR_RETURN(weight_size, bias_size)
        << "The number of weights should be equal to biases. ";
    int64_t n = 0, k = 0;
    /*
    x: (m, k)
    weight: (n, k) need transpose
    bias: (n)
    */
    const auto& x_shape = x->shape();
    k = x_shape->At(1);
    for (int64_t i = 0; i < weight_size; i++) {
      const auto& weight_shape = weights[i]->shape();
      const auto& bias_shape = biases[i]->shape();

      // TODO(): Support Fused batch/broadcast matmul.
      CHECK_EQ_OR_RETURN(weight_shape->NumAxes(), 2) << "Weight's dim should == 2";
      CHECK_EQ_OR_RETURN(bias_shape->NumAxes(), 1) << "Bias's dim should == 1";

      n = weight_shape->At(0);
      CHECK_EQ_OR_RETURN(bias_shape->At(0), n) << "Bias's dim is not equal to weight's last dim. ";
      CHECK_EQ_OR_RETURN(weight_shape->At(1), k)
          << "weight's first dim should be equal to input's last dim. ";

      // Set for next layer.
      k = n;
    }

#if CUDA_VERSION >= 11060
    DeviceType device_type{};
    if (x->is_consistent()) {
      device_type = JUST(x->parallel_desc())->device_type();
    } else {
      device_type = JUST(x->device())->enum_type();
    }

    if ((device_type == DeviceType::kCUDA) && (weight_size <= kMaxInputCount)
        && (!ParseBooleanFromEnv("ONEFLOW_FUNCTOR_DISABLE_FUSED_MLP", false))) {
      TensorTuple input(2 * weight_size + 1);
      input[0] = x;
      std::copy(weights.begin(), weights.end(), input.begin() + 1);
      std::copy(biases.begin(), biases.end(), input.begin() + 1 + weight_size);

      MutableAttrMap attrs;
      JUST(attrs.SetAttr<bool>("skip_final_activation", skip_final_activation));
      return OpInterpUtil::Dispatch<Tensor>(*fused_op_[weight_size], input, attrs);
    }
#endif  // CUDA_VERSION >= 11060

    // Fall back to Naive matmul + bias_add + relu
    std::shared_ptr<one::Tensor> out = x;
    for (int32_t layer_idx = 0; layer_idx < weight_size; layer_idx++) {
      out = JUST(
          functional::BiasAdd(JUST(functional::MatMul(out, weights[layer_idx], false, true, 1.0)),
                              biases[layer_idx], 1));
      if ((layer_idx != weight_size - 1) || (!skip_final_activation)) {
        /*
        When it is not finaly dense layer, or it is final dense layer and skip_final_activate=False,
        we add relu Layer.
        */
        out = JUST(functional::Relu(out, false));
      }
    }
    return out;
  }

 private:
#if CUDA_VERSION >= 11040
  std::vector<std::shared_ptr<OpExpr>> fused_op_;
#endif
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

class PixelShuffleFunctor {
 public:
  PixelShuffleFunctor() {}
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const int64_t& h_upscale_factor,
                           const int64_t& w_upscale_factor) const {
    CHECK_OR_RETURN(x->ndim() == 4) << "Only Accept 4D Tensor";
    const int64_t batch = x->shape()->At(0);
    const int64_t channel = x->shape()->At(1);
    const int64_t height = x->shape()->At(2);
    const int64_t width = x->shape()->At(3);
    std::shared_ptr<one::Tensor> out;
    CHECK_OR_RETURN(channel % (h_upscale_factor * w_upscale_factor) == 0)
        << "The channels of input tensor must be divisible by (upscale_factor * upscale_factor) or "
           "(h_upscale_factor * w_upscale_factor)";
    const int64_t new_c = static_cast<int>(channel / (h_upscale_factor * w_upscale_factor));
    std::vector<int32_t> permute_vec = {0, 1, 4, 2, 5, 3};
    std::vector<int64_t> reshape_vec_1 = {batch, new_c, h_upscale_factor * w_upscale_factor, height,
                                          width};
    Shape reshape_1(DimVector(reshape_vec_1.begin(), reshape_vec_1.end()));
    std::vector<int64_t> reshape_vec_2 = {batch,  new_c, h_upscale_factor, w_upscale_factor,
                                          height, width};
    Shape reshape_2(DimVector(reshape_vec_2.begin(), reshape_vec_2.end()));
    std::vector<int64_t> reshape_vec_3 = {batch, new_c, height * h_upscale_factor,
                                          width * w_upscale_factor};
    Shape reshape_3(DimVector(reshape_vec_3.begin(), reshape_vec_3.end()));
    out = JUST(Reshape(x, reshape_1));
    out = JUST(Reshape(out, reshape_2));
    out = JUST(Permute(out, permute_vec));
    out = JUST(Reshape(out, reshape_3));
    return out;
  }
};

class TFPoolNDFunctor {
 public:
  TFPoolNDFunctor() = default;
  virtual ~TFPoolNDFunctor() = default;
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

class MaxPoolNDFunctor {
 public:
  MaxPoolNDFunctor() = default;
  virtual ~MaxPoolNDFunctor() = default;
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& x,
                                const std::vector<int32_t>& kernel_size,
                                const Optional<std::vector<int32_t>>& stride,
                                const std::vector<int32_t>& padding,
                                const std::vector<int32_t>& dilation, const bool& return_indices,
                                const bool& ceil_mode, const std::string& data_format) const {
    if (x->ndim() == 4 && data_format == "channels_last") {
      if (!return_indices && dilation.at(0) == 1 && dilation.at(1) == 1) {
        // legacy tf style maxpool2d , use cudnn implementation
        // with high performance but do not support dilation/return_indices
        MutableAttrMap attrs;
        std::vector<int32_t> padding_before{padding.at(0), padding.at(1)};
        std::vector<int32_t> padding_after{padding.at(0), padding.at(1)};

        JUST(attrs.SetAttr<std::vector<int32_t>>("pool_size", kernel_size));
        if (stride.has_value()) {
          JUST(attrs.SetAttr<std::vector<int32_t>>("strides", *JUST(stride)));
        } else {
          JUST(attrs.SetAttr<std::vector<int32_t>>("strides", kernel_size));
        }
        JUST(attrs.SetAttr<std::string>("padding", "customized"));
        JUST(attrs.SetAttr<std::vector<int32_t>>("padding_before", padding_before));
        JUST(attrs.SetAttr<std::vector<int32_t>>("padding_after", padding_after));
        JUST(attrs.SetAttr<std::string>("data_format", data_format));
        JUST(attrs.SetAttr<bool>("ceil_mode", ceil_mode));
        TensorTuple output;
        output.emplace_back(JUST(OpInterpUtil::Dispatch<Tensor>(*tf_maxpool_op_, {x}, attrs)));
        return output;
      }
    }

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
  std::shared_ptr<OpExpr> tf_maxpool_op_;
};

class TFAvgPool2DFunctor : public TFPoolNDFunctor {
 public:
  TFAvgPool2DFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("tf_avg_pool_2d").Input("x").Output("y").Build());
  }
};

class MaxPool1DFunctor : public MaxPoolNDFunctor {
 public:
  MaxPool1DFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("max_pool_1d").Input("x").Output("y").Output("indice").Build());
  }
};

class MaxPool2DFunctor : public MaxPoolNDFunctor {
 public:
  MaxPool2DFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("max_pool_2d").Input("x").Output("y").Output("indice").Build());
    tf_maxpool_op_ = CHECK_JUST(one::OpBuilder("tf_max_pool_2d").Input("x").Output("y").Build());
  }
};

class MaxPool3DFunctor : public MaxPoolNDFunctor {
 public:
  MaxPool3DFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("max_pool_3d").Input("x").Output("y").Output("indice").Build());
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
    const auto out = sequence_function(functional::Sub)
                         .then(functional::Square)
                         .call(input, target, /*alpha=*/1.0, /*inplace=*/false);
    return apply_reduction(out, reduction);
  }
};

class L1LossFunctor : public LossFunctorBase {
 public:
  L1LossFunctor() {}
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& target,
                           const std::string& reduction) const {
    const auto out = sequence_function(functional::Sub)
                         .then(functional::Abs)
                         .call(input, target, /*alpha=*/1.0, /*inplace=*/false);
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
            .call(input_1, input_2, /*alpha=*/1.0, /*inplace=*/false);
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
    result = JUST(functional::Reshape((*kernel_result)[0], *target_shape));
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

    if (JUST(logits->parallel_desc())->parallel_num() == 1) { return false; }

    if (logits->shape()->NumAxes() != 2) { return false; }

    const NdSbp& logits_nd_sbp = *(JUST(logits->nd_sbp()));
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

    const NdSbp& logits_nd_sbp = *(JUST(logits->nd_sbp()));
    std::vector<Symbol<SbpParallel>> new_sbp_parallels;
    std::vector<Symbol<SbpParallel>> s0s1_sbp_parallels;
    if (logits_nd_sbp.sbp_parallel_size() == 2) {
      for (int i = 0; i < logits_nd_sbp.sbp_parallel_size(); ++i) {
        const auto& sbp_parallel = logits_nd_sbp.sbp_parallel(i);
        if (sbp_parallel.has_split_parallel()) {
          const int64_t& split_axis = sbp_parallel.split_parallel().axis();
          if (split_axis == axis) {
            SbpParallel sbp;
            sbp.mutable_broadcast_parallel();
            new_sbp_parallels.emplace_back(sbp);
          } else {
            CHECK_EQ_OR_RETURN(split_axis, 0);
            new_sbp_parallels.emplace_back(sbp_parallel);
          }
        } else {
          new_sbp_parallels.emplace_back(sbp_parallel);
        }
      }

      s0s1_sbp_parallels.emplace_back(logits_nd_sbp.sbp_parallel(0));
      s0s1_sbp_parallels.emplace_back(logits_nd_sbp.sbp_parallel(1));
      max_global_stage_input0 = JUST(functional::ToConsistent(
          max_device_stage->at(0), JUST(max_device_stage->at(0)->parallel_desc()),
          new_sbp_parallels, s0s1_sbp_parallels, /* check_meta */ false));
      max_global_stage_input1 = JUST(functional::ToConsistent(
          max_device_stage->at(2), JUST(max_device_stage->at(0)->parallel_desc()),
          new_sbp_parallels, s0s1_sbp_parallels, /* check_meta */ false));
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
          broadcast_sub_input, JUST(max_device_stage->at(0)->parallel_desc()), new_sbp_parallels,
          new_sbp_parallels, /* check_meta */ false));
    }
    // op_broadcast_sub_
    attrs.clear();
    const auto& output_broadcast_sub = JUST(OpInterpUtil::Dispatch<TensorTuple>(
        *op_broadcast_sub_, {logits, broadcast_sub_input}, attrs));
    // op_exp_
    const auto& output_exp =
        JUST(OpInterpUtil::Dispatch<TensorTuple>(*op_exp_, {(*output_broadcast_sub)[0]}, attrs));
    // op_reduce_sum_
    JUST(attrs.SetAttr<std::vector<int32_t>>("axis", {axis}));
    JUST(attrs.SetAttr<bool>("keepdims", true));
    const auto& output_reduce_sum =
        JUST(OpInterpUtil::Dispatch<TensorTuple>(*op_reduce_sum_, {(*output_exp)[0]}, attrs));
    std::shared_ptr<Tensor> broadcast_div_input1 = output_reduce_sum->at(0);
    if (logits_nd_sbp.sbp_parallel_size() == 2) {
      std::vector<Symbol<SbpParallel>> empty_grad_sbp_parallels;
      broadcast_div_input1 = JUST(functional::ToConsistent(
          output_reduce_sum->at(0), JUST(output_reduce_sum->at(0)->parallel_desc()),
          new_sbp_parallels, new_sbp_parallels, /* check_meta */ false));
    }
    // op_broadcast_div_
    attrs.clear();
    const auto& predictions = JUST(OpInterpUtil::Dispatch<TensorTuple>(
        *op_broadcast_div_, {(*output_exp)[0], broadcast_div_input1}, attrs));
    // op_sparse_cross_entropy_ms_
    JUST(attrs.SetAttr<int64_t>("depth", depth));
    const auto& output = JUST(OpInterpUtil::Dispatch<Tensor>(*op_sparse_cross_entropy_ms_,
                                                             {(*predictions)[0], label}, attrs));
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
          .then(std::bind(functional::Cast, std::placeholders::_1, log_probs->dtype(),
                          /*pin_memory=*/false))
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
    auto da_p = JUST(VectorNorm(
        JUST(ScalarAdd(eps, JUST(Sub(anchor, positive, /*alpha=*/1.0, /*inplace=*/false)),
                       /*alpha=*/1)),
        p, dim,
        /*keepdim=*/false, anchor->dtype()));
    auto da_n = JUST(VectorNorm(
        JUST(ScalarAdd(eps, JUST(Sub(anchor, negative, /*alpha=*/1.0, /*inplace=*/false)),
                       /*alpha=*/1)),
        p, dim,
        /*keepdim=*/false, anchor->dtype()));
    if (swap) {
      auto distance_swap = JUST(VectorNorm(
          JUST(ScalarAdd(eps, JUST(Sub(positive, negative, /*alpha=*/1.0, /*inplace=*/false)),
                         /*alpha=*/1)),
          p, dim,
          /*keepdim=*/false, positive->dtype()));
      da_n = JUST(Minimum(distance_swap, da_n));
    }
    auto triplet_loss =
        JUST(Clamp(JUST(ScalarAdd(JUST(Sub(da_p, da_n, /*alpha=*/1.0, /*inplace=*/false)), margin,
                                  /*alpha=*/1, /*inplace=*/false)),
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

class NormalFunctor {
 public:
  NormalFunctor() { op_ = CHECK_JUST(one::OpBuilder("normal").Output("out").Build()); }
  Maybe<Tensor> operator()(const float& mean, const float& std, const Shape& shape,
                           const Optional<one::Tensor>& out,
                           const Optional<Symbol<DType>>& optional_dtype,
                           const Optional<Symbol<Device>>& optional_device,
                           const Optional<one::Generator>& optional_generator,
                           const bool& requires_grad) const {
    Symbol<DType> dtype = DType::Float();
    if (optional_dtype.has_value()) {
      dtype = JUST(optional_dtype);
      if (dtype->data_type() != DataType::kFloat && dtype->data_type() != DataType::kDouble) {
        OF_UNIMPLEMENTED() << "Only support float and double in normal().";
      }
    }
    Symbol<Device> device = JUST(Device::New("cpu"));
    if (optional_device.has_value()) { device = JUST(optional_device); }

    if (out.has_value()) {
      auto out_tensor = JUST(out);
      Symbol<DType> output_tensor_dtype = out_tensor->dtype();
      if (optional_dtype.has_value()) {
        CHECK_OR_RETURN(output_tensor_dtype == dtype)
            << Error::RuntimeError() << "data type " << dtype->name()
            << " does not match data type of out parameter (" << output_tensor_dtype->name();
      }
      dtype = output_tensor_dtype;
      Symbol<Device> out_tensor_device = JUST(out_tensor->device());
      if (optional_device.has_value()) {
        CHECK_OR_RETURN(out_tensor_device == JUST(optional_device))
            << Error::RuntimeError() << "device type " << device->ToString()
            << " does not match device type of out parameter (" << out_tensor_device->ToString();
      }
      device = out_tensor_device;
    }

    const auto gen = optional_generator.value_or(JUST(one::DefaultAutoGenerator()));
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<double>("mean", mean));
    JUST(attrs.SetAttr<double>("std", std));
    JUST(attrs.SetAttr<Shape>("shape", shape));
    JUST(attrs.SetAttr<DataType>("dtype", dtype->data_type()));
    JUST(attrs.SetAttr<int64_t>("seed", gen->current_seed()));

    const auto& distribution_state = std::make_shared<DistributionKernelState>(gen);
    OpExprInterpContext ctx(attrs, distribution_state);
    ctx.device = device;
    if (out.has_value()) {
      std::shared_ptr<TensorTuple> outputs = std::make_shared<TensorTuple>(1);
      (*outputs)[0] = JUST(out);
      JUST(OpInterpUtil::Dispatch(*op_, {}, outputs.get(), ctx));
      return (*outputs)[0];
    }

    auto result = JUST(OpInterpUtil::Dispatch<Tensor>(*op_, {}, ctx));
    JUST(result->set_requires_grad(requires_grad));
    return result;
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ConsistentNormalFunctor {
 public:
  ConsistentNormalFunctor() { op_ = CHECK_JUST(one::OpBuilder("normal").Output("out").Build()); }
  Maybe<Tensor> operator()(const float& mean, const float& std, const Shape& shape,
                           const Optional<one::Tensor>& out, const Symbol<ParallelDesc>& placement,
                           const std::vector<Symbol<SbpParallel>>& sbp_tuple,
                           const Optional<Symbol<DType>>& optional_dtype,
                           const Optional<one::Generator>& optional_generator,
                           const bool& requires_grad) const {
    JUST(CheckDeviceIdsIsValid(placement));

    Symbol<DType> dtype = DType::Float();
    if (optional_dtype.has_value()) {
      dtype = JUST(optional_dtype);
      if (dtype->data_type() != DataType::kFloat && dtype->data_type() != DataType::kDouble) {
        OF_UNIMPLEMENTED() << "Only support float and double in normal().";
      }
    }

    if (out.has_value()) {
      auto out_tensor = JUST(out);
      Symbol<DType> output_tensor_dtype = out_tensor->dtype();
      if (optional_dtype.has_value()) {
        CHECK_OR_RETURN(output_tensor_dtype == dtype)
            << Error::RuntimeError() << "data type " << dtype->name()
            << " does not match data type of out parameter (" << output_tensor_dtype->name();
      }
      dtype = output_tensor_dtype;
    }

    const auto gen = optional_generator.value_or(JUST(one::DefaultAutoGenerator()));
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<double>("mean", mean));
    JUST(attrs.SetAttr<double>("std", std));
    JUST(attrs.SetAttr<Shape>("shape", shape));
    JUST(attrs.SetAttr<DataType>("dtype", dtype->data_type()));
    JUST(attrs.SetAttr<int64_t>("seed", gen->current_seed()));

    const auto& distribution_state = std::make_shared<DistributionKernelState>(gen);
    const auto& nd_sbp = JUST(GetNdSbp(sbp_tuple));
    if (LazyMode::is_enabled()) {
      JUST(attrs.SetAttr<std::vector<std::string>>("nd_sbp", *JUST(GetNdSbpStrList(nd_sbp))));
    }

    if (out.has_value()) {
      std::shared_ptr<TensorTuple> outputs = std::make_shared<TensorTuple>(1);
      (*outputs)[0] = JUST(out);
      JUST(OpInterpUtil::Dispatch(
          *op_, {}, outputs.get(),
          OpExprInterpContext(attrs, placement, nd_sbp, distribution_state)));
      return (*outputs)[0];
    }

    auto result = JUST(OpInterpUtil::Dispatch<Tensor>(
        *op_, {}, OpExprInterpContext(attrs, placement, nd_sbp, distribution_state)));
    JUST(result->set_requires_grad(requires_grad));
    return result;
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
                           const Optional<one::Tensor>& gamma, const Optional<one::Tensor>& beta,
                           const int32_t& axis, const float& epsilon, const float& momentum,
                           const bool& training) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("axis", axis));
    JUST(attrs.SetAttr<float>("epsilon", epsilon));
    // convert torch momentum to tensorflow momentum
    JUST(attrs.SetAttr<float>("momentum", 1.0 - momentum));

    CHECK_OR_RETURN((moving_mean && moving_variance) || (!moving_mean && !moving_variance))
        << "Both moving_mean and moving_variance should be None or Tensor.";

    std::shared_ptr<one::Tensor> gamma_val;
    std::shared_ptr<one::Tensor> beta_val;

    CHECK_GE_OR_RETURN(x->shape()->NumAxes(), 2)
        << "NumAxes of x should be greater or equal than 2. ";
    if (gamma.has_value() && beta.has_value()) {
      gamma_val = JUST(gamma);
      beta_val = JUST(beta);
    } else {
      const Shape gamma_beta_shape = Shape({x->shape()->At(1)});
      gamma_val = JUST(functional::Constant(gamma_beta_shape, 1.0, x->dtype(), JUST(x->device())));
      beta_val = JUST(functional::Constant(gamma_beta_shape, 0.0, x->dtype(), JUST(x->device())));
    }

    if (!training) {
      CHECK_OR_RETURN(moving_mean && moving_variance)
          << "Must have moving_mean and moving_variance in eval mode.";
      return OpInterpUtil::Dispatch<one::Tensor>(
          *norm_eval_op_, {x, JUST(moving_mean), JUST(moving_variance), gamma_val, beta_val},
          attrs);
    }
    if (moving_mean) {
      return OpInterpUtil::Dispatch<one::Tensor>(
          *norm_training_stats_op_,
          {x, JUST(moving_mean), JUST(moving_variance), gamma_val, beta_val}, attrs);
    }
    return OpInterpUtil::Dispatch<one::Tensor>(*norm_training_no_stats_op_,
                                               {x, gamma_val, beta_val}, attrs);
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
      if (IsFloatingDataType(x->dtype()->data_type())
          || x->dtype()->data_type() == DataType::kFloat16) {
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
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const float& p,
                           const bool& training, const bool& inplace,
                           const Optional<one::Generator>& generator,
                           const Optional<one::Tensor>& addend) const {
    auto outputs = std::make_shared<TensorTuple>(1);
    if (inplace) {
      JUST(CheckInplaceValid(x));
      (*outputs)[0] = x;
    }
    const auto gen = generator.value_or(JUST(one::DefaultAutoGenerator()));
    const auto& dropout_state = std::make_shared<FusedDropoutKernelState>(gen);
    MutableAttrMap dropout_attrs;
    JUST(dropout_attrs.SetAttr<float>("rate", p));
    if (addend) {
      if ((!training) || p == 0.0) {
        JUST(OpInterpUtil::Dispatch(*add_op_, {x, JUST(addend)}, outputs.get()));
      } else {
        outputs->resize(2);
        JUST(OpInterpUtil::Dispatch(*dropout_addend_op_, {x, JUST(addend)}, outputs.get(),
                                    OpExprInterpContext(dropout_attrs, dropout_state)));
      }
    } else {
      if (!training || p == 0.0) {
        return x;
      } else {
        outputs->resize(2);
        JUST(OpInterpUtil::Dispatch(*dropout_op_, {x}, outputs.get(),
                                    OpExprInterpContext(dropout_attrs, dropout_state)));
      }
    }
    return (*outputs)[0];
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

class AvgPoolNDFunctor {
 public:
  AvgPoolNDFunctor() = default;
  virtual ~AvgPoolNDFunctor() = default;
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::vector<int32_t>& kernel_size,
                           const Optional<std::vector<int32_t>>& stride,
                           const std::vector<int32_t>& padding, const bool& ceil_mode,
                           const bool& count_include_pad, const int32_t& divisor_override,
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
    JUST(attrs.SetAttr<int32_t>("divisor_override", divisor_override));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 protected:
  std::shared_ptr<OpExpr> op_;
};

class AvgPool1DFunctor : public AvgPoolNDFunctor {
 public:
  AvgPool1DFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("avg_pool_1d").Input("x").Output("y").Build());
  }
};

class AvgPool2DFunctor : public AvgPoolNDFunctor {
 public:
  AvgPool2DFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("avg_pool_2d").Input("x").Output("y").Build());
  }
};

class AvgPool3DFunctor : public AvgPoolNDFunctor {
 public:
  AvgPool3DFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("avg_pool_3d").Input("x").Output("y").Build());
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
      const auto& callback = [&](uint64_t of_blob_ptr) {
        auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
        of_blob->AutoMemCopyTo<int64_t>(&max, 1);  // copy 1 scalar(int64_t) tensor's value to max
      };
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

class CosineSimilarityFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& y, const int32_t& dim,
                           const double& eps) const {
    const auto& x_shape = *(x->shape());
    const auto& y_shape = *(y->shape());
    std::shared_ptr<one::Tensor> x_extend = x;
    std::shared_ptr<one::Tensor> y_extend = y;
    if (x_shape != y_shape) {
      Shape max_shape = Shape::Ones(std::max(x_shape.NumAxes(), y_shape.NumAxes()));
      for (int64_t i = max_shape.NumAxes() - 1; i >= 0; i--) {
        int64_t offset = max_shape.NumAxes() - 1 - i;
        int64_t dim_x = x_shape.NumAxes() - 1 - offset;
        int64_t dim_y = y_shape.NumAxes() - 1 - offset;
        int64_t size_x = (dim_x >= 0) ? x_shape.At(i) : 1;
        int64_t size_y = (dim_y >= 0) ? y_shape.At(i) : 1;
        if (!(size_x == size_y || size_x == 1 || size_y == 1)) {
          return Error::RuntimeError()
                 << "The size of tensor a (" << size_x << ") must match the size of tensor b ("
                 << size_y << ") at non-singleton dimension " << i;
        }
        max_shape.Set(i, std::max(size_x, size_y));
      }
      x_extend = JUST(Expand(x, max_shape));
      y_extend = JUST(Expand(y, max_shape));
    }
    TensorProcessor tensor_processor;
    JUST(tensor_processor.PromoteInputsToCommonDtype(true).AddInputs({x_extend, y_extend}).Apply());
    TensorTuple input_vec = JUST(tensor_processor.GetInputs());
    const auto common_dtype = JUST(oneflow::VectorAt(input_vec, 0))->dtype();
    if (!IsFloatingDataType(common_dtype->data_type())) {
      return Error::RuntimeError()
             << "expected common dtype to be floating point, yet common dtype is "
             << common_dtype->name();
    }
    auto& x_ = JUST(oneflow::VectorAt(input_vec, 0));
    auto& y_ = JUST(oneflow::VectorAt(input_vec, 1));
    std::shared_ptr<Tensor> w12 =
        JUST(functional::ReduceSum(JUST(functional::Mul(x_, y_)), {dim}, false));
    std::shared_ptr<Tensor> w1 =
        JUST(functional::ReduceSum(JUST(functional::Mul(x_, x_)), {dim}, false));
    std::shared_ptr<Tensor> w2 =
        JUST(functional::ReduceSum(JUST(functional::Mul(y_, y_)), {dim}, false));
    std::shared_ptr<Tensor> n12 = JUST(functional::Sqrt(
        JUST(functional::Clamp(JUST(functional::Mul(w1, w2)), Scalar(eps * eps), NullOpt))));
    return functional::Div(w12, n12);
  }
};

class L2NormalizeFunctor {
 public:
  L2NormalizeFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("l2_normalize").Input("x").Output("y").Output("square_x_sum").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input, const int32_t& axis,
                           const float& epsilon) const {
    const auto ndims = input->shape()->NumAxes();
    const auto final_dim = ndims - 1;

    auto axis_ = axis >= 0 ? axis : axis + ndims;
    CHECK_GE_OR_RETURN(axis_, 0) << "Axis should >=0 but axis is " << axis_ << " now.";
    CHECK_LE_OR_RETURN(axis_, final_dim)
        << "Axis should <" << ndims << " but axis is " << axis_ << " now.";

    MutableAttrMap attrs;
    JUST(attrs.SetAttr<float>("epsilon", epsilon));
    JUST(attrs.SetAttr<int32_t>("axis", final_dim));

    if (axis_ == final_dim) { return OpInterpUtil::Dispatch<Tensor>(*op_, {input}, attrs); }

    std::vector<int> input_perm(input->shape()->dim_vec().size(), 0);
    for (size_t i = 0; i < input_perm.size(); ++i) { input_perm[i] = static_cast<int>(i); }
    std::swap(input_perm[final_dim], input_perm[static_cast<size_t>(axis_)]);

    const auto result = JUST(OpInterpUtil::Dispatch<TensorTuple>(
        *op_, {JUST(functional::Transpose(input, input_perm))}, attrs));
    return functional::Transpose((*result)[0], input_perm);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class NormalizeFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input, const float& p,
                           const int32_t& dim, const float& eps,
                           const bool& use_l2_norm_kernel) const {
    if (use_l2_norm_kernel && (std::fabs(p - 2.0f) < std::numeric_limits<float>::min())) {
      return functional::L2Normalize(input, dim, eps);
    }
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
    if (p > 0.0) {
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

class FusedDotFeatureInteractionFunctor {
 public:
  FusedDotFeatureInteractionFunctor() {
    ops_has_output_concat_.resize(kMaxInputCount);
    ops_no_output_concat_.resize(kMaxInputCount);
    for (int n = 0; n < ops_has_output_concat_.size(); ++n) {
      ops_has_output_concat_[n] = CHECK_JUST(one::OpBuilder("fused_dot_feature_interaction")
                                                 .Input("features", n + 1)
                                                 .Input("output_concat")
                                                 .Output("out")
                                                 .Output("padded_concated_features")
                                                 .Build());
    }
    for (int n = 0; n < ops_no_output_concat_.size(); ++n) {
      ops_no_output_concat_[n] = CHECK_JUST(one::OpBuilder("fused_dot_feature_interaction")
                                                .Input("features", n + 1)
                                                .Output("out")
                                                .Output("padded_concated_features")
                                                .Build());
    }
    ops_no_padded_concated_features_.resize(kMaxInputCount);
    for (int n = 0; n < ops_no_padded_concated_features_.size(); ++n) {
      ops_no_padded_concated_features_[n] =
          CHECK_JUST(one::OpBuilder("fused_dot_feature_interaction")
                         .Input("features", n + 1)
                         .Output("out")
                         .Build());
    }
  }

  Maybe<Tensor> operator()(const TensorTuple& features, const Optional<one::Tensor>& output_concat,
                           const bool& self_interaction, const int32_t& output_padding,
                           const std::string& pooling) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<bool>("self_interaction", self_interaction));
    JUST(attrs.SetAttr<int32_t>("output_padding", output_padding));
    JUST(attrs.SetAttr<std::string>("pooling", pooling));
    const int64_t n_features = features.size();
    TensorTuple inputs;
    if (n_features > kMaxInputCount) {
      inputs.push_back(JUST(functional::Concat(features, 1)));
    } else {
      inputs = features;
    }
    CHECK_OR_RETURN(pooling == "sum" || pooling == "none")
        << Error::RuntimeError() << "pooling should be sum or none, but get " << pooling;

    if (pooling == "sum") {
      CHECK_EQ_OR_RETURN(output_padding, 0)
          << Error::RuntimeError() << "output_padding should be equal to 0. ";
      CHECK_OR_RETURN(!output_concat) << Error::RuntimeError() << "output_concat should not exist";
      JUST(attrs.SetAttr<bool>("has_output_concat", false));
      const std::shared_ptr<one::Tensor>& bi_interaction = JUST(OpInterpUtil::Dispatch<Tensor>(
          *JUST(oneflow::VectorAt(ops_no_padded_concated_features_, n_features - 1)), inputs,
          attrs));
      std::vector<int32_t> reduce_axes_vec = {1};
      return functional::ReduceSum(bi_interaction, reduce_axes_vec, true);
    }
    if (output_concat) {
      JUST(attrs.SetAttr<bool>("has_output_concat", true));
      inputs.push_back(JUST(output_concat));
      return OpInterpUtil::Dispatch<Tensor>(
          *JUST(oneflow::VectorAt(ops_has_output_concat_, n_features - 1)), inputs, attrs);
    } else {
      JUST(attrs.SetAttr<bool>("has_output_concat", false));
      return OpInterpUtil::Dispatch<Tensor>(
          *JUST(oneflow::VectorAt(ops_no_output_concat_, n_features - 1)), inputs, attrs);
    }
  }

 private:
  std::vector<std::shared_ptr<OpExpr>> ops_has_output_concat_;
  std::vector<std::shared_ptr<OpExpr>> ops_no_output_concat_;
  std::vector<std::shared_ptr<OpExpr>> ops_no_padded_concated_features_;
};

class OneEmbeddingIdShuffleFunctor {
 public:
  OneEmbeddingIdShuffleFunctor() {
    op_table_ids_has_in_out_ = CHECK_JUST(one::OpBuilder("id_shuffle")
                                              .Input("ids")
                                              .Input("table_ids")
                                              .Output("num_unique_matrix")
                                              .Output("inverse_unique_partition_indices")
                                              .Output("cur_rank_num_unique")
                                              .Output("cur_rank_unique_ids")
                                              .Output("cur_rank_unique_table_ids")
                                              .Output("cur_rank_inverse_indices")
                                              .Build());
    op_table_ids_no_in_has_out_ = CHECK_JUST(one::OpBuilder("id_shuffle")
                                                 .Input("ids")
                                                 .Output("num_unique_matrix")
                                                 .Output("inverse_unique_partition_indices")
                                                 .Output("cur_rank_num_unique")
                                                 .Output("cur_rank_unique_ids")
                                                 .Output("cur_rank_unique_table_ids")
                                                 .Output("cur_rank_inverse_indices")
                                                 .Build());
  }

  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& ids,
                                const Optional<one::Tensor>& table_ids,
                                const int32_t& num_tables) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("num_tables", num_tables));
    if (table_ids) {
      return OpInterpUtil::Dispatch<TensorTuple>(*op_table_ids_has_in_out_, {ids, JUST(table_ids)},
                                                 attrs);
    } else {
      return OpInterpUtil::Dispatch<TensorTuple>(*op_table_ids_no_in_has_out_, {ids}, attrs);
    }
  }

 private:
  std::shared_ptr<OpExpr> op_table_ids_has_in_out_;
  std::shared_ptr<OpExpr> op_table_ids_no_in_has_out_;
};

class OneEmbeddingEmbeddingShuffleFunctor {
 public:
  OneEmbeddingEmbeddingShuffleFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("embedding_shuffle")
                         .Input("cur_rank_embeddings")
                         .Input("num_unique_matrix")
                         .Input("cur_rank_inverse_indices")
                         .Input("inverse_unique_partition_indices")
                         .Output("embeddings")
                         .Build());
  }

  Maybe<Tensor> operator()(
      const std::shared_ptr<one::Tensor>& cur_rank_embeddings,
      const std::shared_ptr<one::Tensor>& num_unique_matrix,
      const std::shared_ptr<one::Tensor>& cur_rank_inverse_indices,
      const std::shared_ptr<one::Tensor>& inverse_unique_partition_indices) const {
    return OpInterpUtil::Dispatch<Tensor>(
        *op_, {cur_rank_embeddings, num_unique_matrix, cur_rank_inverse_indices,
               inverse_unique_partition_indices});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class OneEmbeddingEmbeddingGradientShuffleFunctor {
 public:
  OneEmbeddingEmbeddingGradientShuffleFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("embedding_gradient_shuffle")
                         .Input("embedding_grad")
                         .Input("num_unique_matrix")
                         .Input("cur_rank_inverse_indices")
                         .Input("inverse_unique_partition_indices")
                         .Output("cur_rank_unique_embedding_grad")
                         .Build());
  }

  Maybe<Tensor> operator()(
      const std::shared_ptr<one::Tensor>& embedding_grad,
      const std::shared_ptr<one::Tensor>& num_unique_matrix,
      const std::shared_ptr<one::Tensor>& cur_rank_inverse_indices,
      const std::shared_ptr<one::Tensor>& inverse_unique_partition_indices) const {
    return OpInterpUtil::Dispatch<Tensor>(
        *op_, {embedding_grad, num_unique_matrix, cur_rank_inverse_indices,
               inverse_unique_partition_indices});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class OneEmbeddingLookupFunctor {
 public:
  OneEmbeddingLookupFunctor() {
    op_has_table_ids_ = CHECK_JUST(one::OpBuilder("embedding_lookup_placeholder")
                                       .Input("shadow")
                                       .Input("ids")
                                       .Input("table_ids")
                                       .Output("embeddings")
                                       .Build());
    op_no_table_ids_ = CHECK_JUST(one::OpBuilder("embedding_lookup_placeholder")
                                      .Input("shadow")
                                      .Input("ids")
                                      .Output("embeddings")
                                      .Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& shadow,
                           const std::shared_ptr<one::Tensor>& ids,
                           const Optional<one::Tensor>& table_ids, const Symbol<DType>& dtype,
                           const int64_t embedding_size, const int32_t num_tables,
                           const std::string& embedding_tables,
                           const std::string& key_value_store_options) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<DataType>("dtype", dtype->data_type()));
    JUST(attrs.SetAttr<int64_t>("embedding_size", embedding_size));
    JUST(attrs.SetAttr<int32_t>("num_tables", num_tables));
    JUST(attrs.SetAttr<std::string>("embedding_tables", embedding_tables));
    JUST(attrs.SetAttr<std::string>("key_value_store_options", key_value_store_options));
    if (table_ids) {
      return OpInterpUtil::Dispatch<Tensor>(*op_has_table_ids_, {shadow, ids, JUST(table_ids)},
                                            attrs);
    } else {
      return OpInterpUtil::Dispatch<Tensor>(*op_no_table_ids_, {shadow, ids}, attrs);
    }
  }

 private:
  std::shared_ptr<OpExpr> op_has_table_ids_;
  std::shared_ptr<OpExpr> op_no_table_ids_;
};

class OneEmbeddingUniqueKeyValuePairFunctor {
 public:
  OneEmbeddingUniqueKeyValuePairFunctor() {
    op_has_input_value_ = CHECK_JUST(one::OpBuilder("unique_key_value_pair")
                                         .Input("keys")
                                         .Input("values")
                                         .Output("num_unique")
                                         .Output("unique_keys")
                                         .Output("unique_values")
                                         .Output("inverse_indices")
                                         .Build());
    op_no_input_value_ = CHECK_JUST(one::OpBuilder("unique_key_value_pair")
                                        .Input("keys")
                                        .Output("num_unique")
                                        .Output("unique_keys")
                                        .Output("unique_values")
                                        .Output("inverse_indices")
                                        .Build());
  }

  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& keys,
                                const Optional<one::Tensor>& values,
                                const int32_t num_tables) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("num_tables", num_tables));
    if (values) {
      return OpInterpUtil::Dispatch<TensorTuple>(*op_has_input_value_, {keys, JUST(values)}, attrs);
    } else {
      return OpInterpUtil::Dispatch<TensorTuple>(*op_no_input_value_, {keys}, attrs);
    }
  }

 private:
  std::shared_ptr<OpExpr> op_has_input_value_;
  std::shared_ptr<OpExpr> op_no_input_value_;
};

class OneEmbeddingSgdUpdateFunctor {
 public:
  OneEmbeddingSgdUpdateFunctor() {
    // This functor is just for unittest
    sgd_op_ = CHECK_JUST(one::OpBuilder("sgd_embedding_update")
                             .Input("num_unique_ids")
                             .Input("unique_embeddings")
                             .Input("embedding_grad")
                             .Input("learning_rate")
                             .Input("down_scale_by_tensor")
                             .Input("skip_if")
                             .Output("updated_unique_embeddings")
                             .Build());
    momentum_op_ = CHECK_JUST(one::OpBuilder("momentum_embedding_update")
                                  .Input("num_unique_ids")
                                  .Input("unique_embeddings")
                                  .Input("embedding_grad")
                                  .Input("learning_rate")
                                  .Input("down_scale_by_tensor")
                                  .Input("skip_if")
                                  .Output("updated_unique_embeddings")
                                  .Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& num_unique_ids,
                           const std::shared_ptr<one::Tensor>& unique_embeddings,
                           const std::shared_ptr<one::Tensor>& embedding_grad,
                           const std::shared_ptr<one::Tensor>& learning_rate,
                           const std::shared_ptr<one::Tensor>& down_scale_by_tensor,
                           const std::shared_ptr<one::Tensor>& skip_if, const double scale,
                           const float weight_decay, const float momentum) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<double>("scale", scale));
    JUST(attrs.SetAttr<float>("weight_decay", weight_decay));
    if (momentum == 0) {
      return OpInterpUtil::Dispatch<Tensor>(*sgd_op_,
                                            {num_unique_ids, unique_embeddings, embedding_grad,
                                             learning_rate, down_scale_by_tensor, skip_if},
                                            attrs);
    } else {
      JUST(attrs.SetAttr<float>("beta", momentum));
      return OpInterpUtil::Dispatch<Tensor>(*momentum_op_,
                                            {num_unique_ids, unique_embeddings, embedding_grad,
                                             learning_rate, down_scale_by_tensor, skip_if},
                                            attrs);
    }
  }

 private:
  std::shared_ptr<OpExpr> sgd_op_;
  std::shared_ptr<OpExpr> momentum_op_;
};

class OneEmbeddingAdamUpdateFunctor {
 public:
  OneEmbeddingAdamUpdateFunctor() {
    // This functor is just for unittest
    no_bias_correction_op_ = CHECK_JUST(one::OpBuilder("adam_embedding_update")
                                            .Input("num_unique_ids")
                                            .Input("unique_embeddings")
                                            .Input("embedding_grad")
                                            .Input("learning_rate")
                                            .Input("down_scale_by_tensor")
                                            .Input("skip_if")
                                            .Output("updated_unique_embeddings")
                                            .Build());
    do_bias_correction_op_ = CHECK_JUST(one::OpBuilder("adam_embedding_update")
                                            .Input("num_unique_ids")
                                            .Input("unique_embeddings")
                                            .Input("embedding_grad")
                                            .Input("learning_rate")
                                            .Input("down_scale_by_tensor")
                                            .Input("skip_if")
                                            .Input("bias_correction1")
                                            .Input("bias_correction2")
                                            .Output("updated_unique_embeddings")
                                            .Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& num_unique_ids,
                           const std::shared_ptr<one::Tensor>& unique_embeddings,
                           const std::shared_ptr<one::Tensor>& embedding_grad,
                           const std::shared_ptr<one::Tensor>& learning_rate,
                           const std::shared_ptr<one::Tensor>& down_scale_by_tensor,
                           const std::shared_ptr<one::Tensor>& skip_if,
                           const Optional<one::Tensor>& bias_correction1,
                           const Optional<one::Tensor>& bias_correction2, const double scale,
                           const float weight_decay, const float beta1, const float beta2,
                           const float epsilon, const bool do_bias_correction) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<double>("scale", scale));
    JUST(attrs.SetAttr<float>("weight_decay", weight_decay));
    JUST(attrs.SetAttr<float>("beta1", beta1));
    JUST(attrs.SetAttr<float>("beta2", beta2));
    JUST(attrs.SetAttr<float>("epsilon", epsilon));
    JUST(attrs.SetAttr<bool>("do_bias_correction", do_bias_correction));
    if (do_bias_correction) {
      CHECK(bias_correction1);
      CHECK(bias_correction2);
      return OpInterpUtil::Dispatch<Tensor>(
          *do_bias_correction_op_,
          {num_unique_ids, unique_embeddings, embedding_grad, learning_rate, down_scale_by_tensor,
           skip_if, JUST(bias_correction1), JUST(bias_correction2)},
          attrs);
    } else {
      return OpInterpUtil::Dispatch<Tensor>(*no_bias_correction_op_,
                                            {num_unique_ids, unique_embeddings, embedding_grad,
                                             learning_rate, down_scale_by_tensor, skip_if},
                                            attrs);
    }
  }

 private:
  std::shared_ptr<OpExpr> no_bias_correction_op_;
  std::shared_ptr<OpExpr> do_bias_correction_op_;
};

class OneEmbeddingAdagradUpdateFunctor {
 public:
  OneEmbeddingAdagradUpdateFunctor() {
    // This functor is just for unittest
    op_ = CHECK_JUST(one::OpBuilder("adagrad_embedding_update")
                         .Input("num_unique_ids")
                         .Input("unique_embeddings")
                         .Input("embedding_grad")
                         .Input("learning_rate")
                         .Input("down_scale_by_tensor")
                         .Input("skip_if")
                         .Input("train_step")
                         .Output("updated_unique_embeddings")
                         .Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& num_unique_ids,
                           const std::shared_ptr<one::Tensor>& unique_embeddings,
                           const std::shared_ptr<one::Tensor>& embedding_grad,
                           const std::shared_ptr<one::Tensor>& learning_rate,
                           const std::shared_ptr<one::Tensor>& down_scale_by_tensor,
                           const std::shared_ptr<one::Tensor>& skip_if,
                           const std::shared_ptr<one::Tensor>& train_step, const double scale,
                           const float weight_decay, const float lr_decay,
                           const float epsilon) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<double>("scale", scale));
    JUST(attrs.SetAttr<float>("weight_decay", weight_decay));
    JUST(attrs.SetAttr<float>("lr_decay", lr_decay));
    JUST(attrs.SetAttr<float>("epsilon", epsilon));
    return OpInterpUtil::Dispatch<Tensor>(
        *op_,
        {num_unique_ids, unique_embeddings, embedding_grad, learning_rate, down_scale_by_tensor,
         skip_if, train_step},
        attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class OneEmbeddingFtrlUpdateFunctor {
 public:
  OneEmbeddingFtrlUpdateFunctor() {
    // This functor is just for unittest
    op_ = CHECK_JUST(one::OpBuilder("ftrl_embedding_update")
                         .Input("num_unique_ids")
                         .Input("unique_embeddings")
                         .Input("embedding_grad")
                         .Input("learning_rate")
                         .Input("down_scale_by_tensor")
                         .Input("skip_if")
                         .Output("updated_unique_embeddings")
                         .Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& num_unique_ids,
                           const std::shared_ptr<one::Tensor>& unique_embeddings,
                           const std::shared_ptr<one::Tensor>& embedding_grad,
                           const std::shared_ptr<one::Tensor>& learning_rate,
                           const std::shared_ptr<one::Tensor>& down_scale_by_tensor,
                           const std::shared_ptr<one::Tensor>& skip_if, const double scale,
                           const float weight_decay, const float lr_power, const float lambda1,
                           const float lambda2, const float beta) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<double>("scale", scale));
    JUST(attrs.SetAttr<float>("weight_decay", weight_decay));
    JUST(attrs.SetAttr<float>("lr_power", lr_power));
    JUST(attrs.SetAttr<float>("lambda1", lambda1));
    JUST(attrs.SetAttr<float>("lambda2", lambda2));
    JUST(attrs.SetAttr<float>("beta", beta));
    return OpInterpUtil::Dispatch<Tensor>(*op_,
                                          {num_unique_ids, unique_embeddings, embedding_grad,
                                           learning_rate, down_scale_by_tensor, skip_if},
                                          attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class RocAucScoreFunctor {
 public:
  RocAucScoreFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("roc_auc_score").Input("label").Input("pred").Output("out").Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& label,
                           const std::shared_ptr<one::Tensor>& pred) const {
    return OpInterpUtil::Dispatch<Tensor>(*op_, {label, pred});
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
  m.add_functor<impl::FusedMLPFunctor>("FusedMLP");
  m.add_functor<impl::LayerNormFunctor>("LayerNorm");
  m.add_functor<impl::LayerNormAffineFunctor>("LayerNormAffine");
  m.add_functor<impl::TFAvgPool2DFunctor>("TFAvgPool2D");
  m.add_functor<impl::MaxPool1DFunctor>("MaxPool1D");
  m.add_functor<impl::MaxPool2DFunctor>("MaxPool2D");
  m.add_functor<impl::MaxPool3DFunctor>("MaxPool3D");
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
  m.add_functor<impl::PixelShuffleFunctor>("PixelShuffle");
  m.add_functor<impl::AvgPool1DFunctor>("AvgPool1D");
  m.add_functor<impl::AvgPool2DFunctor>("AvgPool2D");
  m.add_functor<impl::AvgPool3DFunctor>("AvgPool3D");
  m.add_functor<impl::UnfoldFunctor>("Unfold");
  m.add_functor<impl::FoldFunctor>("Fold");
  m.add_functor<impl::OneHotFunctor>("OneHot");
  m.add_functor<impl::FusedSelfAttentionFunctor>("FusedSelfAttention");
  m.add_functor<impl::FusedSelfAttentionGradFunctor>("FusedSelfAttentionGrad");
  m.add_functor<impl::CosineSimilarityFunctor>("CosineSimilarity");
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
  m.add_functor<impl::PariticalFCSampleDisableBoxing>("DistributedPariticalFCSampleDisableBoxing");
  m.add_functor<impl::NmsFunctor>("Nms");
  m.add_functor<impl::RoiAlignFunctor>("RoiAlign");
  m.add_functor<impl::RoiAlignGradFunctor>("RoiAlignGrad");
  m.add_functor<impl::FusedDotFeatureInteractionFunctor>("FusedDotFeatureInteraction");
  m.add_functor<impl::OneEmbeddingIdShuffleFunctor>("OneEmbeddingIdShuffle");
  m.add_functor<impl::OneEmbeddingEmbeddingShuffleFunctor>("OneEmbeddingEmbeddingShuffle");
  m.add_functor<impl::OneEmbeddingEmbeddingGradientShuffleFunctor>(
      "OneEmbeddingEmbeddingGradientShuffle");
  m.add_functor<impl::OneEmbeddingLookupFunctor>("OneEmbeddingLookup");
  m.add_functor<impl::OneEmbeddingUniqueKeyValuePairFunctor>("OneEmbeddingUniqueKeyValuePair");
  m.add_functor<impl::NormalFunctor>("Normal");
  m.add_functor<impl::ConsistentNormalFunctor>("ConsistentNormal");
  m.add_functor<impl::OneEmbeddingSgdUpdateFunctor>("OneEmbeddingSgdUpdate");
  m.add_functor<impl::OneEmbeddingAdamUpdateFunctor>("OneEmbeddingAdamUpdate");
  m.add_functor<impl::OneEmbeddingAdagradUpdateFunctor>("OneEmbeddingAdagradUpdate");
  m.add_functor<impl::OneEmbeddingFtrlUpdateFunctor>("OneEmbeddingFtrlUpdate");
  m.add_functor<impl::RocAucScoreFunctor>("RocAucScore");
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow
