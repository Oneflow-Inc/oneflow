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

#include "oneflow/core/framework/mutable_attr_map.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/tensor_util.h"
#include "oneflow/core/functional/function_library.h"
#include "oneflow/core/functional/sequence_function.h"
#include "oneflow/core/functional/impl/common.h"
#include "oneflow/core/functional/impl/unary_functor.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/user/kernels/random_mask_like_kernel.h"
#include "oneflow/user/kernels/dropout_kernel.h"
#include "oneflow/user/kernels/distributions/common.h"
#include "oneflow/user/kernels/random_seed_util.h"

#include "oneflow/core/common/container_util.h"
#include "fmt/core.h"

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
    int32_t axis_val = axis;
    if (axis_val < 0) {
      const int64_t num_axes = x->shape()->NumAxes();
      axis_val += num_axes;
    }
    CHECK_LT_OR_RETURN(axis_val, x->shape()->NumAxes())
        << Error::IndexError() << "Dimension out of range (expected to be in range of [-"
        << x->shape()->NumAxes() << "," << x->shape()->NumAxes() - 1 << "], but got " << axis_val
        << ")";
    CHECK_EQ_OR_RETURN(x->shape()->At(axis_val), bias->shape()->At(0))
        << Error::RuntimeError() << "The size of tensor x " << x->shape()->ToString()
        << " must match the size of tensor b " << bias->shape()->ToString() << " at dimension "
        << axis_val;
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("axis");
    attrs.SetAllAttrs(axis_val);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x, bias}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ConvBaseFunctor {
 public:
  explicit ConvBaseFunctor(const int& num_spatial_dims) : num_spatial_dims_(num_spatial_dims) {
    bias_op_ = CHECK_JUST(one::OpBuilder("bias_add").Input("a").Input("b").Output("out").Build());
    enable_fused_conv_bias_ = ParseBooleanFromEnv("ONEFLOW_KERNEL_ENABLE_FUSED_CONV_BIAS", false);
  }
  virtual ~ConvBaseFunctor() = default;
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& weight,
                           const Optional<one::Tensor>& bias, const std::vector<int32_t>& stride,
                           const std::vector<int32_t>& padding,
                           const std::vector<int32_t>& dilation, const int32_t& groups,
                           const std::string& channel_pos) const {
    std::shared_ptr<one::Tensor> unsqueezed_input;
    bool is_batched = true;
    std::string func_name;
    if (num_spatial_dims_ == 1) {
      func_name = "conv1d";
    } else if (num_spatial_dims_ == 2) {
      func_name = "conv2d";
    } else {
      func_name = "conv3d";
    }
    std::tie(unsqueezed_input, is_batched) = *JUST(batchify(input, num_spatial_dims_, func_name));
    std::vector<int32_t> kernel_size_vec(num_spatial_dims_);
    int32_t channel_idx = 1;
    int32_t kernel_idx_offset = 2;
    if (channel_pos == "channels_last") {
      kernel_idx_offset = 1;
      channel_idx = kernel_idx_offset + num_spatial_dims_;
    }

    for (int i = 0; i < num_spatial_dims_; i++) {
      kernel_size_vec.at(i) = ((weight->shape())->At(i + kernel_idx_offset));
    }
    auto& conv_attrs =
        THREAD_CACHED_MUTABLE_ATTR_MAP("filters", "kernel_size", "padding_before", "strides",
                                       "dilation_rate", "groups", "data_format");
    conv_attrs.SetAllAttrs(static_cast<int32_t>(weight->shape()->At(0)), kernel_size_vec, padding,
                           stride, dilation, groups, channel_pos);
    if (bias && enable_fused_conv_bias_) {
      return OpInterpUtil::Dispatch<Tensor>(*conv_bias_op_, {input, weight, JUST(bias)},
                                            conv_attrs);
    }
    const std::shared_ptr<one::Tensor>& conv_out =
        JUST(OpInterpUtil::Dispatch<Tensor>(*conv_op_, {unsqueezed_input, weight}, conv_attrs));
    std::shared_ptr<one::Tensor> squeezed_conv_output = conv_out;
    if (!is_batched) {
      squeezed_conv_output = JUST(functional::Squeeze(conv_out, std::vector<int32_t>{0}));
      channel_idx -= 1;
    }
    if (bias) {
      return functional::BiasAdd(squeezed_conv_output, JUST(bias), channel_idx);
    } else {
      return squeezed_conv_output;
    }
  }

 protected:
  std::shared_ptr<OpExpr> conv_op_;
  std::shared_ptr<OpExpr> bias_op_;
  std::shared_ptr<OpExpr> conv_bias_op_;
  int32_t num_spatial_dims_;
  bool enable_fused_conv_bias_;
};

class Conv1dFunctor : public ConvBaseFunctor {
 public:
  Conv1dFunctor() : ConvBaseFunctor(/*num_spatial_dims_=*/1) {
    conv_op_ =
        CHECK_JUST(one::OpBuilder("conv1d").Input("in").Input("weight").Output("out").Build());
    conv_bias_op_ = CHECK_JUST(
        one::OpBuilder("conv1d").Input("in").Input("weight").Input("bias").Output("out").Build());
  }
};

class Conv2dFunctor : public ConvBaseFunctor {
 public:
  Conv2dFunctor() : ConvBaseFunctor(/*num_spatial_dims_=*/2) {
    conv_op_ =
        CHECK_JUST(one::OpBuilder("conv2d").Input("in").Input("weight").Output("out").Build());
    conv_bias_op_ = CHECK_JUST(
        one::OpBuilder("conv2d").Input("in").Input("weight").Input("bias").Output("out").Build());
  }
};

class Conv3dFunctor : public ConvBaseFunctor {
 public:
  Conv3dFunctor() : ConvBaseFunctor(/*num_spatial_dims_=*/3) {
    conv_op_ =
        CHECK_JUST(one::OpBuilder("conv3d").Input("in").Input("weight").Output("out").Build());
    conv_bias_op_ = CHECK_JUST(
        one::OpBuilder("conv3d").Input("in").Input("weight").Input("bias").Output("out").Build());
  }
};

class DeConvBaseFunctor {
 public:
  explicit DeConvBaseFunctor(const int& num_spatial_dims) : num_spatial_dims_(num_spatial_dims) {
    bias_op_ = CHECK_JUST(one::OpBuilder("bias_add").Input("a").Input("b").Output("out").Build());
  }
  virtual ~DeConvBaseFunctor() = default;
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& weight,
                           const Optional<one::Tensor>& bias, const std::vector<int32_t>& stride,
                           const std::vector<int32_t>& padding,
                           const std::vector<int32_t>& output_padding, const int32_t& groups,
                           const std::vector<int32_t>& dilation,
                           const std::string& data_format) const {
    std::shared_ptr<one::Tensor> unsqueezed_input;
    bool is_batched = true;
    std::string func_name;
    if (num_spatial_dims_ == 1) {
      func_name = "deconv1d";
    } else if (num_spatial_dims_ == 2) {
      func_name = "deconv2d";
    } else {
      func_name = "deconv3d";
    }
    std::tie(unsqueezed_input, is_batched) = *JUST(batchify(input, num_spatial_dims_, func_name));
    int32_t channel_idx = 1;
    std::vector<int32_t> kernel_size_vec(num_spatial_dims_);
    int32_t kernel_idx_offset = 2;
    if (data_format == "channels_last") { kernel_idx_offset = 1; }
    for (int i = 0; i < num_spatial_dims_; i++) {
      kernel_size_vec[i] = ((weight->shape())->At(i + kernel_idx_offset));
    }

    auto& deconv_attrs =
        THREAD_CACHED_MUTABLE_ATTR_MAP("filters", "kernel_size", "padding_before", "output_padding",
                                       "strides", "dilation_rate", "groups", "data_format");
    deconv_attrs.SetAllAttrs(static_cast<int32_t>(weight->shape()->At(1) * groups), kernel_size_vec,
                             padding, output_padding, stride, dilation, groups, data_format);
    std::shared_ptr<one::Tensor> deconv_out =
        JUST(OpInterpUtil::Dispatch<Tensor>(*deconv_op_, {unsqueezed_input, weight}, deconv_attrs));
    std::shared_ptr<one::Tensor> squeezed_deconv_output = deconv_out;
    if (!is_batched) {
      squeezed_deconv_output = JUST(functional::Squeeze(deconv_out, std::vector<int32_t>{0}));
      channel_idx -= 1;
    }
    if (bias) {
      auto& bias_attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("axis");
      bias_attrs.SetAllAttrs(static_cast<int32_t>(channel_idx));
      return OpInterpUtil::Dispatch<Tensor>(*bias_op_, {squeezed_deconv_output, JUST(bias)},
                                            bias_attrs);
    } else {
      return squeezed_deconv_output;
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

class EmbeddingReNormFunctor {
 public:
  EmbeddingReNormFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("embedding_renorm").Input("in").Input("indices").Output("out").Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& in,
                           const std::shared_ptr<one::Tensor>& indices, const double& max_norm,
                           const double& norm_type) const {
    CHECK_EQ_OR_RETURN(in->ndim(), 2)
        << Error::RuntimeError() << "The dimension of input should be 2.";
    std::shared_ptr<TensorTuple> outputs = std::make_shared<TensorTuple>(1);
    JUST(oneflow::VectorAt(*outputs, 0)) = in;

    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("max_norm", "norm_type");
    attrs.SetAllAttrs(max_norm, norm_type);

    JUST(OpInterpUtil::Dispatch(*op_, {in, indices}, outputs.get(), attrs));
    return JUST(oneflow::VectorAt(*outputs, 0));
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class EmbeddingFunctor {
 public:
  EmbeddingFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("embedding").Input("weight").Input("indices").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& weight,
                           const std::shared_ptr<one::Tensor>& indices,
                           const Optional<int64_t>& padding_idx,
                           const bool& scale_grad_by_freq) const {
    CHECK_EQ_OR_RETURN(weight->ndim(), 2) << "The dimension of weight should be 2";
    int64_t new_padding_idx = -1;
    if (padding_idx.has_value()) { new_padding_idx = JUST(padding_idx); }
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("padding_idx", "scale_grad_by_freq");
    attrs.SetAllAttrs(new_padding_idx, scale_grad_by_freq);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {weight, indices}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class MatMulNoBroadCastFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& mat2) const {
    const auto& input_shape = input->shape();
    const auto& mat2_shape = mat2->shape();
    CHECK_EQ_OR_RETURN(input_shape->NumAxes(), 2)
        << Error::RuntimeError() << "self must be a matrix";
    CHECK_EQ_OR_RETURN(mat2_shape->NumAxes(), 2)
        << Error::RuntimeError() << "mat2 must be a matrix";
    CHECK_EQ_OR_RETURN(input_shape->at(1), mat2_shape->at(0))
        << Error::RuntimeError() << "mat1 and mat2 shapes cannot be multiplied ("
        << std::to_string(input_shape->at(0)) << "x" << std::to_string(input_shape->at(1))
        << " and " << std::to_string(mat2_shape->at(0)) << "x" << std::to_string(mat2_shape->at(1))
        << ")";
    return JUST(functional::MatMul(input, mat2, false, false, 1.0));
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
    CHECK_GE_OR_RETURN(a_shape->NumAxes(), 1)
        << Error::RuntimeError() << "Tensor a's dim should >= 1";
    CHECK_GE_OR_RETURN(b_shape->NumAxes(), 1)
        << Error::RuntimeError() << "Tensor b's dim should >= 1";

    DeviceType device_type{};
    if (a->is_global()) {
      device_type = JUST(a->parallel_desc())->device_type();
    } else {
      device_type = JUST(a->device())->enum_type();
    }
    std::shared_ptr<one::Tensor> cast_a = a;
    std::shared_ptr<one::Tensor> cast_b = b;
    std::shared_ptr<one::Tensor> result;
    if ((cast_a->dtype()->is_integer()) && (device_type == DeviceType::kCPU)) {
      cast_a = JUST(functional::Cast(a, JUST(DType::Get(DataType::kFloat)), /*pin_memory=*/false));
      cast_b = JUST(functional::Cast(b, JUST(DType::Get(DataType::kFloat)), /*pin_memory=*/false));
    }

    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("transpose_a", "transpose_b", "alpha");
    attrs.SetAllAttrs(transpose_a, transpose_b, alpha);
    const int64_t a_num_axes = a_shape->NumAxes();
    const int64_t b_num_axes = b_shape->NumAxes();
    if (a_num_axes == 1 && b_num_axes == 2) {
      result = JUST(VectorMatrixProduct(cast_a, cast_b));
    } else if (a_num_axes == 2 && b_num_axes == 1) {
      result = JUST(MatrixVectorProduct(cast_a, cast_b));
    } else if (a_num_axes == 2 && b_num_axes == 2) {
      result = JUST(OpInterpUtil::Dispatch<Tensor>(*matmul_op_, {cast_a, cast_b}, attrs));
    } else if (a_num_axes == b_num_axes) {
      bool if_batch_matmul = true;
      for (int i = 0; i < a_num_axes - 2; ++i) {
        if (a_shape->At(i) != b_shape->At(i)) {
          if_batch_matmul = false;
          break;
        }
      }
      if (if_batch_matmul) {
        result = JUST(OpInterpUtil::Dispatch<Tensor>(*batch_matmul_op_, {cast_a, cast_b}, attrs));
      } else {
        result = JUST(OpInterpUtil::Dispatch<Tensor>(*bcast_matmul_op_, {cast_a, cast_b}, attrs));
      }
    } else {
      result = JUST(OpInterpUtil::Dispatch<Tensor>(*bcast_matmul_op_, {cast_a, cast_b}, attrs));
    }

    if ((a->dtype()->is_integer()) && (device_type == DeviceType::kCPU)) {
      return JUST(functional::Cast(result, a->dtype(), /*pin_memory=*/false));
    } else {
      return result;
    }
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
    const int64_t matmul_dim_a = transpose_a ? a_shape->At(1) : a_shape->At(2);
    const int64_t matmul_dim_b = transpose_b ? b_shape->At(2) : b_shape->At(1);
    CHECK_EQ_OR_RETURN(matmul_dim_a, matmul_dim_b)
        << Error::RuntimeError() << "Matmul dim not match, got " << matmul_dim_a << " of mat1 and "
        << matmul_dim_b << " of mat2, please check input!";
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("transpose_a", "transpose_b", "alpha");
    attrs.SetAllAttrs(transpose_a, transpose_b, alpha);

    DeviceType device_type{};
    if (a->is_global()) {
      device_type = JUST(a->parallel_desc())->device_type();
    } else {
      device_type = JUST(a->device())->enum_type();
    }
    std::shared_ptr<one::Tensor> cast_a = a;
    std::shared_ptr<one::Tensor> cast_b = b;
    if ((a->dtype()->is_integer()) && (device_type == DeviceType::kCPU)) {
      cast_a = JUST(functional::Cast(a, JUST(DType::Get(DataType::kFloat)), /*pin_memory=*/false));
      cast_b = JUST(functional::Cast(b, JUST(DType::Get(DataType::kFloat)), /*pin_memory=*/false));
    }

    auto result = JUST(OpInterpUtil::Dispatch<Tensor>(*batch_matmul_op_, {cast_a, cast_b}, attrs));
    if ((a->dtype()->is_integer()) && (device_type == DeviceType::kCPU)) {
      return JUST(functional::Cast(result, a->dtype(), /*pin_memory=*/false));
    } else {
      return result;
    }
  }

 private:
  std::shared_ptr<OpExpr> batch_matmul_op_;
};

class VectorMatrixProductFunctor {
 public:
  VectorMatrixProductFunctor() {
    vector_matrix_product_op_ = CHECK_JUST(
        one::OpBuilder("vector_matrix_product").Input("a").Input("b").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& vec,
                           const std::shared_ptr<one::Tensor>& input) const {
    const auto& vec_shape = vec->shape();
    const auto& input_shape = input->shape();
    CHECK_OR_RETURN(input_shape->NumAxes() == 2 && vec_shape->NumAxes() == 1)
        << Error::RuntimeError() << "vector @ matrix expected, got "
        << "1, " << input_shape->NumAxes() << ", " << vec_shape->NumAxes();
    CHECK_EQ_OR_RETURN(vec_shape->at(0), input_shape->at(0))
        << Error::RuntimeError() << "size mismatch, got " << 1 << ", "
        << std::to_string(vec_shape->at(0)) << " x " << std::to_string(input_shape->at(0)) << ", "
        << std::to_string(input_shape->at(1));
    return OpInterpUtil::Dispatch<Tensor>(*vector_matrix_product_op_, {vec, input});
  }

 private:
  std::shared_ptr<OpExpr> vector_matrix_product_op_;
};

class TensorDotIntDimsFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b,
                           const int32_t dims) const {
    CHECK_GE_OR_RETURN(dims, 0) << Error::RuntimeError()
                                << "tensordot expects dims >= 0, but got dims=" << dims;
    CHECK_LE_OR_RETURN(dims, a->ndim())
        << Error::RuntimeError() << "tensordot expects dims <= a.ndim which is " << a->ndim()
        << ", but got " << dims;
    CHECK_LE_OR_RETURN(dims, b->ndim())
        << Error::RuntimeError() << "tensordot expects dims <= b.ndim which is " << b->ndim()
        << ", but got " << dims;
    std::vector<int32_t> dot_dims_a(dims), dot_dims_b(dims);
    for (int32_t i = 0; i < dims; i++) {
      dot_dims_a[i] = a->ndim() - dims + i;
      dot_dims_b[i] = i;
    }
    return JUST(functional::TensorDot(a, b, dot_dims_a, dot_dims_b));
  }
};

class TensorDotFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b,
                           const std::vector<int32_t>& dims_a,
                           const std::vector<int32_t>& dims_b) const {
    // dims_a and dims_b represent dim indices to calculate dot, and are copied to variables
    // dot_dims_a and dot_dims_b when they need to be modified
    CHECK_EQ_OR_RETURN(dims_a.size(), dims_b.size())
        << Error::RuntimeError() << "both dimension lists should have same length, got "
        << dims_a.size() << " and " << dims_b.size();

    // dims_a.size() == dims_b.size(), and specially treat if both are empty
    if (dims_a.empty()) {
      DimVector shape_sum(a->ndim() + b->ndim());
      for (int64_t i = 0; i < a->ndim(); i++) { shape_sum[i] = a->shape()->At(i); }
      for (int64_t i = 0; i < b->ndim(); i++) { shape_sum[i + a->ndim()] = b->shape()->At(i); }
      std::shared_ptr<Tensor> reshape_a = JUST(Reshape(a, Shape(DimVector{-1, 1})));
      std::shared_ptr<Tensor> reshape_b = JUST(Reshape(b, Shape(DimVector{1, -1})));
      return JUST(Reshape(JUST(functional::MatMul(reshape_a, reshape_b, false, false, 1.0)),
                          Shape(DimVector(shape_sum.begin(), shape_sum.end()))));
    }
    std::vector<int32_t> dot_dims_a(dims_a.begin(), dims_a.end());
    std::vector<int32_t> dot_dims_b(dims_b.begin(), dims_b.end());
    for (int64_t i = 0; i < dot_dims_a.size(); i++) {
      dot_dims_a[i] = JUST(maybe_wrap_dim(dot_dims_a[i], a->ndim()));
      dot_dims_b[i] = JUST(maybe_wrap_dim(dot_dims_b[i], b->ndim()));
    }
    std::vector<bool> if_dot_dims_a(a->ndim(), false);
    std::vector<bool> if_dot_dims_b(b->ndim(), false);
    for (const int32_t dim_idx : dot_dims_a) {
      CHECK_EQ_OR_RETURN(if_dot_dims_a[dim_idx], false)
          << Error::RuntimeError() << "dim " << dim_idx
          << " appears multiple times in the list of dims";
      if_dot_dims_a[dim_idx] = true;
    }
    for (const int32_t dim_idx : dot_dims_b) {
      CHECK_EQ_OR_RETURN(if_dot_dims_b[dim_idx], false)
          << Error::RuntimeError() << "dim " << dim_idx
          << " appears multiple times in the list of dims";
      if_dot_dims_b[dim_idx] = true;
    }

    std::vector<int32_t> broadcast_dims_a, broadcast_dims_b;
    for (int64_t i = 0; i < dot_dims_a.size(); i++) {
      int64_t size_a = a->shape()->At(dot_dims_a[i]);
      int64_t size_b = b->shape()->At(dot_dims_b[i]);
      if (size_a == 1 && size_b > 1) {
        broadcast_dims_b.emplace_back(dot_dims_b[i]);
      } else if (size_b == 1 && size_a > 1) {
        broadcast_dims_a.emplace_back(dot_dims_a[i]);
      } else {
        CHECK_EQ_OR_RETURN(size_a, size_b)
            << Error::RuntimeError() << "contracted dimensions need to match, but first has size "
            << size_a << " in dim " << dot_dims_a[i] << " and second has size " << size_b
            << " in dim " << dot_dims_b[i];
      }
    }

    // calculate ReduceSum for broadcasting of some axis
    std::shared_ptr<Tensor> reduced_sum_a = a;
    std::shared_ptr<Tensor> reduced_sum_b = b;
    if (!broadcast_dims_a.empty())
      reduced_sum_a = JUST(functional::ReduceSum(a, broadcast_dims_a, true));
    if (!broadcast_dims_b.empty())
      reduced_sum_b = JUST(functional::ReduceSum(b, broadcast_dims_b, true));

    // int64_t non_dot_size_a = 1, non_dot_size_b = 1;
    std::vector<int32_t> non_dot_shape_a, non_dot_shape_b;
    non_dot_shape_a.reserve(a->ndim() - dot_dims_a.size() + b->ndim() - dot_dims_b.size());
    non_dot_shape_b.reserve(b->ndim() - dot_dims_b.size());

    std::vector<int32_t> permuted_dims_a, permuted_dims_b;
    permuted_dims_a.reserve(a->ndim());
    permuted_dims_b.reserve(b->ndim());

    for (int32_t i = 0; i < a->ndim(); i++) {
      if (!if_dot_dims_a[i]) {
        permuted_dims_a.emplace_back(i);
        // non_dot_size_a *= reduced_sum_a->shape()->At(i);
        non_dot_shape_a.emplace_back(reduced_sum_a->shape()->At(i));
      }
    }

    for (const int32_t dim_idx : dot_dims_a) permuted_dims_a.emplace_back(dim_idx);
    for (const int32_t dim_idx : dot_dims_b) permuted_dims_b.emplace_back(dim_idx);

    for (int32_t i = 0; i < b->ndim(); i++) {
      if (!if_dot_dims_b[i]) {
        permuted_dims_b.emplace_back(i);
        // non_dot_size_b *= reduced_sum_b->shape()->At(i);
        non_dot_shape_b.emplace_back(reduced_sum_b->shape()->At(i));
      }
    }
    non_dot_shape_a.insert(non_dot_shape_a.end(), non_dot_shape_b.begin(), non_dot_shape_b.end());

    int64_t dot_size = 1;
    for (const int32_t dim_idx : dot_dims_a) dot_size *= reduced_sum_a->shape()->At(dim_idx);
    std::shared_ptr<Tensor> permuted_a = JUST(
        Reshape(JUST(Permute(reduced_sum_a, permuted_dims_a)), Shape(DimVector({-1, dot_size}))));
    std::shared_ptr<Tensor> permuted_b = JUST(
        Reshape(JUST(Permute(reduced_sum_b, permuted_dims_b)), Shape(DimVector({dot_size, -1}))));

    return Reshape(JUST(functional::MatMul(permuted_a, permuted_b, false, false, 1.0)),
                   Shape(DimVector({non_dot_shape_a.begin(), non_dot_shape_a.end()})));
  }
};

class FusedMLPFunctor {
 public:
  FusedMLPFunctor() {
#if CUDA_VERSION >= 11060
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
    CHECK_GE_OR_RETURN(weight_size, 1)
        << Error::RuntimeError() << "The number of weights should be greater equal than 1. ";
    CHECK_EQ_OR_RETURN(weight_size, bias_size)
        << Error::RuntimeError() << "The number of weights should be equal to biases. ";
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
      CHECK_EQ_OR_RETURN(weight_shape->NumAxes(), 2)
          << Error::RuntimeError() << "Weight's dim size should == 2";
      CHECK_EQ_OR_RETURN(bias_shape->NumAxes(), 1)
          << Error::RuntimeError() << "Bias's dim size should == 1";

      n = weight_shape->At(0);
      CHECK_EQ_OR_RETURN(bias_shape->At(0), n)
          << Error::RuntimeError() << "Bias's dim is not equal to weight's first dim. ";
      CHECK_EQ_OR_RETURN(weight_shape->At(1), k)
          << Error::RuntimeError() << "weight's second dim should be equal to input's second dim. ";

      // Set for next layer.
      k = n;
    }

#if CUDA_VERSION >= 11060
    DeviceType device_type{};
    if (x->is_global()) {
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

      auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("skip_final_activation");
      attrs.SetAllAttrs(skip_final_activation);
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
        When it is not last dense layer, or it is last dense layer and skip_final_activate=False,
        we add relu Layer.
        */
        out = JUST(functional::Relu(out, false));
      }
    }
    return out;
  }

 private:
#if CUDA_VERSION >= 11060
  std::vector<std::shared_ptr<OpExpr>> fused_op_;
#endif
};

class FusedMatmulBiasFunctor {
 public:
  FusedMatmulBiasFunctor() {
    _with_add_to_output_op = CHECK_JUST(one::OpBuilder("fused_matmul_bias")
                                            .Input("x")
                                            .Input("weight")
                                            .Input("bias")
                                            .Input("_add_to_output")
                                            .Output("out")
                                            .Build());
    _without_add_to_output_op = CHECK_JUST(one::OpBuilder("fused_matmul_bias")
                                               .Input("x")
                                               .Input("weight")
                                               .Input("bias")
                                               .Output("out")
                                               .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& weight,
                           const std::shared_ptr<one::Tensor>& bias,
                           const Optional<one::Tensor>& _add_to_output, const double& alpha,
                           const double& beta) const {
    /*
    x: (m_i, ... m_0, k)
    weight: (n, k) need transpose
    bias: (n)
    */
    const auto& x_shape = x->shape();
    const int64_t k = x_shape->At(x->shape()->NumAxes() - 1);

    const auto& weight_shape = weight->shape();
    const auto& bias_shape = bias->shape();

    CHECK_EQ_OR_RETURN(weight_shape->NumAxes(), 2)
        << Error::RuntimeError() << "Weight's dim size should == 2";
    CHECK_EQ_OR_RETURN(bias_shape->NumAxes(), 1)
        << Error::RuntimeError() << "Bias's dim size should == 1";

    const int64_t n = weight_shape->At(0);
    CHECK_EQ_OR_RETURN(bias_shape->At(0), n)
        << Error::RuntimeError() << "Bias's dim is not equal to weight's first dim. ";
    CHECK_EQ_OR_RETURN(weight_shape->At(1), k)
        << Error::RuntimeError() << "weight's second dim should be equal to input's second dim. ";

#if CUDA_VERSION >= 11020
    DeviceType device_type{};
    if (x->is_global()) {
      device_type = JUST(x->parallel_desc())->device_type();
    } else {
      device_type = JUST(x->device())->enum_type();
    }

    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("alpha", "beta");
    attrs.SetAllAttrs(alpha, beta);
    if (device_type == DeviceType::kCUDA) {
      if (_add_to_output) {
        return OpInterpUtil::Dispatch<Tensor>(*_with_add_to_output_op,
                                              {x, weight, bias, JUST(_add_to_output)}, attrs);
      } else {
        return OpInterpUtil::Dispatch<Tensor>(*_without_add_to_output_op, {x, weight, bias}, attrs);
      }
    }
#endif  // CUDA_VERSION >= 11020

    auto matmul_bias = JUST(functional::BiasAdd(
        JUST(functional::MatMul(x, weight, false, true, alpha)), bias, x->shape()->NumAxes() - 1));
    if (_add_to_output && beta != 0.0) {
      if (beta == 1.0) {
        return JUST(functional::Add({matmul_bias, JUST(_add_to_output)}, false));
      } else {
        return JUST(functional::Add(
            {matmul_bias, JUST(functional::ScalarMul(JUST(_add_to_output), beta, false))}, false));
      }
    } else {
      return matmul_bias;
    }
  }

 private:
  std::shared_ptr<OpExpr> _with_add_to_output_op;
  std::shared_ptr<OpExpr> _without_add_to_output_op;
};

class FusedMatmulBiasAddReluDropoutFunctor {
 public:
  FusedMatmulBiasAddReluDropoutFunctor() {
#if CUDA_VERSION >= 11060
    fused_op_.resize(kMaxInputCount /*the maximum number of inputs*/);
    for (int n = 1; n < fused_op_.size(); ++n) {
      fused_op_[n] = CHECK_JUST(one::OpBuilder("fused_matmul_bias_add_relu_dropout")
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
                           const TensorTuple& biases, bool skip_final_activation,
                           const std::vector<float>& dropout_rate_list,
                           const Optional<one::Generator>& generator) const {
    const int64_t weight_size = weights.size();
    const int64_t bias_size = biases.size();
    CHECK_GE_OR_RETURN(weight_size, 1)
        << Error::RuntimeError() << "The number of weights should be greater equal than 1. ";
    CHECK_EQ_OR_RETURN(weight_size, bias_size)
        << Error::RuntimeError() << "The number of weights should be equal to biases. ";
    CHECK_EQ_OR_RETURN(weight_size, dropout_rate_list.size())
        << Error::RuntimeError()
        << "The dropout rate list length should be equal to the number of weights. ";
    int64_t n = 0, k = 0;
    /*
    x: (m, k)
    weight: (n, k) need transpose
    bias: (n)
    */
    const auto& x_shape = x->shape();
    k = x_shape->At(1);
    for (int64_t i = 0; i < weight_size; i++) {
      CHECK_GE_OR_RETURN(dropout_rate_list[i], 0.0f)
          << Error::RuntimeError() << "Dropout rate should be >= 0.0";

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

    auto gen = generator.value_or(JUST(one::DefaultAutoGenerator()));

#if CUDA_VERSION >= 11060
    DeviceType device_type{};
    if (x->is_global()) {
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

      gen = JUST(GetGeneratorForLazyOrGlobal(gen, LazyMode::is_enabled(), x));
      auto& attrs =
          THREAD_CACHED_MUTABLE_ATTR_MAP("skip_final_activation", "seed", "dropout_rate_list");
      attrs.SetAllAttrs(skip_final_activation, static_cast<int64_t>(gen->current_seed()),
                        dropout_rate_list);
      const auto& dropout_state = std::make_shared<FusedDropoutKernelState>(gen);
      return OpInterpUtil::Dispatch<Tensor>(*fused_op_[weight_size], input,
                                            OpExprInterpContext(attrs, dropout_state));
    }
#endif  // CUDA_VERSION >= 11060

    // Fall back to Naive matmul + bias_add + relu + dropout
    std::shared_ptr<one::Tensor> out = x;
    for (int32_t layer_idx = 0; layer_idx < weight_size; layer_idx++) {
      out = JUST(
          functional::BiasAdd(JUST(functional::MatMul(out, weights[layer_idx], false, true, 1.0)),
                              biases[layer_idx], 1));
      if ((layer_idx != weight_size - 1) || !skip_final_activation) {
        out = JUST(functional::Relu(out, false));
        out = JUST(functional::Dropout(out, JUST(VectorAt(dropout_rate_list, layer_idx)),
                                       /*training=*/true,
                                       /*inplace=*/false,
                                       /*generator=*/gen, /*addend=*/NullOpt));
      } else {
        out = JUST(functional::Dropout(out, JUST(VectorAt(dropout_rate_list, layer_idx)),
                                       /*training=*/true,
                                       /*inplace=*/false,
                                       /*generator=*/gen, /*addend=*/NullOpt));
      }
    }
    return out;
  }

 private:
#if CUDA_VERSION >= 11060
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
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("begin_norm_axis", "begin_params_axis", "epsilon",
                                                 "center", "scale");
    attrs.SetAllAttrs(begin_norm_axis, begin_params_axis, epsilon, false, false);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class SkipLayerNormFunctor {
 public:
  SkipLayerNormFunctor() {
    std::vector<bool> bool_list = {true, false};

    /* number of skip */
    for (bool has_skip : bool_list) {
      /* has_gamma */
      for (bool has_gamma : bool_list) {
        /* has_beta */
        for (bool has_beta : bool_list) {
          /* has_bias */
          for (bool has_bias : bool_list) {
            one::OpBuilder op_builder = one::OpBuilder("skip_layer_norm").Input("x");
            if (has_gamma) { op_builder = op_builder.Input("gamma"); }
            if (has_beta) { op_builder = op_builder.Input("beta"); }
            if (has_bias) { op_builder = op_builder.Input("bias"); }
            if (has_skip) { op_builder = op_builder.Input("skip"); }
            op_builder = op_builder.Output("y").Output("mean").Output("inv_variance");

            std::shared_ptr<OpExpr> op_expr = CHECK_JUST(op_builder.Build());
            ops_.insert(std::pair<std::tuple<bool, bool, bool, bool>, std::shared_ptr<OpExpr>>(
                std::tuple<bool, bool, bool, bool>(has_skip, has_gamma, has_beta, has_bias),
                op_expr));
          }  // has_bias
        }    // has_beta
      }      // has_gamma
    }        // has_skip
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const Optional<one::Tensor>& gamma, const Optional<one::Tensor>& beta,
                           const Optional<one::Tensor>& bias, const Optional<one::Tensor>& skip,
                           const double& epsilon, const double& alpha) const {
    // check shape of x
    const auto& x_shape = *(x->shape());
    CHECK_GE_OR_RETURN(x_shape.NumAxes(), 2)
        << "number of axes of \'x\' should be greater than or equal to 2, yet get "
        << x_shape.NumAxes();

    if (gamma) {
      const auto& gamma_shape = *(JUST(gamma)->shape());
      CHECK_EQ_OR_RETURN(gamma_shape.NumAxes(), 1)
          << "number of axes of \'gamma\' should have be equal to 1, yet get "
          << gamma_shape.NumAxes();
      CHECK_EQ_OR_RETURN(gamma_shape.At(0), x_shape.At(x_shape.NumAxes() - 1))
          << "the size of \'gamma\'(" << gamma_shape.At(0)
          << ") is not consistant with the last dimension of \'x\'("
          << x_shape.At(x_shape.NumAxes() - 1) << ")";
    }
    if (beta) {
      const auto& beta_shape = *(JUST(beta)->shape());
      CHECK_EQ_OR_RETURN(beta_shape.NumAxes(), 1)
          << "number of axes of \'beta\' should have be equal to 1, yet get "
          << beta_shape.NumAxes();
      CHECK_EQ_OR_RETURN(beta_shape.At(0), x_shape.At(x_shape.NumAxes() - 1))
          << "dimension 1 of \'beta\'(" << beta_shape.At(0)
          << ") is not consistant with the last dimension of \'x\'("
          << x_shape.At(x_shape.NumAxes() - 1) << ")";
    }
    if (bias) {
      const auto& bias_shape = *(JUST(bias)->shape());
      CHECK_EQ_OR_RETURN(bias_shape.NumAxes(), 1)
          << "number of axes of \'bias\' should have be equal to 1, yet get "
          << bias_shape.NumAxes();
      CHECK_EQ_OR_RETURN(bias_shape.At(0), x_shape.At(x_shape.NumAxes() - 1))
          << "dimension 1 of \'bias\'(" << bias_shape.At(0)
          << ") is not consistant with the last dimension of \'x\'("
          << x_shape.At(x_shape.NumAxes() - 1) << ")";
    }
    if (skip) {
      const auto& skip_shape = *(JUST(skip)->shape());
      CHECK_EQ_OR_RETURN(skip_shape, x_shape) << "shape of \'skip\' is not the same as \'x\'";
    }

    // set attributes
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("epsilon", "alpha");
    attrs.SetAllAttrs(epsilon, alpha);

    // count number of all input tensors
    size_t nb_inputs = 1;       // count x
    if (skip) nb_inputs += 1;   // count skip
    if (gamma) nb_inputs += 1;  // count gamma
    if (beta) nb_inputs += 1;   // count beta
    if (bias) nb_inputs += 1;   // count bias

    // construct input tensor tuple
    size_t tensor_index = 1;
    TensorTuple input(nb_inputs);
    bool has_gamma = false, has_beta = false, has_bias = false, has_skip = false;
    input[0] = x;
    if (gamma) {
      input[tensor_index] = JUST(gamma);
      tensor_index += 1;
      has_gamma = true;
    }
    if (beta) {
      input[tensor_index] = JUST(beta);
      tensor_index += 1;
      has_beta = true;
    }
    if (bias) {
      input[tensor_index] = JUST(bias);
      tensor_index += 1;
      has_bias = true;
    }
    if (skip) {
      input[tensor_index] = JUST(skip);
      tensor_index += 1;
      has_skip = true;
    }

    return OpInterpUtil::Dispatch<Tensor>(
        *(ops_.find(std::tuple<bool, bool, bool, bool>(has_skip, has_gamma, has_beta, has_bias))
              ->second),
        input, attrs);
  }

 private:
  /* (nb_skip, has_gamma, has_beta, has_bias) -> op */
  std::map<std::tuple<bool, bool, bool, bool>, std::shared_ptr<OpExpr>> ops_;
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
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("begin_norm_axis", "begin_params_axis", "epsilon",
                                                 "center", "scale");
    attrs.SetAllAttrs(begin_norm_axis, begin_params_axis, epsilon, true, true);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x, gamma, beta}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class GroupNormFunctor {
 public:
  GroupNormFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("group_norm")
                         .Input("x")
                         .Output("y")
                         .Output("mean")
                         .Output("inv_variance")
                         .Attr("affine", false)
                         .Build());
    affine_op_ = CHECK_JUST(one::OpBuilder("group_norm")
                                .Input("x")
                                .Input("gamma")
                                .Input("beta")
                                .Output("y")
                                .Output("mean")
                                .Output("inv_variance")
                                .Attr("affine", true)
                                .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const Optional<one::Tensor>& gamma, const Optional<one::Tensor>& beta,
                           const bool affine, const int32_t num_groups, const double& epsilon,
                           const std::string& data_format, const std::string& activation) const {
    auto& attrs =
        THREAD_CACHED_MUTABLE_ATTR_MAP("num_groups", "epsilon", "data_format", "activation");
    attrs.SetAllAttrs(num_groups, epsilon, data_format, activation);
    if (affine) {
      return OpInterpUtil::Dispatch<Tensor>(*affine_op_, {x, JUST(gamma), JUST(beta)}, attrs);
    } else {
      return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
    }
  }

 private:
  std::shared_ptr<OpExpr> op_;
  std::shared_ptr<OpExpr> affine_op_;
};

bool CheckNormShape(const Shape& x_shape, const Shape& normalized_shape) {
  if (x_shape.size() < normalized_shape.size()) { return false; }
  size_t b_ndim = x_shape.size() - normalized_shape.size();
  for (int i = 0; i < x_shape.size(); ++i) {
    if (i >= b_ndim) {
      if (x_shape[i] != normalized_shape[i - b_ndim]) { return false; }
    }
  }
  return true;
}

class RMSNormFunctor {
 public:
  RMSNormFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("rms_norm").Input("x").Output("y").Output("inv_rms").Build());
    op_affine_ = CHECK_JUST(one::OpBuilder("rms_norm")
                                .Input("x")
                                .Input("weight")
                                .Output("y")
                                .Output("inv_rms")
                                .Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x, const Optional<one::Tensor>& weight,
                           const Shape& normalized_shape, const float epsilon) const {
    const Shape& x_shape = *x->shape();
    if (weight) {
      const Shape& w_shape = *JUST(weight)->shape();
      CHECK_EQ_OR_RETURN(w_shape, normalized_shape)
          << "Expected weight be the same shape with normalized_shape "
          << normalized_shape.ToString() << ", but got " << w_shape.ToString();
    }
    if (!CheckNormShape(x_shape, normalized_shape)) {
      auto shape_str_without_parentheses =
          x_shape.ToString().substr(1, x_shape.ToString().size() - 2);
      return Error::RuntimeError()
             << "Given normalized_shape=" << normalized_shape.ToString()
             << ", expected input with shape (*, " << shape_str_without_parentheses
             << "), but got input of " << x_shape.ToString();
    }

    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("normalized_shape", "epsilon");
    attrs.SetAllAttrs(normalized_shape, epsilon);
    if (weight) {
      const DataType dtype = x->dtype()->data_type();
      if (JUST(weight)->dtype()->data_type() != dtype) {
        auto weight_cast = JUST(functional::Cast(JUST(weight), DType{dtype}, /*pin_memory=*/false));
        return OpInterpUtil::Dispatch<Tensor>(*op_affine_, {x, weight_cast}, attrs);
      }
      return OpInterpUtil::Dispatch<Tensor>(*op_affine_, {x, JUST(weight)}, attrs);
    }
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
  std::shared_ptr<OpExpr> op_affine_;
};

class SkipRMSNormFunctor {
 public:
  SkipRMSNormFunctor() {
    std::vector<bool> bool_list = {true, false};

    for (bool has_weight : bool_list) {
      for (bool has_skip : bool_list) {
        for (bool has_bias : bool_list) {
          one::OpBuilder op_builder = one::OpBuilder("skip_rms_norm").Input("x");
          if (has_weight) { op_builder = op_builder.Input("weight"); }
          if (has_bias) { op_builder = op_builder.Input("bias"); }
          if (has_skip) { op_builder = op_builder.Input("skip"); }
          op_builder = op_builder.Output("y").Output("inv_rms");

          std::shared_ptr<OpExpr> op_expr = CHECK_JUST(op_builder.Build());
          ops_.insert(std::pair<std::tuple<bool, bool, bool>, std::shared_ptr<OpExpr>>(
              std::tuple<bool, bool, bool>(has_weight, has_skip, has_bias), op_expr));
        }  // has_bias
      }    // has_skip
    }      // has_weight
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const Optional<one::Tensor>& weight, const Optional<one::Tensor>& bias,
                           const Optional<one::Tensor>& skip, const double& epsilon,
                           const double& alpha) const {
    // check shape of x
    const auto& x_shape = *(x->shape());
    CHECK_GE_OR_RETURN(x_shape.NumAxes(), 2)
        << "number of axes of \'x\' should be greater than or equal to 2, yet get "
        << x_shape.NumAxes();

    if (weight) {
      const auto& weight_shape = *(JUST(weight)->shape());
      CHECK_EQ_OR_RETURN(weight_shape.NumAxes(), 1)
          << "number of axes of \'weight\' should have be equal to 1, yet get "
          << weight_shape.NumAxes();
      CHECK_EQ_OR_RETURN(weight_shape.At(0), x_shape.At(x_shape.NumAxes() - 1))
          << "dimension 1 of \'weight\'(" << weight_shape.At(0)
          << ") is not consistant with the last dimension of \'x\'("
          << x_shape.At(x_shape.NumAxes() - 1) << ")";
    }

    if (bias) {
      const auto& bias_shape = *(JUST(bias)->shape());
      CHECK_EQ_OR_RETURN(bias_shape.NumAxes(), 1)
          << "number of axes of \'bias\' should have be equal to 1, yet get "
          << bias_shape.NumAxes();
      CHECK_EQ_OR_RETURN(bias_shape.At(0), x_shape.At(x_shape.NumAxes() - 1))
          << "dimension 1 of \'bias\'(" << bias_shape.At(0)
          << ") is not consistant with the last dimension of \'x\'("
          << x_shape.At(x_shape.NumAxes() - 1) << ")";
    }

    if (skip) {
      const auto& skip_shape = *(JUST(skip)->shape());
      CHECK_EQ_OR_RETURN(skip_shape, x_shape) << "shape of \'skip\' is not the same as \'x\'";
    }

    // set attributes
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("epsilon", "alpha");
    attrs.SetAllAttrs(epsilon, alpha);

    // count number of all input tensors
    size_t nb_inputs = 1;        // count x
    if (skip) nb_inputs += 1;    // count skip
    if (weight) nb_inputs += 1;  // count weight
    if (bias) nb_inputs += 1;    // count bias

    // construct input tensor tuple
    size_t tensor_index = 1;
    TensorTuple input(nb_inputs);
    bool has_weight = false, has_bias = false, has_skip = false;
    input[0] = x;
    if (weight) {
      input[tensor_index] = JUST(weight);
      tensor_index += 1;
      has_weight = true;
    }
    if (bias) {
      input[tensor_index] = JUST(bias);
      tensor_index += 1;
      has_bias = true;
    }
    if (skip) {
      input[tensor_index] = JUST(skip);
      tensor_index += 1;
      has_skip = true;
    }

    return OpInterpUtil::Dispatch<Tensor>(
        *(ops_.find(std::tuple<bool, bool, bool>(has_weight, has_skip, has_bias))->second), input,
        attrs);
  }

 private:
  /* (has_weight, has_skip, has_bias) -> op */
  std::map<std::tuple<bool, bool, bool>, std::shared_ptr<OpExpr>> ops_;
};

class PixelShuffleFunctor {
 public:
  PixelShuffleFunctor() {}
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const int64_t& h_upscale_factor,
                           const int64_t& w_upscale_factor) const {
    CHECK_OR_RETURN(x->ndim() == 4) << Error::RuntimeError() << "Only Accept 4D Tensor";
    const int64_t batch = x->shape()->At(0);
    const int64_t channel = x->shape()->At(1);
    const int64_t height = x->shape()->At(2);
    const int64_t width = x->shape()->At(3);
    std::shared_ptr<one::Tensor> out;
    CHECK_OR_RETURN(channel % (h_upscale_factor * w_upscale_factor) == 0)
        << Error::RuntimeError()
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
    auto& attrs =
        THREAD_CACHED_MUTABLE_ATTR_MAP("pool_size", "strides", "padding", "padding_before",
                                       "padding_after", "data_format", "ceil_mode");
    attrs.SetAllAttrs(kernel_size, strides, padding, padding_before, padding_after, data_format,
                      ceil_mode);
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
        std::vector<int32_t> padding_before{padding.at(0), padding.at(1)};
        std::vector<int32_t> padding_after{padding.at(0), padding.at(1)};

        auto& attrs =
            THREAD_CACHED_MUTABLE_ATTR_MAP("pool_size", "strides", "padding", "padding_before",
                                           "padding_after", "data_format", "ceil_mode");
        attrs.SetAllAttrs(kernel_size, stride ? *JUST(stride) : kernel_size,
                          std::string("customized"), padding_before, padding_after, data_format,
                          ceil_mode);
        TensorTuple output;
        output.emplace_back(JUST(OpInterpUtil::Dispatch<Tensor>(*tf_maxpool_op_, {x}, attrs)));
        return output;
      }
    }

    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("kernel_size", "padding", "stride", "dilation",
                                                 "data_format", "return_indices", "ceil_mode");
    // If stride is None, we set it as kernel_size to align Pytorch.
    attrs.SetAllAttrs(kernel_size, padding, stride ? *JUST(stride) : kernel_size, dilation,
                      data_format, return_indices, ceil_mode);
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

template<int N>
class MaxUnpoolNDFunctor {
 public:
  MaxUnpoolNDFunctor()
      : op_(CHECK_JUST(one::OpBuilder(fmt::format("max_unpool_{}d", N))
                           .Input("x")
                           .Input("indices")
                           .Output("y")
                           .Build())){};
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& indices,
                           const std::vector<int32_t>& kernel_size,
                           const Optional<std::vector<int32_t>>& stride,
                           const std::vector<int32_t>& padding,
                           const Optional<Shape>& output_size) const {
    const auto fmt_error_msg = [](const std::string& name, int32_t num, bool check_element) {
      if (check_element) {
        return fmt::format("each element in `{}` must be greater than 0, got {}", name, num);
      }
      return fmt::format("`{}` must be an integer or a list of {} integers", name, N);
    };

    CHECK_EQ_OR_RETURN(kernel_size.size(), N) << fmt_error_msg("kernel_size", N, false);
    for (int32_t pool_dim : kernel_size) {
      CHECK_GT_OR_RETURN(pool_dim, 0) << fmt_error_msg("kernel_size", pool_dim, true);
    }

    if (stride) {
      CHECK_EQ_OR_RETURN(JUST(stride)->size(), N) << fmt_error_msg("stride", N, false);
      for (int32_t stride_dim : *JUST(stride)) {
        CHECK_GT_OR_RETURN(stride_dim, 0) << fmt_error_msg("stride", stride_dim, true);
      }
    }
    for (int32_t i = 0; i < padding.size(); i++) {
      CHECK_GE_OR_RETURN(kernel_size[i], 2 * padding[i])
          << "pad should be smaller than half of kernel size";
    }

    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("kernel_size", "padding", "stride",
                                                 "has_output_size", "output_size");
    attrs.SetAllAttrs(kernel_size, padding, stride ? *JUST(stride) : kernel_size,
                      output_size.has_value(),
                      output_size.has_value() ? *JUST(output_size) : Shape());
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x, indices}, attrs);
  }

 protected:
  std::shared_ptr<OpExpr> op_;
};

class AdaptivePoolNDFunctor {
 public:
  AdaptivePoolNDFunctor() = default;
  virtual ~AdaptivePoolNDFunctor() = default;
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::vector<int64_t>& output_size) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("output_size");
    attrs.SetAllAttrs(output_size);
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

class AdaptiveMaxPoolBaseFunctor {
 public:
  AdaptiveMaxPoolBaseFunctor() = default;
  virtual ~AdaptiveMaxPoolBaseFunctor() = default;
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& x,
                                const std::vector<int64_t>& output_size) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("output_size");
    attrs.SetAllAttrs(output_size);
    return OpInterpUtil::Dispatch<TensorTuple>(*op_, {x}, attrs);
  }

 protected:
  std::shared_ptr<OpExpr> op_;
};

class AdaptiveMaxPool1DFunctor : public AdaptiveMaxPoolBaseFunctor {
 public:
  AdaptiveMaxPool1DFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("adaptive_max_pool1d").Input("x").Output("y").Output("index").Build());
  }
};

class AdaptiveMaxPool2DFunctor : public AdaptiveMaxPoolBaseFunctor {
 public:
  AdaptiveMaxPool2DFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("adaptive_max_pool2d").Input("x").Output("y").Output("index").Build());
  }
};

class AdaptiveMaxPool3DFunctor : public AdaptiveMaxPoolBaseFunctor {
 public:
  AdaptiveMaxPool3DFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("adaptive_max_pool3d").Input("x").Output("y").Output("index").Build());
  }
};
class LossFunctorBase {
 public:
  Maybe<Tensor> apply_reduction(const Maybe<Tensor>& x, const std::string& reduction) const {
    CHECK_OR_RETURN(reduction == "none" || reduction == "sum" || reduction == "mean")
        << Error::RuntimeError() << "Reduction should be none, sum or mean.";
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
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("beta");
    attrs.SetAllAttrs(beta);
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
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("log_target");
    attrs.SetAllAttrs(log_target);
    if (reduction == "batchmean" && input->ndim() != 0) {
      const auto& result = JUST(
          apply_reduction(OpInterpUtil::Dispatch<Tensor>(*op_, {input, target}, attrs), "sum"));
      return ScalarDiv(result, input->shape()->At(0));
    } else {
      return apply_reduction(OpInterpUtil::Dispatch<Tensor>(*op_, {input, target}, attrs),
                             reduction);
    }
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
    auto out = weight ? OpInterpUtil::Dispatch<Tensor>(*op_weight_, {input, target, JUST(weight)})
                      : OpInterpUtil::Dispatch<Tensor>(*op_, {input, target});
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
    op_reduce_mean_ = CHECK_JUST(one::OpBuilder("binary_cross_entropy_with_logits_reduce_mean")
                                     .Input("input")
                                     .Input("target")
                                     .Output("out")
                                     .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& target,
                           const Optional<one::Tensor>& weight,
                           const Optional<one::Tensor>& pos_weight,
                           const std::string& reduction) const {
    if (pos_weight) {
      const auto pos_weight_shape = JUST(pos_weight)->shape();
      // pos weight shape = (), (1,), (1,1)... or (input/target.shape[-1],)
      const bool is_pos_weight_shape_valid =
          (pos_weight_shape->elem_cnt() == 1)
          || (pos_weight_shape->NumAxes() == 1
              && pos_weight_shape->At(0) == target->shape()->back());

      CHECK_OR_RETURN(is_pos_weight_shape_valid)
          << Error::RuntimeError()
          << "pos_weight must be a vector with length equal to the number of classes.";
    }

    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("has_pos_weight");
    attrs.SetAllAttrs(pos_weight.has_value());
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
        if (reduction == "mean") {
          return OpInterpUtil::Dispatch<Tensor>(*op_reduce_mean_, {input, target});
        }
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
  std::shared_ptr<OpExpr> op_reduce_mean_;
};

class NLLLossFunctor {
 public:
  NLLLossFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("nll")
                         .Input("input")
                         .Input("target")
                         .Output("output")
                         .Output("out_weight")
                         .Build());

    op_weight_ = CHECK_JUST(one::OpBuilder("nll")
                                .Input("input")
                                .Input("target")
                                .Input("weight")
                                .Output("output")
                                .Output("out_weight")
                                .Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& target,
                           const Optional<one::Tensor>& weight, const int64_t& ignore_index,
                           const std::string& reduction) const {
    CHECK_OR_RETURN(reduction == "none" || reduction == "sum" || reduction == "mean")
        << Error::RuntimeError() << "Reduction should be none, sum or mean.";

    const auto& input_shape = input->shape();
    const int64_t K = input_shape->NumAxes();
    CHECK_GE_OR_RETURN(K, 2) << Error::RuntimeError() << "Expected 2 or more dimensions";
    const int64_t N = input_shape->At(0);
    const int64_t C = input_shape->At(1);

    const auto& target_shape = target->shape();
    CHECK_EQ_OR_RETURN(target_shape->NumAxes(), K - 1)
        << Error::RuntimeError() << "Expected target dimensions (" << K - 1
        << ") to match input dimensions (" << K << "), got " << target_shape->NumAxes();
    CHECK_EQ_OR_RETURN(target_shape->At(0), N)
        << Error::RuntimeError() << "Expected input batch_size (" << N
        << ") to match target batch_size (" << target_shape->At(0) << ")";

    std::shared_ptr<one::Tensor> input_;
    std::shared_ptr<one::Tensor> target_;
    if (K > 2) {
      DimVector idea_target_dim_vec;
      idea_target_dim_vec.push_back(N);
      for (int64_t i = 2; i < K; ++i) { idea_target_dim_vec.push_back(input_shape->At(i)); }
      Shape idea_target_shape(idea_target_dim_vec);
      CHECK_EQ_OR_RETURN(*target_shape, idea_target_shape)
          << Error::RuntimeError() << "Expected target shape " << idea_target_shape.ToString()
          << ", got " << target_shape->ToString();

      std::vector<int> perm(input_shape->dim_vec().size(), 0);
      perm[perm.size() - 1] = 1;
      for (size_t i = 1; i < perm.size() - 1; ++i) { perm[i] = i + 1; }

      input_ = JUST(sequence_function(functional::Transpose)
                        .then(std::bind(functional::Reshape, std::placeholders::_1, Shape({-1, C})))
                        .call(input, perm));
      target_ = JUST(functional::Flatten(target, 0, K - 2));
    } else {
      input_ = input;
      target_ = target;
    }

    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("ignore_index");
    attrs.SetAllAttrs(ignore_index);

    std::shared_ptr<TensorTuple> nll_result;
    if (weight) {
      nll_result = JUST(
          OpInterpUtil::Dispatch<TensorTuple>(*op_weight_, {input_, target_, JUST(weight)}, attrs));
    } else {
      nll_result = JUST(OpInterpUtil::Dispatch<TensorTuple>(*op_, {input_, target_}, attrs));
    }
    auto output = JUST(VectorAt(*nll_result, 0));

    if (K > 2) { output = JUST(functional::Reshape(output, *target_shape)); }

    if (reduction == "none") { return output; }

    auto sum = JUST(functional::ReduceSum(output, {}, false));

    if (reduction == "sum") { return sum; }

    auto total_weight = JUST(functional::ReduceSum(JUST(VectorAt(*nll_result, 1)), {}, false));
    return functional::Div(sum, total_weight);
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
                             .Output("output")
                             .Output("out_weight")
                             .Build());

    op_nll_weight_ = CHECK_JUST(one::OpBuilder("nll")
                                    .Input("input")
                                    .Input("target")
                                    .Input("weight")
                                    .Output("output")
                                    .Output("out_weight")
                                    .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& target,
                           const Optional<one::Tensor>& weight, const int64_t& ignore_index,
                           const std::string& reduction, const double& label_smoothing) const {
    if (input->shape() == target->shape()) {
      CHECK_OR_RETURN(target->dtype()->is_floating_point())
          << "Expected floating point type for target with class probabilities, got "
          << target->dtype()->name();
      CHECK_LT_OR_RETURN(ignore_index, 0)
          << "ignore_index is not supported for floating point targe";
      return CrossEntropyProb(input, target, weight, reduction, label_smoothing);
    }
    if (label_smoothing > 0.0)
      return CrossEntropyLabelSmoothing(input, target, weight, ignore_index, reduction,
                                        label_smoothing);
    CHECK_OR_RETURN(reduction == "none" || reduction == "sum" || reduction == "mean")
        << Error::RuntimeError() << "Reduction should be none, sum or mean.";
    const auto& input_shape = input->shape();
    const auto& target_shape = target->shape();

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

    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("ignore_index");
    attrs.SetAllAttrs(ignore_index);

    std::shared_ptr<TensorTuple> nll_result;
    if (weight) {
      nll_result = JUST(OpInterpUtil::Dispatch<TensorTuple>(
          *op_nll_weight_, {input_, target_, JUST(weight)}, attrs));
    } else {
      nll_result = JUST(OpInterpUtil::Dispatch<TensorTuple>(*op_nll_, {input_, target_}, attrs));
    }

    auto output = JUST(VectorAt(*nll_result, 0));
    output = JUST(functional::Reshape(output, *target_shape));
    if (reduction == "none") { return output; }

    auto sum = JUST(functional::ReduceSum(output, {}, false));
    if (reduction == "sum") { return sum; }

    auto total_weight = JUST(functional::ReduceSum(JUST(VectorAt(*nll_result, 1)), {}, false));
    return functional::Div(sum, total_weight);
  }

 private:
  std::shared_ptr<OpExpr> op_log_softmax_;
  std::shared_ptr<OpExpr> op_nll_;
  std::shared_ptr<OpExpr> op_nll_weight_;
};

class CrossEntropyLabelSmoothingFunctor {
 public:
  CrossEntropyLabelSmoothingFunctor() {
    op_log_softmax_ = CHECK_JUST(one::OpBuilder("log_softmax").Input("in").Output("prob").Build());

    op_nll_ = CHECK_JUST(one::OpBuilder("nll")
                             .Input("input")
                             .Input("target")
                             .Output("output")
                             .Output("out_weight")
                             .Build());

    op_nll_weight_ = CHECK_JUST(one::OpBuilder("nll")
                                    .Input("input")
                                    .Input("target")
                                    .Input("weight")
                                    .Output("output")
                                    .Output("out_weight")
                                    .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& target,
                           const Optional<one::Tensor>& weight, const int64_t& ignore_index,
                           const std::string& reduction, const double& label_smoothing) const {
    CHECK_OR_RETURN(reduction == "none" || reduction == "sum" || reduction == "mean")
        << Error::RuntimeError() << "Reduction should be none, sum or mean.";
    const auto& input_shape = input->shape();
    const auto& target_shape = target->shape();

    std::vector<int> input_perm(input_shape->dim_vec().size(), 0);
    input_perm[input_perm.size() - 1] = 1;
    for (size_t i = 1; i < input_perm.size() - 1; ++i) { input_perm[i] = i + 1; }
    CHECK_OR_RETURN(label_smoothing > 0.0 && label_smoothing <= 1.0)
        << "label_smoothing must be between 0.0 and 1.0. Got: " << label_smoothing;

    const auto& input_ = JUST(sequence_function(functional::Transpose)
                                  .then(std::bind(functional::Reshape, std::placeholders::_1,
                                                  Shape({-1, input_shape->At(1)})))
                                  .then([this](const std::shared_ptr<one::Tensor>& x) {
                                    return OpInterpUtil::Dispatch<Tensor>(*op_log_softmax_, {x});
                                  })
                                  .call(input, input_perm));
    const auto& target_ = JUST(functional::Flatten(target, 0, target->shape()->NumAxes() - 1));

    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("ignore_index");
    attrs.SetAllAttrs(ignore_index);

    std::shared_ptr<TensorTuple> nll_result;
    if (weight) {
      nll_result = JUST(OpInterpUtil::Dispatch<TensorTuple>(
          *op_nll_weight_, {input_, target_, JUST(weight)}, attrs));
    } else {
      nll_result = JUST(OpInterpUtil::Dispatch<TensorTuple>(*op_nll_, {input_, target_}, attrs));
    }

    const auto& ignore_mask = JUST(Reshape(JUST(ScalarLogicalEqual(target_, ignore_index)), {-1}));

    // smooth_loss = (-(input_ * weight.reshape(1, -1)).sum(1) * ~ignore_mask).reshape_as(target)
    std::shared_ptr<Tensor> smooth_loss = input_;
    if (weight) {
      const auto& weight_2d = JUST(Reshape(JUST(weight), {1, -1}));
      smooth_loss = JUST(Mul(smooth_loss, weight_2d));
    }
    smooth_loss = JUST(Negative(JUST(ReduceSum(smooth_loss, {1}, false))));
    smooth_loss = JUST(MaskedFill(smooth_loss, ignore_mask, 0.0));
    smooth_loss = JUST(Reshape(smooth_loss, *target_shape));

    int64_t n_classes = input->shape()->At(1);
    auto nll_loss = JUST(VectorAt(*nll_result, 0));
    nll_loss = JUST(functional::Reshape(nll_loss, *target_shape));

    // loss = nll_loss * (1 - label_smoothing) + smooth_loss * label_smoothing / num_classes
    if (reduction == "none") {
      return JUST(Add(JUST(ScalarMul(nll_loss, 1 - label_smoothing, false)),
                      JUST(ScalarMul(smooth_loss, label_smoothing / n_classes, false)), 1, false));
    }

    const auto& nll_loss_sum = JUST(ReduceSum(nll_loss, {}, false));
    const auto& smooth_loss_sum = JUST(ReduceSum(smooth_loss, {}, false));
    const auto& cross_entropy_loss_sum =
        JUST(Add(JUST(ScalarMul(nll_loss_sum, 1 - label_smoothing, false)),
                 JUST(ScalarMul(smooth_loss_sum, label_smoothing / n_classes, false)), 1, false));
    if (reduction == "sum") { return cross_entropy_loss_sum; }

    const auto& total_weight = JUST(ReduceSum(JUST(VectorAt(*nll_result, 1)), {}, false));
    return Div(cross_entropy_loss_sum, total_weight);
  }

 private:
  std::shared_ptr<OpExpr> op_log_softmax_;
  std::shared_ptr<OpExpr> op_nll_;
  std::shared_ptr<OpExpr> op_nll_weight_;
};

class CrossEntropyProbFunctor : public LossFunctorBase {
 public:
  CrossEntropyProbFunctor() {
    op_log_softmax_ = CHECK_JUST(one::OpBuilder("log_softmax").Input("in").Output("prob").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& target,
                           const Optional<one::Tensor>& weight, const std::string& reduction,
                           const double& label_smoothing) const {
    const auto& input_shape = input->shape();
    const auto& target_shape = target->shape();

    std::vector<int> input_perm(input_shape->NumAxes(), 0);
    input_perm[input_perm.size() - 1] = 1;
    for (size_t i = 1; i < input_perm.size() - 1; ++i) { input_perm[i] = i + 1; }

    const auto input_ = JUST(sequence_function(functional::Transpose)
                                 .then(std::bind(functional::Reshape, std::placeholders::_1,
                                                 Shape({-1, input_shape->At(1)})))
                                 .then([this](const std::shared_ptr<one::Tensor>& x) {
                                   return OpInterpUtil::Dispatch<Tensor>(*op_log_softmax_, {x});
                                 })
                                 .call(input, input_perm));
    std::shared_ptr<Tensor> target_ =
        JUST(sequence_function(functional::Transpose)
                 .then(std::bind(functional::Reshape, std::placeholders::_1,
                                 Shape({-1, target_shape->At(1)})))
                 .call(target, input_perm));
    if (label_smoothing > 0) {
      int32_t num_classes = input_->shape()->At(1);
      target_ =
          JUST(ScalarAdd(JUST(ScalarMul(target_, static_cast<double>(1) - label_smoothing, false)),
                         label_smoothing / static_cast<double>(num_classes), 1, false));
    }

    auto nll_result = JUST(Negative(JUST(Mul(input_, target_))));
    if (weight) {
      const auto& weight_expand = JUST(Unsqueeze(JUST(weight), 0));
      nll_result = JUST(Mul(nll_result, weight_expand));
    }
    DimVector target_reshape_(input->ndim() - 1);
    for (size_t i = 0; i < target_reshape_.size(); ++i) {
      target_reshape_[i] = input_shape->At(input_perm[i]);
    }
    nll_result = JUST(ReduceSum(nll_result, {-1}, false));
    nll_result = JUST(Reshape(nll_result, Shape(target_reshape_)));
    return apply_reduction(nll_result, reduction);
  }

 private:
  std::shared_ptr<OpExpr> op_log_softmax_;
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
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("depth");
    attrs.SetAllAttrs(depth);

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
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("depth");
    attrs.SetAllAttrs(depth);

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
    if (!(logits->is_global() && label->is_global())) { return false; }

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
    int64_t depth = logits->shape()->At(logits->shape()->NumAxes() - 1);
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("depth");
    attrs.SetAllAttrs(depth);
    const auto& result = JUST(OpInterpUtil::Dispatch<TensorTuple>(*op_sparse_softmax_cross_entropy_,
                                                                  {logits, label}, attrs));
    return result->at(1);
  }

  Maybe<Tensor> LazySparseSoftmaxCrossEntropyMsOperator(
      const std::shared_ptr<one::Tensor>& logits, const std::shared_ptr<one::Tensor>& label) const {
    int64_t depth = logits->shape()->At(logits->shape()->NumAxes() - 1);
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("depth");
    attrs.SetAllAttrs(depth);
    const auto& result = JUST(OpInterpUtil::Dispatch<TensorTuple>(
        *op_sparse_softmax_cross_entropy_ms_, {logits, label}, attrs));
    return result->at(1);
  }

  Maybe<Tensor> EagerSparseSoftmaxCrossEntropyMsOperator(
      const std::shared_ptr<one::Tensor>& logits, const std::shared_ptr<one::Tensor>& label) const {
    // op_reduce_max_device_stage_
    int64_t depth = logits->shape()->At(logits->shape()->NumAxes() - 1);
    int32_t axis = logits->shape()->NumAxes() - 1;

    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("axis");
    attrs.SetAllAttrs(std::vector<int32_t>{axis});
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
            CHECK_EQ_OR_RETURN(split_axis, 0)
                << Error::RuntimeError() << "Split axis must equal to 0. ";
            new_sbp_parallels.emplace_back(sbp_parallel);
          }
        } else {
          new_sbp_parallels.emplace_back(sbp_parallel);
        }
      }

      s0s1_sbp_parallels.emplace_back(logits_nd_sbp.sbp_parallel(0));
      s0s1_sbp_parallels.emplace_back(logits_nd_sbp.sbp_parallel(1));
      max_global_stage_input0 = JUST(functional::ToGlobal(
          (*max_device_stage)[0], JUST((*max_device_stage)[0]->parallel_desc()), new_sbp_parallels,
          s0s1_sbp_parallels, /* check_meta */ false, /*copy=*/false));
      max_global_stage_input1 = JUST(functional::ToGlobal(
          (*max_device_stage)[2], JUST((*max_device_stage)[0]->parallel_desc()), new_sbp_parallels,
          s0s1_sbp_parallels, /* check_meta */ false, /*copy=*/false));
    }
    // op_reduce_max_global_stage_
    auto& reduce_max_global_attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("axis", "keepdims");
    reduce_max_global_attrs.SetAllAttrs(std::vector<int32_t>{axis}, true);
    const auto& max_global_stage = JUST(OpInterpUtil::Dispatch<TensorTuple>(
        *op_reduce_max_global_stage_, {max_global_stage_input0, max_global_stage_input1},
        reduce_max_global_attrs));
    auto& broadcast_sub_input = max_global_stage->at(0);
    if (logits_nd_sbp.sbp_parallel_size() == 2) {
      broadcast_sub_input = JUST(functional::ToGlobal(
          broadcast_sub_input, JUST((*max_device_stage)[0]->parallel_desc()), new_sbp_parallels,
          new_sbp_parallels, /* check_meta */ false, /*copy=*/false));
    }
    // op_broadcast_sub_
    const auto& output_broadcast_sub = JUST(
        OpInterpUtil::Dispatch<TensorTuple>(*op_broadcast_sub_, {logits, broadcast_sub_input}));
    // op_exp_
    const auto& output_exp =
        JUST(OpInterpUtil::Dispatch<TensorTuple>(*op_exp_, {(*output_broadcast_sub)[0]}));
    // op_reduce_sum_
    auto& reduce_sum_attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("axis", "keepdims");
    reduce_sum_attrs.SetAllAttrs(std::vector<int32_t>{axis}, true);
    const auto& output_reduce_sum = JUST(
        OpInterpUtil::Dispatch<TensorTuple>(*op_reduce_sum_, {(*output_exp)[0]}, reduce_sum_attrs));
    std::shared_ptr<Tensor> broadcast_div_input1 = output_reduce_sum->at(0);
    if (logits_nd_sbp.sbp_parallel_size() == 2) {
      std::vector<Symbol<SbpParallel>> empty_grad_sbp_parallels;
      broadcast_div_input1 = JUST(functional::ToGlobal(
          (*output_reduce_sum)[0], JUST((*output_reduce_sum)[0]->parallel_desc()),
          new_sbp_parallels, new_sbp_parallels, /* check_meta */ false, /*copy=*/false));
    }
    // op_broadcast_div_
    const auto& predictions = JUST(OpInterpUtil::Dispatch<TensorTuple>(
        *op_broadcast_div_, {(*output_exp)[0], broadcast_div_input1}));
    // op_sparse_cross_entropy_ms_
    auto& sparse_cross_entropy_ms_attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("depth");
    sparse_cross_entropy_ms_attrs.SetAllAttrs(depth);
    const auto& output = JUST(OpInterpUtil::Dispatch<Tensor>(
        *op_sparse_cross_entropy_ms_, {(*predictions)[0], label}, sparse_cross_entropy_ms_attrs));
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
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("m1", "m2", "m3", "depth");
    attrs.SetAllAttrs(m1, m2, m3, x->shape()->At(1));
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
                           const int64_t& max_target_length, const int64_t& blank,
                           const bool& zero_infinity, const std::string& reduction) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("max_target_length", "blank", "zero_infinity");
    attrs.SetAllAttrs(max_target_length, blank, zero_infinity);
    std::shared_ptr<one::Tensor> out;
    DeviceType log_probs_device_type;  // NOLINT
    if (log_probs->is_local()) {
      log_probs_device_type = JUST(log_probs->device())->enum_type();
    } else {
      log_probs_device_type = JUST(log_probs->parallel_desc())->device_type();
    }
    const std::string& log_probs_device_str = *JUST(DeviceTag4DeviceType(log_probs_device_type));
    std::shared_ptr<one::Tensor> target_lengths_on_log_probs_device =
        JUST(functional::To(target_lengths, log_probs_device_str));
    if (targets->dtype()->data_type() == DataType::kInt32) {
      out = JUST(OpInterpUtil::Dispatch<Tensor>(
          *op_,
          {
              log_probs,
              JUST(functional::To(targets, log_probs_device_str)),
              JUST(functional::To(input_lengths, log_probs_device_str)),
              target_lengths_on_log_probs_device,
          },
          attrs));
    } else {
      out = JUST(OpInterpUtil::Dispatch<Tensor>(
          *op_,
          {
              log_probs,
              JUST(functional::To(targets, Optional<std::string>(log_probs_device_str),
                                  DType::Int64(), false)),
              JUST(functional::To(input_lengths, log_probs_device_str)),
              target_lengths_on_log_probs_device,
          },
          attrs));
    }
    if (zero_infinity) {
      if (out->is_local()) {
        const auto create_constant = [&](const Scalar& scalar) -> Maybe<Tensor> {
          return functional::Constant(*out->shape(), scalar, out->dtype(), JUST(out->device()));
        };

        out = JUST(sequence_function(functional::Constant)
                       .then(std::bind(functional::BroadcastEqual, out, std::placeholders::_1))
                       .then(std::bind(functional::Where, std::placeholders::_1,
                                       JUST(create_constant(Scalar(0))), out))
                       .call(*out->shape(), Scalar(std::numeric_limits<double>::infinity()),
                             out->dtype(), JUST(out->device())));
      } else {
        const auto& placement = JUST(out->parallel_desc());
        const auto& nd_sbp = *JUST(GetSbpList(JUST(out->nd_sbp())));
        const auto create_constant = [&](const Scalar& scalar) -> Maybe<Tensor> {
          return functional::GlobalConstant(*out->shape(), scalar, out->dtype(), placement, nd_sbp);
        };

        out = JUST(sequence_function(functional::GlobalConstant)
                       .then(std::bind(functional::BroadcastEqual, out, std::placeholders::_1))
                       .then(std::bind(functional::Where, std::placeholders::_1,
                                       JUST(create_constant(Scalar(0))), out))
                       .call(*out->shape(), Scalar(std::numeric_limits<double>::infinity()),
                             out->dtype(), placement, nd_sbp));
      }
    }
    CHECK_OR_RETURN([&]() -> bool {
      if ((reduction != "none") && (reduction != "sum") && (reduction != "mean")) return false;
      return true;
    }()) << Error::RuntimeError()
         << "Reduction should be none, sum or mean.";
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
          .call(target_lengths_on_log_probs_device, Scalar(1), NullOpt);
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
    }()) << Error::RuntimeError()
         << "Reduction should be none, sum or mean.";
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
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("size", "align_corners");
    attrs.SetAllAttrs(size, align_corners);
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
    auto& attrs =
        THREAD_CACHED_MUTABLE_ATTR_MAP("interpolation_mode", "padding_mode", "align_corners");
    attrs.SetAllAttrs(interpolation_mode, padding_mode, align_corners);
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
    cast_op_ = CHECK_JUST(one::OpBuilder("cast").Input("in").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const Optional<one::Tensor>& moving_mean,
                           const Optional<one::Tensor>& moving_variance,
                           const Optional<one::Tensor>& gamma, const Optional<one::Tensor>& beta,
                           const int32_t& axis, const float& epsilon, const float& momentum,
                           const bool& training) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("axis", "epsilon", "momentum");
    // convert torch momentum to tensorflow momentum
    attrs.SetAllAttrs(axis, epsilon, static_cast<float>(1.0 - momentum));

    CHECK_OR_RETURN((moving_mean && moving_variance) || (!moving_mean && !moving_variance))
        << Error::RuntimeError()
        << "Both running_mean and running_variance should be None or Tensor.";

    const DataType dtype = x->dtype()->data_type();

    std::shared_ptr<one::Tensor> gamma_val;
    std::shared_ptr<one::Tensor> beta_val;

    CHECK_GE_OR_RETURN(x->shape()->NumAxes(), 2)
        << Error::RuntimeError() << "NumAxes of x should be greater or equal than 2. ";
    if (gamma.has_value() && beta.has_value()) {
      gamma_val = JUST(gamma);
      beta_val = JUST(beta);
    } else {
      const Shape gamma_beta_shape = Shape({x->shape()->At(1)});
      gamma_val = JUST(functional::Constant(gamma_beta_shape, 1.0, x->dtype(), JUST(x->device())));
      beta_val = JUST(functional::Constant(gamma_beta_shape, 0.0, x->dtype(), JUST(x->device())));
    }

    const DataType gamma_dtype = gamma_val->dtype()->data_type();
    const DataType beta_dtype = beta_val->dtype()->data_type();
    CHECK_EQ_OR_RETURN(gamma_dtype, beta_dtype)
        << Error::RuntimeError() << "gamma and beta have different data types.";
    if (gamma_dtype != dtype) {
      gamma_val = JUST(functional::Cast(gamma_val, DType{dtype}, /*pin_memory=*/false));
      beta_val = JUST(functional::Cast(beta_val, DType{dtype}, /*pin_memory=*/false));
    }

    std::shared_ptr<one::Tensor> moving_mean_val;
    std::shared_ptr<one::Tensor> moving_variance_val;
    bool need_cast_moving_stats = false;
    if (moving_mean) {
      const DataType moving_mean_dtype = JUST(moving_mean)->dtype()->data_type();
      CHECK_EQ_OR_RETURN(JUST(moving_variance)->dtype()->data_type(), moving_mean_dtype)
          << Error::RuntimeError() << "moving_mean and moving_variance have different data types.";
      need_cast_moving_stats = (moving_mean_dtype != dtype);
      if (need_cast_moving_stats) {
        moving_mean_val =
            JUST(functional::Cast(JUST(moving_mean), DType{dtype}, /*pin_memory=*/false));
        moving_variance_val =
            JUST(functional::Cast(JUST(moving_variance), DType{dtype}, /*pin_memory=*/false));
      } else {
        moving_mean_val = JUST(moving_mean);
        moving_variance_val = JUST(moving_variance);
      }
    }

    std::shared_ptr<one::Tensor> res;

    if (!training) {
      CHECK_OR_RETURN(moving_mean && moving_variance)
          << Error::RuntimeError() << "Must have moving_mean and moving_variance in eval mode.";
      res = JUST(OpInterpUtil::Dispatch<one::Tensor>(
          *norm_eval_op_, {x, moving_mean_val, moving_variance_val, gamma_val, beta_val}, attrs));
    } else if (moving_mean) {
      res = JUST(OpInterpUtil::Dispatch<one::Tensor>(
          *norm_training_stats_op_, {x, moving_mean_val, moving_variance_val, gamma_val, beta_val},
          attrs));
    } else {
      res = JUST(OpInterpUtil::Dispatch<one::Tensor>(*norm_training_no_stats_op_,
                                                     {x, gamma_val, beta_val}, attrs));
    }

    if (need_cast_moving_stats) {
      // For inplace update moving_mean and moving_variance
      JUST(CheckInplaceValid(JUST(moving_mean)));
      std::shared_ptr<TensorTuple> outputs = std::make_shared<TensorTuple>(1);
      outputs->at(0) = JUST(moving_mean);
      auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("dtype", "pin_memory");
      attrs.SetAllAttrs(JUST(moving_mean)->dtype()->data_type(), false);
      JUST(OpInterpUtil::Dispatch(*cast_op_, {moving_mean_val}, outputs.get(), attrs));
      JUST(CheckInplaceValid(JUST(moving_variance)));
      outputs->at(0) = JUST(moving_variance);
      JUST(OpInterpUtil::Dispatch(*cast_op_, {moving_variance_val}, outputs.get(), attrs));
    }

    return res;
  }

 private:
  std::shared_ptr<OpExpr> norm_eval_op_;
  std::shared_ptr<OpExpr> norm_training_stats_op_;
  std::shared_ptr<OpExpr> norm_training_no_stats_op_;
  std::shared_ptr<OpExpr> cast_op_;
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
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("axis", "epsilon", "momentum");
    // convert torch momentum to tensorflow momentum
    attrs.SetAllAttrs(axis, epsilon, static_cast<float>(1.0 - momentum));

    CHECK_OR_RETURN((moving_mean && moving_variance) || (!moving_mean && !moving_variance))
        << Error::RuntimeError()
        << "Both moving_mean and moving_variance should be None or Tensor.";
    if (!is_training) {
      CHECK_OR_RETURN(moving_mean && moving_variance)
          << Error::RuntimeError() << "Must have moving_mean and moving_variance in eval mode.";
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

class ConstantPadFunctor {
 public:
  ConstantPadFunctor() {
    constant_pad_ = CHECK_JUST(one::OpBuilder("pad").Input("x").Output("y").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::vector<int64_t>& pad, const Scalar& value) const {
    const int64_t ndim = input->shape()->NumAxes();
    const int64_t pad_size = pad.size();
    CHECK_LE_OR_RETURN(pad_size, 2 * ndim)
        << Error::RuntimeError() << "Pad size should less than or equal to input axes * 2.";
    CHECK_EQ_OR_RETURN(pad_size % 2, 0)
        << Error::RuntimeError() << "Length of pad must be even but instead it equals " << pad_size;

    std::vector<int64_t> pad_before(ndim, 0);
    std::vector<int64_t> pad_after(ndim, 0);
    const int64_t pad_pair = pad_size / 2;
    for (int64_t i = 0; i < pad_pair; ++i) {
      pad_before[ndim - i - 1] = pad[2 * i];
      pad_after[ndim - i - 1] = pad[2 * i + 1];
    }
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("padding", "floating_constant_value",
                                                 "integral_constant_value", "padding_before",
                                                 "padding_after");
    if (IsFloatingDataType(input->dtype()->data_type())) {
      attrs.SetAllAttrs(pad, value.As<double>(), static_cast<int64_t>(0), pad_before, pad_after);
    } else if (IsIntegralDataType(input->dtype()->data_type())) {
      attrs.SetAllAttrs(pad, static_cast<double>(0), value.As<int64_t>(), pad_before, pad_after);
    } else if (input->dtype() == DType::Bool()) {
      int64_t bool_value = value.As<int64_t>();
      CHECK_OR_RETURN(bool_value == 1 || bool_value == 0)
          << "value must be 1/0 or True/False for bool Tensor";
      attrs.SetAllAttrs(pad, static_cast<double>(0), value.As<int64_t>(), pad_before, pad_after);
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "Data type should be floating, bool or integral type.";
    }
    return OpInterpUtil::Dispatch<Tensor>(*constant_pad_, {input}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> constant_pad_;
};

class ReflectionPadFunctor {
 public:
  ReflectionPadFunctor() {
    reflect_pad1d_ = CHECK_JUST(one::OpBuilder("reflection_pad1d").Input("x").Output("y").Build());
    reflect_pad2d_ = CHECK_JUST(one::OpBuilder("reflection_pad2d").Input("x").Output("y").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::vector<int64_t>& pad) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("padding");
    attrs.SetAllAttrs(pad);
    const int64_t pad_size = pad.size();
    const size_t ndim = input->ndim();
    CHECK_LE_OR_RETURN(pad_size, 2 * ndim)
        << Error::RuntimeError() << "Pad size should less than or equal to input axes * 2.";

    if (pad_size == 2) {
      // 2D/3D reflect padding
      CHECK_OR_RETURN((ndim == 2 && input->shape()->At(1) != 0)
                      || (ndim == 3 && input->shape()->At(1) != 0 && input->shape()->At(2) != 0))
          << "2D or 3D (batch mode) tensor expected for input, but got: " << ndim;
      const int64_t pad_left = pad[0];
      const int64_t pad_right = pad[1];
      const int64_t dim_w = (ndim == 3) ? 2 : 1;
      const int64_t input_width = input->shape()->At(dim_w);
      const int64_t output_w = input_width + pad_left + pad_right;
      CHECK_OR_RETURN(pad_left < input_width && pad_right < input_width)
          << "Padding size should be less than the corresponding input dimension, but got: "
             "padding ("
          << pad_left << ", " << pad_right << ") at dimension " << dim_w << " of input "
          << input->shape()->ToString();
      CHECK_OR_RETURN(output_w >= 1)
          << "input (W: " << input_width << ")is too small. Calculated output W: " << output_w;

      if (ndim == 2) {
        // for 2D input
        auto unsqueezed_input = JUST(functional::Unsqueeze(input, 0));
        auto unsqueezed_output =
            JUST(OpInterpUtil::Dispatch<Tensor>(*reflect_pad1d_, {unsqueezed_input}, attrs));
        return JUST(functional::Squeeze(unsqueezed_output, std::vector<int32_t>{0}));
      }
      return OpInterpUtil::Dispatch<Tensor>(*reflect_pad1d_, {input}, attrs);
    } else if (pad_size == 4) {
      // 3D/4D reflect padding
      bool valid_dims = input->shape()->At(1) != 0 && input->shape()->At(2) != 0;
      CHECK_OR_RETURN((ndim == 3 && valid_dims)
                      || (ndim == 4 && valid_dims && input->shape()->At(3) != 0))
          << "3D or 4D (batch mode) tensor expected for input, but got: " << ndim;

      int dim_h = 1;
      int dim_w = 2;
      if (ndim == 4) {
        dim_w++;
        dim_h++;
      }

      const int64_t pad_left = pad[0];
      const int64_t pad_right = pad[1];
      const int64_t pad_top = pad[2];
      const int64_t pad_bottom = pad[3];

      const int64_t input_h = input->shape()->At(dim_h);
      const int64_t input_w = input->shape()->At(dim_w);
      const int64_t output_h = input_h + pad_top + pad_bottom;
      const int64_t output_w = input_w + pad_left + pad_right;
      CHECK_OR_RETURN(pad_left < input_w && pad_right < input_w)
          << Error::RuntimeError()
          << "Padding size should be less than the corresponding input "
             "dimension, but got: padding ("
          << pad_left << ", " << pad_right << ") at dimension " << dim_w << " of input " << ndim;

      CHECK_OR_RETURN(pad_top < input_h && pad_bottom < input_h)
          << Error::RuntimeError()
          << "Padding size should be less than the corresponding input "
             "dimension, but got: padding ("
          << pad_top << ", " << pad_bottom << ") at dimension " << dim_h << " of input " << ndim;

      CHECK_OR_RETURN(output_w >= 1 || output_h >= 1)
          << Error::RuntimeError() << "input (H: " << input_h << ", W: " << input_w
          << ")is too small. Calculated output H: " << output_h << " W: " << output_w;

      if (ndim == 3) {
        // for 3D input
        auto unsqueezed_input = JUST(functional::Unsqueeze(input, 0));
        auto unsqueezed_output =
            JUST(OpInterpUtil::Dispatch<Tensor>(*reflect_pad2d_, {unsqueezed_input}, attrs));
        return JUST(functional::Squeeze(unsqueezed_output, std::vector<int32_t>{0}));
      }
      return OpInterpUtil::Dispatch<Tensor>(*reflect_pad2d_, {input}, attrs);
    } else if (pad_size == 6) {
      UNIMPLEMENTED_THEN_RETURN() << "5D reflect padding are not supported for now";
    } else {
      UNIMPLEMENTED_THEN_RETURN()
          << "Only 2D, 3D, 4D, 5D padding with non-constant padding are supported for now";
    }
  }

 private:
  std::shared_ptr<OpExpr> reflect_pad1d_;
  std::shared_ptr<OpExpr> reflect_pad2d_;
};

class ReplicationPadFunctor {
 public:
  ReplicationPadFunctor() {
    replicate_pad1d_ =
        CHECK_JUST(one::OpBuilder("replication_pad1d").Input("x").Output("y").Build());
    replicate_pad2d_ =
        CHECK_JUST(one::OpBuilder("replication_pad2d").Input("x").Output("y").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::vector<int64_t>& pad) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("padding");
    attrs.SetAllAttrs(pad);
    const int64_t pad_size = pad.size();
    const size_t ndim = input->ndim();
    CHECK_LE_OR_RETURN(pad_size, 2 * ndim)
        << Error::RuntimeError() << "Pad size should less than or equal to input axes * 2.";
    if (pad_size == 2) {
      // 2D/3D replicate padding
      CHECK_OR_RETURN((ndim == 2 && input->shape()->At(0) != 0 && input->shape()->At(1) != 0)
                      || (ndim == 3 && input->shape()->At(1) != 0 && input->shape()->At(2) != 0))
          << "Expected 2D or 3D (batch mode) tensor with possibly 0 batch size and other "
             "non-zero dimensions for input, but got: "
          << ndim;
      const int64_t pad_left = pad[0];
      const int64_t pad_right = pad[1];
      const int64_t dim_w = (ndim == 3) ? 2 : 1;
      const int64_t input_width = input->shape()->At(dim_w);
      const int64_t output_w = input_width + pad_left + pad_right;
      CHECK_OR_RETURN(output_w >= 1)
          << "input (W: " << input_width << ")is too small. Calculated output W: " << output_w;

      if (ndim == 2) {
        // for 2D input
        auto unsqueezed_input = JUST(functional::Unsqueeze(input, 0));
        auto unsqueezed_output =
            JUST(OpInterpUtil::Dispatch<Tensor>(*replicate_pad1d_, {unsqueezed_input}, attrs));
        return JUST(functional::Squeeze(unsqueezed_output, std::vector<int32_t>{0}));
      }
      return OpInterpUtil::Dispatch<Tensor>(*replicate_pad1d_, {input}, attrs);
    } else if (pad_size == 4) {
      // 3D/4D replicate padding
      bool valid_dims = input->shape()->At(1) != 0 && input->shape()->At(2) != 0;
      CHECK_OR_RETURN((ndim == 3 && valid_dims)
                      || (ndim == 4 && valid_dims && input->shape()->At(3) != 0))
          << "3D or 4D (batch mode) tensor expected for input, but got: " << ndim;

      int dim_h = 1;
      int dim_w = 2;
      if (ndim == 4) {
        dim_w++;
        dim_h++;
      }

      const int64_t pad_left = pad[0];
      const int64_t pad_right = pad[1];
      const int64_t pad_top = pad[2];
      const int64_t pad_bottom = pad[3];

      const int64_t input_h = input->shape()->At(dim_h);
      const int64_t input_w = input->shape()->At(dim_w);
      const int64_t output_h = input_h + pad_top + pad_bottom;
      const int64_t output_w = input_w + pad_left + pad_right;
      CHECK_OR_RETURN(output_w >= 1 || output_h >= 1)
          << Error::RuntimeError() << "input (H: " << input_h << ", W: " << input_w
          << ")is too small. Calculated output H: " << output_h << " W: " << output_w;

      if (ndim == 3) {
        // for 3D input
        auto unsqueezed_input = JUST(functional::Unsqueeze(input, 0));
        auto unsqueezed_output =
            JUST(OpInterpUtil::Dispatch<Tensor>(*replicate_pad2d_, {unsqueezed_input}, attrs));
        return JUST(functional::Squeeze(unsqueezed_output, std::vector<int32_t>{0}));
      }
      return OpInterpUtil::Dispatch<Tensor>(*replicate_pad2d_, {input}, attrs);
    } else if (pad_size == 6) {
      UNIMPLEMENTED_THEN_RETURN() << "5D replicate padding are not supported for now";
    } else {
      UNIMPLEMENTED_THEN_RETURN()
          << "Only 2D, 3D, 4D, 5D padding with non-constant padding are supported for now";
    }
  }

 private:
  std::shared_ptr<OpExpr> replicate_pad1d_;
  std::shared_ptr<OpExpr> replicate_pad2d_;
};

class PadFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::vector<int64_t>& pad, const std::string& mode,
                           const Scalar& value) const {
    if (mode == "constant") {
      return functional::ConstantPad(input, pad, value);
    } else if (mode == "reflect") {
      return functional::ReflectionPad(input, pad);
    } else if (mode == "replicate") {
      return functional::ReplicationPad(input, pad);
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "Pad mode is " << mode
                                  << ", but only constant, reflect and replicate are valid.";
    }
  }
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

    auto gen = generator.value_or(JUST(one::DefaultAutoGenerator()));
    gen = JUST(GetGeneratorForLazyOrGlobal(gen, LazyMode::is_enabled(), x));
    auto& dropout_attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("rate", "seed");
    dropout_attrs.SetAllAttrs(p, static_cast<int64_t>(gen->current_seed()));

    const auto& dropout_state = std::make_shared<FusedDropoutKernelState>(gen);
    OpExprInterpContext ctx(dropout_attrs, dropout_state);
    if (addend) {
      if ((!training) || p == 0.0) {
        JUST(OpInterpUtil::Dispatch(*add_op_, {x, JUST(addend)}, outputs.get()));
      } else {
        outputs->resize(2);
        JUST(OpInterpUtil::Dispatch(*dropout_addend_op_, {x, JUST(addend)}, outputs.get(), ctx));
      }
    } else {
      if (!training || p == 0.0) {
        return x;
      } else {
        outputs->resize(2);
        JUST(OpInterpUtil::Dispatch(*dropout_op_, {x}, outputs.get(), ctx));
      }
    }
    return (*outputs)[0];
  }

 private:
  std::shared_ptr<OpExpr> dropout_op_;
  std::shared_ptr<OpExpr> dropout_addend_op_;
  std::shared_ptr<OpExpr> add_op_;
};

namespace {
Maybe<Tensor> MakeFeatureNoise(const std::shared_ptr<one::Tensor>& x) {
  const int64_t ndim = x->ndim();
  CHECK_GE_OR_RETURN(ndim, 2) << Error::RuntimeError()
                              << "Feature dropout requires at least 2 dimensions in the input";
  std::vector<int64_t> sizes;
  sizes.reserve(ndim);
  sizes.push_back(x->shape()->At(0));
  sizes.push_back(x->shape()->At(1));
  for (int i = 2; i < ndim; i++) { sizes.push_back(1); }
  return JUST(Empty(Shape(sizes), x->dtype(), JUST(x->device()),
                    /*requires_grad=*/x->requires_grad(),
                    /*pin_memory=*/false));
}

Maybe<Tensor> DropoutImpl(const std::shared_ptr<one::Tensor>& input, const float& p,
                          const bool& train) {
  CHECK_EQ_OR_RETURN(p >= 0 && p <= 1, true)
      << "dropout probability has to be between 0 and 1, but got " << p;
  if (p == 0 || !train || input->shape()->elem_cnt() == 0) { return input; }
  if (p == 1) {
    std::shared_ptr<Tensor> other =
        JUST(Constant(*input->shape(), Scalar(0.0), input->dtype(), JUST(input->device())));
    return Mul(input, other);
  }
  std::shared_ptr<Tensor> noise = JUST(MakeFeatureNoise(input));
  noise =
      JUST(BernoulliProb(noise, 1.0 - p, noise->dtype(), JUST(one::DefaultAutoGenerator()), false));
  noise = JUST(InplaceScalarDiv(noise, Scalar(1.0 - p)));
  return JUST(Mul(input, noise));
}
}  // namespace

class Dropout1dFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input, const float& p,
                           const bool& training) const {
    CHECK_EQ_OR_RETURN(p < 0 || p > 1.0, false)
        << "dropout probability has to be between 0 and 1, but got " << p;
    const int input_dim = input->ndim();
    CHECK_EQ_OR_RETURN(input_dim != 2 && input_dim != 3, false)
        << "dropout1d: Expected 2D or 3D input, but received a " << input_dim
        << "D input. "
           "Note that dropout1d exists to provide channel-wise dropout on inputs with 1 "
           "spatial dimension, a channel dimension, and an optional batch dimension "
           "(i.e. 2D or 3D inputs).";
    bool is_batched = (input_dim == 3);
    std::shared_ptr<one::Tensor> result = input;
    if (!is_batched) { result = JUST(Unsqueeze(input, 0)); }
    result = JUST(DropoutImpl(result, p, training));
    if (!is_batched) { result = JUST(Squeeze(result, std::vector<int32_t>{0})); }
    return result;
  }
};

class Dropout2dFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input, const float& p,
                           const bool& training) const {
    CHECK_EQ_OR_RETURN(p < 0 || p > 1.0, false)
        << "dropout probability has to be between 0 and 1, but got " << p;
    const int input_dim = input->ndim();
    if (input_dim != 3 && input_dim != 4) {
      LOG(WARNING)
          << "dropout2d: Received a " << input_dim
          << "-D input to dropout2d, which is deprecated "
             "and will result in an error in a future release. To retain the behavior "
             "and silence this warning, please use dropout instead. Note that dropout2d "
             "exists to provide channel-wise dropout on inputs with 2 spatial dimensions, "
             "a channel dimension, and an optional batch dimension (i.e. 3D or 4D inputs).";
    }
    if (input_dim == 3) {
      LOG(WARNING)
          << "dropout2d: Received a 3D input to dropout2d and assuming that channel-wise "
             "1D dropout behavior is desired - input is interpreted as shape (N, C, L), where C "
             "is the channel dim. This behavior will change in a future release to interpret the "
             "input as one without a batch dimension, i.e. shape (C, H, W). To maintain the 1D "
             "channel-wise dropout behavior, please switch to using dropout1d instead.";
    }
    return JUST(DropoutImpl(input, p, training));
  }
};

class Dropout3dFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input, const float& p,
                           const bool& training) const {
    CHECK_EQ_OR_RETURN(p < 0 || p > 1.0, false)
        << "dropout probability has to be between 0 and 1, but got " << p;
    const int input_dim = input->ndim();
    if (input_dim != 4 && input_dim != 5) {
      LOG(WARNING)
          << "dropout3d: Received a " << input_dim
          << "-D input to dropout3d, which is deprecated "
             "and will result in an error in a future release. To retain the behavior "
             "and silence this warning, please use dropout instead. Note that dropout3d "
             "exists to provide channel-wise dropout on inputs with 3 spatial dimensions, "
             "a channel dimension, and an optional batch dimension (i.e. 4D or 5D inputs).";
    }
    bool is_batched = (input_dim == 5);
    std::shared_ptr<one::Tensor> result = input;
    if (!is_batched) { result = JUST(Unsqueeze(input, 0)); }
    result = JUST(DropoutImpl(result, p, training));
    if (!is_batched) { result = JUST(Squeeze(result, std::vector<int32_t>{0})); }
    return result;
  }
};

class DropoutGradFunctor {
 public:
  DropoutGradFunctor() {
    dropout_grad_op_ =
        CHECK_JUST(one::OpBuilder("dropout_grad").Input("dy").Input("mask").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& mask, const float& scale) const {
    auto& dropout_grad_attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("scale");
    dropout_grad_attrs.SetAllAttrs(scale);
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
    auto& attrs =
        THREAD_CACHED_MUTABLE_ATTR_MAP("kernel_size", "padding", "stride", "data_format",
                                       "ceil_mode", "count_include_pad", "divisor_override");
    // If stride is None, we set it as kernel_size to align Pytorch.
    attrs.SetAllAttrs(kernel_size, padding, stride ? *JUST(stride) : kernel_size, data_format,
                      ceil_mode, count_include_pad, divisor_override);
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
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::vector<int32_t>& kernel_size,
                           const std::vector<int32_t>& dilation_rate,
                           const std::vector<int32_t>& padding, const std::vector<int32_t>& strides,
                           const std::string& data_format) const {
    const auto& x_shape = x->shape();
    // Only Support 4d tensor now.
    CHECK_EQ_OR_RETURN(x_shape->NumAxes(), 4)
        << Error::RuntimeError() << "Input Tensor dim should == 4";
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("kernel_size", "dilation_rate", "padding",
                                                 "strides", "data_format");
    attrs.SetAllAttrs(kernel_size, dilation_rate, padding, strides, data_format);
    return OpInterpUtil::Dispatch<Tensor>(*unfold_op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> unfold_op_;
};

class FoldFunctor {
 public:
  FoldFunctor() { fold_op_ = CHECK_JUST(one::OpBuilder("fold").Input("x").Output("y").Build()); }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::vector<int32_t>& output_size,
                           const std::vector<int32_t>& kernel_size,
                           const std::vector<int32_t>& dilation_rate,
                           const std::vector<int32_t>& padding, const std::vector<int32_t>& strides,
                           const std::string& data_format) const {
    const auto& x_shape = x->shape();
    // Only Support 3d tensor fold now. format is (N, C*K*K, L)
    CHECK_EQ_OR_RETURN(x_shape->NumAxes(), 3)
        << Error::RuntimeError() << "Input Tensor dim should == 3";
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("output_size", "kernel_size", "dilation_rate",
                                                 "padding", "strides", "data_format");
    attrs.SetAllAttrs(output_size, kernel_size, dilation_rate, padding, strides, data_format);
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
    CHECK_OR_RETURN(!IsFloatingDataType(input->dtype()->data_type()))
        << Error::RuntimeError() << "one_hot is only applicable to index tensor.";
    auto& attrs =
        THREAD_CACHED_MUTABLE_ATTR_MAP("depth", "dtype", "floating_on_value", "floating_off_value",
                                       "integer_on_value", "integer_off_value");
    int64_t depth = num_classes;
    if (num_classes == -1) {
      std::vector<int32_t> axis(input->ndim());
      std::iota(axis.begin(), axis.end(), 0);
      auto tensor_max = JUST(functional::ReduceMax(input, axis, false));

      int64_t max = 0;
      const auto& callback = [&](ep::Stream* stream,
                                 const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object) {
        SyncAutoMemcpy(stream, &max, eager_blob_object->dptr(), sizeof(max),
                       memory::MakeHostMemCase(), eager_blob_object->mem_case());
      };
      JUST(SyncAccessTensorWithTimeOut(tensor_max, callback, "const"));
      depth = max + 1;
    }
    // Refer to: https://github.com/Oneflow-Inc/oneflow/pull/5315/files#r755823506
    bool is_on_value_double = on_value.IsFloatingPoint();
    bool is_off_value_double = off_value.IsFloatingPoint();
    if (is_on_value_double || is_off_value_double) {
      attrs.SetAllAttrs(depth, kFloat, on_value.As<double>(), off_value.As<double>(),
                        static_cast<int64_t>(0), static_cast<int64_t>(0));
    } else {
      attrs.SetAllAttrs(depth, kInt64, static_cast<double>(0), static_cast<double>(0),
                        on_value.As<int64_t>(), off_value.As<int64_t>());
    }
    return OpInterpUtil::Dispatch<Tensor>(*one_hot_op_, {input}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> one_hot_op_;
};

class PairwiseDistanceFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x, const std::shared_ptr<Tensor>& y,
                           const float& p, const double& eps, bool keepdim) const {
    const int64_t xdim = x->ndim();
    const int64_t ydim = y->ndim();
    const int64_t output_dim = xdim > ydim ? xdim : ydim;
    const auto& sub = JUST(ScalarAdd(JUST(Sub(x, y, 1, false)), eps, 1, false));
    return ScalarNorm(sub, p, output_dim - 1, keepdim, NullOpt);
  }
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
        int64_t size_x = (dim_x >= 0) ? x_shape.At(dim_x) : 1;
        int64_t size_y = (dim_y >= 0) ? y_shape.At(dim_y) : 1;
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
    const int32_t ndims = input->shape()->NumAxes();
    const int32_t final_dim = ndims - 1;

    auto axis_ = axis >= 0 ? axis : axis + ndims;
    CHECK_GE_OR_RETURN(axis_, 0) << Error::RuntimeError() << "Axis should >=0 but axis is " << axis_
                                 << " now.";
    CHECK_LE_OR_RETURN(axis_, final_dim) << Error::RuntimeError() << "Axis should < " << ndims
                                         << " but axis is " << axis_ << " now.";

    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("epsilon", "axis");
    attrs.SetAllAttrs(epsilon, final_dim);

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
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("head_size", "alpha");
    attrs.SetAllAttrs(head_size, alpha);
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
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("alpha");
    attrs.SetAllAttrs(alpha);
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
                                const float tril_fill_value,
                                const Optional<one::Generator>& generator) const {
    auto gen = generator.value_or(JUST(one::DefaultAutoGenerator()));
    gen = JUST(GetGeneratorForLazyOrGlobal(gen, LazyMode::is_enabled(), x));
    auto& random_mask_like_attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("rate", "seed");
    random_mask_like_attrs.SetAllAttrs(p, static_cast<int64_t>(gen->current_seed()));
    const auto& random_mask_like_state = std::make_shared<RandomMaskLikeKernelState>(gen);
    const auto& mask = JUST(OpInterpUtil::Dispatch<Tensor>(
        *random_mask_like_op_, {x},
        OpExprInterpContext(random_mask_like_attrs, random_mask_like_state)));

    float mask_scale_value = 1.0;
    if (p != 1.0) { mask_scale_value = 1.0 / (1.0 - p); }
    auto& fused_attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("diagonal", "tril_scale_value",
                                                       "mask_scale_value", "tril_fill_value");
    fused_attrs.SetAllAttrs(diagonal, tril_scale_value, mask_scale_value, tril_fill_value);
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
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("axis", "epsilon");
    attrs.SetAllAttrs(axis, epsilon);
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
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("axis");
    attrs.SetAllAttrs(axis);
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
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("axis");
    attrs.SetAllAttrs(axis);
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
    int32_t axis_val = axis;
    if (axis_val < 0) {
      const int64_t num_axes = a->shape()->NumAxes();
      axis_val += num_axes;
    }
    if (p > 0.0) {
      auto gen = generator.value_or(JUST(one::DefaultAutoGenerator()));
      gen = JUST(GetGeneratorForLazyOrGlobal(gen, LazyMode::is_enabled(), a));
      auto& random_mask_like_attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("rate", "seed");
      random_mask_like_attrs.SetAllAttrs(p, static_cast<int64_t>(gen->current_seed()));
      const auto& random_mask_like_state = std::make_shared<RandomMaskLikeKernelState>(gen);

      float scale = 0.0;
      if (p != 1.0) { scale = 1.0 / (1.0 - p); }
      auto& fused_bias_add_mask_attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("scale", "axis");
      fused_bias_add_mask_attrs.SetAllAttrs(scale, axis_val);

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

class FusedGluFunctor {
 public:
  FusedGluFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("fused_glu")
                         .Input("x")
                         .Input("w")
                         .Input("b")
                         .Output("y")
                         .Output("matmul_wx")
                         .Build());

    op_without_bias_ = CHECK_JUST(
        one::OpBuilder("fused_glu").Input("x").Input("w").Output("y").Output("matmul_wx").Build());

    split_op_ = CHECK_JUST(one::OpBuilder("fused_glu")
                               .Input("x")
                               .Input("w")
                               .Input("b")
                               .Input("v")
                               .Input("c")
                               .Output("y")
                               .Output("matmul_wx")
                               .Output("matmul_vx")
                               .Build());

    split_op_without_bias_ = CHECK_JUST(one::OpBuilder("fused_glu")
                                            .Input("x")
                                            .Input("w")
                                            .Input("v")
                                            .Output("y")
                                            .Output("matmul_wx")
                                            .Output("matmul_vx")
                                            .Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& w, const Optional<one::Tensor>& b,
                           const Optional<one::Tensor>& v, const Optional<one::Tensor>& c,
                           const std::string& activation) const {
    // check whether the user provide weight tensor v
    bool is_split_mode = false;
    if (v) {
      is_split_mode = true;
    } else {
      is_split_mode = false;
    }

    // check whether the user provide bias tensors
    bool has_bias = false;
    if (b) {
      has_bias = true;
      if (is_split_mode) {
        CHECK_OR_RETURN(c) << "expected existance of c, when provide tensors w, v and b";
      }
    } else {
      CHECK_OR_RETURN(!c) << "expected existance of b while providing c";
      has_bias = false;
    }

    // obtain input shape
    const auto& x_shape = *(x->shape());
    const auto& w_shape = *(w->shape());
    std::shared_ptr<const oneflow::Shape> b_shape = nullptr;
    if (has_bias) { b_shape = (JUST(b)->shape()); }

    // check number of axes of x, w and b
    CHECK_GT_OR_RETURN(x_shape.NumAxes(), 1)
        << "number of axes of \'x\' should have be greater than 1, yet get " << x_shape.NumAxes();
    CHECK_EQ_OR_RETURN(w_shape.NumAxes(), 2)
        << "number of axes of \'w\' should have be equal to 2, yet get " << w_shape.NumAxes();
    if (has_bias) {
      CHECK_EQ_OR_RETURN(b_shape->NumAxes(), 1)
          << "number of axes of \'b\' should have be equal to 1, yet get " << b_shape->NumAxes();
    }

    // check input shapes of w and b
    size_t x_num_axes = x_shape.NumAxes();
    CHECK_EQ_OR_RETURN(w_shape.At(1), x_shape.At(x_num_axes - 1))
        << "dimension 1 of \'w\'(" << w_shape.At(1)
        << ") is not consistant with the last dimension of \'x\'(" << x_shape.At(x_num_axes - 1)
        << ")";
    if (has_bias) {
      CHECK_EQ_OR_RETURN(b_shape->At(0), w_shape.At(0))
          << "dimension 0 of \'b\'(" << b_shape->At(0)
          << ") is not consistant with dimension 0 of \'w\'(" << w_shape.At(0) << ")";
    }
    if (!is_split_mode) {
      CHECK_EQ_OR_RETURN(w_shape.At(1) % 2, 0) << "dimension 1 of \'w\' is not divisible by 2";
    }

    // check both dimensions and input shapes of v and c (optional)
    if (is_split_mode) {
      const auto& v_shape = *(JUST(v)->shape());
      std::shared_ptr<const oneflow::Shape> c_shape = NULL;
      if (has_bias) { c_shape = (JUST(c)->shape()); }

      CHECK_EQ_OR_RETURN(v_shape.NumAxes(), 2)
          << "number of axes of \'v\' should have be equal to 2, yet get " << v_shape.NumAxes();
      if (has_bias) {
        CHECK_EQ_OR_RETURN(c_shape->NumAxes(), 1)
            << "number of axes of \'c\' should have be equal to 1, yet get " << c_shape->NumAxes();
      }

      CHECK_OR_RETURN(v_shape == w_shape) << "the shape of \'v\' is not consistant with \'w\'";
      if (has_bias) {
        CHECK_OR_RETURN((*c_shape) == (*b_shape))
            << "the shape of \'c\' is not consistant with \'b\'";
      }
    }

    // set activation attribute
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("activation", "has_bias", "is_split");
    attrs.SetAllAttrs(activation, has_bias, is_split_mode);

    // dispatch corresponding operator
    if (is_split_mode && has_bias) {
      return OpInterpUtil::Dispatch<one::Tensor>(*split_op_, {x, w, JUST(b), JUST(v), JUST(c)},
                                                 attrs);
    } else if (!is_split_mode && has_bias) {
      return OpInterpUtil::Dispatch<one::Tensor>(*op_, {x, w, JUST(b)}, attrs);
    } else if (is_split_mode && !has_bias) {
      return OpInterpUtil::Dispatch<one::Tensor>(*split_op_without_bias_, {x, w, JUST(v)}, attrs);
    } else if (!is_split_mode && !has_bias) {
      return OpInterpUtil::Dispatch<one::Tensor>(*op_without_bias_, {x, w}, attrs);
    } else {
      UNIMPLEMENTED_THEN_RETURN();
    }
  }

 private:
  std::shared_ptr<OpExpr> op_;
  std::shared_ptr<OpExpr> op_without_bias_;
  std::shared_ptr<OpExpr> split_op_;
  std::shared_ptr<OpExpr> split_op_without_bias_;
};

class FusedScaleTrilFunctor {
 public:
  FusedScaleTrilFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("fused_scale_tril").Input("in").Output("out").Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const int64_t& diagonal,
                           const Scalar& fill_value, const Scalar& scale) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP(
        "diagonal", "floating_fill_value", "is_floating_fill_value", "integer_fill_value",
        "floating_scale_value", "is_floating_scale_value", "integer_scale_value");
    bool is_fill_value_double = fill_value.IsFloatingPoint();
    bool is_scale_double = scale.IsFloatingPoint();

    double floating_fill_value = 0;
    int64_t integer_fill_value = 0;
    if (is_fill_value_double) {
      floating_fill_value = fill_value.As<double>();
    } else {
      integer_fill_value = fill_value.As<int64_t>();
    }
    double floating_scale_value = 0;
    int64_t integer_scale_value = 0;
    if (is_scale_double) {
      floating_scale_value = scale.As<double>();
    } else {
      integer_scale_value = scale.As<int64_t>();
    }
    attrs.SetAllAttrs(diagonal, floating_fill_value, is_fill_value_double, integer_fill_value,
                      floating_scale_value, is_scale_double, integer_scale_value);
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
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("scale_value", "mask_fill_value");
    attrs.SetAllAttrs(scale, fill_value);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x, mask}, attrs);
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
    auto gen = generator.value_or(JUST(one::DefaultAutoGenerator()));
    gen = JUST(GetGeneratorForLazyOrGlobal(gen, LazyMode::is_enabled(), x));
    auto& random_mask_like_attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("rate", "seed");
    random_mask_like_attrs.SetAllAttrs(rate, static_cast<int64_t>(gen->current_seed()));
    const auto& random_mask_like_state = std::make_shared<RandomMaskLikeKernelState>(gen);
    const auto& dropout_mask = JUST(OpInterpUtil::Dispatch<Tensor>(
        *random_mask_like_op_, {x},
        OpExprInterpContext(random_mask_like_attrs, random_mask_like_state)));

    float dropout_scale = 0.0;
    if (rate != 1.0) { dropout_scale = 1.0 / (1.0 - rate); }
    auto& fused_scale_mask_softmax_dropout_attrs =
        THREAD_CACHED_MUTABLE_ATTR_MAP("scale_value", "mask_fill_value", "dropout_scale_value");
    fused_scale_mask_softmax_dropout_attrs.SetAllAttrs(scale, fill_value, dropout_scale);
    return OpInterpUtil::Dispatch<TensorTuple>(*fused_scale_mask_softmax_dropout_op_,
                                               {x, mask, dropout_mask},
                                               fused_scale_mask_softmax_dropout_attrs);
  }

 private:
  std::shared_ptr<OpExpr> random_mask_like_op_;
  std::shared_ptr<OpExpr> fused_scale_mask_softmax_dropout_op_;
};

// Equivalent to
// masked = (x + bias) * mask * scale_value
// unmask = (1 - mask).bool()
// masked.masked_fill_(unmask, mask_fill_value)
// softmax_y = softmax(masked, dim=-1)
// y = dropout(softmax_y, p)
class FusedBiasAddScaleMaskSoftmaxDropoutFunctor {
 public:
  FusedBiasAddScaleMaskSoftmaxDropoutFunctor() {
    random_mask_op_ =
        CHECK_JUST(one::OpBuilder("random_mask_like").Input("like").Output("out").Build());
    fused_op_ = CHECK_JUST(one::OpBuilder("fused_bias_add_scale_mask_softmax_dropout")
                               .Input("x")
                               .Input("bias")
                               .Input("mask")
                               .Input("dropout_mask")
                               .Output("y")
                               .Output("softmax_y")
                               .Build());
  }
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& x,
                                const std::shared_ptr<one::Tensor>& bias,
                                const std::shared_ptr<one::Tensor>& mask, const float& fill_value,
                                const float& scale, const float& p, const bool& training,
                                const Optional<one::Generator>& generator) const {
    float rate = p;
    if (!training) rate = 0.0;
    auto gen = generator.value_or(JUST(one::DefaultAutoGenerator()));
    gen = JUST(GetGeneratorForLazyOrGlobal(gen, LazyMode::is_enabled(), x));
    auto& random_mask_like_attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("rate", "seed");
    random_mask_like_attrs.SetAllAttrs(rate, static_cast<int64_t>(gen->current_seed()));
    const auto& random_mask_like_state = std::make_shared<RandomMaskLikeKernelState>(gen);
    const auto& dropout_mask = JUST(OpInterpUtil::Dispatch<Tensor>(
        *random_mask_op_, {x},
        OpExprInterpContext(random_mask_like_attrs, random_mask_like_state)));

    float dropout_scale = 0.0;
    if (rate != 1.0) { dropout_scale = 1.0 / (1.0 - rate); }
    auto& fused_scale_mask_softmax_dropout_attrs =
        THREAD_CACHED_MUTABLE_ATTR_MAP("scale_value", "mask_fill_value", "dropout_scale_value");
    fused_scale_mask_softmax_dropout_attrs.SetAllAttrs(scale, fill_value, dropout_scale);
    return OpInterpUtil::Dispatch<TensorTuple>(*fused_op_, {x, bias, mask, dropout_mask},
                                               fused_scale_mask_softmax_dropout_attrs);
  }

 private:
  std::shared_ptr<OpExpr> random_mask_op_;
  std::shared_ptr<OpExpr> fused_op_;
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
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("merge_repeated");
    attrs.SetAllAttrs(merge_repeated);
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
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("iou_threshold", "keep_n");
    attrs.SetAllAttrs(iou_threshold, keep_n);
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
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("spatial_scale", "pooled_h", "pooled_w",
                                                 "sampling_ratio", "aligned");
    attrs.SetAllAttrs(spatial_scale, pooled_h, pooled_w, sampling_ratio, aligned);
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
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("spatial_scale", "pooled_h", "pooled_w",
                                                 "sampling_ratio", "aligned");
    attrs.SetAllAttrs(spatial_scale, pooled_h, pooled_w, sampling_ratio, aligned);
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
                                                 .Build());
    }
    for (int n = 0; n < ops_no_output_concat_.size(); ++n) {
      ops_no_output_concat_[n] = CHECK_JUST(one::OpBuilder("fused_dot_feature_interaction")
                                                .Input("features", n + 1)
                                                .Output("out")
                                                .Build());
    }
  }

  Maybe<Tensor> operator()(const TensorTuple& features, const Optional<one::Tensor>& output_concat,
                           const bool& self_interaction, const int32_t& output_padding,
                           const std::string& pooling) const {
    const int64_t n_features = features.size();
    TensorTuple inputs;
    if (n_features > kMaxInputCount) {
      inputs.push_back(JUST(functional::Concat(features, 1)));
    } else {
      inputs = features;
    }
    CHECK_OR_RETURN(pooling == "sum" || pooling == "none")
        << Error::RuntimeError() << "pooling should be sum or none, but get " << pooling;

    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("self_interaction", "output_padding", "pooling",
                                                 "has_output_concat");
    if (pooling == "sum") {
      CHECK_EQ_OR_RETURN(output_padding, 0)
          << Error::RuntimeError() << "output_padding should be equal to 0. ";
      CHECK_OR_RETURN(!output_concat) << Error::RuntimeError() << "output_concat should not exist";
      attrs.SetAllAttrs(self_interaction, output_padding, pooling, false);
      const std::shared_ptr<one::Tensor>& bi_interaction = JUST(OpInterpUtil::Dispatch<Tensor>(
          *JUST(oneflow::VectorAt(ops_no_output_concat_, n_features - 1)), inputs, attrs));
      std::vector<int32_t> reduce_axes_vec = {1};
      return functional::ReduceSum(bi_interaction, reduce_axes_vec, true);
    }
    if (output_concat) {
      attrs.SetAllAttrs(self_interaction, output_padding, pooling, true);
      inputs.push_back(JUST(output_concat));
      return OpInterpUtil::Dispatch<Tensor>(
          *JUST(oneflow::VectorAt(ops_has_output_concat_, n_features - 1)), inputs, attrs);
    } else {
      attrs.SetAllAttrs(self_interaction, output_padding, pooling, false);
      return OpInterpUtil::Dispatch<Tensor>(
          *JUST(oneflow::VectorAt(ops_no_output_concat_, n_features - 1)), inputs, attrs);
    }
  }

 private:
  std::vector<std::shared_ptr<OpExpr>> ops_has_output_concat_;
  std::vector<std::shared_ptr<OpExpr>> ops_no_output_concat_;
};

class FusedCrossFeatureInteractionFunctor {
 public:
  FusedCrossFeatureInteractionFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("fused_cross_feature_interaction")
                         .Input("x")
                         .Input("weight")
                         .Input("x0")
                         .Input("bias")
                         .Output("out")
                         .Output("matmul_result")
                         .Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& weight,
                           const std::shared_ptr<one::Tensor>& x0,
                           const std::shared_ptr<one::Tensor>& bias,
                           const std::string& interaction_mode) const {
    if (interaction_mode != "vector" && interaction_mode != "matrix") {
      UNIMPLEMENTED_THEN_RETURN()
          << "Fused Cross Interaction mode only support `vector` and `matrix`. ";
    }
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("interaction_mode");
    attrs.SetAllAttrs(interaction_mode);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x, weight, x0, bias}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
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
                                const Optional<one::Tensor>& table_ids, const int32_t& num_tables,
                                const std::string& embedding_name) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("num_tables", "embedding_name");
    attrs.SetAllAttrs(num_tables, embedding_name);
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

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& cur_rank_embeddings,
                           const std::shared_ptr<one::Tensor>& num_unique_matrix,
                           const std::shared_ptr<one::Tensor>& cur_rank_inverse_indices,
                           const std::shared_ptr<one::Tensor>& inverse_unique_partition_indices,
                           const std::string& embedding_name) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("embedding_size", "embedding_name");
    const int64_t num_axes = cur_rank_embeddings->shape()->NumAxes();
    attrs.SetAllAttrs(cur_rank_embeddings->shape()->At(num_axes - 1), embedding_name);
    return OpInterpUtil::Dispatch<Tensor>(
        *op_,
        {cur_rank_embeddings, num_unique_matrix, cur_rank_inverse_indices,
         inverse_unique_partition_indices},
        attrs);
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

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& embedding_grad,
                           const std::shared_ptr<one::Tensor>& num_unique_matrix,
                           const std::shared_ptr<one::Tensor>& cur_rank_inverse_indices,
                           const std::shared_ptr<one::Tensor>& inverse_unique_partition_indices,
                           const std::string& embedding_name) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("embedding_size", "embedding_name");
    const int64_t num_axes = embedding_grad->shape()->NumAxes();
    attrs.SetAllAttrs(embedding_grad->shape()->At(num_axes - 1), embedding_name);
    return OpInterpUtil::Dispatch<Tensor>(
        *op_,
        {embedding_grad, num_unique_matrix, cur_rank_inverse_indices,
         inverse_unique_partition_indices},
        attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class OneEmbeddingLookupFunctor {
 public:
  OneEmbeddingLookupFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("embedding_lookup")
                         .Input("num_unique_ids")
                         .Input("unique_ids")
                         .Input("table_ids")
                         .Output("unique_values")
                         .Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& num_unique_ids,
                           const std::shared_ptr<one::Tensor>& unique_ids,
                           const std::shared_ptr<one::Tensor>& table_ids,
                           const Symbol<DType>& dtype, const Symbol<DType>& embedding_dtype,
                           const int64_t line_size, const int64_t embedding_size,
                           const std::string& embedding_name, const std::string& embedding_tables,
                           const std::string& state_initializer, const int64_t seed) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("dtype", "embedding_dtype", "line_size",
                                                 "embedding_size", "embedding_name",
                                                 "embedding_tables", "state_initializer", "seed");
    attrs.SetAllAttrs(dtype->data_type(), embedding_dtype->data_type(), line_size, embedding_size,
                      embedding_name, embedding_tables, state_initializer, seed);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {num_unique_ids, unique_ids, table_ids}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class OneEmbeddingFusedLookupFunctor {
 public:
  OneEmbeddingFusedLookupFunctor() {
    op_has_table_ids_ = CHECK_JUST(one::OpBuilder("one_embedding_fused_lookup")
                                       .Input("shadow")
                                       .Input("ids")
                                       .Input("table_ids")
                                       .Output("embeddings")
                                       .Build());
    op_no_table_ids_ = CHECK_JUST(one::OpBuilder("one_embedding_fused_lookup")
                                      .Input("shadow")
                                      .Input("ids")
                                      .Output("embeddings")
                                      .Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& shadow,
                           const std::shared_ptr<one::Tensor>& ids,
                           const Optional<one::Tensor>& table_ids, const Symbol<DType>& dtype,
                           const std::string& embedding_name, const int64_t line_size,
                           const int64_t embedding_size, const bool is_full_cache,
                           const int32_t num_tables, const std::string& embedding_tables,
                           const Optional<int64_t>& padding_idx, const int64_t seed) const {
    int64_t padding_idx_val = -1;
    bool has_padding_idx = false;
    if (padding_idx.has_value()) {
      padding_idx_val = JUST(padding_idx);
      has_padding_idx = true;
    }
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP(
        "dtype", "embedding_name", "line_size", "embedding_size", "is_full_cache", "num_tables",
        "embedding_tables", "seed", "padding_idx", "has_padding_idx");
    attrs.SetAllAttrs(dtype->data_type(), embedding_name, line_size, embedding_size, is_full_cache,
                      num_tables, embedding_tables, seed, padding_idx_val, has_padding_idx);
    if (table_ids) {
      const auto& table_ids_shape = *(JUST(table_ids)->shape());
      const auto& ids_shape = *(ids->shape());
      auto broadcast_table_ids = JUST(table_ids);
      if (table_ids_shape != ids_shape) {
        CHECK_LE_OR_RETURN(table_ids_shape.NumAxes(), ids_shape.NumAxes())
            << "table_ids num_axes should be less equal to ids num_axes, but got table_ids "
               "num_axes "
            << table_ids_shape.NumAxes() << " and ids num_axes " << ids_shape.NumAxes();
        const int64_t left_extend_dims = ids_shape.NumAxes() - table_ids_shape.NumAxes();
        for (int64_t i = 0; i < table_ids_shape.NumAxes(); i++) {
          CHECK_EQ_OR_RETURN(table_ids_shape.at(i), ids_shape.at(left_extend_dims + i))
              << "when table_ids's shape not equals ids shape, table_ids must be able to be "
                 "broadcast to ids_shape "
                 "but got table_ids_shape: "
              << table_ids_shape.DebugStr() << ", ids_shape: " << ids_shape.DebugStr();
        }
        broadcast_table_ids =
            JUST(functional::BroadcastLike(JUST(table_ids), ids, std::vector<int32_t>{}));
      }
      return OpInterpUtil::Dispatch<Tensor>(*op_has_table_ids_, {shadow, ids, broadcast_table_ids},
                                            attrs);
    } else {
      return OpInterpUtil::Dispatch<Tensor>(*op_no_table_ids_, {shadow, ids}, attrs);
    }
  }

 private:
  std::shared_ptr<OpExpr> op_has_table_ids_;
  std::shared_ptr<OpExpr> op_no_table_ids_;
};

class OneEmbeddingFusedLookupGradFunctor {
 public:
  OneEmbeddingFusedLookupGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("one_embedding_fused_lookup_grad")
                         .Input("ids")
                         .Input("embedding_grad")
                         .Build());
  }

  Maybe<void> operator()(const std::shared_ptr<one::Tensor>& ids,
                         const std::shared_ptr<one::Tensor>& embedding_grad,
                         const std::string& embedding_name, const int64_t line_size,
                         const int64_t embedding_size) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("embedding_name", "line_size", "embedding_size");
    attrs.SetAllAttrs(embedding_name, line_size, embedding_size);
    JUST(OpInterpUtil::Dispatch<TensorTuple>(*op_, {ids, embedding_grad}, attrs));
    return Maybe<void>::Ok();
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class OneEmbeddingEmbeddingPutFunctor {
 public:
  OneEmbeddingEmbeddingPutFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("embedding_put")
                         .Input("num_unique_ids")
                         .Input("unique_ids")
                         .Input("unique_embeddings")
                         .Build());
  }

  Maybe<void> operator()(const std::shared_ptr<one::Tensor>& num_unique_ids,
                         const std::shared_ptr<one::Tensor>& unique_ids,
                         const std::shared_ptr<one::Tensor>& unique_embeddings,
                         const std::string& embedding_name, const int64_t line_size) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("embedding_name", "line_size");
    attrs.SetAllAttrs(embedding_name, line_size);
    JUST(OpInterpUtil::Dispatch<TensorTuple>(*op_, {num_unique_ids, unique_ids, unique_embeddings},
                                             attrs));
    return Maybe<void>::Ok();
  }

 private:
  std::shared_ptr<OpExpr> op_;
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
                                const Optional<one::Tensor>& values, const int32_t num_tables,
                                const std::string& embedding_name) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("num_tables", "embedding_name");
    attrs.SetAllAttrs(num_tables, embedding_name);
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
    // This functor is only used in one_embedding eager mode with lr passed by attr and no optional
    // input, we also define functor with all optional input just for unittest. when the optional
    // input learning_rate tensor has passed in, we think all optional input are not None and check
    // them.
    sgd_no_optional_input_op_ = CHECK_JUST(one::OpBuilder("one_embedding_sgd_update")
                                               .Input("num_unique_ids")
                                               .Input("unique_embeddings")
                                               .Input("embedding_grad")
                                               .Output("updated_unique_embeddings")
                                               .Build());
    momentum_no_optional_input_op_ = CHECK_JUST(one::OpBuilder("one_embedding_momentum_update")
                                                    .Input("num_unique_ids")
                                                    .Input("unique_embeddings")
                                                    .Input("embedding_grad")
                                                    .Output("updated_unique_embeddings")
                                                    .Build());
    // This functor is just for unittest
    sgd_op_ = CHECK_JUST(one::OpBuilder("one_embedding_sgd_update")
                             .Input("num_unique_ids")
                             .Input("unique_embeddings")
                             .Input("embedding_grad")
                             .Input("learning_rate")
                             .Input("down_scale_by_tensor")
                             .Input("skip_if")
                             .Output("updated_unique_embeddings")
                             .Build());
    momentum_op_ = CHECK_JUST(one::OpBuilder("one_embedding_momentum_update")
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
                           const Optional<one::Tensor>& learning_rate,
                           const Optional<one::Tensor>& down_scale_by_tensor,
                           const Optional<one::Tensor>& skip_if, const float learning_rate_val,
                           const double scale, const float weight_decay, const float momentum,
                           const int64_t line_size, const int64_t embedding_size,
                           const std::string& embedding_name) const {
    auto& attrs =
        THREAD_CACHED_MUTABLE_ATTR_MAP("learning_rate_val", "scale", "weight_decay", "line_size",
                                       "embedding_size", "embedding_name", "beta");
    if (momentum == 0) {
      attrs.SetAllAttrs(learning_rate_val, scale, weight_decay, line_size, embedding_size,
                        embedding_name, NullOpt);

      if (learning_rate) {
        CHECK(down_scale_by_tensor);
        CHECK(skip_if);
        return OpInterpUtil::Dispatch<Tensor>(
            *sgd_op_,
            {num_unique_ids, unique_embeddings, embedding_grad, JUST(learning_rate),
             JUST(down_scale_by_tensor), JUST(skip_if)},
            attrs);
      } else {
        CHECK(!down_scale_by_tensor);
        CHECK(!skip_if);
        return OpInterpUtil::Dispatch<Tensor>(
            *sgd_no_optional_input_op_, {num_unique_ids, unique_embeddings, embedding_grad}, attrs);
      }
    } else {
      attrs.SetAllAttrs(learning_rate_val, scale, weight_decay, line_size, embedding_size,
                        embedding_name, momentum);
      if (learning_rate) {
        CHECK(down_scale_by_tensor);
        CHECK(skip_if);
        return OpInterpUtil::Dispatch<Tensor>(
            *momentum_op_,
            {num_unique_ids, unique_embeddings, embedding_grad, JUST(learning_rate),
             JUST(down_scale_by_tensor), JUST(skip_if)},
            attrs);
      } else {
        CHECK(!down_scale_by_tensor);
        CHECK(!skip_if);
        return OpInterpUtil::Dispatch<Tensor>(*momentum_no_optional_input_op_,
                                              {num_unique_ids, unique_embeddings, embedding_grad},
                                              attrs);
      }
    }
  }

 private:
  std::shared_ptr<OpExpr> sgd_no_optional_input_op_;
  std::shared_ptr<OpExpr> sgd_op_;
  std::shared_ptr<OpExpr> momentum_no_optional_input_op_;
  std::shared_ptr<OpExpr> momentum_op_;
};

class OneEmbeddingAdamUpdateFunctor {
 public:
  OneEmbeddingAdamUpdateFunctor() {
    // This functor is only used in one_embedding eager mode with lr passed by attr and no optional
    // input, we also define functor with all optional input just for unittest. when the optional
    // input learning_rate tensor has passed in, we think all optional input are not None and check
    // them.
    no_optional_input_op_ = CHECK_JUST(one::OpBuilder("one_embedding_adam_update")
                                           .Input("num_unique_ids")
                                           .Input("unique_embeddings")
                                           .Input("embedding_grad")
                                           .Output("updated_unique_embeddings")
                                           .Build());
    // This functor is just for unittest
    no_bias_correction_op_ = CHECK_JUST(one::OpBuilder("one_embedding_adam_update")
                                            .Input("num_unique_ids")
                                            .Input("unique_embeddings")
                                            .Input("embedding_grad")
                                            .Input("learning_rate")
                                            .Input("down_scale_by_tensor")
                                            .Input("skip_if")
                                            .Output("updated_unique_embeddings")
                                            .Build());
    do_bias_correction_op_ = CHECK_JUST(one::OpBuilder("one_embedding_adam_update")
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

  Maybe<Tensor> operator()(
      const std::shared_ptr<one::Tensor>& num_unique_ids,
      const std::shared_ptr<one::Tensor>& unique_embeddings,
      const std::shared_ptr<one::Tensor>& embedding_grad,
      const Optional<one::Tensor>& learning_rate, const Optional<one::Tensor>& down_scale_by_tensor,
      const Optional<one::Tensor>& skip_if, const Optional<one::Tensor>& bias_correction1,
      const Optional<one::Tensor>& bias_correction2, const float learning_rate_val,
      const double scale, const float weight_decay, const float beta1, const float beta2,
      const float& bias_correction1_val, const float& bias_correction2_val, const float epsilon,
      const bool do_bias_correction, const int64_t line_size, const int64_t embedding_size,
      const std::string& embedding_name) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP(
        "learning_rate_val", "scale", "weight_decay", "beta1", "beta2", "epsilon",
        "bias_correction1_val", "bias_correction2_val", "do_bias_correction", "line_size",
        "embedding_size", "embedding_name");
    attrs.SetAllAttrs(learning_rate_val, scale, weight_decay, beta1, beta2, epsilon,
                      bias_correction1_val, bias_correction2_val, do_bias_correction, line_size,
                      embedding_size, embedding_name);
    if (learning_rate) {
      CHECK(down_scale_by_tensor);
      CHECK(skip_if);
      if (do_bias_correction) {
        CHECK(bias_correction1);
        CHECK(bias_correction2);
        return OpInterpUtil::Dispatch<Tensor>(
            *do_bias_correction_op_,
            {num_unique_ids, unique_embeddings, embedding_grad, JUST(learning_rate),
             JUST(down_scale_by_tensor), JUST(skip_if), JUST(bias_correction1),
             JUST(bias_correction2)},
            attrs);
      } else {
        return OpInterpUtil::Dispatch<Tensor>(
            *no_bias_correction_op_,
            {num_unique_ids, unique_embeddings, embedding_grad, JUST(learning_rate),
             JUST(down_scale_by_tensor), JUST(skip_if)},
            attrs);
      }
    } else {
      CHECK(!down_scale_by_tensor);
      CHECK(!skip_if);
      CHECK(!bias_correction1);
      CHECK(!bias_correction2);
      return OpInterpUtil::Dispatch<Tensor>(
          *no_optional_input_op_, {num_unique_ids, unique_embeddings, embedding_grad}, attrs);
    }
  }

 private:
  std::shared_ptr<OpExpr> no_bias_correction_op_;
  std::shared_ptr<OpExpr> do_bias_correction_op_;
  std::shared_ptr<OpExpr> no_optional_input_op_;
};

class OneEmbeddingAdagradUpdateFunctor {
 public:
  OneEmbeddingAdagradUpdateFunctor() {
    // This functor is only used in one_embedding eager mode with lr passed by attr and no optional
    // input, we also define functor with all optional input just for unittest. when the optional
    // input learning_rate tensor has passed in, we think all optional input are not None and check
    // them.
    op_no_optional_input_ = CHECK_JUST(one::OpBuilder("one_embedding_adagrad_update")
                                           .Input("num_unique_ids")
                                           .Input("unique_embeddings")
                                           .Input("embedding_grad")
                                           .Output("updated_unique_embeddings")
                                           .Build());
    // This functor is just for unittest
    op_ = CHECK_JUST(one::OpBuilder("one_embedding_adagrad_update")
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
                           const Optional<one::Tensor>& learning_rate,
                           const Optional<one::Tensor>& down_scale_by_tensor,
                           const Optional<one::Tensor>& skip_if,
                           const Optional<one::Tensor>& train_step, const int64_t train_step_val,
                           const float learning_rate_val, const double scale,
                           const float weight_decay, const float lr_decay, const float epsilon,
                           const int64_t line_size, const int64_t embedding_size,
                           const std::string& embedding_name) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("train_step_val", "learning_rate_val", "scale",
                                                 "weight_decay", "lr_decay", "epsilon", "line_size",
                                                 "embedding_size", "embedding_name");
    attrs.SetAllAttrs(train_step_val, learning_rate_val, scale, weight_decay, lr_decay, epsilon,
                      line_size, embedding_size, embedding_name);
    if (learning_rate) {
      CHECK(down_scale_by_tensor);
      CHECK(skip_if);
      CHECK(train_step);
      return OpInterpUtil::Dispatch<Tensor>(
          *op_,
          {num_unique_ids, unique_embeddings, embedding_grad, JUST(learning_rate),
           JUST(down_scale_by_tensor), JUST(skip_if), JUST(train_step)},
          attrs);
    } else {
      CHECK(!down_scale_by_tensor);
      CHECK(!skip_if);
      CHECK(!train_step);
      return OpInterpUtil::Dispatch<Tensor>(
          *op_no_optional_input_, {num_unique_ids, unique_embeddings, embedding_grad}, attrs);
    }
  }

 private:
  std::shared_ptr<OpExpr> op_;
  std::shared_ptr<OpExpr> op_no_optional_input_;
};

class OneEmbeddingFtrlUpdateFunctor {
 public:
  OneEmbeddingFtrlUpdateFunctor() {
    // This functor is only used in one_embedding eager mode with lr passed by attr and no optional
    // input, we also define functor with all optional input just for unittest. when the optional
    // input learning_rate tensor has passed in, we think all optional input are not None and check
    // them.
    op_no_optional_input_ = CHECK_JUST(one::OpBuilder("one_embedding_ftrl_update")
                                           .Input("num_unique_ids")
                                           .Input("unique_embeddings")
                                           .Input("embedding_grad")
                                           .Output("updated_unique_embeddings")
                                           .Build());
    // This functor is just for unittest
    op_ = CHECK_JUST(one::OpBuilder("one_embedding_ftrl_update")
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
                           const Optional<one::Tensor>& learning_rate,
                           const Optional<one::Tensor>& down_scale_by_tensor,
                           const Optional<one::Tensor>& skip_if, const float learning_rate_val,
                           const double scale, const float weight_decay, const float lr_power,
                           const float lambda1, const float lambda2, const float beta,
                           const int64_t line_size, const int64_t embedding_size,
                           const std::string& embedding_name) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("learning_rate_val", "scale", "weight_decay",
                                                 "lr_power", "lambda1", "lambda2", "beta",
                                                 "line_size", "embedding_size", "embedding_name");
    attrs.SetAllAttrs(learning_rate_val, scale, weight_decay, lr_power, lambda1, lambda2, beta,
                      line_size, embedding_size, embedding_name);
    if (learning_rate) {
      CHECK(down_scale_by_tensor);
      CHECK(skip_if);
      return OpInterpUtil::Dispatch<Tensor>(
          *op_,
          {num_unique_ids, unique_embeddings, embedding_grad, JUST(learning_rate),
           JUST(down_scale_by_tensor), JUST(skip_if)},
          attrs);
    } else {
      CHECK(!down_scale_by_tensor);
      CHECK(!skip_if);
      return OpInterpUtil::Dispatch<Tensor>(
          *op_no_optional_input_, {num_unique_ids, unique_embeddings, embedding_grad}, attrs);
    }
  }

 private:
  std::shared_ptr<OpExpr> op_;
  std::shared_ptr<OpExpr> op_no_optional_input_;
};

class DeformConv2dFunctor {
 public:
  DeformConv2dFunctor() {
    bias_op_ = CHECK_JUST(one::OpBuilder("bias_add").Input("a").Input("b").Output("out").Build());
    deformconv2d_op_ = CHECK_JUST(one::OpBuilder("deform_conv2d")
                                      .Input("input")
                                      .Input("weight")
                                      .Input("offset")
                                      .Input("mask")
                                      .Output("output")
                                      .Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& weight,
                           const std::shared_ptr<one::Tensor>& offset,
                           const std::shared_ptr<one::Tensor>& mask,
                           const Optional<one::Tensor>& bias, const int32_t& stride_h,
                           const int32_t& stride_w, const int32_t& pad_h, const int32_t& pad_w,
                           const int32_t& dilation_h, const int32_t& dilation_w,
                           const int32_t& groups, const int32_t& offset_groups,
                           const bool& use_mask) const {
    auto& attrs =
        THREAD_CACHED_MUTABLE_ATTR_MAP("stride_h", "stride_w", "pad_h", "pad_w", "dilation_h",
                                       "dilation_w", "groups", "offset_groups", "use_mask");
    attrs.SetAllAttrs(stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups,
                      offset_groups, use_mask);
    const std::shared_ptr<one::Tensor>& deformconv2d_out = JUST(
        OpInterpUtil::Dispatch<Tensor>(*deformconv2d_op_, {input, weight, offset, mask}, attrs));
    if (bias) {
      auto bias_shape = JUST(bias)->shape();
      auto& bias_attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("axis");
      bias_attrs.SetAllAttrs(static_cast<int32_t>(1));
      return OpInterpUtil::Dispatch<Tensor>(*bias_op_, {deformconv2d_out, JUST(bias)}, bias_attrs);
    }
    return deformconv2d_out;
  }

 private:
  std::shared_ptr<OpExpr> deformconv2d_op_;
  std::shared_ptr<OpExpr> bias_op_;
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

class MultiTensorSgdUpdateFunctor {
 public:
  MultiTensorSgdUpdateFunctor() {
    op_.resize(kMaxInputCount /*the maximum number of inputs*/);
    for (int n = 0; n < op_.size(); ++n) {
      op_[n] = CHECK_JUST(one::OpBuilder("multi_tensor_sgd_update")
                              .Input("model", n + 1)
                              .Input("model_diff", n + 1)
                              .Build());
    }
  }

  Maybe<void> operator()(const TensorTuple& model, const TensorTuple& model_diff,
                         const double& scale, const float& weight_decay,
                         const float& learning_rate_val) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("scale", "weight_decay", "learning_rate_val");
    attrs.SetAllAttrs(scale, weight_decay, learning_rate_val);
    const int64_t weight_size = model.size();
    for (int i = 0; i < weight_size; i += kMaxInputCount) {
      size_t size = (i + kMaxInputCount) < weight_size ? kMaxInputCount : weight_size - i;
      TensorTuple input(2 * size);
      std::copy(model.begin() + i, model.begin() + i + size, input.begin());
      std::copy(model_diff.begin() + i, model_diff.begin() + i + size, input.begin() + size);
      JUST(OpInterpUtil::Dispatch<TensorTuple>(*op_[size - 1], input, attrs));
    }
    return Maybe<void>::Ok();
  }

 private:
  std::vector<std::shared_ptr<OpExpr>> op_;
};

class MultiTensorMomentumUpdateFunctor {
 public:
  MultiTensorMomentumUpdateFunctor() {
    op_.resize(kMaxInputCount /*the maximum number of inputs*/);
    for (int n = 0; n < op_.size(); ++n) {
      op_[n] = CHECK_JUST(one::OpBuilder("multi_tensor_momentum_update")
                              .Input("model", n + 1)
                              .Input("model_diff", n + 1)
                              .Input("momentum_buf", n + 1)
                              .Build());
    }
  }

  Maybe<void> operator()(const TensorTuple& model, const TensorTuple& model_diff,
                         const TensorTuple& momentum_buf, const double& scale,
                         const float& weight_decay, const float& learning_rate_val,
                         const float& momentum, const float& dampening, const bool& nesterov,
                         const bool& maximize) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("scale", "weight_decay", "learning_rate_val",
                                                 "momentum", "dampening", "nesterov", "maximize");
    attrs.SetAllAttrs(scale, weight_decay, learning_rate_val, momentum, dampening, nesterov,
                      maximize);
    const int64_t weight_size = model.size();
    for (int i = 0; i < weight_size; i += kMaxInputCount) {
      size_t size = (i + kMaxInputCount) < weight_size ? kMaxInputCount : weight_size - i;
      TensorTuple input(3 * size);
      std::copy(model.begin() + i, model.begin() + i + size, input.begin());
      std::copy(model_diff.begin() + i, model_diff.begin() + i + size, input.begin() + size);
      std::copy(momentum_buf.begin() + i, momentum_buf.begin() + i + size,
                input.begin() + 2 * size);
      JUST(OpInterpUtil::Dispatch<TensorTuple>(*op_[size - 1], input, attrs));
    }
    return Maybe<void>::Ok();
  }

 private:
  std::vector<std::shared_ptr<OpExpr>> op_;
};

class MultiTensorAdamUpdateFunctor {
 public:
  MultiTensorAdamUpdateFunctor() {
    op_.resize(kMaxInputCount /*the maximum number of inputs*/);
    for (int n = 0; n < op_.size(); ++n) {
      op_[n] = CHECK_JUST(one::OpBuilder("multi_tensor_adam_update")
                              .Input("model", n + 1)
                              .Input("model_diff", n + 1)
                              .Input("m", n + 1)
                              .Input("v", n + 1)
                              .Build());
    }
  }

  Maybe<void> operator()(const TensorTuple& model, const TensorTuple& model_diff,
                         const TensorTuple& m, const TensorTuple& v, const float& learning_rate_val,
                         const float& l2, const float& beta1, const float& beta2,
                         const float& bias_correction1_val, const float& bias_correction2_val,
                         const bool& do_bias_correction, const double& scale,
                         const float& weight_decay, const float& epsilon) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP(
        "scale", "weight_decay", "beta1", "beta2", "bias_correction1_val", "bias_correction2_val",
        "do_bias_correction", "learning_rate_val", "l2", "epsilon");
    attrs.SetAllAttrs(scale, weight_decay, beta1, beta2, bias_correction1_val, bias_correction2_val,
                      do_bias_correction, learning_rate_val, l2, epsilon);

    const int64_t weight_size = model.size();
    for (int i = 0; i < weight_size; i += kMaxInputCount) {
      size_t size = (i + kMaxInputCount) < weight_size ? kMaxInputCount : weight_size - i;
      TensorTuple input(4 * size);
      std::copy(model.begin() + i, model.begin() + i + size, input.begin());
      std::copy(model_diff.begin() + i, model_diff.begin() + i + size, input.begin() + size);
      std::copy(m.begin() + i, m.begin() + i + size, input.begin() + 2 * size);
      std::copy(v.begin() + i, v.begin() + i + size, input.begin() + 3 * size);
      JUST(OpInterpUtil::Dispatch<TensorTuple>(*op_[size - 1], input, attrs));
    }
    return Maybe<void>::Ok();
  }

 private:
  std::vector<std::shared_ptr<OpExpr>> op_;
};

class MatrixVectorProductFunctor {
 public:
  MatrixVectorProductFunctor() {
    matrix_vector_product_op_ = CHECK_JUST(
        one::OpBuilder("matrix_vector_product").Input("a").Input("b").Output("out").Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& vec) const {
    const auto& input_shape = input->shape();
    const auto& vec_shape = vec->shape();
    CHECK_OR_RETURN(input_shape->NumAxes() == 2 && vec_shape->NumAxes() == 1)
        << Error::RuntimeError() << "vector + matrix @ vector expected, got "
        << "1, " << input_shape->NumAxes() << ", " << vec_shape->NumAxes();
    CHECK_EQ_OR_RETURN(input_shape->at(1), vec_shape->at(0))
        << Error::RuntimeError() << "size mismatch, got " << std::to_string(input_shape->at(0))
        << ", " << std::to_string(input_shape->at(0)) << "x" << std::to_string(input_shape->at(1))
        << ", " << std::to_string(vec_shape->at(0));
    return OpInterpUtil::Dispatch<Tensor>(*matrix_vector_product_op_, {input, vec});
  }

 private:
  std::shared_ptr<OpExpr> matrix_vector_product_op_;
};

class BatchNormStatsFunctor {
 public:
  BatchNormStatsFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("batch_norm_stats").Input("input").Output("mean").Output("invstd").Build());
  }

  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& input, const int& axis,
                                const float& eps) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("axis", "eps");
    attrs.SetAllAttrs(axis, eps);
    return OpInterpUtil::Dispatch<one::TensorTuple>(*op_, {input}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class BatchNormGatherStatsWithCountsFunctor {
 public:
  BatchNormGatherStatsWithCountsFunctor() {
    op_with_running_mean_and_var_ = CHECK_JUST(one::OpBuilder("batch_norm_gather_stats_with_counts")
                                                   .Input("input")
                                                   .Input("mean")
                                                   .Input("invstd")
                                                   .Input("counts")
                                                   .Input("running_mean")
                                                   .Input("running_var")
                                                   .Output("global_mean")
                                                   .Output("global_invstd")
                                                   .Build());
    op_without_running_mean_and_var_ =
        CHECK_JUST(one::OpBuilder("batch_norm_gather_stats_with_counts")
                       .Input("input")
                       .Input("mean")
                       .Input("invstd")
                       .Input("counts")
                       .Output("global_mean")
                       .Output("global_invstd")
                       .Build());
  }

  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& input,
                                const std::shared_ptr<one::Tensor>& mean,
                                const std::shared_ptr<one::Tensor>& invstd,
                                const Optional<one::Tensor>& running_mean,
                                const Optional<one::Tensor>& running_var, const float& momentum,
                                const float& eps,
                                const std::shared_ptr<one::Tensor>& counts) const {
    CHECK_OR_RETURN((running_mean && running_var) || (!running_mean && !running_var))
        << Error::RuntimeError()
        << "Both running_mean and running_var should be None or Tensor at the same time.";

    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("eps", "momentum");
    attrs.SetAllAttrs(eps, momentum);

    if (running_mean) {
      return OpInterpUtil::Dispatch<one::TensorTuple>(
          *op_with_running_mean_and_var_,
          {input, mean, invstd, counts, JUST(running_mean), JUST(running_var)}, attrs);
    }
    return OpInterpUtil::Dispatch<one::TensorTuple>(*op_without_running_mean_and_var_,
                                                    {input, mean, invstd, counts}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_with_running_mean_and_var_;
  std::shared_ptr<OpExpr> op_without_running_mean_and_var_;
};

class BatchNormElemtFunctor {
 public:
  BatchNormElemtFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("batch_norm_elemt")
                         .Input("input")
                         .Input("weight")
                         .Input("bias")
                         .Input("mean")
                         .Input("invstd")
                         .Output("output")
                         .Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& weight,
                           const std::shared_ptr<one::Tensor>& bias,
                           const std::shared_ptr<one::Tensor>& mean,
                           const std::shared_ptr<one::Tensor>& invstd, const int& axis,
                           const float& eps) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("axis", "eps");
    attrs.SetAllAttrs(axis, eps);
    return OpInterpUtil::Dispatch<one::Tensor>(*op_, {input, weight, bias, mean, invstd}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class BatchNormBackwardReduceFunctor {
 public:
  BatchNormBackwardReduceFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("batch_norm_backward_reduce")
                         .Input("grad_out")
                         .Input("input")
                         .Input("mean")
                         .Input("invstd")
                         .Output("sum_dy")
                         .Output("sum_dy_xmu")
                         .Output("grad_weight")
                         .Output("grad_bias")
                         .Build());
  }

  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& grad_out,
                                const std::shared_ptr<one::Tensor>& input,
                                const std::shared_ptr<one::Tensor>& mean,
                                const std::shared_ptr<one::Tensor>& invstd, const int& axis) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("axis");
    attrs.SetAllAttrs(axis);
    return OpInterpUtil::Dispatch<one::TensorTuple>(*op_, {grad_out, input, mean, invstd}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class BatchNormBackwardElemtFunctor {
 public:
  BatchNormBackwardElemtFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("batch_norm_backward_elemt")
                         .Input("grad_out")
                         .Input("input")
                         .Input("mean")
                         .Input("invstd")
                         .Input("weight")
                         .Input("sum_dy")
                         .Input("sum_dy_xmu")
                         .Input("count")
                         .Output("grad_in")
                         .Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& grad_out,
                           const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& mean,
                           const std::shared_ptr<one::Tensor>& invstd,
                           const std::shared_ptr<one::Tensor>& weight,
                           const std::shared_ptr<one::Tensor>& sum_dy,
                           const std::shared_ptr<one::Tensor>& sum_dy_xmu,
                           const std::shared_ptr<one::Tensor>& count, const int& axis) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("axis");
    attrs.SetAllAttrs(axis);
    return OpInterpUtil::Dispatch<one::Tensor>(
        *op_, {grad_out, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class FusedFastGeluMulFunctor {
 public:
  FusedFastGeluMulFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("fused_fast_gelu_mul")
                         .Input("in")
                         .Input("multiplier")
                         .Output("out")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& multiplier) const {
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x, multiplier});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class FusedFastGeluMulGradFunctor {
 public:
  FusedFastGeluMulGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("fused_fast_gelu_mul_grad")
                         .Input("out_diff")
                         .Input("in")
                         .Input("multiplier")
                         .Output("in_diff")
                         .Output("multiplier_diff")
                         .Build());
  }
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& dy,
                                const std::shared_ptr<one::Tensor>& x,
                                const std::shared_ptr<one::Tensor>& multiplier) const {
    return OpInterpUtil::Dispatch<TensorTuple>(*op_, {dy, x, multiplier});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class GroupedMatmulBiasFunctor {
 public:
  GroupedMatmulBiasFunctor() {
    fused_op_.resize(kMaxInputCount /*the maximum number of inputs*/);
    for (int n = 1; n < fused_op_.size(); ++n) {
      fused_op_[n] = CHECK_JUST(one::OpBuilder("grouped_matmul_bias")
                                    .Input("xs", n)
                                    .Input("weights", n)
                                    .Input("biases", n)
                                    .Output("ys", n)
                                    .Build());
    }
  }
  Maybe<TensorTuple> operator()(const TensorTuple& xs, const TensorTuple& weights,
                                const TensorTuple& biases) const {
    const int64_t input_size = xs.size();
    const int64_t weight_size = weights.size();
    const int64_t bias_size = biases.size();
    CHECK_GE_OR_RETURN(input_size, 1)
        << Error::RuntimeError() << "The number of xs should be greater equal than 1.";
    CHECK_EQ_OR_RETURN(weight_size, input_size)
        << Error::RuntimeError() << "The number of weights should be equal to xs.";
    CHECK_EQ_OR_RETURN(bias_size, input_size)
        << Error::RuntimeError() << "The number of bias should be equal to xs.";
    for (int64_t i = 0; i < input_size; ++i) {
      const auto& input_shape = xs[i]->shape();
      const auto& weight_shape = weights[i]->shape();
      const auto& bias_shape = biases[i]->shape();
      CHECK_GE_OR_RETURN(input_shape->NumAxes(), 2)
          << Error::RuntimeError() << "x's dim size should greater equal than 2.";
      CHECK_EQ_OR_RETURN(weight_shape->NumAxes(), 2)
          << Error::RuntimeError() << "Weight's dim size should == 2";
      CHECK_EQ_OR_RETURN(bias_shape->NumAxes(), 1)
          << Error::RuntimeError() << "Bias's dim size should == 1";
      const int64_t k = input_shape->At(input_shape->NumAxes() - 1);
      CHECK_EQ_OR_RETURN(weight_shape->At(1), k)
          << Error::RuntimeError() << "weight's second dim should be equal to input's last dim. ";
      const int64_t n = weight_shape->At(0);
      CHECK_EQ_OR_RETURN(bias_shape->At(0), n)
          << Error::RuntimeError() << "Bias's dim is not equal to weight's first dim. ";
    }
    TensorTuple input(3 * input_size);
    std::copy(xs.begin(), xs.end(), input.begin() + 0 * input_size);
    std::copy(weights.begin(), weights.end(), input.begin() + 1 * input_size);
    std::copy(biases.begin(), biases.end(), input.begin() + 2 * input_size);
    return OpInterpUtil::Dispatch<TensorTuple>(*fused_op_[input_size], input);
  }

 private:
  std::vector<std::shared_ptr<OpExpr>> fused_op_;
};

class GroupedMatmulFunctor {
 public:
  GroupedMatmulFunctor() {
    fused_op_.resize(kMaxInputCount /*the maximum number of inputs*/);
    for (int n = 1; n < fused_op_.size(); ++n) {
      fused_op_[n] = CHECK_JUST(one::OpBuilder("grouped_matmul_bias")
                                    .Input("xs", n)
                                    .Input("weights", n)
                                    .Output("ys", n)
                                    .Build());
    }
  }
  Maybe<TensorTuple> operator()(const TensorTuple& xs, const TensorTuple& weights) const {
    const int64_t input_size = xs.size();
    const int64_t weight_size = weights.size();
    CHECK_GE_OR_RETURN(input_size, 1)
        << Error::RuntimeError() << "The number of xs should be greater equal than 1.";
    CHECK_EQ_OR_RETURN(weight_size, input_size)
        << Error::RuntimeError() << "The number of weights should be equal to xs.";
    for (int64_t i = 0; i < input_size; ++i) {
      const auto& input_shape = xs[i]->shape();
      const auto& weight_shape = weights[i]->shape();
      CHECK_GE_OR_RETURN(input_shape->NumAxes(), 2)
          << Error::RuntimeError() << "x's dim size should greater equal than 2.";
      CHECK_EQ_OR_RETURN(weight_shape->NumAxes(), 2)
          << Error::RuntimeError() << "Weight's dim size should == 2";
      const int64_t k = input_shape->At(input_shape->NumAxes() - 1);
      CHECK_EQ_OR_RETURN(weight_shape->At(1), k)
          << Error::RuntimeError() << "weight's second dim should be equal to input's last dim. ";
    }
    TensorTuple input(2 * input_size);
    std::copy(xs.begin(), xs.end(), input.begin() + 0 * input_size);
    std::copy(weights.begin(), weights.end(), input.begin() + 1 * input_size);
    return OpInterpUtil::Dispatch<TensorTuple>(*fused_op_[input_size], input);
  }

 private:
  std::vector<std::shared_ptr<OpExpr>> fused_op_;
};

class MultiTensorYoloV5WeightUpdateFunctor {
 public:
  MultiTensorYoloV5WeightUpdateFunctor() {
    op_.resize(kMaxInputCount /*the maximum number of inputs*/);
    for (int n = 0; n < op_.size(); ++n) {
      op_[n] = CHECK_JUST(one::OpBuilder("multi_tensor_yolov5_weight_update")
                              .Input("model", n + 1)
                              .Input("model_update", n + 1)
                              .Build());
    }
  }

  Maybe<void> operator()(const TensorTuple& model, const TensorTuple& model_update,
                         const float& d) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("d");
    attrs.SetAllAttrs(d);
    const int64_t weight_size = model.size();
    for (int i = 0; i < weight_size; i += kMaxInputCount) {
      size_t size = (i + kMaxInputCount) < weight_size ? kMaxInputCount : weight_size - i;
      TensorTuple input(size * 2);
      std::copy(model.begin() + i, model.begin() + i + size, input.begin());
      std::copy(model_update.begin() + i, model_update.begin() + i + size,
                input.begin() + 1 * size);
      JUST(OpInterpUtil::Dispatch<TensorTuple>(*op_[size - 1], input, attrs));
    }
    return Maybe<void>::Ok();
  }

 private:
  std::vector<std::shared_ptr<OpExpr>> op_;
};

class FusedScaleMaskBiasSoftmaxFunctor {
 public:
  FusedScaleMaskBiasSoftmaxFunctor() {
    op_with_bias_ = CHECK_JUST(one::OpBuilder("fused_scale_mask_bias_softmax")
                                   .Input("x")
                                   .Input("mask")
                                   .Input("bias")
                                   .Output("out")
                                   .Build());
    op_without_bias_ = CHECK_JUST(one::OpBuilder("fused_scale_mask_bias_softmax")
                                      .Input("x")
                                      .Input("mask")
                                      .Output("out")
                                      .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& mask,
                           const Optional<one::Tensor>& bias, const float& scale,
                           const bool& inplace = false) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("scale", "inplace");
    attrs.SetAllAttrs(scale, inplace);
    if (bias) {
      if (inplace) {
        std::shared_ptr<TensorTuple> outputs = std::make_shared<TensorTuple>(1);
        outputs->at(0) = x;
        JUST(OpInterpUtil::Dispatch(*op_with_bias_, {x, mask, JUST(bias)}, outputs.get(), attrs));
        return outputs->at(0);
      }
      return OpInterpUtil::Dispatch<Tensor>(*op_with_bias_, {x, mask, JUST(bias)}, attrs);
      ;
    }
    if (inplace) {
      std::shared_ptr<TensorTuple> outputs = std::make_shared<TensorTuple>(1);
      outputs->at(0) = x;
      JUST(OpInterpUtil::Dispatch(*op_without_bias_, {x, mask}, outputs.get(), attrs));
      return outputs->at(0);
    }
    return OpInterpUtil::Dispatch<Tensor>(*op_without_bias_, {x, mask}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_without_bias_;
  std::shared_ptr<OpExpr> op_with_bias_;
};

class FusedScaleMaskBiasSoftmaxGradFunctor {
 public:
  FusedScaleMaskBiasSoftmaxGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("fused_scale_mask_bias_softmax_grad")
                         .Input("y")
                         .Input("dy")
                         .Output("dx")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& y,
                           const std::shared_ptr<one::Tensor>& dy, const float& scale) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("scale");
    attrs.SetAllAttrs(scale);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {y, dy}, attrs);
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
  m.add_functor<impl::EmbeddingReNormFunctor>("EmbeddingReNorm");
  m.add_functor<impl::EmbeddingFunctor>("Embedding");
  m.add_functor<impl::MatMulFunctor>("MatMul");
  m.add_functor<impl::MatMulNoBroadCastFunctor>("MatMulNoBroadCast");
  m.add_functor<impl::BatchMatMulFunctor>("BatchMatMul");
  m.add_functor<impl::MatrixVectorProductFunctor>("MatrixVectorProduct");
  m.add_functor<impl::VectorMatrixProductFunctor>("VectorMatrixProduct");
  m.add_functor<impl::TensorDotFunctor>("TensorDot");
  m.add_functor<impl::TensorDotIntDimsFunctor>("TensorDotIntDims");
  m.add_functor<impl::FusedMLPFunctor>("FusedMLP");
  m.add_functor<impl::FusedMatmulBiasFunctor>("FusedMatmulBias");
  m.add_functor<impl::FusedMatmulBiasAddReluDropoutFunctor>("FusedMatmulBiasAddReluDropout");
  m.add_functor<impl::LayerNormFunctor>("LayerNorm");
  m.add_functor<impl::SkipLayerNormFunctor>("SkipLayerNorm");
  m.add_functor<impl::LayerNormAffineFunctor>("LayerNormAffine");
  m.add_functor<impl::GroupNormFunctor>("GroupNorm");
  m.add_functor<impl::TFAvgPool2DFunctor>("TFAvgPool2D");
  m.add_functor<impl::MaxPool1DFunctor>("MaxPool1D");
  m.add_functor<impl::MaxPool2DFunctor>("MaxPool2D");
  m.add_functor<impl::MaxPool3DFunctor>("MaxPool3D");
  m.add_functor<impl::MaxUnpoolNDFunctor<1>>("MaxUnpool1D");
  m.add_functor<impl::MaxUnpoolNDFunctor<2>>("MaxUnpool2D");
  m.add_functor<impl::MaxUnpoolNDFunctor<3>>("MaxUnpool3D");
  m.add_functor<impl::AdaptiveAvgPool1DFunctor>("AdaptiveAvgPool1D");
  m.add_functor<impl::AdaptiveAvgPool2DFunctor>("AdaptiveAvgPool2D");
  m.add_functor<impl::AdaptiveAvgPool3DFunctor>("AdaptiveAvgPool3D");
  m.add_functor<impl::AdaptiveMaxPool1DFunctor>("AdaptiveMaxPool1D");
  m.add_functor<impl::AdaptiveMaxPool2DFunctor>("AdaptiveMaxPool2D");
  m.add_functor<impl::AdaptiveMaxPool3DFunctor>("AdaptiveMaxPool3D");
  m.add_functor<impl::L1LossFunctor>("L1Loss");
  m.add_functor<impl::MseLossFunctor>("MseLoss");
  m.add_functor<impl::KLDivLossFunctor>("KLDivLoss");
  m.add_functor<impl::NLLLossFunctor>("NLLLoss");
  m.add_functor<impl::BinaryCrossEntropyLossFunctor>("BinaryCrossEntropyLoss");
  m.add_functor<impl::BinaryCrossEntropyWithLogitsLossFunctor>("BinaryCrossEntropyWithLogitsLoss");
  m.add_functor<impl::SparseCrossEntropyFunctor>("SparseCrossEntropy");
  m.add_functor<impl::SparseCrossEntropyMsFunctor>("SparseCrossEntropyMs");
  m.add_functor<impl::CrossEntropyFunctor>("CrossEntropy");
  m.add_functor<impl::CrossEntropyLabelSmoothingFunctor>("CrossEntropyLabelSmoothing");
  m.add_functor<impl::CrossEntropyProbFunctor>("CrossEntropyProb");
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
  m.add_functor<impl::ConstantPadFunctor>("ConstantPad");
  m.add_functor<impl::ReflectionPadFunctor>("ReflectionPad");
  m.add_functor<impl::ReplicationPadFunctor>("ReplicationPad");
  m.add_functor<impl::PadFunctor>("Pad");
  m.add_functor<impl::DropoutFunctor>("Dropout");
  m.add_functor<impl::DropoutGradFunctor>("DropoutGrad");
  m.add_functor<impl::Dropout1dFunctor>("Dropout1d");
  m.add_functor<impl::Dropout2dFunctor>("Dropout2d");
  m.add_functor<impl::Dropout3dFunctor>("Dropout3d");
  m.add_functor<impl::PixelShuffleFunctor>("PixelShuffle");
  m.add_functor<impl::AvgPool1DFunctor>("AvgPool1D");
  m.add_functor<impl::AvgPool2DFunctor>("AvgPool2D");
  m.add_functor<impl::AvgPool3DFunctor>("AvgPool3D");
  m.add_functor<impl::UnfoldFunctor>("Unfold");
  m.add_functor<impl::FoldFunctor>("Fold");
  m.add_functor<impl::OneHotFunctor>("OneHot");
  m.add_functor<impl::FusedSelfAttentionFunctor>("FusedSelfAttention");
  m.add_functor<impl::FusedSelfAttentionGradFunctor>("FusedSelfAttentionGrad");
  m.add_functor<impl::PairwiseDistanceFunctor>("PairwiseDistance");
  m.add_functor<impl::CosineSimilarityFunctor>("CosineSimilarity");
  m.add_functor<impl::NormalizeFunctor>("Normalize");
  m.add_functor<impl::L2NormalizeFunctor>("L2Normalize");
  m.add_functor<impl::L2NormalizeGradFunctor>("L2NormalizeGrad");
  m.add_functor<impl::FusedBiasAddGeluFunctor>("FusedBiasAddGelu");
  m.add_functor<impl::FusedBiasAddGeluGradFunctor>("FusedBiasAddGeluGrad");
  m.add_functor<impl::FusedGluFunctor>("FusedGlu");
  m.add_functor<impl::FusedBiasAddDropoutFunctor>("FusedBiasAddDropout");
  m.add_functor<impl::FusedScaleMaskSoftmaxFunctor>("FusedScaleMaskSoftmax");
  m.add_functor<impl::FusedScaleMaskSoftmaxDropoutFunctor>("FusedScaleMaskSoftmaxDropout");
  m.add_functor<impl::FusedBiasAddScaleMaskSoftmaxDropoutFunctor>(
      "FusedBiasAddScaleMaskSoftmaxDropout");
  m.add_functor<impl::FusedScaleTrilSoftmaxMaskScaleFunctor>("FusedScaleTrilSoftmaxMaskScale");
  m.add_functor<impl::FusedScaleTrilFunctor>("FusedScaleTril");
  m.add_functor<impl::CtcGreedyDecoderFunctor>("CtcGreedyDecoder");
  m.add_functor<impl::PariticalFCSampleDisableBoxing>("DistributedPariticalFCSampleDisableBoxing");
  m.add_functor<impl::NmsFunctor>("Nms");
  m.add_functor<impl::RoiAlignFunctor>("RoiAlign");
  m.add_functor<impl::RoiAlignGradFunctor>("RoiAlignGrad");
  m.add_functor<impl::FusedDotFeatureInteractionFunctor>("FusedDotFeatureInteraction");
  m.add_functor<impl::FusedCrossFeatureInteractionFunctor>("FusedCrossFeatureInteraction");
  m.add_functor<impl::OneEmbeddingIdShuffleFunctor>("OneEmbeddingIdShuffle");
  m.add_functor<impl::OneEmbeddingEmbeddingShuffleFunctor>("OneEmbeddingEmbeddingShuffle");
  m.add_functor<impl::OneEmbeddingEmbeddingGradientShuffleFunctor>(
      "OneEmbeddingEmbeddingGradientShuffle");
  m.add_functor<impl::OneEmbeddingLookupFunctor>("OneEmbeddingLookup");
  m.add_functor<impl::OneEmbeddingFusedLookupFunctor>("OneEmbeddingFusedLookup");
  m.add_functor<impl::OneEmbeddingFusedLookupGradFunctor>("OneEmbeddingFusedLookupGrad");
  m.add_functor<impl::OneEmbeddingEmbeddingPutFunctor>("OneEmbeddingEmbeddingPut");
  m.add_functor<impl::OneEmbeddingUniqueKeyValuePairFunctor>("OneEmbeddingUniqueKeyValuePair");
  m.add_functor<impl::OneEmbeddingSgdUpdateFunctor>("OneEmbeddingSgdUpdate");
  m.add_functor<impl::OneEmbeddingAdamUpdateFunctor>("OneEmbeddingAdamUpdate");
  m.add_functor<impl::OneEmbeddingAdagradUpdateFunctor>("OneEmbeddingAdagradUpdate");
  m.add_functor<impl::OneEmbeddingFtrlUpdateFunctor>("OneEmbeddingFtrlUpdate");
  m.add_functor<impl::RocAucScoreFunctor>("RocAucScore");
  m.add_functor<impl::MultiTensorSgdUpdateFunctor>("MultiTensorSgdUpdate");
  m.add_functor<impl::MultiTensorMomentumUpdateFunctor>("MultiTensorMomentumUpdate");
  m.add_functor<impl::MultiTensorAdamUpdateFunctor>("MultiTensorAdamUpdate");
  m.add_functor<impl::DeformConv2dFunctor>("DeformConv2d");
  m.add_functor<impl::BatchNormStatsFunctor>("BatchNormStats");
  m.add_functor<impl::BatchNormGatherStatsWithCountsFunctor>("BatchNormGatherStatsWithCounts");
  m.add_functor<impl::BatchNormElemtFunctor>("BatchNormElemt");
  m.add_functor<impl::BatchNormBackwardReduceFunctor>("BatchNormBackwardReduce");
  m.add_functor<impl::BatchNormBackwardElemtFunctor>("BatchNormBackwardElemt");
  m.add_functor<impl::FusedFastGeluMulFunctor>("FusedFastGeluMul");
  m.add_functor<impl::FusedFastGeluMulGradFunctor>("FusedFastGeluMulGrad");
  m.add_functor<impl::GroupedMatmulBiasFunctor>("GroupedMatmulBias");
  m.add_functor<impl::GroupedMatmulFunctor>("GroupedMatmul");
  m.add_functor<impl::RMSNormFunctor>("RMSNorm");
  m.add_functor<impl::SkipRMSNormFunctor>("SkipRMSNorm");
  m.add_functor<impl::FusedScaleMaskBiasSoftmaxFunctor>("FusedScaleMaskBiasSoftmax");
  m.add_functor<impl::FusedScaleMaskBiasSoftmaxGradFunctor>("FusedScaleMaskBiasSoftmaxGrad");
  m.add_functor<impl::MultiTensorYoloV5WeightUpdateFunctor>("MultiTensorYoloV5WeightUpdate");
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow
