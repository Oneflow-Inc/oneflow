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

#include "oneflow/core/common/cached_functor_ptr.h"
#include "oneflow/core/common/container_util.h"
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
  struct BiasAdd {
    Maybe<AttrMap> operator()(int32_t axis_val) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<int32_t>("axis", axis_val));
      return AttrMap(attrs);
    }
  };
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
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(BiasAdd);
    const auto attrs = *JUST(GetAttrs(axis_val));
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

  struct ConvBase {
    Maybe<AttrMap> operator()(int num_spatial_dims, const std::vector<int32_t>& stride,
                              const std::vector<int32_t>& padding,
                              const std::vector<int32_t>& dilation, const int32_t& groups,
                              const std::string& channel_pos, const Shape& weight_shape) {
      std::vector<int32_t> kernel_size_vec(num_spatial_dims);
      int32_t channel_idx = 1;
      int32_t kernel_idx_offset = 2;
      if (channel_pos == "channels_last") {
        kernel_idx_offset = 1;
        channel_idx = kernel_idx_offset + num_spatial_dims;
      }

      for (int i = 0; i < num_spatial_dims; i++) {
        kernel_size_vec.at(i) = (weight_shape.At(i + kernel_idx_offset));
      }
      MutableAttrMap conv_attrs;
      JUST(conv_attrs.SetAttr<int32_t>("filters", weight_shape.At(0)));
      JUST(conv_attrs.SetAttr<std::vector<int32_t>>("padding_before", padding));
      JUST(conv_attrs.SetAttr<std::vector<int32_t>>("kernel_size", kernel_size_vec));
      JUST(conv_attrs.SetAttr<std::vector<int32_t>>("strides", stride));
      JUST(conv_attrs.SetAttr<std::vector<int32_t>>("dilation_rate", dilation));
      JUST(conv_attrs.SetAttr<int32_t>("groups", groups));
      JUST(conv_attrs.SetAttr<std::string>("data_format", channel_pos));
      JUST(conv_attrs.SetAttr<int32_t>("channel_idx", channel_idx));
      return AttrMap(conv_attrs);
    }
  };
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& weight,
                           const Optional<one::Tensor>& bias, const std::vector<int32_t>& stride,
                           const std::vector<int32_t>& padding,
                           const std::vector<int32_t>& dilation, const int32_t& groups,
                           const std::string& channel_pos) const {
    constexpr static auto* GetAttrs = CACHED_FUNCTOR_PTR(ConvBase);
    const auto& conv_attrs = JUST(GetAttrs(num_spatial_dims_, stride, padding, dilation, groups,
                                           channel_pos, *weight->shape()));
    const std::shared_ptr<one::Tensor>& conv_out =
        JUST(OpInterpUtil::Dispatch<Tensor>(*conv_op_, {x, weight}, *conv_attrs));
    if (bias) {
      int32_t channel_idx = JUST(conv_attrs->GetAttr<int32_t>("channel_idx"));
      return functional::BiasAdd(conv_out, JUST(bias), channel_idx);
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
  struct DeConvBase {
    Maybe<AttrMap> operator()(int num_spatial_dims, const Shape weight_shape,
                              const std::vector<int32_t>& stride,
                              const std::vector<int32_t>& padding,
                              const std::vector<int32_t>& output_padding, const int32_t& groups,
                              const std::vector<int32_t>& dilation,
                              const std::string& data_format) {
      std::vector<int32_t> kernel_size_vec(num_spatial_dims);
      int32_t kernel_idx_offset = 2;
      if (data_format == "channels_last") { kernel_idx_offset = 1; }
      for (int i = 0; i < num_spatial_dims; i++) {
        kernel_size_vec[i] = (weight_shape.At(i + kernel_idx_offset));
      }

      MutableAttrMap deconv_attrs;
      JUST(deconv_attrs.SetAttr<int32_t>("filters", weight_shape.At(1) * groups));
      JUST(deconv_attrs.SetAttr<std::vector<int32_t>>("padding_before", padding));
      JUST(deconv_attrs.SetAttr<std::vector<int32_t>>("kernel_size", kernel_size_vec));
      JUST(deconv_attrs.SetAttr<std::vector<int32_t>>("output_padding", output_padding));
      JUST(deconv_attrs.SetAttr<std::vector<int32_t>>("strides", stride));
      JUST(deconv_attrs.SetAttr<std::vector<int32_t>>("dilation_rate", dilation));
      JUST(deconv_attrs.SetAttr<int32_t>("groups", groups));
      JUST(deconv_attrs.SetAttr<std::string>("data_format", data_format));
      return AttrMap(deconv_attrs);
    }
  };
  struct Bias {
    Maybe<AttrMap> operator()() {
      MutableAttrMap bias_attrs;
      JUST(bias_attrs.SetAttr<int32_t>("axis", 1));
      return AttrMap(bias_attrs);
    }
  };
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& weight,
                           const Optional<one::Tensor>& bias, const std::vector<int32_t>& stride,
                           const std::vector<int32_t>& padding,
                           const std::vector<int32_t>& output_padding, const int32_t& groups,
                           const std::vector<int32_t>& dilation,
                           const std::string& data_format) const {
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(DeConvBase);
    const auto deconv_attrs = *JUST(GetAttrs(num_spatial_dims_, *weight->shape(), stride, padding,
                                             output_padding, groups, dilation, data_format));
    std::shared_ptr<one::Tensor> deconv_out = nullptr;
    deconv_out = JUST(OpInterpUtil::Dispatch<Tensor>(*deconv_op_, {x, weight}, deconv_attrs));
    if (bias) {
      constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(Bias);
      const auto bias_attrs = *JUST(GetAttrs());
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

class EmbeddingReNormFunctor {
 public:
  EmbeddingReNormFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("embedding_renorm").Input("in").Input("indices").Output("out").Build());
  }

  struct EmbeddingReNorm {
    Maybe<AttrMap> operator()(const double& max_norm, const double& norm_type) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<double>("max_norm", max_norm));
      JUST(attrs.SetAttr<double>("norm_type", norm_type));
      return AttrMap(attrs);
    }
  };

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& in,
                           const std::shared_ptr<one::Tensor>& indices, const double& max_norm,
                           const double& norm_type) const {
    CHECK_EQ_OR_RETURN(in->ndim(), 2)
        << Error::RuntimeError() << "The dimension of input should be 2.";
    std::shared_ptr<TensorTuple> outputs = std::make_shared<TensorTuple>(1);
    JUST(oneflow::VectorAt(*outputs, 0)) = in;

    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(EmbeddingReNorm);
    const auto attrs = *JUST(GetAttrs(max_norm, norm_type));
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
  struct Embedding {
    Maybe<AttrMap> operator()(int64_t new_padding_idx, bool scale_grad_by_freq) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<int64_t>("padding_idx", new_padding_idx));
      JUST(attrs.SetAttr<bool>("scale_grad_by_freq", scale_grad_by_freq));
      return AttrMap(attrs);
    }
  };
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& weight,
                           const std::shared_ptr<one::Tensor>& indices,
                           const Optional<int64_t>& padding_idx,
                           const bool& scale_grad_by_freq) const {
    CHECK_EQ_OR_RETURN(weight->ndim(), 2) << "The dimension of weight should be 2";
    int64_t new_padding_idx = -1;
    if (padding_idx.has_value()) { new_padding_idx = JUST(padding_idx); }
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(Embedding);
    const auto attrs = *JUST(GetAttrs(new_padding_idx, scale_grad_by_freq));
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
  struct MatMul {
    Maybe<AttrMap> operator()(const bool& transpose_a, const bool& transpose_b,
                              const double& alpha) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<bool>("transpose_a", transpose_a));
      JUST(attrs.SetAttr<bool>("transpose_b", transpose_b));
      JUST(attrs.SetAttr<double>("alpha", alpha));
      return AttrMap(attrs);
    }
  };
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& a,
                           const std::shared_ptr<one::Tensor>& b, const bool& transpose_a,
                           const bool& transpose_b, const double& alpha) const {
    const auto& a_shape = a->shape();
    const auto& b_shape = b->shape();

    CHECK_GE_OR_RETURN(a_shape->NumAxes(), 1)
        << Error::RuntimeError() << "Tensor a's dim should >= 1";
    CHECK_GE_OR_RETURN(b_shape->NumAxes(), 1)
        << Error::RuntimeError() << "Tensor b's dim should >= 1";
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(MatMul);
    const auto attrs = *JUST(GetAttrs(transpose_a, transpose_b, alpha));
    const int64_t a_num_axes = a_shape->NumAxes();
    const int64_t b_num_axes = b_shape->NumAxes();
    if (a_num_axes == 1 && b_num_axes == 2) { return VectorMatrixProduct(a, b); }
    if (a_num_axes == 2 && b_num_axes == 1) { return MatrixVectorProduct(a, b); }
    if (a_num_axes == 2 && b_num_axes == 2) {
      return OpInterpUtil::Dispatch<Tensor>(*matmul_op_, {a, b}, attrs);
    }
    if (a_num_axes == b_num_axes) {
      bool if_batch_matmul = true;
      for (int i = 0; i < a_num_axes - 2; ++i) {
        if (a_shape->At(i) != b_shape->At(i)) {
          if_batch_matmul = false;
          break;
        }
      }
      if (if_batch_matmul) {
        return OpInterpUtil::Dispatch<Tensor>(*batch_matmul_op_, {a, b}, attrs);
      }
    }
    return OpInterpUtil::Dispatch<Tensor>(*bcast_matmul_op_, {a, b}, attrs);
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
  struct BatchMatMul {
    Maybe<AttrMap> operator()(const bool& transpose_a, const bool& transpose_b,
                              const double& alpha) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<bool>("transpose_a", transpose_a));
      JUST(attrs.SetAttr<bool>("transpose_b", transpose_b));
      JUST(attrs.SetAttr<double>("alpha", alpha));
      return AttrMap(attrs);
    }
  };
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
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(BatchMatMul);
    const auto attrs = *JUST(GetAttrs(transpose_a, transpose_b, alpha));
    return OpInterpUtil::Dispatch<Tensor>(*batch_matmul_op_, {a, b}, attrs);
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
      struct FusedAttr {
        Maybe<AttrMap> operator()(bool skip_final_activation) {
          MutableAttrMap attrs;
          JUST(attrs.SetAttr<bool>("skip_final_activation", skip_final_activation));
          return AttrMap(attrs);
        };
      };
      constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(FusedAttr);
      const auto attrs = *JUST(GetAttrs(skip_final_activation));
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
    const auto gen = generator.value_or(JUST(one::DefaultAutoGenerator()));
    const auto& dropout_state = std::make_shared<FusedDropoutKernelState>(gen);
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
      struct FuseAttr {
        Maybe<AttrMap> operator()(bool skip_final_activation,
                                  const std::vector<float>& dropout_rate_list) {
          MutableAttrMap attrs;
          JUST(attrs.SetAttr<bool>("skip_final_activation", skip_final_activation));
          JUST(attrs.SetAttr<std::vector<float>>("dropout_rate_list", dropout_rate_list));
          return AttrMap(attrs);
        }
      };
      constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(FuseAttr);
      const auto attrs = *JUST(GetAttrs(skip_final_activation, dropout_rate_list));
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
  struct LayerNorm {
    Maybe<AttrMap> operator()(const int64_t& begin_norm_axis, const int64_t& begin_params_axis,
                              const double& epsilon) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<int64_t>("begin_norm_axis", begin_norm_axis));
      JUST(attrs.SetAttr<int64_t>("begin_params_axis", begin_params_axis));
      JUST(attrs.SetAttr<double>("epsilon", epsilon));
      JUST(attrs.SetAttr<bool>("center", false));
      JUST(attrs.SetAttr<bool>("scale", false));
      return AttrMap(attrs);
    }
  };
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const int64_t& begin_norm_axis,
                           const int64_t& begin_params_axis, const double& epsilon) const {
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(LayerNorm);
    const auto attrs = *JUST(GetAttrs(begin_norm_axis, begin_params_axis, epsilon));
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

  struct LayerNormAffine {
    Maybe<AttrMap> operator()(const int64_t& begin_norm_axis, const int64_t& begin_params_axis,
                              const double& epsilon) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<int64_t>("begin_norm_axis", begin_norm_axis));
      JUST(attrs.SetAttr<int64_t>("begin_params_axis", begin_params_axis));
      JUST(attrs.SetAttr<double>("epsilon", epsilon));
      JUST(attrs.SetAttr<bool>("center", true));
      JUST(attrs.SetAttr<bool>("scale", true));
      return AttrMap(attrs);
    }
  };

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& gamma,
                           const std::shared_ptr<one::Tensor>& beta, const int64_t& begin_norm_axis,
                           const int64_t& begin_params_axis, const double& epsilon) const {
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(LayerNormAffine);
    const auto attrs = *JUST(GetAttrs(begin_norm_axis, begin_params_axis, epsilon));
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
  struct TFPoolND {
    Maybe<AttrMap> operator()(const std::vector<int32_t>& kernel_size,
                              const std::vector<int32_t>& strides, const std::string& padding,
                              const std::vector<int32_t>& padding_before,
                              const std::vector<int32_t>& padding_after,
                              const std::string& data_format, const bool& ceil_mode) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<std::vector<int32_t>>("pool_size", kernel_size));
      JUST(attrs.SetAttr<std::vector<int32_t>>("strides", strides));
      JUST(attrs.SetAttr<std::string>("padding", padding));
      JUST(attrs.SetAttr<std::vector<int32_t>>("padding_before", padding_before));
      JUST(attrs.SetAttr<std::vector<int32_t>>("padding_after", padding_after));
      JUST(attrs.SetAttr<std::string>("data_format", data_format));
      JUST(attrs.SetAttr<bool>("ceil_mode", ceil_mode));
      return AttrMap(attrs);
    }
  };
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::vector<int32_t>& kernel_size,
                           const std::vector<int32_t>& strides, const std::string& padding,
                           const std::vector<int32_t>& padding_before,
                           const std::vector<int32_t>& padding_after,
                           const std::string& data_format, const bool& ceil_mode) const {
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(TFPoolND);
    const auto attrs = *JUST(GetAttrs(kernel_size, strides, padding, padding_before, padding_after,
                                      data_format, ceil_mode));
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
        struct TfMaxPooling {
          Maybe<AttrMap> operator()(const std::vector<int32_t>& kernel_size,
                                    const std::vector<int32_t>& strides,
                                    const std::vector<int32_t>& padding_before,
                                    const std::vector<int32_t>& padding_after,
                                    const std::string& data_format, bool ceil_mode) {
            MutableAttrMap attrs;
            JUST(attrs.SetAttr<std::vector<int32_t>>("pool_size", kernel_size));
            JUST(attrs.SetAttr<std::vector<int32_t>>("strides", strides));
            JUST(attrs.SetAttr<std::string>("padding", "customized"));
            JUST(attrs.SetAttr<std::vector<int32_t>>("padding_before", padding_before));
            JUST(attrs.SetAttr<std::vector<int32_t>>("padding_after", padding_after));
            JUST(attrs.SetAttr<std::string>("data_format", data_format));
            JUST(attrs.SetAttr<bool>("ceil_mode", ceil_mode));
            return AttrMap(attrs);
          }
        };
        constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(TfMaxPooling);
        const auto attrs =
            *JUST(GetAttrs(kernel_size, stride.has_value() ? *JUST(stride) : kernel_size,
                           padding_before, padding_after, data_format, ceil_mode));
        TensorTuple output;
        output.emplace_back(JUST(OpInterpUtil::Dispatch<Tensor>(*tf_maxpool_op_, {x}, attrs)));
        return output;
      }
    }

    struct MaxPoolND {
      Maybe<AttrMap> operator()(const std::string& data_format, const std::vector<int32_t>& padding,
                                const std::vector<int32_t>& kernel_size,
                                const std::vector<int32_t>& stride,
                                const std::vector<int32_t>& dilation, bool return_indices,
                                bool ceil_mode) {
        MutableAttrMap attrs;
        JUST(attrs.SetAttr<std::string>("data_format", data_format));
        JUST(attrs.SetAttr<std::vector<int32_t>>("padding", padding));
        JUST(attrs.SetAttr<std::vector<int32_t>>("kernel_size", kernel_size));
        JUST(attrs.SetAttr<std::vector<int32_t>>("stride", stride));
        JUST(attrs.SetAttr<std::vector<int32_t>>("dilation", dilation));
        JUST(attrs.SetAttr<bool>("return_indices", return_indices));
        JUST(attrs.SetAttr<bool>("ceil_mode", ceil_mode));
        return AttrMap(attrs);
      }
    };
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(MaxPoolND);
    // If stride is None, we set it as kernel_size to align Pytorch.
    const auto attrs = *JUST(GetAttrs(data_format, padding, kernel_size,
                                      (stride.has_value() ? *JUST(stride) : kernel_size), dilation,
                                      return_indices, ceil_mode));
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
  struct AdaptivePoolND {
    Maybe<AttrMap> operator()(const std::vector<int64_t>& output_size) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<std::vector<int64_t>>("output_size", output_size));
      return AttrMap(attrs);
    }
  };
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::vector<int64_t>& output_size) const {
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(AdaptivePoolND);
    const auto attrs = *JUST(GetAttrs(output_size));
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
  struct SmoothL1Loss {
    Maybe<AttrMap> operator()(float beta) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<float>("beta", beta));
      return AttrMap(attrs);
    }
  };
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& target, const float& beta,
                           const std::string& reduction) const {
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(SmoothL1Loss);
    const auto attrs = *JUST(GetAttrs(beta));
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
  struct KLDivLoss {
    Maybe<AttrMap> operator()(bool log_target) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<bool>("log_target", log_target));
      return AttrMap(attrs);
    }
  };
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& target, const bool log_target,
                           const std::string& reduction) const {
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(KLDivLoss);
    const auto attrs = *JUST(GetAttrs(log_target));
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
    AttrMap attrs{};
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
    op_reduce_mean_ = CHECK_JUST(one::OpBuilder("binary_cross_entropy_with_logits_reduce_mean")
                                     .Input("input")
                                     .Input("target")
                                     .Output("out")
                                     .Build());
  }
  struct BinaryCrossEntropyWithLogitsLoss {
    Maybe<AttrMap> operator()(bool has_pos_weight) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<bool>("has_pos_weight", has_pos_weight));
      return AttrMap(attrs);
    }
  };
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& target,
                           const Optional<one::Tensor>& weight,
                           const Optional<one::Tensor>& pos_weight,
                           const std::string& reduction) const {
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(BinaryCrossEntropyWithLogitsLoss);
    const auto attrs = *JUST(GetAttrs(pos_weight.has_value()));

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
    struct OpWeightAttrs {
      Maybe<AttrMap> operator()(int64_t ignore_index) {
        MutableAttrMap attrs;
        JUST(attrs.SetAttr<int64_t>("ignore_index", ignore_index));
        return AttrMap(attrs);
      }
    };
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(OpWeightAttrs);
    const auto attrs = *JUST(GetAttrs(ignore_index));

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
                           const std::string& reduction) const {
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

    struct OpNLLAttrs {
      Maybe<AttrMap> operator()(int64_t ignore_index) {
        MutableAttrMap attrs;
        JUST(attrs.SetAttr<int64_t>("ignore_index", ignore_index));
        return AttrMap(attrs);
      }
    };
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(OpNLLAttrs);
    const auto attrs = *JUST(GetAttrs(ignore_index));

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

class SparseCrossEntropyFunctor {
 public:
  SparseCrossEntropyFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("sparse_cross_entropy")
                         .Input("prediction")
                         .Input("label")
                         .Output("out")
                         .Build());
  }
  struct SparseCrossEntropy {
    Maybe<AttrMap> operator()(int64_t depth) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<int64_t>("depth", depth));
      return AttrMap(attrs);
    }
  };
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& prediction,
                           const std::shared_ptr<one::Tensor>& label, const int64_t& depth) const {
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(SparseCrossEntropy);
    const auto attrs = *JUST(GetAttrs(depth));
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
  struct SparseCrossEntropyMs {
    Maybe<AttrMap> operator()(int64_t depth) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<int64_t>("depth", depth));
      return AttrMap(attrs);
    }
  };
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& prediction,
                           const std::shared_ptr<one::Tensor>& label, const int64_t& depth) const {
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(SparseCrossEntropyMs);
    const auto attrs = *JUST(GetAttrs(depth));
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

  struct SparseSoftmaxCrossEntropy {
    Maybe<AttrMap> operator()(int64_t depth) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<int64_t>("depth", depth));
      return AttrMap(attrs);
    }
  };

  Maybe<Tensor> SparseSoftmaxCrossEntropyOperator(const std::shared_ptr<one::Tensor>& logits,
                                                  const std::shared_ptr<one::Tensor>& label) const {
    int64_t depth = logits->shape()->At(logits->shape()->NumAxes() - 1);
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(SparseSoftmaxCrossEntropy);
    const auto attrs = *JUST(GetAttrs(depth));
    const auto& result = JUST(OpInterpUtil::Dispatch<TensorTuple>(*op_sparse_softmax_cross_entropy_,
                                                                  {logits, label}, attrs));
    return result->at(1);
  }

  struct LazySparseSoftmaxCrossEntropyMs {
    Maybe<AttrMap> operator()(int64_t depth) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<int64_t>("depth", depth));
      return AttrMap(attrs);
    }
  };

  Maybe<Tensor> LazySparseSoftmaxCrossEntropyMsOperator(
      const std::shared_ptr<one::Tensor>& logits, const std::shared_ptr<one::Tensor>& label) const {
    int64_t depth = logits->shape()->At(logits->shape()->NumAxes() - 1);
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(LazySparseSoftmaxCrossEntropyMs);
    const auto attrs = *JUST(GetAttrs(depth));
    const auto& result = JUST(OpInterpUtil::Dispatch<TensorTuple>(
        *op_sparse_softmax_cross_entropy_ms_, {logits, label}, attrs));
    return result->at(1);
  }

  Maybe<Tensor> EagerSparseSoftmaxCrossEntropyMsOperator(
      const std::shared_ptr<one::Tensor>& logits, const std::shared_ptr<one::Tensor>& label) const {
    // op_reduce_max_device_stage_
    int64_t depth = logits->shape()->At(logits->shape()->NumAxes() - 1);
    int32_t axis = logits->shape()->NumAxes() - 1;
    struct ReduceMaxDeviceStage {
      Maybe<AttrMap> operator()(int32_t axis) {
        MutableAttrMap attrs;
        JUST(attrs.SetAttr<std::vector<int32_t>>("axis", {axis}));
        return AttrMap(attrs);
      }
    };
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(ReduceMaxDeviceStage);
    auto attrs = *JUST(GetAttrs(axis));
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
          s0s1_sbp_parallels, /* check_meta */ false));
      max_global_stage_input1 = JUST(functional::ToGlobal(
          (*max_device_stage)[2], JUST((*max_device_stage)[0]->parallel_desc()), new_sbp_parallels,
          s0s1_sbp_parallels, /* check_meta */ false));
    }
    // op_reduce_max_global_stage_
    struct ReduceMaxGlobal {
      Maybe<AttrMap> operator()(int32_t axis) {
        MutableAttrMap attrs{};
        JUST(attrs.SetAttr<std::vector<int32_t>>("axis", {axis}));
        JUST(attrs.SetAttr<bool>("keepdims", true));
        return AttrMap(attrs);
      }
    };
    constexpr auto* GetAttrsReduceMaxGlobal = CACHED_FUNCTOR_PTR(ReduceMaxGlobal);
    attrs = *JUST(GetAttrsReduceMaxGlobal(axis));

    const auto& max_global_stage = JUST(OpInterpUtil::Dispatch<TensorTuple>(
        *op_reduce_max_global_stage_, {max_global_stage_input0, max_global_stage_input1}, attrs));
    auto& broadcast_sub_input = max_global_stage->at(0);
    if (logits_nd_sbp.sbp_parallel_size() == 2) {
      broadcast_sub_input = JUST(
          functional::ToGlobal(broadcast_sub_input, JUST((*max_device_stage)[0]->parallel_desc()),
                               new_sbp_parallels, new_sbp_parallels, /* check_meta */ false));
    }
    // op_broadcast_sub_
    attrs = AttrMap{};
    const auto& output_broadcast_sub = JUST(OpInterpUtil::Dispatch<TensorTuple>(
        *op_broadcast_sub_, {logits, broadcast_sub_input}, attrs));
    // op_exp_
    const auto& output_exp =
        JUST(OpInterpUtil::Dispatch<TensorTuple>(*op_exp_, {(*output_broadcast_sub)[0]}, attrs));
    // op_reduce_sum_
    struct ReduceSumAttrs {
      Maybe<AttrMap> operator()(int32_t axis) {
        MutableAttrMap attrs{};
        JUST(attrs.SetAttr<std::vector<int32_t>>("axis", {axis}));
        JUST(attrs.SetAttr<bool>("keepdims", true));
        return AttrMap(attrs);
      }
    };
    constexpr auto* GetAttrsReduceSumAttrs = CACHED_FUNCTOR_PTR(ReduceSumAttrs);
    attrs = *JUST(GetAttrsReduceSumAttrs(axis));

    const auto& output_reduce_sum =
        JUST(OpInterpUtil::Dispatch<TensorTuple>(*op_reduce_sum_, {(*output_exp)[0]}, attrs));
    std::shared_ptr<Tensor> broadcast_div_input1 = output_reduce_sum->at(0);
    if (logits_nd_sbp.sbp_parallel_size() == 2) {
      std::vector<Symbol<SbpParallel>> empty_grad_sbp_parallels;
      broadcast_div_input1 = JUST(functional::ToGlobal(
          (*output_reduce_sum)[0], JUST((*output_reduce_sum)[0]->parallel_desc()),
          new_sbp_parallels, new_sbp_parallels, /* check_meta */ false));
    }
    // op_broadcast_div_
    attrs = AttrMap{};
    const auto& predictions = JUST(OpInterpUtil::Dispatch<TensorTuple>(
        *op_broadcast_div_, {(*output_exp)[0], broadcast_div_input1}, attrs));
    // op_sparse_cross_entropy_ms_
    struct SparseCrossEntropy {
      Maybe<AttrMap> operator()(int64_t depth) {
        MutableAttrMap attrs{};
        JUST(attrs.SetAttr<int64_t>("depth", depth));
        return AttrMap(attrs);
      }
    };
    constexpr auto* GetAttrsSparseCrossEntropy = CACHED_FUNCTOR_PTR(SparseCrossEntropy);
    attrs = *JUST(GetAttrsSparseCrossEntropy(depth));

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
  struct CombinedMarginLoss {
    Maybe<AttrMap> operator()(float m1, float m2, float m3, int64_t depth) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<float>("m1", m1));
      JUST(attrs.SetAttr<float>("m2", m2));
      JUST(attrs.SetAttr<float>("m3", m3));
      JUST(attrs.SetAttr<int64_t>("depth", depth));
      return AttrMap(attrs);
    }
  };

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& label, const float& m1,
                           const float& m2, const float& m3) const {
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(CombinedMarginLoss);
    const auto attrs = *JUST(GetAttrs(m1, m2, m3, x->shape()->At(1)));
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
  struct CtcLoss {
    Maybe<AttrMap> operator()(int64_t max_target_length, int32_t blank, bool zero_infinity) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<int64_t>("max_target_length", max_target_length));
      JUST(attrs.SetAttr<int32_t>("blank", blank));
      JUST(attrs.SetAttr<bool>("zero_infinity", zero_infinity));
      return AttrMap(attrs);
    }
  };

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& log_probs,
                           const std::shared_ptr<one::Tensor>& targets,
                           const std::shared_ptr<one::Tensor>& input_lengths,
                           const std::shared_ptr<one::Tensor>& target_lengths,
                           const int64_t& max_target_length, const int& blank,
                           const bool& zero_infinity, const std::string& reduction) const {
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(CtcLoss);
    const auto attrs = *JUST(GetAttrs(max_target_length, blank, zero_infinity));
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
  struct AffineGrid {
    Maybe<AttrMap> operator()(const Shape& size, const bool& align_corners) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<Shape>("size", size));
      JUST(attrs.SetAttr<bool>("align_corners", align_corners));
      return AttrMap(attrs);
    }
  };

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& theta, const Shape& size,
                           const bool& align_corners) const {
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(AffineGrid);
    const auto attrs = *JUST(GetAttrs(size, align_corners));
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
  struct GridSample {
    Maybe<AttrMap> operator()(const std::string& interpolation_mode,
                              const std::string& padding_mode, const bool& align_corners) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<std::string>("interpolation_mode", interpolation_mode));
      JUST(attrs.SetAttr<std::string>("padding_mode", padding_mode));
      JUST(attrs.SetAttr<bool>("align_corners", align_corners));
      return AttrMap(attrs);
    }
  };

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& grid,
                           const std::string& interpolation_mode, const std::string& padding_mode,
                           const bool& align_corners) const {
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(GridSample);
    const auto attrs = *JUST(GetAttrs(interpolation_mode, padding_mode, align_corners));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {input, grid}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class NormalFunctor {
 public:
  NormalFunctor() { op_ = CHECK_JUST(one::OpBuilder("normal").Output("out").Build()); }
  struct Normal {
    Maybe<AttrMap> operator()(double mean, double std, const Shape& shape, DataType dtype,
                              int64_t seed) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<double>("mean", mean));
      JUST(attrs.SetAttr<double>("std", std));
      JUST(attrs.SetAttr<Shape>("shape", shape));
      JUST(attrs.SetAttr<DataType>("dtype", dtype));
      JUST(attrs.SetAttr<int64_t>("seed", seed));
      return AttrMap(attrs);
    }
  };

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
            << " does not match data type of out parameter " << output_tensor_dtype->name();
      }
      dtype = output_tensor_dtype;
      Symbol<Device> out_tensor_device = JUST(out_tensor->device());
      if (optional_device.has_value()) {
        CHECK_OR_RETURN(out_tensor_device == JUST(optional_device))
            << Error::RuntimeError() << "device type " << device->ToString()
            << " does not match device type of out parameter " << out_tensor_device->ToString();
      }
      device = out_tensor_device;
    }

    const auto gen = optional_generator.value_or(JUST(one::DefaultAutoGenerator()));
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(Normal);
    const auto attrs = *JUST(GetAttrs(mean, std, shape, dtype->data_type(), gen->current_seed()));

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

class Normal2Functor {
 public:
  Maybe<Tensor> operator()(const float& mean, const float& std, const int32_t& shape,
                           const Optional<one::Tensor>& out,
                           const Optional<Symbol<DType>>& optional_dtype,
                           const Optional<Symbol<Device>>& optional_device,
                           const Optional<one::Generator>& optional_generator,
                           const bool& requires_grad) const {
    const Shape size = Shape({shape});
    return Normal(mean, std, size, out, optional_dtype, optional_device, optional_generator,
                  requires_grad);
  }
};

class GlobalNormalFunctor {
 public:
  GlobalNormalFunctor() { op_ = CHECK_JUST(one::OpBuilder("normal").Output("out").Build()); }
  struct GlobalNormal {
    Maybe<AttrMap> operator()(double mean, double std, const Shape& shape, DataType dtype,
                              int64_t seed, bool is_lazy_mode, Symbol<NdSbp> nd_sbp) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<double>("mean", mean));
      JUST(attrs.SetAttr<double>("std", std));
      JUST(attrs.SetAttr<Shape>("shape", shape));
      JUST(attrs.SetAttr<DataType>("dtype", dtype));
      JUST(attrs.SetAttr<int64_t>("seed", seed));
      if (is_lazy_mode) {
        JUST(attrs.SetAttr<std::vector<std::string>>("nd_sbp", *JUST(GetNdSbpStrList(nd_sbp))));
      }
      return AttrMap(attrs);
    }
  };

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
    const auto& nd_sbp = JUST(GetNdSbp(sbp_tuple));
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(GlobalNormal);
    const auto attrs = *JUST(GetAttrs(mean, std, shape, dtype->data_type(), gen->current_seed(),
                                      LazyMode::is_enabled(), nd_sbp));

    const auto& distribution_state = std::make_shared<DistributionKernelState>(gen);

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

class GlobalNormal2Functor {
 public:
  Maybe<Tensor> operator()(const float& mean, const float& std, const int32_t& shape,
                           const Optional<one::Tensor>& out, const Symbol<ParallelDesc>& placement,
                           const std::vector<Symbol<SbpParallel>>& sbp_tuple,
                           const Optional<Symbol<DType>>& optional_dtype,
                           const Optional<one::Generator>& optional_generator,
                           const bool& requires_grad) const {
    const Shape size = Shape({shape});
    return GlobalNormal(mean, std, size, out, placement, sbp_tuple, optional_dtype,
                        optional_generator, requires_grad);
  }
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
  struct Normalization {
    Maybe<AttrMap> operator()(int32_t axis, float epsilon, float momentum) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<int32_t>("axis", axis));
      JUST(attrs.SetAttr<float>("epsilon", epsilon));
      // convert torch momentum to tensorflow momentum.
      JUST(attrs.SetAttr<float>("momentum", 1.0 - momentum));
      return AttrMap(attrs);
    }
  };

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const Optional<one::Tensor>& moving_mean,
                           const Optional<one::Tensor>& moving_variance,
                           const Optional<one::Tensor>& gamma, const Optional<one::Tensor>& beta,
                           const int32_t& axis, const float& epsilon, const float& momentum,
                           const bool& training) const {
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(Normalization);
    const auto attrs = *JUST(GetAttrs(axis, epsilon, momentum));

    CHECK_OR_RETURN((moving_mean && moving_variance) || (!moving_mean && !moving_variance))
        << Error::RuntimeError()
        << "Both moving_mean and moving_variance should be None or Tensor.";

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

    if (!training) {
      CHECK_OR_RETURN(moving_mean && moving_variance)
          << Error::RuntimeError() << "Must have moving_mean and moving_variance in eval mode.";
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
  struct NormalizationAddRelu {
    Maybe<AttrMap> operator()(const int32_t& axis, const float& epsilon, const float& momentum) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<int32_t>("axis", axis));
      JUST(attrs.SetAttr<float>("epsilon", epsilon));
      // convert torch momentum to tensorflow momentum
      JUST(attrs.SetAttr<float>("momentum", 1.0f - momentum));
      return AttrMap(attrs);
    }
  };

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const Optional<one::Tensor>& addend,
                           const Optional<one::Tensor>& moving_mean,
                           const Optional<one::Tensor>& moving_variance,
                           const std::shared_ptr<one::Tensor>& gamma,
                           const std::shared_ptr<one::Tensor>& beta, const int32_t& axis,
                           const float& epsilon, const float& momentum,
                           const bool& is_training) const {
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(NormalizationAddRelu);
    const auto attrs = *JUST(GetAttrs(axis, epsilon, momentum));
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

class PadFunctor {
 public:
  PadFunctor() {
    pad_ = CHECK_JUST(one::OpBuilder("pad").Input("x").Output("y").Build());
    reflect_pad1d_ = CHECK_JUST(one::OpBuilder("reflection_pad1d").Input("x").Output("y").Build());
    reflect_pad2d_ = CHECK_JUST(one::OpBuilder("reflection_pad2d").Input("x").Output("y").Build());
    replicate_pad1d_ =
        CHECK_JUST(one::OpBuilder("replication_pad1d").Input("x").Output("y").Build());
    replicate_pad2d_ =
        CHECK_JUST(one::OpBuilder("replication_pad2d").Input("x").Output("y").Build());
  }
  struct PadAttr {
    Maybe<AttrMap> operator()(const std::vector<int64_t>& pad) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<std::vector<int64_t>>("padding", pad));
      return AttrMap(attrs);
    }
  };
  struct ConstantPadAttr {
    Maybe<AttrMap> operator()(DataType in_dtype, int64_t ndim, const std::vector<int64_t>& pad,
                              const Scalar& value) {
      MutableAttrMap attrs;
      const int64_t pad_size = pad.size();
      if (IsFloatingDataType(in_dtype) || in_dtype == DataType::kFloat16) {
        JUST(attrs.SetAttr<double>("floating_constant_value", value.As<double>()));
        JUST(attrs.SetAttr<int64_t>("integral_constant_value", 0));
      } else if (IsIntegralDataType(in_dtype)) {
        JUST(attrs.SetAttr<double>("floating_constant_value", 0));
        JUST(attrs.SetAttr<int64_t>("integral_constant_value", value.As<int64_t>()));
      } else {
        UNIMPLEMENTED_THEN_RETURN() << "Data type should be floating or integral type.";
      }

      std::vector<int64_t> pad_before(ndim, 0);
      std::vector<int64_t> pad_after(ndim, 0);
      const int64_t pad_pair = pad_size / 2;
      for (int64_t i = 0; i < pad_pair; ++i) {
        pad_before[ndim - i - 1] = pad[2 * i];
        pad_after[ndim - i - 1] = pad[2 * i + 1];
      }
      JUST(attrs.SetAttr<std::vector<int64_t>>("padding_before", pad_before));
      JUST(attrs.SetAttr<std::vector<int64_t>>("padding_after", pad_after));
      return AttrMap(attrs);
    }
  };

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::vector<int64_t>& pad, const std::string& mode,
                           const Scalar& value) const {
    const int64_t ndim = input->shape()->NumAxes();
    const int64_t pad_size = pad.size();
    CHECK_LE_OR_RETURN(pad_size, 2 * ndim)
        << Error::RuntimeError() << "Pad size should less than or equal to input axes * 2.";
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(PadAttr);
    const auto pad_attrs = *JUST(GetAttrs(pad));
    if (mode == "constant") {
      CHECK_EQ_OR_RETURN(pad_size % 2, 0)
          << Error::RuntimeError() << "Length of pad must be even but instead it equals "
          << pad_size;
      constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(ConstantPadAttr);
      const auto attrs = *JUST(GetAttrs(input->dtype()->data_type(), ndim, pad, value));
      return OpInterpUtil::Dispatch<Tensor>(*pad_, {input}, attrs);
    } else if (mode == "reflect") {
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
              JUST(OpInterpUtil::Dispatch<Tensor>(*reflect_pad1d_, {unsqueezed_input}, pad_attrs));
          return JUST(functional::Squeeze(unsqueezed_output, std::vector<int32_t>{0}));
        }
        return OpInterpUtil::Dispatch<Tensor>(*reflect_pad1d_, {input}, pad_attrs);
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
              JUST(OpInterpUtil::Dispatch<Tensor>(*reflect_pad2d_, {unsqueezed_input}, pad_attrs));
          return JUST(functional::Squeeze(unsqueezed_output, std::vector<int32_t>{0}));
        }
        return OpInterpUtil::Dispatch<Tensor>(*reflect_pad2d_, {input}, pad_attrs);
      } else if (pad_size == 6) {
        UNIMPLEMENTED_THEN_RETURN() << "5D reflect padding are not supported for now";
      } else {
        UNIMPLEMENTED_THEN_RETURN()
            << "Only 2D, 3D, 4D, 5D padding with non-constant padding are supported for now";
      }

    } else if (mode == "replicate") {
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
          auto unsqueezed_output = JUST(
              OpInterpUtil::Dispatch<Tensor>(*replicate_pad1d_, {unsqueezed_input}, pad_attrs));
          return JUST(functional::Squeeze(unsqueezed_output, std::vector<int32_t>{0}));
        }
        return OpInterpUtil::Dispatch<Tensor>(*replicate_pad1d_, {input}, pad_attrs);
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
          auto unsqueezed_output = JUST(
              OpInterpUtil::Dispatch<Tensor>(*replicate_pad2d_, {unsqueezed_input}, pad_attrs));
          return JUST(functional::Squeeze(unsqueezed_output, std::vector<int32_t>{0}));
        }
        return OpInterpUtil::Dispatch<Tensor>(*replicate_pad2d_, {input}, pad_attrs);
      } else if (pad_size == 6) {
        UNIMPLEMENTED_THEN_RETURN() << "5D replicate padding are not supported for now";
      } else {
        UNIMPLEMENTED_THEN_RETURN()
            << "Only 2D, 3D, 4D, 5D padding with non-constant padding are supported for now";
      }

    } else {
      UNIMPLEMENTED_THEN_RETURN() << "Pad mode is " << mode
                                  << ", but only constant, reflect and replicate are valid.";
    }
  }

 private:
  std::shared_ptr<OpExpr> pad_;
  std::shared_ptr<OpExpr> reflect_pad1d_;
  std::shared_ptr<OpExpr> reflect_pad2d_;
  std::shared_ptr<OpExpr> replicate_pad1d_;
  std::shared_ptr<OpExpr> replicate_pad2d_;
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

  struct Dropout {
    Maybe<AttrMap> operator()(float p) {
      MutableAttrMap dropout_attrs;
      JUST(dropout_attrs.SetAttr<float>("rate", p));
      return AttrMap(dropout_attrs);
    }
  };

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
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(Dropout);
    const auto dropout_attrs = *JUST(GetAttrs(p));
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
  struct DropoutGrad {
    Maybe<AttrMap> operator()(float scale) {
      MutableAttrMap dropout_grad_attrs;
      JUST(dropout_grad_attrs.SetAttr<float>("scale", scale));
      return AttrMap(dropout_grad_attrs);
    }
  };

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& mask, const float& scale) const {
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(DropoutGrad);
    const auto dropout_grad_attrs = *JUST(GetAttrs(scale));
    return OpInterpUtil::Dispatch<Tensor>(*dropout_grad_op_, {dy, mask}, dropout_grad_attrs);
  }

 private:
  std::shared_ptr<OpExpr> dropout_grad_op_;
};

class AvgPoolNDFunctor {
 public:
  AvgPoolNDFunctor() = default;
  virtual ~AvgPoolNDFunctor() = default;
  struct AvgPoolND {
    Maybe<AttrMap> operator()(const std::string& data_format, const std::vector<int32_t>& padding,
                              const std::vector<int32_t>& kernel_size,
                              const std::vector<int32_t>& stride, bool ceil_mode,
                              bool count_include_pad, int32_t divisor_override) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<std::string>("data_format", data_format));
      JUST(attrs.SetAttr<std::vector<int32_t>>("padding", padding));
      JUST(attrs.SetAttr<std::vector<int32_t>>("kernel_size", kernel_size));
      JUST(attrs.SetAttr<std::vector<int32_t>>("stride", stride));
      JUST(attrs.SetAttr<bool>("ceil_mode", ceil_mode));
      JUST(attrs.SetAttr<bool>("count_include_pad", count_include_pad));
      JUST(attrs.SetAttr<int32_t>("divisor_override", divisor_override));
      return AttrMap(attrs);
    }
  };

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::vector<int32_t>& kernel_size,
                           const Optional<std::vector<int32_t>>& stride,
                           const std::vector<int32_t>& padding, const bool& ceil_mode,
                           const bool& count_include_pad, const int32_t& divisor_override,
                           const std::string& data_format) const {
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(AvgPoolND);
    // If stride is None, we set it as kernel_size to align Pytorch
    const auto attrs = *JUST(GetAttrs(data_format, padding, kernel_size,
                                      (stride.has_value() ? *JUST(stride) : kernel_size), ceil_mode,
                                      count_include_pad, divisor_override));
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
  struct Unfold {
    Maybe<AttrMap> operator()(const std::string& data_format,
                              const std::vector<int32_t>& kernel_size,
                              const std::vector<int32_t>& dilation_rate,
                              const std::vector<int32_t>& padding,
                              const std::vector<int32_t>& strides) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<std::string>("data_format", data_format));
      JUST(attrs.SetAttr<std::vector<int32_t>>("kernel_size", kernel_size));
      JUST(attrs.SetAttr<std::vector<int32_t>>("dilation_rate", dilation_rate));
      JUST(attrs.SetAttr<std::vector<int32_t>>("padding", padding));
      JUST(attrs.SetAttr<std::vector<int32_t>>("strides", strides));
      return AttrMap(attrs);
    }
  };

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::vector<int32_t>& kernel_size,
                           const std::vector<int32_t>& dilation_rate,
                           const std::vector<int32_t>& padding, const std::vector<int32_t>& strides,
                           const std::string& data_format) const {
    const auto& x_shape = x->shape();
    // Only Support 4d tensor now.
    CHECK_EQ_OR_RETURN(x_shape->NumAxes(), 4)
        << Error::RuntimeError() << "Input Tensor dim should == 4";
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(Unfold);
    const auto attrs = *JUST(GetAttrs(data_format, kernel_size, dilation_rate, padding, strides));
    return OpInterpUtil::Dispatch<Tensor>(*unfold_op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> unfold_op_;
};

class FoldFunctor {
 public:
  FoldFunctor() { fold_op_ = CHECK_JUST(one::OpBuilder("fold").Input("x").Output("y").Build()); }
  struct Fold {
    Maybe<AttrMap> operator()(const std::string& data_format,
                              const std::vector<int32_t>& output_size,
                              const std::vector<int32_t>& kernel_size,
                              const std::vector<int32_t>& dilation_rate,
                              const std::vector<int32_t>& padding,
                              const std::vector<int32_t>& strides) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<std::string>("data_format", data_format));
      JUST(attrs.SetAttr<std::vector<int32_t>>("output_size", output_size));
      JUST(attrs.SetAttr<std::vector<int32_t>>("kernel_size", kernel_size));
      JUST(attrs.SetAttr<std::vector<int32_t>>("dilation_rate", dilation_rate));
      JUST(attrs.SetAttr<std::vector<int32_t>>("padding", padding));
      JUST(attrs.SetAttr<std::vector<int32_t>>("strides", strides));
      return AttrMap(attrs);
    }
  };

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

    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(Fold);
    const auto attrs =
        *JUST(GetAttrs(data_format, output_size, kernel_size, dilation_rate, padding, strides));
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

  struct OneHot {
    Maybe<AttrMap> operator()(int64_t depth, const Scalar& on_value, const Scalar& off_value) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<int64_t>("depth", depth));
      // Refer to: https://github.com/Oneflow-Inc/oneflow/pull/5315/files#r755823506
      bool is_on_value_double = on_value.IsFloatingPoint();
      bool is_off_value_double = off_value.IsFloatingPoint();
      if (is_on_value_double || is_off_value_double) {
        JUST(attrs.SetAttr<DataType>("dtype", kFloat));
        JUST(attrs.SetAttr<double>("floating_on_value", on_value.As<double>()));
        JUST(attrs.SetAttr<double>("floating_off_value", off_value.As<double>()));
        JUST(attrs.SetAttr<int64_t>("integer_on_value", 0));
        JUST(attrs.SetAttr<int64_t>("integer_off_value", 0));
      } else {
        JUST(attrs.SetAttr<DataType>("dtype", kInt64));
        JUST(attrs.SetAttr<double>("floating_on_value", 0));
        JUST(attrs.SetAttr<double>("floating_off_value", 0));
        JUST(attrs.SetAttr<int64_t>("integer_on_value", on_value.As<int64_t>()));
        JUST(attrs.SetAttr<int64_t>("integer_off_value", off_value.As<int64_t>()));
      }
      return AttrMap(attrs);
    }
  };

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input, const int64_t& num_classes,
                           const Scalar& on_value, const Scalar& off_value) const {
    CHECK_OR_RETURN(!IsFloatingDataType(input->dtype()->data_type()))
        << Error::RuntimeError() << "one_hot is only applicable to index tensor.";
    int64_t depth = 0;
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
      depth = max + 1;
    } else {
      depth = num_classes;
    }
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(OneHot);
    const auto attrs = *JUST(GetAttrs(depth, on_value, off_value));
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
  struct L2Normalize {
    Maybe<AttrMap> operator()(float epsilon, int32_t final_dim) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<float>("epsilon", epsilon));
      JUST(attrs.SetAttr<int32_t>("axis", final_dim));
      return AttrMap(attrs);
    }
  };

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input, const int32_t& axis,
                           const float& epsilon) const {
    const auto ndims = input->shape()->NumAxes();
    const auto final_dim = ndims - 1;

    auto axis_ = axis >= 0 ? axis : axis + ndims;
    CHECK_GE_OR_RETURN(axis_, 0) << Error::RuntimeError() << "Axis should >=0 but axis is " << axis_
                                 << " now.";
    CHECK_LE_OR_RETURN(axis_, final_dim) << Error::RuntimeError() << "Axis should < " << ndims
                                         << " but axis is " << axis_ << " now.";

    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(L2Normalize);
    const auto attrs = *JUST(GetAttrs(epsilon, final_dim));

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
  struct FusedSelfAttention {
    Maybe<AttrMap> operator()(int64_t head_size, float alpha) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<int64_t>("head_size", head_size));
      JUST(attrs.SetAttr<float>("alpha", alpha));
      return AttrMap(attrs);
    }
  };

  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& hidden_states,
                                const int64_t& head_size, const float& alpha) const {
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(FusedSelfAttention);
    const auto attrs = *JUST(GetAttrs(head_size, alpha));
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
  struct FusedSelfAttentionGrad {
    Maybe<AttrMap> operator()(float alpha) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<float>("alpha", alpha));
      return AttrMap(attrs);
    }
  };

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& query_mul_key_grad,
                           const std::shared_ptr<one::Tensor>& value_grad,
                           const std::shared_ptr<one::Tensor>& hidden_states,
                           const float& alpha) const {
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(FusedSelfAttentionGrad);
    const auto attrs = *JUST(GetAttrs(alpha));
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
  struct RandomMaskLike {
    Maybe<AttrMap> operator()(float p, int64_t seed) {
      MutableAttrMap random_mask_like_attrs;
      JUST(random_mask_like_attrs.SetAttr<float>("rate", p));
      JUST(random_mask_like_attrs.SetAttr<int64_t>("seed", seed));
      return AttrMap(random_mask_like_attrs);
    }
  };

  struct FuseOpAttr {
    Maybe<AttrMap> operator()(int64_t diagonal, float tril_scale_value, float mask_scale_value,
                              float tril_fill_value) {
      MutableAttrMap fused_attrs;
      JUST(fused_attrs.SetAttr<int64_t>("diagonal", diagonal));
      JUST(fused_attrs.SetAttr<float>("tril_scale_value", tril_scale_value));
      JUST(fused_attrs.SetAttr<float>("mask_scale_value", mask_scale_value));
      JUST(fused_attrs.SetAttr<float>("tril_fill_value", tril_fill_value));
      return AttrMap(fused_attrs);
    }
  };

  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& x, const float p,
                                const int64_t diagonal, const float tril_scale_value,
                                const float tril_fill_value,
                                const Optional<one::Generator>& generator) const {
    const auto gen = generator.value_or(JUST(one::DefaultAutoGenerator()));
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(RandomMaskLike);
    const auto random_mask_like_attrs = *JUST(GetAttrs(p, gen->current_seed()));
    const auto& random_mask_like_state = std::make_shared<RandomMaskLikeKernelState>(gen);

    const auto& mask = JUST(OpInterpUtil::Dispatch<Tensor>(
        *random_mask_like_op_, {x},
        OpExprInterpContext(random_mask_like_attrs, random_mask_like_state)));

    float mask_scale_value = 1.0;
    if (p != 1.0) { mask_scale_value = 1.0 / (1.0 - p); }
    constexpr auto* GetAttrsFuseOpAttr = CACHED_FUNCTOR_PTR(FuseOpAttr);
    const auto fused_attrs =
        *JUST(GetAttrsFuseOpAttr(diagonal, tril_scale_value, mask_scale_value, tril_fill_value));

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
  struct L2NormalizeGrad {
    Maybe<AttrMap> operator()(int32_t axis, float epsilon) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<int32_t>("axis", axis));
      JUST(attrs.SetAttr<float>("epsilon", epsilon));
      return AttrMap(attrs);
    }
  };

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& y,
                           const std::shared_ptr<one::Tensor>& square_x_sum, const int32_t& axis,
                           const float& epsilon) const {
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(L2NormalizeGrad);
    const auto attrs = *JUST(GetAttrs(axis, epsilon));
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
  struct FusedBiasAddGelu {
    Maybe<AttrMap> operator()(int32_t axis) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<int32_t>("axis", axis));
      return AttrMap(attrs);
    }
  };

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& a,
                           const std::shared_ptr<one::Tensor>& b, const int32_t& axis) const {
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(FusedBiasAddGelu);
    const auto attrs = *JUST(GetAttrs(axis));
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
  struct FusedBiasAddGeluGrad {
    Maybe<AttrMap> operator()(int32_t axis) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<int32_t>("axis", axis));
      return AttrMap(attrs);
    }
  };

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& a,
                           const std::shared_ptr<one::Tensor>& b,
                           const std::shared_ptr<one::Tensor>& dy, const int32_t& axis) const {
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(FusedBiasAddGeluGrad);
    const auto attrs = *JUST(GetAttrs(axis));
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
  struct RanddomMaskLike {
    Maybe<AttrMap> operator()(float rate, int64_t seed) {
      MutableAttrMap random_mask_like_attrs;
      JUST(random_mask_like_attrs.SetAttr<float>("rate", rate));
      JUST(random_mask_like_attrs.SetAttr<int64_t>("seed", seed));
      return AttrMap(random_mask_like_attrs);
    }
  };

  struct FusedBiasAddMask {
    Maybe<AttrMap> operator()(float scale, int32_t axis) {
      MutableAttrMap fused_bias_add_mask_attrs;
      JUST(fused_bias_add_mask_attrs.SetAttr<float>("scale", scale));
      JUST(fused_bias_add_mask_attrs.SetAttr<int32_t>("axis", axis));
      return AttrMap(fused_bias_add_mask_attrs);
    }
  };

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& a,
                           const std::shared_ptr<one::Tensor>& b, const float& p,
                           const int32_t& axis, const Optional<one::Generator>& generator) const {
    const auto gen = generator.value_or(JUST(one::DefaultAutoGenerator()));
    constexpr auto* GetAttrsRanddomMaskLike = CACHED_FUNCTOR_PTR(RanddomMaskLike);
    const auto random_mask_like_attrs = *JUST(GetAttrsRanddomMaskLike(p, gen->current_seed()));
    const auto& random_mask_like_state = std::make_shared<RandomMaskLikeKernelState>(gen);
    float scale = 0.0;
    if (p != 1.0) { scale = 1.0 / (1.0 - p); }
    int32_t axis_val = axis;
    if (axis_val < 0) {
      const int64_t num_axes = a->shape()->NumAxes();
      axis_val += num_axes;
    }
    constexpr auto* GetAttrsFusedBiasAddMask = CACHED_FUNCTOR_PTR(FusedBiasAddMask);
    const auto fused_bias_add_mask_attrs = *JUST(GetAttrsFusedBiasAddMask(scale, axis_val));
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

  struct FusedScaleTril {
    Maybe<AttrMap> operator()(const int64_t& diagonal, const Scalar& fill_value,
                              const Scalar& scale) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<int64_t>("diagonal", diagonal));
      bool is_fill_value_double = fill_value.IsFloatingPoint();
      bool is_scale_double = scale.IsFloatingPoint();
      if (is_fill_value_double) {
        JUST(attrs.SetAttr<double>("floating_fill_value", fill_value.As<double>()));
        JUST(attrs.SetAttr<int64_t>("integer_fill_value", 0));
        JUST(attrs.SetAttr<bool>("is_floating_fill_value", true));
      } else {
        JUST(attrs.SetAttr<double>("floating_fill_value", 0));
        JUST(attrs.SetAttr<int64_t>("integer_fill_value", fill_value.As<int64_t>()));
        JUST(attrs.SetAttr<bool>("is_floating_fill_value", false));
      }

      if (is_scale_double) {
        JUST(attrs.SetAttr<double>("floating_scale_value", scale.As<double>()));
        JUST(attrs.SetAttr<int64_t>("integer_scale_value", 0));
        JUST(attrs.SetAttr<bool>("is_floating_scale_value", true));
      } else {
        JUST(attrs.SetAttr<double>("floating_scale_value", 0));
        JUST(attrs.SetAttr<int64_t>("integer_scale_value", scale.As<int64_t>()));
        JUST(attrs.SetAttr<bool>("is_floating_scale_value", false));
      }
      return AttrMap(attrs);
    }
  };

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const int64_t& diagonal,
                           const Scalar& fill_value, const Scalar& scale) const {
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(FusedScaleTril);
    const auto attrs = *JUST(GetAttrs(diagonal, fill_value, scale));
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
  struct FusedScaleMaskSoftmax {
    Maybe<AttrMap> operator()(const float& fill_value, const float& scale) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<float>("scale_value", scale));
      JUST(attrs.SetAttr<float>("mask_fill_value", fill_value));
      return AttrMap(attrs);
    }
  };

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& mask, const float& fill_value,
                           const float& scale) const {
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(FusedScaleMaskSoftmax);
    const auto attrs = *JUST(GetAttrs(scale, fill_value));
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
  struct RandomMaskLike {
    Maybe<AttrMap> operator()(float rate, int64_t seed) {
      MutableAttrMap random_mask_like_attrs;
      JUST(random_mask_like_attrs.SetAttr<float>("rate", rate));
      JUST(random_mask_like_attrs.SetAttr<int64_t>("seed", seed));
      return AttrMap(random_mask_like_attrs);
    }
  };

  struct FusedScaleMaskSoftmaxDropout {
    Maybe<AttrMap> operator()(float scale, float fill_value, float dropout_scale) {
      MutableAttrMap fused_scale_mask_softmax_dropout_attrs;
      JUST(fused_scale_mask_softmax_dropout_attrs.SetAttr<float>("scale_value", scale));
      JUST(fused_scale_mask_softmax_dropout_attrs.SetAttr<float>("mask_fill_value", fill_value));
      JUST(fused_scale_mask_softmax_dropout_attrs.SetAttr<float>("dropout_scale_value",
                                                                 dropout_scale));
      return AttrMap(fused_scale_mask_softmax_dropout_attrs);
    }
  };

  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& x,
                                const std::shared_ptr<one::Tensor>& mask, const float& fill_value,
                                const float& scale, const float& p, const bool& training,
                                const Optional<one::Generator>& generator) const {
    float rate = p;
    if (!training) rate = 0.0;
    const auto gen = generator.value_or(JUST(one::DefaultAutoGenerator()));
    constexpr auto* GetAttrsRandomMaskLike = CACHED_FUNCTOR_PTR(RandomMaskLike);
    const auto random_mask_like_attrs = *JUST(GetAttrsRandomMaskLike(rate, gen->current_seed()));
    const auto& random_mask_like_state = std::make_shared<RandomMaskLikeKernelState>(gen);

    const auto& dropout_mask = JUST(OpInterpUtil::Dispatch<Tensor>(
        *random_mask_like_op_, {x},
        OpExprInterpContext(random_mask_like_attrs, random_mask_like_state)));

    float dropout_scale = 0.0;
    if (rate != 1.0) { dropout_scale = 1.0 / (1.0 - rate); }
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(FusedScaleMaskSoftmaxDropout);
    const auto fused_scale_mask_softmax_dropout_attrs =
        *JUST(GetAttrs(scale, fill_value, dropout_scale));

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
  struct CtcGreedyDecoder {
    Maybe<AttrMap> operator()(bool merge_repeated) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<bool>("merge_repeated", merge_repeated));
      return AttrMap(attrs);
    }
  };

  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& log_probs,
                                const std::shared_ptr<one::Tensor>& input_lengths,
                                const bool& merge_repeated) const {
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(CtcGreedyDecoder);
    const auto attrs = *JUST(GetAttrs(merge_repeated));
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

  struct Nms {
    Maybe<AttrMap> operator()(float iou_threshold, int32_t keep_n) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<float>("iou_threshold", iou_threshold));
      JUST(attrs.SetAttr<int32_t>("keep_n", keep_n));
      return AttrMap(attrs);
    }
  };

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const float& iou_threshold,
                           const int32_t& keep_n) const {
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(Nms);
    const auto attrs = *JUST(GetAttrs(iou_threshold, keep_n));
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

  struct RoiAlign {
    Maybe<AttrMap> operator()(const float& spatial_scale, const int32_t& pooled_h,
                              const int32_t& pooled_w, const int32_t& sampling_ratio,
                              const bool& aligned) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<float>("spatial_scale", spatial_scale));
      JUST(attrs.SetAttr<int32_t>("pooled_h", pooled_h));
      JUST(attrs.SetAttr<int32_t>("pooled_w", pooled_w));
      JUST(attrs.SetAttr<int32_t>("sampling_ratio", sampling_ratio));
      JUST(attrs.SetAttr<bool>("aligned", aligned));
      return AttrMap(attrs);
    }
  };

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& rois, const float& spatial_scale,
                           const int32_t& pooled_h, const int32_t& pooled_w,
                           const int32_t& sampling_ratio, const bool& aligned) const {
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(RoiAlign);
    const auto attrs = *JUST(GetAttrs(spatial_scale, pooled_h, pooled_w, sampling_ratio, aligned));
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

  struct RoiAlignGrad {
    Maybe<AttrMap> operator()(const float& spatial_scale, const int32_t& pooled_h,
                              const int32_t& pooled_w, const int32_t& sampling_ratio,
                              const bool& aligned) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<float>("spatial_scale", spatial_scale));
      JUST(attrs.SetAttr<int32_t>("pooled_h", pooled_h));
      JUST(attrs.SetAttr<int32_t>("pooled_w", pooled_w));
      JUST(attrs.SetAttr<int32_t>("sampling_ratio", sampling_ratio));
      JUST(attrs.SetAttr<bool>("aligned", aligned));
      return AttrMap(attrs);
    }
  };

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& x_like,
                           const std::shared_ptr<one::Tensor>& rois, const float& spatial_scale,
                           const int32_t& pooled_h, const int32_t& pooled_w,
                           const int32_t& sampling_ratio, const bool& aligned) const {
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(RoiAlignGrad);
    const auto attrs = *JUST(GetAttrs(spatial_scale, pooled_h, pooled_w, sampling_ratio, aligned));
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

  struct FusedDotFeatureInteraction {
    Maybe<AttrMap> operator()(const bool& self_interaction, const int32_t& output_padding,
                              const std::string& pooling, bool has_output_concat) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<bool>("self_interaction", self_interaction));
      JUST(attrs.SetAttr<int32_t>("output_padding", output_padding));
      JUST(attrs.SetAttr<std::string>("pooling", pooling));
      JUST(attrs.SetAttr<bool>("has_output_concat", has_output_concat));
      return AttrMap(attrs);
    }
  };

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

    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(FusedDotFeatureInteraction);
    if (pooling == "sum") {
      CHECK_EQ_OR_RETURN(output_padding, 0)
          << Error::RuntimeError() << "output_padding should be equal to 0. ";
      CHECK_OR_RETURN(!output_concat) << Error::RuntimeError() << "output_concat should not exist";
      const auto attrs = *JUST(GetAttrs(self_interaction, output_padding, pooling, false));
      const std::shared_ptr<one::Tensor>& bi_interaction = JUST(OpInterpUtil::Dispatch<Tensor>(
          *JUST(oneflow::VectorAt(ops_no_output_concat_, n_features - 1)), inputs, attrs));
      std::vector<int32_t> reduce_axes_vec = {1};
      return functional::ReduceSum(bi_interaction, reduce_axes_vec, true);
    }
    if (output_concat) {
      const auto attrs = *JUST(GetAttrs(self_interaction, output_padding, pooling, true));
      inputs.push_back(JUST(output_concat));
      return OpInterpUtil::Dispatch<Tensor>(
          *JUST(oneflow::VectorAt(ops_has_output_concat_, n_features - 1)), inputs, attrs);
    } else {
      const auto attrs = *JUST(GetAttrs(self_interaction, output_padding, pooling, false));
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

  struct FusedCrossFeatureInteraction {
    Maybe<AttrMap> operator()(const std::string& interaction_mode) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<std::string>("interaction_mode", interaction_mode));
      return AttrMap(attrs);
    }
  };

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& weight,
                           const std::shared_ptr<one::Tensor>& x0,
                           const std::shared_ptr<one::Tensor>& bias,
                           const std::string& interaction_mode) const {
    if (interaction_mode != "vector" && interaction_mode != "matrix") {
      UNIMPLEMENTED_THEN_RETURN()
          << "Fused Cross Interaction mode only support `vector` and `matrix`. ";
    }
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(FusedCrossFeatureInteraction);
    const auto attrs = *JUST(GetAttrs(interaction_mode));
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

  struct OneEmbeddingIdShuffle {
    Maybe<AttrMap> operator()(const int32_t& num_tables, const std::string& embedding_name) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<int32_t>("num_tables", num_tables));
      JUST(attrs.SetAttr<std::string>("embedding_name", embedding_name));
      return AttrMap(attrs);
    }
  };

  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& ids,
                                const Optional<one::Tensor>& table_ids, const int32_t& num_tables,
                                const std::string& embedding_name) const {
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(OneEmbeddingIdShuffle);
    const auto attrs = *JUST(GetAttrs(num_tables, embedding_name));
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

  struct OneEmbeddingEmbeddingShuffle {
    Maybe<AttrMap> operator()(int64_t embedding_size, const std::string& embedding_name) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<int64_t>("embedding_size", embedding_size));
      JUST(attrs.SetAttr<std::string>("embedding_name", embedding_name));
      return AttrMap(attrs);
    }
  };

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& cur_rank_embeddings,
                           const std::shared_ptr<one::Tensor>& num_unique_matrix,
                           const std::shared_ptr<one::Tensor>& cur_rank_inverse_indices,
                           const std::shared_ptr<one::Tensor>& inverse_unique_partition_indices,
                           const std::string& embedding_name) const {
    const int64_t num_axes = cur_rank_embeddings->shape()->NumAxes();
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(OneEmbeddingEmbeddingShuffle);
    const auto attrs =
        *JUST(GetAttrs(cur_rank_embeddings->shape()->At(num_axes - 1), embedding_name));
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

  struct OneEmbeddingEmbeddingGradientShuffle {
    Maybe<AttrMap> operator()(int64_t embedding_size, const std::string& embedding_name) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<int64_t>("embedding_size", embedding_size));
      JUST(attrs.SetAttr<std::string>("embedding_name", embedding_name));
      return AttrMap(attrs);
    }
  };

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& embedding_grad,
                           const std::shared_ptr<one::Tensor>& num_unique_matrix,
                           const std::shared_ptr<one::Tensor>& cur_rank_inverse_indices,
                           const std::shared_ptr<one::Tensor>& inverse_unique_partition_indices,
                           const std::string& embedding_name) const {
    const int64_t num_axes = embedding_grad->shape()->NumAxes();
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(OneEmbeddingEmbeddingGradientShuffle);
    const auto attrs = *JUST(GetAttrs(embedding_grad->shape()->At(num_axes - 1), embedding_name));
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

  struct OneEmbeddingLookup {
    Maybe<AttrMap> operator()(const Symbol<DType>& dtype, const int64_t embedding_size,
                              const int32_t num_tables, const std::string& embedding_tables,
                              const std::string& key_value_store_options) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<DataType>("dtype", dtype->data_type()));
      JUST(attrs.SetAttr<int64_t>("embedding_size", embedding_size));
      JUST(attrs.SetAttr<int32_t>("num_tables", num_tables));
      JUST(attrs.SetAttr<std::string>("embedding_tables", embedding_tables));
      JUST(attrs.SetAttr<std::string>("key_value_store_options", key_value_store_options));
      return AttrMap(attrs);
    }
  };

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& shadow,
                           const std::shared_ptr<one::Tensor>& ids,
                           const Optional<one::Tensor>& table_ids, const Symbol<DType>& dtype,
                           const int64_t embedding_size, const int32_t num_tables,
                           const std::string& embedding_tables,
                           const std::string& key_value_store_options) const {
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(OneEmbeddingLookup);
    const auto attrs = *JUST(
        GetAttrs(dtype, embedding_size, num_tables, embedding_tables, key_value_store_options));
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

  struct OneEmbeddingUniqueKeyValuePair {
    Maybe<AttrMap> operator()(int32_t num_tables) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<int32_t>("num_tables", num_tables));
      return AttrMap(attrs);
    }
  };

  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& keys,
                                const Optional<one::Tensor>& values,
                                const int32_t num_tables) const {
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(OneEmbeddingUniqueKeyValuePair);
    const auto attrs = *JUST(GetAttrs(num_tables));
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

  struct OneEmbeddingSgdUpdate {
    Maybe<AttrMap> operator()(const double scale, const float weight_decay, const int64_t line_size,
                              const int64_t embedding_size) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<double>("scale", scale));
      JUST(attrs.SetAttr<float>("weight_decay", weight_decay));
      JUST(attrs.SetAttr<int64_t>("line_size", line_size));
      JUST(attrs.SetAttr<int64_t>("embedding_size", embedding_size));
      return AttrMap(attrs);
    }
  };

  struct OneEmbeddingMomentumUpdate {
    Maybe<AttrMap> operator()(const double scale, const float weight_decay, const float momentum,
                              const int64_t line_size, const int64_t embedding_size) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<double>("scale", scale));
      JUST(attrs.SetAttr<float>("weight_decay", weight_decay));
      JUST(attrs.SetAttr<float>("beta", momentum));
      JUST(attrs.SetAttr<int64_t>("line_size", line_size));
      JUST(attrs.SetAttr<int64_t>("embedding_size", embedding_size));
      return AttrMap(attrs);
    }
  };

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& num_unique_ids,
                           const std::shared_ptr<one::Tensor>& unique_embeddings,
                           const std::shared_ptr<one::Tensor>& embedding_grad,
                           const std::shared_ptr<one::Tensor>& learning_rate,
                           const std::shared_ptr<one::Tensor>& down_scale_by_tensor,
                           const std::shared_ptr<one::Tensor>& skip_if, const double scale,
                           const float weight_decay, const float momentum, const int64_t line_size,
                           const int64_t embedding_size) const {
    if (momentum == 0) {
      constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(OneEmbeddingSgdUpdate);
      const auto attrs = *JUST(GetAttrs(scale, weight_decay, line_size, embedding_size));
      return OpInterpUtil::Dispatch<Tensor>(*sgd_op_,
                                            {num_unique_ids, unique_embeddings, embedding_grad,
                                             learning_rate, down_scale_by_tensor, skip_if},
                                            attrs);
    } else {
      constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(OneEmbeddingMomentumUpdate);
      const auto attrs = *JUST(GetAttrs(scale, weight_decay, momentum, line_size, embedding_size));
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

  struct OneEmbeddingAdamUpdate {
    Maybe<AttrMap> operator()(const double scale, const float weight_decay, const float beta1,
                              const float beta2, const float epsilon, const bool do_bias_correction,
                              const int64_t line_size, const int64_t embedding_size) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<double>("scale", scale));
      JUST(attrs.SetAttr<float>("weight_decay", weight_decay));
      JUST(attrs.SetAttr<float>("beta1", beta1));
      JUST(attrs.SetAttr<float>("beta2", beta2));
      JUST(attrs.SetAttr<float>("epsilon", epsilon));
      JUST(attrs.SetAttr<bool>("do_bias_correction", do_bias_correction));
      JUST(attrs.SetAttr<int64_t>("line_size", line_size));
      JUST(attrs.SetAttr<int64_t>("embedding_size", embedding_size));
      return AttrMap(attrs);
    }
  };

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& num_unique_ids,
                           const std::shared_ptr<one::Tensor>& unique_embeddings,
                           const std::shared_ptr<one::Tensor>& embedding_grad,
                           const std::shared_ptr<one::Tensor>& learning_rate,
                           const std::shared_ptr<one::Tensor>& down_scale_by_tensor,
                           const std::shared_ptr<one::Tensor>& skip_if,
                           const Optional<one::Tensor>& bias_correction1,
                           const Optional<one::Tensor>& bias_correction2, const double scale,
                           const float weight_decay, const float beta1, const float beta2,
                           const float epsilon, const bool do_bias_correction,
                           const int64_t line_size, const int64_t embedding_size) const {
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(OneEmbeddingAdamUpdate);
    const auto attrs = *JUST(GetAttrs(scale, weight_decay, beta1, beta2, epsilon,
                                      do_bias_correction, line_size, embedding_size));
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

  struct OneEmbeddingAdagradUpdate {
    Maybe<AttrMap> operator()(const double scale, const float weight_decay, const float lr_decay,
                              const float epsilon, const int64_t line_size,
                              const int64_t embedding_size) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<double>("scale", scale));
      JUST(attrs.SetAttr<float>("weight_decay", weight_decay));
      JUST(attrs.SetAttr<float>("lr_decay", lr_decay));
      JUST(attrs.SetAttr<float>("epsilon", epsilon));
      JUST(attrs.SetAttr<int64_t>("line_size", line_size));
      JUST(attrs.SetAttr<int64_t>("embedding_size", embedding_size));
      return AttrMap(attrs);
    }
  };

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& num_unique_ids,
                           const std::shared_ptr<one::Tensor>& unique_embeddings,
                           const std::shared_ptr<one::Tensor>& embedding_grad,
                           const std::shared_ptr<one::Tensor>& learning_rate,
                           const std::shared_ptr<one::Tensor>& down_scale_by_tensor,
                           const std::shared_ptr<one::Tensor>& skip_if,
                           const std::shared_ptr<one::Tensor>& train_step, const double scale,
                           const float weight_decay, const float lr_decay, const float epsilon,
                           const int64_t line_size, const int64_t embedding_size) const {
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(OneEmbeddingAdagradUpdate);
    const auto attrs =
        *JUST(GetAttrs(scale, weight_decay, lr_decay, epsilon, line_size, embedding_size));
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

  struct OneEmbeddingFtrlUpdate {
    Maybe<AttrMap> operator()(const double scale, const float weight_decay, const float lr_power,
                              const float lambda1, const float lambda2, const float beta,
                              const int64_t line_size, const int64_t embedding_size) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<double>("scale", scale));
      JUST(attrs.SetAttr<float>("weight_decay", weight_decay));
      JUST(attrs.SetAttr<float>("lr_power", lr_power));
      JUST(attrs.SetAttr<float>("lambda1", lambda1));
      JUST(attrs.SetAttr<float>("lambda2", lambda2));
      JUST(attrs.SetAttr<float>("beta", beta));
      JUST(attrs.SetAttr<int64_t>("line_size", line_size));
      JUST(attrs.SetAttr<int64_t>("embedding_size", embedding_size));
      return AttrMap(attrs);
    }
  };

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& num_unique_ids,
                           const std::shared_ptr<one::Tensor>& unique_embeddings,
                           const std::shared_ptr<one::Tensor>& embedding_grad,
                           const std::shared_ptr<one::Tensor>& learning_rate,
                           const std::shared_ptr<one::Tensor>& down_scale_by_tensor,
                           const std::shared_ptr<one::Tensor>& skip_if, const double scale,
                           const float weight_decay, const float lr_power, const float lambda1,
                           const float lambda2, const float beta, const int64_t line_size,
                           const int64_t embedding_size) const {
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(OneEmbeddingFtrlUpdate);
    const auto attrs = *JUST(
        GetAttrs(scale, weight_decay, lr_power, lambda1, lambda2, beta, line_size, embedding_size));
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

class MultiTensorSgdUpdateFunctor {
 public:
  MultiTensorSgdUpdateFunctor() {
    // This functor is just for unittest
    op_.resize(kMaxInputCount /*the maximum number of inputs*/);
    for (int n = 0; n < op_.size(); ++n) {
      op_[n] = CHECK_JUST(one::OpBuilder("multi_tensor_sgd_update")
                              .Input("model", n + 1)
                              .Input("model_diff", n + 1)
                              .Input("learning_rate")
                              .Build());
    }
  }

  struct MultiTensorSgdUpdate {
    Maybe<AttrMap> operator()(double scale, float weight_decay) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<double>("scale", scale));
      JUST(attrs.SetAttr<float>("weight_decay", weight_decay));
      return AttrMap(attrs);
    }
  };

  Maybe<void> operator()(const TensorTuple& model, const TensorTuple& model_diff,
                         const std::shared_ptr<one::Tensor>& learning_rate, const double& scale,
                         const float& weight_decay) const {
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(MultiTensorSgdUpdate);
    const auto attrs = *JUST(GetAttrs(scale, weight_decay));
    const int64_t weight_size = model.size();
    for (int i = 0; i < weight_size; i += kMaxInputCount) {
      size_t size = (i + kMaxInputCount) < weight_size ? kMaxInputCount : weight_size - i;
      TensorTuple input(2 * size + 1);
      std::copy(model.begin() + i, model.begin() + i + size, input.begin());
      std::copy(model_diff.begin() + i, model_diff.begin() + size, input.begin() + size);
      input[2 * size] = learning_rate;
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
    // This functor is just for unittest
    op_.resize(kMaxInputCount /*the maximum number of inputs*/);
    for (int n = 0; n < op_.size(); ++n) {
      op_[n] = CHECK_JUST(one::OpBuilder("multi_tensor_adam_update")
                              .Input("model", n + 1)
                              .Input("model_diff", n + 1)
                              .Input("m", n + 1)
                              .Input("v", n + 1)
                              .Input("learning_rate")
                              .Build());
    }
  }

  struct MultiTensorAdamUpdate {
    Maybe<AttrMap> operator()(const float& beta1, const float& beta2,
                              const float& bias_correction1_val, const float& bias_correction2_val,
                              const bool& do_bias_correction, const double& scale,
                              const float& weight_decay) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<double>("scale", scale));
      JUST(attrs.SetAttr<float>("weight_decay", weight_decay));
      JUST(attrs.SetAttr<float>("beta1", beta1));
      JUST(attrs.SetAttr<float>("beta2", beta2));
      JUST(attrs.SetAttr<float>("bias_correction1_val", bias_correction1_val));
      JUST(attrs.SetAttr<float>("bias_correction2_val", bias_correction2_val));
      JUST(attrs.SetAttr<bool>("do_bias_correction", do_bias_correction));
      return AttrMap(attrs);
    }
  };

  Maybe<void> operator()(const TensorTuple& model, const TensorTuple& model_diff,
                         const TensorTuple& m, const TensorTuple& v,
                         const std::shared_ptr<one::Tensor>& learning_rate, const float& beta1,
                         const float& beta2, const float& bias_correction1_val,
                         const float& bias_correction2_val, const bool& do_bias_correction,
                         const double& scale, const float& weight_decay) const {
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(MultiTensorAdamUpdate);
    const auto attrs = *JUST(GetAttrs(beta1, beta2, bias_correction1_val, bias_correction2_val,
                                      do_bias_correction, scale, weight_decay));
    const int64_t weight_size = model.size();

    for (int i = 0; i < weight_size; i += kMaxInputCount) {
      size_t size = (i + kMaxInputCount) < weight_size ? kMaxInputCount : weight_size - i;
      TensorTuple input(4 * size + 1);
      std::copy(model.begin() + i, model.begin() + i + size, input.begin());
      std::copy(model_diff.begin() + i, model_diff.begin() + i + size, input.begin() + size);
      std::copy(m.begin() + i, m.begin() + i + size, input.begin() + 2 * size);
      std::copy(v.begin() + i, v.begin() + i + size, input.begin() + 3 * size);
      input[4 * size] = learning_rate;
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
  m.add_functor<impl::FusedMatmulBiasAddReluDropoutFunctor>("FusedMatmulBiasAddReluDropout");
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
  m.add_functor<impl::NLLLossFunctor>("NLLLoss");
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
  m.add_functor<impl::FusedCrossFeatureInteractionFunctor>("FusedCrossFeatureInteraction");
  m.add_functor<impl::OneEmbeddingIdShuffleFunctor>("OneEmbeddingIdShuffle");
  m.add_functor<impl::OneEmbeddingEmbeddingShuffleFunctor>("OneEmbeddingEmbeddingShuffle");
  m.add_functor<impl::OneEmbeddingEmbeddingGradientShuffleFunctor>(
      "OneEmbeddingEmbeddingGradientShuffle");
  m.add_functor<impl::OneEmbeddingLookupFunctor>("OneEmbeddingLookup");
  m.add_functor<impl::OneEmbeddingUniqueKeyValuePairFunctor>("OneEmbeddingUniqueKeyValuePair");
  m.add_functor<impl::NormalFunctor>("Normal");
  m.add_functor<impl::Normal2Functor>("Normal2");
  m.add_functor<impl::GlobalNormalFunctor>("GlobalNormal");
  m.add_functor<impl::GlobalNormal2Functor>("GlobalNormal2");
  m.add_functor<impl::OneEmbeddingSgdUpdateFunctor>("OneEmbeddingSgdUpdate");
  m.add_functor<impl::OneEmbeddingAdamUpdateFunctor>("OneEmbeddingAdamUpdate");
  m.add_functor<impl::OneEmbeddingAdagradUpdateFunctor>("OneEmbeddingAdagradUpdate");
  m.add_functor<impl::OneEmbeddingFtrlUpdateFunctor>("OneEmbeddingFtrlUpdate");
  m.add_functor<impl::RocAucScoreFunctor>("RocAucScore");
  m.add_functor<impl::MultiTensorSgdUpdateFunctor>("MultiTensorSgdUpdate");
  m.add_functor<impl::MultiTensorAdamUpdateFunctor>("MultiTensorAdamUpdate");
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow
