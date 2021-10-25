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

#include "oneflow/core/framework/op_expr_helper.h"

#include "oneflow/core/framework/id_util.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_builder.h"

namespace oneflow {
namespace op_expr_helper {

static std::string UniqueOpName(const std::string& prefix) {
  return *CHECK_JUST(UniqueStr(prefix));
}

Maybe<one::UserOpExpr> AddNOp(int32_t n) { return AddNOp(n, UniqueOpName("add_n")); }
Maybe<one::UserOpExpr> AddNOp(int32_t n, const std::string& name) {
  return one::OpBuilder("add_n", name).Input("in", n).Output("out").Build();
}

Maybe<one::UserOpExpr> AddOp() { return AddOp(UniqueOpName("add_n")); }
Maybe<one::UserOpExpr> AddOp(const std::string& name) { return AddNOp(2, name); }

Maybe<one::UserOpExpr> ZeroLikeOp() { return ZeroLikeOp(UniqueOpName("zero_like")); }
Maybe<one::UserOpExpr> ZeroLikeOp(const std::string& name) {
  return one::OpBuilder("zero_like", name).Input("like").Output("out").Build();
}

Maybe<one::UserOpExpr> OnesLikeOp() { return OnesLikeOp(UniqueOpName("ones_like")); }
Maybe<one::UserOpExpr> OnesLikeOp(const std::string& name) {
  return one::OpBuilder("ones_like", name).Input("like").Output("out").Build();
}

#define DEFINE_FLOATING_CONSTATNT_OP(cpp_type, data_type)                        \
  template<>                                                                     \
  Maybe<one::UserOpExpr> ConstantOp(const Shape& shape, const cpp_type& value,   \
                                    const std::string& name) {                   \
    return one::OpBuilder("constant", name)                                      \
        .Output("out")                                                           \
        .Attr<double>("floating_value", value)                                   \
        .Attr<int64_t>("integer_value", 0)                                       \
        .Attr<bool>("is_floating_value", true)                                   \
        .Attr<DataType>("dtype", data_type)                                      \
        .Attr<Shape>("shape", shape)                                             \
        .Build();                                                                \
  }                                                                              \
  template<>                                                                     \
  Maybe<one::UserOpExpr> ConstantOp(const Shape& shape, const cpp_type& value) { \
    return ConstantOp(shape, value, UniqueOpName("constant"));                   \
  }
OF_PP_FOR_EACH_TUPLE(DEFINE_FLOATING_CONSTATNT_OP, FLOATING_DATA_TYPE_SEQ);
#undef DEFINE_FLOATING_CONSTATNT_OP

#define DEFINE_INTEGER_CONSTATNT_OP(cpp_type, data_type)                         \
  template<>                                                                     \
  Maybe<one::UserOpExpr> ConstantOp(const Shape& shape, const cpp_type& value,   \
                                    const std::string& name) {                   \
    return one::OpBuilder("constant", name)                                      \
        .Output("out")                                                           \
        .Attr<double>("floating_value", 0.f)                                     \
        .Attr<int64_t>("integer_value", value)                                   \
        .Attr<bool>("is_floating_value", false)                                  \
        .Attr<DataType>("dtype", data_type)                                      \
        .Attr<Shape>("shape", shape)                                             \
        .Build();                                                                \
  }                                                                              \
  template<>                                                                     \
  Maybe<one::UserOpExpr> ConstantOp(const Shape& shape, const cpp_type& value) { \
    return ConstantOp(shape, value, UniqueOpName("constant"));                   \
  }
OF_PP_FOR_EACH_TUPLE(DEFINE_INTEGER_CONSTATNT_OP, INT_DATA_TYPE_SEQ)
#undef DEFINE_INTEGER_CONSTATNT_OP

Maybe<one::UserOpExpr> ZerosOp(const Shape& shape, const DataType& dtype) {
  return ZerosOp(shape, dtype, UniqueOpName("constant"));
}
Maybe<one::UserOpExpr> ZerosOp(const Shape& shape, const DataType& dtype, const std::string& name) {
  switch (dtype) {
#define CONSTANT_DATA_TYPE_CASE(cpp_type, data_type) \
  case data_type: return ConstantOp(shape, (cpp_type)0, name);
    OF_PP_FOR_EACH_TUPLE(CONSTANT_DATA_TYPE_CASE, FLOATING_DATA_TYPE_SEQ INT_DATA_TYPE_SEQ);
#undef CONSTANT_DATA_TYPE_CASE
    default: UNIMPLEMENTED_THEN_RETURN();
  }
}

Maybe<one::UserOpExpr> OnesOp(const Shape& shape, const DataType& dtype) {
  return OnesOp(shape, dtype, UniqueOpName("constant"));
}
Maybe<one::UserOpExpr> OnesOp(const Shape& shape, const DataType& dtype, const std::string& name) {
  switch (dtype) {
#define CONSTANT_DATA_TYPE_CASE(cpp_type, data_type) \
  case data_type: return ConstantOp(shape, (cpp_type)1, name);
    OF_PP_FOR_EACH_TUPLE(CONSTANT_DATA_TYPE_CASE, FLOATING_DATA_TYPE_SEQ INT_DATA_TYPE_SEQ);
#undef CONSTANT_DATA_TYPE_CASE
    default: UNIMPLEMENTED_THEN_RETURN();
  }
}

Maybe<one::UserOpExpr> EmptyOp(const Shape& shape, const DataType& dtype) {
  return EmptyOp(shape, dtype, UniqueOpName("empty"));
}
Maybe<one::UserOpExpr> EmptyOp(const Shape& shape, const DataType& dtype, const std::string& name) {
  return one::OpBuilder("empty", name)
      .Output("out")
      .Attr<DataType>("dtype", dtype)
      .Attr<Shape>("shape", shape)
      .Build();
}

Maybe<one::UserOpExpr> IdentityOp() { return IdentityOp(UniqueOpName("identity")); }
Maybe<one::UserOpExpr> IdentityOp(const std::string& name) {
  return one::OpBuilder("identity", name).Input("in").Output("out").Build();
}

Maybe<one::UserOpExpr> ReshapeOp(const Shape& shape) {
  return ReshapeOp(shape, UniqueOpName("reshape"));
}
Maybe<one::UserOpExpr> ReshapeOp(const Shape& shape, const std::string& name) {
  return one::OpBuilder("reshape", name)
      .Input("in")
      .Output("out")
      .Attr<Shape>("shape", shape)
      .Build();
}

Maybe<one::UserOpExpr> ReshapeLikeOp() { return ReshapeLikeOp(UniqueOpName("reshape_like")); }
Maybe<one::UserOpExpr> ReshapeLikeOp(const std::string& name) {
  return one::OpBuilder("reshape_like", name).Input("in").Input("like").Output("out").Build();
}

Maybe<one::UserOpExpr> ReduceSumOp(const std::vector<int32_t>& reduce_axes, const bool& keepdims) {
  return ReduceSumOp(reduce_axes, keepdims, UniqueOpName("reduce_sum"));
}
Maybe<one::UserOpExpr> ReduceSumOp(const std::vector<int32_t>& reduce_axes, const bool& keepdims,
                                   const std::string& name) {
  return one::OpBuilder("reduce_sum", name)
      .Input("input_tensor")
      .Output("output_tensor")
      .Attr<std::vector<int32_t>>("axis", reduce_axes)
      .Attr<bool>("keepdims", keepdims)
      .Build();
}

Maybe<one::UserOpExpr> ReduceSumLikeOp(const std::vector<int32_t>& axis) {
  return ReduceSumLikeOp(axis, UniqueOpName("reduce_sum_like"));
}
Maybe<one::UserOpExpr> ReduceSumLikeOp(const std::vector<int32_t>& axis, const std::string& name) {
  return one::OpBuilder("reduce_sum_like", name)
      .Input("x")
      .Input("like")
      .Attr<std::vector<int32_t>>("axis", axis)
      .Output("y")
      .Build();
}

Maybe<one::UserOpExpr> ScalarPowOp(const double& exponent) {
  return ScalarPowOp(exponent, UniqueOpName("scalar_pow"));
}

Maybe<one::UserOpExpr> ScalarPowOp(const double& exponent, const std::string& name) {
  return one::OpBuilder("scalar_pow", name)
      .Input("in")
      .Attr<double>("exponent", exponent)
      .Output("out")
      .Build();
}

template<>
Maybe<one::UserOpExpr> ScalarMulOp(const float& scalar, const std::string& name) {
  return one::OpBuilder("scalar_mul", name)
      .Input("in")
      .Attr<bool>("has_int_operand", false)
      .Attr<bool>("has_float_operand", true)
      .Attr<int64_t>("int_operand", 0)
      .Attr<double>("float_operand", scalar)
      .Output("out")
      .Build();
}

template<>
Maybe<one::UserOpExpr> ScalarMulOp(const float& scalar) {
  return ScalarMulOp<float>(scalar, UniqueOpName("scalar_mul"));
}

template<>
Maybe<one::UserOpExpr> ScalarMulOp(const int32_t& scalar, const std::string& name) {
  return one::OpBuilder("scalar_mul", name)
      .Input("in")
      .Attr<bool>("has_int_operand", true)
      .Attr<bool>("has_float_operand", false)
      .Attr<int64_t>("int_operand", scalar)
      .Attr<double>("float_operand", 0.f)
      .Output("out")
      .Build();
}

template<>
Maybe<one::UserOpExpr> ScalarMulOp(const int32_t& scalar) {
  return ScalarMulOp<int32_t>(scalar, UniqueOpName("scalar_mul"));
}

template<>
Maybe<one::UserOpExpr> ScalarAddOp(const float& scalar, const std::string& name) {
  return one::OpBuilder("scalar_add", name)
      .Input("in")
      .Attr<bool>("has_int_operand", false)
      .Attr<bool>("has_float_operand", true)
      .Attr<int64_t>("int_operand", 0)
      .Attr<double>("float_operand", scalar)
      .Output("out")
      .Build();
}

template<>
Maybe<one::UserOpExpr> ScalarAddOp(const float& scalar) {
  return ScalarAddOp<float>(scalar, UniqueOpName("scalar_add"));
}

template<>
Maybe<one::UserOpExpr> ScalarAddOp(const int32_t& scalar, const std::string& name) {
  return one::OpBuilder("scalar_add", name)
      .Input("in")
      .Attr<bool>("has_int_operand", true)
      .Attr<bool>("has_float_operand", false)
      .Attr<int64_t>("int_operand", scalar)
      .Attr<double>("float_operand", 0.f)
      .Output("out")
      .Build();
}

template<>
Maybe<one::UserOpExpr> ScalarAddOp(const int32_t& scalar) {
  return ScalarAddOp<int32_t>(scalar, UniqueOpName("scalar_add"));
}

Maybe<one::UserOpExpr> RsqrtOp() { return RsqrtOp(UniqueOpName("rsqrt")); }
Maybe<one::UserOpExpr> RsqrtOp(const std::string& name) {
  return one::OpBuilder("rsqrt", name).Input("x").Output("y").Build();
}

Maybe<one::UserOpExpr> BroadcastAddOp() { return BroadcastAddOp(UniqueOpName("broadcast_add")); }
Maybe<one::UserOpExpr> BroadcastAddOp(const std::string& name) {
  return one::OpBuilder("broadcast_add", name).Input("x").Input("y").Output("z").Build();
}

Maybe<one::UserOpExpr> BroadcastSubOp() { return BroadcastSubOp(UniqueOpName("broadcast_sub")); }
Maybe<one::UserOpExpr> BroadcastSubOp(const std::string& name) {
  return one::OpBuilder("broadcast_sub", name).Input("x").Input("y").Output("z").Build();
}

Maybe<one::UserOpExpr> BroadcastMulOp() { return BroadcastMulOp(UniqueOpName("broadcast_mul")); }
Maybe<one::UserOpExpr> BroadcastMulOp(const std::string& name) {
  return one::OpBuilder("broadcast_mul", name).Input("x").Input("y").Output("z").Build();
}

Maybe<one::UserOpExpr> BroadcastDivOp() { return BroadcastDivOp(UniqueOpName("broadcast_div")); }
Maybe<one::UserOpExpr> BroadcastDivOp(const std::string& name) {
  return one::OpBuilder("broadcast_div", name).Input("x").Input("y").Output("z").Build();
}

Maybe<one::UserOpExpr> BroadcastLikeOp(const std::vector<int32_t>& axis) {
  return BroadcastLikeOp(axis, UniqueOpName("broadcast_like"));
}
Maybe<one::UserOpExpr> BroadcastLikeOp(const std::vector<int32_t>& axis, const std::string& name) {
  return one::OpBuilder("broadcast_like", name)
      .Input("x")
      .Input("like")
      .Output("y")
      .Attr<std::vector<int32_t>>("broadcast_axes", axis)
      .Build();
}

Maybe<one::UserOpExpr> BroadcastEqualOp() {
  return BroadcastEqualOp(UniqueOpName("broadcast_equal"));
}
Maybe<one::UserOpExpr> BroadcastEqualOp(const std::string& name) {
  return one::OpBuilder("broadcast_equal", name).Input("x").Input("y").Output("z").Build();
}

Maybe<one::UserOpExpr> CastOp(const DataType& to_type) {
  return CastOp(to_type, UniqueOpName("cast"));
}
Maybe<one::UserOpExpr> CastOp(const DataType& to_type, const std::string& name) {
  return one::OpBuilder("cast", name)
      .Input("in")
      .Output("out")
      .Attr<DataType>("dtype", to_type)
      .Build();
}

Maybe<one::UserOpExpr> CopyOp(const std::string& device_type, const int64_t device_id) {
  return CopyOp(device_type, device_id, UniqueOpName("copy"));
}
Maybe<one::UserOpExpr> CopyOp(const std::string& device_type, const int64_t device_id,
                              const std::string& name) {
  return one::OpBuilder("copy", name)
      .Input("in")
      .Output("out")
      .Attr<std::string>("device_type", device_type)
      .Attr<int64_t>("device_id", device_id)
      .Build();
}

Maybe<one::UserOpExpr> CastLikeOp() { return CastLikeOp(UniqueOpName("cast_like")); }
Maybe<one::UserOpExpr> CastLikeOp(const std::string& name) {
  return one::OpBuilder("cast_like", name).Input("in").Input("dtype_like").Output("out").Build();
}

Maybe<one::UserOpExpr> NormalizationGradOp(const int32_t& axis, const float& epsilon) {
  return NormalizationGradOp(axis, epsilon, UniqueOpName("normalization_grad"));
}

Maybe<one::UserOpExpr> NormalizationGradOp(const int32_t& axis, const float& epsilon,
                                           const std::string& name) {
  return one::OpBuilder("normalization_grad", name)
      .Input("x")
      .Input("dy")
      .Input("gamma")
      .Input("mean")
      .Input("inv_variance")
      .Output("dx")
      .Output("gamma_diff")
      .Output("beta_diff")
      .Attr<int32_t>("axis", axis)
      .Attr<float>("epsilon", epsilon)
      .Build();
}

Maybe<one::UserOpExpr> BroadcastDivGradOp() {
  return BroadcastDivGradOp(UniqueOpName("broadcast_div_grad"));
}
Maybe<one::UserOpExpr> BroadcastDivGradOp(const std::string& name) {
  return one::OpBuilder("broadcast_div_grad", name)
      .Input("dz")
      .Input("z")
      .Input("y")
      .Output("dy")
      .Build();
}

Maybe<one::UserOpExpr> ConcatOp(const int& n, const int64_t& axis, const int64_t& max_dim_size) {
  return ConcatOp(n, axis, max_dim_size, UniqueOpName("concat"));
}

Maybe<one::UserOpExpr> ConcatOp(const int& n, const int64_t& axis, const int64_t& max_dim_size,
                                const std::string& name) {
  return one::OpBuilder("concat", name)
      .Input("in", n)
      .Output("out")
      .Attr<int64_t>("axis", axis)
      .Attr<int64_t>("max_dim_size", max_dim_size)
      .Build();
}

Maybe<one::UserOpExpr> ScalarAddByTensorOp() {
  return ScalarAddByTensorOp(UniqueOpName("scalar_add_by_tensor"));
}
Maybe<one::UserOpExpr> ScalarAddByTensorOp(const std::string& name) {
  return one::OpBuilder("scalar_add_by_tensor", name)
      .Input("x")
      .Input("scalar")
      .Output("y")
      .Build();
}

Maybe<one::UserOpExpr> ScalarSubByTensorOp() {
  return ScalarSubByTensorOp(UniqueOpName("scalar_sub_by_tensor"));
}
Maybe<one::UserOpExpr> ScalarSubByTensorOp(const std::string& name) {
  return one::OpBuilder("scalar_sub_by_tensor", name)
      .Input("x")
      .Input("scalar")
      .Output("y")
      .Build();
}

Maybe<one::UserOpExpr> ScalarMulByTensorOp() {
  return ScalarMulByTensorOp(UniqueOpName("scalar_mul_by_tensor"));
}
Maybe<one::UserOpExpr> ScalarMulByTensorOp(const std::string& name) {
  return one::OpBuilder("scalar_mul_by_tensor", name)
      .Input("x")
      .Input("scalar")
      .Output("y")
      .Build();
}

Maybe<one::UserOpExpr> ScalarDivByTensorOp() {
  return ScalarDivByTensorOp(UniqueOpName("scalar_div_by_tensor"));
}
Maybe<one::UserOpExpr> ScalarDivByTensorOp(const std::string& name) {
  return one::OpBuilder("scalar_div_by_tensor", name)
      .Input("x")
      .Input("scalar")
      .Output("y")
      .Build();
}

Maybe<one::UserOpExpr> MultiplyOp() { return MultiplyOp(UniqueOpName("multiply")); }
Maybe<one::UserOpExpr> MultiplyOp(const std::string& name) {
  return one::OpBuilder("multiply", name).Input("x").Input("y").Output("out").Build();
}

Maybe<one::UserOpExpr> ConvNdOp(const int& filters, const std::vector<int32_t>& kernel_size,
                                const std::vector<int32_t>& strides,
                                const std::vector<int32_t>& padding_before,
                                const std::vector<int32_t>& dilation_rate, const int& groups,
                                const std::string& data_format) {
  return ConvNdOp(filters, kernel_size, strides, padding_before, dilation_rate, groups, data_format,
                  UniqueOpName("conv_nd"));
}
Maybe<one::UserOpExpr> ConvNdOp(const int& filters, const std::vector<int32_t>& kernel_size,
                                const std::vector<int32_t>& strides,
                                const std::vector<int32_t>& padding_before,
                                const std::vector<int32_t>& dilation_rate, const int& groups,
                                const std::string& data_format, const std::string& name) {
  int ndims = kernel_size.size();
  std::string op_type_name = "conv" + std::to_string(ndims) + "d";
  return one::OpBuilder(op_type_name, name)
      .Input("in")
      .Input("weight")
      .Output("out")
      .Attr<int32_t>("filters", filters)
      .Attr<std::vector<int32_t>>("kernel_size", kernel_size)
      .Attr<std::vector<int32_t>>("strides", strides)
      .Attr<std::vector<int32_t>>("padding_before", padding_before)
      .Attr<std::vector<int32_t>>("dilation_rate", dilation_rate)
      .Attr<int32_t>("groups", groups)
      .Attr<std::string>("data_format", data_format)
      .Build();
}

Maybe<one::UserOpExpr> ConvNdFilterGradOp(const std::vector<int32_t>& kernel_size,
                                          const std::vector<int32_t>& strides,
                                          const std::vector<int32_t>& padding_before,
                                          const std::vector<int32_t>& dilation_rate,
                                          const int& groups, const std::string& data_format) {
  return ConvNdFilterGradOp(kernel_size, strides, padding_before, dilation_rate, groups,
                            data_format, UniqueOpName("conv_filter_grad"));
}
Maybe<one::UserOpExpr> ConvNdFilterGradOp(const std::vector<int32_t>& kernel_size,
                                          const std::vector<int32_t>& strides,
                                          const std::vector<int32_t>& padding_before,
                                          const std::vector<int32_t>& dilation_rate,
                                          const int& groups, const std::string& data_format,
                                          const std::string& name) {
  return one::OpBuilder("conv_filter_grad", name)
      .Input("dy")
      .Input("x")
      .Output("filter_diff")
      .Attr<int32_t>("num_spatial_dims", kernel_size.size())
      .Attr<std::vector<int32_t>>("kernel_size", kernel_size)
      .Attr<std::vector<int32_t>>("strides", strides)
      .Attr<std::vector<int32_t>>("padding_before", padding_before)
      .Attr<std::vector<int32_t>>("dilation_rate", dilation_rate)
      .Attr<int32_t>("groups", groups)
      .Attr<std::string>("data_format", data_format)
      .Build();
}

Maybe<one::UserOpExpr> ConvNdDataGradOp(const std::vector<int32_t>& kernel_size,
                                        const std::vector<int32_t>& strides,
                                        const std::vector<int32_t>& padding_before,
                                        const std::vector<int32_t>& dilation_rate,
                                        const int& groups, const std::string& data_format) {
  return ConvNdDataGradOp(kernel_size, strides, padding_before, dilation_rate, groups, data_format,
                          UniqueOpName("conv_data_grad"));
}
Maybe<one::UserOpExpr> ConvNdDataGradOp(const std::vector<int32_t>& kernel_size,
                                        const std::vector<int32_t>& strides,
                                        const std::vector<int32_t>& padding_before,
                                        const std::vector<int32_t>& dilation_rate,
                                        const int& groups, const std::string& data_format,
                                        const std::string& name) {
  return one::OpBuilder("conv_data_grad", name)
      .Input("dy")
      .Input("filter")
      .Input("x_like")
      .Output("dx")
      .Attr<int32_t>("num_spatial_dims", kernel_size.size())
      .Attr<std::vector<int32_t>>("padding_before", padding_before)
      .Attr<std::string>("data_format", data_format)
      .Attr<std::vector<int32_t>>("kernel_size", kernel_size)
      .Attr<std::vector<int32_t>>("strides", strides)
      .Attr<std::vector<int32_t>>("dilation_rate", dilation_rate)
      .Attr<int32_t>("groups", groups)
      .Build();
}

Maybe<one::UserOpExpr> CTCLossGradOp(const int32_t& blank, const bool& zero_infinity) {
  return CTCLossGradOp(blank, zero_infinity, UniqueOpName("ctc_loss_grad"));
}
Maybe<one::UserOpExpr> CTCLossGradOp(const int32_t& blank, const bool& zero_infinity,
                                     const std::string& name) {
  return one::OpBuilder("ctc_loss_grad", name)
      .Input("grad_out")
      .Input("log_probs")
      .Input("targets")
      .Input("input_lengths")
      .Input("target_lengths")
      .Input("loss")
      .Input("alpha")
      .Output("grad")
      .Attr<int32_t>("blank", blank)
      .Attr<bool>("zero_infinity", zero_infinity)
      .Build();
}

Maybe<one::UserOpExpr> SparseSoftmaxCrossEntropyGradOp(const int64_t& depth) {
  return SparseSoftmaxCrossEntropyGradOp(depth, UniqueOpName("sparse_softmax_cross_entropy"));
}
Maybe<one::UserOpExpr> SparseSoftmaxCrossEntropyGradOp(const int64_t& depth,
                                                       const std::string& name) {
  return one::OpBuilder("sparse_softmax_cross_entropy_grad", name)
      .Input("prob")
      .Input("label")
      .Input("dy")
      .Output("prediction_diff")
      .Attr<int64_t>("depth", depth)
      .Build();
}

Maybe<one::UserOpExpr> SparseSoftmaxCrossEntropyMsGradOp(const int64_t& depth) {
  return SparseSoftmaxCrossEntropyMsGradOp(depth, UniqueOpName("sparse_softmax_cross_entropy_ms"));
}
Maybe<one::UserOpExpr> SparseSoftmaxCrossEntropyMsGradOp(const int64_t& depth,
                                                         const std::string& name) {
  return one::OpBuilder("sparse_softmax_cross_entropy_ms_grad", name)
      .Input("prob")
      .Input("label")
      .Input("dy")
      .Output("prediction_diff")
      .Attr<int64_t>("depth", depth)
      .Build();
}

Maybe<one::UserOpExpr> UpsampleGradOp(const float& height_scale, const float& width_scale,
                                      const bool& align_corners, const std::string& data_format,
                                      const std::string& interpolation) {
  return UpsampleGradOp(height_scale, width_scale, align_corners, data_format, interpolation,
                        UniqueOpName("upsample_grad"));
}
Maybe<one::UserOpExpr> UpsampleGradOp(const float& height_scale, const float& width_scale,
                                      const bool& align_corners, const std::string& data_format,
                                      const std::string& interpolation, const std::string& name) {
  return one::OpBuilder("upsample_grad", name)
      .Input("dy")
      .Input("x")
      .Output("dx")
      .Attr<float>("height_scale", height_scale)
      .Attr<float>("width_scale", width_scale)
      .Attr<bool>("align_corners", align_corners)
      .Attr<std::string>("data_format", data_format)
      .Attr<std::string>("interpolation", interpolation)
      .Build();
}

Maybe<one::UserOpExpr> DimScatterAddLikeOp(const int32_t dim) {
  return DimScatterAddLikeOp(dim, UniqueOpName("dim_scatter_add_like"));
}
Maybe<one::UserOpExpr> DimScatterAddLikeOp(const int32_t dim, const std::string& name) {
  return one::OpBuilder("dim_scatter_add_like", name)
      .Input("like")
      .Input("index")
      .Input("src")
      .Output("output")
      .Attr<int32_t>("dim", dim)
      .Build();
}

Maybe<one::UserOpExpr> TransposeOp(const std::vector<int32_t>& perm) {
  return TransposeOp(perm, UniqueOpName("transpose"));
}
Maybe<one::UserOpExpr> TransposeOp(const std::vector<int32_t>& perm, const std::string& name) {
  return one::OpBuilder("transpose", name)
      .Input("input")
      .Output("output")
      .Attr<std::vector<int32_t>>("perm", perm)
      .Build();
}

Maybe<one::UserOpExpr> SplitLikeOp(const int n, const int64_t axis) {
  return SplitLikeOp(n, axis, UniqueOpName("split_like"));
}
Maybe<one::UserOpExpr> SplitLikeOp(const int n, const int64_t axis, const std::string& name) {
  return one::OpBuilder("split_like", name)
      .Input("in")
      .Input("like", n)
      .Output("out", n)
      .Attr<int64_t>("axis", axis)
      .Build();
}

Maybe<one::UserOpExpr> WhereOp() { return WhereOp(UniqueOpName("where")); }
Maybe<one::UserOpExpr> WhereOp(const std::string& name) {
  return one::OpBuilder("where", name)
      .Input("condition")
      .Input("x")
      .Input("y")
      .Output("out")
      .Build();
}

Maybe<one::UserOpExpr> ExpandGradOp(const std::vector<int32_t>& logical_out_shape,
                                    const std::vector<int32_t>& logical_expand_shape) {
  return ExpandGradOp(logical_out_shape, logical_expand_shape, UniqueOpName("expand_grad"));
}
Maybe<one::UserOpExpr> ExpandGradOp(const std::vector<int32_t>& logical_out_shape,
                                    const std::vector<int32_t>& logical_expand_shape,
                                    const std::string& name) {
  return one::OpBuilder("expand_grad", name)
      .Input("in")
      .Output("out")
      .Attr<std::vector<int32_t>>("logical_out_shape", logical_out_shape)
      .Attr<std::vector<int32_t>>("logical_expand_shape", logical_expand_shape)
      .Build();
}

Maybe<one::UserOpExpr> UnaryGradOp(const std::string& unary_op_type) {
  return UnaryGradOp(unary_op_type, UniqueOpName(unary_op_type + "_grad"));
}
Maybe<one::UserOpExpr> UnaryGradOp(const std::string& unary_op_type, const std::string& name) {
  return one::OpBuilder(unary_op_type + "_grad", name).Input("x").Input("dy").Output("dx").Build();
}

Maybe<one::UserOpExpr> BinaryXGradOp(const std::string& binary_op_type) {
  return BinaryXGradOp(binary_op_type, UniqueOpName(binary_op_type + "_x_grad"));
}
Maybe<one::UserOpExpr> BinaryXGradOp(const std::string& binary_op_type, const std::string& name) {
  return one::OpBuilder(binary_op_type + "_x_grad", name)
      .Input("x")
      .Input("y")
      .Input("dz")
      .Output("dx")
      .Build();
}

Maybe<one::UserOpExpr> BinaryYGradOp(const std::string& binary_op_type) {
  return BinaryYGradOp(binary_op_type, UniqueOpName(binary_op_type + "_y_grad"));
}
Maybe<one::UserOpExpr> BinaryYGradOp(const std::string& binary_op_type, const std::string& name) {
  return one::OpBuilder(binary_op_type + "_y_grad", name)
      .Input("x")
      .Input("y")
      .Input("dz")
      .Output("dy")
      .Build();
}

#define MATMUL_SERIES_OPS(op_type_name)       \
  return one::OpBuilder(op_type_name, name)   \
      .Input("a")                             \
      .Input("b")                             \
      .Output("out")                          \
      .Attr<bool>("transpose_a", transpose_a) \
      .Attr<bool>("transpose_b", transpose_b) \
      .Attr<double>("alpha", alpha)           \
      .Build();

Maybe<one::UserOpExpr> MatmulOp(const bool& transpose_a, const bool& transpose_b,
                                const double& alpha) {
  return MatmulOp(transpose_a, transpose_b, alpha, UniqueOpName("matmul"));
}

Maybe<one::UserOpExpr> MatmulOp(const bool& transpose_a, const bool& transpose_b,
                                const double& alpha, const std::string& name) {
  MATMUL_SERIES_OPS("matmul");
}

Maybe<one::UserOpExpr> BatchMatmulOp(const bool& transpose_a, const bool& transpose_b,
                                     const double& alpha) {
  return BatchMatmulOp(transpose_a, transpose_b, alpha, UniqueOpName("batch_matmul"));
}

Maybe<one::UserOpExpr> BatchMatmulOp(const bool& transpose_a, const bool& transpose_b,
                                     const double& alpha, const std::string& name) {
  MATMUL_SERIES_OPS("batch_matmul");
}

Maybe<one::UserOpExpr> BroadcastMatmulOp(const bool& transpose_a, const bool& transpose_b,
                                         const double& alpha) {
  return BroadcastMatmulOp(transpose_a, transpose_b, alpha, UniqueOpName("broadcast_matmul"));
}

Maybe<one::UserOpExpr> BroadcastMatmulOp(const bool& transpose_a, const bool& transpose_b,
                                         const double& alpha, const std::string& name) {
  MATMUL_SERIES_OPS("broadcast_matmul");
}

#undef MATMUL_SERIES_OPS

Maybe<one::UserOpExpr> BroadcastMatmulGradBOp(const double& alpha) {
  return BroadcastMatmulGradBOp(alpha, UniqueOpName("broadcast_matmul_grad_b"));
}
Maybe<one::UserOpExpr> BroadcastMatmulGradBOp(const double& alpha, const std::string& name) {
  return one::OpBuilder("broadcast_matmul_grad_b", name)
      .Input("a")
      .Input("b")
      .Output("out")
      .Attr<double>("alpha", alpha)
      .Build();
}

Maybe<one::UserOpExpr> PoolNdGradOp(const std::string& mode, const std::string& data_format,
                                    const std::string& padding,
                                    const std::vector<int32_t>& padding_before,
                                    const std::vector<int32_t>& padding_after,
                                    const std::vector<int32_t>& pool_size,
                                    const std::vector<int32_t>& strides, const bool& ceil_mode) {
  return PoolNdGradOp(mode, data_format, padding, padding_before, padding_after, pool_size, strides,
                      ceil_mode, UniqueOpName(mode + "_pool_nd_grad"));
}

Maybe<one::UserOpExpr> PoolNdGradOp(const std::string& mode, const std::string& data_format,
                                    const std::string& padding,
                                    const std::vector<int32_t>& padding_before,
                                    const std::vector<int32_t>& padding_after,
                                    const std::vector<int32_t>& pool_size,
                                    const std::vector<int32_t>& strides, const bool& ceil_mode,
                                    const std::string& name) {
  int ndims = pool_size.size();
  std::string op_type_name = mode + "_pool_" + std::to_string(ndims) + "d_grad";
  return one::OpBuilder(op_type_name, name)
      .Input("x")
      .Input("y")
      .Input("dy")
      .Output("dx")
      .Attr<std::string>("data_format", data_format)
      .Attr<std::string>("padding", padding)
      .Attr<std::vector<int32_t>>("padding_before", padding_before)
      .Attr<std::vector<int32_t>>("padding_after", padding_after)
      .Attr<std::vector<int32_t>>("pool_size", pool_size)
      .Attr<std::vector<int32_t>>("strides", strides)
      .Attr<bool>("ceil_mode", ceil_mode)
      .Build();
}

Maybe<one::UserOpExpr> UnsortedSegmentSumLikeOp(const int64_t& axis) {
  return UnsortedSegmentSumLikeOp(axis, UniqueOpName("unsorted_segment_sum_like"));
}
Maybe<one::UserOpExpr> UnsortedSegmentSumLikeOp(const int64_t& axis, const std::string& name) {
  return one::OpBuilder("unsorted_segment_sum_like", name)
      .Input("data")
      .Input("segment_ids")
      .Input("like")
      .Output("out")
      .Attr<int64_t>("axis", axis)
      .Build();
}

Maybe<one::UserOpExpr> SoftmaxGradOp() { return SoftmaxGradOp("softmax_grad"); }

Maybe<one::UserOpExpr> SoftmaxGradOp(const std::string& name) {
  return one::OpBuilder("softmax_grad", name).Input("y").Input("dy").Output("dx").Build();
}

}  // namespace op_expr_helper
}  // namespace oneflow
