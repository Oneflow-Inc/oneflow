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
#include "oneflow/core/operator/op_attribute.cfg.h"

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
  return one::OpBuilder("zero_like", name).Input("in").Output("out").Build();
}

Maybe<one::UserOpExpr> OnesLikeOp() { return OnesLikeOp(UniqueOpName("constant_like")); }
Maybe<one::UserOpExpr> OnesLikeOp(const std::string& name) {
  AttrValue conf;
  conf.set_at_float(1.0);
  return one::OpBuilder("constant_like", name)
      .Input("like")
      .Output("out")
      .Attr("value", conf)
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
  return one::OpBuilder("reshape", name).Input("in").Output("out").Attr("shape", shape).Build();
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
      .Attr("axis", reduce_axes)
      .Attr("keepdims", keepdims)
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

template<>
Maybe<one::UserOpExpr> ScalarMulOp(const float& scalar, const std::string& name) {
  return one::OpBuilder("scalar_mul", name)
      .Input("in")
      .Attr("has_int_operand", false)
      .Attr("has_float_operand", true)
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
      .Attr("has_int_operand", true)
      .Attr("has_float_operand", false)
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
      .Attr("has_int_operand", false)
      .Attr("has_float_operand", true)
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
      .Attr("has_int_operand", true)
      .Attr("has_float_operand", false)
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

Maybe<one::UserOpExpr> CastOp(const DataType& to_type) {
  return CastOp(to_type, UniqueOpName("cast"));
}
Maybe<one::UserOpExpr> CastOp(const DataType& to_type, const std::string& name) {
  return one::OpBuilder("cast", name).Input("in").Output("out").Attr("dtype", to_type).Build();
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
      .Attr("axis", axis)
      .Attr("epsilon", epsilon)
      .Build();
}

Maybe<one::UserOpExpr> BroadcastDivGradOp() {
  return BroadcastDivGradOp(UniqueOpName("broadcast_div_grad"));
}
Maybe<one::UserOpExpr> BroadcastDivGradOp(const std::string& name) {
  return one::OpBuilder("broadcast_div_grad", name)
      .Input("dz")
      .Input("y")
      .Input("z")
      .Output("dy")
      .Build();
}

Maybe<one::UserOpExpr> LayerNormGradOp(const int64_t& begin_norm_axis, const double& epsilon) {
  return LayerNormGradOp(begin_norm_axis, epsilon, UniqueOpName("layer_norm_grad"));
}
Maybe<one::UserOpExpr> LayerNormGradOp(const int64_t& begin_norm_axis, const double& epsilon,
                                       const std::string& name) {
  return one::OpBuilder("layer_norm_grad", name)
      .Input("x")
      .Input("mean")
      .Input("inv_variance")
      .Input("dy")
      .Output("dx")
      .Attr("begin_norm_axis", begin_norm_axis)
      .Attr("epsilon", epsilon)
      .Build();
}

Maybe<one::UserOpExpr> LayerNormParamGradOp(const int64_t& begin_params_axis,
                                            const bool& has_beta_diff, const bool& has_gamma_diff,
                                            const bool& has_normalized_diff) {
  return LayerNormParamGradOp(begin_params_axis, has_beta_diff, has_gamma_diff, has_normalized_diff,
                              UniqueOpName("layer_norm_param_grad"));
}
Maybe<one::UserOpExpr> LayerNormParamGradOp(const int64_t& begin_params_axis,
                                            const bool& has_beta_diff, const bool& has_gamma_diff,
                                            const bool& has_normalized_diff,
                                            const std::string& name) {
  auto builder = one::OpBuilder("layer_norm_param_grad", name).Input("dy");
  if (has_gamma_diff || has_normalized_diff) { builder.Input("gamma"); }
  if (has_gamma_diff) { builder.Input("normalized"); }
  if (has_beta_diff) { builder.Output("beta_diff"); }
  if (has_gamma_diff) { builder.Output("gamma_diff"); }
  if (has_normalized_diff) { builder.Output("normalized_diff"); }
  if (has_beta_diff || has_gamma_diff) { builder.Output("reduce_buf"); }
  return builder.Attr("begin_params_axis", begin_params_axis).Build();
}

Maybe<one::UserOpExpr> ConcatOp(const int& n, const int64_t& axis, const int64_t& max_dim_size) {
  return ConcatOp(n, axis, max_dim_size, UniqueOpName("concat"));
}

Maybe<one::UserOpExpr> ConcatOp(const int& n, const int64_t& axis, const int64_t& max_dim_size,
                                const std::string& name) {
  return one::OpBuilder("concat", name)
      .Input("in", n)
      .Output("out")
      .Attr("axis", axis)
      .Attr("max_dim_size", max_dim_size)
      .Build();
}

Maybe<one::UserOpExpr> UnsortedBatchSegmentSumOp(const int& num_segments) {
  return UnsortedBatchSegmentSumOp(num_segments, UniqueOpName("unsorted_batch_segment_sum"));
}
Maybe<one::UserOpExpr> UnsortedBatchSegmentSumOp(const int& num_segments, const std::string& name) {
  return one::OpBuilder("unsorted_batch_segment_sum", name)
      .Input("data")
      .Input("segment_ids")
      .Output("out")
      .Attr("num_segments", num_segments)
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
      .Attr("strides", strides)
      .Attr("padding_before", padding_before)
      .Attr("dilation_rate", dilation_rate)
      .Attr("groups", groups)
      .Attr("data_format", data_format)
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
      .Attr("num_spatial_dims", (int)kernel_size.size())
      .Attr("kernel_size", kernel_size)
      .Attr("strides", strides)
      .Attr("padding_before", padding_before)
      .Attr("dilation_rate", dilation_rate)
      .Attr("groups", groups)
      .Attr("data_format", data_format)
      .Build();
}

}  // namespace op_expr_helper
}  // namespace oneflow
