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

Maybe<one::UserOpExpr> BroadcastMulOp() { return BroadcastMulOp(UniqueOpName("broadcast_mul")); }
Maybe<one::UserOpExpr> BroadcastMulOp(const std::string& name) {
  return one::OpBuilder("broadcast_mul", name).Input("x").Input("y").Output("z").Build();
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

}  // namespace op_expr_helper
}  // namespace oneflow
