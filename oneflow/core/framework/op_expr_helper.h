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

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/shape.h"

namespace oneflow {

namespace one {

class UserOpExpr;

}  // namespace one

namespace op_expr_helper {

Maybe<one::UserOpExpr> AddNOp(int32_t n);
Maybe<one::UserOpExpr> AddNOp(int32_t n, const std::string& name);

Maybe<one::UserOpExpr> AddOp();
Maybe<one::UserOpExpr> AddOp(const std::string& name);

Maybe<one::UserOpExpr> ZeroLikeOp();
Maybe<one::UserOpExpr> ZeroLikeOp(const std::string& name);

Maybe<one::UserOpExpr> OnesLikeOp();
Maybe<one::UserOpExpr> OnesLikeOp(const std::string& name);

Maybe<one::UserOpExpr> IdentityOp();
Maybe<one::UserOpExpr> IdentityOp(const std::string& name);

Maybe<one::UserOpExpr> ReshapeOp(const Shape& shape);
Maybe<one::UserOpExpr> ReshapeOp(const Shape& shape, const std::string& name);

Maybe<one::UserOpExpr> ReshapeLikeOp();
Maybe<one::UserOpExpr> ReshapeLikeOp(const std::string& name);

Maybe<one::UserOpExpr> ReduceSumOp(const std::vector<int32_t>& reduce_axes, const bool& keepdims);
Maybe<one::UserOpExpr> ReduceSumOp(const std::vector<int32_t>& reduce_axes, const bool& keepdims,
                                   const std::string& name);

Maybe<one::UserOpExpr> ReduceSumLikeOp(const std::vector<int32_t>& axis);
Maybe<one::UserOpExpr> ReduceSumLikeOp(const std::vector<int32_t>& axis, const std::string& name);

template<typename T>
Maybe<one::UserOpExpr> ScalarAddOp(const T& scalar);

template<typename T>
Maybe<one::UserOpExpr> ScalarAddOp(const T& scalar, const std::string& name);

template<typename T>
Maybe<one::UserOpExpr> ScalarMulOp(const T& scalar);

template<typename T>
Maybe<one::UserOpExpr> ScalarMulOp(const T& scalar, const std::string& name);

Maybe<one::UserOpExpr> RsqrtOp();
Maybe<one::UserOpExpr> RsqrtOp(const std::string& name);

Maybe<one::UserOpExpr> BroadcastAddOp();
Maybe<one::UserOpExpr> BroadcastAddOp(const std::string& name);

Maybe<one::UserOpExpr> BroadcastSubOp();
Maybe<one::UserOpExpr> BroadcastSubOp(const std::string& name);

Maybe<one::UserOpExpr> BroadcastMulOp();
Maybe<one::UserOpExpr> BroadcastMulOp(const std::string& name);

Maybe<one::UserOpExpr> BroadcastDivOp();
Maybe<one::UserOpExpr> BroadcastDivOp(const std::string& name);

Maybe<one::UserOpExpr> CastOp(const DataType& to_type);
Maybe<one::UserOpExpr> CastOp(const DataType& to_type, const std::string& name);

Maybe<one::UserOpExpr> NormalizationGradOp(const int32_t& axis, const float& epsilon);
Maybe<one::UserOpExpr> NormalizationGradOp(const int32_t& axis, const float& epsilon,
                                           const std::string& name);

Maybe<one::UserOpExpr> BroadcastDivGradOp();
Maybe<one::UserOpExpr> BroadcastDivGradOp(const std::string& name);

Maybe<one::UserOpExpr> LayerNormGradOp(const int64_t& begin_norm_axis, const double& epsilon);
Maybe<one::UserOpExpr> LayerNormGradOp(const int64_t& begin_norm_axis, const double& epsilon,
                                       const std::string& name);

Maybe<one::UserOpExpr> LayerNormParamGradOp(const int64_t& begin_params_axis,
                                            const bool& has_beta_diff, const bool& has_gamma_diff,
                                            const bool& has_normalized_diff);
Maybe<one::UserOpExpr> LayerNormParamGradOp(const int64_t& begin_params_axis,
                                            const bool& has_beta_diff, const bool& has_gamma_diff,
                                            const bool& has_normalized_diff,
                                            const std::string& name);

}  // namespace op_expr_helper

}  // namespace oneflow
