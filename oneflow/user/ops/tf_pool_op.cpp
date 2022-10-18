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
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/utils/pool_util.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

namespace {

// Logically computation cost of pool op is the product of output data amount and pool kernal data
// amount. After adding sbp, we just divide it by parallel number if output data is splitted because
// splitting input and using partial sum for output is not a valid sbp for this op for now.
Maybe<double> GetComputationCost(user_op::ComputeComplexityFnContext* ctx) {
  const std::vector<int32_t> pool_size = ctx->Attr<std::vector<int32_t>>("pool_size");
  double logical_computation_cost =
      std::accumulate(pool_size.begin(), pool_size.end(),
                      ctx->Shape4ArgNameAndIndex("y", 0).elem_cnt(), std::multiplies<double>());
  const auto& parallel_hierarchy = ctx->parallel_desc().hierarchy();
  const auto& nd_sbp_y = ctx->NdSbp4ArgNameAndIndex("y", 0);
  for (int32_t dim_sbp = 0; dim_sbp < nd_sbp_y.sbp_parallel_size(); dim_sbp++) {
    if (nd_sbp_y.sbp_parallel(dim_sbp).has_split_parallel()) {
      logical_computation_cost /= parallel_hierarchy->At(dim_sbp);
    }
  }
  return logical_computation_cost;
}

typedef std::function<Maybe<void>(user_op::InferContext* ctx)> TensorDescInferFn;

TensorDescInferFn MakeFwTensorDescInferFn(const int32_t dim) {
  return [dim](user_op::InferContext* ctx) -> Maybe<void> {
    const Shape& x_shape = ctx->InputShape("x", 0);
    const std::string& data_format = ctx->Attr<std::string>("data_format");
    const std::string& padding = ctx->Attr<std::string>("padding");
    const auto& padding_before = ctx->Attr<std::vector<int32_t>>("padding_before");
    const auto& padding_after = ctx->Attr<std::vector<int32_t>>("padding_after");
    const std::vector<int32_t> pool_size = ctx->Attr<std::vector<int32_t>>("pool_size");
    const std::vector<int32_t> strides = ctx->Attr<std::vector<int32_t>>("strides");
    const bool ceil_mode = ctx->Attr<bool>("ceil_mode");

    CHECK_EQ_OR_RETURN(pool_size.size(), dim);
    for (int32_t pool_dim : pool_size) { CHECK_GT_OR_RETURN(pool_dim, 0); }
    CHECK_EQ_OR_RETURN(strides.size(), dim);
    for (int32_t stride_dim : strides) { CHECK_GT_OR_RETURN(stride_dim, 0); }

    const Params3D params_3d(dim, x_shape, data_format, padding, padding_before, padding_after,
                             pool_size, strides, ceil_mode);
    user_op::TensorDesc* y_desc = ctx->MutOutputTensorDesc("y", 0);
    y_desc->set_shape(params_3d.GetYShape());
    y_desc->set_is_dynamic(ctx->InputIsDynamic("x", 0));
    return Maybe<void>::Ok();
  };
}

Maybe<void> BwTensorDescInferFn(user_op::InferContext* ctx) {
  ctx->SetOutputShape("dx", 0, ctx->InputShape("x", 0));
  ctx->SetOutputIsDynamic("dx", 0, ctx->InputIsDynamic("x", 0));
  return Maybe<void>::Ok();
}

Maybe<void> FwInferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("y", 0, ctx->InputDType("x", 0));
  return Maybe<void>::Ok();
}

Maybe<void> BwInferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("dx", 0, ctx->InputDType("x", 0));
  return Maybe<void>::Ok();
}

Maybe<void> FwGetSbpFn(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  FOR_RANGE(int64_t, i, 0, tensor.shape().NumAxes()) {
    ctx->NewBuilder().Split(user_op::OpArg("x", 0), i).Split(user_op::OpArg("y", 0), i).Build();
  }
  return Maybe<void>::Ok();
}

Maybe<void> BwGetSbpFn(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  FOR_RANGE(int64_t, i, 0, tensor.shape().NumAxes()) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("x", 0), i)
        .Split(user_op::OpArg("y", 0), i)
        .Split(user_op::OpArg("dy", 0), i)
        .Split(user_op::OpArg("dx", 0), i)
        .Build();
  }
  return Maybe<void>::Ok();
}

}  // namespace

#define IMPLEMENT_TF_POOL_FUNCS(name, dim)                                                      \
  /*static*/ Maybe<void> name##Op::GetSbp(user_op::SbpContext* ctx) { return FwGetSbpFn(ctx); } \
  /*static*/ Maybe<void> name##Op::InferLogicalTensorDesc(user_op::InferContext* ctx) {         \
    return MakeFwTensorDescInferFn(dim)(ctx);                                                   \
  }                                                                                             \
  /*static*/ Maybe<void> name##Op::InferPhysicalTensorDesc(user_op::InferContext* ctx) {        \
    return InferLogicalTensorDesc(ctx);                                                         \
  }                                                                                             \
  /*static*/ Maybe<void> name##Op::InferDataType(user_op::InferContext* ctx) {                  \
    return FwInferDataType(ctx);                                                                \
  }                                                                                             \
  /*static*/ Maybe<double> name##Op::GetComputeComplexity(                                      \
      user_op::ComputeComplexityFnContext* ctx) {                                               \
    return GetComputationCost(ctx);                                                             \
  }

IMPLEMENT_TF_POOL_FUNCS(TfAvgPool1D, 1)
IMPLEMENT_TF_POOL_FUNCS(TfAvgPool2D, 2)
IMPLEMENT_TF_POOL_FUNCS(TfAvgPool3D, 3)
IMPLEMENT_TF_POOL_FUNCS(TfMaxPool1D, 1)
IMPLEMENT_TF_POOL_FUNCS(TfMaxPool2D, 2)
IMPLEMENT_TF_POOL_FUNCS(TfMaxPool3D, 3)
#undef IMPLEMENT_TF_POOL_FUNCS

#define IMPLEMENT_TF_POOL_BACKWARD_FUNCS(name)                                               \
  /*static*/ Maybe<void> name##GradOp::GetSbp(user_op::SbpContext* ctx) {                    \
    return BwGetSbpFn(ctx);                                                                  \
  }                                                                                          \
  /*static*/ Maybe<void> name##GradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {  \
    return BwTensorDescInferFn(ctx);                                                         \
  }                                                                                          \
  /*static*/ Maybe<void> name##GradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) { \
    return InferLogicalTensorDesc(ctx);                                                      \
  }                                                                                          \
  /*static*/ Maybe<void> name##GradOp::InferDataType(user_op::InferContext* ctx) {           \
    return BwInferDataType(ctx);                                                             \
  }                                                                                          \
  /*static*/ Maybe<double> name##GradOp::GetComputeComplexity(                               \
      user_op::ComputeComplexityFnContext* ctx) {                                            \
    return GetComputationCost(ctx);                                                          \
  }

IMPLEMENT_TF_POOL_BACKWARD_FUNCS(TfAvgPool1D)
IMPLEMENT_TF_POOL_BACKWARD_FUNCS(TfAvgPool2D)
IMPLEMENT_TF_POOL_BACKWARD_FUNCS(TfAvgPool3D)
IMPLEMENT_TF_POOL_BACKWARD_FUNCS(TfMaxPool1D)
IMPLEMENT_TF_POOL_BACKWARD_FUNCS(TfMaxPool2D)
IMPLEMENT_TF_POOL_BACKWARD_FUNCS(TfMaxPool3D)
#undef IMPLEMENT_TF_POOL_BACKWARD_FUNCS

}  // namespace oneflow
