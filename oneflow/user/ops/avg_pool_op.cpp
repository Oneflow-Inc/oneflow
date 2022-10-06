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
#include "oneflow/user/kernels/avg_pool_kernel_util.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

namespace {

typedef std::function<Maybe<void>(user_op::InferContext* ctx)> TensorDescInferFn;

TensorDescInferFn AvgPoolMakeForwardTensorDescInferFn(const int32_t dim) {
  return [dim](user_op::InferContext* ctx) -> Maybe<void> {
    const Shape& x_shape = ctx->Shape4ArgNameAndIndex("x", 0);
    const std::string& data_format = ctx->Attr<std::string>("data_format");
    const std::vector<int32_t>& padding = ctx->Attr<std::vector<int32_t>>("padding");
    const std::vector<int32_t>& kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
    const std::vector<int32_t>& stride = ctx->Attr<std::vector<int32_t>>("stride");
    const bool ceil_mode = ctx->Attr<bool>("ceil_mode");
    const bool count_include_pad = ctx->Attr<bool>("count_include_pad");
    const int32_t& divisor_override = ctx->Attr<int32_t>("divisor_override");

    CHECK_EQ_OR_RETURN(kernel_size.size(), dim)
        << Error::RuntimeError() << "kernel size.size() should equal to dim.";
    for (int32_t pool_dim : kernel_size) {
      CHECK_GT_OR_RETURN(pool_dim, 0)
          << Error::RuntimeError() << "kernel size should great than 0, but got: " << pool_dim;
    }
    CHECK_EQ_OR_RETURN(stride.size(), dim)
        << Error::RuntimeError() << "stride.size() should equal to dim.";
    for (int32_t stride_dim : stride) {
      CHECK_GT_OR_RETURN(stride_dim, 0)
          << Error::RuntimeError() << "stride size should great than 0, but got: " << stride_dim;
    }
    for (int32_t i = 0; i < padding.size(); i++) {
      CHECK_GE_OR_RETURN(kernel_size[i], 2 * padding[i])
          << "pad should be smaller than half of kernel size";
    }

    const AvgPoolParams3D params_3d(dim, x_shape, data_format, padding, kernel_size, stride,
                                    ceil_mode, count_include_pad, divisor_override);
    user_op::TensorDesc* y_desc = ctx->MutOutputTensorDesc("y", 0);
    *y_desc = ctx->InputTensorDesc("x", 0);
    y_desc->set_shape(params_3d.GetYShape());

    return Maybe<void>::Ok();
  };
}

Maybe<void> AvgPoolForwardGetSbpFn(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  FOR_RANGE(int64_t, i, 0, std::min(2, (int)tensor.shape().NumAxes() - 2)) {
    ctx->NewBuilder().Split(user_op::OpArg("x", 0), i).Split(user_op::OpArg("y", 0), i).Build();
  }
  return Maybe<void>::Ok();
}

Maybe<void> AvgPoolBackwardGetSbpFn(user_op::SbpContext* ctx) {
  FOR_RANGE(int64_t, i, 0, 2) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("x", 0), i)
        .Split(user_op::OpArg("dy", 0), i)
        .Split(user_op::OpArg("dx", 0), i)
        .Build();
  }
  return Maybe<void>::Ok();
}

// Logically computation cost of pool op is the product of output data amount and pool kernal data
// amount. After adding sbp, we just divide it by parallel number if output data is splitted because
// splitting input and using partial sum for output is not a valid sbp for this op for now.
Maybe<double> GetComputationCost(user_op::ComputeComplexityFnContext* ctx,
                                 const std::string& blob_name) {
  const std::vector<int32_t> pool_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
  double logical_computation_cost = std::accumulate(
      pool_size.begin(), pool_size.end(), ctx->Shape4ArgNameAndIndex(blob_name, 0).elem_cnt(),
      std::multiplies<double>());
  const auto& parallel_hierarchy = ctx->parallel_desc().hierarchy();
  const auto& nd_sbp_y = ctx->NdSbp4ArgNameAndIndex(blob_name, 0);
  for (int32_t dim_sbp = 0; dim_sbp < nd_sbp_y.sbp_parallel_size(); dim_sbp++) {
    if (nd_sbp_y.sbp_parallel(dim_sbp).has_split_parallel()) {
      logical_computation_cost /= parallel_hierarchy->At(dim_sbp);
    }
  }
  return logical_computation_cost;
}

Maybe<void> BackwardTensorDescInferFn(user_op::InferContext* ctx) {
  *ctx->MutOutputTensorDesc("dx", 0) = ctx->InputTensorDesc("x", 0);
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

}  // namespace

#define IMPLEMENT_AVGPOOL_FUNCS(name, ndim)                                              \
  /*static*/ Maybe<void> name##Op::GetSbp(user_op::SbpContext* ctx) {                    \
    return AvgPoolForwardGetSbpFn(ctx);                                                  \
  }                                                                                      \
  /*static*/ Maybe<void> name##Op::InferLogicalTensorDesc(user_op::InferContext* ctx) {  \
    return AvgPoolMakeForwardTensorDescInferFn(ndim)(ctx);                               \
  }                                                                                      \
  /*static*/ Maybe<void> name##Op::InferPhysicalTensorDesc(user_op::InferContext* ctx) { \
    return InferLogicalTensorDesc(ctx);                                                  \
  }                                                                                      \
  /*static*/ Maybe<void> name##Op::InferDataType(user_op::InferContext* ctx) {           \
    return FwInferDataType(ctx);                                                         \
  }                                                                                      \
  /*static*/ Maybe<double> name##Op::GetComputeComplexity(                               \
      user_op::ComputeComplexityFnContext* ctx) {                                        \
    return GetComputationCost(ctx, "y");                                                 \
  }

IMPLEMENT_AVGPOOL_FUNCS(AvgPool1D, 1)
IMPLEMENT_AVGPOOL_FUNCS(AvgPool2D, 2)
IMPLEMENT_AVGPOOL_FUNCS(AvgPool3D, 3)
#undef IMPLEMENT_AVGPOOL_FUNCS

#define IMPLEMENT_AVGPOOL_BACKWARD_FUNCS(name)                                               \
  /*static*/ Maybe<void> name##GradOp::GetSbp(user_op::SbpContext* ctx) {                    \
    return AvgPoolBackwardGetSbpFn(ctx);                                                     \
  }                                                                                          \
  /*static*/ Maybe<void> name##GradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {  \
    return BackwardTensorDescInferFn(ctx);                                                   \
  }                                                                                          \
  /*static*/ Maybe<void> name##GradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) { \
    return InferLogicalTensorDesc(ctx);                                                      \
  }                                                                                          \
  /*static*/ Maybe<void> name##GradOp::InferDataType(user_op::InferContext* ctx) {           \
    return BwInferDataType(ctx);                                                             \
  }                                                                                          \
  /*static*/ Maybe<double> name##GradOp::GetComputeComplexity(                               \
      user_op::ComputeComplexityFnContext* ctx) {                                            \
    return GetComputationCost(ctx, "dy");                                                    \
  }

IMPLEMENT_AVGPOOL_BACKWARD_FUNCS(AvgPool1D)
IMPLEMENT_AVGPOOL_BACKWARD_FUNCS(AvgPool2D)
IMPLEMENT_AVGPOOL_BACKWARD_FUNCS(AvgPool3D)
#undef IMPLEMENT_AVGPOOL_BACKWARD_FUNCS

}  // namespace oneflow
