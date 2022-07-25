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

typedef std::function<Maybe<void>(user_op::InferContext* ctx)> TensorDescInferFn;
typedef std::function<Maybe<void>(const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp)>
    GenBackwardOpConfFn;

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
    user_op::TensorDesc* y_desc = ctx->OutputTensorDesc("y", 0);
    *y_desc->mut_shape() = params_3d.GetYShape();
    *y_desc->mut_is_dynamic() = ctx->InputIsDynamic("x", 0);
    return Maybe<void>::Ok();
  };
}

Maybe<void> BwTensorDescInferFn(user_op::InferContext* ctx) {
  *ctx->MutOutputShape("dx", 0) = ctx->InputShape("x", 0);
  *ctx->MutOutputIsDynamic("dx", 0) = ctx->InputIsDynamic("x", 0);
  return Maybe<void>::Ok();
}

Maybe<void> FwInferDataType(user_op::InferContext* ctx) {
  *ctx->MutOutputDType("y", 0) = ctx->InputDType("x", 0);
  return Maybe<void>::Ok();
}

Maybe<void> BwInferDataType(user_op::InferContext* ctx) {
  *ctx->MutOutputDType("dx", 0) = ctx->InputDType("x", 0);
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

GenBackwardOpConfFn MakeGenBackwardOpConfFn(const std::string& mode, const int32_t dim) {
  return [mode, dim](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) -> Maybe<void> {
    if (op.NeedGenGradTensor4OpInput("x", 0)) {
      user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
      user_op::UserOpConfWrapper grad_op =
          builder.Op(mode + "_pool_" + std::to_string(dim) + "d_grad")
              .Input("x", op.input("x", 0))
              .Input("y", op.output("y", 0))
              .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
              .Output("dx")
              .Attr("data_format", op.attr<std::string>("data_format"))
              .Attr("padding", op.attr<std::string>("padding"))
              .Attr("padding_before", op.attr<std::vector<int32_t>>("padding_before"))
              .Attr("padding_after", op.attr<std::vector<int32_t>>("padding_after"))
              .Attr("pool_size", op.attr<std::vector<int32_t>>("pool_size"))
              .Attr("strides", op.attr<std::vector<int32_t>>("strides"))
              .Attr("ceil_mode", op.attr<bool>("ceil_mode"))
              .Build();
      op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);
      AddOp(grad_op);
    }
    return Maybe<void>::Ok();
  };
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
  }

IMPLEMENT_TF_POOL_BACKWARD_FUNCS(TfAvgPool1D)
IMPLEMENT_TF_POOL_BACKWARD_FUNCS(TfAvgPool2D)
IMPLEMENT_TF_POOL_BACKWARD_FUNCS(TfAvgPool3D)
IMPLEMENT_TF_POOL_BACKWARD_FUNCS(TfMaxPool1D)
IMPLEMENT_TF_POOL_BACKWARD_FUNCS(TfMaxPool2D)
IMPLEMENT_TF_POOL_BACKWARD_FUNCS(TfMaxPool3D)
#undef IMPLEMENT_TF_POOL_BACKWARD_FUNCS

REGISTER_USER_OP_GRAD("tf_avg_pool_1d")
    .SetGenBackwardOpConfFn(MakeGenBackwardOpConfFn("tf_avg", 1));
REGISTER_USER_OP_GRAD("tf_avg_pool_2d")
    .SetGenBackwardOpConfFn(MakeGenBackwardOpConfFn("tf_avg", 2));
REGISTER_USER_OP_GRAD("tf_avg_pool_3d")
    .SetGenBackwardOpConfFn(MakeGenBackwardOpConfFn("tf_avg", 3));

REGISTER_USER_OP_GRAD("tf_max_pool_1d")
    .SetGenBackwardOpConfFn(MakeGenBackwardOpConfFn("tf_max", 1));
REGISTER_USER_OP_GRAD("tf_max_pool_2d")
    .SetGenBackwardOpConfFn(MakeGenBackwardOpConfFn("tf_max", 2));
REGISTER_USER_OP_GRAD("tf_max_pool_3d")
    .SetGenBackwardOpConfFn(MakeGenBackwardOpConfFn("tf_max", 3));

}  // namespace oneflow
