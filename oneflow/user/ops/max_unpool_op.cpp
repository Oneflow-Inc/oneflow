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
#include "oneflow/core/common/shape.h"
#include "oneflow/user/kernels/max_unpool_kernel_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

namespace {

typedef std::function<Maybe<void>(user_op::InferContext* ctx)> TensorDescInferFn;

TensorDescInferFn MaxUnpoolMakeForwardTensorDescInferFn(const int32_t dim) {
  return [dim](user_op::InferContext* ctx) -> Maybe<void> {
    const Shape& x_shape = ctx->InputShape("x", 0);
    const std::vector<int32_t>& padding = ctx->Attr<std::vector<int32_t>>("padding");
    const std::vector<int32_t>& kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
    const std::vector<int32_t>& stride = ctx->Attr<std::vector<int32_t>>("stride");
    Shape output_shape = Shape();
    if (ctx->Attr<bool>("has_output_size")) {
      output_shape = ctx->Attr<Shape>("output_size");
    } else {
      const MaxUnpoolParams3D params_3d(dim, x_shape, padding, kernel_size, stride);
      output_shape = params_3d.GetYShape();
    }

    user_op::TensorDesc* y_desc = ctx->MutOutputTensorDesc("y", 0);
    *y_desc = ctx->InputTensorDesc("x", 0);
    y_desc->set_shape(output_shape);

    return Maybe<void>::Ok();
  };
}

Maybe<void> MaxUnpoolForwardGetSbpFn(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  FOR_RANGE(int64_t, i, 0, std::min(2L, tensor.shape().NumAxes() - 2)) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("x", 0), i)
        .Split(user_op::OpArg("indices", 0), i)
        .Split(user_op::OpArg("y", 0), i)
        .Build();
  }
  return Maybe<void>::Ok();
}

Maybe<void> MaxUnpoolBackwardGetSbpFn(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  FOR_RANGE(int64_t, i, 0, std::min(2L, tensor.shape().NumAxes())) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("x", 0), i)
        .Split(user_op::OpArg("indices", 0), i)
        .Split(user_op::OpArg("dy", 0), i)
        .Split(user_op::OpArg("dx", 0), i)
        .Build();
  }
  return Maybe<void>::Ok();
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

#define IMPLEMENT_MAXUNPOOL_FUNCS(name, dim)                                             \
  /*static*/ Maybe<void> name##Op::GetSbp(user_op::SbpContext* ctx) {                    \
    return MaxUnpoolForwardGetSbpFn(ctx);                                                \
  }                                                                                      \
  /*static*/ Maybe<void> name##Op::InferLogicalTensorDesc(user_op::InferContext* ctx) {  \
    return MaxUnpoolMakeForwardTensorDescInferFn(dim)(ctx);                              \
  }                                                                                      \
  /*static*/ Maybe<void> name##Op::InferPhysicalTensorDesc(user_op::InferContext* ctx) { \
    return InferLogicalTensorDesc(ctx);                                                  \
  }                                                                                      \
  /*static*/ Maybe<void> name##Op::InferDataType(user_op::InferContext* ctx) {           \
    return FwInferDataType(ctx);                                                         \
  }

IMPLEMENT_MAXUNPOOL_FUNCS(MaxUnpool1D, 1)
IMPLEMENT_MAXUNPOOL_FUNCS(MaxUnpool2D, 2)
IMPLEMENT_MAXUNPOOL_FUNCS(MaxUnpool3D, 3)
#undef IMPLEMENT_MAXUNPOOL_FUNCS

#define IMPLEMENT_MAXUNPOOL_BACKWARD_FUNCS(name)                                             \
  /*static*/ Maybe<void> name##GradOp::GetSbp(user_op::SbpContext* ctx) {                    \
    return MaxUnpoolBackwardGetSbpFn(ctx);                                                   \
  }                                                                                          \
  /*static*/ Maybe<void> name##GradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {  \
    return BackwardTensorDescInferFn(ctx);                                                   \
  }                                                                                          \
  /*static*/ Maybe<void> name##GradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) { \
    return InferLogicalTensorDesc(ctx);                                                      \
  }                                                                                          \
  /*static*/ Maybe<void> name##GradOp::InferDataType(user_op::InferContext* ctx) {           \
    return BwInferDataType(ctx);                                                             \
  }

IMPLEMENT_MAXUNPOOL_BACKWARD_FUNCS(MaxUnpool1D)
IMPLEMENT_MAXUNPOOL_BACKWARD_FUNCS(MaxUnpool2D)
IMPLEMENT_MAXUNPOOL_BACKWARD_FUNCS(MaxUnpool3D)
#undef IMPLEMENT_MAXUNPOOL_BACKWARD_FUNCS

}  // namespace oneflow
