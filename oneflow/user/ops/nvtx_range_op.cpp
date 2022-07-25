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
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

#ifdef WITH_CUDA

/* static */ Maybe<void> NvtxStartOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  *ctx->MutOutputShape("out", 0) = ctx->InputShape("in", 0);
  *ctx->MutOutputIsDynamic("out", 0) = ctx->InputIsDynamic("in", 0);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> NvtxStartOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> NvtxStartOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
  FOR_RANGE(int64_t, i, 0, in_tensor.shape().NumAxes()) {
    ctx->NewBuilder().Split(user_op::OpArg("in", 0), i).Split(user_op::OpArg("out", 0), i).Build();
  }
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("in", 0))
      .PartialSum(user_op::OpArg("out", 0))
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> NvtxStartOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->MutOutputDType("out", 0) = ctx->InputDType("in", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> NvtxEndOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  *ctx->MutOutputShape("out", 0) = ctx->InputShape("in", 0);
  *ctx->MutOutputIsDynamic("out", 0) = ctx->InputIsDynamic("in", 0);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> NvtxEndOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> NvtxEndOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
  FOR_RANGE(int64_t, i, 0, in_tensor.shape().NumAxes()) {
    ctx->NewBuilder().Split(user_op::OpArg("in", 0), i).Split(user_op::OpArg("out", 0), i).Build();
  }
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("in", 0))
      .PartialSum(user_op::OpArg("out", 0))
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> NvtxEndOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->MutOutputDType("out", 0) = ctx->InputDType("in", 0);
  return Maybe<void>::Ok();
}

#else

/* static */ Maybe<void> NvtxStartOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return Error::UnimplementedError() << "require CUDA to use NVTX";
}

/*static*/ Maybe<void> NvtxStartOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> NvtxStartOp::GetSbp(user_op::SbpContext* ctx) {
  return Error::UnimplementedError() << "require CUDA to use NVTX";
}

/* static */ Maybe<void> NvtxStartOp::InferDataType(user_op::InferContext* ctx) {
  return Error::UnimplementedError() << "require CUDA to use NVTX";
}

/* static */ Maybe<void> NvtxEndOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return Error::UnimplementedError() << "require CUDA to use NVTX";
}

/*static*/ Maybe<void> NvtxEndOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return Error::UnimplementedError() << "require CUDA to use NVTX";
}

/* static */ Maybe<void> NvtxEndOp::GetSbp(user_op::SbpContext* ctx) {
  return Error::UnimplementedError() << "require CUDA to use NVTX";
}

/* static */ Maybe<void> NvtxEndOp::InferDataType(user_op::InferContext* ctx) {
  return Error::UnimplementedError() << "require CUDA to use NVTX";
}

#endif  // WITH_CUDA

REGISTER_USER_OP_GRAD("nvtx_start")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                               user_op::AddOpFn AddOp) -> Maybe<void> {
      if (op.NeedGenGradTensor4OpInput("in", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper nvtx_end_op =
            builder.Op("nvtx_end")
                .Input("in", op.GetGradTensorWithOpOutput("out", 0))
                .Output("out")
                .Attr("mark_prefix", op.attr<std::string>("mark_prefix") + "-bw")
                .Build();
        op.BindGradTensorWithOpInput(nvtx_end_op.output("out", 0), "in", 0);
        AddOp(nvtx_end_op);
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("nvtx_end")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                               user_op::AddOpFn AddOp) -> Maybe<void> {
      if (op.NeedGenGradTensor4OpInput("in", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper nvtx_start_op =
            builder.Op("nvtx_start")
                .Input("in", op.GetGradTensorWithOpOutput("out", 0))
                .Output("out")
                .Attr("mark_prefix", op.attr<std::string>("mark_prefix") + "-bw")
                .Build();
        op.BindGradTensorWithOpInput(nvtx_start_op.output("out", 0), "in", 0);
        AddOp(nvtx_start_op);
      }
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
