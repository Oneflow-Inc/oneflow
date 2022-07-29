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
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_generated.h"
#include "oneflow/core/functional/impl/common.h"

namespace oneflow {

/*static*/ Maybe<void> WkvOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}
/*static*/ Maybe<void> WkvOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& v_shape = ctx->InputShape("v", 0);
  *ctx->MutOutputShape("y", 0) = v_shape;
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> WkvOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> WkvOp::InferDataType(user_op::InferContext* ctx) {
  DataType v_dtype = ctx->InputDType("v", 0);
  *ctx->MutOutputDType("y", 0) = v_dtype;
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> WkvGradOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}
/*static*/ Maybe<void> WkvGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const int64_t B = ctx->Attr<int64_t>("B");
  const int64_t C = ctx->Attr<int64_t>("C");
  const Shape& gy_shape = ctx->InputShape("gy", 0);
  *ctx->MutOutputShape("gw", 0) = Shape({B, C});
  *ctx->MutOutputShape("gu", 0) = Shape({B, C});
  *ctx->MutOutputShape("gk", 0) = gy_shape;
  *ctx->MutOutputShape("gv", 0) = gy_shape;
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> WkvGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> WkvGradOp::InferDataType(user_op::InferContext* ctx) {
  DataType gy_dtype = ctx->InputDType("gy", 0);
  *ctx->MutOutputDType("gw", 0) = gy_dtype;
  *ctx->MutOutputDType("gu", 0) = gy_dtype;
  *ctx->MutOutputDType("gk", 0) = gy_dtype;
  *ctx->MutOutputDType("gv", 0) = gy_dtype;
  return Maybe<void>::Ok();
}

namespace {

Maybe<void> WkvGraphGradOp(user_op::BackwardOpConfContext* ctx) {
  const std::string wkv_grad_op = ctx->FwOp().op_name() + "_grad";
  ctx->DefineOp(wkv_grad_op, [&](user_op::BackwardOpBuilder& builder) {
    return builder.OpTypeName("wkv_grad")
        .InputBind("w", ctx->FwOp().input("w", 0))
        .InputBind("u", ctx->FwOp().input("u", 0))
        .InputBind("k", ctx->FwOp().input("k", 0))
        .InputBind("v", ctx->FwOp().input("v", 0))
        .InputBind("gy", ctx->FwOp().GetGradTensorWithOpOutput("y", 0))
        .Attr("B", ctx->FwOp().attr<int64_t>("B"))
        .Attr("T", ctx->FwOp().attr<int64_t>("T"))
        .Attr("C", ctx->FwOp().attr<int64_t>("C"))
        .Output("gw")
        .Output("gu")
        .Output("gk")
        .Output("gv")
        .Build();
  });
  const std::string reduce_sum_w_op = ctx->FwOp().op_name() + "_grad_reduce_sum_w";
  std::vector<int32_t> reduce_axes_vec;
  reduce_axes_vec.emplace_back(0);
  ctx->DefineOp(reduce_sum_w_op, [&](user_op::BackwardOpBuilder& builder) {
    return builder.OpTypeName("reduce_sum")
        .InputBind("input_tensor", ctx->GetOp(wkv_grad_op).output("gw", 0))
        .Attr("axis", reduce_axes_vec)
        .Attr("keepdims", false)
        .Output("output_tensor")
        .Build();
  });
  const std::string reduce_sum_u_op = ctx->FwOp().op_name() + "_grad_reduce_sum_u";
  ctx->DefineOp(reduce_sum_u_op, [&](user_op::BackwardOpBuilder& builder) {
    return builder.OpTypeName("reduce_sum")
        .InputBind("input_tensor", ctx->GetOp(wkv_grad_op).output("gu", 0))
        .Attr("axis", reduce_axes_vec)
        .Attr("keepdims", false)
        .Output("output_tensor")
        .Build();
  });
  ctx->FwOp().InputGradBind(user_op::OpArg("w", 0), [&]() -> const std::string& {
    return ctx->GetOp(reduce_sum_w_op).output("output_tensor", 0);
  });
  ctx->FwOp().InputGradBind(user_op::OpArg("u", 0), [&]() -> const std::string& {
    return ctx->GetOp(reduce_sum_u_op).output("output_tensor", 0);
  });
  ctx->FwOp().InputGradBind(user_op::OpArg("k", 0), [&]() -> const std::string& {
    return ctx->GetOp(wkv_grad_op).output("gk", 0);
  });
  ctx->FwOp().InputGradBind(user_op::OpArg("v", 0), [&]() -> const std::string& {
    return ctx->GetOp(wkv_grad_op).output("gv", 0);
  });
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_USER_OP_GRAD("wkv").SetBackwardOpConfGenFn(WkvGraphGradOp);

}  // namespace oneflow
