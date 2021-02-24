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
#include "oneflow/core/operator/reduce_sbp_util.h"

namespace oneflow {

namespace {

Maybe<void> GetSbpSignatures(user_op::SbpContext* ctx) {
  int32_t num_axes = ctx->LogicalTensorDesc4InputArgNameAndIndex("like", 0).shape().NumAxes();
  const auto& reduced_axes = ctx->Attr<std::vector<int32_t>>("broadcast_axes");
  HashSet<int32_t> conf_axes;
  ReduceSbpUtil::GetRegularAxes(num_axes, reduced_axes, &conf_axes);
  auto IsReducedAxis = ReduceSbpUtil::MakePredicatorIsReducedAxis(conf_axes, num_axes);
  int32_t num_reduced_axis = 0;
  FOR_RANGE(int64_t, i, 0, num_axes) {
    if (IsReducedAxis(i)) {
      ctx->NewBuilder()
          .Broadcast(user_op::OpArg("x", 0))
          .Split(user_op::OpArg("like", 0), i)
          .Split(user_op::OpArg("y", 0), i)
          .Build();
      num_reduced_axis += 1;
    } else {
      ctx->NewBuilder()
          .Split(user_op::OpArg("x", 0), i - num_reduced_axis)
          .Split(user_op::OpArg("like", 0), i)
          .Split(ctx->outputs(), i)
          .Build();
    }
  }
  ctx->NewBuilder().PartialSum(ctx->inputs()).PartialSum(ctx->outputs()).Build();
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("x", 0))
      .Broadcast(user_op::OpArg("like", 0))
      .PartialSum(user_op::OpArg("y", 0))
      .Build();
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("x", 0))
      .PartialSum(user_op::OpArg("like", 0))
      .Broadcast(user_op::OpArg("y", 0))
      .Build();
  return Maybe<void>::Ok();
}

bool IsAxesLegal(const AxisVector& axis_vec, const Shape& like_shape, const Shape& in_shape) {
  Shape reduced_shape = CreateReducedShape(like_shape, axis_vec);
  if (like_shape.NumAxes() > in_shape.NumAxes()) {
    reduced_shape = reduced_shape.RemoveOnes(axis_vec);
  }
  return reduced_shape.dim_vec() == in_shape.dim_vec();
}

Maybe<void> InferTensorDesc(user_op::InferContext* ctx) {
  const auto& broadcast_axes = ctx->Attr<std::vector<int32_t>>("broadcast_axes");
  CHECK_OR_RETURN(!broadcast_axes.empty());
  const Shape* in_shape = ctx->Shape4ArgNameAndIndex("x", 0);
  const Shape* like_shape = ctx->Shape4ArgNameAndIndex("like", 0);
  Shape* out_shape = ctx->Shape4ArgNameAndIndex("y", 0);
  const AxisVector axis_vec = {broadcast_axes.begin(), broadcast_axes.end()};
  CHECK_OR_RETURN(IsAxesLegal(axis_vec, *like_shape, *in_shape));
  *out_shape = *like_shape;
  *ctx->Dtype4ArgNameAndIndex("y", 0) = *ctx->Dtype4ArgNameAndIndex("like", 0);
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_USER_OP("broadcast_like")
    .Input("x")
    .Input("like")
    .Attr<std::vector<int32_t>>("broadcast_axes")
    .Output("y")
    .SetTensorDescInferFn(InferTensorDesc)
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper&) {
      user_op::InputArgModifier* like_modifier = GetInputArgModifierFn("like", 0);
      CHECK(like_modifier != nullptr);
      like_modifier->set_requires_grad(false);
    })
    .SetGetSbpFn(GetSbpSignatures);

REGISTER_USER_OP_GRAD("broadcast_like")
    .SetBackwardOpConfGenFn([](user_op::BackwardOpConfContext* ctx) {
      const auto x_grad_op_name = ctx->FwOp().op_name() + "_x_grad";
      ctx->DefineOp(x_grad_op_name, [&ctx](user_op::BackwardOpBuilder& builder) {
        return builder.OpTypeName("reduce_sum_like")
            .InputBind("x", ctx->FwOp().output_grad("y", 0))
            .InputBind("like", ctx->FwOp().input("x", 0))
            .Output("y")
            .Attr("axis", ctx->FwOp().attr<std::vector<int32_t>>("broadcast_axes"))
            .Build();
      });

      ctx->FwOp().InputGradBind(user_op::OpArg("x", 0),
                                [&ctx, &x_grad_op_name]() -> const std::string& {
                                  return ctx->GetOp(x_grad_op_name).output("y", 0);
                                });
    });

}  // namespace oneflow
