#include "oneflow/core/framework/framework.h"
#include "oneflow/core/operator/reduce_sbp_util.h"

namespace oneflow {

namespace {

Maybe<void> GetSbpSignatures(user_op::SbpContext* ctx) {
  const Shape& like_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("like", 0).shape();
  const int64_t num_axes = like_shape.NumAxes();
  const auto& broadcast_axes_vec = ctx->Attr<AxisVector>("broadcast_axes");
  auto IsReducedAxis = ReduceSbpUtil::MakePredicatorIsReducedAxis(broadcast_axes_vec);
  int64_t num_reduced_axes = 0;
  FOR_RANGE(int64_t, i, 0, num_axes) {
    if (IsReducedAxis(i)) {
      ctx->NewBuilder()
          .Broadcast(user_op::OpArg("x", 0))
          .Split(user_op::OpArg("like", 0), i)
          .Split(user_op::OpArg("y", 0), i)
          .Build();
      num_reduced_axes += 1;
    } else {
      ctx->NewBuilder()
          .Split(user_op::OpArg("x", 0), i - num_reduced_axes)
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
  const auto& broadcast_axes_vec = ctx->Attr<AxisVector>("broadcast_axes");
  const Shape* x_shape = ctx->Shape4ArgNameAndIndex("x", 0);
  const Shape* like_shape = ctx->Shape4ArgNameAndIndex("like", 0);
  CHECK_OR_RETURN(IsAxesLegal(broadcast_axes_vec, *like_shape, *x_shape));
  *ctx->Shape4ArgNameAndIndex("y", 0) = *like_shape;
  *ctx->Dtype4ArgNameAndIndex("y", 0) = *ctx->Dtype4ArgNameAndIndex("x", 0);
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_USER_OP("broadcast_like")
    .Input("x")
    .Input("like")
    .Attr("broadcast_axes", UserOpAttrType::kAtListInt64)
    .Output("y")
    .SetTensorDescInferFn(InferTensorDesc)
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper&) {
      user_op::InputArgModifier* like_arg_modifier = GetInputArgModifierFn("like", 0);
      CHECK(like_arg_modifier != nullptr);
      like_arg_modifier->set_use_header_only(true);
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("y", 0) = *ctx->BatchAxis4ArgNameAndIndex("like", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn(GetSbpSignatures);

}  // namespace oneflow
