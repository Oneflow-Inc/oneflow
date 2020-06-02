#include "oneflow/core/framework/framework.h"
#include "oneflow/core/operator/reduce_sbp_util.h"

namespace oneflow {

namespace {

bool IsReduceAxesLegal(const AxisVector& reduce_axes_vec, const Shape& reduce_shape,
                       const Shape& broadcast_shape) {
  if (reduce_axes_vec.empty()) { return reduce_shape == broadcast_shape; }
  Shape reduced_shape = CreateReducedShape(ShapeView(broadcast_shape), reduce_axes_vec);
  if (broadcast_shape.NumAxes() > reduce_shape.NumAxes()) {
    reduced_shape = reduced_shape.RemoveOnes(reduce_axes_vec);
  }
  return reduced_shape == reduce_shape;
}

Maybe<void> InferTensorDesc(user_op::InferContext* ctx) {
  const Shape* x_shape = ctx->Shape4ArgNameAndIndex("x", 0);
  const Shape* like_shape = ctx->Shape4ArgNameAndIndex("like", 0);
  const auto& reduce_axes_vec = ctx->Attr<AxisVector>("reduce_axes");
  CHECK_OR_RETURN(IsReduceAxesLegal(reduce_axes_vec, *like_shape, *x_shape));
  *ctx->Shape4ArgNameAndIndex("y", 0) = *like_shape;
  *ctx->Dtype4ArgNameAndIndex("y", 0) = *ctx->Dtype4ArgNameAndIndex("x", 0);
  return Maybe<void>::Ok();
}

Maybe<void> InferBatchAxis(user_op::BatchAxisContext* ctx) {
  *ctx->BatchAxis4ArgNameAndIndex("y", 0) = *ctx->BatchAxis4ArgNameAndIndex("like", 0);
  return Maybe<void>::Ok();
}

Maybe<void> GetSbpSignature(user_op::SbpContext* ctx) {
  const auto& x_desc = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  const int64_t num_axes = x_desc.shape().NumAxes();
  const auto& reduce_axes_vec = ctx->Attr<AxisVector>("reduce_axes");
  auto IsReducedAxis = ReduceSbpUtil::MakePredicatorIsReducedAxis(reduce_axes_vec);
  int64_t num_reduced_axes = 0;
  FOR_RANGE(int64_t, i, 0, num_axes) {
    if (IsReducedAxis(i)) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("x", 0), i)
          .Broadcast(user_op::OpArg("like", 0))
          .PartialSum(user_op::OpArg("y", 0))
          .Build();
      num_reduced_axes += 1;
    } else {
      ctx->NewBuilder()
          .Split(user_op::OpArg("x", 0), i)
          .Split(user_op::OpArg("like", 0), i - num_reduced_axes)
          .Split(user_op::OpArg("y", 0), i - num_reduced_axes)
          .Build();
    }
  }
  return Maybe<void>::Ok();
}

void SetInputArgModifier(user_op::GetInputArgModifier GetInputArgModifierFn,
                         const user_op::UserOpConfWrapper& op_conf) {
  user_op::InputArgModifier* like_arg_modifier = GetInputArgModifierFn("like", 0);
  CHECK(like_arg_modifier != nullptr);
  like_arg_modifier->set_use_header_only(true);
  like_arg_modifier->set_requires_grad(false);
}

}  // namespace

REGISTER_USER_OP("reduce_sum_like")
    .Input("x")
    .Input("like")
    .Output("y")
    .Attr("reduce_axes", UserOpAttrType::kAtListInt64)
    .SetTensorDescInferFn(InferTensorDesc)
    .SetBatchAxisInferFn(InferBatchAxis)
    .SetGetSbpFn(GetSbpSignature)
    .SetInputArgModifyFn(SetInputArgModifier);

}  // namespace oneflow
