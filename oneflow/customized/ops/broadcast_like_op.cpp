#include "oneflow/core/framework/batch_axis_context.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/operator/reduce_sbp_util.h"

namespace oneflow {

namespace {

Maybe<void> GetSbpSignatures(user_op::SbpContext* ctx) {
  int32_t num_axes = ctx->LogicalTensorDesc4InputArgNameAndIndex("like", 0).shape().NumAxes();
  const auto& reduced_axes = ctx->GetAttr<std::vector<int32_t>>("axis");
  HashSet<int32_t> conf_axes = {reduced_axes.begin(), reduced_axes.end()};
  auto IsReducedAxis = ReduceSbpUtil::MakePredicatorIsReducedAxis(conf_axes, num_axes);
  FOR_RANGE(int64_t, i, 0, num_axes) {
    if (IsReducedAxis(i)) {
      ctx->NewBuilder()
          .Broadcast(user_op::OpArg("x", 0))
          .Split(user_op::OpArg("like", 0), i)
          .Split(user_op::OpArg("y", 0), i)
          .Build();
    } else {
      ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
    }
  }
  return Maybe<void>::Ok();
}

REGISTER_USER_OP("broadcast_like")
    .Input("x")
    .Input("like")
    .Attr("axis", UserOpAttrType::kAtListInt32)
    .Output("y")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* like_shape = ctx->Shape4ArgNameAndIndex("like", 0);
      Shape* out_shape = ctx->Shape4ArgNameAndIndex("out", 0);
      *out_shape = *like_shape;

      //*ctx->Shape4ArgNameAndIndex("out", 0) = *ctx->Shape4ArgNameAndIndex("like", 0);
      //*ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn) {
      user_op::InputArgModifier* like_arg_modifier = GetInputArgModifierFn("like", 0);
      CHECK_OR_RETURN(like_arg_modifier != nullptr);
      like_arg_modifier->set_use_header_only(true);
    })
    .SetBatchAxisInferFn(user_op::BatchAxisInferFnUtil::NaiveInferBatchAxis)
    .SetGetSbpFn(GetSbpSignatures);
}  // namespace
}  // namespace oneflow