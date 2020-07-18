#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

Maybe<void> InferTensorDesc(user_op::InferContext* ctx) {
  const int64_t axis = ctx->Attr<int64_t>("axis");
  const user_op::TensorDesc* in_desc = ctx->TensorDesc4ArgNameAndIndex("in", 0);
  int64_t dynamic_dims = 0;
  int64_t static_dims = 0;
  FOR_RANGE(int32_t, i, 0, ctx->outputs().size()) {
    const user_op::TensorDesc* like_i_desc = ctx->TensorDesc4ArgNameAndIndex("like", i);
    user_op::TensorDesc* out_i_desc = ctx->TensorDesc4ArgNameAndIndex("out", i);
    CHECK_EQ_OR_RETURN(like_i_desc->shape().NumAxes(), in_desc->shape().NumAxes());
    FOR_RANGE(int64_t, j, 0, in_desc->shape().NumAxes()) {
      if (j == axis) {
        if (like_i_desc->is_dynamic()) {
          dynamic_dims = std::max(dynamic_dims, like_i_desc->shape().At(j));
        } else {
          static_dims += like_i_desc->shape().At(j);
        }
      } else {
        CHECK_EQ_OR_RETURN(in_desc->shape().At(j), like_i_desc->shape().At(j));
      }
    }
    *out_i_desc->mut_shape() = like_i_desc->shape();
    *out_i_desc->mut_data_type() = in_desc->data_type();
    out_i_desc->set_is_dynamic(like_i_desc->is_dynamic());
  }
  CHECK_LE_OR_RETURN(static_dims + dynamic_dims, in_desc->shape().At(axis));
  if (!in_desc->is_dynamic()) { CHECK_EQ_OR_RETURN(dynamic_dims, 0); }
  return Maybe<void>::Ok();
}

void SetLikeArgModifier(user_op::GetInputArgModifier GetInputArgModifierFn,
                        const user_op::UserOpConfWrapper& user_op_conf) {
  FOR_RANGE(int32_t, i, 0, user_op_conf.input_size("like")) {
    user_op::InputArgModifier* like_modifier = GetInputArgModifierFn("like", i);
    CHECK_NOTNULL(like_modifier);
    like_modifier->set_use_header_only(true);
    like_modifier->set_requires_grad(false);
  }
}

Maybe<void> InferBatchAxis(user_op::BatchAxisContext* ctx) {
  FOR_RANGE(int32_t, i, 0, ctx->outputs().size()) {
    *ctx->BatchAxis4ArgNameAndIndex("out", i) = *ctx->BatchAxis4ArgNameAndIndex("like", i);
  }
  return Maybe<void>::Ok();
}

Maybe<void> GetSbpSignature(user_op::SbpContext* ctx) {
  const int64_t axis = ctx->Attr<int64_t>("axis");
  const user_op::TensorDesc& in_desc = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
  FOR_RANGE(int64_t, i, 0, in_desc.shape().NumAxes()) {
    if (i == axis) { continue; }
    ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
  }
  std::vector<user_op::OpArg> like_arg_vec;
  const size_t like_arg_size = ctx->outputs().size();
  like_arg_vec.reserve(like_arg_size);
  FOR_RANGE(int32_t, i, 0, like_arg_size) { like_arg_vec.push_back(user_op::OpArg("like", i)); }
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("in", 0))
      .PartialSum(like_arg_vec)
      .PartialSum(ctx->outputs())
      .Build();
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("in", 0))
      .Broadcast(like_arg_vec)
      .PartialSum(ctx->outputs())
      .Build();
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_USER_OP("split_like")
    .Input("in")
    .InputWithMinimum("like", 2)
    .OutputWithMinimum("out", 2)
    .Attr("axis", UserOpAttrType::kAtInt64)
    .SetTensorDescInferFn(InferTensorDesc)
    .SetInputArgModifyFn(SetLikeArgModifier)
    .SetBatchAxisInferFn(InferBatchAxis)
    .SetGetSbpFn(GetSbpSignature);

}  // namespace oneflow
