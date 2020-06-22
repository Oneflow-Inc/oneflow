#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

Maybe<void> DynamicConcatInferTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc* first_in_desc = ctx->TensorDesc4ArgNameAndIndex("in", 0);
  const int64_t axis = ctx->Attr<int64_t>("axis");
  CHECK_GE_OR_RETURN(axis, 0);
  CHECK_LT_OR_RETURN(axis, first_in_desc->shape().NumAxes());
  const int64_t max_dims = ctx->Attr<int64_t>("max_dims");
  for (const auto& in_arg_pair : ctx->inputs()) {
    const user_op::TensorDesc* in_desc =
        ctx->TensorDesc4ArgNameAndIndex(in_arg_pair.first, in_arg_pair.second);
    CHECK_EQ_OR_RETURN(in_desc->shape().NumAxes(), first_in_desc->shape().NumAxes());
    FOR_RANGE(int64_t, i, 0, in_desc->shape().NumAxes()) {
      if (i != axis) { CHECK_EQ_OR_RETURN(in_desc->shape().At(i), first_in_desc->shape().At(i)); }
    }
    CHECK_EQ_OR_RETURN(in_desc->data_type(), first_in_desc->data_type());
    CHECK_OR_RETURN(in_desc->is_dynamic());
  }
  DimVector dim_vec = first_in_desc->shape().dim_vec();
  dim_vec.at(axis) = max_dims;
  user_op::TensorDesc* out_desc = ctx->TensorDesc4ArgNameAndIndex("out", 0);
  *out_desc->mut_shape() = Shape(dim_vec);
  *out_desc->mut_data_type() = first_in_desc->data_type();
  out_desc->set_is_dynamic(true);
  return Maybe<void>::Ok();
}

Maybe<void> DynamicConcatGetSbpSignature(user_op::SbpContext* ctx) {
  const int64_t axis = ctx->Attr<int64_t>("axis");
  const user_op::TensorDesc& first_in_desc = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
  FOR_RANGE(int64_t, i, 0, first_in_desc.shape().NumAxes()) {
    if (i == axis) { continue; }
    ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
  }
  ctx->NewBuilder().PartialSum(ctx->inputs()).PartialSum(ctx->outputs()).Build();
  return Maybe<void>::Ok();
}

Maybe<void> DynamicSplitLikeInferTensorDesc(user_op::InferContext* ctx) {
  const int64_t axis = ctx->Attr<int64_t>("axis");
  const user_op::TensorDesc* in_desc = ctx->TensorDesc4ArgNameAndIndex("in", 0);
  CHECK_OR_RETURN(in_desc->is_dynamic());
  FOR_RANGE(int32_t, i, 0, ctx->outputs().size()) {
    const user_op::TensorDesc* like_i_desc = ctx->TensorDesc4ArgNameAndIndex("like", i);
    user_op::TensorDesc* out_i_desc = ctx->TensorDesc4ArgNameAndIndex("out", i);
    CHECK_EQ_OR_RETURN(like_i_desc->shape().NumAxes(), in_desc->shape().NumAxes());
    FOR_RANGE(int64_t, j, 0, in_desc->shape().NumAxes()) {
      if (j != axis) { CHECK_EQ_OR_RETURN(in_desc->shape().At(j), like_i_desc->shape().At(j)); }
    }
    *out_i_desc->mut_shape() = like_i_desc->shape();
    *out_i_desc->mut_data_type() = in_desc->data_type();
    out_i_desc->set_is_dynamic(true);
  }
  return Maybe<void>::Ok();
}

void SetDynamicSplitLikeModifier(user_op::GetInputArgModifier GetInputArgModifierFn,
                                 const user_op::UserOpConfWrapper& user_op_conf) {
  FOR_RANGE(int32_t, i, 0, user_op_conf.input_size("like")) {
    user_op::InputArgModifier* like_modifier = GetInputArgModifierFn("like", i);
    CHECK_NOTNULL(like_modifier);
    like_modifier->set_use_header_only(true);
    like_modifier->set_requires_grad(false);
  }
}

Maybe<void> DynamicSplitLikeInferBatchAxis(user_op::BatchAxisContext* ctx) {
  FOR_RANGE(int32_t, i, 0, ctx->outputs().size()) {
    *ctx->BatchAxis4ArgNameAndIndex("out", i) = *ctx->BatchAxis4ArgNameAndIndex("like", i);
  }
  return Maybe<void>::Ok();
}

Maybe<void> DynamicSplitLikeGetSbpSignature(user_op::SbpContext* ctx) {
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

void GenDynamicConcatGrapOp(const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
  bool need_grad = false;
  const int32_t in_size = op.input_size("in");
  FOR_RANGE(int32_t, i, 0, in_size) {
    if (op.NeedGenGradTensor4OpInput("in", i)) { need_grad = true; }
  }
  if (need_grad) {
    user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
    builder = builder.Op("dynamic_split_like");
    FOR_RANGE(int32_t, i, 0, in_size) { builder = builder.Input("like", op.input("in", i)); }
    user_op::UserOpConfWrapper grad_op = builder.Input("in", op.GetGradTensorWithOpOutput("out", 0))
                                             .Output("out", in_size)
                                             .Attr("axis", op.attr<int64_t>("axis"))
                                             .Build();

    FOR_RANGE(int32_t, i, 0, in_size) {
      if (op.NeedGenGradTensor4OpInput("in", i)) {
        op.BindGradTensorWithOpInput(grad_op.output("out", i), "in", i);
      }
    }
    AddOp(grad_op);
  }
}

}  // namespace

REGISTER_USER_OP("dynamic_concat")
    .InputWithMinimum("in", 2)
    .Output("out")
    .Attr("axis", UserOpAttrType::kAtInt64)
    .Attr("max_dims", UserOpAttrType::kAtInt64)
    .SetTensorDescInferFn(DynamicConcatInferTensorDesc)
    .SetBatchAxisInferFn(user_op::BatchAxisInferFnUtil::NaiveInferBatchAxis)
    .SetGetSbpFn(DynamicConcatGetSbpSignature);

REGISTER_USER_OP("dynamic_split_like")
    .Input("in")
    .InputWithMinimum("like", 2)
    .OutputWithMinimum("out", 2)
    .Attr("axis", UserOpAttrType::kAtInt64)
    .SetTensorDescInferFn(DynamicSplitLikeInferTensorDesc)
    .SetInputArgModifyFn(SetDynamicSplitLikeModifier)
    .SetBatchAxisInferFn(DynamicSplitLikeInferBatchAxis)
    .SetGetSbpFn(DynamicSplitLikeGetSbpSignature);

REGISTER_USER_OP_GRAD("dynamic_concat").SetGenBackwardOpConfFn(GenDynamicConcatGrapOp);

}  // namespace oneflow
