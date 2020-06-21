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
    CHECK_LE_OR_RETURN(in_desc->shape().At(axis), max_dims);
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

void GenDynamicConcatGrapOp(const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
  bool need_grad = false;
  int32_t in_size = op.input_size("in");
  for (size_t i = 0; i < in_size; ++i) {
    if (op.NeedGenGradTensor4OpInput("in", i)) { need_grad = true; }
  }
  if (need_grad) {
    user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
    builder = builder.Op("split_like");
    for (size_t i = 0; i < in_size; ++i) { builder = builder.Input("like", op.input("in", i)); }
    user_op::UserOpConfWrapper grad_op =
        builder.Input("in", op.GetGradTensorWithOpOutput("out", 0))
            .Output("out", in_size)
            .Attr("axis", static_cast<int32_t>(op.attr<int64_t>("axis")))
            .Build();

    for (size_t i = 0; i < in_size; ++i) {
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

REGISTER_USER_OP_GRAD("dynamic_concat").SetGenBackwardOpConfFn(GenDynamicConcatGrapOp);

}  // namespace oneflow
