#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

Maybe<void> InferWhereTensorDesc(user_op::InferContext* ctx) {
  // shape infer
  const Shape* cond_shape = ctx->Shape4ArgNameAndIndex("condition", 0);
  OF_CHECK_EQ(*cond_shape, *ctx->Shape4ArgNameAndIndex("x", 0));
  OF_CHECK_EQ(*cond_shape, *ctx->Shape4ArgNameAndIndex("y", 0));
  *ctx->Shape4ArgNameAndIndex("out", 0) = *cond_shape;
  // data_type infer
  DataType cond_dtype = *ctx->Dtype4ArgNameAndIndex("condition", 0);
  OF_CHECK(IsIntegralDataType(cond_dtype));
  DataType x_dtype = *ctx->Dtype4ArgNameAndIndex("x", 0);
  OF_CHECK_EQ(x_dtype, *ctx->Dtype4ArgNameAndIndex("y", 0));
  *ctx->Dtype4ArgNameAndIndex("out", 0) = x_dtype;
  return Maybe<void>::Ok();
}

Maybe<void> InferWhereBatchAxis(user_op::BatchAxisContext* ctx) {
  OptInt64* x_batch_axis = ctx->BatchAxis4ArgNameAndIndex("x", 0);
  OptInt64* y_batch_axis = ctx->BatchAxis4ArgNameAndIndex("y", 0);
  OF_CHECK((x_batch_axis->has_value() && y_batch_axis->has_value())
           || (!x_batch_axis->has_value() && !y_batch_axis->has_value()));
  if (x_batch_axis->has_value()) {
    OF_CHECK_EQ(x_batch_axis->value(), y_batch_axis->value());
    ctx->BatchAxis4ArgNameAndIndex("out", 0)->set_value(x_batch_axis->value());
  } else {
    ctx->BatchAxis4ArgNameAndIndex("out", 0)->clear_value();
  }
  return Maybe<void>::Ok();
}

Maybe<void> GetWhereSbpSignatures(user_op::SbpContext* ctx) {
  int64_t num_axes = ctx->LogicalTensorDesc4InputArgNameAndIndex("out", 0).shape().NumAxes();
  SbpSignatureBuilder()
      .Split("condition", 0, 0)
      .Split("x", 0, 0)
      .Split("y", 0, 0)
      .Split("out", 0, 0)
      .MakeSplitSignatureListBuilder(num_axes)
      .Build(ctx->sbp_sig_list());
  SbpSignatureBuilder()
      .Broadcast("condition", 0)
      .PartialSum("x", 0)
      .PartialSum("y", 0)
      .PartialSum("out", 0)
      .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_USER_OP("where")
    .Input("condition")
    .Input("x")
    .Input("y")
    .Output("out")
    .SetTensorDescInferFn(InferWhereTensorDesc)
    .SetBatchAxisInferFn(InferWhereBatchAxis)
    .SetGetSbpFn(GetWhereSbpSignatures);

REGISTER_USER_OP_GRAD("where").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                                                         user_op::AddOpFn AddOp) {
  if (op.NeedGenGradTensor4OpInput("x", 0)) {
    user_op::UserOpConfWrapperBuilder zero_y_builder(op.op_name() + "_zero_y");
    user_op::UserOpConfWrapper zero_y_op =
        zero_y_builder.Op("zero_like").Input("like", op.input("y", 0)).Output("out").Build();
    AddOp(zero_y_op);
    user_op::UserOpConfWrapperBuilder x_grad_builder(op.op_name() + "_x_grad");
    user_op::UserOpConfWrapper x_grad_op = x_grad_builder.Op("where")
                                               .Input("condition", op.input("condition", 0))
                                               .Input("x", op.GetGradTensorWithOpOutput("out", 0))
                                               .Input("y", zero_y_op.output("out", 0))
                                               .Output("out")
                                               .Build();
    op.BindGradTensorWithOpInput(x_grad_op.output("out", 0), "x", 0);
    AddOp(x_grad_op);
  }
  if (op.NeedGenGradTensor4OpInput("y", 0)) {
    user_op::UserOpConfWrapperBuilder zero_x_builder(op.op_name() + "_zero_x");
    user_op::UserOpConfWrapper zero_x_op =
        zero_x_builder.Op("zero_like").Input("like", op.input("x", 0)).Output("out").Build();
    AddOp(zero_x_op);
    user_op::UserOpConfWrapperBuilder y_grad_builder(op.op_name() + "_y_grad");
    user_op::UserOpConfWrapper y_grad_op = y_grad_builder.Op("where")
                                               .Input("condition", op.input("condition", 0))
                                               .Input("x", zero_x_op.output("out", 0))
                                               .Input("y", op.GetGradTensorWithOpOutput("out", 0))
                                               .Output("out")
                                               .Build();
    op.BindGradTensorWithOpInput(y_grad_op.output("out", 0), "y", 0);
    AddOp(y_grad_op);
  }
});

}  // namespace oneflow
