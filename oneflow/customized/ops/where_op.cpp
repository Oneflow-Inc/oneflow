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

Maybe<void> InferWhereGradTensorDesc(user_op::InferContext* ctx) {
  // shape infer
  const Shape* cond_shape = ctx->Shape4ArgNameAndIndex("condition", 0);
  const Shape* dz_shape = ctx->Shape4ArgNameAndIndex("dz", 0);
  OF_CHECK_EQ(*cond_shape, *dz_shape);
  *ctx->Shape4ArgNameAndIndex("dx", 0) = *dz_shape;
  *ctx->Shape4ArgNameAndIndex("dy", 0) = *dz_shape;
  // data_type infer
  DataType cond_dtype = *ctx->Dtype4ArgNameAndIndex("condition", 0);
  OF_CHECK(IsIntegralDataType(cond_dtype));
  DataType dz_dtype = *ctx->Dtype4ArgNameAndIndex("dz", 0);
  *ctx->Dtype4ArgNameAndIndex("dx", 0) = dz_dtype;
  *ctx->Dtype4ArgNameAndIndex("dy", 0) = dz_dtype;
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

Maybe<void> InferWhereGradBatchAxis(user_op::BatchAxisContext* ctx) {
  *ctx->BatchAxis4ArgNameAndIndex("dx", 0) = *ctx->BatchAxis4ArgNameAndIndex("dz", 0);
  *ctx->BatchAxis4ArgNameAndIndex("dy", 0) = *ctx->BatchAxis4ArgNameAndIndex("dz", 0);
  return Maybe<void>::Ok();
}

Maybe<void> GetWhereGradSbpSignatures(user_op::SbpContext* ctx) {
  int64_t num_axes = ctx->LogicalTensorDesc4InputArgNameAndIndex("dz", 0).shape().NumAxes();
  SbpSignatureBuilder()
      .Split("condition", 0, 0)
      .Split("dz", 0, 0)
      .Split("dx", 0, 0)
      .Split("dy", 0, 0)
      .MakeSplitSignatureListBuilder(num_axes)
      .Build(ctx->sbp_sig_list());
  SbpSignatureBuilder()
      .Broadcast("condition", 0)
      .PartialSum("dz", 0)
      .PartialSum("dx", 0)
      .PartialSum("dy", 0)
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

REGISTER_USER_OP("where_grad")
    .Input("condition")
    .Input("dz")
    .Output("dx")
    .Output("dy")
    .SetTensorDescInferFn(InferWhereGradTensorDesc)
    .SetBatchAxisInferFn(InferWhereGradBatchAxis)
    .SetGetSbpFn(GetWhereGradSbpSignatures);

REGISTER_USER_OP_GRAD("where").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                                                         user_op::AddOpFn AddOp) {
  bool x_need_grad = op.NeedGenGradTensor4OpInput("x", 0);
  bool y_need_grad = op.NeedGenGradTensor4OpInput("y", 0);
  if (x_need_grad || y_need_grad) {
    user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
    user_op::UserOpConfWrapper grad_op = builder.Op("where_grad")
                                             .Input("condition", op.input("condition", 0))
                                             .Input("dz", op.GetGradTensorWithOpOutput("out", 0))
                                             .Output("dx")
                                             .Output("dy")
                                             .Build();
    if (x_need_grad) { op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0); }
    if (y_need_grad) { op.BindGradTensorWithOpInput(grad_op.output("dy", 0), "y", 0); }
    AddOp(grad_op);
  }
});

}  // namespace oneflow
