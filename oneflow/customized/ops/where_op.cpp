#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

Maybe<void> InferWhereTensorDesc(user_op::InferContext* ctx) {
  const Shape* cond_shape = ctx->Shape4ArgNameAndIndex("condition", 0);
  CHECK_EQ_OR_RETURN(*cond_shape, *ctx->Shape4ArgNameAndIndex("x", 0));
  CHECK_EQ_OR_RETURN(*cond_shape, *ctx->Shape4ArgNameAndIndex("y", 0));
  *ctx->Shape4ArgNameAndIndex("out", 0) = *cond_shape;
  DataType cond_dtype = *ctx->Dtype4ArgNameAndIndex("condition", 0);
  CHECK_OR_RETURN(IsIntegralDataType(cond_dtype));
  DataType x_dtype = *ctx->Dtype4ArgNameAndIndex("x", 0);
  CHECK_EQ_OR_RETURN(x_dtype, *ctx->Dtype4ArgNameAndIndex("y", 0));
  *ctx->Dtype4ArgNameAndIndex("out", 0) = x_dtype;
  return Maybe<void>::Ok();
}

Maybe<void> GetWhereSbpSignatures(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& condition_tensor =
      ctx->LogicalTensorDesc4InputArgNameAndIndex("condition", 0);
  FOR_RANGE(int64_t, i, 0, condition_tensor.shape().NumAxes()) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("condition", 0), i)
        .Split(user_op::OpArg("x", 0), i)
        .Split(user_op::OpArg("y", 0), i)
        .Split(user_op::OpArg("out", 0), i)
        .Build();
  }
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("condition", 0))
      .PartialSum(user_op::OpArg("x", 0))
      .PartialSum(user_op::OpArg("y", 0))
      .PartialSum(user_op::OpArg("out", 0))
      .Build();
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_USER_OP("where")
    .Input("condition")
    .Input("x")
    .Input("y")
    .Output("out")
    .SetTensorDescInferFn(InferWhereTensorDesc)
    .SetBatchAxisInferFn(user_op::BatchAxisInferFnUtil::NaiveInferBatchAxis)
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn) {
      user_op::InputArgModifier* cond_arg_modifier = GetInputArgModifierFn("condition", 0);
      cond_arg_modifier->set_requires_grad(false);
    })
    .SetGetSbpFn(GetWhereSbpSignatures);

REGISTER_USER_OP_GRAD("where").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                                                         user_op::AddOpFn AddOp) {
  bool need_grad_x = op.NeedGenGradTensor4OpInput("x", 0);
  bool need_grad_y = op.NeedGenGradTensor4OpInput("y", 0);
  if (need_grad_x || need_grad_y) {
    user_op::UserOpConfWrapperBuilder zero_builder(op.op_name() + "_zero_grad");
    user_op::UserOpConfWrapper zero_op =
        zero_builder.Op("zero_like").Input("like", op.input("x", 0)).Output("out").Build();
    AddOp(zero_op);
    if (need_grad_x) {
      user_op::UserOpConfWrapperBuilder x_grad_builder(op.op_name() + "_x_grad");
      user_op::UserOpConfWrapper x_grad_op = x_grad_builder.Op("where")
                                                 .Input("condition", op.input("condition", 0))
                                                 .Input("x", op.GetGradTensorWithOpOutput("out", 0))
                                                 .Input("y", zero_op.output("out", 0))
                                                 .Output("out")
                                                 .Build();
      op.BindGradTensorWithOpInput(x_grad_op.output("out", 0), "x", 0);
      AddOp(x_grad_op);
    }
    if (need_grad_y) {
      user_op::UserOpConfWrapperBuilder y_grad_builder(op.op_name() + "_y_grad");
      user_op::UserOpConfWrapper y_grad_op = y_grad_builder.Op("where")
                                                 .Input("condition", op.input("condition", 0))
                                                 .Input("x", zero_op.output("out", 0))
                                                 .Input("y", op.GetGradTensorWithOpOutput("out", 0))
                                                 .Output("out")
                                                 .Build();
      op.BindGradTensorWithOpInput(y_grad_op.output("out", 0), "y", 0);
      AddOp(y_grad_op);
    }
  }
});

}  // namespace oneflow
