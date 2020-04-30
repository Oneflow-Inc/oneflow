#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("prelu")
    .Input("x")
    .Input("alpha")
    .Output("y")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* x_shape = ctx->Shape4ArgNameAndIndex("x", 0);
      Shape* y_shape = ctx->Shape4ArgNameAndIndex("y", 0);
      CHECK_EQ_OR_RETURN(x_shape->NumAxes(), ctx->Shape4ArgNameAndIndex("alpha", 0)->NumAxes() + 1);
      *y_shape = *x_shape;
      *ctx->Dtype4ArgNameAndIndex("y", 0) = *ctx->Dtype4ArgNameAndIndex("x", 0);
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("y", 0) = *ctx->BatchAxis4ArgNameAndIndex("x", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      SbpSignatureBuilder()
          .Split("x", 0, 0)
          .Broadcast("alpha", 0)
          .Split("y", 0, 0)
          .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("prelu_x_grad")
    .Input("dy")
    .Input("x")
    .Input("alpha")
    .Output("dx")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* x_shape = ctx->Shape4ArgNameAndIndex("x", 0);
      const Shape* dy_shape = ctx->Shape4ArgNameAndIndex("dy", 0);
      Shape* dx_shape = ctx->Shape4ArgNameAndIndex("dx", 0);
      CHECK_EQ_OR_RETURN(x_shape->NumAxes(), ctx->Shape4ArgNameAndIndex("alpha", 0)->NumAxes() + 1);
      CHECK_EQ_OR_RETURN(*dy_shape, *x_shape);
      *dx_shape = *dy_shape;
      *ctx->Dtype4ArgNameAndIndex("dx", 0) = *ctx->Dtype4ArgNameAndIndex("x", 0);
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("dx", 0) = *ctx->BatchAxis4ArgNameAndIndex("x", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      SbpSignatureBuilder()
          .Split("dy", 0, 0)
          .Split("x", 0, 0)
          .Broadcast("alpha", 0)
          .Split("dx", 0, 0)
          .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("prelu_alpha_grad")
    .Input("dy")
    .Input("x")
    .Input("alpha")
    .Output("alpha_diff")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* x_shape = ctx->Shape4ArgNameAndIndex("x", 0);
      const Shape* dy_shape = ctx->Shape4ArgNameAndIndex("dy", 0);
      CHECK_EQ_OR_RETURN(x_shape->NumAxes(), ctx->Shape4ArgNameAndIndex("alpha", 0)->NumAxes() + 1);
      CHECK_EQ_OR_RETURN(*dy_shape, *x_shape);
      *ctx->Shape4ArgNameAndIndex("alpha_diff", 0) = *ctx->Shape4ArgNameAndIndex("alpha", 0);
      *ctx->Dtype4ArgNameAndIndex("alpha_diff", 0) = *ctx->Dtype4ArgNameAndIndex("alpha", 0);
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("alpha_diff", 0) =
          *ctx->BatchAxis4ArgNameAndIndex("alpha", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      SbpSignatureBuilder()
          .Split("dy", 0, 0)
          .Split("x", 0, 0)
          .Broadcast("alpha", 0)
          .PartialSum("alpha_diff", 0)
          .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("prelu").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                                                         user_op::AddOpFn AddOp) {
  if (op.NeedGenGradTensor4OpInput("x", 0)) {
    user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_x_grad");
    user_op::UserOpConfWrapper x_grad_op = builder.Op("prelu_x_grad")
                                               .Input("x", op.input("x", 0))
                                               .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
                                               .Input("alpha", op.input("alpha", 0))
                                               .Output("dx")
                                               .Build();
    op.BindGradTensorWithOpInput(x_grad_op.output("dx", 0), "x", 0);
    AddOp(x_grad_op);
  }
  if (op.NeedGenGradTensor4OpInput("alpha", 0)) {
    user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_alpha_grad");
    user_op::UserOpConfWrapper alpha_grad_op =
        builder.Op("prelu_alpha_grad")
            .Input("x", op.input("x", 0))
            .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
            .Input("alpha", op.input("alpha", 0))
            .Output("alpha_diff")
            .Build();
    op.BindGradTensorWithOpInput(alpha_grad_op.output("alpha_diff", 0), "alpha", 0);
    AddOp(alpha_grad_op);
  }
});

}  // namespace oneflow
