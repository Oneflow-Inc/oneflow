#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("prelu")
    .Input("x")
    .Input("alpha")
    .Output("y")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* x_desc = ctx->TensorDesc4ArgNameAndIndex("x", 0);
      user_op::TensorDesc* y_desc = ctx->TensorDesc4ArgNameAndIndex("y", 0);
      const Shape* alpha_shape = ctx->Shape4ArgNameAndIndex("alpha", 0);
      CHECK_EQ_OR_RETURN(x_desc->shape().NumAxes(), alpha_shape->NumAxes() + 1);
      FOR_RANGE(int64_t, i, 1, x_desc->shape().NumAxes()) {
        CHECK_OR_RETURN((alpha_shape->At(i - 1) == x_desc->shape().At(i))
                        || (alpha_shape->At(i - 1) == 1));
      }
      *y_desc = *x_desc;
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("y", 0) = *ctx->BatchAxis4ArgNameAndIndex("x", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& x_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
      const user_op::TensorDesc& alpha_tensor =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("alpha", 0);
      ctx->NewBuilder()
          .Split(user_op::OpArg("x", 0), 0)
          .Broadcast(user_op::OpArg("alpha", 0))
          .Split(user_op::OpArg("y", 0), 0)
          .Build();
      FOR_RANGE(int64_t, i, 1, x_tensor.shape().NumAxes()) {
        if (x_tensor.shape().At(i) == alpha_tensor.shape().At(i - 1)) {
          ctx->NewBuilder()
              .Split(user_op::OpArg("x", 0), i)
              .Split(user_op::OpArg("alpha", 0), i - 1)
              .Split(user_op::OpArg("y", 0), i)
              .Build();
        }
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("prelu_x_grad")
    .Input("dy")
    .Input("x")
    .Input("alpha")
    .Output("dx")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* x_desc = ctx->TensorDesc4ArgNameAndIndex("x", 0);
      const user_op::TensorDesc* dy_desc = ctx->TensorDesc4ArgNameAndIndex("dy", 0);
      user_op::TensorDesc* dx_desc = ctx->TensorDesc4ArgNameAndIndex("dx", 0);
      const Shape* alpha_shape = ctx->Shape4ArgNameAndIndex("alpha", 0);
      CHECK_EQ_OR_RETURN(x_desc->shape().NumAxes(), alpha_shape->NumAxes() + 1);
      FOR_RANGE(int64_t, i, 1, x_desc->shape().NumAxes()) {
        CHECK_OR_RETURN((alpha_shape->At(i - 1) == x_desc->shape().At(i))
                        || (alpha_shape->At(i - 1) == 1));
      }
      CHECK_EQ_OR_RETURN(dy_desc->shape(), x_desc->shape());
      CHECK_EQ_OR_RETURN(dy_desc->data_type(), x_desc->data_type());
      *dx_desc = *x_desc;
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("dx", 0) = *ctx->BatchAxis4ArgNameAndIndex("x", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& x_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
      const user_op::TensorDesc& alpha_tensor =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("alpha", 0);
      ctx->NewBuilder()
          .Split(user_op::OpArg("dy", 0), 0)
          .Split(user_op::OpArg("x", 0), 0)
          .Broadcast(user_op::OpArg("alpha", 0))
          .Split(user_op::OpArg("dx", 0), 0)
          .Build();
      ctx->NewBuilder()
          .PartialSum(user_op::OpArg("dy", 0))
          .Broadcast(user_op::OpArg("x", 0))
          .Broadcast(user_op::OpArg("alpha", 0))
          .PartialSum(user_op::OpArg("dx", 0))
          .Build();
      FOR_RANGE(int64_t, i, 1, x_tensor.shape().NumAxes()) {
        if (x_tensor.shape().At(i) == alpha_tensor.shape().At(i - 1)) {
          ctx->NewBuilder()
              .Split(user_op::OpArg("dy", 0), i)
              .Split(user_op::OpArg("x", 0), i)
              .Split(user_op::OpArg("alpha", 0), i - 1)
              .Split(user_op::OpArg("dx", 0), i)
              .Build();
        }
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("prelu_alpha_grad")
    .Input("dy")
    .Input("x")
    .Input("alpha")
    .Output("alpha_diff")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* x_desc = ctx->TensorDesc4ArgNameAndIndex("x", 0);
      const user_op::TensorDesc* dy_desc = ctx->TensorDesc4ArgNameAndIndex("dy", 0);
      const user_op::TensorDesc* alpha_desc = ctx->TensorDesc4ArgNameAndIndex("alpha", 0);
      CHECK_EQ_OR_RETURN(x_desc->shape().NumAxes(), alpha_desc->shape().NumAxes() + 1);
      FOR_RANGE(int64_t, i, 1, x_desc->shape().NumAxes()) {
        CHECK_OR_RETURN((alpha_desc->shape().At(i - 1) == x_desc->shape().At(i))
                        || (alpha_desc->shape().At(i - 1) == 1));
      }
      CHECK_EQ_OR_RETURN(dy_desc->shape(), x_desc->shape());
      CHECK_EQ_OR_RETURN(dy_desc->data_type(), x_desc->data_type());
      *ctx->TensorDesc4ArgNameAndIndex("alpha_diff", 0) = *alpha_desc;
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("alpha_diff", 0) =
          *ctx->BatchAxis4ArgNameAndIndex("alpha", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& x_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
      const user_op::TensorDesc& alpha_tensor =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("alpha", 0);
      ctx->NewBuilder()
          .Split(user_op::OpArg("dy", 0), 0)
          .Split(user_op::OpArg("x", 0), 0)
          .Broadcast(user_op::OpArg("alpha", 0))
          .PartialSum(user_op::OpArg("alpha_diff", 0))
          .Build();
      ctx->NewBuilder()
          .PartialSum(user_op::OpArg("dy", 0))
          .Broadcast(user_op::OpArg("x", 0))
          .Broadcast(user_op::OpArg("alpha", 0))
          .PartialSum(user_op::OpArg("alpha_diff", 0))
          .Build();
      FOR_RANGE(int64_t, i, 1, x_tensor.shape().NumAxes()) {
        if (x_tensor.shape().At(i) == alpha_tensor.shape().At(i - 1)) {
          ctx->NewBuilder()
              .Split(user_op::OpArg("dy", 0), i)
              .Split(user_op::OpArg("x", 0), i)
              .Split(user_op::OpArg("alpha", 0), i - 1)
              .Split(user_op::OpArg("alpha_diff", 0), i - 1)
              .Build();
        }
      }
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
