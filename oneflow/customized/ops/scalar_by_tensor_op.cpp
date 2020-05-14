#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

Maybe<void> InferBroadcastTensorDescFn(user_op::InferContext* ctx) {
  const user_op::TensorDesc* x = ctx->TensorDesc4ArgNameAndIndex("x", 0);
  const user_op::TensorDesc* scalar = ctx->TensorDesc4ArgNameAndIndex("scalar", 0);
  CHECK_EQ_OR_RETURN(x->data_type(), scalar->data_type());
  CHECK_EQ_OR_RETURN(scalar->shape().elem_cnt(), 1);
  user_op::TensorDesc* y = ctx->TensorDesc4ArgNameAndIndex("y", 0);
  *y = *x;
  return Maybe<void>::Ok();
}

Maybe<void> GetBasicSbpSignature(user_op::SbpContext* ctx) {
  const auto& x = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  FOR_RANGE(int64_t, i, 0, x.shape().NumAxes()) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("x", 0), i)
        .Split(user_op::OpArg("y", 0), i)
        .Broadcast(user_op::OpArg("scalar", 0))
        .Build();
  }
  return Maybe<void>::Ok();
}

using GetSbpFn = std::function<Maybe<void>(user_op::SbpContext*)>;
GetSbpFn MakeGetSbpFn(GetSbpFn extra) {
  return [extra](user_op::SbpContext* ctx) -> Maybe<void> {
    JUST(extra(ctx));
    GetBasicSbpSignature(ctx);
    return Maybe<void>::Ok();
  };
}

}  // namespace

REGISTER_USER_OP("scalar_add_by_tensor")
    .Input("x")
    .Input("scalar")
    .Output("y")

    .SetTensorDescInferFn(InferBroadcastTensorDescFn)
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("y", 0) = *ctx->BatchAxis4ArgNameAndIndex("x", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn(MakeGetSbpFn([](user_op::SbpContext* ctx) {
      ctx->NewBuilder()
          .PartialSum(user_op::OpArg("x", 0))
          .PartialSum(user_op::OpArg("scalar", 0))
          .PartialSum(user_op::OpArg("y", 0))
          .Build();
      return Maybe<void>::Ok();
    }));

REGISTER_USER_OP_GRAD("scalar_add_by_tensor")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        op.BindGradTensorWithOpInput(op.GetGradTensorWithOpOutput("y", 0), "x", 0);
      }
    });

}  // namespace oneflow
