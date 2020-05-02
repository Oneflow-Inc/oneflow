#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("gelu")
    .Input("in")
    .Output("out")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* in_shape = ctx->Shape4ArgNameAndIndex("in", 0);
      Shape* out_shape = ctx->Shape4ArgNameAndIndex("out", 0);
      *out_shape = *in_shape;
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("out", 0) = *ctx->BatchAxis4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
      SbpSignatureBuilder()
          .Split("in", 0, 0)
          .Split("out", 0, 0)
          .MakeSplitSignatureListBuilder(tensor.shape().NumAxes())
          .Build(ctx->sbp_sig_list());
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("gelu_grad")
    .Input("x")
    .Input("dy")
    .Output("dx")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* x_shape = ctx->Shape4ArgNameAndIndex("x", 0);
      const Shape* dy_shape = ctx->Shape4ArgNameAndIndex("dy", 0);
      Shape* dx_shape = ctx->Shape4ArgNameAndIndex("dx", 0);
      CHECK(*dy_shape == *x_shape);
      *dx_shape = *dy_shape;
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("dx", 0) = *ctx->BatchAxis4ArgNameAndIndex("x", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
      SbpSignatureBuilder()
          .Split("x", 0, 0)
          .Split("dy", 0, 0)
          .Split("dx", 0, 0)
          .MakeSplitSignatureListBuilder(tensor.shape().NumAxes())
          .Build(ctx->sbp_sig_list());
      SbpSignatureBuilder().Broadcast("x", 0).PartialSum("dy", 0).PartialSum("dx", 0).Build(
          ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("gelu").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                                                        user_op::AddOpFn AddOp) {
  if (op.NeedGenGradTensor4OpInput("in", 0)) {
    user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
    user_op::UserOpConfWrapper grad_op = builder.Op("gelu_grad")
                                             .Input("x", op.input("in", 0))
                                             .Input("dy", op.GetGradTensorWithOpOutput("out", 0))
                                             .Output("dx")
                                             .Build();
    op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "in", 0);
    AddOp(grad_op);
  }
});

}  // namespace oneflow
