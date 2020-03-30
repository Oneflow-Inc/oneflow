#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("smooth_l1")
    .Input("x")
    .Input("label")
    .Output("y")
    .Attr("beta", UserOpAttrType::kAtFloat)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      CHECK_EQ_OR_RETURN(*ctx->Shape4ArgNameAndIndex("x", 0),
                         *ctx->Shape4ArgNameAndIndex("label", 0));
      CHECK_EQ_OR_RETURN(*ctx->Dtype4ArgNameAndIndex("x", 0),
                         *ctx->Dtype4ArgNameAndIndex("label", 0));
      CHECK_GE_OR_RETURN(ctx->GetAttr<float>("beta"), 0);
      *ctx->Shape4ArgNameAndIndex("y", 0) = *ctx->Shape4ArgNameAndIndex("x", 0);
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
          .Split("label", 0, 0)
          .Split("y", 0, 0)
          .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("smooth_l1_grad")
    .Input("dy")
    .Input("x")
    .Input("label")
    .Output("dx")
    .Attr("beta", UserOpAttrType::kAtFloat)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      CHECK_EQ_OR_RETURN(*ctx->Shape4ArgNameAndIndex("dy", 0), *ctx->Shape4ArgNameAndIndex("x", 0));
      CHECK_EQ_OR_RETURN(*ctx->Dtype4ArgNameAndIndex("dy", 0), *ctx->Dtype4ArgNameAndIndex("x", 0));
      CHECK_EQ_OR_RETURN(*ctx->Shape4ArgNameAndIndex("x", 0),
                         *ctx->Shape4ArgNameAndIndex("label", 0));
      CHECK_EQ_OR_RETURN(*ctx->Dtype4ArgNameAndIndex("x", 0),
                         *ctx->Dtype4ArgNameAndIndex("label", 0));
      CHECK_GE_OR_RETURN(ctx->GetAttr<float>("beta"), 0);
      *ctx->Shape4ArgNameAndIndex("dx", 0) = *ctx->Shape4ArgNameAndIndex("dy", 0);
      *ctx->Dtype4ArgNameAndIndex("dx", 0) = *ctx->Dtype4ArgNameAndIndex("dy", 0);
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
          .Split("label", 0, 0)
          .Split("dx", 0, 0)
          .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("smooth_l1")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op = builder.Op("smooth_l1_grad")
                                                 .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
                                                 .Input("x", op.input("x", 0))
                                                 .Input("label", op.input("label", 0))
                                                 .Output("dx")
                                                 .Attr("beta", op.attr<float>("beta"))
                                                 .Build();
        op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);
        AddOp(grad_op);
      }
    });

}  // namespace oneflow
