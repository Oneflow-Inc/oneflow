#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("sparse_softmax_cross_entropy")
    .Input("prediction")
    .Input("label")
    .Output("prob")
    .Output("out")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* prediction_shape = ctx->Shape4ArgNameAndIndex("prediction", 0);
      const Shape* label_shape = ctx->Shape4ArgNameAndIndex("label", 0);
      Shape* out_shape = ctx->Shape4ArgNameAndIndex("out", 0);
      CHECK_OR_RETURN(IsIndexDataType(*ctx->Dtype4ArgNameAndIndex("label", 0)));
      const int64_t num_out_axes = prediction_shape->NumAxes() - 1;
      CHECK_GE_OR_RETURN(label_shape->NumAxes(), num_out_axes);
      CHECK_EQ_OR_RETURN(label_shape->Count(num_out_axes), 1);
      FOR_RANGE(int64_t, i, 0, num_out_axes) {
        CHECK_EQ_OR_RETURN(prediction_shape->At(i), label_shape->At(i));
      }

      *ctx->Dtype4ArgNameAndIndex("prob", 0) = *ctx->Dtype4ArgNameAndIndex("prediction", 0);
      *ctx->Shape4ArgNameAndIndex("prob", 0) = *ctx->Shape4ArgNameAndIndex("prediction", 0);

      *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("prediction", 0);
      *out_shape = Shape(
          DimVector(prediction_shape->dim_vec().cbegin(), prediction_shape->dim_vec().cend() - 1));

      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("prob", 0) = *ctx->BatchAxis4ArgNameAndIndex("prediction", 0);
      *ctx->BatchAxis4ArgNameAndIndex("out", 0) = *ctx->BatchAxis4ArgNameAndIndex("prediction", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      SbpSignatureBuilder()
          .Split("prediction", 0, 0)
          .Split("label", 0, 0)
          .Split("prob", 0, 0)
          .Split("out", 0, 0)
          .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("sparse_softmax_cross_entropy_grad")
    .Input("dy")
    .Input("label")
    .Input("prob")
    .Output("dx")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      // todo add check
      *ctx->Dtype4ArgNameAndIndex("dx", 0) = *ctx->Dtype4ArgNameAndIndex("prob", 0);
      *ctx->Shape4ArgNameAndIndex("dx", 0) = *ctx->Shape4ArgNameAndIndex("prob", 0);
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("dx", 0) = *ctx->BatchAxis4ArgNameAndIndex("prob", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      SbpSignatureBuilder()
          .Split("dy", 0, 0)
          .Split("label", 0, 0)
          .Split("prob", 0, 0)
          .Split("dx", 0, 0)
          .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("sparse_softmax_cross_entropy")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("prediction", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("sparse_softmax_cross_entropy_grad")
                .Input("prob", op.output("prob", 0))
                .Input("label", op.input("label", 0))
                .Input("dy", op.GetGradTensorWithOpOutput("out", 0))
                .Output("dx")
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "prediction", 0);
        AddOp(grad_op);
      }
    });

}  // namespace oneflow
