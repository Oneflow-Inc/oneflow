#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("sparse_softmax_cross_entropy")
    .Input("prediction")
    .Input("label")
    .Output("prob")
    .Output("out")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* prediction_desc = ctx->TensorDesc4ArgNameAndIndex("prediction", 0);
      const user_op::TensorDesc* label_desc = ctx->TensorDesc4ArgNameAndIndex("label", 0);
      CHECK_OR_RETURN(IsIndexDataType(label_desc->data_type()));
      CHECK_EQ_OR_RETURN(prediction_desc->is_dynamic(), label_desc->is_dynamic());
      CHECK_GE_OR_RETURN(prediction_desc->shape().NumAxes(), 2);
      const int64_t num_out_axes = prediction_desc->shape().NumAxes() - 1;
      CHECK_GE_OR_RETURN(label_desc->shape().NumAxes(), num_out_axes);
      CHECK_EQ_OR_RETURN(label_desc->shape().Count(num_out_axes), 1);
      FOR_RANGE(int64_t, i, 0, num_out_axes) {
        CHECK_EQ_OR_RETURN(prediction_desc->shape().At(i), label_desc->shape().At(i));
      }
      *ctx->TensorDesc4ArgNameAndIndex("prob", 0) =
          *ctx->TensorDesc4ArgNameAndIndex("prediction", 0);
      user_op::TensorDesc* out_desc = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      *out_desc = *prediction_desc;
      *out_desc->mut_shape() = Shape(DimVector(prediction_desc->shape().dim_vec().cbegin(),
                                               prediction_desc->shape().dim_vec().cend() - 1));
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
    .Output("prediction_diff")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->TensorDesc4ArgNameAndIndex("prediction_diff", 0) =
          *ctx->TensorDesc4ArgNameAndIndex("prob", 0);
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("prediction_diff", 0) =
          *ctx->BatchAxis4ArgNameAndIndex("prob", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      SbpSignatureBuilder()
          .Split("dy", 0, 0)
          .Split("label", 0, 0)
          .Split("prob", 0, 0)
          .Split("prediction_diff", 0, 0)
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
                .Output("prediction_diff")
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("prediction_diff", 0), "prediction", 0);
        AddOp(grad_op);
      }
    });

}  // namespace oneflow
