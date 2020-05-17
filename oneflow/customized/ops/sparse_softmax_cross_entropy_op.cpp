#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("sparse_softmax_cross_entropy")
    .Input("prediction")
    .Input("label")
    .Output("prob")  //'prob' is just for compute prediction's grad, prob's grad will be ignored
    .Output("out")
    .Attr("depth", UserOpAttrType::kAtInt64)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* prediction_desc = ctx->TensorDesc4ArgNameAndIndex("prediction", 0);
      const user_op::TensorDesc* label_desc = ctx->TensorDesc4ArgNameAndIndex("label", 0);
      CHECK_OR_RETURN(IsIndexDataType(label_desc->data_type()));
      CHECK_EQ_OR_RETURN(prediction_desc->is_dynamic(), label_desc->is_dynamic());
      CHECK_GE_OR_RETURN(prediction_desc->shape().NumAxes(), 2);
      const int64_t num_out_axes = prediction_desc->shape().NumAxes() - 1;
      CHECK_EQ_OR_RETURN(label_desc->shape().NumAxes(), num_out_axes);
      FOR_RANGE(int64_t, i, 0, num_out_axes) {
        CHECK_EQ_OR_RETURN(prediction_desc->shape().At(i), label_desc->shape().At(i));
      }
      *ctx->TensorDesc4ArgNameAndIndex("prob", 0) = *prediction_desc;
      user_op::TensorDesc* out_desc = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      *out_desc = *prediction_desc;
      *out_desc->mut_shape() = label_desc->shape();
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("prob", 0) = *ctx->BatchAxis4ArgNameAndIndex("prediction", 0);
      *ctx->BatchAxis4ArgNameAndIndex("out", 0) = *ctx->BatchAxis4ArgNameAndIndex("label", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder()
          .Split(user_op::OpArg("prediction", 0), 0)
          .Split(user_op::OpArg("label", 0), 0)
          .Split(user_op::OpArg("prob", 0), 0)
          .Split(user_op::OpArg("out", 0), 0)
          .Build();
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("sparse_softmax_cross_entropy_grad")
    .Input("dy")
    .Input("label")
    .Input("prob")
    .Output("prediction_diff")
    .Attr("depth", UserOpAttrType::kAtInt64)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* prob_desc = ctx->TensorDesc4ArgNameAndIndex("prob", 0);
      const user_op::TensorDesc* label_desc = ctx->TensorDesc4ArgNameAndIndex("label", 0);
      const user_op::TensorDesc* dy_desc = ctx->TensorDesc4ArgNameAndIndex("dy", 0);
      CHECK_OR_RETURN(IsIndexDataType(label_desc->data_type()));
      CHECK_EQ_OR_RETURN(prob_desc->is_dynamic(), label_desc->is_dynamic());
      CHECK_GE_OR_RETURN(prob_desc->shape().NumAxes(), 2);
      const int64_t num_out_axes = prob_desc->shape().NumAxes() - 1;
      CHECK_EQ_OR_RETURN(label_desc->shape().NumAxes(), num_out_axes);
      FOR_RANGE(int64_t, i, 0, num_out_axes) {
        CHECK_EQ_OR_RETURN(prob_desc->shape().At(i), label_desc->shape().At(i));
      }
      CHECK_EQ_OR_RETURN(dy_desc->shape(), label_desc->shape());
      CHECK_EQ_OR_RETURN(dy_desc->data_type(), prob_desc->data_type());
      *ctx->TensorDesc4ArgNameAndIndex("prediction_diff", 0) = *prob_desc;
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("prediction_diff", 0) =
          *ctx->BatchAxis4ArgNameAndIndex("prob", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder()
          .Split(user_op::OpArg("dy", 0), 0)
          .Split(user_op::OpArg("label", 0), 0)
          .Split(user_op::OpArg("prob", 0), 0)
          .Split(user_op::OpArg("prediction_diff", 0), 0)
          .Build();
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
                .Attr("depth", op.attr<int64_t>("depth"))
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("prediction_diff", 0), "prediction", 0);
        AddOp(grad_op);
      }
    });

REGISTER_USER_OP("sparse_softmax_cross_entropy_ms")
    .Input("prediction")
    .Input("label")
    .Output("prob")
    .Output("out")
    .Attr("depth", UserOpAttrType::kAtInt64)
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
      const user_op::TensorDesc& tensor =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("prediction", 0);
      ctx->NewBuilder()
          .Split(user_op::OpArg("prediction", 0), tensor.shape().NumAxes() - 1)
          .Split(user_op::OpArg("prob", 0), tensor.shape().NumAxes() - 1)
          .Broadcast(user_op::OpArg("label", 0))
          .PartialSum(user_op::OpArg("out", 0))
          .Build();
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("sparse_softmax_cross_entropy_ms_grad")
    .Input("dy")
    .Input("label")
    .Input("prob")
    .Output("prediction_diff")
    .Attr("depth", UserOpAttrType::kAtInt64)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->Dtype4ArgNameAndIndex("prediction_diff", 0) = *ctx->Dtype4ArgNameAndIndex("prob", 0);
      *ctx->Shape4ArgNameAndIndex("prediction_diff", 0) = *ctx->Shape4ArgNameAndIndex("prob", 0);
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("prediction_diff", 0) =
          *ctx->BatchAxis4ArgNameAndIndex("prob", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("prob", 0);
      ctx->NewBuilder()
          .Split(user_op::OpArg("prob", 0), tensor.shape().NumAxes() - 1)
          .Broadcast(user_op::OpArg("label", 0))
          .Broadcast(user_op::OpArg("dy", 0))
          .Split(user_op::OpArg("prediction_diff", 0), tensor.shape().NumAxes() - 1)
          .Build();
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("sparse_softmax_cross_entropy_ms")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("prediction", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("sparse_softmax_cross_entropy_ms_grad")
                .Input("prob", op.output("prob", 0))
                .Input("label", op.input("label", 0))
                .Input("dy", op.GetGradTensorWithOpOutput("out", 0))
                .Output("prediction_diff")
                .Attr("depth", op.attr<int64_t>("depth"))
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("prediction_diff", 0), "prediction", 0);
        AddOp(grad_op);
      }
    });

}  // namespace oneflow
