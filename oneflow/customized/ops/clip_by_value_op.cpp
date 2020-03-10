#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("clip_by_value")
    .Input("x")
    .OptionalInput("min")
    .OptionalInput("max")
    .Output("y")
    .SetShapeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* min_shape = ctx->Shape4ArgNameAndIndex("min", 0);
      Shape* max_shape = ctx->Shape4ArgNameAndIndex("max", 0);
      OF_CHECK_EQ(min_shape->NumAxes(), 1);
      OF_CHECK_EQ(max_shape->NumAxes(), 1);
      OF_CHECK_EQ(min_shape->At(0), 1);
      OF_CHECK_EQ(max_shape->At(0), 1);
      *ctx->Shape4ArgNameAndIndex("y", 0) = *ctx->Shape4ArgNameAndIndex("x", 0);
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      DataType x_data_type = *ctx->Dtype4ArgNameAndIndex("x", 0);
      OF_CHECK_EQ(x_data_type, *ctx->Dtype4ArgNameAndIndex("min", 0));
      OF_CHECK_EQ(x_data_type, *ctx->Dtype4ArgNameAndIndex("max", 0));
      *ctx->Dtype4ArgNameAndIndex("y", 0) = x_data_type;
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("y", 0) = *ctx->BatchAxis4ArgNameAndIndex("x", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      int64_t num_axes = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0).shape().NumAxes();
      FOR_RANGE(int64_t, axis, 0, num_axes) {
        SbpSignatureBuilder()
            .Split("x", 0, axis)
            .Broadcast("min", 0)
            .Broadcast("max", 0)
            .Split("y", 0, axis)
            .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("clip_by_value_grad")
    .Input("dy")
    .Input("x")
    .OptionalInput("min")
    .OptionalInput("max")
    .Output("dx")
    .SetShapeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* x_shape = ctx->Shape4ArgNameAndIndex("x", 0);
      Shape* min_shape = ctx->Shape4ArgNameAndIndex("min", 0);
      Shape* max_shape = ctx->Shape4ArgNameAndIndex("max", 0);
      OF_CHECK_EQ(min_shape->NumAxes(), 1);
      OF_CHECK_EQ(max_shape->NumAxes(), 1);
      OF_CHECK_EQ(min_shape->At(0), 1);
      OF_CHECK_EQ(max_shape->At(0), 1);
      OF_CHECK_EQ(*x_shape, *ctx->Shape4ArgNameAndIndex("dy", 0));
      *ctx->Shape4ArgNameAndIndex("dx", 0) = *x_shape;
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      DataType x_data_type = *ctx->Dtype4ArgNameAndIndex("x", 0);
      OF_CHECK_EQ(x_data_type, *ctx->Dtype4ArgNameAndIndex("min", 0));
      OF_CHECK_EQ(x_data_type, *ctx->Dtype4ArgNameAndIndex("max", 0));
      OF_CHECK_EQ(x_data_type, *ctx->Dtype4ArgNameAndIndex("dy", 0));
      *ctx->Dtype4ArgNameAndIndex("dx", 0) = x_data_type;
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("dx", 0) = *ctx->BatchAxis4ArgNameAndIndex("x", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      int64_t num_axes = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0).shape().NumAxes();
      FOR_RANGE(int64_t, axis, 0, num_axes) {
        SbpSignatureBuilder()
            .Broadcast("min", 0)
            .Broadcast("max", 0)
            .Split("dy", 0, axis)
            .Split("x", 0, axis)
            .Split("dx", 0, axis)
            .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("clip_by_value")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op = builder.Op("clip_by_value_grad")
                                                 .Input("min", op.input("min", 0))
                                                 .Input("max", op.input("max", 0))
                                                 .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
                                                 .Input("x", op.input("x", 0))
                                                 .Output("dx")
                                                 .Build();
        op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);
        AddOp(grad_op);
      }
    });

}  // namespace oneflow
