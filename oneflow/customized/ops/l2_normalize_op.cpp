#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("l2_normalize")
    .Input("x")
    .Output("y")
    .Output("square_x_sum")
    .Attr("axis", UserOpAttrType::kAtInt32)
    .Attr("epsilon", UserOpAttrType::kAtFloat)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* x_shape = ctx->Shape4ArgNameAndIndex("x", 0);
      Shape* y_shape = ctx->Shape4ArgNameAndIndex("y", 0);
      Shape* square_x_sum_shape = ctx->Shape4ArgNameAndIndex("square_x_sum", 0);
      const int32_t axis = ctx->GetAttr<int32_t>("axis");
      const float epsilon = ctx->GetAttr<float>("epsilon");
      CHECK_GE_OR_RETURN(axis, 0);
      CHECK_LT_OR_RETURN(axis, x_shape->NumAxes());
      CHECK_GT_OR_RETURN(epsilon, 0);
      *y_shape = *x_shape;
      *square_x_sum_shape = *x_shape;
      square_x_sum_shape->Set(axis, 1);
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("y", 0) = *ctx->BatchAxis4ArgNameAndIndex("x", 0);
      if (ctx->BatchAxis4ArgNameAndIndex("x", 0)->value() != ctx->GetAttr<int32_t>("axis")) {
        *ctx->BatchAxis4ArgNameAndIndex("square_x_sum", 0) =
            *ctx->BatchAxis4ArgNameAndIndex("x", 0);
      } else {
        ctx->BatchAxis4ArgNameAndIndex("square_x_sum", 0)->clear_value();
      }
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const int32_t num_axes =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0).shape().NumAxes();
      const int32_t axis = ctx->GetAttr<int32_t>("axis");
      for (int64_t i = 0; i < num_axes; ++i) {
        if (i != axis) {
          SbpSignatureBuilder()
              .Split("x", 0, i)
              .Split("y", 0, i)
              .Split("square_x_sum", 0, i)
              .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
        }
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("l2_normalize_grad")
    .Input("dy")
    .Input("y")
    .Input("square_x_sum")
    .Output("dx")
    .Attr("axis", UserOpAttrType::kAtInt32)
    .Attr("epsilon", UserOpAttrType::kAtFloat)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* dy_shape = ctx->Shape4ArgNameAndIndex("dy", 0);
      const Shape* y_shape = ctx->Shape4ArgNameAndIndex("y", 0);
      const Shape* square_x_sum_shape = ctx->Shape4ArgNameAndIndex("square_x_sum", 0);
      Shape* dx_shape = ctx->Shape4ArgNameAndIndex("dx", 0);
      const int32_t axis = ctx->GetAttr<int32_t>("axis");
      const float epsilon = ctx->GetAttr<float>("epsilon");
      CHECK_EQ_OR_RETURN(*dy_shape, *y_shape);
      CHECK_GE_OR_RETURN(axis, 0);
      CHECK_LT_OR_RETURN(axis, dy_shape->NumAxes());
      CHECK_GT_OR_RETURN(epsilon, 0);
      FOR_RANGE(int32_t, i, 0, dy_shape->NumAxes()) {
        if (i == axis) {
          CHECK_EQ_OR_RETURN(square_x_sum_shape->At(i), 1);
        } else {
          CHECK_EQ_OR_RETURN(square_x_sum_shape->At(i), dy_shape->At(i));
        }
      }
      *dx_shape = *dy_shape;
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("dx", 0) = *ctx->BatchAxis4ArgNameAndIndex("dy", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const int32_t num_axes =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("y", 0).shape().NumAxes();
      const int32_t axis = ctx->GetAttr<int32_t>("axis");
      for (int64_t i = 0; i < num_axes; ++i) {
        if (i != axis) {
          SbpSignatureBuilder()
              .Split("y", 0, i)
              .Split("dy", 0, i)
              .Split("square_x_sum", 0, i)
              .Split("dx", 0, i)
              .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
        }
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("l2_normalize")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("l2_normalize_grad")
                .Input("y", op.output("y", 0))
                .Input("square_x_sum", op.output("square_x_sum", 0))
                .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
                .Output("dx")
                .Attr("axis", op.attr<int32_t>("axis"))
                .Attr("epsilon", op.attr<float>("epsilon"))
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);
        AddOp(grad_op);
      }
    });

}  // namespace oneflow
