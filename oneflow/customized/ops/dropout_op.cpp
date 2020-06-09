#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

REGISTER_USER_OP("dropout")
    .Input("in")
    .Input("mask")
    .Output("out")
    .Attr("scale", UserOpAttrType::kAtFloat)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* in_shape = ctx->Shape4ArgNameAndIndex("in", 0);
      *ctx->TensorDesc4ArgNameAndIndex("out", 0) = *ctx->TensorDesc4ArgNameAndIndex("in", 0);
      CHECK_EQ_OR_RETURN(*ctx->Shape4ArgNameAndIndex("mask", 0), *in_shape);
      CHECK_EQ_OR_RETURN(*ctx->Dtype4ArgNameAndIndex("mask", 0), DataType::kInt8);
      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper&) {
      user_op::InputArgModifier* mask = GetInputArgModifierFn("mask", 0);
      mask->set_requires_grad(false);
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("out", 0) = *ctx->BatchAxis4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
      FOR_RANGE(int64_t, axis, 0, in_tensor.shape().NumAxes()) {
        ctx->NewBuilder()
            .Split(user_op::OpArg("in", 0), axis)
            .Split(user_op::OpArg("mask", 0), axis)
            .Split(user_op::OpArg("out", 0), axis)
            .Build();
      }
      return Maybe<void>::Ok();
    })
    .SetCheckAttrFn([](const user_op::UserOpDefWrapper& op_def,
                       const user_op::UserOpConfWrapper& op_conf) -> Maybe<void> {
      float scale = op_conf.attr<float>("scale");
      CHECK_GT_OR_RETURN(scale, 1);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("dropout_grad")
    .Input("dy")
    .Input("mask")
    .Output("dx")
    .Attr("scale", UserOpAttrType::kAtFloat)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* dy_shape = ctx->Shape4ArgNameAndIndex("dy", 0);
      *ctx->TensorDesc4ArgNameAndIndex("dx", 0) = *ctx->TensorDesc4ArgNameAndIndex("dy", 0);
      CHECK_EQ_OR_RETURN(*ctx->Shape4ArgNameAndIndex("mask", 0), *dy_shape);
      CHECK_EQ_OR_RETURN(*ctx->Dtype4ArgNameAndIndex("mask", 0), DataType::kInt8);
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("dx", 0) = *ctx->BatchAxis4ArgNameAndIndex("dy", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& dy_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("dy", 0);
      FOR_RANGE(int64_t, axis, 0, dy_tensor.shape().NumAxes()) {
        ctx->NewBuilder()
            .Split(user_op::OpArg("dy", 0), axis)
            .Split(user_op::OpArg("mask", 0), axis)
            .Split(user_op::OpArg("dx", 0), axis)
            .Build();
      }
      return Maybe<void>::Ok();
    })
    .SetCheckAttrFn([](const user_op::UserOpDefWrapper& op_def,
                       const user_op::UserOpConfWrapper& op_conf) -> Maybe<void> {
      float scale = op_conf.attr<float>("scale");
      CHECK_GT_OR_RETURN(scale, 1);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("dropout").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                                                           user_op::AddOpFn AddOp) {
  if (op.NeedGenGradTensor4OpInput("in", 0)) {
    user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
    user_op::UserOpConfWrapper dropout_grad_op =
        builder.Op("dropout_grad")
            .Input("dy", op.GetGradTensorWithOpOutput("out", 0))
            .Input("mask", op.input("mask", 0))
            .Output("dx")
            .Attr("scale", op.attr<float>("scale"))
            .Build();
    op.BindGradTensorWithOpInput(dropout_grad_op.output("dx", 0), "in", 0);
    AddOp(dropout_grad_op);
  }
});

REGISTER_USER_OP("random_mask_like")
    .Input("like")
    .Output("out")
    .Attr("rate", UserOpAttrType::kAtFloat)
    .Attr("seed", UserOpAttrType::kAtInt64)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->Shape4ArgNameAndIndex("out", 0) = *ctx->Shape4ArgNameAndIndex("like", 0);
      *ctx->Dtype4ArgNameAndIndex("out", 0) = DataType::kInt8;
      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper&) {
      user_op::InputArgModifier* like_arg_modifier = GetInputArgModifierFn("like", 0);
      CHECK(like_arg_modifier != nullptr);
      like_arg_modifier->set_use_header_only(true);
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("out", 0) = *ctx->BatchAxis4ArgNameAndIndex("like", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& like_tensor =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("like", 0);
      FOR_RANGE(int64_t, axis, 0, like_tensor.shape().NumAxes()) {
        ctx->NewBuilder()
            .Split(user_op::OpArg("like", 0), axis)
            .Split(user_op::OpArg("out", 0), axis)
            .Build();
      }
      return Maybe<void>::Ok();
    })
    .SetCheckAttrFn([](const user_op::UserOpDefWrapper& op_def,
                       const user_op::UserOpConfWrapper& op_conf) -> Maybe<void> {
      float rate = op_conf.attr<float>("rate");
      CHECK_GE_OR_RETURN(rate, 0);
      CHECK_LT_OR_RETURN(rate, 1);
      return Maybe<void>::Ok();
    });

}  // namespace

}  // namespace oneflow
