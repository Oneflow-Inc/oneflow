#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

REGISTER_USER_OP("dropout_v2")
    .Input("in")
    .Output("out")
    .Output("mask")
    .Attr("scale", UserOpAttrType::kAtFloat)
    .Attr("rate", UserOpAttrType::kAtFloat)
    .Attr("seed", UserOpAttrType::kAtInt64)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* in_desc = ctx->TensorDesc4ArgNameAndIndex("in", 0);
      user_op::TensorDesc* mask_desc = ctx->TensorDesc4ArgNameAndIndex("mask", 0);
      *ctx->TensorDesc4ArgNameAndIndex("out", 0) = *in_desc;
      *mask_desc = *in_desc;
      *mask_desc->mut_data_type() = DataType::kInt8;
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("out", 0) = *ctx->BatchAxis4ArgNameAndIndex("in", 0);
      *ctx->BatchAxis4ArgNameAndIndex("mask", 0) = *ctx->BatchAxis4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
      FOR_RANGE(int64_t, axis, 0, in_tensor.shape().NumAxes()) {
        ctx->NewBuilder()
            .Split(user_op::OpArg("in", 0), axis)
            .Split(user_op::OpArg("out", 0), axis)
            .Split(user_op::OpArg("mask", 0), axis)
            .Build();
      }
      return Maybe<void>::Ok();
    })
    .SetCheckAttrFn([](const user_op::UserOpDefWrapper& op_def,
                       const user_op::UserOpConfWrapper& op_conf) -> Maybe<void> {
      float scale = op_conf.attr<float>("scale");
      CHECK_GT_OR_RETURN(scale, 1);
      float rate = op_conf.attr<float>("rate");
      CHECK_GE_OR_RETURN(rate, 0);
      CHECK_LT_OR_RETURN(rate, 1);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("dropout_v2")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("in", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper dropout_grad_op =
            builder.Op("dropout_grad")
                .Input("dy", op.GetGradTensorWithOpOutput("out", 0))
                .Input("mask", op.output("mask", 0))
                .Output("dx")
                .Attr("scale", op.attr<float>("scale"))
                .Build();
        op.BindGradTensorWithOpInput(dropout_grad_op.output("dx", 0), "in", 0);
        AddOp(dropout_grad_op);
      }
    });

}  // namespace

}  // namespace oneflow
