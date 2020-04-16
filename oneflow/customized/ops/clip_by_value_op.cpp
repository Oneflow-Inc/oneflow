#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

Maybe<void> InferClipTensorDesc(user_op::InferContext* ctx) {
  *ctx->Shape4ArgNameAndIndex("y", 0) = *ctx->Shape4ArgNameAndIndex("x", 0);
  *ctx->Dtype4ArgNameAndIndex("y", 0) = *ctx->Dtype4ArgNameAndIndex("x", 0);
  return Maybe<void>::Ok();
}

Maybe<void> InferClipBatchAxis(user_op::BatchAxisContext* ctx) {
  *ctx->BatchAxis4ArgNameAndIndex("y", 0) = *ctx->BatchAxis4ArgNameAndIndex("x", 0);
  return Maybe<void>::Ok();
}

Maybe<void> GetClipSbpSignature(user_op::SbpContext* ctx) {
  SbpSignatureBuilder()
      .Split("x", 0, 0)
      .Split("y", 0, 0)
      .MakeSplitSignatureListBuilder(
          ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0).shape().NumAxes())
      .Build(ctx->sbp_sig_list());
  return Maybe<void>::Ok();
}

Maybe<void> InferClipGradTensorDesc(user_op::InferContext* ctx) {
  *ctx->Shape4ArgNameAndIndex("dx", 0) = *ctx->Shape4ArgNameAndIndex("x", 0);
  *ctx->Dtype4ArgNameAndIndex("dx", 0) = *ctx->Dtype4ArgNameAndIndex("x", 0);
  return Maybe<void>::Ok();
}

Maybe<void> InferClipGradBatchAxis(user_op::BatchAxisContext* ctx) {
  *ctx->BatchAxis4ArgNameAndIndex("dx", 0) = *ctx->BatchAxis4ArgNameAndIndex("x", 0);
  return Maybe<void>::Ok();
}

Maybe<void> GetClipGradSbpSignature(user_op::SbpContext* ctx) {
  SbpSignatureBuilder()
      .Split("dy", 0, 0)
      .Split("x", 0, 0)
      .Split("dx", 0, 0)
      .MakeSplitSignatureListBuilder(
          ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0).shape().NumAxes())
      .Build(ctx->sbp_sig_list());
  SbpSignatureBuilder().Broadcast("x", 0).PartialSum("dy", 0).PartialSum("dx", 0).Build(
      ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_USER_OP("clip_by_scalar")
    .Input("x")
    .Attr("floating_min", UserOpAttrType::kAtDouble)
    .Attr("integral_min", UserOpAttrType::kAtInt64)
    .Attr("floating_max", UserOpAttrType::kAtDouble)
    .Attr("integral_max", UserOpAttrType::kAtInt64)
    .Output("y")
    .SetTensorDescInferFn(InferClipTensorDesc)
    .SetBatchAxisInferFn(InferClipBatchAxis)
    .SetGetSbpFn(GetClipSbpSignature);

REGISTER_USER_OP("clip_by_scalar_min")
    .Input("x")
    .Attr("floating_min", UserOpAttrType::kAtDouble)
    .Attr("integral_min", UserOpAttrType::kAtInt64)
    .Output("y")
    .SetTensorDescInferFn(InferClipTensorDesc)
    .SetBatchAxisInferFn(InferClipBatchAxis)
    .SetGetSbpFn(GetClipSbpSignature);

REGISTER_USER_OP("clip_by_scalar_max")
    .Input("x")
    .Attr("floating_max", UserOpAttrType::kAtDouble)
    .Attr("integral_max", UserOpAttrType::kAtInt64)
    .Output("y")
    .SetTensorDescInferFn(InferClipTensorDesc)
    .SetBatchAxisInferFn(InferClipBatchAxis)
    .SetGetSbpFn(GetClipSbpSignature);

REGISTER_USER_OP("clip_by_scalar_grad")
    .Input("dy")
    .Input("x")
    .Attr("floating_min", UserOpAttrType::kAtDouble)
    .Attr("integral_min", UserOpAttrType::kAtInt64)
    .Attr("floating_max", UserOpAttrType::kAtDouble)
    .Attr("integral_max", UserOpAttrType::kAtInt64)
    .Output("dx")
    .SetTensorDescInferFn(InferClipGradTensorDesc)
    .SetBatchAxisInferFn(InferClipGradBatchAxis)
    .SetGetSbpFn(GetClipGradSbpSignature);

REGISTER_USER_OP("clip_by_scalar_min_grad")
    .Input("dy")
    .Input("x")
    .Attr("floating_min", UserOpAttrType::kAtDouble)
    .Attr("integral_min", UserOpAttrType::kAtInt64)
    .Output("dx")
    .SetTensorDescInferFn(InferClipGradTensorDesc)
    .SetBatchAxisInferFn(InferClipGradBatchAxis)
    .SetGetSbpFn(GetClipGradSbpSignature);

REGISTER_USER_OP("clip_by_scalar_max_grad")
    .Input("dy")
    .Input("x")
    .Attr("floating_max", UserOpAttrType::kAtDouble)
    .Attr("integral_max", UserOpAttrType::kAtInt64)
    .Output("dx")
    .SetTensorDescInferFn(InferClipGradTensorDesc)
    .SetBatchAxisInferFn(InferClipGradBatchAxis)
    .SetGetSbpFn(GetClipGradSbpSignature);

REGISTER_USER_OP_GRAD("clip_by_scalar")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("clip_by_scalar_grad")
                .Attr("floating_min", op.attr<double>("floating_min"))
                .Attr("integral_min", op.attr<int64_t>("integral_min"))
                .Attr("floating_max", op.attr<double>("floating_max"))
                .Attr("integral_max", op.attr<int64_t>("integral_max"))
                .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
                .Input("x", op.input("x", 0))
                .Output("dx")
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);
        AddOp(grad_op);
      }
    });

REGISTER_USER_OP_GRAD("clip_by_scalar_min")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("clip_by_scalar_min_grad")
                .Attr("floating_min", op.attr<double>("floating_min"))
                .Attr("integral_min", op.attr<int64_t>("integral_min"))
                .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
                .Input("x", op.input("x", 0))
                .Output("dx")
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);
        AddOp(grad_op);
      }
    });

REGISTER_USER_OP_GRAD("clip_by_scalar_max")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("clip_by_scalar_max_grad")
                .Attr("floating_max", op.attr<double>("floating_max"))
                .Attr("integral_max", op.attr<int64_t>("integral_max"))
                .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
                .Input("x", op.input("x", 0))
                .Output("dx")
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);
        AddOp(grad_op);
      }
    });

}  // namespace oneflow
