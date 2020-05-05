#include "oneflow/core/framework/framework.h"

namespace oneflow {

Maybe<void> TensorDescInfer(user_op::InferContext *ctx) {
  *ctx->TensorDesc4ArgNameAndIndex("y", 0) = *ctx->TensorDesc4ArgNameAndIndex("x", 0);
  return Maybe<void>::Ok();
}

Maybe<void> BatchAxisInfer(user_op::BatchAxisContext *ctx) {
  *ctx->BatchAxis4ArgNameAndIndex("y", 0) = *ctx->BatchAxis4ArgNameAndIndex("x", 0);
  return Maybe<void>::Ok();
}

Maybe<void> AddPartialSumSbpSignature(user_op::SbpContext *ctx) {
  SbpSignatureBuilder()
      .PartialSum(ctx->inputs())
      .PartialSum(ctx->outputs())
      .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

Maybe<void> NoExtraSbpSignature(user_op::SbpContext *ctx) { return Maybe<void>::Ok(); }

template<Maybe<void> (*GetExtraSbpSignature)(user_op::SbpContext *)>
Maybe<void> GetSbp(user_op::SbpContext *ctx) {
  const user_op::TensorDesc &tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  SbpSignatureBuilder()
      .Split(ctx->inputs(), 0)
      .Split(ctx->outputs(), 0)
      .MakeSplitSignatureListBuilder(tensor.shape().NumAxes())
      .Build(ctx->sbp_sig_list());
  JUST(GetExtraSbpSignature(ctx));
  return Maybe<void>::Ok();
}

#define REGISTER_SCALAR_BINARY_USER_OP(op_name, extra_sbp_sig) \
  REGISTER_USER_OP(op_name)                                    \
      .Input("x")                                              \
      .Output("y")                                             \
      .Attr("has_int_operand", UserOpAttrType::kAtBool)        \
      .Attr("has_float_operand", UserOpAttrType::kAtBool)      \
      .Attr("int_operand", UserOpAttrType::kAtInt64)           \
      .Attr("float_operand", UserOpAttrType::kAtDouble)        \
      .SetTensorDescInferFn(TensorDescInfer)                   \
      .SetBatchAxisInferFn(BatchAxisInfer)                     \
      .SetGetSbpFn(GetSbp<extra_sbp_sig>);

REGISTER_SCALAR_BINARY_USER_OP("scalar_add", NoExtraSbpSignature);
// TODO: add sub op
// REGISTER_SCALAR_BINARY_USER_OP(scalar_sub_left_scalar, NoExtraSbp);
// REGISTER_SCALAR_BINARY_USER_OP(scalar_sub_right_scalar, NoExtraSbp);
REGISTER_SCALAR_BINARY_USER_OP("scalar_mul", AddPartialSumSbpSignature);
REGISTER_SCALAR_BINARY_USER_OP("left_scalar_div", NoExtraSbpSignature);
REGISTER_SCALAR_BINARY_USER_OP("right_scalar_div", AddPartialSumSbpSignature);

REGISTER_USER_OP_GRAD("scalar_add")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper &op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        op.BindGradTensorWithOpInput(op.GetGradTensorWithOpOutput("y", 0), "x", 0);
      }
    });

REGISTER_USER_OP_GRAD("scalar_mul")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper &op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("scalar_mul")
                .Input("x", op.GetGradTensorWithOpOutput("y", 0))
                .Output("y")
                .Attr("has_int_operand", op.attr<bool>("has_int_operand"))
                .Attr("int_operand", op.attr<int64_t>("int_operand"))
                .Attr("has_float_operand", op.attr<bool>("has_float_operand"))
                .Attr("float_operand", op.attr<double>("float_operand"))
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("y", 0), "x", 0);
        AddOp(grad_op);
      }
    });

// TODO: add div grad

}  // namespace oneflow
