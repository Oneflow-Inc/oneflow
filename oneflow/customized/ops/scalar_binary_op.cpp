#include "oneflow/core/framework/framework.h"

namespace oneflow {

Maybe<void> TensorDescInfer(user_op::InferContext *ctx) {
  *ctx->TensorDesc4ArgNameAndIndex("out", 0) = *ctx->TensorDesc4ArgNameAndIndex("in", 0);
  return Maybe<void>::Ok();
}

Maybe<void> BatchAxisInfer(user_op::BatchAxisContext *ctx) {
  *ctx->BatchAxis4ArgNameAndIndex("out", 0) = *ctx->BatchAxis4ArgNameAndIndex("in", 0);
  return Maybe<void>::Ok();
}

Maybe<void> GetSbp(user_op::SbpContext *ctx, const user_op::GetSbpFn &extra_get_sbp_fn) {
  const user_op::TensorDesc &tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
  SbpSignatureBuilder()
      .Split(ctx->inputs(), 0)
      .Split(ctx->outputs(), 0)
      .MakeSplitSignatureListBuilder(tensor.shape().NumAxes())
      .Build(ctx->sbp_sig_list());
  JUST(extra_get_sbp_fn(ctx));
  return Maybe<void>::Ok();
}

Maybe<void> NoExtraSbp(user_op::SbpContext *ctx) { return Maybe<void>::Ok(); }

Maybe<void> AddPs2PsSbp(user_op::SbpContext *ctx) {
  SbpSignatureBuilder()
      .PartialSum(ctx->inputs())
      .PartialSum(ctx->outputs())
      .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

// TODO: use JUST(GetSbp(ctx, EXTRA_GET_SBP_FN));
#define REGISTER_SCALAR_BINARY_USER_OP(OP_NAME, EXTRA_GET_SBP_FN) \
  REGISTER_USER_OP(OF_PP_STRINGIZE(OP_NAME))                                       \
      .Input("in")                                                \
      .Output("out")                                              \
      .Attr("has_int_operand", UserOpAttrType::kAtBool)           \
      .Attr("has_float_operand", UserOpAttrType::kAtBool)         \
      .Attr("int_operand", UserOpAttrType::kAtInt64)              \
      .Attr("float_operand", UserOpAttrType::kAtDouble)           \
      .SetTensorDescInferFn(TensorDescInfer)                      \
      .SetBatchAxisInferFn(BatchAxisInfer)                        \
      .SetGetSbpFn([](user_op::SbpContext *ctx) -> Maybe<void> {  \
        GetSbp(ctx, EXTRA_GET_SBP_FN);                            \
        return Maybe<void>::Ok();                                 \
      });

REGISTER_SCALAR_BINARY_USER_OP(scalar_add, NoExtraSbp);
// TODO: add sub op
// REGISTER_SCALAR_BINARY_USER_OP(scalar_sub_left_scalar, NoExtraSbp);
// REGISTER_SCALAR_BINARY_USER_OP(scalar_sub_right_scalar, NoExtraSbp);
REGISTER_SCALAR_BINARY_USER_OP(scalar_mul, AddPs2PsSbp);
REGISTER_SCALAR_BINARY_USER_OP(scalar_div_left_scalar, NoExtraSbp);
REGISTER_SCALAR_BINARY_USER_OP(scalar_div_right_scalar, AddPs2PsSbp);

REGISTER_USER_OP_GRAD("scalar_add")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper &op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("in", 0)) {
        op.BindGradTensorWithOpInput(op.GetGradTensorWithOpOutput("out", 0), "in", 0);
      }
    });

REGISTER_USER_OP_GRAD("scalar_mul")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("in", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("scalar_mul")
                .Input("in", op.GetGradTensorWithOpOutput("out", 0))
                .Output("out")
                .Attr("has_int_operand", op.attr<bool>("has_int_operand"))
                .Attr("int_operand", op.attr<int64_t>("int_operand"))
                .Attr("has_float_operand", op.attr<bool>("has_float_operand"))
                .Attr("float_operand", op.attr<double>("float_operand"))
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("out", 0), "in", 0);
        AddOp(grad_op);
      }
    });

// TODO: add div grad

}  // namespace oneflow
