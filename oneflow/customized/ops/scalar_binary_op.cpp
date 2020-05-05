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
REGISTER_SCALAR_BINARY_USER_OP("left_scalar_sub", NoExtraSbpSignature);
REGISTER_SCALAR_BINARY_USER_OP("right_scalar_sub", NoExtraSbpSignature);
REGISTER_SCALAR_BINARY_USER_OP("scalar_mul", AddPartialSumSbpSignature);
REGISTER_SCALAR_BINARY_USER_OP("left_scalar_div", NoExtraSbpSignature);
REGISTER_SCALAR_BINARY_USER_OP("right_scalar_div", AddPartialSumSbpSignature);

REGISTER_USER_OP_GRAD("scalar_add")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper &op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        op.BindGradTensorWithOpInput(op.GetGradTensorWithOpOutput("y", 0), "x", 0);
      }
    });

REGISTER_USER_OP_GRAD("left_scalar_sub")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper &op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op = builder.Op("scalar_mul")
                                                 .Input("x", op.GetGradTensorWithOpOutput("y", 0))
                                                 .Output("y")
                                                 .Attr("has_int_operand", true)
                                                 .Attr<int64_t>("int_operand", -1)
                                                 .Attr("has_float_operand", false)
                                                 .Attr<double>("float_operand", 0.)
                                                 .Build();
        op.BindGradTensorWithOpInput(grad_op.output("y", 0), "x", 0);
        AddOp(grad_op);
      }
    });

REGISTER_USER_OP_GRAD("right_scalar_sub")
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

// TODO: there is no multiply user op so the grad of left_scalar_div doesn't work
REGISTER_USER_OP_GRAD("left_scalar_div")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper &op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        user_op::UserOpConfWrapperBuilder pow_builder(op.op_name() + "_grad_sqaure");
        user_op::UserOpConfWrapper square_op = pow_builder.Op("multiply")
                                                   .Input("in_0", op.input("x", 0))
                                                   .Input("in_1", op.input("x", 0))
                                                   .Output("out")
                                                   .Build();
        AddOp(square_op);
        user_op::UserOpConfWrapperBuilder div_builder(op.op_name() + "_grad_div");
        user_op::UserOpConfWrapper div_op = div_builder.Op("left_scalar_div")
                                                .Input("x", square_op.output("out", 0))
                                                .Attr("has_int_operand", true)
                                                .Attr<int64_t>("int_operand", -1)
                                                .Attr("has_float_operand", false)
                                                .Attr<double>("float_operand", 0.)
                                                .Output("y")
                                                .Build();
        AddOp(div_op);
        user_op::UserOpConfWrapperBuilder mul_builder(op.op_name() + "_grad_mul");
        user_op::UserOpConfWrapper mul_op = mul_builder.Op("multiply")
                                                .Input("in_0", div_op.output("y", 0))
                                                .Input("in_1", op.GetGradTensorWithOpOutput("y", 0))
                                                .Output("out")
                                                .Build();
        AddOp(mul_op);
        op.BindGradTensorWithOpInput(mul_op.output("out", 0), "x", 0);
      }
    });

REGISTER_USER_OP_GRAD("right_scalar_div")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper &op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("right_scalar_div")
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

}  // namespace oneflow
