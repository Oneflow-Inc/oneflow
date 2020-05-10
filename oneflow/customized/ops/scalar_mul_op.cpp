#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("scalar_mul")
    .Input("in")
    .Output("out")
    .Attr("has_int_operand", UserOpAttrType::kAtBool)
    .Attr("has_float_operand", UserOpAttrType::kAtBool)
    .Attr("int_operand", UserOpAttrType::kAtInt64)
    .Attr("float_operand", UserOpAttrType::kAtDouble)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->TensorDesc4ArgNameAndIndex("out", 0) = *ctx->TensorDesc4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
      FOR_RANGE(int64_t, i, 0, in_tensor.shape().NumAxes()) {
        ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
      }
      ctx->NewBuilder().PartialSum(ctx->inputs()).PartialSum(ctx->outputs()).Build();
      return Maybe<void>::Ok();
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

}  // namespace oneflow
