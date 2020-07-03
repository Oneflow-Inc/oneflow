#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("bias_add")
    .Input("a")
    .Input("b")
    .Output("out")
    .Attr("axis", UserOpAttrType::kAtInt32)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const auto* a_tensor_desc = ctx->TensorDesc4ArgNameAndIndex("a", 0);
      const auto* b_tensor_desc = ctx->TensorDesc4ArgNameAndIndex("b", 0);
      const auto bias_add_axis = ctx->Attr<int32_t>("axis");
      CHECK_EQ_OR_RETURN(b_tensor_desc->shape().NumAxes(), 1);
      CHECK_GE_OR_RETURN(bias_add_axis, 0);
      CHECK_LT_OR_RETURN(bias_add_axis, a_tensor_desc->shape().NumAxes());
      CHECK_EQ_OR_RETURN(a_tensor_desc->shape().At(bias_add_axis), b_tensor_desc->shape().At(0));
      *ctx->TensorDesc4ArgNameAndIndex("out", 0) = *a_tensor_desc;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const auto axis = ctx->Attr<int32_t>("axis");
      for (int64_t i = 0; i < ctx->LogicalTensorDesc4InputArgNameAndIndex("a", 0).shape().NumAxes();
           ++i) {
        if (i == axis) { continue; }
        ctx->NewBuilder()
            .Split(user_op::OpArg("a", 0), i)
            .Broadcast(user_op::OpArg("b", 0))
            .Split(ctx->outputs(), i)
            .Build();
      }
      ctx->NewBuilder()
          .Split(user_op::OpArg("b", 0), 0)
          .Split(user_op::OpArg("a", 0), axis)
          .Split(ctx->outputs(), axis)
          .Build();
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("bias_add")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("a", 0)) {
        op.BindGradTensorWithOpInput(op.GetGradTensorWithOpOutput("out", 0), "a", 0);
      }
      if (op.NeedGenGradTensor4OpInput("b", 0)) {
        const int64_t num_axes = op.TensorDesc4ArgNameAndIndex("a", 0).shape().NumAxes();
        const int32_t bias_add_axis = op.attr<int32_t>("axis");
        std::vector<int32_t> reduce_axes_vec(num_axes);
        FOR_RANGE(int64_t, i, 0, num_axes) {
          if (i != bias_add_axis) { reduce_axes_vec.push_back(i); }
        }
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        auto grad_op = builder.Op("reduce_sum")
                           .Input("input_tensor", op.GetGradTensorWithOpOutput("out", 0))
                           .Output("output_tensor")
                           .Attr("axis", reduce_axes_vec)
                           .Attr("keepdims", false)
                           .Build();
        AddOp(grad_op);
        op.BindGradTensorWithOpInput(grad_op.output("output_tensor", 0), "b", 0);
      }
    });

}  // namespace oneflow
