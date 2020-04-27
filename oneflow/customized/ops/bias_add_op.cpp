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
      const auto bias_add_axis = ctx->GetAttr<int32_t>("axis");
      CHECK_EQ_OR_RETURN(b_tensor_desc->shape().NumAxes(), 1);
      CHECK_GE_OR_RETURN(bias_add_axis, 0);
      CHECK_LT_OR_RETURN(bias_add_axis, a_tensor_desc->shape().NumAxes());
      CHECK_EQ_OR_RETURN(a_tensor_desc->shape().At(bias_add_axis), b_tensor_desc->shape().At(0));
      *ctx->TensorDesc4ArgNameAndIndex("out", 0) = *a_tensor_desc;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const auto axis = ctx->GetAttr<int32_t>("axis");
      for (int64_t i = 0; i < ctx->LogicalTensorDesc4InputArgNameAndIndex("a", 0).shape().NumAxes();
           ++i) {
        if (i == axis) { continue; }
        SbpSignatureBuilder().Split("a", i, 0).Broadcast("b").Split("out", i, 0).Build(
            ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      }
      SbpSignatureBuilder()
          .Split("b", 0, 0)
          .Split("a", axis, 0)
          .Split("out", axis, 0)
          .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("bias_add")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("a", 0)) {
        op.BindGradTensorWithOpInput(op.GetGradTensorWithOpOutput("out", 0), "a", 0);
      }
      if (op.NeedGenGradTensor4OpInput("b", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        auto grad_op_builder = builder.Op("reduce_sum")
                                   .Input("in", op.GetGradTensorWithOpOutput("out", 0))
                                   .Output("out");

        const int32_t bias_add_axis = op.attr<int32_t>("axis");
        const int32_t num_axes = op.TensorDesc4ArgNameAndIndex("a", 0).shape().NumAxes();
        std::vector<int32_t> reduce_sum_axes;
        FOR_RANGE(int32_t, i, 0, num_axes) {
          if (i != bias_add_axis) { reduce_sum_axes.push_back(i); }
        }
        grad_op_builder.Attr("keep_dims", false);
        const auto grad_op = grad_op_builder.Build();
        AddOp(grad_op);
        op.BindGradTensorWithOpInput(grad_op.output("out", 0), "b", 0);
      }
    });

}  // namespace oneflow
