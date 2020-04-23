#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("layer_norm")
    .Input("x")
    .OptionalInput("beta")
    .OptionalInput("gamma")
    .Output("y")
    .OptionalOutput("normalized")
    .OptionalOutput("mean")
    .OptionalOutput("inv_variance")
    .Attr("alpha", UserOpAttrType::kAtFloat)
    .Attr("center", UserOpAttrType::kAtBool)
    .Attr("scale", UserOpAttrType::kAtBool)
    .Attr("begin_norm_axis", UserOpAttrType::kAtInt64)
    .Attr("begin_params_axis", UserOpAttrType::kAtInt64)
    .Attr("epsilon", UserOpAttrType::kAtFloat)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* x_shape = ctx->Shape4ArgNameAndIndex("x", 0);
      Shape* y_shape = ctx->Shape4ArgNameAndIndex("y", 0);
      *y_shape = *x_shape;
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("y", 0) = *ctx->BatchAxis4ArgNameAndIndex("x", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
      SbpSignatureBuilder()
          .Split("x", 0, 0)
          .Split("y", 0, 0)
          .MakeSplitSignatureListBuilder(tensor.shape().NumAxes())
          .Build(ctx->sbp_sig_list());
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("layer_norm_grad")
    .Input("x")
    .Output("y")
    .Attr("alpha", UserOpAttrType::kAtFloat)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* x_shape = ctx->Shape4ArgNameAndIndex("x", 0);
      Shape* y_shape = ctx->Shape4ArgNameAndIndex("y", 0);
      *y_shape = *x_shape;
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("y", 0) = *ctx->BatchAxis4ArgNameAndIndex("x", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
      SbpSignatureBuilder()
          .Split("x", 0, 0)
          .Split("y", 0, 0)
          .MakeSplitSignatureListBuilder(tensor.shape().NumAxes())
          .Build(ctx->sbp_sig_list());
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("layer_norm")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op = builder.Op("layer_norm_grad")
                                                 .Input("x", op.input("x", 0))
                                                 .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
                                                 .Output("dx")
                                                 .Attr("alpha", op.attr<float>("alpha"))
                                                 .Build();
        op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);
        AddOp(grad_op);
      }
    });

}  // namespace oneflow
