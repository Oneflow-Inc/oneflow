#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("binary")
    .Input("x")
    .Input("y")
    .Output("z")
    .Attr("binary_math_type", UserOpAttrType::kAtString)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* x_shape = ctx->Shape4ArgNameAndIndex("x", 0);
      Shape* y_shape = ctx->Shape4ArgNameAndIndex("y", 0);
      Shape* z_shape = ctx->Shape4ArgNameAndIndex("z", 0);
      CHECK(*y_shape == *x_shape);
      *z_shape = *x_shape;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& tensor_x = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
      // const user_op::TensorDesc& tensor_y = ctx->LogicalTensorDesc4InputArgNameAndIndex("y", 0);
      SbpSignatureBuilder()
          .Split(ctx->inputs(), 0)
          .Split(ctx->outputs(), 0)
          .MakeSplitSignatureListBuilder(tensor_x.shape().NumAxes())
          .Build(ctx->sbp_sig_list());
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("binary_x_grad")
    .Input("x")
    .Input("y")
    .Input("dz")
    .Output("dx")
    .Attr("binary_math_type", UserOpAttrType::kAtString)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* x_shape = ctx->Shape4ArgNameAndIndex("x", 0);
      Shape* y_shape = ctx->Shape4ArgNameAndIndex("y", 0);
      Shape* dz_shape = ctx->Shape4ArgNameAndIndex("dz", 0);
      Shape* dx_shape = ctx->Shape4ArgNameAndIndex("dx", 0);
      CHECK((*y_shape == *x_shape) && (*dz_shape == *x_shape));
      *dx_shape = *x_shape;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("dz", 0);
      SbpSignatureBuilder()
          .Split(ctx->inputs(), 0)
          .Split(ctx->outputs(), 0)
          .MakeSplitSignatureListBuilder(tensor.shape().NumAxes())
          .Build(ctx->sbp_sig_list());
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("binary_y_grad")
    .Input("x")
    .Input("y")
    .Input("dz")
    .Output("dy")
    .Attr("binary_math_type", UserOpAttrType::kAtString)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* x_shape = ctx->Shape4ArgNameAndIndex("x", 0);
      Shape* y_shape = ctx->Shape4ArgNameAndIndex("y", 0);
      Shape* dz_shape = ctx->Shape4ArgNameAndIndex("dz", 0);
      Shape* dy_shape = ctx->Shape4ArgNameAndIndex("dy", 0);
      CHECK((*y_shape == *x_shape) && (*dz_shape == *x_shape));
      *dy_shape = *y_shape;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("dz", 0);
      SbpSignatureBuilder()
          .Split(ctx->inputs(), 0)
          .Split(ctx->outputs(), 0)
          .MakeSplitSignatureListBuilder(tensor.shape().NumAxes())
          .Build(ctx->sbp_sig_list());
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("binary").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                                                          user_op::AddOpFn AddOp) {
  if (op.NeedGenGradTensor4OpInput("x", 0)) {
    user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_x_grad");
    user_op::UserOpConfWrapper binary_grad_op =
        builder.Op("binary_x_grad")
            .Input("x", op.input("x", 0))
            .Input("y", op.input("y", 0))
            .Input("dz", op.GetGradTensorWithOpOutput("z", 0))
            .Output("dx")
            .Attr<std::string>("binary_math_type", op.attr<std::string>("binary_math_type"))
            .Build();
    op.BindGradTensorWithOpInput(binary_grad_op.output("dx", 0), "x", 0);
    AddOp(binary_grad_op);
  }
  if (op.NeedGenGradTensor4OpInput("y", 0)) {
    user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_y_grad");
    user_op::UserOpConfWrapper binary_grad_op =
        builder.Op("binary_y_grad")
            .Input("x", op.input("x", 0))
            .Input("y", op.input("y", 0))
            .Input("dz", op.GetGradTensorWithOpOutput("z", 0))
            .Output("dy")
            .Attr<std::string>("binary_math_type", op.attr<std::string>("binary_math_type"))
            .Build();
    op.BindGradTensorWithOpInput(binary_grad_op.output("dy", 0), "y", 0);
    AddOp(binary_grad_op);
  }
});
}  // namespace oneflow
