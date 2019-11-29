#include "oneflow/core/framework/op_registration.h"
#include "oneflow/core/framework/grad_registration.h"
#include "oneflow/core/common/shape.h"

namespace oneflow {

REGISTER_USER_OP("ccrelu").Input("in").Output("out").SetShapeInferFn(
    [](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* in_shape = ctx->Shape4ArgNameAndIndex("in", 0);
      Shape* out_shape = ctx->Shape4ArgNameAndIndex("out", 0);
      *out_shape = *in_shape;
      // int32_t last_axis = in_shape->NumAxes() - 1;
      // out_shape->Set(last_axis, in_shape->At(last_axis) * 2);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("ccrelu_grad")
    .Input("y")
    .Input("dy")
    .Output("dx")
    .SetShapeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* y_shape = ctx->Shape4ArgNameAndIndex("y", 0);
      Shape* dy_shape = ctx->Shape4ArgNameAndIndex("dy", 0);
      Shape* dx_shape = ctx->Shape4ArgNameAndIndex("dx", 0);
      CHECK(*dy_shape == *y_shape);
      *dx_shape = *y_shape;
      // int32_t last_axis = y_shape->NumAxes() - 1;
      // dx_shape->Set(last_axis, y_shape->At(last_axis) / 2);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("ccrelu").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                                                          user_op::AddOpFn AddOp) {
  if (op.NeedGenGradBlob4OpInput("in", 0)) {
    user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
    user_op::UserOpConfWrapper ccrelu_grad_op =
        builder.Op("ccrelu_grad")
            .Input("y", op.output("out", 0))
            .Input("dy", op.GetGradBlobWithOpOutput("out", 0))
            .Output("dx")
            .Build();
    op.BindGradBlobWithOpInput(ccrelu_grad_op.output("dx", 0), "in", 0);
    AddOp(ccrelu_grad_op);
  }
});

}  // namespace oneflow
