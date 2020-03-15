#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

Maybe<Shape> GetWhereBroadcastedShape(const Shape& cond_shape, const Shape& x_shape,
                                      const Shape& y_shape) {
  int64_t max_num_axes = std::max(x_shape.NumAxes(), y_shape.NumAxes());
  max_num_axes = std::max(max_num_axes, cond_shape.NumAxes());
  Shape cond_extend_shape = CreateLeftExtendedShape(cond_shape, max_num_axes);
  Shape x_extend_shape = CreateLeftExtendedShape(x_shape, max_num_axes);
  Shape y_extend_shape = CreateLeftExtendedShape(y_shape, max_num_axes);
  Shape broadcasted_shape(DimVector(max_num_axes, 1));

  auto CheckDoBroadcast = [](const Shape& x, Shape* y, int64_t axis) -> Maybe<void> {
    if (x.At(axis) != 1) {
      if (y->At(axis) == 1) {
        y->Set(axis, x.At(axis));
      } else {
        OF_CHECK_EQ(x.At(axis), y->At(axis));
      }
    }
    return Maybe<void>::Ok();
  };
  for (size_t i = 0; i < broadcasted_shape.NumAxes(); ++i) {
    CheckDoBroadcast(cond_extend_shape, &broadcasted_shape, i);
    CheckDoBroadcast(x_extend_shape, &broadcasted_shape, i);
    CheckDoBroadcast(y_extend_shape, &broadcasted_shape, i);
  }
  return broadcasted_shape;
}

Maybe<void> CheckShapeBroadcastable(const Shape& shape, const Shape& broadcasted_shape) {
  Shape extend_shape = CreateLeftExtendedShape(shape, broadcasted_shape.NumAxes());
  FOR_RANGE(int64_t, i, 0, broadcasted_shape.NumAxes()) {
    OF_CHECK_LE(extend_shape.At(i), broadcasted_shape.At(i));
    if (extend_shape.At(i) != 1) { OF_CHECK_EQ(extend_shape.At(i), broadcasted_shape.At(i)); }
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferWhereTensorDesc(user_op::InferContext* ctx) {
  // shape infer
  const Shape* cond_shape = ctx->Shape4ArgNameAndIndex("condition", 0);
  const Shape* x_shape = ctx->Shape4ArgNameAndIndex("x", 0);
  const Shape* y_shape = ctx->Shape4ArgNameAndIndex("y", 0);
  *ctx->Shape4ArgNameAndIndex("out", 0) =
      *JUST(GetWhereBroadcastedShape(*cond_shape, *x_shape, *y_shape));
  // data_type infer
  DataType cond_dtype = *ctx->Dtype4ArgNameAndIndex("condition", 0);
  OF_CHECK(IsIntegralDataType(cond_dtype));
  DataType x_dtype = *ctx->Dtype4ArgNameAndIndex("x", 0);
  OF_CHECK_EQ(x_dtype, *ctx->Dtype4ArgNameAndIndex("y", 0));
  *ctx->Dtype4ArgNameAndIndex("out", 0) = x_dtype;
  return Maybe<void>::Ok();
}

Maybe<void> InferWhereGradTensorDesc(user_op::InferContext* ctx) {
  // shape infer
  const Shape* cond_shape = ctx->Shape4ArgNameAndIndex("condition", 0);
  const Shape* x_shape = ctx->Shape4ArgNameAndIndex("x", 0);
  const Shape* y_shape = ctx->Shape4ArgNameAndIndex("y", 0);
  const Shape* dz_shape = ctx->Shape4ArgNameAndIndex("dz", 0);
  CheckShapeBroadcastable(*cond_shape, *dz_shape);
  CheckShapeBroadcastable(*x_shape, *dz_shape);
  CheckShapeBroadcastable(*y_shape, *dz_shape);
  *ctx->Shape4ArgNameAndIndex("dx", 0) = *x_shape;
  *ctx->Shape4ArgNameAndIndex("dy", 0) = *y_shape;
  // data_type infer
  DataType cond_dtype = *ctx->Dtype4ArgNameAndIndex("condition", 0);
  OF_CHECK(IsIntegralDataType(cond_dtype));
  DataType x_dtype = *ctx->Dtype4ArgNameAndIndex("x", 0);
  DataType y_dtype = *ctx->Dtype4ArgNameAndIndex("y", 0);
  DataType dz_dtype = *ctx->Dtype4ArgNameAndIndex("dz", 0);
  OF_CHECK_EQ(x_dtype, dz_dtype);
  OF_CHECK_EQ(y_dtype, dz_dtype);
  *ctx->Dtype4ArgNameAndIndex("dx", 0) = x_dtype;
  *ctx->Dtype4ArgNameAndIndex("dy", 0) = y_dtype;
  return Maybe<void>::Ok();
}

Maybe<void> InferWhereBatchAxis(user_op::BatchAxisContext* ctx) {
  // TODO
  return Maybe<void>::Ok();
}

Maybe<void> GetWhereSbpSignatures(user_op::SbpContext* ctx) {
  // TODO
  return Maybe<void>::Ok();
}

Maybe<void> InferWhereGradBatchAxis(user_op::BatchAxisContext* ctx) {
  // TODO
  return Maybe<void>::Ok();
}

Maybe<void> GetWhereGradSbpSignatures(user_op::SbpContext* ctx) {
  // TODO
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_USER_OP("where")
    .Input("condition")
    .Input("x")
    .Input("y")
    .Output("out")
    .SetTensorDescInferFn(InferWhereTensorDesc)
    .SetBatchAxisInferFn(InferWhereBatchAxis)
    .SetGetSbpFn(GetWhereSbpSignatures);

REGISTER_USER_OP("where_grad")
    .Input("condition")
    .Input("dz")
    .Input("x")
    .Input("y")
    .Output("dx")
    .Output("dy")
    .SetTensorDescInferFn(InferWhereGradTensorDesc)
    .SetBatchAxisInferFn(InferWhereGradBatchAxis)
    .SetGetSbpFn(GetWhereGradSbpSignatures)
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn) {
      user_op::InputArgModifier* x_arg_modifier = GetInputArgModifierFn("x", 0);
      user_op::InputArgModifier* y_arg_modifier = GetInputArgModifierFn("y", 0);
      CHECK(x_arg_modifier != nullptr);
      CHECK(y_arg_modifier != nullptr);
      x_arg_modifier->set_use_header_only(true);
      y_arg_modifier->set_use_header_only(true);
    });

REGISTER_USER_OP_GRAD("where").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                                                         user_op::AddOpFn AddOp) {
  bool x_need_grad = op.NeedGenGradTensor4OpInput("x", 0);
  bool y_need_grad = op.NeedGenGradTensor4OpInput("y", 0);
  if (x_need_grad || y_need_grad) {
    user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
    user_op::UserOpConfWrapper grad_op = builder.Op("where_grad")
                                             .Input("condition", op.input("condition", 0))
                                             .Input("x", op.input("x", 0))
                                             .Input("y", op.input("y", 0))
                                             .Input("dz", op.GetGradTensorWithOpOutput("out", 0))
                                             .Output("dx")
                                             .Output("dy")
                                             .Build();
    if (x_need_grad) { op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0); }
    if (y_need_grad) { op.BindGradTensorWithOpInput(grad_op.output("dy", 0), "y", 0); }
    AddOp(grad_op);
  }
});

}  // namespace oneflow
