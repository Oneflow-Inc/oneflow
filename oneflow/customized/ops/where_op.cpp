#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

Maybe<Shape> GetWhereBroadcastedShape(const ShapeView& cond_shape, const ShapeView& x_shape,
                                      const ShapeView& y_shape) {
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

Maybe<Shape> GetWhereBroadcastedShape(const Shape& cond_shape, const Shape& x_shape,
                                      const Shape& y_shape) {
  return GetWhereBroadcastedShape(ShapeView(cond_shape), ShapeView(x_shape), ShapeView(y_shape));
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

Maybe<void> InferWhereBatchAxis(user_op::BatchAxisContext* ctx) {
  // TODO
  return Maybe<void>::Ok();
}

Maybe<void> GetWhereSbpSignatures(user_op::SbpContext* ctx) {
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

}  // namespace oneflow
