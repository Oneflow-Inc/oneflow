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
  OptInt64* x_batch_axis = ctx->BatchAxis4ArgNameAndIndex("x", 0);
  OptInt64* y_batch_axis = ctx->BatchAxis4ArgNameAndIndex("y", 0);
  OF_CHECK((x_batch_axis->has_value() && y_batch_axis->has_value())
           || (!x_batch_axis->has_value() && !y_batch_axis->has_value()));
  if (x_batch_axis->has_value()) {
    ctx->BatchAxis4ArgNameAndIndex("out", 0)->set_value(0);
  } else {
    ctx->BatchAxis4ArgNameAndIndex("out", 0)->clear_value();
  }
  return Maybe<void>::Ok();
}

Maybe<void> GetWhereSbpSignatures(user_op::SbpContext* ctx) {
  const Shape& cond_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("condition", 0).shape();
  const Shape& x_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0).shape();
  const Shape& y_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("y", 0).shape();
  std::shared_ptr<Shape> broadcasted_shape =
      JUST(GetWhereBroadcastedShape(cond_shape, x_shape, y_shape));
  Shape cond_extend_shape =
      CreateLeftExtendedShape(ShapeView(cond_shape), broadcasted_shape->NumAxes());
  Shape x_extend_shape = CreateLeftExtendedShape(ShapeView(x_shape), broadcasted_shape->NumAxes());
  Shape y_extend_shape = CreateLeftExtendedShape(ShapeView(y_shape), broadcasted_shape->NumAxes());
  FOR_RANGE(int64_t, i, 0, broadcasted_shape->NumAxes()) {
    int64_t cond_origin_axis = i - (broadcasted_shape->NumAxes() - cond_shape.NumAxes());
    int64_t x_origin_axis = i - (broadcasted_shape->NumAxes() - x_shape.NumAxes());
    int64_t y_origin_axis = i - (broadcasted_shape->NumAxes() - y_shape.NumAxes());
    if (cond_extend_shape.At(i) != 1 && x_extend_shape.At(i) != 1 && y_extend_shape.At(i) != 1) {
      SbpSignatureBuilder()
          .Split("condition", 0, cond_origin_axis)
          .Split("x", 0, x_origin_axis)
          .Split("y", 0, y_origin_axis)
          .Split("out", 0, i)
          .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
    } else if (cond_extend_shape.At(i) != 1 && x_extend_shape.At(i) != 1) {
      SbpSignatureBuilder()
          .Split("condition", 0, cond_origin_axis)
          .Split("x", 0, x_origin_axis)
          .Broadcast("y", 0)
          .Split("out", 0, i)
          .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
    } else if (cond_extend_shape.At(i) != 1 && y_extend_shape.At(i) != 1) {
      SbpSignatureBuilder()
          .Split("condition", 0, cond_origin_axis)
          .Broadcast("x", 0)
          .Split("y", 0, y_origin_axis)
          .Split("out", 0, i)
          .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
    } else if (x_extend_shape.At(i) != 1 && y_extend_shape.At(i) != 1) {
      SbpSignatureBuilder()
          .Broadcast("condition", 0)
          .Split("x", 0, x_origin_axis)
          .Split("y", 0, y_origin_axis)
          .Split("out", 0, i)
          .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
    } else if (cond_extend_shape.At(i) != 1) {
      SbpSignatureBuilder()
          .Split("condition", 0, cond_origin_axis)
          .Broadcast("x", 0)
          .Broadcast("y", 0)
          .Split("out", 0, i)
          .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
    } else if (x_extend_shape.At(i) != 1) {
      SbpSignatureBuilder()
          .Broadcast("condition", 0)
          .Split("x", 0, x_origin_axis)
          .Broadcast("y", 0)
          .Split("out", 0, i)
          .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
    } else if (y_extend_shape.At(i) != 1) {
      SbpSignatureBuilder()
          .Broadcast("condition", 0)
          .Broadcast("x", 0)
          .Split("y", 0, y_origin_axis)
          .Split("out", 0, i)
          .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
    }
  }
  SbpSignatureBuilder()
      .Broadcast("condition", 0)
      .PartialSum("x", 0)
      .PartialSum("y", 0)
      .PartialSum("out", 0)
      .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

Maybe<void> InferWhereGradBatchAxis(user_op::BatchAxisContext* ctx) {
  *ctx->BatchAxis4ArgNameAndIndex("dx", 0) = *ctx->BatchAxis4ArgNameAndIndex("x", 0);
  *ctx->BatchAxis4ArgNameAndIndex("dy", 0) = *ctx->BatchAxis4ArgNameAndIndex("y", 0);
  return Maybe<void>::Ok();
}

Maybe<void> GetWhereGradSbpSignatures(user_op::SbpContext* ctx) {
  const Shape& cond_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("condition", 0).shape();
  const Shape& x_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0).shape();
  const Shape& y_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("y", 0).shape();
  const Shape& dz_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("dz", 0).shape();
  Shape cond_extend_shape = CreateLeftExtendedShape(ShapeView(cond_shape), dz_shape.NumAxes());
  Shape x_extend_shape = CreateLeftExtendedShape(ShapeView(x_shape), dz_shape.NumAxes());
  Shape y_extend_shape = CreateLeftExtendedShape(ShapeView(y_shape), dz_shape.NumAxes());
  FOR_RANGE(int64_t, i, 0, dz_shape.NumAxes()) {
    int64_t cond_origin_axis = i - (dz_shape.NumAxes() - cond_shape.NumAxes());
    int64_t x_origin_axis = i - (dz_shape.NumAxes() - x_shape.NumAxes());
    int64_t y_origin_axis = i - (dz_shape.NumAxes() - y_shape.NumAxes());
    if (cond_extend_shape.At(i) != 1 && x_extend_shape.At(i) != 1 && y_extend_shape.At(i) != 1) {
      SbpSignatureBuilder()
          .Split("dz", 0, i)
          .Split("condition", 0, cond_origin_axis)
          .Split("x", 0, x_origin_axis)
          .Split("y", 0, y_origin_axis)
          .Split("dx", 0, x_origin_axis)
          .Split("dy", 0, y_origin_axis)
          .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
    } else if (cond_extend_shape.At(i) != 1 && x_extend_shape.At(i) != 1) {
      SbpSignatureBuilder()
          .Split("dz", 0, i)
          .Split("condition", 0, cond_origin_axis)
          .Split("x", 0, x_origin_axis)
          .Broadcast("y", 0)
          .Split("dx", 0, x_origin_axis)
          .Broadcast("dy", 0)
          .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
    } else if (cond_extend_shape.At(i) != 1 && y_extend_shape.At(i) != 1) {
      SbpSignatureBuilder()
          .Split("dz", 0, i)
          .Split("condition", 0, cond_origin_axis)
          .Broadcast("x", 0)
          .Split("y", 0, y_origin_axis)
          .Broadcast("dx", 0)
          .Split("dy", 0, y_origin_axis)
          .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
    } else if (x_extend_shape.At(i) != 1 && y_extend_shape.At(i) != 1) {
      SbpSignatureBuilder()
          .Split("dz", 0, i)
          .Broadcast("condition", 0)
          .Split("x", 0, x_origin_axis)
          .Split("y", 0, y_origin_axis)
          .Split("dx", 0, x_origin_axis)
          .Split("dy", 0, y_origin_axis)
          .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
    } else if (cond_extend_shape.At(i) != 1) {
      SbpSignatureBuilder()
          .Split("dz", 0, i)
          .Split("condition", 0, cond_origin_axis)
          .Broadcast("x", 0)
          .Broadcast("y", 0)
          .Broadcast("dx", 0)
          .Broadcast("dy", 0)
          .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
    } else if (x_extend_shape.At(i) != 1) {
      SbpSignatureBuilder()
          .Split("dz", 0, i)
          .Broadcast("condition", 0)
          .Split("x", 0, x_origin_axis)
          .Broadcast("y", 0)
          .Split("dx", 0, x_origin_axis)
          .Broadcast("dy", 0)
          .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
    } else if (y_extend_shape.At(i) != 1) {
      SbpSignatureBuilder()
          .Split("dz", 0, i)
          .Broadcast("condition", 0)
          .Broadcast("x", 0)
          .Split("y", 0, y_origin_axis)
          .Broadcast("dx", 0)
          .Split("dy", 0, y_origin_axis)
          .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
    }
  }
  SbpSignatureBuilder()
      .PartialSum("dz", 0)
      .Broadcast("condition", 0)
      .PartialSum("x", 0)
      .PartialSum("y", 0)
      .PartialSum("dx", 0)
      .PartialSum("dy", 0)
      .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
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
